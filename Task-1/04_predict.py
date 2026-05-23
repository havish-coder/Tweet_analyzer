"""
04_predict.py
=============
Phase 4 of the Tweet Likes Prediction pipeline.

Inference on the two competition test sets using the classify-then-regress
cascade. For each test row:

  1. Generate the same features + MiniLM embedding as during training.
  2. Classifier predicts probabilities over 3 popularity buckets.
  3. Each specialist regressor predicts log_likes.
  4. Final prediction is PROBABILITY-WEIGHTED:
        expected_likes = Σ p(class_k) · expm1(r_k(x))

Outputs:
  outputs/submission_company.xlsx
  outputs/submission_time.xlsx
  outputs/predictions_company.csv
  outputs/predictions_time.csv
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from importlib.machinery import SourceFileLoader
_features_mod = SourceFileLoader("features_mod", "01_features.py").load_module()
TabularFeatureBuilder = _features_mod.TabularFeatureBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_DIR        = "data/test"
TEST_COMPANY    = f"{TEST_DIR}/behaviour_simulation_test_company.xlsx"
TEST_TIME       = f"{TEST_DIR}/behaviour_simulation_test_time.xlsx"

COMPANY_STATS   = "models/company_stats.joblib"
CLF_PATH        = "models/classifier_model.joblib"
REG_PREFIX      = "models/regressor_class_"
FEATURE_COLS    = "models/feature_cols.joblib"
SCALER_PATH     = "models/tabular_scaler.joblib"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE      = 128

OUT_SUB_COMPANY = "outputs/submission_company.xlsx"
OUT_SUB_TIME    = "outputs/submission_time.xlsx"
OUT_PRED_COMP   = "outputs/predictions_company.csv"
OUT_PRED_TIME   = "outputs/predictions_time.csv"


def predict_regime(
    xlsx_path: str,
    sub_path: str,
    pred_path: str,
    regime_label: str,
    company_stats: dict,
    clf,
    specialists: dict,
    feature_cols: list,
    scaler,
    embed_model: SentenceTransformer,
) -> pd.DataFrame:
    logger.info("─" * 70)
    logger.info(f"REGIME: {regime_label}")
    logger.info("─" * 70)

    df = pd.read_excel(xlsx_path)
    logger.info(f"  Loaded {len(df)} rows from {xlsx_path}")

    builder = TabularFeatureBuilder(df, is_train=False, company_stats=company_stats)
    feat_df = builder.run()

    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    X_tab = feat_df[feature_cols].fillna(0).astype(np.float32).values
    X_tab_scaled = scaler.transform(X_tab).astype(np.float32)

    texts = feat_df["content_clean"].fillna("").astype(str).tolist()
    emb = embed_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    X = np.hstack([X_tab_scaled, emb]).astype(np.float32)
    logger.info(f"  Feature matrix: {X.shape}")

    # Stage A: class probabilities
    proba = clf.predict_proba(X)  # (n, 3)

    # Stage B: each specialist regressor predicts log_likes for all rows
    n = X.shape[0]
    reg_pred = np.zeros((n, 3), dtype=np.float32)
    for cls_id in [0, 1, 2]:
        if cls_id in specialists:
            reg_pred[:, cls_id] = specialists[cls_id].predict(X)

    # SOFT routing on the raw-likes scale
    raw_per_class = np.clip(np.expm1(reg_pred), 0, None)
    soft_pred = (proba * raw_per_class).sum(axis=1)
    pred = np.clip(soft_pred, 0, None).round().astype(int)

    # HARD routing (for inspection only)
    hard_class = proba.argmax(axis=1)
    hard_log   = reg_pred[np.arange(n), hard_class]
    hard_pred  = np.clip(np.expm1(hard_log), 0, None).round().astype(int)

    submission = pd.DataFrame({
        "id":               df["id"],
        "date":             df["date"],
        "username":         df["username"],
        "media":            df["media"],
        "inferred company": df["inferred company"],
        "content":          df.get("content", ""),
        "predicted_likes":  pred,
    })
    os.makedirs("outputs", exist_ok=True); submission.to_excel(sub_path, index=False)
    logger.info(f"  Wrote {sub_path}")

    debug = submission.copy()
    debug["pred_class"]     = hard_class
    debug["hard_pred"]      = hard_pred
    debug["soft_pred"]      = pred
    debug["p_common"]       = proba[:, 0]
    debug["p_popular"]      = proba[:, 1]
    debug["p_viral"]        = proba[:, 2]
    if "likes" in df.columns:
        actuals = df["likes"].astype(float).values
        debug["actual_likes"] = actuals
        debug["abs_error"]    = np.abs(actuals - pred)
        from sklearn.metrics import mean_squared_error
        rmse = float(np.sqrt(mean_squared_error(actuals, pred)))
        logger.info(f"  Soft-routed RMSE ({regime_label}) = {rmse:,.2f}")

    debug.to_csv(pred_path, index=False)
    logger.info(f"  Wrote {pred_path}")
    return submission


def main():
    for p in (COMPANY_STATS, CLF_PATH, FEATURE_COLS, SCALER_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — run 03b_train_class_reg.py first.")
    for p in (TEST_COMPANY, TEST_TIME):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found.")

    company_stats = joblib.load(COMPANY_STATS)
    clf           = joblib.load(CLF_PATH)
    feature_cols  = joblib.load(FEATURE_COLS)
    scaler        = joblib.load(SCALER_PATH)

    specialists = {}
    for cls_id in [0, 1, 2]:
        p = f"{REG_PREFIX}{cls_id}.joblib"
        if os.path.exists(p):
            specialists[cls_id] = joblib.load(p)
            logger.info(f"Loaded specialist regressor for class {cls_id}")

    logger.info(f"Loading embedding model {EMBED_MODEL}...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    predict_regime(
        TEST_COMPANY, OUT_SUB_COMPANY, OUT_PRED_COMP,
        "Unseen Brands", company_stats, clf, specialists,
        feature_cols, scaler, embed_model,
    )
    predict_regime(
        TEST_TIME, OUT_SUB_TIME, OUT_PRED_TIME,
        "Unseen Time", company_stats, clf, specialists,
        feature_cols, scaler, embed_model,
    )

    logger.info("All classification-then-regression submissions written.")


if __name__ == "__main__":
    main()
