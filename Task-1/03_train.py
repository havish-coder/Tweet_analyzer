"""
03_train.py
===========
Phase 3 of the Tweet Likes Prediction pipeline.

Two-stage model on the combined feature matrix (32 tabular + 384 embedding):

  Stage A — XGBoost CLASSIFIER on popularity buckets:
              Class 0  (common):  likes in [0, 75th pct)
              Class 1  (popular): likes in [75th, 95th pct)
              Class 2  (viral):   likes in [95th pct, ∞)

  Stage B — Three SPECIALIST XGBoost regressors, one per bucket,
            each trained on log1p(likes) using only that bucket's rows.

At inference time, predictions are PROBABILITY-WEIGHTED across all three
specialist regressors (soft routing) — see 04_predict.py.

Outputs:
  models/classifier_model.joblib
  models/regressor_class_{0,1,2}.joblib
  models/class_bins.joblib
  models/feature_cols.joblib
  models/tabular_scaler.joblib
  models/metrics.json
"""

import json
import logging
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_FEATURES = "features_train.csv"
INPUT_EMB      = "embeddings_train.npy"
INPUT_EMB_IDS  = "embeddings_train_ids.npy"

OUT_CLF        = "models/classifier_model.joblib"
OUT_REG_PREFIX = "models/regressor_class_"
OUT_BINS       = "models/class_bins.joblib"
OUT_COLS       = "models/feature_cols.joblib"
OUT_SCALER     = "models/tabular_scaler.joblib"
OUT_METRICS    = "models/metrics.json"

SEED            = 42
EVAL_BRAND_FRAC = 0.05
EVAL_TIME_FRAC  = 0.05

# Bucket quantiles
PCT_LOW, PCT_HIGH = 0.75, 0.95


DROP_COLS = {
    "id", "date", "content", "content_clean", "media", "username",
    "inferred company", "likes", "log_likes",
    "image_url", "video_thumb_url", "video_mp4_url", "target_download_url",
}


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def regime_split(df: pd.DataFrame, brand_frac: float, time_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    brand_col = "inferred company" if "inferred company" in df.columns else "username"

    brands = pd.Series(df[brand_col]).dropna().unique()
    n_brands = max(1, int(round(len(brands) * brand_frac)))
    held = set(rng.choice(brands, size=n_brands, replace=False).tolist())
    brand_mask = df[brand_col].isin(held)

    remaining = df.loc[~brand_mask].copy()
    remaining["_dt"] = pd.to_datetime(remaining["date"], errors="coerce")
    remaining = remaining.sort_values("_dt", kind="stable")
    n_time = int(round(len(remaining) * time_frac))
    time_idx = remaining.tail(n_time).index if n_time else pd.Index([])

    eval_pos = sorted(set(df.index[brand_mask].tolist()) | set(time_idx.tolist()))
    train_pos = sorted(set(df.index.tolist()) - set(eval_pos))
    logger.info(
        f"Regime split: held-out brands={n_brands}/{len(brands)} "
        f"({len(df.index[brand_mask])} rows), "
        f"latest-time held out={n_time} rows, "
        f"total train={len(train_pos)}, eval={len(eval_pos)}"
    )
    return np.array(train_pos), np.array(eval_pos)


def assign_class(likes: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """0 if < q_low, 1 if < q_high, 2 otherwise."""
    out = np.zeros(likes.shape, dtype=int)
    out[likes >= q_low]  = 1
    out[likes >= q_high] = 2
    return out


# ---------------------------------------------------------------------------
def main():
    if not (os.path.exists(INPUT_FEATURES) and os.path.exists(INPUT_EMB)):
        raise FileNotFoundError("Run 01_features.py + 02_embed.py first.")

    df = pd.read_csv(INPUT_FEATURES)
    emb = np.load(INPUT_EMB)
    emb_ids = np.load(INPUT_EMB_IDS, allow_pickle=True)
    if not np.array_equal(emb_ids, df["id"].values):
        logger.warning("Embedding ids don't match feature order — realigning by id.")
        idx_map = {tid: i for i, tid in enumerate(emb_ids)}
        emb = np.stack([emb[idx_map[tid]] for tid in df["id"].values])

    feature_cols = select_feature_columns(df)
    logger.info(f"Numeric features: {len(feature_cols)}")

    X_tab = df[feature_cols].fillna(0).astype(np.float32).values
    scaler = StandardScaler().fit(X_tab)
    X = np.hstack([scaler.transform(X_tab).astype(np.float32), emb]).astype(np.float32)
    y_likes = df["likes"].astype(float).values
    y_log   = df["log_likes"].astype(np.float32).values

    train_pos, eval_pos = regime_split(df, EVAL_BRAND_FRAC, EVAL_TIME_FRAC, SEED)

    # ---------------- Bucket boundaries (from TRAIN only) ----------------
    train_likes = y_likes[train_pos]
    q_low  = float(np.quantile(train_likes, PCT_LOW))
    q_high = float(np.quantile(train_likes, PCT_HIGH))
    logger.info(f"Bucket boundaries (train-only quantiles): low={q_low:.0f}, high={q_high:.0f}")

    y_cls       = assign_class(y_likes, q_low, q_high)
    y_cls_train = y_cls[train_pos]
    y_cls_eval  = y_cls[eval_pos]
    logger.info(
        f"Class distribution (train): {np.bincount(y_cls_train)} "
        f"({np.bincount(y_cls_train) / len(y_cls_train) * 100})"
    )

    # ---------------- Stage A: Classifier ----------------
    class_counts = np.bincount(y_cls_train)
    class_weights = len(y_cls_train) / (len(class_counts) * class_counts.astype(float))
    sample_weight = np.take(class_weights, y_cls_train)
    logger.info(f"Class weights: {class_weights}")

    clf = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
        verbosity=1,
    )

    logger.info("Training classifier (stage A)...")
    clf.fit(
        X[train_pos], y_cls_train,
        sample_weight=sample_weight,
        eval_set=[(X[eval_pos], y_cls_eval)],
        verbose=100,
    )
    cls_pred_eval  = clf.predict(X[eval_pos])
    cls_proba_eval = clf.predict_proba(X[eval_pos])

    logger.info("Classifier eval report:\n" + classification_report(y_cls_eval, cls_pred_eval, digits=3))
    cm = confusion_matrix(y_cls_eval, cls_pred_eval)
    logger.info(f"Confusion matrix:\n{cm}")

    # ---------------- Stage B: 3 specialist regressors ----------------
    specialists = {}
    for cls_id in [0, 1, 2]:
        mask = (y_cls_train == cls_id)
        if mask.sum() < 30:
            logger.warning(f"  class {cls_id}: only {mask.sum()} train rows — skipping (will fall back).")
            continue
        logger.info(f"Training regressor for class {cls_id} ({mask.sum()} rows)...")

        # Validation slice = eval rows that BELONG to this class (per ground truth)
        eval_mask = (y_cls_eval == cls_id)
        eval_X = X[eval_pos][eval_mask]
        eval_y = y_log[eval_pos][eval_mask]
        eval_set = [(eval_X, eval_y)] if len(eval_X) > 0 else None

        reg = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            early_stopping_rounds=50 if eval_set else None,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        reg.fit(
            X[train_pos][mask], y_log[train_pos][mask],
            eval_set=eval_set,
            verbose=False,
        )
        specialists[cls_id] = reg
        joblib.dump(reg, f"{OUT_REG_PREFIX}{cls_id}.joblib")

    # ---------------- Build predictions on eval set ----------------
    n_eval = len(eval_pos)
    reg_pred_per_class = np.zeros((n_eval, 3), dtype=np.float32)
    for cls_id, reg in specialists.items():
        reg_pred_per_class[:, cls_id] = reg.predict(X[eval_pos])

    # HARD routing — pick predicted class, take its regressor
    hard_log = reg_pred_per_class[np.arange(n_eval), cls_pred_eval]
    hard_pred = np.clip(np.expm1(hard_log), 0, None)

    # SOFT routing — probability-weighted log prediction
    soft_log = (cls_proba_eval * reg_pred_per_class).sum(axis=1)
    soft_pred = np.clip(np.expm1(soft_log), 0, None)

    actual_eval = np.clip(np.expm1(y_log[eval_pos]), 0, None)
    rmse_hard_raw = float(np.sqrt(mean_squared_error(actual_eval, hard_pred)))
    rmse_soft_raw = float(np.sqrt(mean_squared_error(actual_eval, soft_pred)))
    rmse_hard_log = float(np.sqrt(mean_squared_error(y_log[eval_pos], hard_log)))
    rmse_soft_log = float(np.sqrt(mean_squared_error(y_log[eval_pos], soft_log)))

    # Oracle: cheat and use the TRUE class to pick the regressor.
    oracle_log = reg_pred_per_class[np.arange(n_eval), y_cls_eval]
    oracle_pred = np.clip(np.expm1(oracle_log), 0, None)
    rmse_oracle_raw = float(np.sqrt(mean_squared_error(actual_eval, oracle_pred)))

    logger.info("─" * 70)
    logger.info(f"  HARD-routed   RMSE (raw likes): {rmse_hard_raw:,.2f}")
    logger.info(f"  SOFT-routed   RMSE (raw likes): {rmse_soft_raw:,.2f}")
    logger.info(f"  Oracle-routed RMSE (raw likes): {rmse_oracle_raw:,.2f}  ← upper bound if classifier were perfect")
    logger.info(f"  HARD-routed   RMSE (log scale): {rmse_hard_log:.4f}")
    logger.info(f"  SOFT-routed   RMSE (log scale): {rmse_soft_log:.4f}")
    logger.info("─" * 70)

    # ---------------- Persist artifacts ----------------
    os.makedirs("models", exist_ok=True); joblib.dump(clf, OUT_CLF)
    joblib.dump({"q_low": q_low, "q_high": q_high}, OUT_BINS)
    joblib.dump(feature_cols, OUT_COLS)
    joblib.dump(scaler, OUT_SCALER)

    metrics = {
        "q_low": q_low,
        "q_high": q_high,
        "n_train": int(len(train_pos)),
        "n_eval":  int(len(eval_pos)),
        "class_distribution_train": np.bincount(y_cls_train).tolist(),
        "class_distribution_eval":  np.bincount(y_cls_eval).tolist(),
        "rmse_hard_raw":   rmse_hard_raw,
        "rmse_soft_raw":   rmse_soft_raw,
        "rmse_oracle_raw": rmse_oracle_raw,
        "rmse_hard_log":   rmse_hard_log,
        "rmse_soft_log":   rmse_soft_log,
        "classifier_acc": float((cls_pred_eval == y_cls_eval).mean()),
        "confusion_matrix": cm.tolist(),
    }
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics → {OUT_METRICS}")


if __name__ == "__main__":
    main()
