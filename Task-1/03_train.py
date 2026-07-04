"""
03_train.py
===========
Phase 3 of the Tweet Likes Prediction pipeline.

Two-stage model on the combined feature matrix (32 tabular + 384 embedding):

  Stage A — XGBoost CLASSIFIER over 7 popularity buckets
            (edges at 100 / 250 / 500 / 1k / 2.5k / 5k likes).

  Stage B — Seven SPECIALIST XGBoost regressors, one per bucket,
            each trained on log1p(likes) using only that bucket's rows.

At inference time, predictions are PROBABILITY-WEIGHTED across all seven
specialist regressors in raw-likes space (soft routing) — the same rule
as 04_predict.py. A single-regressor baseline (plus smearing-corrected and
tweedie variants) is trained alongside as the ablation the cascade must beat.

Leakage discipline: the regime-mirrored eval set is used only for reporting;
early stopping runs on an inner slice of train, and company priors + scaler
are computed from train rows only.

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
SMOOTHING_PRIOR = 30    # must match 01_features.py
INNER_ES_FRAC   = 0.10  # slice of TRAIN used for early stopping, so the
                        # regime-mirrored eval set is only ever used for reporting

# 7-bucket cascade — explicit edges (not quantiles) for clean alignment with engagement levels
BUCKET_EDGES = [100, 250, 500, 1000, 2500, 5000]
N_CLASSES    = len(BUCKET_EDGES) + 1   # 7


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


def assign_class(likes: np.ndarray, edges) -> np.ndarray:
    """Assign one of N_CLASSES buckets based on explicit edges."""
    out = np.zeros(likes.shape, dtype=int)
    for i, edge in enumerate(edges):
        out[likes >= edge] = i + 1
    return out


def recompute_company_prior(df: pd.DataFrame, train_pos: np.ndarray) -> pd.DataFrame:
    """
    Rebuild company_avg_log_likes / company_tweet_count using ONLY train rows.

    01_features.py computes the LOO prior over the FULL training file, so eval
    rows from held-out brands still carry that brand's real history — but a truly
    unseen brand at test time gets the smoothed global fallback. Recomputing from
    train rows only makes held-out-brand eval rows see exactly what unseen brands
    see at test time (and held-out-time rows see only past data for their brand).
    """
    brand_col = "inferred company" if "inferred company" in df.columns else "username"
    tr = df.iloc[train_pos]
    global_mean = float(tr["log_likes"].mean())
    stats = tr.groupby(brand_col)["log_likes"].agg(["sum", "count"])

    sums   = df[brand_col].map(stats["sum"]).fillna(0.0)
    counts = df[brand_col].map(stats["count"]).fillna(0.0)

    in_train = np.zeros(len(df), dtype=bool)
    in_train[train_pos] = True
    in_train = pd.Series(in_train, index=df.index)

    # Train rows: leave-one-out.  Eval rows: plain smoothed mean — same formula
    # as the inference-time lookup in 01_features.py (unseen brand → global mean).
    eff_sum   = sums - df["log_likes"].where(in_train, 0.0)
    eff_count = (counts - in_train.astype(float)).clip(lower=0)

    df["company_avg_log_likes"] = (eff_sum + SMOOTHING_PRIOR * global_mean) / (eff_count + SMOOTHING_PRIOR)
    df["company_tweet_count"]   = counts.astype(int)
    n_unseen = int((counts[~in_train] == 0).sum())
    logger.info(f"Company priors recomputed from train rows only "
                f"({n_unseen} eval rows now use the unseen-brand global fallback).")
    return df


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

    # Split FIRST, then compute any statistic (priors, scaler) from train rows only.
    train_pos, eval_pos = regime_split(df, EVAL_BRAND_FRAC, EVAL_TIME_FRAC, SEED)
    df = recompute_company_prior(df, train_pos)

    X_tab = df[feature_cols].fillna(0).astype(np.float32).values
    scaler = StandardScaler().fit(X_tab[train_pos])
    X = np.hstack([scaler.transform(X_tab).astype(np.float32), emb]).astype(np.float32)
    y_likes = df["likes"].astype(float).values
    y_log   = df["log_likes"].astype(np.float32).values

    # Inner early-stopping slice carved from TRAIN — the regime-mirrored eval
    # set is used exclusively for the reported metrics.
    rng = np.random.default_rng(SEED)
    inner_mask = np.zeros(len(train_pos), dtype=bool)
    inner_mask[rng.choice(len(train_pos), size=int(len(train_pos) * INNER_ES_FRAC), replace=False)] = True
    fit_pos = train_pos[~inner_mask]
    es_pos  = train_pos[inner_mask]
    logger.info(f"Inner split: fit={len(fit_pos)}, early-stopping={len(es_pos)}, report eval={len(eval_pos)}")

    # ---------------- Bucket boundaries (fixed edges, not quantiles) ----------------
    logger.info(f"Bucket edges: {BUCKET_EDGES}  ({N_CLASSES} buckets)")

    y_cls       = assign_class(y_likes, BUCKET_EDGES)
    y_cls_train = y_cls[train_pos]
    y_cls_fit   = y_cls[fit_pos]
    y_cls_es    = y_cls[es_pos]
    y_cls_eval  = y_cls[eval_pos]
    logger.info(
        f"Class distribution (train): {np.bincount(y_cls_train)} "
        f"({np.bincount(y_cls_train) / len(y_cls_train) * 100})"
    )

    # ---------------- Stage A: Classifier ----------------
    class_counts = np.bincount(y_cls_fit, minlength=N_CLASSES)
    class_weights = len(y_cls_fit) / (len(class_counts) * np.maximum(class_counts.astype(float), 1))
    sample_weight = np.take(class_weights, y_cls_fit)
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
        num_class=N_CLASSES,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
        verbosity=1,
    )

    logger.info("Training classifier (stage A)...")
    clf.fit(
        X[fit_pos], y_cls_fit,
        sample_weight=sample_weight,
        eval_set=[(X[es_pos], y_cls_es)],   # early stopping on inner slice, NOT the report set
        verbose=100,
    )
    cls_pred_eval  = clf.predict(X[eval_pos])
    cls_proba_eval = clf.predict_proba(X[eval_pos])

    logger.info("Classifier eval report:\n" + classification_report(y_cls_eval, cls_pred_eval, digits=3))
    cm = confusion_matrix(y_cls_eval, cls_pred_eval)
    logger.info(f"Confusion matrix:\n{cm}")

    # ---------------- Stage B: N_CLASSES specialist regressors ----------------
    specialists = {}
    for cls_id in range(N_CLASSES):
        mask = (y_cls_fit == cls_id)
        if mask.sum() < 30:
            logger.warning(f"  class {cls_id}: only {mask.sum()} train rows — skipping (will fall back).")
            continue
        logger.info(f"Training regressor for class {cls_id} ({mask.sum()} rows)...")

        # Early-stopping slice = inner-ES rows that BELONG to this class (per ground truth)
        es_mask = (y_cls_es == cls_id)
        eval_X = X[es_pos][es_mask]
        eval_y = y_log[es_pos][es_mask]
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
            X[fit_pos][mask], y_log[fit_pos][mask],
            eval_set=eval_set,
            verbose=False,
        )
        specialists[cls_id] = reg
        joblib.dump(reg, f"{OUT_REG_PREFIX}{cls_id}.joblib")

    # ---------------- Build predictions on eval set ----------------
    n_eval = len(eval_pos)
    reg_pred_per_class = np.zeros((n_eval, N_CLASSES), dtype=np.float32)
    for cls_id, reg in specialists.items():
        reg_pred_per_class[:, cls_id] = reg.predict(X[eval_pos])

    # HARD routing — pick predicted class, take its regressor
    hard_log = reg_pred_per_class[np.arange(n_eval), cls_pred_eval]
    hard_pred = np.clip(np.expm1(hard_log), 0, None)

    # SOFT routing — probability-weighted in RAW-likes space.
    # This is the exact rule 04_predict.py ships: Σ p_k · expm1(r_k).
    # (The old log-space variant expm1(Σ p·r) validated a different model
    # than the one deployed — kept below only as a reference metric.)
    raw_per_class = np.clip(np.expm1(reg_pred_per_class), 0, None)
    soft_pred = (cls_proba_eval * raw_per_class).sum(axis=1)
    soft_log  = np.log1p(np.clip(soft_pred, 0, None))

    soft_logspace_pred = np.clip(np.expm1((cls_proba_eval * reg_pred_per_class).sum(axis=1)), 0, None)

    actual_eval = np.clip(np.expm1(y_log[eval_pos]), 0, None)
    rmse_hard_raw = float(np.sqrt(mean_squared_error(actual_eval, hard_pred)))
    rmse_soft_raw = float(np.sqrt(mean_squared_error(actual_eval, soft_pred)))
    rmse_soft_logspace_raw = float(np.sqrt(mean_squared_error(actual_eval, soft_logspace_pred)))
    rmse_hard_log = float(np.sqrt(mean_squared_error(y_log[eval_pos], hard_log)))
    rmse_soft_log = float(np.sqrt(mean_squared_error(y_log[eval_pos], soft_log)))

    # Oracle: cheat and use the TRUE class to pick the regressor.
    oracle_log = reg_pred_per_class[np.arange(n_eval), y_cls_eval]
    oracle_pred = np.clip(np.expm1(oracle_log), 0, None)
    rmse_oracle_raw = float(np.sqrt(mean_squared_error(actual_eval, oracle_pred)))

    # ---------------- Hybrid: cascade + PER-TIER smearing ----------------
    # classifier -> tier regressor -> per-tier Duan smearing -> prediction.
    # Each tier's factor comes from that tier's inner-ES residuals (>= 20 rows,
    # else 1.0). Answers whether the cascade was losing to the smeared baseline
    # because of retransformation bias (this fixes it) or routing error (this doesn't).
    smear_k = np.ones(N_CLASSES, dtype=np.float64)
    for cls_id, reg in specialists.items():
        es_mask_k = (y_cls_es == cls_id)
        if es_mask_k.sum() >= 20:
            resid = y_log[es_pos][es_mask_k] - reg.predict(X[es_pos][es_mask_k])
            smear_k[cls_id] = float(np.mean(np.exp(resid)))
    logger.info(f"Per-tier smearing factors: {np.round(smear_k, 3).tolist()}")

    raw_smeared = np.clip(np.exp(reg_pred_per_class) * smear_k[None, :] - 1.0, 0, None)
    soft_smear_pred = (cls_proba_eval * raw_smeared).sum(axis=1)
    hard_smear_pred = raw_smeared[np.arange(n_eval), cls_pred_eval]
    rmse_soft_smear_raw = float(np.sqrt(mean_squared_error(actual_eval, soft_smear_pred)))
    rmse_hard_smear_raw = float(np.sqrt(mean_squared_error(actual_eval, hard_smear_pred)))

    # ---------------- Baseline ablation: single regressor, no cascade ----------------
    # The cascade has to beat this to justify its existence.
    logger.info("Training single-regressor baseline (log target)...")
    base = xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=8, min_child_weight=2,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:squarederror", tree_method="hist",
        early_stopping_rounds=50, random_state=SEED, n_jobs=-1, verbosity=0,
    )
    base.fit(X[fit_pos], y_log[fit_pos], eval_set=[(X[es_pos], y_log[es_pos])], verbose=False)
    base_log_eval = base.predict(X[eval_pos])
    base_pred = np.clip(np.expm1(base_log_eval), 0, None)
    rmse_baseline_raw = float(np.sqrt(mean_squared_error(actual_eval, base_pred)))
    rmse_baseline_log = float(np.sqrt(mean_squared_error(y_log[eval_pos], base_log_eval)))

    # Duan's smearing correction — expm1(mean_log) underestimates E[likes|x];
    # rescale by the mean of exp(residual) measured on the inner-ES slice.
    smear = float(np.mean(np.exp(y_log[es_pos] - base.predict(X[es_pos]))))
    base_smear_pred = np.clip(np.exp(base_log_eval) * smear - 1.0, 0, None)
    rmse_baseline_smear_raw = float(np.sqrt(mean_squared_error(actual_eval, base_smear_pred)))

    # Tweedie on raw likes — optimizes closer to the raw-RMSE metric directly.
    logger.info("Training single-regressor baseline (tweedie, raw target)...")
    tweedie = xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=8, min_child_weight=2,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:tweedie", tweedie_variance_power=1.5, tree_method="hist",
        early_stopping_rounds=50, random_state=SEED, n_jobs=-1, verbosity=0,
    )
    tweedie.fit(X[fit_pos], y_likes[fit_pos], eval_set=[(X[es_pos], y_likes[es_pos])], verbose=False)
    tweedie_pred = np.clip(tweedie.predict(X[eval_pos]), 0, None)
    rmse_tweedie_raw = float(np.sqrt(mean_squared_error(actual_eval, tweedie_pred)))

    # ---------------- Ensemble: smeared baseline + smeared cascade ----------------
    # The cascade doesn't need to beat the single model to be useful — only to make
    # different errors. Blend in log space at a few weights; report each.
    ens_results = {}
    for w in (0.3, 0.5, 0.7):  # weight on the single-regressor baseline
        blend_log = w * np.log1p(base_smear_pred) + (1 - w) * np.log1p(soft_smear_pred)
        blend = np.clip(np.expm1(blend_log), 0, None)
        ens_results[w] = float(np.sqrt(mean_squared_error(actual_eval, blend)))

    logger.info("─" * 70)
    for w, r in ens_results.items():
        logger.info(f"  ENSEMBLE {w:.0%} baseline + {1-w:.0%} cascade  RMSE: {r:,.2f}")
    logger.info(f"  CASCADE soft (raw-space, shipped) RMSE: {rmse_soft_raw:,.2f}")
    logger.info(f"  CASCADE soft + per-tier smearing  RMSE: {rmse_soft_smear_raw:,.2f}")
    logger.info(f"  CASCADE hard + per-tier smearing  RMSE: {rmse_hard_smear_raw:,.2f}")
    logger.info(f"  CASCADE soft (log-space, old)     RMSE: {rmse_soft_logspace_raw:,.2f}")
    logger.info(f"  CASCADE hard                      RMSE: {rmse_hard_raw:,.2f}")
    logger.info(f"  CASCADE oracle (perfect clf)      RMSE: {rmse_oracle_raw:,.2f}")
    logger.info(f"  BASELINE single reg (log)         RMSE: {rmse_baseline_raw:,.2f}")
    logger.info(f"  BASELINE + smearing ({smear:.3f})     RMSE: {rmse_baseline_smear_raw:,.2f}")
    logger.info(f"  BASELINE tweedie (raw)            RMSE: {rmse_tweedie_raw:,.2f}")
    logger.info(f"  soft RMSE (log scale): {rmse_soft_log:.4f}  |  baseline RMSE (log scale): {rmse_baseline_log:.4f}")
    logger.info("─" * 70)

    # ---------------- Persist artifacts ----------------
    os.makedirs("models", exist_ok=True); joblib.dump(clf, OUT_CLF)
    # Baseline bundle — 04_predict.py ships whichever predictor won on val.
    joblib.dump({"model": base, "smearing_factor": smear}, "models/baseline_regressor.joblib")
    joblib.dump({"edges": BUCKET_EDGES, "n_classes": N_CLASSES}, OUT_BINS)
    joblib.dump(feature_cols, OUT_COLS)
    joblib.dump(scaler, OUT_SCALER)

    metrics = {
        "bucket_edges": BUCKET_EDGES,
        "n_classes": N_CLASSES,
        "n_train": int(len(train_pos)),
        "n_fit":   int(len(fit_pos)),
        "n_inner_es": int(len(es_pos)),
        "n_eval":  int(len(eval_pos)),
        "class_distribution_train": np.bincount(y_cls_train, minlength=N_CLASSES).tolist(),
        "class_distribution_eval":  np.bincount(y_cls_eval, minlength=N_CLASSES).tolist(),
        "rmse_hard_raw":   rmse_hard_raw,
        "rmse_soft_raw":   rmse_soft_raw,
        "rmse_soft_logspace_raw": rmse_soft_logspace_raw,
        "rmse_oracle_raw": rmse_oracle_raw,
        "rmse_hard_log":   rmse_hard_log,
        "rmse_soft_log":   rmse_soft_log,
        "rmse_baseline_raw":       rmse_baseline_raw,
        "rmse_baseline_log":       rmse_baseline_log,
        "rmse_baseline_smear_raw": rmse_baseline_smear_raw,
        "smearing_factor":         smear,
        "rmse_cascade_soft_smear_raw": rmse_soft_smear_raw,
        "rmse_cascade_hard_smear_raw": rmse_hard_smear_raw,
        "per_tier_smearing_factors": smear_k.tolist(),
        "rmse_ensemble_by_baseline_weight": {str(w): r for w, r in ens_results.items()},
        "rmse_tweedie_raw":        rmse_tweedie_raw,
        "classifier_acc": float((cls_pred_eval == y_cls_eval).mean()),
        "confusion_matrix": cm.tolist(),
        "notes": "Eval set is regime-mirrored (unseen brands + latest time) and used ONLY for reporting; "
                 "early stopping uses an inner 10% slice of train. Company priors recomputed from train rows "
                 "only, so held-out-brand rows use the unseen-brand fallback. Soft routing is raw-space, "
                 "matching 04_predict.py.",
    }
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics → {OUT_METRICS}")


if __name__ == "__main__":
    main()
