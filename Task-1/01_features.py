"""
01_features.py
==============
Phase 1 of the Tweet Likes Prediction pipeline.

Builds a clean numerical feature matrix from raw tweet metadata.

Design principles:
  * Cyclical time encoding (sin/cos) — preserves hour-23 / hour-0 adjacency.
  * COVID regime flag — explicit binary for the March-2020 distribution shift.
  * Media regex parsing — extracts video duration / views / target URL.
  * Leave-one-out company prior — strong signal without row-level leakage.
  * Company stats serialised to joblib so test-time inference can look them up
    (unseen brands fall back to smoothed global mean).

This script is fit on the FULL training table; the resulting `company_stats.joblib`
and the row-level features are written to disk for downstream consumption.

Usage:
    python 01_features.py
"""

import logging
import os
import re

import joblib
import numpy as np
import pandas as pd

try:
    import emoji  # optional — counts emoji properly
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV          = "data/train.csv"
OUTPUT_FEATURES    = "features_train.csv"
OUTPUT_STATS       = "models/company_stats.joblib"
COVID_START        = pd.Timestamp("2020-03-01")
SMOOTHING_PRIOR    = 30  # rows of pseudo-evidence pulled toward global mean


# ---------------------------------------------------------------------------
# TabularFeatureBuilder
# ---------------------------------------------------------------------------
class TabularFeatureBuilder:
    """Turns a raw tweet table into a numeric feature matrix + target."""

    def __init__(self, df: pd.DataFrame, is_train: bool = True, company_stats: dict | None = None):
        self.df = df.copy()
        self.is_train = is_train
        # company_stats: {company: {"mean_log_likes": float, "count": int}}
        self.company_stats = company_stats or {}
        # Filled in by transform_target during training:
        self.global_mean_log_likes = 0.0

    # ----------------------------------------------------------------
    # Target
    # ----------------------------------------------------------------
    def transform_target(self):
        if "likes" not in self.df.columns:
            return
        if self.is_train:
            logger.info("Transforming target 'likes' → 'log_likes'...")
            self.df["log_likes"] = np.log1p(self.df["likes"].astype(float))
            self.global_mean_log_likes = float(self.df["log_likes"].mean())
        else:
            # Keep likes if present (some test files have it for sanity-check)
            if "likes" in self.df.columns:
                self.df["log_likes"] = np.log1p(self.df["likes"].astype(float))

    # ----------------------------------------------------------------
    # Cyclical time
    # ----------------------------------------------------------------
    def engineer_cyclical_time(self):
        logger.info("Extracting cyclical temporal features...")
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

        hour  = self.df["date"].dt.hour.fillna(0)
        dow   = self.df["date"].dt.dayofweek.fillna(0)
        month = self.df["date"].dt.month.fillna(1)

        self.df["hour_sin"]  = np.sin(2 * np.pi * hour / 24.0)
        self.df["hour_cos"]  = np.cos(2 * np.pi * hour / 24.0)
        self.df["day_sin"]   = np.sin(2 * np.pi * dow / 7.0)
        self.df["day_cos"]   = np.cos(2 * np.pi * dow / 7.0)
        self.df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        self.df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

        self.df["is_weekend"]    = dow.isin([5, 6]).astype(int)
        self.df["is_post_covid"] = (self.df["date"] >= COVID_START).astype(int)
        self.df["year"]          = self.df["date"].dt.year.fillna(2019).astype(int)

    # ----------------------------------------------------------------
    # Text structure
    # ----------------------------------------------------------------
    def extract_text_metadata(self):
        logger.info("Extracting structural text metadata...")
        self.df["content"] = self.df["content"].fillna("").astype(str)

        self.df["num_mentions"]   = self.df["content"].str.count("<mention>")
        self.df["num_hyperlinks"] = self.df["content"].str.count("<hyperlink>")
        self.df["num_hashtags"]   = self.df["content"].str.count(r"#\w+")
        self.df["num_exclaim"]    = self.df["content"].str.count("!")
        self.df["num_question"]   = self.df["content"].str.count(r"\?")
        self.df["num_caps_words"] = self.df["content"].apply(
            lambda x: sum(1 for w in str(x).split() if len(w) > 1 and w.isupper())
        )
        self.df["char_length"]    = self.df["content"].str.len()
        self.df["word_count"]     = self.df["content"].str.split().str.len().fillna(0).astype(int)
        self.df["avg_word_len"]   = self.df["char_length"] / (self.df["word_count"] + 1)
        self.df["uppercase_ratio"] = self.df["content"].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
        )

        if HAS_EMOJI:
            self.df["num_emojis"] = self.df["content"].apply(
                lambda x: emoji.emoji_count(str(x))
            )
        else:
            self.df["num_emojis"] = self.df["content"].apply(
                lambda x: len(re.findall(
                    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", str(x)))
            )

        # Clean text for embedding model
        self.df["content_clean"] = (
            self.df["content"]
            .str.replace("<mention>", "@user", regex=False)
            .str.replace("<hyperlink>", "[URL]", regex=False)
        )

        if "inferred company" in self.df.columns:
            self.df["inferred company"] = (
                self.df["inferred company"].astype(str).str.lower().str.strip()
            )
        if "username" in self.df.columns:
            self.df["username"] = self.df["username"].astype(str)
            self.df["username_length"]      = self.df["username"].str.len()
            self.df["username_has_digit"]   = self.df["username"].str.contains(r"\d", na=False).astype(int)
            self.df["username_has_under"]   = self.df["username"].str.contains("_", na=False).astype(int)

    # ----------------------------------------------------------------
    # Media regex
    # ----------------------------------------------------------------
    @staticmethod
    def _parse_media(media_str) -> pd.Series:
        duration, views = 0.0, 0.0
        image_url, thumb_url, mp4_url = "", "", ""
        s = "" if pd.isna(media_str) else str(media_str)

        if "Video(" in s:
            m = re.search(r"duration=([\d\.]+)", s)
            if m: duration = float(m.group(1))
            m = re.search(r"views=(\d+)", s)
            if m: views = float(m.group(1))
            m = re.search(r"thumbnailUrl='([^']+)'", s)
            if m: thumb_url = m.group(1)
            m = re.search(r"url='([^']+\.mp4)[^']*'", s)
            if m: mp4_url = m.group(1)
        elif "Gif(" in s:
            m = re.search(r"thumbnailUrl='([^']+)'", s)
            if m: thumb_url = m.group(1)
            m = re.search(r"url='([^']+\.mp4)[^']*'", s)
            if m: mp4_url = m.group(1)
        elif "Photo(" in s:
            m = re.search(r"fullUrl='([^']+)'", s)
            if m: image_url = m.group(1)

        return pd.Series([duration, views, image_url, thumb_url, mp4_url])

    def extract_media_metadata(self):
        logger.info("Parsing media metadata via regex...")
        self.df["media"] = self.df["media"].fillna("")

        self.df["has_photo"] = self.df["media"].str.contains("Photo\\(", regex=True).astype(int)
        self.df["has_video"] = self.df["media"].str.contains("Video\\(", regex=True).astype(int)
        self.df["has_gif"]   = self.df["media"].str.contains("Gif\\(",   regex=True).astype(int)
        self.df["has_media"] = ((self.df["has_photo"] + self.df["has_video"] + self.df["has_gif"]) > 0).astype(int)

        cols = ["video_duration", "video_views", "image_url", "video_thumb_url", "video_mp4_url"]
        self.df[cols] = self.df["media"].apply(self._parse_media)

        # Log-scaled video views — long tail
        self.df["log_video_views"] = np.log1p(self.df["video_views"])

        # Unified URL for downstream image processing (if ever wanted)
        self.df["target_download_url"] = self.df["image_url"].where(
            self.df["image_url"] != "", self.df["video_thumb_url"]
        )

    # ----------------------------------------------------------------
    # Company prior (LEAVE-ONE-OUT smoothed)
    # ----------------------------------------------------------------
    def compute_company_prior(self):
        """
        company_avg_log_likes  =  smoothed leave-one-out mean of log_likes
                                  per brand on the training set, written into
                                  every row.

        For unseen brands at test time → fall back to the global mean.
        """
        if "inferred company" not in self.df.columns:
            return

        if self.is_train:
            logger.info("Computing leave-one-out company priors (training)...")
            global_mean = float(self.df["log_likes"].mean())
            grouped = self.df.groupby("inferred company")["log_likes"]
            group_sum   = grouped.transform("sum")
            group_count = grouped.transform("count")

            # Leave-one-out smoothed mean:
            #   loo_mean = ((sum - row) + prior * global_mean) / ((count - 1) + prior)
            loo = ((group_sum - self.df["log_likes"]) + SMOOTHING_PRIOR * global_mean) / \
                  ((group_count - 1).clip(lower=0) + SMOOTHING_PRIOR)
            self.df["company_avg_log_likes"] = loo
            self.df["company_tweet_count"]   = group_count.astype(int)

            # Build the lookup table for downstream test-time use
            agg = self.df.groupby("inferred company").agg(
                mean_log_likes=("log_likes", "mean"),
                count=("log_likes", "count"),
            )
            self.company_stats = {
                "per_company": agg.to_dict("index"),
                "global_mean_log_likes": global_mean,
                "smoothing_prior": SMOOTHING_PRIOR,
            }
        else:
            logger.info("Applying saved company priors (inference)...")
            per_company = self.company_stats.get("per_company", {})
            global_mean = self.company_stats.get(
                "global_mean_log_likes", self.df.get("log_likes", pd.Series([0.0])).mean()
            )
            prior = self.company_stats.get("smoothing_prior", SMOOTHING_PRIOR)

            def lookup(company):
                stats = per_company.get(company)
                if stats is None:
                    return pd.Series([global_mean, 0])
                count = stats["count"]
                smoothed = (stats["mean_log_likes"] * count + global_mean * prior) / (count + prior)
                return pd.Series([smoothed, count])

            self.df[["company_avg_log_likes", "company_tweet_count"]] = (
                self.df["inferred company"].apply(lookup)
            )

    # ----------------------------------------------------------------
    # Main pipeline
    # ----------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        logger.info("Starting Phase 1 feature pipeline...")
        self.transform_target()
        self.engineer_cyclical_time()
        self.extract_text_metadata()
        self.extract_media_metadata()
        self.compute_company_prior()
        logger.info(f"Phase 1 complete: {len(self.df)} rows × {len(self.df.columns)} columns.")
        return self.df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Expected the raw train CSV with columns "
            "[id, date, likes, content, username, media, inferred company]."
        )

    logger.info(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"  {len(df)} rows loaded.")

    builder = TabularFeatureBuilder(df, is_train=True)
    processed = builder.run()

    processed.to_csv(OUTPUT_FEATURES, index=False)
    logger.info(f"Saved features → {OUTPUT_FEATURES}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(builder.company_stats, OUTPUT_STATS)
    logger.info(f"Saved company stats → {OUTPUT_STATS}")
    logger.info(
        f"  brands tracked: {len(builder.company_stats['per_company'])}, "
        f"global_mean_log_likes: {builder.company_stats['global_mean_log_likes']:.3f}"
    )


if __name__ == "__main__":
    main()
