"""
02_embed.py
===========
Phase 2 of the Tweet Likes Prediction pipeline.

Generates 384-dim sentence-transformer embeddings of the cleaned tweet text.
MiniLM-L6-v2 was chosen over BGE-Base for laptop friendliness — it runs at
~600 tweets/sec on CPU and uses ~120 MB of RAM. The information loss vs BGE
is real but acceptable on this task because the tabular features carry most
of the regression signal.

Outputs:
  embeddings.npy      — float32, shape (n_rows, 384), row-aligned with features
  embedding_index.npy — int64, tweet ids in the same order (paranoia check)

Usage:
    python 02_embed.py
"""

import logging
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_FEATURES = "features_train.csv"
OUTPUT_EMB     = "embeddings_train.npy"
OUTPUT_IDS     = "embeddings_train_ids.npy"
MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE     = 128


def embed_texts(texts: list[str]) -> np.ndarray:
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Encoding {len(texts)} texts (batch_size={BATCH_SIZE})...")
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def main():
    if not os.path.exists(INPUT_FEATURES):
        raise FileNotFoundError(f"{INPUT_FEATURES} not found — run 01_features.py first.")

    df = pd.read_csv(INPUT_FEATURES, usecols=["id", "content_clean"])
    logger.info(f"Loaded {len(df)} rows from {INPUT_FEATURES}")

    texts = df["content_clean"].fillna("").astype(str).tolist()
    emb   = embed_texts(texts)

    np.save(OUTPUT_EMB, emb)
    np.save(OUTPUT_IDS, df["id"].values)
    logger.info(f"Saved embeddings: {OUTPUT_EMB}  shape={emb.shape}")
    logger.info(f"Saved id index:   {OUTPUT_IDS}")


if __name__ == "__main__":
    main()
