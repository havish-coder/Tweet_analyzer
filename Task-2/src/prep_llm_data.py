"""
Converts train_enriched.csv → llm_train_data.jsonl using the shared
prompt builder from prompt_utils.py.  Re-run this after enrich_vlm.py
to pick up any new vlm_description values.

Each JSONL record carries the row 'id' so finetune_qwen.py can verify
JSONL↔CSV alignment by id instead of trusting row order.

Env flags:
  EXAMPLES_IN_TRAIN=1  — include up to 3 of the brand's *earlier* tweets in each
                         prompt (temporal leave-one-out, no leakage). Only enable
                         if you will re-finetune AND run eval.py with USE_RETRIEVAL=1,
                         so train and inference prompts stay consistent.
"""

import json
import os
import pandas as pd
from prompt_utils import build_messages

EXAMPLES_IN_TRAIN = os.environ.get("EXAMPLES_IN_TRAIN", "0").lower() in {"1", "true", "yes"}
N_EXAMPLES = 3


def main():
    print("Loading data...")
    for path in ["data/train_enriched.csv", "data/train.csv", "train_enriched.csv", "train.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} rows from {path}")
            break
    else:
        print("ERROR: no training CSV found.")
        return

    df = df[df["content"].notna() & (df["content"].str.strip() != "")]
    print(f"  After content filter: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Per-row few-shot examples: the brand's up-to-3 most recent EARLIER tweets.
    # Temporal ordering means a row never sees its own or future tweets — no leakage.
    examples_by_index = {}
    if EXAMPLES_IN_TRAIN:
        print(f"  Building temporal few-shot examples (up to {N_EXAMPLES} per row)...")
        brand_col = "inferred company" if "inferred company" in df.columns else "username"
        work = df.copy()
        work["_dt"] = pd.to_datetime(work["date"], errors="coerce")
        for _, group in work.groupby(brand_col):
            group = group.sort_values("_dt", kind="stable")
            history = []
            for idx, row in group.iterrows():
                examples_by_index[idx] = list(history[-N_EXAMPLES:])
                history.append(str(row["content"]))

    records = []
    for idx, row in df.iterrows():
        messages = build_messages(
            row.to_dict(),
            include_response=True,
            examples=examples_by_index.get(idx),
        )
        records.append({"id": row.get("id", idx), "messages": messages})

    out_path = "data/llm_train_data.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(f"\nSaved {len(records)} records to {out_path}")
    print(f"  Few-shot examples in prompts: {'ON' if EXAMPLES_IN_TRAIN else 'off'}")
    print("\n--- Example ---")
    # ensure_ascii=True avoids Windows cp1252 console encoding errors
    print(json.dumps(records[0], indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
