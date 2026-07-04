"""
Score Task-2 generations against the supplied ground-truth answers.

The competition test CSVs carry no content column; the ground-truth xlsx files
(one 'answers' column, row-aligned with the test CSVs) live outside the repo.
Predictions are joined to references via id -> row position (id - 1), which
works for any sample of rows — including the random samples eval.py now draws.

Usage:
    python score_vs_gt.py [predictions_dir]     # default: Task-2/outputs
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Task-2", "src"))
from gen_metrics import bootstrap_ci, compute_bleu, compute_cider, compute_rouge

GT_BRANDS = "C:/Users/ponmu/Downloads/content_simulation_test_company.xlsx"
GT_TIME   = "C:/Users/ponmu/Downloads/content_simulation_test_time.xlsx"


def score(pred_csv: str, gt_xlsx: str, label: str):
    if not os.path.exists(pred_csv):
        print(f"[skip] {pred_csv} not found")
        return
    p = pd.read_csv(pred_csv)
    gt = pd.read_excel(gt_xlsx)

    ids = p["id"].astype(int).values
    assert ids.min() >= 1 and ids.max() <= len(gt), \
        f"id range [{ids.min()}, {ids.max()}] outside GT rows (1..{len(gt)})"
    refs = [str(gt["answers"].iloc[i - 1]) for i in ids]
    preds = [str(x) for x in p["generated"].fillna("")]

    print(f"\n{'=' * 66}\n{label}  (n={len(preds)}, {pred_csv})\n{'=' * 66}")
    for name, val in compute_bleu(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, val in compute_rouge(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, val in compute_cider(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, (lo, hi) in bootstrap_ci(preds, refs).items():
        print(f"  {name:<12}: [{lo:.4f}, {hi:.4f}]")
    gen_len = np.mean([len(x.split()) for x in preds])
    ref_len = np.mean([len(x.split()) for x in refs])
    print(f"  Gen length  : {gen_len:.1f} words (ref: {ref_len:.1f})")


def main():
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "Task-2/outputs"
    score(os.path.join(pred_dir, "predictions_unseen_brands.csv"), GT_BRANDS,
          "Unseen Brands")
    score(os.path.join(pred_dir, "predictions_unseen_time.csv"), GT_TIME,
          "Unseen Time")


if __name__ == "__main__":
    main()
