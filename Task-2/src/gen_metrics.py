"""
Generation metrics shared by eval.py and select_checkpoint.py:
BLEU 1-4 (nltk), ROUGE-1/2/L (rouge-score), CIDEr (pycocoevalcap),
plus bootstrap confidence intervals (valid only because sampling is random).
"""

import numpy as np


def compute_bleu(preds: list, refs: list) -> dict:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoother = SmoothingFunction().method3
        refs_tok = [[r.lower().split()] for r in refs]
        hyps_tok = [p.lower().split() for p in preds]
        return {
            "BLEU-1": corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=smoother),
            "BLEU-2": corpus_bleu(refs_tok, hyps_tok, weights=(.5, .5, 0, 0), smoothing_function=smoother),
            "BLEU-3": corpus_bleu(refs_tok, hyps_tok, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoother),
            "BLEU-4": corpus_bleu(refs_tok, hyps_tok, weights=(.25, .25, .25, .25), smoothing_function=smoother),
        }
    except ImportError:
        print("  [warn] nltk not installed — pip install nltk")
        return {}


def compute_rouge(preds: list, refs: list) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
        return {
            "ROUGE-1": float(np.mean([s["rouge1"].fmeasure for s in scores])),
            "ROUGE-2": float(np.mean([s["rouge2"].fmeasure for s in scores])),
            "ROUGE-L": float(np.mean([s["rougeL"].fmeasure for s in scores])),
        }
    except ImportError:
        print("  [warn] rouge-score not installed — pip install rouge-score")
        return {}


def compute_cider(preds: list, refs: list) -> dict:
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        gts = {i: [r] for i, r in enumerate(refs)}
        res = {i: [p] for i, p in enumerate(preds)}
        score, _ = scorer.compute_score(gts, res)
        return {"CIDEr": float(score)}
    except ImportError:
        print("  [warn] pycocoevalcap not installed — pip install pycocoevalcap")
        return {}
    except Exception as e:
        print(f"  [warn] CIDEr failed: {e}")
        return {}


def bootstrap_ci(preds: list, refs: list, n_boot: int = 1000, seed: int = 42) -> dict:
    """
    95% bootstrap CIs for BLEU-1 and ROUGE-L by resampling (pred, ref) pairs.
    Only meaningful when the evaluated rows are a RANDOM sample of the test set.
    """
    rng = np.random.default_rng(seed)
    n = len(preds)
    if n < 10:
        return {}

    b1_samples, rl_samples = [], []
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
    except ImportError:
        return {}

    smoother = SmoothingFunction().method3
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # Precompute per-sentence pieces so 1000 resamples stay fast
    refs_tok = [[r.lower().split()] for r in refs]
    hyps_tok = [p.lower().split() for p in preds]
    rl_per = np.array([scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)])

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b1_samples.append(corpus_bleu(
            [refs_tok[i] for i in idx], [hyps_tok[i] for i in idx],
            weights=(1, 0, 0, 0), smoothing_function=smoother,
        ))
        rl_samples.append(rl_per[idx].mean())

    def ci(samples):
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return float(lo), float(hi)

    return {"BLEU-1_ci95": ci(b1_samples), "ROUGE-L_ci95": ci(rl_samples)}
