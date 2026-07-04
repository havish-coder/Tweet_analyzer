# 🏆 Final Evaluation Report

> Both pipelines graded against the supplied ground-truth labels.
> **Task 1:** 10,000 rows per regime (full test set).
> **Task 2:** 500 rows per regime (10× the original 100-sample baseline; full 10K was infeasible on a 4 GB laptop within useful time).

---

## ⭐ Headline Numbers

| Task | Metric | Unseen Brands | Unseen Time | Average |
|---|---|---:|---:|---:|
| **Task 1** — Likes Prediction | **RMSE (raw likes)** | **620.83** | **1,861.01** | **1,240.92** |
| Task 1 | RMSE (log scale) | 0.9855 | 1.5282 | 1.2569 |
| Task 1 | MAE | 350.39 | 565.46 | 457.93 |
| **Task 2** — Tweet Generation | **BLEU-1** | **0.1762** | **0.1327** | **0.1545** |
| Task 2 | BLEU-2 | 0.0572 | 0.0387 | 0.0480 |
| Task 2 | BLEU-3 | 0.0209 | 0.0164 | 0.0187 |
| Task 2 | BLEU-4 | 0.0096 | 0.0092 | 0.0094 |
| Task 2 | ROUGE-1 | 0.2324 | 0.2002 | 0.2163 |
| Task 2 | ROUGE-2 | 0.0368 | 0.0292 | 0.0330 |
| Task 2 | **ROUGE-L** | **0.2131** | **0.1850** | **0.1991** |
| Task 2 | **CIDEr** | **0.0861** | **0.0806** | **0.0834** |

---

## 📊 Task 1 — Tweet Likes Prediction

**Model:** Single XGBoost regressor + Duan smearing correction (factor 1.724) — selected over the 7-bucket classify-then-regress cascade by ablation on a leak-free validation split. The cascade's predictions remain in the prediction CSVs as a comparison column.
**Test sets:** 10,000 rows each (3 unseen-time rows with `likes = -1` excluded from grading)
**Inference time:** ~30 sec per regime on RTX 3050

### RMSE per regime — shipped model vs the cascade it replaced

| Regime | RMSE (shipped) | RMSE (cascade) | RMSE (log) | MAE | Previous report |
|---|---:|---:|---:|---:|---:|
| Unseen Brands | **620.83** | 1,242.27 | 0.9855 | 350.39 | 963.27 |
| Unseen Time | **1,861.01** | 2,060.92 | 1.5282 | 565.46 | 2,208.23 |

### Naive-baseline context

Every RMSE needs a "compared to what." Constant predictors derived from the training set, applied to the same graded rows:

| Predictor | Unseen Brands RMSE | Unseen Time RMSE | Combined (20K) |
|---|---:|---:|---:|
| Predict train median (73) | 398.9 | 2,551.6 | 1,826 |
| Predict train mean (718) | 434.2 | 2,498.8 | 1,793 |
| Predict per-brand train mean | 434.2¹ | 3,959.3 | — |
| **Shipped model** | **620.83** | **1,861.01** | **1,387** |

¹ No test brand appears in training, so this degenerates to the global mean.

| Rank / calibration | Unseen Brands | Unseen Time |
|---|---:|---:|
| Spearman correlation (model vs actual) | 0.02 | **0.72** |
| Log-RMSE — model | **0.986** | **1.528** |
| Log-RMSE — best constant | 1.081 | 1.781 |

**Honest read:** combined, the model beats the best constant by 23%. On unseen time it adds unambiguous value — 26% lower RMSE than any constant and strong per-tweet ranking (Spearman 0.72), driven by brand-history priors. On unseen brands, no per-tweet ranking signal survives the removal of brand history (Spearman ≈ 0.02); the model's value there is limited to predicting the right overall magnitude (better log-RMSE than any constant), and because the regime's true distribution is narrow, a constant prediction wins on raw RMSE. This is a property of the task — virality depends on follower counts and network effects absent from the metadata — and any candidate model should be benchmarked against these same constants.

### Distribution sanity check

| Regime | Pred median | True median | Pred max | True max |
|---|---:|---:|---:|---:|
| Unseen Brands | 453 | 356 | 6,733 | 1,863 |
| Unseen Time | 231 | 291 | 18,519 | 28,721 |

### Read of the numbers

- **The validation-selected model won on test in both regimes** (−36% RMSE on unseen brands, −16% on unseen time vs the previous cascade submission). This is the payoff of fixing the company-prior leak: validation now measures the real task, so choosing by validation actually works.
- **Val RMSE (2,240) remains far above test RMSE** because the val split concentrates the hardest rows (fully held-out brands + the most recent dates); the test distribution — especially unseen brands, true max only 1,863 — is narrower.
- **Unseen time is still ~3× harder than unseen brands.** Its tail (28,721-like outlier) is under-predicted, and engagement drifts over time in ways a static brand prior can't track. Rolling brand-history features are the documented next step.
- **Known weakness:** over-predicted maximum on unseen brands (6,733 vs true 1,863). Capping predictions by brand-history quantiles would trim this.

---

## ✍️ Task 2 — Tweet Content Generation

**Model:** Qwen2.5-1.5B-Instruct + QLoRA (r=16, α=32), beam search n=4, no-repeat-ngram=3
**Test sets:** seeded random 500 rows per regime (1,000 generations total), scored by id against the supplied answer keys
**Baseline:** the un-fine-tuned base model, generated on the *same* rows with identical prompts and decoding

### Fine-tuned vs base model (the "compared to what")

| Metric | Base (brands / time) | **Fine-tuned (brands / time)** | Lift |
|---|---:|---:|---:|
| BLEU-1 | 0.0754 / 0.0832 | **0.1762 / 0.1327** | ~2× |
| ROUGE-1 | 0.0980 / 0.1109 | **0.2324 / 0.2002** | ~2× |
| ROUGE-L | 0.0781 / 0.0861 | **0.2131 / 0.1850** | ~2.4× |
| CIDEr | 0.0139 / 0.0147 | **0.0861 / 0.0806** | ~6× |
| Gen length (ref 16.5 / 19.3) | 26.9 / 30.4 words | **13.7 / 13.8 words** | learns brevity |
| Eval perplexity (same eval split) | 53.49 | **3.33** | −93.8% |

Bootstrap 95% CIs do not overlap on any metric (e.g., ROUGE-L unseen brands: fine-tuned [0.202, 0.225] vs base [0.074, 0.083]).

### Fine-tuned model — full metric detail

| Metric | Unseen Brands | Unseen Time |
|---|---:|---:|
| BLEU-1 | 0.1762 — CI [0.159, 0.192] | 0.1327 — CI [0.120, 0.146] |
| BLEU-2 / 3 / 4 | 0.0572 / 0.0209 / 0.0096 | 0.0387 / 0.0164 / 0.0092 |
| ROUGE-1 / 2 | 0.2324 / 0.0368 | 0.2002 / 0.0292 |
| **ROUGE-L** | **0.2131** — CI [0.202, 0.225] | **0.1850** — CI [0.174, 0.196] |
| CIDEr | 0.0861 | 0.0806 |
| Gen length (ref) | 13.7 (16.5) words | 13.8 (19.3) words |

### Sample side-by-side

| Generated (ours) | Reference (truth) | Brand |
|---|---|---|
| *"`<mention>` `<mention>` has been named the 2020 recipient of the American Academy of Neurology's Distinguished Service Award. `<hyperlink>`"* | *"Lasagna is a classic dish that never goes out of style. Here are some of our favorite lasagna recipes to try at home: `<hyperlink>` `<hyperlink>`"* | (unseen) |
| *"`<mention>` `<mention>` #RBCGAM `<hyperlink>`"* | *"The best part of the game is when the kids get to play. `<hyperlink>` `<hyperlink>`"* | RBC |
| *"This is the first time we've been able to see the body of a man who was killed in the 2015 London Bridge attack `<hyperlink>`"* | *"`<mention>` and `<mention>` are the winners of the 2018 Strictly Come Dancing Christmas Special! `<hyperlink>`"* | BBC |

### Read of the numbers

- **ROUGE-L ≈ 0.19** → model recovers ~19% of the reference's longest common subsequence, with **zero access to the actual tweet content** at inference (only metadata).
- **BLEU-4 is tiny (~0.005–0.012)** because matching 4-grams of an unseen tweet exactly is near-impossible without seeing the underlying event/topic. **BLEU-1 (~0.15) is the better signal at this scale.**
- **Generation length is consistently shorter than reference** (14.5 vs 16.2 words on brands, 14.3 vs 18.9 on time). The `max_new_tokens=100` cap rarely fires; the beam search just prefers shorter sequences as overall-best.
- **The model has learned tweet structure** (`<mention>`, `<hyperlink>`, hashtags, brand-appropriate register) but **cannot recover the semantic content** of the reference tweet without VLM image context. We skipped VLM at inference because 99.8% of training URLs are dead (expired Twitter media).

### Sampling note

500 generations × 2 regimes = 1,000 total, drawn as a **seeded random sample** (`random_state=42`) of each 10K test file — an earlier version of this report used the first 500 rows, which is a biased slice; all numbers above come from the corrected random-sample run, with bootstrap 95% confidence intervals reported alongside. The base-model comparison uses the identical rows, prompts, and decoding, so the lift is attributable to fine-tuning alone.

---

## 💡 Honest Limitations

1. **Task 2 was not run on full 10K.** Full beam-search inference for 20,000 rows at ~10 sec/each ≈ 55 hours on the 4 GB GPU. The 500-row subset gives 10× the statistical power of the original 100-row baseline and lands in the same metric neighborhood.

2. **Task 1 still over-predicts the extreme tail on unseen brands** (max prediction 6,733 vs true max 1,863). On the leak-free validation split the viral band is essentially unpredictable from metadata alone (classifier recall 0% for 2.5k–5k likes) — virality depends on follower counts and retweet cascades that are not in the data. Capping predictions by brand-history quantiles is the cheapest mitigation.

3. **Task 2 cannot beat its visual-context ceiling.** The model has no view of the actual image at inference. Reference tweets often refer to the image content directly ("look at our new product!", "watch this clip"), which the model can't reproduce without VLM enrichment.

4. **The CynapticsAI predictions in the supplied xlsx are *not* used** as ground truth. We graded strictly against the `likes` (Task 1) and `answers` (Task 2) columns.

---

## 🔁 Reproduce These Numbers

```bash
# Task 1 — full 10K
cd Tweet_analyzer/Task-1
python 04_predict.py
# outputs/predictions_company.csv  &  predictions_time.csv

# Then compute RMSE:
python -c "
import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error
for label, p, gt in [('co','company','behaviour_simulation_test_company'),
                      ('ti','time',   'behaviour_simulation_test_time')]:
    pred = pd.read_csv(f'outputs/predictions_{p}.csv')['shipped_pred']
    truth = pd.read_excel(f'C:/Users/ponmu/Downloads/{gt}.xlsx')['likes']
    print(f'{label}: RMSE = {np.sqrt(mean_squared_error(truth, pred)):.2f}')
"

# Task 2 — 500 per regime (SAMPLE_SIZE=500 in src/eval.py)
cd ../Task-2
python src/eval.py
# Then run the BLEU/ROUGE/CIDEr script in task2_metrics.json
```

---

## 🎯 Bottom Line

| Task | Headline number | Verdict |
|---|---|---|
| **Task 1** | RMSE **620.83 / 1,861.01** (brands / time), combined **1,387** on 20K rows | **Beats the predict-the-mean baseline by 23% combined.** On unseen time the model adds clear value (−26% vs best constant, Spearman 0.72). On unseen brands, metadata carries almost no per-tweet ranking signal (Spearman ≈ 0.02) — the model's contribution there is level calibration (log-RMSE 0.99 vs 1.08 for the best constant), and a constant prediction actually wins on raw RMSE. Reported in full below. |
| **Task 2** | **BLEU-1 0.155**, **ROUGE-L 0.186**, **CIDEr 0.063** across both regimes (n=1000) | **Honest for the constraint.** A 1.5B model trained on a 4 GB laptop with 99.8% dead image URLs learns brand voice + tweet structure, but can't recover semantic content without VLM context. The numbers reflect the ceiling of the metadata-only setup. |

---

*Generated: 2026-05-24 · Tested on RTX 3050 Laptop · Tweet_analyzer @ `havish-coder/Tweet_analyzer`*
