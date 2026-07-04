# Task 1 — Interview Notes

> Everything you need to explain this project convincingly, in the order an
> interview usually unfolds. Numbers here match `models/metrics.json` and
> `FINAL_REPORT.md` — one source of truth.

---

## 1. The 30-second pitch

> "I predict how many likes a marketing tweet gets from its metadata alone.
> I designed a tiered classify-then-regress cascade, built a leak-free
> validation split that mirrors the competition's two test regimes, and ran a
> pre-registered ablation against a simple baseline. The baseline — a single
> XGBoost regressor with a smearing correction — won, so I shipped it, and the
> validation choice held up on 20,000 test rows: RMSE 621 on unseen brands and
> 1,861 on an unseen time period, beating the predict-the-mean baseline by 23%."

The two words that make this pitch senior-sounding: **pre-registered** (I fixed
the number to beat *before* running the experiment) and **shipped the winner**
(I didn't keep my pet architecture after it lost).

---

## 2. The problem and the data

- **Input per tweet:** date, content text, username, media URL, inferred company.
- **Target:** number of likes. Heavy-tailed: train median **73**, mean **718**,
  max **254,931**. A handful of viral tweets dominate any squared-error metric.
- **Test:** 10K rows of **unseen brands** + 10K rows of an **unseen (later) time
  period**. These are two different generalization problems and I treat them
  separately everywhere.
- **What's NOT in the data:** follower counts, paid promotion, retweet cascades —
  the things that actually cause virality. This puts a hard ceiling on any
  model, and being able to say so (with evidence, §7) is a strength, not an excuse.

---

## 3. The feature vector — memorize this breakdown

**416 dimensions = 384 embedding + 32 engineered.**

| Group | Count | Features |
|---|---|---|
| Temporal | 9 | hour/day-of-week/month as sin+cos pairs (6), is_weekend, is_post_covid, year |
| Text structure | 11 | counts of mentions, hyperlinks, hashtags, `!`, `?`, ALL-CAPS words, emojis; char length, word count, avg word length, uppercase ratio |
| Username | 3 | length, has-digit, has-underscore |
| Media | 7 | photo/video/gif/any flags, video duration, video views, log(video views) |
| Brand prior | 2 | leave-one-out smoothed mean log-likes, brand tweet count |
| Embedding | 384 | all-MiniLM-L6-v2 sentence embedding of cleaned text |

**Why sin/cos for time:** hour 23 and hour 0 are adjacent in reality but far
apart as raw numbers. `sin(2πh/24), cos(2πh/24)` puts hours on a circle so the
model sees the adjacency.

**The brand prior is the strongest feature** — "how popular is this brand
usually?" It's computed **leave-one-out** (each row's own likes are excluded
from its brand mean) with **smoothing** toward the global mean
(`prior strength = 30 pseudo-rows`), and unseen brands fall back to the smoothed
global mean at test time.

---

## 4. The story arc (tell it in this order)

### Act 1 — the cascade design
Likes are power-law distributed, so one model fitting the whole range gets
pulled in different directions by quiet vs viral tweets. My design: a
**7-tier classifier** (edges at 100/250/500/1k/2.5k/5k likes) routes each tweet
to a **tier-specialist regressor** trained on log-likes within that tier.
At inference I use **soft routing**: instead of trusting the classifier's top
pick, I weight all seven specialists by the classifier's probabilities —
`ŷ = Σ p(k|x) · expm1(r_k(x))` — so a misclassification degrades gracefully
instead of catastrophically.

### Act 2 — the leak I found in my own validation
My brand-prior feature was computed over the **full** training file *before*
the train/val split. So validation rows from "held-out" brands still carried
their brand's real popularity history — but genuinely unseen brands at test
time get only the global fallback. My validation was easier than the real task.
**Fixing it dropped classifier accuracy from 65% → 48%** and made every
validation number honest. Rule I learned: *every statistic must be computable
from training rows only, and the split happens first.*

Related hygiene, same fix: early stopping moved to an inner 10% slice of train
(the eval set had been doing triple duty: early-stopping the classifier, early-
stopping every regressor, AND reporting), and the scaler is fit on train only.

### Act 3 — the ablation
Any architecture must beat the obvious baseline on honest validation. Results
(regime-mirrored val, 1,227 rows):

| Candidate | Val RMSE |
|---|---:|
| **Single XGB regressor + Duan smearing** ← shipped | **2,240** |
| Ensemble 70% single / 30% cascade | 2,244 |
| Cascade soft + per-tier smearing | 2,319 |
| Cascade, soft-routed | 2,341 |
| Single regressor, no smearing | 2,419 |
| Tweedie regressor (raw target) | 2,440 |
| Cascade, hard-routed | 2,451 |
| Cascade + *perfect* classifier (oracle) | 2,111 |

The oracle row is the killer argument: even a PERFECT router only reaches
2,111 — ~6% better than the shipped model. The cascade's ceiling was never
high enough to justify its complexity.

### Act 4 — why the cascade lost (the diagnostics)
- **Per-tier smearing factors came out at 1.00–1.13** vs the single model's
  1.72. Meaning: bucketing already acts as an implicit smearing correction
  (each tier spans a narrow range → tiny log-residual variance → tiny bias).
  So the cascade's handicap was never retransformation bias — it was
  **routing error** (48%-accurate classifier).
- **Ensembling failed monotonically** (more cascade weight → worse). The
  cascade's errors are a *noisier superset* of the single model's errors, not
  different errors — both models see identical features and use the identical
  algorithm, so there was no informational diversity to harvest.

### Act 5 — ship and verify
Shipped the validation winner *without touching test*, then graded on the
answer key: **621 / 1,861 RMSE** (previous cascade submission: 963 / 2,208).
The validation-selected model won on both regimes — proof the honest split
works as a leaderboard proxy.

---

## 5. Duan smearing — be able to derive this

**The problem.** We train on `log1p(likes)` (needed for a target spanning
0→255K). The model predicts the *mean of the logs*. But
`exp(mean of logs) < mean` — always — because exp is convex (Jensen's
inequality). Worked example: tweets get 100 or 10,000 likes, 50/50.
- True mean: (100 + 10,000)/2 = **5,050**
- exp(mean of logs): exp((4.6 + 9.2)/2) = exp(6.9) ≈ **1,000**

Same data, 5× underprediction — purely from where you take the average. The
wider the residual spread, the worse it gets, and likes are extremely spread.

**The fix (Duan 1983).** Measure the gap empirically on held-out data:
`S = mean(exp(actual_log − predicted_log))`. If the model were perfect, every
residual is 0 and S=1. Our S = **1.724**. Final prediction:
`ŷ = exp(pred_log) · S − 1`. Three lines of code, worth ~180 RMSE — the best
effort-to-impact ratio in the repo.

**Why it matters for RMSE specifically:** RMSE is minimized by the conditional
**mean**; naive `expm1(pred_log)` estimates something closer to the conditional
**median**. Smearing moves the estimate from median-ish to mean. Matching
evidence: our *median* absolute error barely changed (136→137) while RMSE
dropped a lot — the correction moved the tails, not the middle. And log-scale
RMSE got slightly *worse* — expected, you can't be optimal on both scales; you
pick the graded one.

**Caveats to volunteer if asked:** residuals for S must come from data the
model didn't train on (ours: the inner early-stopping slice); one global S
assumes homoscedastic residuals — otherwise compute S per segment.

---

## 6. The validation split — "regime-mirroring" defined

**My term, standard construction.** Say it like this: *"a grouped holdout on
brand — 5% of brands held out entirely, GroupShuffleSplit in sklearn terms —
unioned with a temporal holdout of the latest 5% of dates, so validation
mirrors the competition's two test regimes."* Seen brands appearing in the
temporal slice is deliberate: the real unseen-time test also contains seen
brands.

---

## 7. Baselines — the "good compared to what?" answer

Constant predictors from the training set, graded on the same test rows:

| Predictor | Brands RMSE | Time RMSE | Combined |
|---|---:|---:|---:|
| Predict train median (73) | 398.9 | 2,551.6 | 1,826 |
| Predict train mean (718) | 434.2 | 2,498.8 | 1,793 |
| **Shipped model** | **620.8** | **1,861.0** | **1,387** |

| Rank / calibration | Brands | Time |
|---|---:|---:|
| Spearman (model vs actual) | **0.02** | **0.72** |
| Log-RMSE model / best constant | 0.99 / 1.08 | 1.53 / 1.78 |

**Own this proactively:** combined, we beat the best constant by 23%. On
unseen **time** the model adds unambiguous value (−26% RMSE, Spearman 0.72 —
it genuinely ranks tweets, powered by brand history). On unseen **brands** a
constant wins on raw RMSE and our Spearman is ~0 — without brand history the
metadata simply contains no per-tweet ranking signal (virality lives in
follower counts we don't have). Our value there is *level calibration* only
(we beat every constant on log-RMSE). Volunteering this before the interviewer
finds it is the single most credibility-building move available.

---

## 8. Numbers to memorize (one card)

| | |
|---|---|
| Dataset | 17,331 train tweets, 194 brands; 2×10K test |
| Target stats | median 73 · mean 718 · max 254,931 |
| Features | 416 = 384 MiniLM + 32 engineered |
| Val split | 1,227 rows = 379 (10 held-out brands) + 848 (latest dates) |
| Classifier accuracy | 48.2% honest (was 65% with the leak) |
| Smearing factor | 1.724 |
| Val RMSE: shipped / cascade / oracle | 2,240 / 2,341 / 2,111 |
| Test RMSE: brands / time / combined | 620.8 / 1,861.0 / 1,387 |
| Naive-baseline combined | 1,793 (we're −23%) |
| Spearman: brands / time | 0.02 / 0.72 |

---

## 9. Hard questions, model answers

**"Is 621 RMSE good?"**
→ "Compared to 963 from my first submission and 434 from predict-the-mean on
that regime — so no, on unseen brands raw RMSE alone doesn't beat a constant,
and I can tell you exactly why: Spearman there is ~0 because virality depends
on follower data we don't have. Where the metadata does carry signal — unseen
time — I beat every constant by 26% with 0.72 rank correlation."

**"Why did your fancy architecture lose to a baseline?"**
→ "Two reasons I can prove: the router is only 48% accurate so it injects
noise, and the oracle experiment shows even a perfect router caps the gain at
6%. The cascade solved a problem — retransformation bias — that a one-scalar
smearing correction solves for free."

**"Could there be leakage?"**
→ "There *was* — I found it and fixed it." (Tell Act 2. This is your best
answer in the whole interview; don't rush it.)

**"Why XGBoost and not a neural net?"**
→ "17K rows of mostly-tabular features is squarely gradient-boosting
territory; trees handle the mixed scales and interactions natively, train in
minutes on CPU, and my 4 GB GPU budget was spent on Task 2's LLM."

**"Why log-transform the target?"**
→ "Range 0→255K; squared error on raw likes would make the model fit five
viral tweets and ignore everything else. But log-training creates
retransformation bias, which I correct with Duan smearing." (Then §5.)

**"What would you do next?"**
→ "Rolling brand-history features — recent-N engagement stats computed from
each brand's earlier tweets only. Engagement is autocorrelated, and unseen-time
is where the remaining error lives. Then prediction capping by brand-history
quantiles, and Optuna with brand-grouped CV."

**"What's your single biggest takeaway?"**
→ "Fix validation before improving models. Every real gain in this project —
picking smearing over the cascade, the test-set wins — became possible only
after the validation split stopped lying to me."

---

## 10. Weaknesses to own (before they're found)

1. Unseen-brands ranking is ~random (Spearman 0.02) — task property, but ours to disclose.
2. Max prediction overshoots on unseen brands (6,733 vs true max 1,863) — capping is the known fix.
3. No hyperparameter search yet — params are hand-picked; grouped-CV Optuna is roadmapped.
4. Test grading is self-run against the released answer key — no leaderboard percentile exists.
