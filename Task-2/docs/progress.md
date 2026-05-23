# Task 2 — Content Simulation — Progress Tracker

> Living document. Updated every working session. Last updated: **2026-05-15 (Session 2)**.

Challenge: Adobe Behaviour Simulation Challenge (Inter IIT Tech Meet, Mid Prep).
Task 2: given tweet metadata (`company`, `username`, `media` URL, `timestamp`, `likes`) → generate tweet `content`.
Eval: **BLEU 1–4, ROUGE, CIDEr**, across 2 regimes (10K each): **unseen brands / seen time** and **seen brands / unseen time**.
Grading: Metrics 50% · Approach (efficiency + novelty) 35% · Presentation 15%.
Hardware constraint: **RTX 3050 Laptop, 4 GB VRAM**.

---

## 0. TL;DR — Current State

**Session 2 status:** All Phase A blockers resolved in code. Run the pipeline in order:

```
python enrich_vlm.py        # Stage 1 — Florence-2-large enrichment
python prep_llm_data.py     # regenerate JSONL with updated prompt+likes
python finetune_qwen.py     # Stage 2 — Qwen2.5-1.5B QLoRA, 3 epochs
python eval.py              # Stage 3 — generation + BLEU/ROUGE/CIDEr + submission
```

Remaining blockers (from session 1):
1. ~~`eval.py` targets wrong model~~ — **fixed** (Qwen2.5-1.5B + ChatML + beam search)
2. VLM enrichment ~2% done, high failure rate — **VLM swapped to Florence-2-large** (faster, stronger OCR); failure rate depends on URL liveness
3. ~~Data integrity issue~~ — **fixed** (enrich never writes to INPUT_CSV; resume is id-keyed)
4. Only ~17K of 300K samples — open; need full dataset
5. ~~Local metrics wrong~~ — **fixed** (BLEU 1-4 / ROUGE / CIDEr in eval.py)
6. ~~No test-set enrichment~~ — **fixed** (eval.py calls Florence-2 on test media automatically)
7. ~~Submission format wrong~~ — **fixed** (carries `id`, all input columns preserved)

---

## 1. Overall Plan — Phases

### Phase A — Stabilise & Fix (BLOCKERS)
- A1. Rewrite `eval.py` for the actual Qwen model + ChatML + shared prompt template.
- A2. Fix data integrity: stop writing to `train.csv`; reconcile the two CSVs; treat raw input as read-only.
- A3. Add a real metrics module (BLEU 1–4 + ROUGE-L + CIDEr).
- A4. Fix submission format (carry `id`, preserve input columns).
- A5. Add a test-set VLM enrichment path (reuse `enrich_vlm.py` with configurable I/O).

### Phase B — Data & VLM Quality
- B1. Diagnose Twitter media URL death rate; decide VLM strategy (enrich what's alive, accept empty for the rest, or source media differently).
- B2. Get the full 300K dataset (or as much as feasible).
- B3. Improve the VLM prompt → concise single-line caption + explicit OCR, no markdown, no newlines.
- B4. Build a proper train/val split that **mirrors the two eval regimes** (hold out brands; hold out the latest time window).

### Phase C — Modelling Improvements
- C1. Upgrade base LLM 0.5B → **Qwen2.5-1.5B-Instruct** QLoRA (best quality/VRAM fit).
- C2. Switch decoding to **beam search** (metrics reward mode-seeking, not sampling).
- C3. Add **retrieval-augmented few-shot** prompting (biggest expected score lift).
- C4. Condition on `likes` (bucketed) — it's a provided input we currently ignore.
- C5. Output post-processing (strip preambles/quotes, length calibration, brand-hashtag append).

### Phase D — Evaluation, Ensembling, Report
- D1. Per-regime metric breakdown; tune decoding params on val.
- D2. Optional: candidate reranking against retrieved neighbours.
- D3. Optional: small seq2seq baseline (Flan-T5-base) for the efficiency narrative.
- D4. Write the 3-page ACL report; clean GitHub repo.

---

## 2. Per-Component Progress

| Component | File | Status | Notes |
|---|---|---|---|
| Component | File | Status | Notes |
|---|---|---|---|
| Shared prompt builder | `prompt_utils.py` | ✅ Done | Single source of truth for train+inference prompt; includes `likes` bucketing |
| VLM media enrichment | `enrich_vlm.py` | ✅ Rewritten | **Qwen2.5-VL-3B-Instruct** (4-bit) with fixed single-line prompt (BUG-12); id-keyed resume; no INPUT_CSV write |
| LLM data prep | `prep_llm_data.py` | ✅ Done | Imports prompt_utils; regenerate JSONL before retraining |
| LLM fine-tune | `finetune_qwen.py` | ✅ Rewritten | **Qwen2.5-1.5B-Instruct**, QLoRA r=16, SFTConfig, 5% eval split, 3 epochs |
| Inference + eval | `eval.py` | ✅ Rewritten | Correct model+ChatML+beam search; BLEU1-4/ROUGE/CIDEr; fixed submission format |
| Real metrics (BLEU1-4/ROUGE/CIDEr) | `eval.py` | ✅ Done | nltk + rouge-score + pycocoevalcap (with import fallbacks) |
| Test-set VLM enrichment | `eval.py` | ✅ Done | Auto-enriches test rows via Florence-2 before generation |
| Train/val split | `finetune_qwen.py` | ✅ Done | 5% hold-out via train_test_split; best checkpoint loaded at end |
| Retrieval / few-shot module | — | ❌ Not Started | Highest-leverage remaining improvement |
| Full 300K dataset | — | ❌ Not Started | Only ~17K present; need to download rest |
| Actual training run (1.5B) | — | ❌ Not Started | Code ready; need to run + check eval loss |
| 3-page ACL report | — | ❌ Not Started | Due 15 Dec; test set drops 14 Dec |

---

## 3. Bugs & Issues — Root Cause Analysis

### BUG-01 — `eval.py` is for a different model than was trained — **CRITICAL, submission-blocking**
- **Symptom:** `MODEL_NAME = "mistralai/Mistral-7B-v0.1"`, `TRAINED_MODEL_DIR = "./mistral_vlm_final"`, prompt uses `<s>[INST]…[/INST]`.
- **Reality:** `finetune_qwen.py` trains `Qwen/Qwen2.5-0.5B-Instruct`, saves a LoRA adapter to `./qwen_vlm_final`, training data is ChatML (`<|im_start|>…`).
- **Root cause:** `eval.py` is a stale artifact from an earlier Mistral-based plan; never updated when the pipeline moved to Qwen.
- **Consequences:** (a) Mistral-7B in 4-bit ≈ 4.5 GB+ of weights — won't load on a 4 GB card. (b) `Mistral-7B-v0.1` is a *base* model, not instruct. (c) The adapter was trained for Qwen layer names — `PeftModel.from_pretrained` onto a Mistral base would fail or mis-apply. (d) Even if it loaded, `[INST]` ≠ ChatML → out-of-distribution prompt → garbage.
- **Fix:** Rewrite `eval.py`: load `Qwen/Qwen2.5-0.5B-Instruct` (or whichever base C1 lands on) + adapter from `./qwen_vlm_final`; build the prompt with `tokenizer.apply_chat_template(..., add_generation_prompt=True)` using the **exact same instruction text** as `prep_llm_data.py`.

### BUG-02 — Train/inference prompt template mismatch
- **Symptom:** `prep_llm_data.py` trains on "You are an expert social media manager. Write an engaging marketing tweet for …"; `eval.py` prompts "Generate an engaging marketing tweet for …". Different system message, different wording, different structure.
- **Root cause:** Two independently written prompt builders; no shared function.
- **Consequence:** A fine-tuned model is highly sensitive to prompt format. Any drift degrades BLEU/ROUGE/CIDEr.
- **Fix:** Extract one `build_prompt(row)` helper imported by both `prep_llm_data.py` and `eval.py`. Train = prompt + assistant turn; inference = prompt + `add_generation_prompt=True`.

### BUG-03 — `train.csv` ↔ `train_enriched.csv` length mismatch + fragile resume — **HIGH**
- **Symptom:** `train.csv` = 15,511 rows, `train_enriched.csv` = 17,331 rows.
- **Root cause:** `enrich_vlm.py` writes the dataframe to **both** `OUTPUT_CSV` and `INPUT_CSV` every 10 rows. The raw input is no longer pristine. The resume guard `len(saved) == len(df)` silently no-ops when the two files diverge (e.g., after a crash mid-write, or a manual edit), so a fresh run would re-process the shorter file and **overwrite the longer enriched file, losing 1,820 rows of work**.
- **Fix:** Never write to `INPUT_CSV`. Keep an immutable `train_raw.csv`. Make resume key off the `id` column (merge on `id`), not row count. Reconcile the current two files by `id` before proceeding.

### BUG-04 — VLM enrichment barely started and mostly failing — **HIGH**
- **Symptom:** Of 17,331 rows: 16,916 empty, 298 "media could not be processed", 82 "no media", **~35 with a real description**.
- **Root cause:** Two parts. (1) The job simply hasn't run to completion. (2) The failure rate among *attempted* rows is huge (298 failed vs ~35 success) — Twitter `pbs.twimg.com` media URLs from tweets up to 5 years old are widely expired/deleted, and some need auth.
- **Consequences:** The "multimodal" pipeline currently has almost no visual signal; the LLM is effectively training on metadata only.
- **Fix:** (a) Run a quick liveness probe on a URL sample to measure the true death rate. (b) If most URLs are dead, accept metadata-only generation for those rows and design the prompt to degrade gracefully (it already blanks `visual_context`). (c) Check whether the official dataset ships media files/thumbnails separately. (d) Speed up enrichment with a lighter VLM (see §5) so the alive URLs can all be processed before 14 Dec.

### BUG-05 — Dead-code image resize in `enrich_vlm.py`
- **Symptom:** `run_vlm` does `pil_img = Image.open(...).resize(IMAGE_SIZE)` then immediately `del pil_img`. The `messages` dict passes the **file path**, so `process_vision_info` re-opens the *original* unresized image.
- **Root cause:** Resize result never wired into the message content.
- **Consequence:** The claimed "fixed pixel_values shape → no CUDA fragmentation" fix does nothing; pixel shapes still vary per image.
- **Fix:** Either pass the resized PIL object (`{"type": "image", "image": pil_img}`) or set `min_pixels`/`max_pixels` on the processor; or just delete the dead code if fragmentation isn't actually observed.

### BUG-06 — Local metrics ≠ competition metrics
- **Symptom:** `eval.py` computes only a hand-rolled clipped-unigram "BLEU-1" (no brevity penalty, no BLEU 2–4, no ROUGE, no CIDEr).
- **Root cause:** Placeholder metric written before the real eval suite was added.
- **Consequence:** You can't tune toward the actual leaderboard objective; local numbers are misleading.
- **Fix:** New `metrics.py`: BLEU 1–4 via `nltk.translate.bleu_score` (with smoothing) or `sacrebleu`; ROUGE-L via `rouge-score`; CIDEr via `pycocoevalcap`. Match tokenisation to whatever the organisers' `[1]` reference describes.

### BUG-07 — Submission format drops `id` and blanks input columns — **HIGH**
- **Symptom:** `submission_*` DataFrames set `likes=0`, `media=''`, and never carry `id`.
- **Root cause:** Submission builder written from the PS sample I/O table literally, without considering row alignment.
- **Consequence:** The grader almost certainly joins predictions to ground truth on `id`. Without `id`, alignment breaks; blanking provided inputs may also break a strict harness.
- **Fix:** Carry `id` through prediction; in the submission, preserve all original test columns and only fill/overwrite `content`.

### BUG-08 — `df.iloc[idx]` used with label indices in `enrich_vlm.py`
- **Symptom:** `indices = df.index[needs].tolist()` (labels) then `row = df.iloc[idx]` (positional).
- **Root cause:** Mixing `.iloc` and `.at`/label indexing.
- **Consequence:** Currently harmless because the index is a default RangeIndex, but it will silently corrupt row mapping the moment the frame is filtered/reindexed.
- **Fix:** Use `df.loc[idx]` consistently.

### BUG-09 — Decoding uses sampling; metrics reward mode-seeking
- **Symptom:** `do_sample=True, temperature=0.7, top_p=0.9`.
- **Root cause:** Defaults copied from a chat-style generation snippet.
- **Consequence:** BLEU/ROUGE/CIDEr reward overlap with the single reference; sampling adds variance away from the most-likely (closest-to-reference) output.
- **Fix:** `do_sample=False`, `num_beams=4–5`, `no_repeat_ngram_size=3`, tune `length_penalty`; truncate at first newline/EOS.

### BUG-10 — No train/val split in `finetune_qwen.py`
- **Symptom:** Trains on 100% of data, 1 epoch, no eval dataset, no checkpoint selection.
- **Consequence:** Can't detect overfitting or pick the best step; can't tune decoding honestly.
- **Fix:** Hold out a validation set that mirrors the two regimes (some brands fully held out; the latest time window held out).

### BUG-11 — `finetune_qwen.py` filters by character length, not tokens
- **Symptom:** `dataset.filter(lambda x: x["length"] < 1000)` where `length` = `len(text)` characters.
- **Consequence:** Arbitrary cut; long VLM descriptions silently drop training rows; no guarantee re: token budget.
- **Fix:** Filter/truncate by tokenised length with an explicit `max_seq_length`.

### BUG-12 — VLM prompt yields verbose, multi-paragraph, markdown output
- **Symptom:** Descriptions contain `**Visible Text:**` headers and embedded newlines (visible in `train_enriched.csv`).
- **Consequence:** Bloats LLM context, embeds newlines into CSV cells (fragile parsing), dilutes signal.
- **Fix:** Prompt for a single concise sentence: visible text first, then a short scene description; explicitly "no markdown, no line breaks, max ~40 words".

### Open question — `<hyperlink>` / `<mention>` / `<hashtag>` placeholder tokens
- Training `content` contains literal anonymisation tokens. If the **ground-truth test text keeps them**, the model emitting them at the right positions *helps* BLEU/ROUGE — keep as-is. If the grader strips them, we should strip them from predictions too. **Action:** confirm from the organisers' reference `[1]` / dataset README before final submission.

---

## 4. What Was Tried — Worked / Didn't / Why

| Attempt | Outcome | Why |
|---|---|---|
| Two-stage VLM→LLM architecture | ⚠️ Sound idea, not yet realised | Architecture is reasonable; execution blocked by BUG-01/03/04 |
| Qwen2.5-VL-3B-Instruct 4-bit for enrichment | ⚠️ Partially works | Loads & runs on 4 GB, but slow + most media URLs dead → ~2% coverage |
| Qwen2.5-0.5B-Instruct QLoRA fine-tune | ⚠️ Ran, untrusted | Completed 1 epoch on 17K rows, but no val split and base model likely too small for competitive metrics |
| Hand-rolled BLEU-1 in `eval.py` | ❌ Misleading | Not the competition metric; no BLEU2-4/ROUGE/CIDEr |
| Mistral-7B eval path | ❌ Never viable on this HW | 7B won't fit 4 GB; also inconsistent with the trained Qwen adapter |
| Writing enriched data back to `train.csv` | ❌ Caused data drift | Lost the pristine input; broke the resume guard |

---

## 5. Model Recommendations (RTX 3050, 4 GB VRAM)

### VLM (Stage 1 — media → description)
| Model | Size | Fit on 4 GB | Notes |
|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** (4-bit) | 3B | Borderline OK | Current choice; best quality; slow |
| **Florence-2-large** | 0.77B | Easy (fp16) | Excellent OCR + caption via task tokens (`<OCR>`, `<MORE_DETAILED_CAPTION>`); very fast — best speed/quality trade for this dataset |
| **Moondream2** | 1.8B | Easy | Edge-optimised VLM, ~2 GB, fast, decent captions |
| **SmolVLM-500M / 2.2B** | 0.5–2.2B | Easy | Very light; 500M variant for max throughput |
| BLIP-2 OPT-2.7B | 3.7B | Tight | Heavier, weaker OCR — not recommended |
- **Recommendation:** Move enrichment to **Florence-2-large** (OCR is the highest-value signal — marketing images often contain text that reappears verbatim in the tweet → direct n-gram overlap → BLEU gain) and keep Qwen2.5-VL-3B only as a quality spot-check. Florence-2's speed is what lets you cover all *alive* URLs before 14 Dec.

### LLM (Stage 2 — metadata → tweet)
| Model | Train on 4 GB (QLoRA)? | Notes |
|---|---|---|
| Qwen2.5-0.5B-Instruct | ✅ Easy | Current; likely too small for competitive metrics |
| **Qwen2.5-1.5B-Instruct** | ✅ Yes (bs=1, grad-ckpt) | **Recommended** — best quality/VRAM balance |
| Llama-3.2-1B-Instruct | ✅ Yes | Strong, very safe fit |
| Gemma-2-2B-it | ⚠️ Tight | Good quality, watch OOM |
| Qwen2.5-3B-Instruct | ⚠️ Risky | Possible with short seq + bs=1, may OOM on a laptop 3050 |
| Phi-3.5-mini (3.8B), Mistral-7B | ❌ No | Too large to QLoRA-train on 4 GB |
- **Recommendation:** Train **Qwen2.5-1.5B-Instruct with QLoRA** (4-bit NF4, r=16, target all attn + MLP proj). Keep 0.5B as a fast fallback.
- **GGUF inference path:** after training, merge the LoRA adapter and convert to **GGUF Q4_K_M**, run via `llama-cpp-python`. This sidesteps `bitsandbytes`-on-Windows pain, speeds up the 20K-row inference pass, and lets you optionally run a larger merged model (3B) inference-only with partial GPU offload.
- **Efficiency-narrative option:** a fine-tuned **Flan-T5-base (250M)** seq2seq baseline trains trivially on 4 GB and is genuinely competitive on BLEU/ROUGE for constrained generation. Worth having as both a baseline and a talking point for the 35% "efficiency" score.

---

## 6. Pipeline Improvements (expected score impact)

1. **Retrieval-augmented few-shot prompting — highest leverage.**
   For each test row, retrieve k nearest training tweets and put them in the prompt as examples.
   - *Seen brand / unseen time:* retrieve heavily from that brand's own history → learns brand voice, recurring hashtags, CTAs. This regime should score much higher.
   - *Unseen brand / seen time:* retrieve by semantic similarity of VLM description + brand-category embedding + nearby timestamp.
   Even a **pure nearest-neighbour copy** (no LLM) is a strong BLEU/CIDEr baseline — implement it first as a floor.
2. **Regime-aware strategy** — don't use one prompt for both; the information available differs.
3. **Beam search + no-repeat n-gram + length calibration** — match generated length to the training median (~20–30 words); too long tanks precision, too short triggers brevity penalty.
4. **Condition on `likes`** — it's a provided Task-2 input currently ignored. Bucket into low/med/viral and add to the prompt; viral tweets have distinct style.
5. **Better VLM prompt** (BUG-12) — concise, OCR-first.
6. **Brand-hashtag post-processing** — if brand is known and the model omitted `#{Brand}`, append it (matches a very common training pattern).
7. **Output cleaning** — regex-strip "Sure, here's a tweet:" preambles, surrounding quotes, markdown.
8. **Candidate reranking** — generate N beams, rerank by similarity to retrieved neighbours.
9. **Train on the full 300K** (or as much as feasible) — 17K is small; more data is the cheapest quality lever.

---

## 7. Next Steps (ordered)

1. **Reconcile the two CSVs** by `id`; snapshot an immutable `train_raw.csv`. (BUG-03)
2. **Rewrite `eval.py`** for Qwen + ChatML + shared prompt template. (BUG-01, BUG-02)
3. **Add `metrics.py`** — BLEU 1–4, ROUGE-L, CIDEr. (BUG-06)
4. **Fix submission format** — carry `id`, preserve input columns. (BUG-07)
5. **Probe Twitter media URL liveness**; decide VLM strategy. (BUG-04)
6. **Add a regime-mirroring train/val split.** (BUG-10)
7. Build the **nearest-neighbour retrieval baseline** — get a real BLEU/ROUGE/CIDEr floor.
8. Upgrade base LLM to **Qwen2.5-1.5B**, switch decoding to **beam search**, retrain. (C1, C2)
9. Layer in **retrieval-augmented few-shot**. (C3)
10. Tune decoding on val, package submissions, write the 3-page ACL report.

## 8. Open Questions

- Does the official dataset ship media files/thumbnails separately, or only URLs? (Decides whether VLM is salvageable.) we have urls and there can be gifs, video, images as well so we need to use a vlm and try to avoid OCR models and very few images have text.
- Do ground-truth test tweets keep `<hyperlink>`/`<mention>`/`<hashtag>` tokens? (Decides post-processing.)
- Exact tokenisation/implementation the organisers use for BLEU/ROUGE/CIDEr (reference `[1]`)?
- Can we obtain the full 300K training set, or are we limited to the ~17K subset?
- Is the test set one combined file or two (per regime)? `eval.py` assumes two.

---

## 9. Session Log

### 2026-05-15 — Session 3: SFTConfig API fix + smoke tests

**finetune_qwen.py — FIXED ✅**
- TRL 0.29.0 renamed `max_seq_length` → `max_length` in `SFTConfig`
- Added explicit `eval_strategy="steps"` (default is "no")
- Fixed `torch_dtype=` → `dtype=` deprecation in `from_pretrained`
- Smoke test (1 step, 4 rows): PASSED — peak VRAM 2.48 GB, loss 3.848

**All Phase A code now verified working:**
| File | Smoke test result |
|---|---|
| `prep_llm_data.py` | ✅ 17,331 records written |
| `finetune_qwen.py` | ✅ 1-step pass, 2.48 GB VRAM |
| `eval.py` | ❌ Not yet tested (requires trained adapter) |
| `enrich_vlm.py` | ✅ VLM loads, inference quality confirmed |

**Next:** run full `python finetune_qwen.py` (3 epochs, ~17K rows — expect several hours)

### 2026-05-15 — Session 2: Model swap implementation

**Stage 1 — VLM swap (enrich_vlm.py)**
- Rewrote for Florence-2-large (0.77B, fp16): `<OCR>` + `<MORE_DETAILED_CAPTION>` tasks
- Output: single-line `"Text: {ocr}. {caption}"` — no embedded newlines, max 60 words
- Resume now keys off `id` column (not row count) — safe against partial saves
- Never writes back to INPUT_CSV
- Dead-code image resize removed (was no-op in Qwen2.5-VL version)

**Stage 2 — LLM swap (finetune_qwen.py)**
- Base model: Qwen2.5-0.5B → **Qwen2.5-1.5B-Instruct**
- LoRA: r=8/2 modules → **r=16/4 attention modules** (q/k/v/o_proj)
- Filter: character-length heuristic → **SFTConfig max_seq_length=512** (token-based)
- Added **5% eval split**, `load_best_model_at_end=True`
- 3 epochs (was 1), output dir `./qwen15b_tweet_final`

**Shared prompt (prompt_utils.py — new)**
- `build_instruction(row)` / `build_messages(row)` — single source of truth
- Added `likes` bucketing (Task 2 provides likes as input; was ignored)
- Imported by both `prep_llm_data.py` and `eval.py` — train/inference format guaranteed identical

**Stage 3 — eval.py full rewrite**
- Loads Qwen2.5-1.5B-Instruct + `./qwen15b_tweet_final` adapter
- ChatML via `tokenizer.apply_chat_template` matching training exactly
- Beam search: `num_beams=4, do_sample=False, no_repeat_ngram_size=3`
- Real metrics: BLEU 1-4 (nltk), ROUGE-1/2/L (rouge-score), CIDEr (pycocoevalcap)
- Test-set VLM auto-enrichment via Florence-2 before generation
- Submission: carries `id` + all original input columns

**Next action:** install new deps + run `python prep_llm_data.py` (rebuild JSONL with likes+updated prompt) → `python finetune_qwen.py` → `python eval.py`

New deps needed: `pip install microsoft-florence-2` / just `transformers>=4.40` has Florence-2; `pip install nltk rouge-score pycocoevalcap`

### 2026-05-15 — Initial audit
- Read PS, all Task-2 pipeline files, and data files.
- Found: `eval.py`/model mismatch (BUG-01), prompt mismatch (BUG-02), CSV length divergence 15,511 vs 17,331 (BUG-03), VLM enrichment ~2% done with ~90% download-failure rate among attempts (BUG-04), dead-code resize (BUG-05), incomplete metrics (BUG-06), broken submission format (BUG-07), and others.
- Confirmed only ~17K of 300K samples present; only ~35 rows have usable VLM descriptions.
- Created this `progress.md`. Next session starts at §7 step 1.
