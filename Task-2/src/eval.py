"""
Inference + evaluation for the Qwen2.5-1.5B tweet generation model.

Fixes over v2:
  - RANDOM eval sample (seeded) instead of head(N) — head() silently graded the
    first rows only, which is biased when test files are sorted by date/brand.
  - Bootstrap 95% CIs for BLEU-1 / ROUGE-L (only valid with random sampling).
  - Batched left-padded generation (GEN_BATCH) — makes a full-10K run feasible.
  - VLM enrichment gated by a fast concurrent URL-liveness check, so dead links
    (99.8% of old media) don't cost a 10s timeout each.
  - Optional retrieval few-shot (USE_RETRIEVAL=1): inserts the brand's recent
    train tweets into the prompt. OFF by default because the shipped adapter was
    fine-tuned WITHOUT examples — enable only for A/B runs or after re-finetuning
    with EXAMPLES_IN_TRAIN=1.

Env config (all optional):
  SAMPLE_SIZE=500   rows per regime; 0 = full file
  SKIP_VLM=1        skip image enrichment entirely (default: enabled with liveness check)
  USE_RETRIEVAL=0   few-shot brand examples in prompt
  GEN_BATCH=4       generation batch size
  LENGTH_PENALTY=1.0, MIN_NEW_TOKENS=0   length-control knobs for beam search
"""

import os
import re
import gc
import random
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from prompt_utils import build_messages
from gen_metrics import compute_bleu, compute_rouge, compute_cider, bootstrap_ci

# ---------------------------------------------------------------------------
# Config — update MODEL_NAME if you change the base during training
# ---------------------------------------------------------------------------
MODEL_NAME        = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINED_MODEL_DIR = "./adapter"
TEST_BRANDS_PATH  = "data/test_unseen_brands.csv"
TEST_TIME_PATH    = "data/test_unseen_time.csv"
TRAIN_CSV_PATH    = "data/train.csv"
OUTPUT_DIR        = os.environ.get("OUTPUT_DIR", "outputs")
MAX_NEW_TOKENS    = 100
NUM_BEAMS         = 4
SEED              = 42

def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}

SAMPLE_SIZE    = int(os.environ.get("SAMPLE_SIZE", "500"))     # 0 = full file
SKIP_VLM       = _env_flag("SKIP_VLM", "0")                    # liveness check makes default-on cheap
USE_RETRIEVAL  = _env_flag("USE_RETRIEVAL", "0")               # adapter was trained WITHOUT examples
NO_ADAPTER     = _env_flag("NO_ADAPTER", "0")                  # base-model baseline run (no LoRA)
GEN_BATCH      = int(os.environ.get("GEN_BATCH", "4"))
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", "1.0"))
MIN_NEW_TOKENS = int(os.environ.get("MIN_NEW_TOKENS", "0"))

print("=" * 70)
print("TWEET GENERATION EVALUATION")
print(f"  sample={SAMPLE_SIZE or 'full'}  skip_vlm={SKIP_VLM}  retrieval={USE_RETRIEVAL}  "
      f"batch={GEN_BATCH}  length_penalty={LENGTH_PENALTY}")
print("=" * 70)

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Shared 4-bit config (reused for both VLM and LLM)
# ---------------------------------------------------------------------------
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------------------------------------------------------
# URL liveness — cheap concurrent HEAD requests so we only load the VLM (and pay
# per-row fetch costs) for media that still exists. Old Twitter media is ~99.8%
# dead in the training set, but the unseen-TIME test set is recent, so its links
# are the most likely to be alive — exactly where captions help most.
# ---------------------------------------------------------------------------
def check_liveness(urls: list, timeout: float = 3.0, workers: int = 16) -> list:
    import requests
    from concurrent.futures import ThreadPoolExecutor

    def alive(url):
        if not url:
            return False
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            return r.status_code < 400
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(alive, urls))


# ---------------------------------------------------------------------------
# VLM enrichment for test rows
# Runs FIRST, then frees VRAM, then LLM is loaded — avoids 3 GB+ co-residence.
# ---------------------------------------------------------------------------
def maybe_enrich_test(df: pd.DataFrame) -> pd.DataFrame:
    """Run Qwen2.5-VL-3B enrichment on test rows whose media URL is still live."""
    if "vlm_description" not in df.columns:
        df["vlm_description"] = ""

    needs = df["vlm_description"].isna() | (df["vlm_description"].astype(str).str.strip() == "")
    has_media = df["media"].notna() & (df["media"].astype(str).str.strip() != "")
    candidates = df.index[needs & has_media].tolist()
    if not candidates:
        return df

    try:
        import enrich_vlm as ev
    except Exception as e:
        print(f"  VLM enrichment unavailable ({e}) — skipping.")
        return df

    urls = [
        ev.extract_image_url(str(df.at[i, "media"])) or ev.extract_video_url(str(df.at[i, "media"]))
        for i in candidates
    ]
    print(f"  Checking liveness of {len(urls)} media URLs...")
    live = check_liveness(urls)
    todo = [i for i, ok in zip(candidates, live) if ok]
    print(f"  {len(todo)}/{len(candidates)} media URLs are live.")
    for i, ok in zip(candidates, live):
        if not ok:
            df.at[i, "vlm_description"] = "no media"
    if not todo:
        return df

    print(f"  Enriching {len(todo)} test rows with Qwen2.5-VL-3B...")
    try:
        import tempfile
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        vlm_processor = AutoProcessor.from_pretrained(ev.MODEL_ID)
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ev.MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=QUANT_CONFIG,
            low_cpu_mem_usage=True,
        )
        vlm_model.eval()
        tmp = os.path.join(tempfile.gettempdir(), "qwen_test_img.jpg")

        for idx in tqdm(todo, desc="Test VLM"):
            media_str = str(df.at[idx, "media"])
            try:
                desc = ev.fetch_media_caption(vlm_model, vlm_processor, media_str, tmp)
                df.at[idx, "vlm_description"] = desc or "media could not be processed"
            except Exception as e:
                print(f"    [vlm error] {e}")
                df.at[idx, "vlm_description"] = "media could not be processed"

        # Free VLM before LLM loads
        del vlm_model, vlm_processor
        gc.collect()
        torch.cuda.empty_cache()
        print("  VLM freed — VRAM headroom restored.")
    except Exception as e:
        print(f"  VLM enrichment skipped: {e}")

    return df


# ---------------------------------------------------------------------------
# Retrieval few-shot: brand → its most recent training tweets.
# For unseen brands there are no tweets to retrieve; we fall back to a
# closest-name match (we only know the brand's NAME at test time, so
# embedding the brand's tweets — as we do for seen brands — isn't possible).
# ---------------------------------------------------------------------------
def build_retrieval_index(train_csv: str, k: int = 3) -> dict:
    if not os.path.exists(train_csv):
        print(f"  [retrieval] {train_csv} not found — few-shot disabled.")
        return {}
    df = pd.read_csv(train_csv)
    df = df[df["content"].notna() & (df["content"].str.strip() != "")]
    brand_col = "inferred company" if "inferred company" in df.columns else "username"
    df["_brand"] = df[brand_col].astype(str).str.lower().str.strip()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("_dt", kind="stable")
    index = {b: g["content"].tail(k).tolist() for b, g in df.groupby("_brand")}
    print(f"  [retrieval] index built: {len(index)} brands from {train_csv}")
    return index


def lookup_examples(index: dict, row: dict) -> list:
    if not index:
        return []
    brand = str(row.get("company", row.get("inferred company", ""))).lower().strip()
    if brand in index:
        return index[brand]
    import difflib
    close = difflib.get_close_matches(brand, list(index.keys()), n=1, cutoff=0.8)
    return index[close[0]] if close else []


# ---------------------------------------------------------------------------
# LLM loader — called AFTER VLM enrichment so we never have both in VRAM
# ---------------------------------------------------------------------------
def load_llm():
    print(f"\nLoading {'BASE model (no adapter — baseline run)' if NO_ADAPTER else 'fine-tuned LLM'}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    if NO_ADAPTER:
        model = base_model
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    else:
        model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_DIR)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR, trust_remote_code=True)
    # Left padding so batched generation reads a contiguous prompt suffix
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("LLM loaded.")
    return model, tokenizer

# ---------------------------------------------------------------------------
# 3. Load test data
# ---------------------------------------------------------------------------
print("\nLoading test datasets...")

def load_test(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  {path} not found — using dummy data for demo")
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "date": ["2024-01-15 10:00:00"] * 3,
            "likes": [100, 500, 50],
            "username": ["Nike", "adidas", "PUMA"],
            "media": [""] * 3,
            "inferred company": ["nike", "adidas", "puma"],
            "content": ["Sample tweet 1", "Sample tweet 2", "Sample tweet 3"],
        })
    if "inferred company" in df.columns:
        df = df.rename(columns={"inferred company": "company"})
    return df

test_brands = load_test(TEST_BRANDS_PATH)
test_time   = load_test(TEST_TIME_PATH)

if SAMPLE_SIZE:
    # RANDOM sample, not head(): test files are typically sorted (by date/brand),
    # so head() would grade a biased slice and invalidate any CI claims.
    test_brands = test_brands.sample(n=min(SAMPLE_SIZE, len(test_brands)), random_state=SEED).reset_index(drop=True)
    test_time   = test_time.sample(n=min(SAMPLE_SIZE, len(test_time)), random_state=SEED).reset_index(drop=True)

print(f"  Unseen brands : {len(test_brands)} rows")
print(f"  Unseen time   : {len(test_time)} rows")

# Enrich test media (loads Qwen2.5-VL, runs, then frees it before LLM loads)
if not SKIP_VLM:
    test_brands = maybe_enrich_test(test_brands)
    test_time   = maybe_enrich_test(test_time)
else:
    print("  VLM enrichment skipped (SKIP_VLM=1)")

retrieval_index = build_retrieval_index(TRAIN_CSV_PATH) if USE_RETRIEVAL else {}

# Now load the LLM — VLM is already freed, no co-residence in VRAM
model, tokenizer = load_llm()

# ---------------------------------------------------------------------------
# 4. Tweet generation
# ---------------------------------------------------------------------------
def clean_tweet(text: str) -> str:
    """Strip common LLM preamble artifacts."""
    text = text.strip()
    text = re.sub(r'^["\'`]+|["\'`]+$', '', text).strip()
    # Remove "Sure, here's a tweet:" style preambles
    text = re.sub(r'^(sure[,!]?\s*here[\'s]*\s*(is|a)[^:]*:?\s*)', '', text,
                  flags=re.IGNORECASE).strip()
    # Truncate at first double newline
    text = text.split("\n\n")[0].strip()
    return text


def build_prompt_text(row: dict) -> str:
    examples = lookup_examples(retrieval_index, row) if USE_RETRIEVAL else None
    messages = build_messages(row, include_response=False, examples=examples)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_tweets(prompts: list) -> list:
    """Batched, left-padded beam-search generation."""
    outputs_text = []
    for start in tqdm(range(0, len(prompts), GEN_BATCH)):
        batch = prompts[start:start + GEN_BATCH]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                no_repeat_ngram_size=3,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for seq in out:
            new_tokens = seq[prompt_len:]
            outputs_text.append(clean_tweet(tokenizer.decode(new_tokens, skip_special_tokens=True)))
    return outputs_text


def generate_batch(test_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    print(f"\nGenerating: {task_name}")
    prompts = [build_prompt_text(row.to_dict()) for _, row in test_df.iterrows()]
    generations = generate_tweets(prompts)

    results = []
    for (_, row), generated in zip(test_df.iterrows(), generations):
        results.append({
            "id":               row.get("id", ""),
            "date":             row.get("date", ""),
            "likes":            row.get("likes", ""),
            "username":         row.get("username", ""),
            "media":            row.get("media", ""),
            "inferred company": row.get("company", row.get("inferred company", "")),
            "generated":        generated,
            "actual":           str(row.get("content", "")),
        })
    return pd.DataFrame(results)

print("\n" + "=" * 70)
preds_brands = generate_batch(test_brands, "Unseen Brands")
preds_time   = generate_batch(test_time,   "Unseen Time Period")

os.makedirs(OUTPUT_DIR, exist_ok=True)
preds_brands.to_csv(f"{OUTPUT_DIR}/predictions_unseen_brands.csv", index=False)
preds_time.to_csv(f"{OUTPUT_DIR}/predictions_unseen_time.csv",   index=False)
print("Predictions saved.")

# ---------------------------------------------------------------------------
# 5. Metrics: BLEU 1-4, ROUGE-1/2/L, CIDEr (+ bootstrap CIs)
# ---------------------------------------------------------------------------
def print_metrics(df: pd.DataFrame, label: str):
    preds = df["generated"].tolist()
    refs  = df["actual"].tolist()

    print(f"\n{'='*70}")
    print(f"RESULTS — {label}")
    print(f"{'='*70}")

    has_refs = any(r and str(r).strip() not in ("", "nan") for r in refs)
    if not has_refs:
        print("  [No ground-truth content column — metrics skipped for competition test set]")
        avg_len_gen = np.mean([len(p.split()) for p in preds])
        print(f"  Gen length  : {avg_len_gen:.1f} words")
        return

    for name, val in compute_bleu(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, val in compute_rouge(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, val in compute_cider(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, (lo, hi) in bootstrap_ci(preds, refs).items():
        print(f"  {name:<12}: [{lo:.4f}, {hi:.4f}]")

    avg_len_gen = np.mean([len(p.split()) for p in preds])
    avg_len_ref = np.mean([len(str(r).split()) for r in refs])
    print(f"  Gen length  : {avg_len_gen:.1f} words (ref: {avg_len_ref:.1f})")


print_metrics(preds_brands, "Unseen Brands")
print_metrics(preds_time,   "Unseen Time Period")

# ---------------------------------------------------------------------------
# 6. Submission files — preserve id + all original input columns
# ---------------------------------------------------------------------------
def make_submission(preds_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "id":               preds_df["id"],
        "date":             preds_df["date"],
        "likes":            preds_df["likes"],
        "username":         preds_df["username"],
        "media":            preds_df["media"],
        "inferred company": preds_df["inferred company"],
        "content":          preds_df["generated"],
    })

make_submission(preds_brands).to_csv(f"{OUTPUT_DIR}/submission_unseen_brands.csv", index=False)
make_submission(preds_time).to_csv(f"{OUTPUT_DIR}/submission_unseen_time.csv",   index=False)
print(f"\nSubmission files saved in {OUTPUT_DIR}/")
if SAMPLE_SIZE:
    print(f"[note] SAMPLE_SIZE={SAMPLE_SIZE}: these files cover a random subset, "
          f"NOT the full test set. Run with SAMPLE_SIZE=0 for a real submission.")

# Cleanup
del model
gc.collect()
torch.cuda.empty_cache()
print("\nDone.")
