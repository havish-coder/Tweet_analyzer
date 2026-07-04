"""
Quantify the EngageX draft -> predict -> regenerate loop on real held-out rows.

For 50 rows sampled from the regime-mirrored eval split (25 unseen-brand +
25 unseen-time), measures:
  A. DRAFT      : beam-search generation, no engagement line in the prompt
  B. TARGETED   : regeneration conditioned on the draft's PREDICTED likes
  C. BEST-OF-4  : four sampled candidates, reranked by predicted likes

Reported: predicted engagement of A vs B (mean/median % change, improved/
regressed/unchanged fractions, one success + one failure example), and the
B-vs-A comparison for the reranking arm C.

Honesty note printed with results: all deltas are in PREDICTED engagement
(the Task-1 model's score). Generated tweets are never posted, so true
engagement is unobservable; C is selected BY the predictor, so its uplift on
the predictor's own scale is expected by construction — the informative
numbers are the size of that uplift and the A-vs-B distribution.

Run from repo root: python quantify_pipeline.py   (~25-35 min on RTX 3050)
"""

import io
import os
import sys

# Windows consoles default to cp1252 and choke on emoji in generated tweets
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from importlib.machinery import SourceFileLoader

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
T1, T2 = os.path.join(ROOT, "Task-1"), os.path.join(ROOT, "Task-2")
sys.path.insert(0, os.path.join(T2, "src"))
from prompt_utils import build_messages

feat_mod = SourceFileLoader("features_mod", os.path.join(T1, "01_features.py")).load_module()
train_mod = SourceFileLoader("train_mod", os.path.join(T1, "03_train.py")).load_module()

SEED = 42
N_PER_REGIME = 25
GEN_BATCH = 4
SAMPLE_BATCH = 8
N_CANDIDATES = 4
MAX_NEW_TOKENS = 100

rng = np.random.default_rng(SEED)


# ------------------------- sample eval rows -------------------------
df = pd.read_csv(os.path.join(T1, "features_train.csv"))
train_pos, eval_pos = train_mod.regime_split(
    df, train_mod.EVAL_BRAND_FRAC, train_mod.EVAL_TIME_FRAC, train_mod.SEED
)
train_brands = set(df["inferred company"].iloc[train_pos])
brand_rows = [i for i in eval_pos if df["inferred company"].iloc[i] not in train_brands]
time_rows = [i for i in eval_pos if df["inferred company"].iloc[i] in train_brands]
picked = (list(rng.choice(brand_rows, N_PER_REGIME, replace=False)) +
          list(rng.choice(time_rows, N_PER_REGIME, replace=False)))
regime = ["unseen-brand"] * N_PER_REGIME + ["unseen-time"] * N_PER_REGIME
rows = df.iloc[picked][["date", "username", "inferred company", "media"]].copy()
rows["vlm_description"] = ""
row_dicts = rows.to_dict("records")
print(f"Sampled {len(row_dicts)} eval rows ({N_PER_REGIME} per regime).")


# ------------------------- Task-1 predictor -------------------------
class Predictor:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.stats = joblib.load(os.path.join(T1, "models/company_stats.joblib"))
        self.cols = joblib.load(os.path.join(T1, "models/feature_cols.joblib"))
        self.scaler = joblib.load(os.path.join(T1, "models/tabular_scaler.joblib"))
        b = joblib.load(os.path.join(T1, "models/baseline_regressor.joblib"))
        self.model, self.smear = b["model"], b["smearing_factor"]
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def predict(self, metas: list, contents: list) -> np.ndarray:
        d = pd.DataFrame([{**m, "content": c, "id": i}
                          for i, (m, c) in enumerate(zip(metas, contents))])
        f = feat_mod.TabularFeatureBuilder(d, is_train=False, company_stats=self.stats).run()
        for c in self.cols:
            if c not in f.columns:
                f[c] = 0.0
        x_tab = self.scaler.transform(f[self.cols].fillna(0).astype(np.float32).values)
        emb = self.embedder.encode(f["content_clean"].fillna("").astype(str).tolist(),
                                   convert_to_numpy=True, normalize_embeddings=True)
        x = np.hstack([x_tab, emb]).astype(np.float32)
        return np.clip(np.exp(self.model.predict(x)) * self.smear - 1.0, 0, None)


# ------------------------- Task-2 generator -------------------------
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", quantization_config=quant, device_map="auto",
    dtype=torch.float16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, os.path.join(T2, "adapter"))
model.eval()
tok = AutoTokenizer.from_pretrained(os.path.join(T2, "adapter"), trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def prompts_for(metas, likes_list):
    out = []
    for m, lk in zip(metas, likes_list):
        msgs = build_messages({**m, "likes": int(lk)}, include_response=False)
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


@torch.no_grad()
def generate(prompts, sample=False, batch=GEN_BATCH, desc="gen"):
    outs = []
    for s in tqdm(range(0, len(prompts), batch), desc=desc):
        enc = tok(prompts[s:s + batch], return_tensors="pt", truncation=True,
                  max_length=512, padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n = enc["input_ids"].shape[1]
        kw = dict(max_new_tokens=MAX_NEW_TOKENS, no_repeat_ngram_size=3,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if sample:
            kw.update(do_sample=True, temperature=0.9, top_p=0.95)
        else:
            kw.update(do_sample=False, num_beams=4)
        out = model.generate(**enc, **kw)
        outs += [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]
    return outs


# ------------------------- run the loop -------------------------
predictor = Predictor()

print("\n[A] Drafting (beam, no engagement line)...")
drafts = generate(prompts_for(row_dicts, [0] * len(row_dicts)), desc="draft")
draft_pred = predictor.predict(row_dicts, drafts)

print("[B] Regenerating targeted at predicted engagement...")
regens = generate(prompts_for(row_dicts, draft_pred), desc="regen")
regen_pred = predictor.predict(row_dicts, regens)

print("[C] Best-of-4 sampled candidates, reranked by predicted likes...")
cand_prompts, cand_meta = [], []
for m, lk in zip(row_dicts, draft_pred):
    p = prompts_for([m], [lk])[0]
    cand_prompts += [p] * N_CANDIDATES
    cand_meta += [m] * N_CANDIDATES
cands = generate(cand_prompts, sample=True, batch=SAMPLE_BATCH, desc="sample")
cand_pred = predictor.predict(cand_meta, cands)
best_pred, best_text = [], []
for i in range(len(row_dicts)):
    block = slice(i * N_CANDIDATES, (i + 1) * N_CANDIDATES)
    j = int(np.argmax(cand_pred[block]))
    best_pred.append(cand_pred[block][j])
    best_text.append(cands[block][j])
best_pred = np.array(best_pred)

# ------------------------- report -------------------------
res = pd.DataFrame({
    "regime": regime,
    "company": [m["inferred company"] for m in row_dicts],
    "draft": drafts, "draft_pred": draft_pred.round(1),
    "targeted": regens, "targeted_pred": regen_pred.round(1),
    "best_of_4": best_text, "best_of_4_pred": np.array(best_pred).round(1),
})
out_csv = os.path.join(T2, "outputs", "pipeline_quantification.csv")
res.to_csv(out_csv, index=False)

pct = (regen_pred - draft_pred) / np.maximum(draft_pred, 1) * 100
identical = np.array([d == r for d, r in zip(drafts, regens)])
pct_bo4 = (best_pred - draft_pred) / np.maximum(draft_pred, 1) * 100

print("\n" + "=" * 70)
print("A -> B (targeted regeneration vs draft), predicted likes")
print(f"  mean change  : {pct.mean():+.1f}%   median: {np.median(pct):+.1f}%")
print(f"  improved >1% : {(pct > 1).mean() * 100:.0f}%   regressed <-1%: {(pct < -1).mean() * 100:.0f}%   ~unchanged: {(np.abs(pct) <= 1).mean() * 100:.0f}%")
print(f"  textually identical draft==targeted: {identical.mean() * 100:.0f}% of rows")
print(f"  by regime — mean change unseen-brand: {pct[:N_PER_REGIME].mean():+.1f}%  unseen-time: {pct[N_PER_REGIME:].mean():+.1f}%")
print("\nA -> C (best-of-4 reranking vs draft), predicted likes")
print(f"  mean uplift  : {pct_bo4.mean():+.1f}%   median: {np.median(pct_bo4):+.1f}%   improved: {(pct_bo4 > 1).mean() * 100:.0f}%")

worst, bestix = int(np.argmin(pct)), int(np.argmax(pct))
print("\n--- clearest SUCCESS (A->B) ---")
print(f"  [{res.iloc[bestix]['company']}] {draft_pred[bestix]:.0f} -> {regen_pred[bestix]:.0f} predicted likes ({pct[bestix]:+.0f}%)")
print(f"  draft   : {drafts[bestix]}")
print(f"  targeted: {regens[bestix]}")
print("--- clearest FAILURE (A->B) ---")
print(f"  [{res.iloc[worst]['company']}] {draft_pred[worst]:.0f} -> {regen_pred[worst]:.0f} predicted likes ({pct[worst]:+.0f}%)")
print(f"  draft   : {drafts[worst]}")
print(f"  targeted: {regens[worst]}")
print(f"\nSaved per-row results -> {out_csv}")
print("\nNOTE: all deltas are in PREDICTED engagement (Task-1 model score); "
      "true engagement of generated tweets is unobservable. Best-of-4 uplift "
      "is measured on the same score it selects by — report it as such.")
