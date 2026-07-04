"""
EngageX end-to-end pipeline: Task 1 (engagement prediction) feeding Task 2
(tweet generation) as one system.

Task 1 predicts likes FROM content + metadata; Task 2 generates content FROM
metadata + a likes target. Chaining them naively is circular (no content
exists before generation), so the pipeline runs a draft -> score -> regenerate
loop:

  1. DRAFT     Task 2 generates a tweet from metadata alone (no engagement line
               in the prompt — the model trained on that case too).
  2. SCORE     Task 1 featurizes the draft (same TabularFeatureBuilder +
               MiniLM embedding as training) and predicts its like count with
               the shipped smeared regressor.
  3. TARGETED  Task 2 regenerates with the PREDICTED like count injected into
               the prompt as the engagement target.

Output: final tweet + its predicted engagement — a single metadata-in,
(content, engagement)-out system.

Usage (from repo root):
    python engagex_pipeline.py --company nike --username Nike --date "2023-06-01 15:00:00"
"""

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.join(ROOT, "Task-1")
T2 = os.path.join(ROOT, "Task-2")
sys.path.insert(0, os.path.join(T2, "src"))

from prompt_utils import build_messages  # shared Task-2 prompt template

TabularFeatureBuilder = SourceFileLoader(
    "features_mod", os.path.join(T1, "01_features.py")
).load_module().TabularFeatureBuilder

LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 100
NUM_BEAMS = 4


# ---------------------------------------------------------------------------
# Task 1: engagement predictor (shipped smeared regressor)
# ---------------------------------------------------------------------------
class EngagementPredictor:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.company_stats = joblib.load(os.path.join(T1, "models/company_stats.joblib"))
        self.feature_cols = joblib.load(os.path.join(T1, "models/feature_cols.joblib"))
        self.scaler = joblib.load(os.path.join(T1, "models/tabular_scaler.joblib"))
        bundle = joblib.load(os.path.join(T1, "models/baseline_regressor.joblib"))
        self.model, self.smear = bundle["model"], bundle["smearing_factor"]
        # CPU keeps the GPU free for the LLM; one row is instant either way
        self.embedder = SentenceTransformer(EMBED_NAME, device="cpu")

    def predict(self, row: dict, content: str) -> int:
        df = pd.DataFrame([{**row, "content": content, "id": 0}])
        feat = TabularFeatureBuilder(df, is_train=False, company_stats=self.company_stats).run()
        for col in self.feature_cols:
            if col not in feat.columns:
                feat[col] = 0.0
        x_tab = self.scaler.transform(
            feat[self.feature_cols].fillna(0).astype(np.float32).values
        )
        emb = self.embedder.encode(
            [feat["content_clean"].iloc[0]], convert_to_numpy=True, normalize_embeddings=True
        )
        x = np.hstack([x_tab, emb]).astype(np.float32)
        pred_log = float(self.model.predict(x)[0])
        return int(round(max(np.exp(pred_log) * self.smear - 1.0, 0)))


# ---------------------------------------------------------------------------
# Task 2: tweet generator (fine-tuned QLoRA adapter)
# ---------------------------------------------------------------------------
class TweetGenerator:
    def __init__(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, quantization_config=quant, device_map="auto",
            dtype=torch.float16, trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, os.path.join(T2, "adapter"))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(T2, "adapter"), trust_remote_code=True
        )

    def generate(self, row: dict, likes: int) -> str:
        messages = build_messages({**row, "likes": likes}, include_response=False)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_prompt = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="EngageX: metadata -> (tweet, predicted engagement)")
    ap.add_argument("--company", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--date", required=True, help='e.g. "2023-06-01 15:00:00"')
    ap.add_argument("--media", default="", help="optional media string")
    args = ap.parse_args()

    row = {
        "date": args.date,
        "username": args.username,
        "inferred company": args.company.lower().strip(),
        "media": args.media,
        "vlm_description": "",
    }

    print("Loading Task-1 engagement predictor (CPU)...")
    predictor = EngagementPredictor()
    print("Loading Task-2 generator (4-bit, GPU)...")
    generator = TweetGenerator()

    print("\n[1/3] Drafting tweet from metadata alone...")
    draft = generator.generate(row, likes=0)  # likes<100 -> no engagement line in prompt
    print(f"      draft: {draft}")

    print("[2/3] Task 1 predicts the draft's engagement...")
    predicted_likes = predictor.predict(row, draft)
    print(f"      predicted likes: {predicted_likes}")

    print("[3/3] Regenerating with the predicted engagement as target...")
    final = generator.generate(row, likes=predicted_likes)

    print("\n" + "=" * 66)
    print(f"  Company           : {args.company}")
    print(f"  Final tweet       : {final}")
    print(f"  Predicted likes   : {predicted_likes}")
    print("=" * 66)


if __name__ == "__main__":
    main()
