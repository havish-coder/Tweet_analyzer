"""
Baseline for the reported fine-tune eval loss.

Recomputes mean token NLL (and perplexity) under IDENTICAL conditions —
same eval rows, same chat template, same tokenization, same max length —
for (a) the un-fine-tuned base model and (b) base + the shipped LoRA adapter.
This is the "compared to what" for the eval_loss number: the SFTTrainer value
(1.080) is only meaningful next to the base model's loss on the same data.

Usage (from Task-2/):
    python src/eval_base_loss.py            # 400-row seeded subset of eval split
    N_EVAL=0 python src/eval_base_loss.py   # full eval split (slower)
"""

import gc
import json
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from finetune_qwen import (
    CSV_PATH, DATA_PATH, EVAL_BRAND_FRAC, EVAL_TIME_FRAC, MAX_SEQ_LEN,
    MODEL_NAME, SEED, make_regime_split,
)

ADAPTER_DIR = "./adapter"
N_EVAL = int(os.environ.get("N_EVAL", "400"))  # 0 = full eval split

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def load_eval_texts(tokenizer) -> list:
    data = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8")]
    csv_to_use = CSV_PATH if os.path.exists(CSV_PATH) else "data/train.csv"
    df = pd.read_csv(csv_to_use)
    df = df[df["content"].notna() & (df["content"].str.strip() != "")].reset_index(drop=True)
    assert len(df) == len(data), "JSONL/CSV mismatch — rerun prep_llm_data.py"

    texts = [
        tokenizer.apply_chat_template(item["messages"], tokenize=False,
                                      add_generation_prompt=False)
        for item in data
    ]
    _, eval_pos, _ = make_regime_split(df, EVAL_BRAND_FRAC, EVAL_TIME_FRAC, SEED)
    eval_texts = [texts[i] for i in eval_pos]
    if N_EVAL and len(eval_texts) > N_EVAL:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(eval_texts), size=N_EVAL, replace=False)
        eval_texts = [eval_texts[i] for i in idx]
    return eval_texts


@torch.no_grad()
def mean_token_nll(model, tokenizer, texts, label) -> float:
    total_nll, total_tokens = 0.0, 0
    for t in tqdm(texts, desc=label):
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN)["input_ids"].to(model.device)
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += out.loss.item() * n
        total_tokens += n
    return total_nll / total_tokens


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    texts = load_eval_texts(tokenizer)
    print(f"Eval texts: {len(texts)} (seed {SEED})")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=QUANT_CONFIG, device_map="auto",
        dtype=torch.float16, trust_remote_code=True,
    )
    base.eval()
    base_nll = mean_token_nll(base, tokenizer, texts, "base model")

    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)
    tuned.eval()
    tuned_nll = mean_token_nll(tuned, tokenizer, texts, "base + adapter")

    print("\n" + "=" * 60)
    print(f"{'Model':<28} {'mean token NLL':>15} {'perplexity':>12}")
    print("-" * 60)
    print(f"{'Qwen2.5-1.5B base':<28} {base_nll:>15.4f} {np.exp(base_nll):>12.2f}")
    print(f"{'  + fine-tuned adapter':<28} {tuned_nll:>15.4f} {np.exp(tuned_nll):>12.2f}")
    print(f"{'improvement':<28} {base_nll - tuned_nll:>15.4f} "
          f"{(1 - np.exp(tuned_nll) / np.exp(base_nll)) * 100:>11.1f}%")
    print("=" * 60)
    print("Note: full-sequence NLL (prompt + completion tokens), computed "
          "identically for both models — directly comparable to each other; "
          "close to but not identical to the SFTTrainer eval_loss.")

    del tuned, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
