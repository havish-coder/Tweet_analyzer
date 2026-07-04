"""
Pick the best LoRA checkpoint by GENERATION metrics, not eval_loss.

eval_loss (token-level) is dominated by templated tokens (<hyperlink>, ChatML
scaffolding) and does not track BLEU/ROUGE — which is what the competition
grades. This script generates tweets from each saved checkpoint on a random
slice of the regime-mirrored eval split and reports BLEU-1 / ROUGE-L per
checkpoint, so the shipped adapter is the one that actually generates best.

Usage (after finetune_qwen.py has left checkpoints in ./adapter):
    python src/select_checkpoint.py            # 200 eval rows per checkpoint
    N_EVAL_GEN=50 python src/select_checkpoint.py   # quicker pass

Copy the winning checkpoint's files over ./adapter (or point eval.py's
TRAINED_MODEL_DIR at it) once you've picked.
"""

import gc
import glob
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from finetune_qwen import make_regime_split, EVAL_BRAND_FRAC, EVAL_TIME_FRAC, SEED
from gen_metrics import compute_bleu, compute_rouge
from prompt_utils import build_messages

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./adapter"
CSV_PATH    = "data/train_enriched.csv"
N_EVAL_GEN  = int(os.environ.get("N_EVAL_GEN", "200"))
GEN_BATCH   = int(os.environ.get("GEN_BATCH", "4"))
MAX_NEW_TOKENS = 100
NUM_BEAMS   = 4

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def load_eval_rows() -> pd.DataFrame:
    csv_to_use = CSV_PATH if os.path.exists(CSV_PATH) else "data/train.csv"
    df = pd.read_csv(csv_to_use)
    df = df[df["content"].notna() & (df["content"].str.strip() != "")].reset_index(drop=True)
    _, eval_pos, _ = make_regime_split(df, EVAL_BRAND_FRAC, EVAL_TIME_FRAC, SEED)
    eval_df = df.iloc[eval_pos]
    if len(eval_df) > N_EVAL_GEN:
        eval_df = eval_df.sample(n=N_EVAL_GEN, random_state=SEED)
    return eval_df.reset_index(drop=True)


def generate_all(model, tokenizer, eval_df: pd.DataFrame) -> list:
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(row.to_dict(), include_response=False),
            tokenize=False, add_generation_prompt=True,
        )
        for _, row in eval_df.iterrows()
    ]
    outs = []
    for start in tqdm(range(0, len(prompts), GEN_BATCH), desc="generate"):
        batch = prompts[start:start + GEN_BATCH]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for seq in generated:
            outs.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip())
    return outs


def main():
    checkpoints = sorted(glob.glob(os.path.join(ADAPTER_DIR, "checkpoint-*")),
                         key=lambda p: int(p.rsplit("-", 1)[-1]))
    candidates = checkpoints + [ADAPTER_DIR]  # final saved adapter last
    print(f"Candidates: {candidates}")

    eval_df = load_eval_rows()
    refs = [str(c) for c in eval_df["content"].tolist()]
    print(f"Eval rows: {len(eval_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for ckpt in candidates:
        print(f"\n{'='*60}\nScoring {ckpt}\n{'='*60}")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=QUANT_CONFIG, device_map="auto",
            dtype=torch.float16, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, ckpt)
        model.eval()

        preds = generate_all(model, tokenizer, eval_df)
        bleu  = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        row = {"checkpoint": ckpt,
               "BLEU-1": bleu.get("BLEU-1", float("nan")),
               "ROUGE-L": rouge.get("ROUGE-L", float("nan"))}
        results.append(row)
        print(f"  BLEU-1={row['BLEU-1']:.4f}  ROUGE-L={row['ROUGE-L']:.4f}")

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\nSUMMARY (higher is better)\n{'='*60}")
    results.sort(key=lambda r: (r["BLEU-1"] + r["ROUGE-L"]), reverse=True)
    for r in results:
        print(f"  {r['checkpoint']:<40} BLEU-1={r['BLEU-1']:.4f}  ROUGE-L={r['ROUGE-L']:.4f}")
    print(f"\nBest: {results[0]['checkpoint']}")


if __name__ == "__main__":
    main()
