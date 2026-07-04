"""
20-step training probe: exact peak VRAM + seconds-per-optimizer-step for the
QLoRA fine-tune, under the IDENTICAL config to finetune_qwen.py (same 4-bit
quantization, LoRA r=16 on q/k/v/o, paged_adamw_8bit, batch 1 x grad-accum 16,
max_length 256, gradient checkpointing, VRAMGuard).

Settles the documentation contradiction: DEEP_DIVE.md claims 10 s/it,
explain.md claims ~21 s/step. This measures it.

Run from Task-2/:  python src/probe_train.py
"""

import io
import json
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from finetune_qwen import MODEL_NAME, DATA_PATH, MAX_SEQ_LEN, VRAMGuardCallback

N_STEPS = 20


class StepTimer(TrainerCallback):
    def __init__(self):
        self.times = []

    def on_step_end(self, args, state, control, **kwargs):
        self.times.append(time.perf_counter())


def main():
    print("Loading data + tokenizer...")
    texts = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False))
            if len(texts) >= 400:  # 20 steps x 16 grad-accum = 320 samples needed
                break
    ds = Dataset.from_dict({"text": texts})

    print(f"Loading {MODEL_NAME} in 4-bit...")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant, device_map="auto",
        dtype=torch.float16, trust_remote_code=True)
    model.config.use_cache = False
    reserved_after_load = torch.cuda.memory_reserved(0)

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    cfg = SFTConfig(
        output_dir="./_probe_out", max_steps=N_STEPS,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        optim="paged_adamw_8bit", learning_rate=2e-4, weight_decay=0.001,
        lr_scheduler_type="cosine", warmup_steps=2, max_grad_norm=0.3,
        fp16=False, gradient_checkpointing=True,
        save_strategy="no", eval_strategy="no", logging_steps=1,
        report_to="none", max_length=MAX_SEQ_LEN, dataset_text_field="text",
        average_tokens_across_devices=False,
    )
    timer = StepTimer()
    trainer = SFTTrainer(model=model, train_dataset=ds, peft_config=peft_config,
                         args=cfg, processing_class=tokenizer,
                         callbacks=[VRAMGuardCallback(), timer])

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    trainer.train()
    total = time.perf_counter() - t0

    peak_reserved = torch.cuda.max_memory_reserved(0)
    peak_alloc = torch.cuda.max_memory_allocated(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory

    steps = np.diff([t0] + timer.times)
    steady = steps[3:]  # skip warmup/compile steps
    gb = 1024 ** 3
    print("\n" + "=" * 62)
    print(f"steps measured              : {len(steps)} (grad-accum 16, batch 1, seq {MAX_SEQ_LEN})")
    print(f"total train() wall time     : {total:.1f} s")
    print(f"per-step: mean all          : {steps.mean():.2f} s/step")
    print(f"per-step: median steady     : {np.median(steady):.2f} s/step (steps 4-{len(steps)})")
    print(f"per-step: min / max steady  : {steady.min():.2f} / {steady.max():.2f} s")
    print(f"throughput                  : {16 * len(steps) / total:.2f} samples/s "
          f"({3600 / np.median(steady):.0f} steps/hour steady-state)")
    print(f"VRAM reserved after load    : {reserved_after_load / gb:.2f} GiB")
    print(f"PEAK VRAM reserved (train)  : {peak_reserved / gb:.2f} GiB "
          f"({peak_reserved / total_vram * 100:.1f}% of {total_vram / gb:.1f} GiB)")
    print(f"PEAK VRAM allocated (train) : {peak_alloc / gb:.2f} GiB")
    print("=" * 62)


if __name__ == "__main__":
    main()
