"""
VLM caption-latency probe: Qwen2.5-VL-3B-Instruct (4-bit NF4), 10 images,
same prompt and generation settings as enrich_vlm.py.

Reports model load time separately from per-image caption time (first caption
reported separately — it includes CUDA kernel warmup).

Images are locally generated (PIL) at a typical tweet-media resolution, so the
measurement is pure VLM inference, no network variance.

Run from Task-2/:  python src/probe_vlm.py
"""

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
from PIL import Image, ImageDraw

from enrich_vlm import MODEL_ID, VLM_PROMPT, MAX_NEW_TOKENS

N_IMAGES = 10


def make_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(768, 1024, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 900, 300], fill=(255, 255, 255))
    d.text((120, 150), f"BRAND {seed} — SUMMER SALE 50% OFF", fill=(0, 0, 0))
    return img


def main():
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)

    print(f"Loading {MODEL_ID} (4-bit)...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto",
        quantization_config=quant, low_cpu_mem_usage=True)
    model.eval()
    load_time = time.perf_counter() - t0

    times, captions = [], []
    for i in range(N_IMAGES):
        img = make_image(i)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": VLM_PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        times.append(time.perf_counter() - t1)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        captions.append(processor.decode(trimmed, skip_special_tokens=True).strip())
        print(f"  image {i + 1:2d}: {times[-1]:6.2f} s | {captions[-1][:70]}")

    steady = np.array(times[1:])
    gb = 1024 ** 3
    print("\n" + "=" * 62)
    print(f"model load time             : {load_time:.1f} s")
    print(f"first caption (warmup)      : {times[0]:.2f} s")
    print(f"steady-state caption time   : mean {steady.mean():.2f} s  median {np.median(steady):.2f} s  "
          f"min {steady.min():.2f}  max {steady.max():.2f}  (n={len(steady)})")
    print(f"peak VRAM reserved          : {torch.cuda.max_memory_reserved(0) / gb:.2f} GiB")
    print("=" * 62)


if __name__ == "__main__":
    main()
