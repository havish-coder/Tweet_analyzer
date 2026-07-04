"""
Qwen2.5-VL-3B-Instruct Tweet Media Enrichment
===============================================
Reverted from Florence-2: Qwen2.5-VL-3B gives better scene understanding
for lifestyle/photography images (the majority of marketing tweets).
The original failure was BUG-12 — a verbose markdown prompt. Fixed here.

Output format: single line, max 60 words, no markdown, no newlines.
  "Brand name / visible text. One-sentence scene description."

Resume: keyed on 'id' column — safe when files differ in row count.
Safety: INPUT_CSV is never written to.
"""

import os
import re
import gc
import sys
import tempfile
import torch
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV      = "data/train_enriched.csv"
OUTPUT_CSV     = "data/train_enriched.csv"
MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_INTERVAL  = 10
MAX_NEW_TOKENS = 80
TIMEOUT        = 10
MAX_DESC_WORDS = 60

# Video fallback (used when a row's static thumbnail is dead but an mp4 variant is live).
# Requires opencv-python (cv2); imported lazily in extract_video_frames so the module still
# loads — and eval.py can still import it — when cv2 is absent (fallback degrades to no-op).
#
# ESCAPE HATCH: set env VIDEO_FALLBACK=0 to disable the video path entirely and run the
# proven image-only enrichment (use this if a video clip triggers a CUDA device-side assert).
VIDEO_FALLBACK   = os.environ.get("VIDEO_FALLBACK", "1").lower() not in {"0", "false", "no"}
VIDEO_TIMEOUT    = 20                 # mp4s are larger than images
MAX_VIDEO_BYTES  = 25 * 1024 * 1024   # stream-download cap (~25 MB) — we only need a few frames
VIDEO_NUM_FRAMES = 6                  # K frames spanning the clip — primary VRAM knob
VIDEO_MAX_PIXELS = 256 * 28 * 28      # per-frame resolution cap fed to the VLM (low → fits 4 GB)

TMP_IMAGE_PATH = os.path.join(tempfile.gettempdir(), "qwen_current_img.jpg")
TMP_VIDEO_PATH = os.path.join(tempfile.gettempdir(), "qwen_current_vid.mp4")
TMP_FRAME_DIR  = os.path.join(tempfile.gettempdir(), "qwen_video_frames")

# Fixed prompt — concise, single-line, OCR-first (BUG-12 fix)
VLM_PROMPT = (
    "In one sentence, describe this image for a marketing tweet: "
    "first state any visible text or brand names, then describe the scene. "
    "No bullet points, no markdown, no line breaks."
)

# Same shape as VLM_PROMPT but for a multi-frame clip — asks for a whole-video summary.
VLM_VIDEO_PROMPT = (
    "In one sentence, summarize this video for a marketing tweet: "
    "first state any visible text or brand names, then describe the scene and action. "
    "No bullet points, no markdown, no line breaks."
)


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------
def extract_image_url(media_string: str):
    if not isinstance(media_string, str):
        return None
    m = re.search(r"fullUrl='(https?://[^']+)'", media_string)
    if m:
        return m.group(1)
    m = re.search(r"thumbnailUrl='(https?://[^']+)'", media_string)
    if m:
        return m.group(1)
    if media_string.startswith("http"):
        return media_string
    return None


def extract_video_url(media_string: str):
    """
    Return the lowest-bitrate .mp4 variant URL from a Video(...)/Gif(...) media string,
    or None if no mp4 variant is present (e.g. m3u8-only rows). Lowest bitrate = smallest
    download — we only need a handful of frames, not playback quality.
    """
    if not isinstance(media_string, str):
        return None

    # Variants look like: contentType='video/mp4', url='...mp4?tag=3', bitrate=2176000
    variants = re.findall(
        r"contentType='video/mp4',\s*url='(https?://[^']+?\.mp4[^']*)',\s*bitrate=(\d+|None)",
        media_string,
    )
    if variants:
        def _rate(v):
            return int(v[1]) if v[1].isdigit() else float("inf")
        return min(variants, key=_rate)[0]

    # Fallback: any .mp4 URL we can find, even without an advertised bitrate.
    m = re.search(r"url='(https?://[^']+?\.mp4[^']*)'", media_string)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_to_disk(url: str, save_path: str) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=TIMEOUT, headers=headers, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(save_path, format="JPEG")
        log(f"    [dl ok]  {url[:70]}  ({os.path.getsize(save_path)//1024} KB)")
        return True
    except Exception as e:
        log(f"    [dl fail]  {e}")
        return False


# ---------------------------------------------------------------------------
# Caption post-processing (shared by image and video paths)
# ---------------------------------------------------------------------------
def _finalize_caption(result: str) -> str:
    # Collapse any newlines the model snuck in despite the prompt
    result = re.sub(r"\s+", " ", result).strip()
    # Truncate to word budget
    words = result.split()
    if len(words) > MAX_DESC_WORDS:
        result = " ".join(words[:MAX_DESC_WORDS])
    return result


# ---------------------------------------------------------------------------
# VLM inference — fixed pixel shape, aggressive cleanup
# ---------------------------------------------------------------------------
def run_vlm(model, processor, image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": VLM_PROMPT},
            ],
        }
    ]

    inputs        = None
    generated_ids = None
    image_inputs  = None
    video_inputs  = None

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        del image_inputs, video_inputs
        image_inputs = video_inputs = None

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        trimmed = [out[len(inp):]
                   for inp, out in zip(inputs["input_ids"], generated_ids)]

        result = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        result = _finalize_caption(result)
        log(f"    [vlm]  {result[:120]}")
        return result

    finally:
        if inputs        is not None: del inputs
        if generated_ids is not None: del generated_ids
        if image_inputs  is not None: del image_inputs
        if video_inputs  is not None: del video_inputs
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Video fallback — sample frames spanning the whole clip, then summarize them
# ---------------------------------------------------------------------------
def extract_video_frames(video_url: str, out_dir: str, k: int = VIDEO_NUM_FRAMES) -> list:
    """
    Download the mp4 (capped at MAX_VIDEO_BYTES) and write up to `k` JPEG frames spread
    evenly across the full duration. Returns the list of frame paths (possibly fewer than
    `k`), or [] on any failure — including cv2 not being installed (graceful degrade).
    """
    try:
        import cv2
    except ImportError:
        log("    [video] opencv-python not installed — skipping video fallback")
        return []

    os.makedirs(out_dir, exist_ok=True)
    # Clear any frames left over from a previous row.
    for f in os.listdir(out_dir):
        delete_temp(os.path.join(out_dir, f))

    # --- stream-download the mp4 with a hard byte cap ---
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(video_url, timeout=VIDEO_TIMEOUT, headers=headers, stream=True)
        resp.raise_for_status()
        total = 0
        with open(TMP_VIDEO_PATH, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                fh.write(chunk)
                total += len(chunk)
                if total >= MAX_VIDEO_BYTES:
                    # Keep the partial file — it still decodes the leading frames.
                    log(f"    [video] size cap hit at {total // (1<<20)} MB — using partial clip")
                    break
        log(f"    [video] downloaded {total // 1024} KB")
    except Exception as e:
        log(f"    [video dl fail]  {e}")
        delete_temp(TMP_VIDEO_PATH)
        return []

    # --- sample frames ---
    cap = None
    frames = []
    try:
        cap = cv2.VideoCapture(TMP_VIDEO_PATH)
        if not cap.isOpened():
            log("    [video] cv2 could not open clip")
            return []

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            # Evenly-spaced indices across the full duration.
            idxs = [int(i * count / k) for i in range(k)]
            for n, fi in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                fp = os.path.join(out_dir, f"frame_{n:02d}.jpg")
                cv2.imwrite(fp, frame)
                frames.append(fp)
        else:
            # Frame count unreliable — read sequentially until we have k frames.
            n = 0
            while len(frames) < k:
                ok, frame = cap.read()
                if not ok:
                    break
                fp = os.path.join(out_dir, f"frame_{n:02d}.jpg")
                cv2.imwrite(fp, frame)
                frames.append(fp)
                n += 1

        log(f"    [video] sampled {len(frames)} frame(s)")
        return frames
    except Exception as e:
        log(f"    [video frame fail]  {e}")
        return frames
    finally:
        if cap is not None:
            cap.release()
        delete_temp(TMP_VIDEO_PATH)


def run_vlm_video(model, processor, frame_paths: list) -> str:
    """
    Summarize a clip from its sampled frames using Qwen2.5-VL's video input mode.
    On CUDA OOM, retries with a single frame so the run never dies on 4 GB.
    """
    if not frame_paths:
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_paths,
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {"type": "text", "text": VLM_VIDEO_PROMPT},
            ],
        }
    ]

    inputs        = None
    generated_ids = None
    image_inputs  = None
    video_inputs  = None

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        del image_inputs, video_inputs
        image_inputs = video_inputs = None

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        trimmed = [out[len(inp):]
                   for inp, out in zip(inputs["input_ids"], generated_ids)]

        result = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        result = _finalize_caption(result)
        log(f"    [vlm-video]  {result[:120]}")
        return result

    except torch.cuda.OutOfMemoryError:
        # Too many frames for 4 GB — fall back to captioning a single frame as an image.
        log("    [vlm-video] OOM — retrying on a single frame")
        gc.collect()
        torch.cuda.empty_cache()
        return run_vlm(model, processor, frame_paths[0])

    finally:
        if inputs        is not None: del inputs
        if generated_ids is not None: del generated_ids
        if image_inputs  is not None: del image_inputs
        if video_inputs  is not None: del video_inputs
        gc.collect()
        torch.cuda.empty_cache()


def fetch_media_caption(model, processor, media_string: str, img_path: str) -> str:
    """
    Unified media → caption entry point, shared by training enrichment and test-time
    enrichment so both see the same caption distribution.

      1. static image (Photo fullUrl / Video|Gif thumbnailUrl) → run_vlm
      2. fallback: lowest-bitrate mp4 → frames spanning the clip → run_vlm_video

    Returns "" when neither path yields a caption (caller maps "" → failure label).
    """
    url = extract_image_url(media_string)
    if url:
        log(f"    [url]  {url[:80]}")
        if download_to_disk(url, img_path):
            try:
                return run_vlm(model, processor, img_path)
            finally:
                delete_temp(img_path)

    if not VIDEO_FALLBACK:
        return ""   # video path disabled (VIDEO_FALLBACK=0) — image-only enrichment

    video_url = extract_video_url(media_string)
    if video_url:
        log(f"    [video url]  {video_url[:80]}")
        frames = extract_video_frames(video_url, TMP_FRAME_DIR)
        if frames:
            try:
                return run_vlm_video(model, processor, frames)
            finally:
                for fp in frames:
                    delete_temp(fp)

    return ""


# ---------------------------------------------------------------------------
# Delete temp file
# ---------------------------------------------------------------------------
def delete_temp(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Resume by id — safe when row counts differ
# ---------------------------------------------------------------------------
def load_with_resume(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "vlm_description" not in df.columns:
        df["vlm_description"] = ""

    if os.path.exists(output_path) and output_path != input_path:
        saved = pd.read_csv(output_path)
        if "id" in saved.columns and "vlm_description" in saved.columns:
            saved = saved[["id", "vlm_description"]].rename(
                columns={"vlm_description": "_saved"})
            df = df.merge(saved, on="id", how="left")
            mask = df["_saved"].notna() & (df["_saved"].astype(str).str.strip() != "")
            df.loc[mask, "vlm_description"] = df.loc[mask, "_saved"]
            df.drop(columns=["_saved"], inplace=True)
            log(f"Resumed: merged saved descriptions by id")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 60)
    log(f"  Qwen2.5-VL-3B Enrichment  (fixed prompt)")
    log(f"  Input  : {INPUT_CSV}")
    log(f"  Output : {OUTPUT_CSV}")
    log("=" * 60)

    if not os.path.exists(INPUT_CSV):
        log(f"ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    log("Loading model...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log("Model ready.\n")

    df = load_with_resume(INPUT_CSV, OUTPUT_CSV)

    needs   = df["vlm_description"].isna() | (df["vlm_description"].astype(str).str.strip() == "")
    indices = df.index[needs].tolist()
    log(f"Already done : {len(df) - len(indices)} / {len(df)}")
    log(f"Remaining    : {len(indices)}\n")

    if not indices:
        log("Nothing to do.")
        return

    loop_n = 0
    try:
        for idx in tqdm(indices, desc="VLM Progress", file=sys.stdout):
            row = df.loc[idx]
            log(f"\n{'─'*60}")
            log(f"  Row {idx}  |  @{row.get('username','?')}  |  #{loop_n+1}")

            media_str = str(row.get("media", ""))
            # "no media" only when there is neither a static image URL nor a video URL.
            if not extract_image_url(media_str) and not extract_video_url(media_str):
                df.at[idx, "vlm_description"] = "no media"
                loop_n += 1
                continue

            try:
                # Static image first; falls back to a whole-video summary internally.
                desc = fetch_media_caption(model, processor, media_str, TMP_IMAGE_PATH)
                df.at[idx, "vlm_description"] = desc if desc else "media could not be processed"
            except Exception as e:
                log(f"    [error]  {e}")
                df.at[idx, "vlm_description"] = "media could not be processed"

            loop_n += 1

            if loop_n % SAVE_INTERVAL == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                log(f"\n  [checkpoint]  {loop_n} done → {OUTPUT_CSV}")

    except KeyboardInterrupt:
        log("\nInterrupted — saving...")
    finally:
        delete_temp(TMP_IMAGE_PATH)
        df.to_csv(OUTPUT_CSV, index=False)
        log(f"\nFinal save → {OUTPUT_CSV}")

    valid  = df["vlm_description"].notna() & ~df["vlm_description"].isin(
                 ["", "no media", "media could not be processed"])
    log(f"\n{'='*60}")
    log(f"  Valid desc  : {valid.sum()}")
    log(f"  No media    : {(df['vlm_description']=='no media').sum()}")
    log(f"  Failed      : {(df['vlm_description']=='media could not be processed').sum()}")
    log(f"  Pending     : {df['vlm_description'].isna().sum()}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
