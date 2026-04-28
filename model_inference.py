"""
model_inference.py — Load Qwen3-VL and run zero-shot inference on video frames.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import List, Optional

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

import config

logger = logging.getLogger(__name__)

_model = None
_processor: Optional[AutoProcessor] = None
_loaded_model_id: Optional[str] = None


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _should_use_4bit(model_id: str) -> bool:
    lowered = model_id.lower()
    large_markers = ("7b", "8b", "9b", "27b", "30b", "32b", "35b", "72b", "122b")
    return any(marker in lowered for marker in large_markers)


def load_model(model_id: str = config.DEFAULT_MODEL) -> None:
    global _model, _processor, _loaded_model_id

    if _model is not None and _loaded_model_id == model_id:
        return

    if _model is not None:
        logger.info("Unloading previous model: %s", _loaded_model_id)
        del _model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Loading model: %s", model_id)

    model_kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }
    if _should_use_4bit(model_id):
        logger.info("Using 4-bit quantisation (BitsAndBytes) for %s", model_id)
        model_kwargs["quantization_config"] = _build_bnb_config()

    _model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    _model.eval()

    _processor = AutoProcessor.from_pretrained(model_id)
    _loaded_model_id = model_id

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("GPU memory after loading: %.2f GB", allocated)


def run_inference(frames: List[Image.Image], prompt: str, model_id: str = config.DEFAULT_MODEL) -> str:
    load_model(model_id)

    image_content = [{"type": "image", "image": frame} for frame in frames]
    messages = [{"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}]

    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generation_kwargs = {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "do_sample": config.DO_SAMPLE,
    }
    if config.DO_SAMPLE:
        generation_kwargs["temperature"] = config.TEMPERATURE

    with torch.no_grad():
        output_ids = _model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    return _processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]