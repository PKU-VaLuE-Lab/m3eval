from __future__ import annotations

import math
import os
from pathlib import Path
from types import MethodType

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MODEL = None
_TOKENIZER = None
_MODEL_PATH = None
_INPUT_SIZE = None
_RUNTIME_CONFIG = None


def _split_internvl_model(num_layers: int) -> dict[str, int]:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise ValueError("INTERNVL_DEVICE_MAP=split requires at least two visible GPUs")

    layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    allocation = [layers_per_gpu] * world_size
    allocation[0] = math.ceil(allocation[0] * 0.5)
    device_map: dict[str, int] = {}
    layer_index = 0
    for device_index, count in enumerate(allocation):
        for _ in range(count):
            if layer_index >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_index}"] = device_index
            layer_index += 1
    device_map.update(
        {
            "vision_model": 0,
            "mlp1": 0,
            "language_model.model.embed_tokens": 0,
            "language_model.model.norm": 0,
            "language_model.lm_head": 0,
            f"language_model.model.layers.{num_layers - 1}": 0,
        }
    )
    return device_map


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_choice_env(name: str, default: str, allowed: set[str]) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in allowed:
        return value
    raise ValueError(f"Unsupported {name}={raw!r}. Allowed: {sorted(allowed)}")


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num}
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    tiles_per_row = target_width // image_size
    for tile_idx in range(blocks):
        box = (
            (tile_idx % tiles_per_row) * image_size,
            (tile_idx // tiles_per_row) * image_size,
            ((tile_idx % tiles_per_row) + 1) * image_size,
            ((tile_idx // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image: Image.Image, *, input_size: int, max_num: int) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


def _sync_num_image_token_for_input_size(model, *, input_size: int) -> None:
    vision_config = getattr(model.config, "vision_config", None)
    patch_size = int(getattr(vision_config, "patch_size", 14) or 14)
    downsample_ratio = float(getattr(model.config, "downsample_ratio", 0.5) or 0.5)
    if input_size % patch_size != 0:
        raise ValueError(f"InternVL input_size={input_size} must be divisible by patch_size={patch_size}")

    runtime_num_image_token = int((input_size // patch_size) ** 2 * (downsample_ratio**2))
    checkpoint_num_image_token = int(getattr(model, "num_image_token", runtime_num_image_token))
    if checkpoint_num_image_token != runtime_num_image_token:
        setattr(model, "num_image_token", runtime_num_image_token)


def _get_model_bundle():
    global _MODEL, _TOKENIZER, _MODEL_PATH, _INPUT_SIZE, _RUNTIME_CONFIG

    model_path = str(Path(os.getenv("INTERNVL_MODEL_PATH", "models/InternVL3_5-8B")).resolve())
    input_size = int(os.getenv("INTERNVL_INPUT_SIZE", "280") or "280")
    use_flash_attn = _parse_bool_env("INTERNVL_USE_FLASH_ATTN", default=True)
    device_map_name = os.getenv("INTERNVL_DEVICE_MAP", "cuda:0").strip()
    vision_batch_size = int(os.getenv("INTERNVL_VISION_BATCH_SIZE", "0") or "0")
    llm_attention = os.getenv("INTERNVL_LLM_ATTENTION", "").strip()
    runtime_config = (device_map_name, vision_batch_size, llm_attention, use_flash_attn)

    if _MODEL is not None and _TOKENIZER is not None and _MODEL_PATH == model_path and _INPUT_SIZE == input_size and _RUNTIME_CONFIG == runtime_config:
        return _MODEL, _TOKENIZER

    device_map: str | dict[str, int] = device_map_name
    if device_map_name == "split":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        device_map = _split_internvl_model(int(config.llm_config.num_hidden_layers))

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    if llm_attention:
        model.language_model.config._attn_implementation = llm_attention
        for layer in model.language_model.model.layers:
            layer.self_attn.config._attn_implementation = llm_attention
    if vision_batch_size > 0:
        original_extract_feature = model.extract_feature

        def chunked_extract_feature(model_self, pixel_values):
            if pixel_values.shape[0] <= vision_batch_size:
                return original_extract_feature(pixel_values)
            outputs = [original_extract_feature(chunk) for chunk in pixel_values.split(vision_batch_size, dim=0)]
            return torch.cat(outputs, dim=0)

        model.extract_feature = MethodType(chunked_extract_feature, model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    _sync_num_image_token_for_input_size(model, input_size=input_size)

    _MODEL = model
    _TOKENIZER = tokenizer
    _MODEL_PATH = model_path
    _INPUT_SIZE = input_size
    _RUNTIME_CONFIG = runtime_config
    return _MODEL, _TOKENIZER


def call_internvl_batched_yes_no(
    clip_frames: list[list[Image.Image]],
    question: str,
) -> str:
    model, tokenizer = _get_model_bundle()
    input_size = int(os.getenv("INTERNVL_INPUT_SIZE", "280") or "280")
    max_num = int(os.getenv("INTERNVL_MAX_NUM", "1") or "1")
    total_max_num = int(os.getenv("INTERNVL_TOTAL_MAX_NUM", "96") or "96")
    max_new_tokens = int(os.getenv("INTERNVL_MAX_NEW_TOKENS", "256") or "256")
    prompt_hint_mode = _parse_choice_env(
        "NBACK_PROMPT_HINT_MODE",
        default="structured",
        allowed={"structured", "minimal"},
    )

    visuals: list[Image.Image] = []
    prompt_lines: list[str] = []
    for clip_idx, frames in enumerate(clip_frames, start=1):
        if prompt_hint_mode == "structured":
            prompt_lines.append(f"Clip {clip_idx} in playback order.")
        for frame_idx, frame in enumerate(frames, start=1):
            visuals.append(frame)
            if prompt_hint_mode == "structured":
                prompt_lines.append(f"Clip {clip_idx} Frame {frame_idx}: <image>")
            else:
                prompt_lines.append("<image>")

    if not visuals:
        return "Error"

    dynamic_max_num = max(1, min(max_num, total_max_num // len(visuals)))
    processed_visuals = [load_image(frame, input_size=input_size, max_num=dynamic_max_num).to(torch.bfloat16).to("cuda:0") for frame in visuals]
    pixel_values = torch.cat(processed_visuals, dim=0)
    num_patches_list = [tensor.size(0) for tensor in processed_visuals]
    full_prompt = "\n".join(prompt_lines) + "\n" + question.strip()

    gen_kwargs = {
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    with torch.inference_mode():
        response, _ = model.chat(
            tokenizer,
            pixel_values,
            full_prompt,
            gen_kwargs,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
    return str(response).strip()
