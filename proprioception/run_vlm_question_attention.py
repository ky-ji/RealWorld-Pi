#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import safetensors.torch
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_runtime_root() -> Path:
    candidates = [
        os.environ.get("OPENPI_RUNTIME_ROOT"),
        str(REPO_ROOT.parent / "openpi"),
        "/data3/yinmenghao/code/openpi",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "src" / "openpi").exists():
            return path
    raise FileNotFoundError(
        "Could not find a usable OpenPI runtime checkout. "
        "Set OPENPI_RUNTIME_ROOT to a repo that contains src/openpi."
    )


RUNTIME_ROOT = resolve_runtime_root()
RUNTIME_SRC = RUNTIME_ROOT / "src"
for path in (str(RUNTIME_ROOT), str(RUNTIME_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from examples.convert_jax_model_to_pytorch import (  # noqa: E402
    slice_gemma_state_dict,
    slice_initial_orbax_checkpoint,
    slice_paligemma_state_dict,
)
from openpi import transforms as _transforms  # noqa: E402
from openpi.models import gemma as _gemma  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.models import tokenizer as _tokenizer  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402


DEFAULT_PROMPTS = [
    "Where is the phone in this image?",
    "Where is the stand in this image?",
    "Where is the robot gripper in this image?",
]

VIEW_NAMES = ("base_0_rgb", "left_wrist_0_rgb")
VIEW_LABELS = {
    "base_0_rgb": "front",
    "left_wrist_0_rgb": "wrist",
}


class VideoFrameReader:
    def __init__(self, video_path: Path):
        self._path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

    def read_frame(self, frame_idx: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self._path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pi0.5 VLM-only attention overlays on a PlacePhone frame.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/data3/yinmenghao/code/openpi/checkpoints/place_phone_lora_0211/"
            "pi05_place_phone_lora/place_phone_lora_v2/19999"
        ),
        help="JAX checkpoint directory used only when the cached PyTorch checkpoint is missing.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("/data3/jikangye/openpi_attention/pi05_place_phone_lora_19999_pytorch"),
        help="Converted PyTorch checkpoint cache directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data1/vla-data/processed/PI/data/009_place_phone"),
        help="LeRobot dataset root containing data/ videos/ meta/.",
    )
    parser.add_argument("--episode-id", type=int, default=155, help="Episode index to inspect.")
    parser.add_argument("--step", type=int, default=400, help="Frame index inside the episode.")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt/question to visualize. Repeat the flag to add more prompts.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="12",
        help="Comma-separated Gemma language-model attention layers to save.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Precision used when caching the converted PyTorch model.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.35,
        help="Overlay blending alpha.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "proprioception" / "artifacts" / "vlm_question_attention" / "place_phone_episode0155_step00400",
        help="Directory to write overlays, heatmaps, and summary files.",
    )
    return parser.parse_args()


def parse_layer_spec(spec: str) -> list[int]:
    return [int(token.strip()) for token in spec.split(",") if token.strip()]


def build_model_config() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=7,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )


def create_flexible_pi0_pytorch(model_config: pi0_config.Pi0Config) -> PI0Pytorch:
    model = PI0Pytorch(model_config)
    action_expert_config = _gemma.get_config(model_config.action_expert_variant)
    model.action_in_proj = torch.nn.Linear(model_config.action_dim, action_expert_config.width)
    model.action_out_proj = torch.nn.Linear(action_expert_config.width, model_config.action_dim)
    if not model_config.pi05:
        model.state_proj = torch.nn.Linear(model_config.action_dim, action_expert_config.width)
    return model


def build_paligemma_config_proxy() -> Any:
    class PaliGemmaConfigProxy:
        def __init__(self):
            self.vision_config = type(
                "VisionConfig",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "TextConfig",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    return PaliGemmaConfigProxy()


def convert_jax_checkpoint_to_state_dict(
    checkpoint_dir: Path, model_config: pi0_config.Pi0Config
) -> dict[str, torch.Tensor]:
    initial_params = slice_initial_orbax_checkpoint(str(checkpoint_dir), restore_precision="float32")

    projection_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    projection_params: dict[str, torch.Tensor] = {}
    for key in projection_keys:
        kernel = initial_params["projection_params"][key]["kernel"]
        bias = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel, dict):
            kernel = kernel["value"]
            bias = bias["value"]
        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(kernel)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    paligemma_config = build_paligemma_config_proxy()
    action_expert_config = _gemma.get_config("gemma_300m")
    paligemma_params, expert_params = slice_paligemma_state_dict(
        initial_params["paligemma_params"], paligemma_config
    )
    gemma_params = slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        checkpoint_dir=str(checkpoint_dir),
        pi05=model_config.pi05,
    )
    return {**paligemma_params, **gemma_params, **projection_params}


def maybe_cache_converted_model(
    checkpoint_dir: Path,
    cache_dir: Path,
    model_config: pi0_config.Pi0Config,
    precision: str,
) -> None:
    model_path = cache_dir / "model.safetensors"
    if model_path.exists():
        return

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Converted model cache is missing and checkpoint_dir is not accessible: {checkpoint_dir}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    model = create_flexible_pi0_pytorch(model_config)
    state_dict = convert_jax_checkpoint_to_state_dict(checkpoint_dir, model_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.float32 if precision == "float32" else torch.bfloat16)

    safetensors.torch.save_model(model, str(model_path))

    assets_dest = cache_dir / "assets"
    for assets_source in (checkpoint_dir / "assets", checkpoint_dir.parent / "assets"):
        if assets_source.exists() and not assets_dest.exists():
            shutil.copytree(assets_source, assets_dest)
            break

    config_json = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "max_token_len": model_config.max_token_len,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
        "runtime_root": str(RUNTIME_ROOT),
    }
    (cache_dir / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")


def load_model_from_cache(cache_dir: Path, model_config: pi0_config.Pi0Config, device: str) -> PI0Pytorch:
    model = create_flexible_pi0_pytorch(model_config)
    safetensors.torch.load_model(model, str(cache_dir / "model.safetensors"))
    model = model.to(device)
    model.eval()
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    return model


def load_norm_stats(
    cache_dir: Path,
    checkpoint_dir: Path,
    asset_id: str,
) -> dict[str, _normalize.NormStats]:
    for base_dir in (
        cache_dir / "assets",
        checkpoint_dir / "assets",
        checkpoint_dir.parent / "assets",
    ):
        stats_dir = base_dir / asset_id
        stats_path = stats_dir / "norm_stats.json"
        if stats_path.exists():
            return _normalize.load(stats_dir)
    raise FileNotFoundError(
        f"Could not find norm_stats.json for asset_id={asset_id!r} under the cached model or checkpoint assets."
    )


def load_episode_parquet(dataset_dir: Path, episode_id: int) -> pd.DataFrame:
    parquet_path = dataset_dir / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def load_step_observation(dataset_dir: Path, episode_id: int, step_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = load_episode_parquet(dataset_dir, episode_id)
    if step_idx < 0 or step_idx >= len(df):
        raise IndexError(f"Step {step_idx} is out of range for episode {episode_id} with {len(df)} steps.")

    state = np.asarray(df["state"].iloc[step_idx], dtype=np.float32)
    front_reader = VideoFrameReader(dataset_dir / "videos" / "chunk-000" / "front_view" / f"episode_{episode_id:06d}.mp4")
    wrist_reader = VideoFrameReader(dataset_dir / "videos" / "chunk-000" / "wrist_view" / f"episode_{episode_id:06d}.mp4")
    try:
        front_image = front_reader.read_frame(step_idx)
        wrist_image = wrist_reader.read_frame(step_idx)
    finally:
        front_reader.close()
        wrist_reader.close()

    return front_image, wrist_image, state


def place_phone_inputs(front_image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, prompt: str) -> dict[str, Any]:
    return {
        "state": np.asarray(state, dtype=np.float32),
        "image": {
            "base_0_rgb": np.asarray(front_image, dtype=np.uint8),
            "left_wrist_0_rgb": np.asarray(wrist_image, dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros_like(front_image, dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        },
        "prompt": prompt,
    }


def build_input_pipeline(
    model_config: pi0_config.Pi0Config,
    norm_stats: dict[str, _normalize.NormStats],
    tokenizer: _tokenizer.PaligemmaTokenizer,
) -> _transforms.DataTransformFn:
    transforms = [
        _transforms.Normalize(norm_stats, use_quantiles=True),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            tokenizer,
            discrete_state_input=model_config.discrete_state_input,
        ),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]
    return _transforms.compose(transforms)


def to_torch_batch(data: dict[str, Any], device: str) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.array(value)).to(device)[None, ...]
        if isinstance(value, (np.bool_, bool, np.number)):
            return torch.as_tensor(np.asarray(value), device=device)[None, ...]
        raise TypeError(f"Unsupported value type: {type(value)}")

    return {key: convert(value) for key, value in data.items()}


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - heatmap.min()
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def build_attention_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image.astype(np.uint8), 1.0 - alpha, colored, alpha, 0.0)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


def sanitize_name(text: str) -> str:
    keep = []
    for char in text.lower():
        if char.isalnum():
            keep.append(char)
        elif char in {" ", "_", "-"}:
            keep.append("_")
    return "".join(keep).strip("_")[:80] or "prompt"


def infer_focus_phrase(prompt: str) -> str | None:
    normalized = prompt.strip().lower().replace("?", "")
    prefix = "where is the "
    suffix = " in this image"
    if normalized.startswith(prefix) and normalized.endswith(suffix):
        return normalized[len(prefix) : -len(suffix)].strip()
    return None


def find_focus_token_positions(
    tokenizer: _tokenizer.PaligemmaTokenizer,
    prompt: str,
    prompt_tokens: torch.Tensor,
    prompt_len: int,
    text_start: int,
) -> list[int]:
    focus_phrase = infer_focus_phrase(prompt)
    if not focus_phrase:
        return []

    target_tokens = tokenizer._tokenizer.encode(focus_phrase, add_bos=False)  # noqa: SLF001
    if not target_tokens:
        return []

    prompt_token_list = prompt_tokens[:prompt_len].detach().cpu().tolist()
    matched_positions: list[int] = []
    window = len(target_tokens)
    for idx in range(0, len(prompt_token_list) - window + 1):
        if prompt_token_list[idx : idx + window] == target_tokens:
            matched_positions.extend(range(idx, idx + window))
    return [text_start + idx for idx in matched_positions]


def compute_vlm_heatmaps(
    model: PI0Pytorch,
    processed_inputs: dict[str, Any],
    view_images: dict[str, np.ndarray],
    layer_idx: int,
    prompt: str,
    tokenizer: _tokenizer.PaligemmaTokenizer,
    device: str,
) -> dict[str, np.ndarray]:
    batched = to_torch_batch(processed_inputs, device)
    observation = _model.Observation.from_dict(batched)
    images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(observation, train=False)

    image_token_counts: list[int] = []
    with torch.inference_mode():
        for image in images:
            image_token_counts.append(int(model.paligemma_with_expert.embed_image(image).shape[1]))

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_attn_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_attn_mask_4d = model._prepare_attention_masks_4d(prefix_attn_mask)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        outputs = model.paligemma_with_expert.paligemma.language_model(
            attention_mask=prefix_attn_mask_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    attention = outputs.attentions[layer_idx].detach().float().cpu()[0]
    prompt_len = int(lang_masks[0].sum().item())
    if prompt_len <= 0:
        raise RuntimeError("Prompt tokenization produced zero valid tokens.")

    text_start = int(sum(image_token_counts))
    prompt_positions = find_focus_token_positions(tokenizer, prompt, lang_tokens[0], prompt_len, text_start)
    if not prompt_positions:
        prompt_positions = list(range(text_start, text_start + prompt_len))
        if prompt_len > 2:
            prompt_positions = prompt_positions[1:-1]
    if not prompt_positions:
        prompt_positions = list(range(text_start, text_start + prompt_len))

    heatmaps: dict[str, np.ndarray] = {}
    image_offset = 0
    for view_name, image, image_mask, token_count in zip(
        VIEW_NAMES,
        images[: len(VIEW_NAMES)],
        img_masks[: len(VIEW_NAMES)],
        image_token_counts[: len(VIEW_NAMES)],
        strict=True,
    ):
        _ = image
        if not bool(image_mask[0].item()):
            image_offset += token_count
            continue

        token_attention = attention[:, prompt_positions, image_offset : image_offset + token_count]
        token_attention = token_attention.mean(dim=(0, 1)).numpy()
        side = int(round(np.sqrt(token_count)))
        if side * side != token_count:
            raise RuntimeError(f"Expected a square number of image tokens, got {token_count}")

        heatmap = token_attention.reshape(side, side)
        height, width = view_images[view_name].shape[:2]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmaps[view_name] = normalize_heatmap(heatmap)
        image_offset += token_count

    return heatmaps


def save_prompt_outputs(
    prompt_dir: Path,
    layer_idx: int,
    view_images: dict[str, np.ndarray],
    heatmaps: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, dict[str, str]]:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, dict[str, str]] = {}
    for view_name in VIEW_NAMES:
        image = view_images[view_name]
        input_path = prompt_dir / f"{view_name}_input.jpg"
        save_rgb(input_path, image)
        saved.setdefault(view_name, {})["input_image"] = str(input_path)

        if view_name not in heatmaps:
            continue

        overlay = build_attention_overlay(image, heatmaps[view_name], alpha)
        overlay_path = prompt_dir / f"{view_name}_layer_{layer_idx:02d}_overlay.jpg"
        heatmap_path = prompt_dir / f"{view_name}_layer_{layer_idx:02d}_heatmap.npy"
        save_rgb(overlay_path, overlay)
        np.save(heatmap_path, heatmaps[view_name].astype(np.float32))
        saved[view_name]["overlay_image"] = str(overlay_path)
        saved[view_name]["heatmap_npy"] = str(heatmap_path)
    return saved


def save_summary_figure(
    summary_path: Path,
    layer_idx: int,
    prompts: list[str],
    view_images: dict[str, np.ndarray],
    heatmaps_by_prompt: list[dict[str, np.ndarray]],
    alpha: float,
) -> None:
    nrows = len(VIEW_NAMES)
    ncols = len(prompts) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])

    for row_idx, view_name in enumerate(VIEW_NAMES):
        input_ax = axes[row_idx, 0]
        input_ax.imshow(view_images[view_name])
        input_ax.set_title(f"{VIEW_LABELS[view_name]} input", fontsize=12)
        input_ax.axis("off")

        for col_idx, prompt in enumerate(prompts, start=1):
            ax = axes[row_idx, col_idx]
            heatmap = heatmaps_by_prompt[col_idx - 1][view_name]
            overlay = build_attention_overlay(view_images[view_name], heatmap, alpha)
            ax.imshow(overlay)
            ax.set_title(prompt, fontsize=11)
            ax.axis("off")

    fig.suptitle(f"Pi0.5 VLM attention overlays, layer {layer_idx}", fontsize=16)
    fig.tight_layout()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    layers = parse_layer_spec(args.layers)
    prompts = args.prompt or DEFAULT_PROMPTS
    model_config = build_model_config()

    print(f"[vlm_attention] runtime_root: {RUNTIME_ROOT}")
    print(f"[vlm_attention] checkpoint: {args.checkpoint_dir}")
    print(f"[vlm_attention] model_cache: {args.model_cache_dir}")
    print(f"[vlm_attention] dataset: {args.dataset_dir}")
    print(f"[vlm_attention] episode: {args.episode_id}")
    print(f"[vlm_attention] step: {args.step}")
    print(f"[vlm_attention] prompts: {len(prompts)}")
    print(f"[vlm_attention] layers: {layers}")
    print(f"[vlm_attention] output_dir: {args.output_dir}")

    maybe_cache_converted_model(args.checkpoint_dir, args.model_cache_dir, model_config, args.precision)
    model = load_model_from_cache(args.model_cache_dir, model_config, args.device)
    norm_stats = load_norm_stats(args.model_cache_dir, args.checkpoint_dir, args.dataset_dir.name)
    prompt_tokenizer = _tokenizer.PaligemmaTokenizer(model_config.max_token_len)
    input_pipeline = build_input_pipeline(model_config, norm_stats, prompt_tokenizer)

    front_image, wrist_image, state = load_step_observation(args.dataset_dir, args.episode_id, args.step)

    summary: dict[str, Any] = {
        "runtime_root": str(RUNTIME_ROOT),
        "checkpoint_dir": str(args.checkpoint_dir),
        "model_cache_dir": str(args.model_cache_dir),
        "dataset_dir": str(args.dataset_dir),
        "episode_id": args.episode_id,
        "step": args.step,
        "layers": layers,
        "prompts": prompts,
        "results": [],
    }

    for layer_idx in layers:
        heatmaps_by_prompt: list[dict[str, np.ndarray]] = []
        view_images: dict[str, np.ndarray] | None = None

        for prompt in prompts:
            structured = place_phone_inputs(front_image, wrist_image, state, prompt)
            processed = input_pipeline(structured)
            view_images = {
                "base_0_rgb": np.asarray(processed["image"]["base_0_rgb"], dtype=np.uint8),
                "left_wrist_0_rgb": np.asarray(processed["image"]["left_wrist_0_rgb"], dtype=np.uint8),
            }

            heatmaps = compute_vlm_heatmaps(
                model,
                processed,
                view_images,
                layer_idx,
                prompt,
                prompt_tokenizer,
                args.device,
            )
            heatmaps_by_prompt.append(heatmaps)

            prompt_dir = args.output_dir / f"layer_{layer_idx:02d}" / sanitize_name(prompt)
            saved_files = save_prompt_outputs(
                prompt_dir=prompt_dir,
                layer_idx=layer_idx,
                view_images=view_images,
                heatmaps=heatmaps,
                alpha=args.overlay_alpha,
            )
            summary["results"].append(
                {
                    "layer": layer_idx,
                    "prompt": prompt,
                    "saved_files": saved_files,
                }
            )
            print(f"[vlm_attention] layer={layer_idx} prompt='{prompt}' saved")

        if view_images is None:
            continue

        summary_path = args.output_dir / f"layer_{layer_idx:02d}" / "summary_overlay.png"
        save_summary_figure(summary_path, layer_idx, prompts, view_images, heatmaps_by_prompt, args.overlay_alpha)
        print(f"[vlm_attention] layer={layer_idx} summary: {summary_path}")

    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[vlm_attention] wrote summary json: {json_path}")


if __name__ == "__main__":
    main()
