#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

import run_vlm_question_attention as attn


@dataclass(frozen=True)
class CaseSpec:
    name: str
    episode_id: int
    steps: list[int]
    prompts: list[str]


@dataclass(frozen=True)
class BundleSpec:
    name: str
    checkpoint_dir: Path
    model_cache_dir: Path
    dataset_dir: Path
    cases: list[CaseSpec]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-task Pi0.5 VLM-only attention suite.")
    parser.add_argument(
        "--layers",
        type=str,
        default="8,10,12,14,16",
        help="Comma-separated Gemma language-model layers to visualize.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Precision used for converted PyTorch model caches.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.35,
        help="Overlay blending alpha.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Optional bundle or case filter. Repeat to keep multiple names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=attn.REPO_ROOT / "proprioception" / "artifacts" / "vlm_attention_suite" / "default_suite",
        help="Directory to write suite artifacts.",
    )
    return parser.parse_args()


def build_default_suite() -> list[BundleSpec]:
    return [
        BundleSpec(
            name="place_phone",
            checkpoint_dir=Path(
                "/data3/yinmenghao/code/openpi/checkpoints/place_phone_lora_0211/"
                "pi05_place_phone_lora/place_phone_lora_v2/19999"
            ),
            model_cache_dir=Path("/data3/jikangye/openpi_attention/pi05_place_phone_lora_19999_pytorch"),
            dataset_dir=Path("/data1/vla-data/processed/PI/data/009_place_phone"),
            cases=[
                CaseSpec(
                    name="place_phone_main",
                    episode_id=155,
                    steps=[120, 240, 360, 480],
                    prompts=[
                        "Where is the phone in this image?",
                        "Where is the stand in this image?",
                        "Where is the robot gripper in this image?",
                    ],
                )
            ],
        ),
        BundleSpec(
            name="stack_bowls",
            checkpoint_dir=Path(
                "/data3/yinmenghao/code/openpi/checkpoints/stack_bowls_lora_0208/"
                "pi05_stack_bowls_lora/stack_bowls_lora_v2/29999"
            ),
            model_cache_dir=Path("/data3/jikangye/openpi_attention/pi05_stack_bowls_lora_29999_pytorch"),
            dataset_dir=Path("/data1/vla-data/processed/PI/data/010_stack_bowls_0209"),
            cases=[
                CaseSpec(
                    name="stack_bowls_main",
                    episode_id=0,
                    steps=[120, 240, 360, 480],
                    prompts=[
                        "Where is the bowl stack in this image?",
                        "Where is the bowl in this image?",
                        "Where is the robot gripper in this image?",
                    ],
                )
            ],
        ),
        BundleSpec(
            name="assembly_things",
            checkpoint_dir=Path(
                "/data3/yinmenghao/code/openpi/checkpoints/assembly_things_lora_0209/"
                "pi05_assembly_things_lora/assembly_things_lora_v1/14999"
            ),
            model_cache_dir=Path("/data3/jikangye/openpi_attention/pi05_assembly_things_lora_14999_pytorch"),
            dataset_dir=Path("/data1/vla-data/processed/PI/data/001_assembly_things_0209"),
            cases=[
                CaseSpec(
                    name="pineapple_bun",
                    episode_id=41,
                    steps=[80, 160, 240],
                    prompts=[
                        "Where is the pineapple bun in this image?",
                        "Where is the plate in this image?",
                        "Where is the robot gripper in this image?",
                    ],
                ),
                CaseSpec(
                    name="toast",
                    episode_id=216,
                    steps=[80, 140, 200],
                    prompts=[
                        "Where is the toast in this image?",
                        "Where is the plate in this image?",
                        "Where is the robot gripper in this image?",
                    ],
                ),
                CaseSpec(
                    name="chocolate",
                    episode_id=420,
                    steps=[80, 140, 200],
                    prompts=[
                        "Where is the chocolate in this image?",
                        "Where is the plate in this image?",
                        "Where is the robot gripper in this image?",
                    ],
                ),
            ],
        ),
    ]


def filter_suite(suite: list[BundleSpec], keep_names: set[str] | None) -> list[BundleSpec]:
    if not keep_names:
        return suite

    filtered: list[BundleSpec] = []
    for bundle in suite:
        if bundle.name in keep_names:
            filtered.append(bundle)
            continue

        cases = [case for case in bundle.cases if case.name in keep_names]
        if cases:
            filtered.append(
                BundleSpec(
                    name=bundle.name,
                    checkpoint_dir=bundle.checkpoint_dir,
                    model_cache_dir=bundle.model_cache_dir,
                    dataset_dir=bundle.dataset_dir,
                    cases=cases,
                )
            )
    return filtered


def compose_step_layer_grid(
    image_paths_by_step_layer: dict[int, dict[int, Path]],
    steps: list[int],
    layers: list[int],
    output_path: Path,
) -> None:
    first_path = next(iter(next(iter(image_paths_by_step_layer.values())).values()))
    sample = cv2.imread(str(first_path))
    if sample is None:
        raise RuntimeError(f"Failed to read summary image: {first_path}")

    tile_w = 360
    tile_h = max(220, int(sample.shape[0] * tile_w / sample.shape[1]))
    label_h = 36
    margin = 12
    header_w = 96
    canvas_h = label_h + len(steps) * (tile_h + margin) + margin
    canvas_w = header_w + len(layers) * (tile_w + margin) + margin
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for col_idx, layer_idx in enumerate(layers):
        x = header_w + margin + col_idx * (tile_w + margin)
        cv2.putText(
            canvas,
            f"layer {layer_idx}",
            (x + 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    for row_idx, step in enumerate(steps):
        y = label_h + margin + row_idx * (tile_h + margin)
        cv2.putText(
            canvas,
            f"step {step}",
            (10, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for col_idx, layer_idx in enumerate(layers):
            summary_path = image_paths_by_step_layer[step][layer_idx]
            image = cv2.imread(str(summary_path))
            if image is None:
                raise RuntimeError(f"Failed to read summary image: {summary_path}")
            image = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            x = header_w + margin + col_idx * (tile_w + margin)
            canvas[y : y + tile_h, x : x + tile_w] = image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def run_bundle(
    bundle: BundleSpec,
    layers: list[int],
    args: argparse.Namespace,
    model_config: object,
) -> dict[str, object]:
    attn.maybe_cache_converted_model(bundle.checkpoint_dir, bundle.model_cache_dir, model_config, args.precision)
    model = attn.load_model_from_cache(bundle.model_cache_dir, model_config, args.device)
    norm_stats = attn.load_norm_stats(bundle.model_cache_dir, bundle.checkpoint_dir, bundle.dataset_dir.name)
    prompt_tokenizer = attn._tokenizer.PaligemmaTokenizer(model_config.max_token_len)
    input_pipeline = attn.build_input_pipeline(model_config, norm_stats, prompt_tokenizer)

    bundle_result: dict[str, object] = {
        "bundle": bundle.name,
        "checkpoint_dir": str(bundle.checkpoint_dir),
        "model_cache_dir": str(bundle.model_cache_dir),
        "dataset_dir": str(bundle.dataset_dir),
        "cases": [],
    }

    for case in bundle.cases:
        print(f"[suite] bundle={bundle.name} case={case.name} episode={case.episode_id} steps={case.steps}")
        case_root = args.output_dir / bundle.name / case.name / f"episode_{case.episode_id:06d}"
        summary_paths_by_step_layer: dict[int, dict[int, Path]] = {}
        case_result: dict[str, object] = {
            "case": case.name,
            "episode_id": case.episode_id,
            "steps": case.steps,
            "prompts": case.prompts,
            "layers": layers,
            "outputs": [],
        }

        for step in case.steps:
            front_image, wrist_image, state = attn.load_step_observation(bundle.dataset_dir, case.episode_id, step)
            summary_paths_by_step_layer[step] = {}
            step_root = case_root / f"step_{step:04d}"

            for layer_idx in layers:
                heatmaps_by_prompt: list[dict[str, np.ndarray]] = []
                view_images: dict[str, np.ndarray] | None = None
                saved_prompts: list[dict[str, object]] = []

                for prompt in case.prompts:
                    structured = attn.place_phone_inputs(front_image, wrist_image, state, prompt)
                    processed = input_pipeline(structured)
                    view_images = {
                        "base_0_rgb": np.asarray(processed["image"]["base_0_rgb"], dtype=np.uint8),
                        "left_wrist_0_rgb": np.asarray(processed["image"]["left_wrist_0_rgb"], dtype=np.uint8),
                    }
                    heatmaps = attn.compute_vlm_heatmaps(
                        model,
                        processed,
                        view_images,
                        layer_idx,
                        prompt,
                        prompt_tokenizer,
                        args.device,
                    )
                    heatmaps_by_prompt.append(heatmaps)

                    prompt_dir = step_root / f"layer_{layer_idx:02d}" / attn.sanitize_name(prompt)
                    saved_files = attn.save_prompt_outputs(
                        prompt_dir=prompt_dir,
                        layer_idx=layer_idx,
                        view_images=view_images,
                        heatmaps=heatmaps,
                        alpha=args.overlay_alpha,
                    )
                    saved_prompts.append({"prompt": prompt, "saved_files": saved_files})

                if view_images is None:
                    raise RuntimeError(f"No images were processed for {bundle.name}/{case.name}/step {step}")

                summary_path = step_root / f"layer_{layer_idx:02d}" / "summary_overlay.png"
                attn.save_summary_figure(
                    summary_path,
                    layer_idx,
                    case.prompts,
                    view_images,
                    heatmaps_by_prompt,
                    args.overlay_alpha,
                )
                summary_paths_by_step_layer[step][layer_idx] = summary_path
                case_result["outputs"].append(
                    {
                        "step": step,
                        "layer": layer_idx,
                        "summary_overlay": str(summary_path),
                        "prompt_outputs": saved_prompts,
                    }
                )
                print(f"[suite] saved {bundle.name}/{case.name}/step_{step:04d}/layer_{layer_idx:02d}")

        grid_path = case_root / "step_layer_grid.png"
        compose_step_layer_grid(summary_paths_by_step_layer, case.steps, layers, grid_path)
        case_result["step_layer_grid"] = str(grid_path)
        bundle_result["cases"].append(case_result)
        print(f"[suite] grid {grid_path}")

    return bundle_result


def main() -> None:
    args = parse_args()
    layers = attn.parse_layer_spec(args.layers)
    keep_names = set(args.only or [])
    suite = filter_suite(build_default_suite(), keep_names)
    if not suite:
        raise ValueError(f"No suite entries matched --only={sorted(keep_names)}")

    model_config = attn.build_model_config()
    summary = {
        "layers": layers,
        "output_dir": str(args.output_dir),
        "runtime_root": str(attn.RUNTIME_ROOT),
        "suite": [],
    }

    for bundle in suite:
        summary["suite"].append(run_bundle(bundle, layers, args, model_config))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[suite] wrote summary {summary_path}")


if __name__ == "__main__":
    main()
