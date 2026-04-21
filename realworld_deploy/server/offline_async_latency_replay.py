#!/usr/bin/env python3
"""
离线回放 + latency injection 工具。

用途：
  - 读取 LeRobot episode（parquet + mp4）
  - 用录制观测重放 Pi0.5 policy
  - 在 chunk 级别注入固定/可变 delay steps
  - 比较 naive async 和 RTC 的 chunk splice discontinuity

示例：
  REPO_ROOT=/path/to/RealWorld-Pi
  CUDA_VISIBLE_DEVICES=3 "${REPO_ROOT}/.venv/bin/python" \
      "${REPO_ROOT}/realworld_deploy/server/offline_async_latency_replay.py" \
      --checkpoint_dir "${REPO_ROOT}/checkpoints/stack_bowls_lora_0208/pi05_stack_bowls_lora/stack_bowls_lora_v2/29999" \
      --config_name pi05_stack_bowls_lora \
      --dataset_dir /data1/vla-data/processed/PI/data/010_stack_bowls_0209 \
      --episode_id 50 \
      --mode both \
      --delay_steps 2 \
      --execute_horizon 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
CLIENT_PKG_DIR = REPO_ROOT / "packages" / "openpi-client" / "src"
ROBOT_INFERENCE_DIR = REPO_ROOT / "realworld_deploy" / "robot_inference"
for path in (CURRENT_DIR, SRC_DIR, CLIENT_PKG_DIR, ROBOT_INFERENCE_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from async_chunk_scheduler import AsyncActionChunkScheduler


def load_episode_parquet(dataset_dir: str, episode_id: int) -> pd.DataFrame:
    parquet_path = Path(dataset_dir) / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")
    return pd.read_parquet(parquet_path)


class VideoFrameReader:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.path = video_path

    def read_frame(self, frame_idx: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"无法读取视频帧 {frame_idx}: {self.path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()

    def __del__(self):
        self.close()


@dataclass
class PendingOfflineResponse:
    obs_seq: int
    available_step: int
    action_chunk: np.ndarray
    mode: str


def build_policy(config_name: str, checkpoint_dir: str, prompt: str):
    from openpi.policies import policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=prompt,
    )
    dummy_obs = {
        "observation/front_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": prompt,
    }
    _ = policy.infer(dummy_obs)
    return policy


def parse_delay_schedule(delay_steps: str, max_cycles: int) -> list[int]:
    pieces = [piece.strip() for piece in str(delay_steps).split(",") if piece.strip()]
    if not pieces:
        return [0] * max_cycles
    values = [max(0, int(piece)) for piece in pieces]
    if len(values) >= max_cycles:
        return values[:max_cycles]
    return values + [values[-1]] * (max_cycles - len(values))


def make_observation(
    states: np.ndarray,
    front_reader: VideoFrameReader,
    wrist_reader: VideoFrameReader,
    step_idx: int,
    prompt: str,
) -> dict:
    return {
        "observation/front_image": front_reader.read_frame(step_idx),
        "observation/wrist_image": wrist_reader.read_frame(step_idx),
        "observation/state": np.asarray(states[step_idx], dtype=np.float32),
        "prompt": prompt,
    }


def simulate_async_execution(
    *,
    policy,
    states: np.ndarray,
    front_reader: VideoFrameReader,
    wrist_reader: VideoFrameReader,
    prompt: str,
    execute_horizon: int,
    delay_schedule: list[int],
    mode: str,
) -> dict:
    if mode not in ("naive_async", "rtc"):
        raise ValueError(f"unsupported mode: {mode}")

    total_steps = len(states)
    initial_obs = make_observation(states, front_reader, wrist_reader, 0, prompt)
    initial_chunk = np.asarray(policy.infer(initial_obs)["actions"], dtype=np.float32)
    if initial_chunk.ndim == 1:
        initial_chunk = initial_chunk.reshape(1, -1)

    scheduler = AsyncActionChunkScheduler(
        action_horizon=int(initial_chunk.shape[0]),
        action_dim=int(initial_chunk.shape[1]),
        execute_horizon=int(execute_horizon),
        dt_exec=1.0,
    )
    scheduler.set_initial_chunk(initial_chunk, obs_seq=0, chunk_id=0)

    executed_actions = []
    boundary_discontinuities = []
    cycle_records = []
    pending_responses: list[PendingOfflineResponse] = []
    current_step = 0
    cycle_id = 0

    while current_step < total_steps:
        if current_step % execute_horizon == 0:
            obs_step = min(current_step, total_steps - 1)
            obs = make_observation(states, front_reader, wrist_reader, obs_step, prompt)
            delay_steps = delay_schedule[min(cycle_id, len(delay_schedule) - 1)]

            if mode == "naive_async":
                next_chunk = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
            else:
                prefix_abs = scheduler.current_chunk()
                prefix_delta = prefix_abs.copy()
                prefix_delta[:, :6] -= np.asarray(states[obs_step], dtype=np.float32)[:6]
                next_chunk = np.asarray(
                    policy.infer_realtime_chunking(
                        obs,
                        prefix_actions=prefix_delta,
                        inference_delay=int(delay_steps),
                        prefix_attention_horizon=int(prefix_abs.shape[0] - execute_horizon),
                        prefix_attention_schedule="exp",
                        max_guidance_weight=5.0,
                    )["actions"],
                    dtype=np.float32,
                )

            scheduler.register_request(obs_seq=cycle_id + 1, send_timestamp_ns=current_step)
            pending_responses.append(
                PendingOfflineResponse(
                    obs_seq=cycle_id + 1,
                    available_step=current_step + delay_steps,
                    action_chunk=next_chunk,
                    mode=mode,
                )
            )
            cycle_records.append(
                {
                    "cycle_id": cycle_id,
                    "send_step": current_step,
                    "delay_steps": delay_steps,
                    "mode": mode,
                }
            )
            cycle_id += 1

        ready = [resp for resp in pending_responses if resp.available_step <= current_step]
        pending_responses = [resp for resp in pending_responses if resp.available_step > current_step]
        for response in sorted(ready, key=lambda item: (item.available_step, item.obs_seq)):
            scheduler.integrate_response(
                obs_seq=response.obs_seq,
                chunk=response.action_chunk,
                recv_timestamp_ns=response.available_step,
                chunk_id=response.obs_seq,
            )

        current_action = scheduler.current_action()
        if executed_actions:
            prev_action = executed_actions[-1]
            if current_step % execute_horizon == 0:
                boundary_discontinuities.append(
                    {
                        "step": current_step,
                        "l2_all": float(np.linalg.norm(current_action - prev_action)),
                        "l2_pose": float(np.linalg.norm(current_action[:6] - prev_action[:6])),
                    }
                )

        executed_actions.append(current_action.copy())
        scheduler.advance(1)
        current_step += 1

    boundary_pose_values = [item["l2_pose"] for item in boundary_discontinuities]
    boundary_all_values = [item["l2_all"] for item in boundary_discontinuities]
    return {
        "mode": mode,
        "execute_horizon": execute_horizon,
        "delay_schedule_steps": delay_schedule,
        "avg_boundary_l2_pose": None if not boundary_pose_values else float(np.mean(boundary_pose_values)),
        "max_boundary_l2_pose": None if not boundary_pose_values else float(np.max(boundary_pose_values)),
        "avg_boundary_l2_all": None if not boundary_all_values else float(np.mean(boundary_all_values)),
        "max_boundary_l2_all": None if not boundary_all_values else float(np.max(boundary_all_values)),
        "boundary_discontinuities": boundary_discontinuities,
        "executed_actions": np.asarray(executed_actions, dtype=np.float32).tolist(),
        "cycle_records": cycle_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline async replay + latency injection for Pi0.5")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, default="pi05_stack_bowls_lora")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="stack the bowls")
    parser.add_argument("--execute_horizon", type=int, default=4)
    parser.add_argument("--delay_steps", type=str, default="2", help="单个整数或逗号分隔的 per-cycle delay steps")
    parser.add_argument("--mode", type=str, choices=["naive_async", "rtc", "both"], default="both")
    parser.add_argument("--output_dir", type=str, default="./offline_async_latency_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_episode_parquet(args.dataset_dir, args.episode_id)
    states = np.asarray(df["state"].tolist(), dtype=np.float32)
    total_steps = len(states)
    if total_steps < 2:
        raise ValueError(f"Episode 太短，无法回放: {total_steps}")

    front_video_path = Path(args.dataset_dir) / "videos" / "chunk-000" / "front_view" / f"episode_{args.episode_id:06d}.mp4"
    wrist_video_path = Path(args.dataset_dir) / "videos" / "chunk-000" / "wrist_view" / f"episode_{args.episode_id:06d}.mp4"
    front_reader = VideoFrameReader(str(front_video_path))
    wrist_reader = VideoFrameReader(str(wrist_video_path))

    try:
        policy = build_policy(args.config_name, args.checkpoint_dir, args.prompt)
        max_cycles = max(1, math.ceil(total_steps / args.execute_horizon))
        delay_schedule = parse_delay_schedule(args.delay_steps, max_cycles)

        modes = ["naive_async", "rtc"] if args.mode == "both" else [args.mode]
        results = {}
        for mode in modes:
            print(f"[Replay] mode={mode}, execute_horizon={args.execute_horizon}, delay_schedule={delay_schedule}")
            results[mode] = simulate_async_execution(
                policy=policy,
                states=states,
                front_reader=front_reader,
                wrist_reader=wrist_reader,
                prompt=args.prompt,
                execute_horizon=args.execute_horizon,
                delay_schedule=delay_schedule,
                mode=mode,
            )

        summary = {
            "config_name": args.config_name,
            "checkpoint_dir": args.checkpoint_dir,
            "dataset_dir": args.dataset_dir,
            "episode_id": args.episode_id,
            "prompt": args.prompt,
            "execute_horizon": args.execute_horizon,
            "delay_schedule_steps": delay_schedule,
            "results": results,
        }
        output_path = Path(args.output_dir) / f"offline_async_latency_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Replay] 结果已保存: {output_path}")

        if "naive_async" in results and "rtc" in results:
            print(
                "[Replay] boundary l2 pose: "
                f"naive={results['naive_async']['avg_boundary_l2_pose']} "
                f"rtc={results['rtc']['avg_boundary_l2_pose']}"
            )
    finally:
        front_reader.close()
        wrist_reader.close()


if __name__ == "__main__":
    main()
