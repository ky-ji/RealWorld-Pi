#!/usr/bin/env python3
"""
在离线数据集上调试 OpenPI Pi0.5 推理效果：
  - 直接读取 LeRobot 格式数据集 (parquet + mp4)，无需安装 lerobot 包
  - 每隔 N 步取一帧进行推理
  - OpenPI 的 policy.infer() 返回绝对目标位置（已由 AbsoluteActions 转换）
  - 使用增量轴角动作空间: [x, y, z, ax, ay, az, gripper]

数据集目录结构：
  dataset_dir/
  ├── data/chunk-000/episode_000000.parquet   (state, actions 等结构化数据)
  ├── videos/chunk-000/front_view/episode_000000.mp4
  ├── videos/chunk-000/wrist_view/episode_000000.mp4
  └── meta/info.json

使用示例：
  CUDA_VISIBLE_DEVICES=3 /home/yinmenghao/code/openpi/.venv/bin/python \
   /home/yinmenghao/code/openpi/realworld_deploy/server/eval_on_dataset_openpi.py \
      --checkpoint_dir /home/yinmenghao/code/openpi/checkpoints/stack_bowls_lora_0208/pi05_stack_bowls_lora/stack_bowls_lora_v2/29999 \
      --config_name pi05_stack_bowls_lora \
      --dataset_dir /data1/vla-data/processed/PI/data/010_stack_bowls_0209 \
      --episode_id 50 -N 10

 CUDA_VISIBLE_DEVICES=3 /home/yinmenghao/code/openpi/.venv/bin/python \
   /home/yinmenghao/code/openpi/realworld_deploy/server/eval_on_dataset_openpi.py \
      --checkpoint_dir /home/yinmenghao/code/openpi/checkpoints/place_phone_lora_0211/pi05_place_phone_lora/place_phone_lora_v2/19999 \
      --config_name pi05_place_phone_lora \
      --dataset_dir /data1/vla-data/processed/PI/data/009_place_phone \
      --episode_id 155 -N 10

输出：
  - 一张包含 7 维 (xyz + axis-angle + gripper) 的对比曲线图
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# =============================================================================
# 路径设置: 确保 openpi 包可导入
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# =============================================================================
# 数据集加载工具 (直接读 parquet + mp4，无需 lerobot 包)
# =============================================================================

def load_episode_parquet(dataset_dir: str, episode_id: int) -> pd.DataFrame:
    """
    从 LeRobot 格式数据集中加载指定 episode 的 parquet 数据。

    Returns:
        DataFrame, 包含 state (np.array 7D), actions (np.array 7D), frame_index 等列
    """
    parquet_path = os.path.join(
        dataset_dir, "data", "chunk-000", f"episode_{episode_id:06d}.parquet"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    return df


class VideoFrameReader:
    """使用 cv2 按帧索引读取 mp4 视频帧"""

    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._path = video_path

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """读取指定帧，返回 RGB (H, W, 3) uint8"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"无法读取帧 {frame_idx} (共 {self.total_frames} 帧): {self._path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()

    def __del__(self):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Debug OpenPI Pi0.5 inference on LeRobot dataset")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="OpenPI checkpoint 目录路径")
    parser.add_argument("--config_name", type=str, default="pi05_stack_bowls_lora",
                        help="OpenPI 训练配置名称")
    parser.add_argument("--dataset_dir", type=str,
                        default="/data1/vla-data/processed/PI/data/010_stack_bowls_0209",
                        help="LeRobot 格式数据集目录 (包含 data/, videos/, meta/ 子目录)")
    parser.add_argument("--episode_id", type=int, default=0, help="episode 索引 (从 0 开始)")
    parser.add_argument("--step_stride", type=int, default=1,
                        help="每隔多少帧做一次推理（被 -N 覆盖）")
    parser.add_argument("--chunk_take", type=int, default=1,
                        help="每个 chunk 取前多少步动作用于对比（被 -N 覆盖）")
    parser.add_argument("-N", "--infer_steps", type=int, default=1,
                        help="每 N 步推理一次，取 chunk 前 N 个 action 做对比")
    parser.add_argument("--prompt", type=str, default="stack the bowls",
                        help="任务指令 prompt")
    parser.add_argument("--output_dir", type=str, default="./debug_eval_outputs_openpi",
                        help="可视化结果输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =============================================================
    # 加载数据集
    # =============================================================
    print(f"[Config] config_name={args.config_name}")
    print(f"[Config] checkpoint_dir={args.checkpoint_dir}")
    print(f"[Config] dataset_dir={args.dataset_dir}")
    print(f"[Config] episode_id={args.episode_id}")

    # 加载 parquet 数据
    df = load_episode_parquet(args.dataset_dir, args.episode_id)
    total_steps = len(df)
    print(f"[Dataset] Episode {args.episode_id}: {total_steps} 帧")

    if total_steps < 2:
        raise ValueError(f"Episode 太短，无法推理: {total_steps} 帧")

    # 打开视频流（按需读帧，避免一次性全部解码）
    front_video_path = os.path.join(
        args.dataset_dir, "videos", "chunk-000", "front_view",
        f"episode_{args.episode_id:06d}.mp4"
    )
    wrist_video_path = os.path.join(
        args.dataset_dir, "videos", "chunk-000", "wrist_view",
        f"episode_{args.episode_id:06d}.mp4"
    )
    front_reader = VideoFrameReader(front_video_path)
    wrist_reader = VideoFrameReader(wrist_video_path)
    print(f"[Dataset] 视频: front={front_reader.total_frames} 帧, wrist={wrist_reader.total_frames} 帧")

    # =============================================================
    # 提取 GT: state 和 action
    # =============================================================
    states = np.array(df["state"].tolist(), dtype=np.float32)      # (N, 7)
    actions_gt = np.array(df["actions"].tolist(), dtype=np.float32)  # (N, 7)

    # parquet 中的 actions 是绝对目标位置（DeltaActions 转换在训练 pipeline 中进行）
    # 模型推理输出经 AbsoluteActions 逆转换后也是绝对目标位置
    # 因此 GT 直接使用 parquet 中的 actions
    gt_state = np.array(actions_gt, copy=True)

    # 预测结果: 用 NaN 预填充
    pred_state = np.full_like(gt_state, fill_value=np.nan)

    # =============================================================
    # 初始化 OpenPI 推理策略
    # =============================================================
    from openpi.training import config as _config
    from openpi.policies import policy_config

    train_config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.prompt,
    )
    print("[Policy] ✓ OpenPI 策略加载成功")

    # 预热
    dummy_obs = {
        "observation/front_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": args.prompt,
    }
    _ = policy.infer(dummy_obs)
    print("[Policy] ✓ 预热完成")

    # =============================================================
    # 逐帧推理
    # =============================================================
    N = args.infer_steps
    infer_stride = N if N > 1 else args.step_stride
    chunk_take = N if N > 1 else args.chunk_take
    chunk_records = []
    infer_points = []

    print(f"[Config] infer_steps(N)={N}, infer_stride={infer_stride}, "
          f"chunk_take={chunk_take}, total_steps={total_steps}")

    for t in range(0, total_steps - 1, infer_stride):
        # 读取视频帧 (RGB uint8)
        front_image = front_reader.read_frame(t)
        wrist_image = wrist_reader.read_frame(t)

        # 提取 state (7D 轴角)
        state7 = states[t]

        # 构建 OpenPI 观测字典
        obs = {
            "observation/front_image": front_image,
            "observation/wrist_image": wrist_image,
            "observation/state": state7,
            "prompt": args.prompt,
        }

        # 推理
        result = policy.infer(obs)
        pred_chunk = result["actions"]  # (action_horizon, 7) 绝对目标位置

        # 取前 N 个 action 填充对应位置
        fill_len = min(chunk_take, len(pred_chunk), total_steps - t)
        pred_state[t: t + fill_len] = pred_chunk[:fill_len]
        infer_points.append(t)

        chunk_records.append({
            "start_step": int(t),
            "fill_len": int(fill_len),
            "prompt": args.prompt,
            "state": state7.tolist(),
            "pred_chunk": pred_chunk[:chunk_take].tolist(),
        })

        if t % 50 == 0:
            print(f"[Progress] Step {t}/{total_steps}")

    # 关闭视频流
    front_reader.close()
    wrist_reader.close()

    print(f"[Done] {len(infer_points)} 次推理完成")

    # =============================================================
    # 保存 JSON 结果
    # =============================================================
    json_output_path = os.path.join(
        args.output_dir, f"episode_{args.episode_id}_{timestamp}.json"
    )
    result_json = {
        "episode_id": args.episode_id,
        "dataset_dir": args.dataset_dir,
        "config_name": args.config_name,
        "checkpoint_dir": args.checkpoint_dir,
        "action_format": "7D_axis_angle",
        "action_type": "delta (AbsoluteActions auto-converted)",
        "infer_steps": N,
        "infer_stride": infer_stride,
        "chunk_take": chunk_take,
        "num_inferences": len(infer_points),
        "prompt": args.prompt,
        "timestamp": timestamp,
        "gt_state": gt_state.tolist(),
        "pred_state": pred_state.tolist(),
        "infer_points": infer_points,
        "chunk_records": chunk_records,
    }
    with open(json_output_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"[OK] JSON saved to: {json_output_path}")

    # =============================================================
    # 可视化对比
    # =============================================================
    dim_names = ["x", "y", "z", "ax", "ay", "az", "gripper"]
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)

    title_line2 = (
        f"N={N}, num_inferences={len(infer_points)}"
        if N > 1
        else f"stride={infer_stride}, chunk_take={chunk_take}"
    )
    fig.suptitle(
        f"OpenPI Pi0.5 Inference vs GT (episode {args.episode_id})\n"
        f"action_type=delta (axis-angle), {title_line2}",
        fontsize=13,
    )

    x_axis = np.arange(total_steps)
    for i, ax in enumerate(axes):
        if N > 1:
            for ip in infer_points:
                ax.axvline(x=ip, color="red", linestyle=":", linewidth=0.6, alpha=0.4)

        ax.plot(x_axis, gt_state[:, i], label="GT", color="tab:blue", linewidth=1.2)
        ax.plot(x_axis, pred_state[:, i], label="Pred", color="tab:orange", linewidth=1.2)
        ax.set_ylabel(dim_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            if N > 1:
                from matplotlib.lines import Line2D
                ax.legend(
                    handles=[
                        Line2D([], [], color="tab:blue", linewidth=1.2, label="GT"),
                        Line2D([], [], color="tab:orange", linewidth=1.2, label="Pred"),
                        Line2D([], [], color="red", linestyle=":", linewidth=1, label=f"Infer (N={N})"),
                    ],
                    loc="upper right",
                )
            else:
                ax.legend(loc="upper right")

    axes[-1].set_xlabel("Step")

    output_path = os.path.join(
        args.output_dir, f"episode_{args.episode_id}_{timestamp}.png"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    print(f"[OK] Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
