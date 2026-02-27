#!/usr/bin/env python3
"""
单步推理调试脚本：
  - 读取 test_jsons/ 下的 JSON 文件（包含 state7_axisangle, prompt, front_view, wrist_view）
  - 加载 OpenPI checkpoint，执行一次推理，输出 8 个 action 的 chunk
  - 同时记录 action7_axisangle（模型原始输出）和 action8_xyzw（四元数转换后）
  - 结果写回原 JSON 文件

使用示例：
  CUDA_VISIBLE_DEVICES=3 /home/yinmenghao/code/openpi/.venv/bin/python \
      /home/yinmenghao/code/openpi/realworld_deploy/server/debug/single_step_inference.py \
      --json_path /home/yinmenghao/code/openpi/realworld_deploy/server/debug/test_jsons/run_20260226_135534.json \
      --checkpoint_dir /home/yinmenghao/code/openpi/checkpoints/place_phone_lora_0211/pi05_place_phone_lora/place_phone_lora_v3/15000 \
      --config_name pi05_place_phone_lora
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import cv2

# =============================================================================
# 路径设置: 确保 openpi 包可导入
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.spatial.transform import Rotation as R


# =============================================================================
# 坐标转换工具 (与 inference_server_openpi_pi05.py 保持一致)
# =============================================================================

def axisangle_to_quaternion_xyzw(axisangle):
    """轴角 [ax, ay, az] -> 四元数 [qx, qy, qz, qw]"""
    r = R.from_rotvec(axisangle)
    return r.as_quat()  # scipy 默认 xyzw


def action7_axisangle_to_pose8_xyzw(action7_aa):
    """
    7D 轴角 [x, y, z, ax, ay, az, gripper] -> 8D 四元数 [x, y, z, qx, qy, qz, qw, gripper]
    """
    pos = action7_aa[:3]
    axisangle = action7_aa[3:6]
    gripper = action7_aa[6]
    quat_xyzw = axisangle_to_quaternion_xyzw(axisangle)
    return np.concatenate([pos, quat_xyzw, [gripper]]).astype(np.float32)


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Single-step OpenPI inference for debugging")
    parser.add_argument("--json_path", type=str, required=True,
                        help="输入 JSON 文件路径 (包含 state7_axisangle, prompt, front_view, wrist_view)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="OpenPI checkpoint 目录路径")
    parser.add_argument("--config_name", type=str, required=True,
                        help="OpenPI 训练配置名称 (如 pi05_place_phone_lora)")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="从 action chunk 中取前多少步 (默认 8)")
    args = parser.parse_args()

    # ----- 读取输入 JSON -----
    print(f"[Input] 读取: {args.json_path}")
    with open(args.json_path, "r") as f:
        data = json.load(f)

    inp = data["input"]
    state7 = np.array(inp["state7_axisangle"], dtype=np.float32)
    prompt = inp["prompt"]
    front_view_path = inp["front_view"]
    wrist_view_path = inp["wrist_view"]

    print(f"[Input] prompt: {prompt}")
    print(f"[Input] state7_axisangle: {state7.tolist()}")
    print(f"[Input] front_view: {front_view_path}")
    print(f"[Input] wrist_view: {wrist_view_path}")

    # ----- 读取图像 -----
    front_image = cv2.imread(front_view_path)
    if front_image is None:
        raise FileNotFoundError(f"无法读取 front_view 图像: {front_view_path}")
    front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)

    wrist_image = cv2.imread(wrist_view_path)
    if wrist_image is None:
        raise FileNotFoundError(f"无法读取 wrist_view 图像: {wrist_view_path}")
    wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)

    print(f"[Input] front_image shape: {front_image.shape}, wrist_image shape: {wrist_image.shape}")

    # ----- 加载模型 -----
    print(f"[Model] 加载中... config={args.config_name}, checkpoint={args.checkpoint_dir}")
    from openpi.training import config as _config
    from openpi.policies import policy_config

    train_config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=prompt,
    )
    print("[Model] ✓ 策略加载成功")

    # ----- 预热 -----
    dummy_obs = {
        "observation/front_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": prompt,
    }
    _ = policy.infer(dummy_obs)
    print("[Model] ✓ 预热完成")

    # ----- 推理 -----
    obs = {
        "observation/front_image": front_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state7,
        "prompt": prompt,
    }

    print("[Infer] 开始推理...")
    infer_start = time.time()
    result = policy.infer(obs)
    infer_end = time.time()
    inference_ms = (infer_end - infer_start) * 1000

    action_chunk_7d = result["actions"]  # (action_horizon, 7) 绝对目标位置
    print(f"[Infer] ✓ 推理完成, 耗时 {inference_ms:.1f}ms, 原始 chunk shape: {action_chunk_7d.shape}")

    # 截断到 chunk_size
    if len(action_chunk_7d) > args.chunk_size:
        action_chunk_7d = action_chunk_7d[:args.chunk_size]
    print(f"[Infer] 截断后 chunk size: {len(action_chunk_7d)}")

    # ----- 转换 action7_axisangle -> action8_xyzw -----
    action_chunk_8d = []
    prev_quat = None
    for i in range(len(action_chunk_7d)):
        pose8 = action7_axisangle_to_pose8_xyzw(action_chunk_7d[i])
        curr_quat = pose8[3:7]
        # 四元数符号一致性：确保相邻四元数走最短路径
        if prev_quat is not None:
            if np.dot(prev_quat, curr_quat) < 0:
                pose8[3:7] = -curr_quat
                curr_quat = -curr_quat
        action_chunk_8d.append(pose8)
        prev_quat = curr_quat

    action_chunk_8d = np.array(action_chunk_8d)

    # 确保第一帧四元数与当前 state 在同一半球
    if len(action_chunk_8d) > 0:
        current_quat = axisangle_to_quaternion_xyzw(state7[3:6])
        first_action_quat = action_chunk_8d[0, 3:7]
        if np.dot(current_quat, first_action_quat) < 0:
            action_chunk_8d[:, 3:7] = -action_chunk_8d[:, 3:7]

    # ----- 打印结果 -----
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)
    for i in range(len(action_chunk_7d)):
        print(f"\n  [Action {i}]")
        print(f"    action7_axisangle: {action_chunk_7d[i].tolist()}")
        print(f"    action8_xyzw:      {action_chunk_8d[i].tolist()}")

    # ----- 写回 JSON -----
    data["output"] = {
        "action7_axisangle": action_chunk_7d.tolist(),
        "action8_xyzw": action_chunk_8d.tolist(),
        "chunk_size": len(action_chunk_7d),
        "action_dim_7d": 7,
        "action_dim_8d": 8,
    }
    data["meta"] = {
        "config_name": args.config_name,
        "checkpoint_dir": args.checkpoint_dir,
        "inference_latency_ms": round(inference_ms, 2),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    with open(args.json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[Output] ✓ 结果已写回: {args.json_path}")


if __name__ == "__main__":
    main()
