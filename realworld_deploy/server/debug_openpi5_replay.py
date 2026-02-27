#!/usr/bin/env python3
"""
调试脚本：OpenPI Pi0.5 轨迹重放 (Trajectory Replay)

用途：
1. 读取 OpenPI 训练数据集 (Parquet) 中的 action (7D 轴角格式)
2. 将 7D 轴角格式转换为 8D 四元数格式 (XYZW 标准顺序) 发送给客户端
3. 模拟服务器，将 action chunk 按固定频率发送给客户端
4. 验证真实数据集回放链路与坐标转换

数据格式说明：
  - Parquet 中的数据: 7D 轴角 [x, y, z, ax, ay, az, gripper]
  - 客户端期望: 8D XYZW [x, y, z, qx, qy, qz, qw, gripper]
  - 四元数全程使用标准 XYZW 顺序
"""

import argparse
import json
import socket
import select
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from server_config_openpi_pi05 import (
    SERVER_IP,
    SERVER_PORT,
    SOCKET_TIMEOUT,
    BUFFER_SIZE,
    ENCODING,
    MAX_CLIENTS,
)


# 数据集根目录（用户指定）
DATASET_ROOT = "/data1/vla-data/processed/PI/data/009_place_phone"

# 回放参数
REPLAY_CHUNK_SIZE = 10
DEFAULT_EPISODE_INDEX = 0
DEFAULT_FPS = 20.0


def axisangle_to_quaternion_xyzw(axisangle):
    """轴角 [ax, ay, az] -> 四元数 [qx, qy, qz, qw] (XYZW)."""
    r = R.from_rotvec(axisangle)
    return r.as_quat()


def quaternion_xyzw_to_axisangle(quat_xyzw):
    """四元数 [qx, qy, qz, qw] (XYZW) -> 轴角 [ax, ay, az]."""
    r = R.from_quat(quat_xyzw)
    return r.as_rotvec()


def state7_axisangle_to_pose8_xyzw(state7):
    """
    7D 轴角格式 -> 8D 四元数格式
    state7: [x, y, z, ax, ay, az, gripper]
    pose8 : [x, y, z, qx, qy, qz, qw, gripper]
    """
    pos = state7[:3]
    axisangle = state7[3:6]
    gripper = state7[6]
    quat_xyzw = axisangle_to_quaternion_xyzw(axisangle)
    return np.concatenate([pos, quat_xyzw, [gripper]]).astype(np.float32)


def pose8_xyzw_to_state7_axisangle(pose8):
    """
    8D 四元数格式 -> 7D 轴角格式
    pose8 : [x, y, z, qx, qy, qz, qw, gripper]
    state7: [x, y, z, ax, ay, az, gripper]
    """
    pos = pose8[:3]
    quat_xyzw = pose8[3:7]
    gripper = pose8[7]
    axisangle = quaternion_xyzw_to_axisangle(quat_xyzw)
    return np.concatenate([pos, axisangle, [gripper]]).astype(np.float32)


def _resolve_episode_path(dataset_root: Path, episode_index: int) -> Path:
    """根据 episode_index 解析 parquet 路径。"""
    episode_chunk = episode_index // 1000
    return dataset_root / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _load_dataset_fps(dataset_root: Path) -> float:
    """从 meta/info.json 读取 fps，失败时返回默认值。"""
    info_path = dataset_root / "meta" / "info.json"
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        fps = float(info.get("fps", DEFAULT_FPS))
        return fps if fps > 0 else DEFAULT_FPS
    except Exception:
        return DEFAULT_FPS


def load_parquet_trajectory(dataset_root: Path, episode_index: int):
    """
    加载单条 episode 的 action 轨迹并转换为 8D XYZW。

    Returns:
        poses_8d: List[np.ndarray], 每个元素 shape=(8,)
    """
    episode_path = _resolve_episode_path(dataset_root, episode_index)
    print(f"[调试] 加载 Episode: {episode_index}")
    print(f"[调试] Parquet 路径: {episode_path}")

    if not episode_path.exists():
        print(f"[错误] 文件不存在: {episode_path}")
        return []

    try:
        df = pd.read_parquet(episode_path)
        if "actions" not in df.columns:
            print("[错误] Parquet 缺少 actions 列")
            return []

        poses_8d = []
        for action in df["actions"].values:
            action7 = np.asarray(action, dtype=np.float32).reshape(-1)
            if action7.shape[0] < 7:
                continue
            pose8 = state7_axisangle_to_pose8_xyzw(action7[:7])
            poses_8d.append(pose8)

        print(f"[调试] 加载了 {len(poses_8d)} 步轨迹")

        if len(poses_8d) > 0:
            first_action7 = np.asarray(df["actions"].iloc[0], dtype=np.float32).reshape(-1)
            first_pose8 = poses_8d[0]
            print("[调试] 第一步数据:")
            print(f"  7D Axis-Angle (原始): {first_action7[:7]}")
            print(f"    位置 [x, y, z]: {first_action7[:3]}")
            print(f"    轴角 [ax, ay, az]: {first_action7[3:6]}")
            print(f"  8D XYZW (转换): {first_pose8}")
            print(f"    四元数 [qx, qy, qz, qw]: {first_pose8[3:7]}")

        return poses_8d
    except Exception as e:
        print(f"[错误] 读取 parquet 失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_debug_server(dataset_root: str, episode_index: int, replay_chunk_size: int):
    dataset_root = Path(dataset_root)
    fps = _load_dataset_fps(dataset_root)
    send_interval_s = replay_chunk_size / fps

    poses_8d = load_parquet_trajectory(dataset_root, episode_index)
    if not poses_8d:
        return

    target_start_pose8 = poses_8d[0]
    rotation_offset = None
    replay_step_idx = 0

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(MAX_CLIENTS)

        print(f"\n[调试服务器] 正在监听 {SERVER_IP}:{SERVER_PORT}")
        print("[调试服务器] 数据格式: 7D Axis-Angle -> 8D XYZW")
        print(f"[调试服务器] 数据集: {dataset_root}")
        print(f"[调试服务器] Episode: {episode_index}, chunk_size={replay_chunk_size}, fps={fps}")
        print("[调试服务器] 请启动客户端 (inference_client.py) ...")

        client_socket, client_addr = server_socket.accept()
        print(f"[调试服务器] 客户端已连接: {client_addr}")
        client_socket.settimeout(SOCKET_TIMEOUT)

        print("[调试服务器] 等待 2 秒...")
        time.sleep(2.0)

        STATE_WAIT_OBS = 0
        STATE_REPLAY = 1
        current_state = STATE_WAIT_OBS

        buffer = b""
        running = True

        while running:
            readable, _, _ = select.select([client_socket], [], [], 0.0)

            if readable:
                try:
                    data_received = client_socket.recv(BUFFER_SIZE)
                    if not data_received:
                        print("[调试服务器] 客户端断开连接")
                        break
                    buffer += data_received

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        msg = line.decode(ENCODING).strip()
                        if not msg:
                            continue

                        req = json.loads(msg)
                        msg_type = req.get("type")

                        if msg_type == "reset":
                            print("[调试服务器] 收到 Reset 请求")
                            client_socket.sendall((json.dumps({"type": "reset_ack"}) + "\n").encode(ENCODING))
                            current_state = STATE_WAIT_OBS
                            replay_step_idx = 0
                            rotation_offset = None

                        elif msg_type == "observation":
                            poses = req.get("poses", [])
                            if not poses:
                                continue

                            # poses 最后一帧: [x, y, z, qx, qy, qz, qw]
                            current_pose7 = np.asarray(poses[-1], dtype=np.float32).reshape(-1)
                            if current_pose7.shape[0] < 7:
                                continue

                            grippers = req.get("grippers", [1.0])
                            val = grippers[-1] if isinstance(grippers, list) else 1.0
                            if isinstance(val, (list, np.ndarray)):
                                val = val[0] if len(val) > 0 else 1.0
                            current_gripper_val = float(val)

                            current_pose8 = np.concatenate([current_pose7[:7], [current_gripper_val]])

                            if current_state == STATE_WAIT_OBS and rotation_offset is None:
                                cur_q_xyzw = current_pose8[3:7]
                                tgt_q_xyzw = target_start_pose8[3:7]

                                r_cur = R.from_quat(cur_q_xyzw)
                                r_tgt = R.from_quat(tgt_q_xyzw)
                                r_offset = r_cur * r_tgt.inv()
                                rotation_offset = r_offset.as_quat()

                                euler_deg = r_offset.as_euler("xyz", degrees=True)
                                print(f"[调试] Offset Euler (deg): {euler_deg}")
                                print("[调试] 开始回放 (Streaming 模式)...")
                                current_state = STATE_REPLAY
                                replay_step_idx = 0
                except Exception as e:
                    print(f"[错误] 数据处理异常: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            if current_state == STATE_REPLAY:
                try:
                    if replay_step_idx < len(poses_8d):
                        end_idx = min(replay_step_idx + replay_chunk_size, len(poses_8d))
                        chunk_poses = poses_8d[replay_step_idx:end_idx]

                        if len(chunk_poses) < replay_chunk_size:
                            last_pose = chunk_poses[-1]
                            chunk_poses = chunk_poses + [last_pose] * (replay_chunk_size - len(chunk_poses))

                        chunk8 = np.asarray(chunk_poses, dtype=np.float32)

                        # 应用初始姿态 offset，使轨迹与当前机器人初始姿态对齐
                        if rotation_offset is not None:
                            r_offset = R.from_quat(rotation_offset)
                            for i in range(len(chunk8)):
                                q_xyzw = chunk8[i, 3:7]
                                r = R.from_quat(q_xyzw)
                                chunk8[i, 3:7] = (r_offset * r).as_quat()

                        response = {"type": "action_sequence", "actions": chunk8.tolist()}
                        client_socket.sendall((json.dumps(response) + "\n").encode(ENCODING))

                        if replay_step_idx % 100 == 0:
                            print(f"Streaming Replay {replay_step_idx}/{len(poses_8d)}")

                        replay_step_idx += replay_chunk_size
                        time.sleep(send_interval_s)
                    else:
                        print("Replay finished, looping...")
                        replay_step_idx = 0
                        time.sleep(send_interval_s)
                except Exception as e:
                    print(f"[错误] 发送异常: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                if not readable:
                    time.sleep(0.01)
    except Exception as e:
        print(f"[严重错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if "client_socket" in locals():
            client_socket.close()
        if "server_socket" in locals():
            server_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenPI Pi0.5 数据集回放调试服务器")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DATASET_ROOT,
        help="数据集根目录 (包含 data/ 和 meta/)",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=DEFAULT_EPISODE_INDEX,
        help="回放的 episode index",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=REPLAY_CHUNK_SIZE,
        help="每次发送的 action chunk 大小",
    )
    args = parser.parse_args()

    run_debug_server(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        replay_chunk_size=args.chunk_size,
    )
