#!/usr/bin/env python3
"""
OpenPI Pi0.5 推理服务器

使用 OpenPI 仓库的标准推理管线 (policy_config.create_trained_policy)，
仅支持增量 (delta) 模式，动作空间为 7D 轴角格式 [x, y, z, ax, ay, az, gripper]。

与 RealWorld-DP 的 inference_client.py 兼容：
- 协议：TCP socket + JSON Lines
- 消息类型：reset / observation → action

支持 VLA-Lab 统一日志格式（可选启用）

数据格式：
  - OpenPI 模型使用 7D 轴角格式: [x, y, z, ax, ay, az, gripper]
  - 客户端发送 8D 四元数格式: [x, y, z, qx, qy, qz, qw] + [gripper] (XYZW 标准顺序)
  - 服务器内部进行 四元数 XYZW ↔ 轴角 的自动转换

推理流程：
  1. 接收客户端的图像和 8D 四元数状态
  2. 转换为 7D 轴角格式 (axis-angle)
  3. 调用 OpenPI policy.infer() 获得 action chunk（绝对目标位置，已由 AbsoluteActions 转换）
  4. 截断到 TARGET_CHUNK_SIZE
  5. 可选: 动作放大、安全限制、平滑
  6. 转换回 8D 四元数格式发送给客户端

使用示例（需要使用 openpi 仓库的 uv 虚拟环境）：
  # 使用默认配置启动
  CUDA_VISIBLE_DEVICES=4 /home/yinmenghao/code/openpi/.venv/bin/python \
      /home/yinmenghao/code/openpi/realworld_deploy/server/inference_server_openpi_pi05.py

  # 或使用 uv run 启动
  cd /home/yinmenghao/code/openpi && CUDA_VISIBLE_DEVICES=3 uv run \
      realworld_deploy/server/inference_server_openpi_pi05.py

  # 指定参数
  CUDA_VISIBLE_DEVICES=3 /home/yinmenghao/code/openpi/.venv/bin/python \
      /home/yinmenghao/code/openpi/realworld_deploy/server/inference_server_openpi_pi05.py \
      --checkpoint_dir /home/yinmenghao/code/openpi/checkpoints/stack_bowls_lora_0208/pi05_stack_bowls_lora/stack_bowls_lora_v1/19999 \
      --config_name pi05_stack_bowls_lora \
      --prompt "stack the bowls" \
      --port 8007
"""

import socket
import json
import numpy as np
import cv2
import base64
import time
import os
import sys
from pathlib import Path
from datetime import datetime

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

from server_config_openpi_pi05 import (
    SERVER_IP, SERVER_PORT, CONFIG_NAME, CHECKPOINT_DIR, DEVICE,
    NUM_IMAGES, ACTION_DIM, TASK_PROMPT,
    INFERENCE_FREQ, TARGET_CHUNK_SIZE,
    ACTION_AMPLIFY_POS, ACTION_AMPLIFY_ROT,
    ENABLE_ACTION_LIMIT, MAX_POS_DELTA_PER_STEP, MAX_ROT_DELTA_PER_STEP,
    MAX_POS_ACCELERATION, MAX_ROT_ACCELERATION,
    SMOOTH_START_STEPS, ACTION_SMOOTHING_ALPHA, SAFE_WORKSPACE,
    GRIPPER_CHUNK_CONSISTENCY, GRIPPER_THRESHOLD,
    SOCKET_TIMEOUT, BUFFER_SIZE, ENCODING, MAX_CLIENTS, VERBOSE,
)

from scipy.spatial.transform import Rotation as R


# =============================================================================
# 坐标转换工具函数 (四元数 XYZW ↔ 轴角 axis-angle)
# =============================================================================

def quaternion_xyzw_to_axisangle(quat_xyzw):
    """
    将四元数 [qx, qy, qz, qw] (XYZW 标准顺序) 转换为轴角 [ax, ay, az]

    Args:
        quat_xyzw: 四元数 [qx, qy, qz, qw]

    Returns:
        轴角向量 [ax, ay, az] (弧度)
    """
    r = R.from_quat(quat_xyzw)
    return r.as_rotvec()


def canonicalize_quaternion_xyzw(quat_xyzw, prev_quat_xyzw=None):
    """
    规范化四元数半球，避免 q 与 -q 抖动导致的表示不连续。

    规则：
      1) 若有上一帧四元数，优先与上一帧同半球（dot >= 0）
      2) 首帧无参考时，强制 qw >= 0，保证确定性
    """
    q = np.asarray(quat_xyzw, dtype=np.float32).copy()
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q /= norm

    if prev_quat_xyzw is not None:
        prev = np.asarray(prev_quat_xyzw, dtype=np.float32)
        prev_norm = np.linalg.norm(prev)
        if prev_norm > 1e-8:
            prev = prev / prev_norm
            if float(np.dot(q, prev)) < 0.0:
                q = -q
            return q

    # 无参考时使用确定性规则，减少首帧随机半球抖动
    if q[3] < 0:
        q = -q
    return q


# 训练数据 009_place_phone 的 rotvec 均值，用于首帧分支选择参考
_TRAIN_ROTVEC_MEAN = np.array([1.88379622, -0.88216197, -0.15260544], dtype=np.float32)


def canonicalize_axisangle_equivalent(axisangle, prev_axisangle=None):
    """
    选择轴角的等价表示，降低接近 pi 区域的分支跳变。

    rotvec 的等价表示之一：
      v = theta * u
      v' = -(2*pi - theta) * u
    当 theta 接近 pi 时，v 与 v' 都可能出现；这里优先选与上一帧更接近的分支。

    首帧无参考时，使用训练数据的 rotvec 均值作为参考，
    避免 180° 附近落入与训练分布相反的分支。
    """
    v = np.asarray(axisangle, dtype=np.float32)
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return v

    # scipy as_rotvec 的 theta 默认在 [0, pi]
    u = v / theta
    v_equiv = (-(2.0 * np.pi - theta) * u).astype(np.float32)

    if prev_axisangle is None:
        # 首帧无参考时，使用训练数据的 rotvec 均值作为参考
        # 训练数据 009_place_phone 的 state mean (ax, ay, az)
        ref = _TRAIN_ROTVEC_MEAN
    else:
        ref = np.asarray(prev_axisangle, dtype=np.float32)

    d_main = np.linalg.norm(v - ref)
    d_equiv = np.linalg.norm(v_equiv - ref)
    return v_equiv if d_equiv < d_main else v


def axisangle_to_quaternion_xyzw(axisangle):
    """
    将轴角 [ax, ay, az] 转换为四元数 [qx, qy, qz, qw] (XYZW 标准顺序)

    Args:
        axisangle: 轴角向量 [ax, ay, az] (弧度)

    Returns:
        四元数 [qx, qy, qz, qw]
    """
    r = R.from_rotvec(axisangle)
    return r.as_quat()


def pose8_xyzw_to_state7_axisangle(
    pose8_xyzw,
    verbose=False,
    prev_quat_xyzw=None,
    prev_axisangle=None,
):
    """
    将客户端的 8D 四元数格式转换为 7D 轴角格式

    Args:
        pose8_xyzw: [x, y, z, qx, qy, qz, qw, gripper] (8D, XYZW 标准顺序)
        verbose: 是否打印调试信息

    Returns:
        state7_aa: [x, y, z, ax, ay, az, gripper] (7D, 轴角格式)
    """
    pos = pose8_xyzw[:3]
    quat_xyzw = pose8_xyzw[3:7]
    gripper = pose8_xyzw[7]

    quat_xyzw = canonicalize_quaternion_xyzw(quat_xyzw, prev_quat_xyzw=prev_quat_xyzw)
    axisangle = quaternion_xyzw_to_axisangle(quat_xyzw)
    axisangle = canonicalize_axisangle_equivalent(axisangle, prev_axisangle=prev_axisangle)

    state7 = np.concatenate([pos, axisangle, [gripper]]).astype(np.float32)

    if verbose:
        angle_deg = np.degrees(np.linalg.norm(axisangle))
        print(f"  四元数 XYZW: [{quat_xyzw[0]:.4f}, {quat_xyzw[1]:.4f}, {quat_xyzw[2]:.4f}, {quat_xyzw[3]:.4f}]")
        print(f"  轴角: [{axisangle[0]:.4f}, {axisangle[1]:.4f}, {axisangle[2]:.4f}] (角度: {angle_deg:.1f}°)")

    return state7


def action7_axisangle_to_pose8_xyzw(action7_aa):
    """
    将模型输出的 7D 轴角格式转换为 8D 四元数格式

    Args:
        action7_aa: [x, y, z, ax, ay, az, gripper] (7D, 轴角格式)

    Returns:
        pose8_xyzw: [x, y, z, qx, qy, qz, qw, gripper] (8D, XYZW 标准顺序)
    """
    pos = action7_aa[:3]
    axisangle = action7_aa[3:6]
    gripper = action7_aa[6]

    quat_xyzw = axisangle_to_quaternion_xyzw(axisangle)

    pose8 = np.concatenate([pos, quat_xyzw, [gripper]]).astype(np.float32)
    return pose8


# =============================================================================
# VLA-Lab 集成（自动启用：安装即生效，设置 VLALAB_DISABLED=1 可禁用）
# =============================================================================
try:
    import vlalab
    VLALAB_AVAILABLE = os.environ.get("VLALAB_DISABLED", "").lower() not in ("1", "true", "yes")
except ImportError:
    VLALAB_AVAILABLE = False
    vlalab = None


# =============================================================================
# 推理服务器
# =============================================================================

class OpenPiPi05InferenceServer:
    """OpenPI Pi0.5 推理服务器 (增量轴角模式)"""

    def __init__(
        self,
        config_name: str = CONFIG_NAME,
        checkpoint_dir: str = CHECKPOINT_DIR,
        device: str = DEVICE,
        task_prompt: str = TASK_PROMPT,
        inference_freq: float = INFERENCE_FREQ,
        server_ip: str = SERVER_IP,
        server_port: int = SERVER_PORT,
        max_clients: int = MAX_CLIENTS,
        verbose: bool = VERBOSE,
    ):
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.task_prompt = task_prompt
        self.inference_freq = inference_freq
        self.server_ip = server_ip
        self.server_port = server_port
        self.max_clients = max_clients
        self.verbose = verbose

        self.policy = None
        self.running = False

        # Episode 状态
        self.episode_start_time = None
        self.last_recv_timestamp = None
        self.last_sent_quat = None
        self.last_obs_quat = None
        self.last_obs_axisangle = None
        self.skipped_obs_count = 0
        self.episode_step_count = 0
        self._view_mapping_logged = False

        # 动作安全限制状态（跨 chunk 保持）
        self.prev_action_chunk = None

        # 轨迹记录
        self.inference_log = {
            'meta': {
                'config_name': config_name,
                'checkpoint_dir': str(checkpoint_dir),
                'model_type': 'openpi_pi05',
                'action_format': '7D_axis_angle',
                'action_type': 'delta (OpenPI AbsoluteActions auto-converts)',
                'amplify_pos': ACTION_AMPLIFY_POS,
                'amplify_rot': ACTION_AMPLIFY_ROT,
                'quaternion_order': 'XYZW (standard)',
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'steps': []
        }

        # VLA-Lab 日志
        self.vlalab_run = None
        self._current_images_b64 = {}  # 用于暂存当前步骤参与推理的图像

        # 时间记录文件夹
        self.time_recordings_dir = Path(__file__).parent / "time_recordings"
        self.time_recordings_dir.mkdir(parents=True, exist_ok=True)
        self.inference_count = 0  # 推理计数
        self.time_records = []  # 存储所有推理记录
        # 生成唯一的文件名（启动时生成）
        self.time_record_filename = f"time_recordings_{datetime.now().strftime('%m%d_%H%M')}.json"

        if VLALAB_AVAILABLE:
            ckpt_name = Path(checkpoint_dir).name[:20]
            # vlalab_runs 目录创建在 openpi 根目录下
            openpi_root = Path(__file__).parent.parent.parent
            vlalab_dir = openpi_root / "vlalab_runs"
            try:
                # 显式创建目录，避免依赖 vlalab.init 的隐式行为
                vlalab_dir.mkdir(parents=True, exist_ok=True)
                self.vlalab_run = vlalab.init(
                    project=f"openpi_pi05_{ckpt_name}",
                    config={
                        "model": "openpi_pi05",
                        "config_name": config_name,
                        "checkpoint_dir": str(checkpoint_dir),
                        "task_prompt": task_prompt,
                        "inference_freq": inference_freq,
                        "action_dim": ACTION_DIM,
                        "num_images": NUM_IMAGES,
                        "action_type": "delta",
                    },
                    dir=str(vlalab_dir),
                )
                print(f"[OpenPI Pi0.5 推理服务器] VLA-Lab 已启用，日志目录: {vlalab_dir}")
            except Exception as e:
                self.vlalab_run = None
                print(f"[OpenPI Pi0.5 推理服务器] VLA-Lab 初始化失败，已禁用: {e}")
        else:
            if vlalab is None:
                print("[OpenPI Pi0.5 推理服务器] VLA-Lab 未安装，跳过日志记录")
            else:
                print("[OpenPI Pi0.5 推理服务器] VLA-Lab 已被环境变量 VLALAB_DISABLED 禁用")

        print("[OpenPI Pi0.5 推理服务器] 初始化...")
        print(f"[OpenPI Pi0.5 推理服务器] 动作格式: 7D 轴角 [x, y, z, ax, ay, az, gripper]")
        print(f"[OpenPI Pi0.5 推理服务器] 四元数顺序: XYZW (标准顺序)")
        print(f"[OpenPI Pi0.5 推理服务器] 动作类型: 增量 (delta) - OpenPI 内部自动转换为绝对量")
        print(f"[OpenPI Pi0.5 推理服务器] 训练配置: {config_name}")
        print(f"[OpenPI Pi0.5 推理服务器] Checkpoint: {checkpoint_dir}")
        print(f"[OpenPI Pi0.5 推理服务器] chunk_size={TARGET_CHUNK_SIZE}")
        print(f"[OpenPI Pi0.5 推理服务器] amplify_pos={ACTION_AMPLIFY_POS}, amplify_rot={ACTION_AMPLIFY_ROT}")
        self._load_policy()

    def _load_policy(self):
        """使用 OpenPI 标准 API 加载策略"""
        from openpi.training import config as _config
        from openpi.policies import policy_config

        print(f"[OpenPI Pi0.5 推理服务器] 加载训练配置: {self.config_name}")
        train_config = _config.get_config(self.config_name)

        print(f"[OpenPI Pi0.5 推理服务器] 加载策略 (checkpoint: {self.checkpoint_dir})...")
        self.policy = policy_config.create_trained_policy(
            train_config,
            self.checkpoint_dir,
            default_prompt=self.task_prompt,
        )
        print("[OpenPI Pi0.5 推理服务器] ✓ 策略加载成功")

        # 预热
        self._warmup()

    def _warmup(self):
        """预热模型（首次推理较慢，需要 JIT 编译）"""
        print("[OpenPI Pi0.5 推理服务器] 预热模型...")
        dummy_obs = {
            "observation/front_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.array([0.55, -0.05, 0.4, 0.0, 0.15, -1.0, 1.0], dtype=np.float32),
            "prompt": self.task_prompt,
        }
        try:
            result = self.policy.infer(dummy_obs)
            actions = result["actions"]
            print(f"[OpenPI Pi0.5 推理服务器] ✓ 预热完成 (output shape: {actions.shape})")
        except Exception as e:
            print(f"[OpenPI Pi0.5 推理服务器] 预热失败: {e}")
            import traceback
            traceback.print_exc()

    def _limit_and_smooth_action(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """
        对输出动作进行安全限制和平滑处理

        处理流程:
          1. 工作空间硬限制
          2. 缓启动速度缩放
          3. 逐步速度限制 + 加速度限制
          4. EMA 跨 chunk 平滑
        """
        action = action.copy()

        if not ENABLE_ACTION_LIMIT:
            return action

        n_steps = len(action)

        # --- 1. 工作空间硬限制 ---
        if SAFE_WORKSPACE:
            action[:, 0] = np.clip(action[:, 0], SAFE_WORKSPACE['x'][0], SAFE_WORKSPACE['x'][1])
            action[:, 1] = np.clip(action[:, 1], SAFE_WORKSPACE['y'][0], SAFE_WORKSPACE['y'][1])
            action[:, 2] = np.clip(action[:, 2], SAFE_WORKSPACE['z'][0], SAFE_WORKSPACE['z'][1])

        # --- 2. 缓启动: 前 N 步线性缩放速度上限 ---
        pos_limit = MAX_POS_DELTA_PER_STEP if MAX_POS_DELTA_PER_STEP else 1e9
        rot_limit = MAX_ROT_DELTA_PER_STEP if MAX_ROT_DELTA_PER_STEP else 1e9

        if SMOOTH_START_STEPS > 0 and self.episode_step_count < SMOOTH_START_STEPS:
            scale = 0.1 + (self.episode_step_count / SMOOTH_START_STEPS) * 0.9
            pos_limit *= scale
            rot_limit *= scale
            if self.verbose and self.episode_step_count < 5:
                print(f"[SmoothStart] Step {self.episode_step_count}: scale={scale:.2f}, "
                      f"max_pos={pos_limit*100:.2f}cm, max_rot={np.degrees(rot_limit):.1f}°")

        # --- 3. 逐步速度 + 加速度限制 ---
        ref_pos = current_state[:3].copy()
        ref_rot = current_state[3:6].copy()
        prev_pos_vel = np.zeros(3)
        prev_rot_vel = np.zeros(3)

        for i in range(n_steps):
            # ---- 位置 ----
            pos_delta = action[i, :3] - ref_pos
            pos_speed = np.linalg.norm(pos_delta)

            if pos_speed > pos_limit:
                pos_delta = pos_delta * (pos_limit / pos_speed)

            if MAX_POS_ACCELERATION and MAX_POS_ACCELERATION > 0:
                acc = pos_delta - prev_pos_vel
                acc_norm = np.linalg.norm(acc)
                if acc_norm > MAX_POS_ACCELERATION:
                    acc = acc * (MAX_POS_ACCELERATION / acc_norm)
                    pos_delta = prev_pos_vel + acc
                    vel_norm = np.linalg.norm(pos_delta)
                    if vel_norm > pos_limit:
                        pos_delta = pos_delta * (pos_limit / vel_norm)

            action[i, :3] = ref_pos + pos_delta
            prev_pos_vel = pos_delta.copy()
            ref_pos = action[i, :3].copy()

            # ---- 旋转 (轴角: 直接计算旋转向量差) ----
            rot_delta = action[i, 3:6] - ref_rot
            rot_speed = np.linalg.norm(rot_delta)

            if rot_speed > rot_limit:
                rot_delta = rot_delta * (rot_limit / rot_speed)

            if MAX_ROT_ACCELERATION and MAX_ROT_ACCELERATION > 0:
                rot_acc = rot_delta - prev_rot_vel
                rot_acc_norm = np.linalg.norm(rot_acc)
                if rot_acc_norm > MAX_ROT_ACCELERATION:
                    rot_acc = rot_acc * (MAX_ROT_ACCELERATION / rot_acc_norm)
                    rot_delta = prev_rot_vel + rot_acc
                    vel_norm = np.linalg.norm(rot_delta)
                    if vel_norm > rot_limit:
                        rot_delta = rot_delta * (rot_limit / vel_norm)

            action[i, 3:6] = ref_rot + rot_delta
            prev_rot_vel = rot_delta.copy()
            ref_rot = action[i, 3:6].copy()

        # --- 4. EMA 跨 chunk 平滑 ---
        alpha = ACTION_SMOOTHING_ALPHA
        if alpha > 0 and self.prev_action_chunk is not None:
            prev = self.prev_action_chunk
            action[0, :6] = alpha * prev[0, :6] + (1 - alpha) * action[0, :6]
            for i in range(1, min(n_steps, 3)):
                decay = alpha * (0.5 ** i)
                if i < len(prev):
                    action[i, :6] = decay * prev[i, :6] + (1 - decay) * action[i, :6]

        self.prev_action_chunk = action.copy()

        return action

    def start(self):
        """启动服务器"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.server_ip, self.server_port))
            server_socket.listen(self.max_clients)

            print(f"[OpenPI Pi0.5 推理服务器] ✓ 监听 {self.server_ip}:{self.server_port}")
            print(f"[OpenPI Pi0.5 推理服务器] 任务指令: {self.task_prompt}")
            self.running = True

            while self.running:
                try:
                    client_socket, client_addr = server_socket.accept()
                    print(f"[OpenPI Pi0.5 推理服务器] 客户端连接: {client_addr}")
                    self._handle_client(client_socket, client_addr)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[OpenPI Pi0.5 推理服务器] 错误: {e}")

            server_socket.close()
            self._save_inference_log()
            self._save_all_time_records()

        except Exception as e:
            print(f"[OpenPI Pi0.5 推理服务器] 启动失败: {e}")
            import traceback
            traceback.print_exc()

    def _handle_client(self, client_socket: socket.socket, client_addr: tuple):
        """处理客户端连接"""
        try:
            client_socket.settimeout(SOCKET_TIMEOUT)
            buffer = b''

            while self.running:
                try:
                    data = client_socket.recv(BUFFER_SIZE)
                    if not data:
                        break
                    buffer += data
                    msgs = []
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        msg = line.decode(ENCODING).strip()
                        if msg:
                            msgs.append(msg)

                    if not msgs:
                        continue

                    recv_timestamp = time.time()
                    latest_obs_msg = None
                    skipped = 0
                    for msg in msgs:
                        try:
                            parsed = json.loads(msg)
                        except Exception:
                            continue
                        if parsed.get('type') == 'reset':
                            self._process_message(client_socket, msg, recv_timestamp)
                        elif parsed.get('type') == 'observation':
                            latest_obs_msg = msg
                            skipped += 1

                    # 仅处理最新观测，丢弃堆积的旧观测
                    if latest_obs_msg is not None:
                        if skipped > 1:
                            self.skipped_obs_count += (skipped - 1)
                            if self.verbose:
                                print(f"[OpenPI Pi0.5 推理服务器] 丢弃 {skipped - 1} 个旧观测 (累计 {self.skipped_obs_count})")
                        self._process_message(client_socket, latest_obs_msg, recv_timestamp)

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[OpenPI Pi0.5 推理服务器] 接收错误: {e}")
                    break
            client_socket.close()
        except Exception as e:
            print(f"[OpenPI Pi0.5 推理服务器] 客户端错误: {e}")

    def _process_message(self, client_socket: socket.socket, message: str, recv_timestamp: float):
        """处理消息"""
        try:
            data = json.loads(message)

            if data.get('type') == 'reset':
                # 重置 episode
                self.episode_start_time = time.time()
                self.last_recv_timestamp = None
                self.last_sent_quat = None
                self.last_obs_quat = None
                self.last_obs_axisangle = None
                self.skipped_obs_count = 0
                self.episode_step_count = 0
                self._view_mapping_logged = False
                self.prev_action_chunk = None
                response = {'type': 'reset_ack'}
                client_socket.sendall((json.dumps(response) + '\n').encode(ENCODING))
                if self.verbose:
                    print("[OpenPI Pi0.5 推理服务器] ✓ Episode 重置")

            elif data.get('type') == 'observation':
                process_start_time = time.time()
                self._current_images_b64 = {}

                # 解码数据
                images_b64 = data.get('images', [])
                poses_list = data.get('poses', [])
                grippers_list = data.get('grippers', [])
                client_timestamps = data.get('timestamps', [])
                client_send_timestamp = data.get('send_timestamp')

                self.last_recv_timestamp = recv_timestamp

                # =============================================================
                # 图像解码与视角排序
                # OpenPI Pi0.5 模型期望的图像顺序: [front, wrist]
                # =============================================================

                desired_keys = ["front_view", "wrist_view"]

                view_aliases = {
                    "front": "front_view",
                    "front_cam": "front_view",
                    "front_camera": "front_view",
                    "wrist": "wrist_view",
                    "wrist_cam": "wrist_view",
                    "wrist_camera": "wrist_view",
                    "ego_view": "wrist_view",
                }

                images = []

                if isinstance(images_b64, dict):
                    selected_images_b64 = {}
                    if self.verbose and not self._view_mapping_logged:
                        print(f"[OpenPI Pi0.5 推理服务器] 视角映射: {list(images_b64.keys())} → 模型顺序: {desired_keys}")
                        self._view_mapping_logged = True

                    for key in desired_keys:
                        img_b64 = None
                        if key in images_b64:
                            img_b64 = images_b64[key]
                        else:
                            for alias, canonical in view_aliases.items():
                                if canonical == key and alias in images_b64:
                                    img_b64 = images_b64[alias]
                                    break

                        if isinstance(img_b64, list) and len(img_b64) > 0:
                            img_b64 = img_b64[-1]

                        if img_b64 is None:
                            print(f"[警告] 缺少视角 {key}，无法完成推理")
                            return

                        selected_images_b64[key] = img_b64
                        img_data = base64.b64decode(img_b64)
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
                    self._current_images_b64 = selected_images_b64
                else:
                    selected_images_b64 = {}
                    if len(images_b64) > NUM_IMAGES:
                        images_b64 = images_b64[-NUM_IMAGES:]
                    for idx, img_b64 in enumerate(images_b64):
                        if idx >= NUM_IMAGES:
                            break
                        selected_images_b64[f"camera_{idx}"] = img_b64
                        img_data = base64.b64decode(img_b64)
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
                    self._current_images_b64 = selected_images_b64

                # =============================================================
                # 准备输入 State: 8D XYZW 四元数 → 7D 轴角
                # =============================================================
                poses = np.array(poses_list, dtype=np.float32)
                grippers = np.array(grippers_list, dtype=np.float32)

                last_pose7 = poses[-1]  # [x, y, z, qx, qy, qz, qw] (XYZW)
                last_gripper1 = grippers[-1]

                if isinstance(last_gripper1, np.ndarray):
                    last_gripper1 = last_gripper1.flatten()
                    if last_gripper1.size == 0:
                        last_gripper1 = np.array([1.0], dtype=np.float32)
                    elif last_gripper1.size > 1:
                        last_gripper1 = last_gripper1[:1]
                else:
                    last_gripper1 = np.array([last_gripper1], dtype=np.float32)

                pose8_xyzw = np.concatenate([last_pose7, last_gripper1]).astype(np.float32)

                state7_aa = pose8_xyzw_to_state7_axisangle(
                    pose8_xyzw,
                    verbose=(self.verbose and self.episode_step_count == 0),
                    prev_quat_xyzw=self.last_obs_quat,
                    prev_axisangle=self.last_obs_axisangle,
                )
                self.last_obs_quat = canonicalize_quaternion_xyzw(pose8_xyzw[3:7], prev_quat_xyzw=self.last_obs_quat)
                self.last_obs_axisangle = state7_aa[3:6].copy()

                if self.verbose and self.episode_step_count == 0:
                    print(f"[Debug] 输入状态转换:")
                    print(f"  8D XYZW:  {pose8_xyzw}")
                    print(f"  7D 轴角:  {state7_aa}")
                    angle_deg = np.degrees(np.linalg.norm(state7_aa[3:6]))
                    print(f"  旋转角度: {angle_deg:.1f}°")

                # =============================================================
                # 构建 OpenPI 观测字典并推理
                # =============================================================
                obs = {
                    "observation/front_image": images[0],   # RGB (H, W, 3)
                    "observation/wrist_image": images[1],    # RGB (H, W, 3)
                    "observation/state": state7_aa,          # [x, y, z, ax, ay, az, gripper]
                    "prompt": self.task_prompt,
                }

                infer_start_time = time.time()
                result = self.policy.infer(obs)
                infer_end_time = time.time()

                # OpenPI 返回的 actions 已经是绝对目标位置
                # (经过 AbsoluteActions transform: delta + state → absolute)
                action_chunk_7d = result["actions"]  # (action_horizon, 7)

                # 截断到 TARGET_CHUNK_SIZE
                if len(action_chunk_7d) > TARGET_CHUNK_SIZE:
                    action_chunk_7d = action_chunk_7d[:TARGET_CHUNK_SIZE]

                # [Debug] 打印初始动作偏差 (仅第一步)
                if self.episode_step_count == 0 and len(action_chunk_7d) > 0:
                    cur_pos = state7_aa[:3]
                    cur_rot = state7_aa[3:6]
                    act_pos = action_chunk_7d[0, :3]
                    act_rot = action_chunk_7d[0, 3:6]
                    pos_dist = np.linalg.norm(cur_pos - act_pos)
                    rot_diff = np.linalg.norm(cur_rot - act_rot)
                    print(f"[Debug] First Step Delta:")
                    print(f"  Pos Dist: {pos_dist*100:.2f} cm")
                    print(f"  Rot Diff: {np.degrees(rot_diff):.2f} deg (轴角向量范数差)")
                    print(f"  Cur Rot: {cur_rot}")
                    print(f"  Act Rot: {act_rot}")

                # =============================================================
                # 动作放大（补偿控制器跟踪衰减）
                # 增量模型：OpenPI 的输出已经是绝对量（delta + state），
                # 放大逻辑：从绝对量中提取 delta，放大后再加回 state
                # =============================================================
                if ACTION_AMPLIFY_POS != 1.0 or ACTION_AMPLIFY_ROT != 1.0:
                    for i in range(len(action_chunk_7d)):
                        # 位置放大 (x, y, z)
                        if ACTION_AMPLIFY_POS != 1.0:
                            delta_pos = action_chunk_7d[i, :3] - state7_aa[:3]
                            action_chunk_7d[i, :3] = state7_aa[:3] + delta_pos * ACTION_AMPLIFY_POS
                        # 旋转放大 (ax, ay, az - 轴角)
                        if ACTION_AMPLIFY_ROT != 1.0:
                            delta_rot = action_chunk_7d[i, 3:6] - state7_aa[3:6]
                            action_chunk_7d[i, 3:6] = state7_aa[3:6] + delta_rot * ACTION_AMPLIFY_ROT
                        # gripper 不放大

                # =============================================================
                # 动作安全限制 + 平滑
                # =============================================================
                action_chunk_7d = self._limit_and_smooth_action(action_chunk_7d, state7_aa)

                # =============================================================
                # 保存模型原始输出（用于日志记录）
                # =============================================================
                action_chunk_7d_raw = action_chunk_7d.copy()

                # =============================================================
                # 夹爪 chunk 一致性保持
                # chunk 内多数投票：只有当新夹爪状态的连续预测数 >= chunk_size//2 时才切换，
                # 否则保持第一个 action 的夹爪值
                # =============================================================
                if GRIPPER_CHUNK_CONSISTENCY and len(action_chunk_7d) > 0:
                    grippers = action_chunk_7d[:, 6]
                    binary = (grippers < GRIPPER_THRESHOLD).astype(int)  # 1=闭合, 0=打开
                    first_state = binary[0]
                    # 从第一个 action 开始，找到第一个不同状态的位置，统计后续连续新状态的长度
                    new_state_count = 0
                    for idx in range(len(binary)):
                        if binary[idx] != first_state:
                            new_state_count += 1
                        else:
                            if new_state_count > 0:
                                break  # 新状态中断，停止计数
                    min_switch = len(action_chunk_7d) // 2  # 向下取整
                    if new_state_count >= min_switch:
                        pass  # 新状态足够多，保持模型原始输出
                    elif new_state_count > 0:
                        # 新状态不够多，统一为第一个 action 的夹爪值
                        action_chunk_7d[:, 6] = action_chunk_7d[0, 6]
                        if self.verbose:
                            print(f"[GripperConsistency] 新状态连续数 {new_state_count} < {min_switch}，统一为首值: {action_chunk_7d[0, 6]:.4f}")

                # =============================================================
                # 转换回 8D 四元数格式发送给客户端 (XYZW 标准顺序)
                # =============================================================
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
                    current_quat = axisangle_to_quaternion_xyzw(state7_aa[3:6])
                    first_action_quat = action_chunk_8d[0, 3:7]
                    if np.dot(current_quat, first_action_quat) < 0:
                        action_chunk_8d[:, 3:7] = -action_chunk_8d[:, 3:7]

                # 与上一次发送的末帧保持同半球
                if len(action_chunk_8d) > 0 and self.last_sent_quat is not None:
                    first_q = action_chunk_8d[0, 3:7]
                    if np.dot(self.last_sent_quat, first_q) < 0:
                        action_chunk_8d[:, 3:7] = -action_chunk_8d[:, 3:7]

                if len(action_chunk_8d) > 0:
                    self.last_sent_quat = action_chunk_8d[-1, 3:7].copy()

                # =============================================================
                # 发送响应
                # =============================================================
                response = {'type': 'action_sequence', 'actions': action_chunk_8d.tolist()}
                msg = json.dumps(response) + '\n'
                client_socket.sendall(msg.encode(ENCODING))

                send_timestamp = time.time()

                # 计算推理时间差（从接收到观测到下发指令）
                inference_duration_ms = (send_timestamp - recv_timestamp) * 1000

                # 保存时间记录到 JSON 文件
                self._save_time_recording(
                    recv_timestamp=recv_timestamp,
                    send_timestamp=send_timestamp,
                    inference_duration_ms=inference_duration_ms,
                    task_prompt=self.task_prompt,
                    step_count=self.episode_step_count
                )

                if self.verbose:
                    inference_time = (infer_end_time - infer_start_time) * 1000
                    total_time = (send_timestamp - process_start_time) * 1000
                    print(f"[OpenPI Pi0.5 推理服务器] 推理: {inference_time:.1f}ms | 总计: {total_time:.1f}ms | 动作: {action_chunk_7d.shape}")

                # 记录日志
                self.episode_step_count += 1
                transport_latency_ms = (recv_timestamp - client_send_timestamp) * 1000 if client_send_timestamp else None
                total_latency_ms = (send_timestamp - client_send_timestamp) * 1000 if client_send_timestamp else None
                current_step = {
                    'step': len(self.inference_log['steps']),
                    'timing': {
                        'client_send': float(client_send_timestamp) if client_send_timestamp else None,
                        'server_recv': float(recv_timestamp),
                        'infer_start': float(infer_start_time),
                        'infer_end': float(infer_end_time),
                        'send_timestamp': float(send_timestamp),
                        'inference_latency_ms': float((infer_end_time - infer_start_time) * 1000),
                        'transport_latency_ms': float(transport_latency_ms) if transport_latency_ms is not None else None,
                        'total_latency_ms': float(total_latency_ms) if total_latency_ms is not None else None,
                    },
                    'input': {
                        'state7_axisangle': state7_aa.tolist(),
                        'prompt': self.task_prompt,
                        'obs_timestamps': client_timestamps,
                        'client_send_timestamp': client_send_timestamp,
                    },
                    'action': {
                        'action7_axisangle': action_chunk_7d_raw.tolist(),
                        'action8_xyzw': action_chunk_8d.tolist(),
                    }
                }
                self.inference_log['steps'].append(current_step)

                # VLA-Lab 日志
                if self.vlalab_run is not None:
                    images_dict = None
                    if isinstance(self._current_images_b64, dict) and len(self._current_images_b64) > 0:
                        images_dict = self._current_images_b64

                    step_idx = len(self.inference_log['steps']) - 1
                    inference_latency_ms = (infer_end_time - infer_start_time) * 1000

                    vlalab.log({
                        "state": state7_aa.tolist(),
                        "action": action_chunk_7d.tolist(),
                        "images": images_dict if images_dict else None,
                        "inference_latency_ms": inference_latency_ms,
                        "transport_latency_ms": transport_latency_ms,
                        "total_latency_ms": total_latency_ms,
                    }, step=step_idx)

        except Exception as e:
            print(f"[OpenPI Pi0.5 推理服务器] 处理错误: {e}")
            import traceback
            traceback.print_exc()

    def _save_inference_log(self):
        """保存推理日志"""
        try:
            log_dir = Path(__file__).parent / "log"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"inference_log_openpi_pi05_{timestamp}.json"
            with open(log_file, 'w') as f:
                json.dump(self.inference_log, f, indent=2)
            print(f"[OpenPI Pi0.5 推理服务器] 日志已保存: {log_file}")

            if self.vlalab_run is not None:
                vlalab.finish()
        except Exception as e:
            print(f"[错误] 保存日志失败: {e}")

    def _save_time_recording(
        self,
        recv_timestamp: float,
        send_timestamp: float,
        inference_duration_ms: float,
        task_prompt: str,
        step_count: int
    ):
        """
        记录单次推理的时间信息（追加到列表，服务器关闭时统一保存）

        Args:
            recv_timestamp: 接收观测信息的时间戳
            send_timestamp: 发送指令的时间戳
            inference_duration_ms: 推理消耗时间（毫秒）
            task_prompt: 任务指令
            step_count: 当前 episode 的步数
        """
        try:
            self.inference_count += 1

            # 构建记录数据
            record = {
                "inference_id": self.inference_count,
                "timestamp": {
                    "receive_observation": float(recv_timestamp),
                    "send_command": float(send_timestamp),
                    "receive_observation_iso": datetime.fromtimestamp(recv_timestamp).isoformat(),
                    "send_command_iso": datetime.fromtimestamp(send_timestamp).isoformat(),
                },
                "inference_duration_ms": float(inference_duration_ms),
                "task_prompt": task_prompt,
                "episode_step": step_count,
            }

            # 追加到记录列表
            self.time_records.append(record)

        except Exception as e:
            print(f"[错误] 记录时间失败: {e}")

    def _save_all_time_records(self):
        """保存所有时间记录到 JSON 文件"""
        try:
            if not self.time_records:
                print("[时间记录] 无记录可保存")
                return

            filepath = self.time_recordings_dir / self.time_record_filename

            # 提取所有 inference_duration_ms
            durations = [r["inference_duration_ms"] for r in self.time_records]

            # 构建完整的 JSON 数据
            full_record = {
                "inference_records": self.time_records,
                "summary": {
                    "inference_duration_ms": durations
                }
            }

            # 保存到 JSON 文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(full_record, f, indent=2, ensure_ascii=False)

            print(f"[时间记录] 已保存: {filepath.name}, 共 {len(self.time_records)} 条记录")

        except Exception as e:
            print(f"[错误] 保存时间记录失败: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OpenPI Pi0.5 推理服务器 (增量轴角模式)')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='OpenPI checkpoint 目录路径')
    parser.add_argument('--config_name', type=str, default=CONFIG_NAME,
                        help='OpenPI 训练配置名称 (如 pi05_stack_bowls_lora)')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='推理设备')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                        help='服务器端口')
    parser.add_argument('--prompt', type=str, default=TASK_PROMPT,
                        help='任务指令 prompt')
    args = parser.parse_args()

    server = OpenPiPi05InferenceServer(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        server_port=args.port,
        task_prompt=args.prompt,
    )
    server.start()
