#!/usr/bin/env python3
"""
OpenPI Pi0.5 真机推理服务器

职责拆分：
1. 沿用 CosmosVLA 的双端口 ZeroMQ + 二进制协议传输层
2. 保留 Pi/OpenPI 的推理语义：
   - 服务端内部使用 7D 轴角状态/动作
   - 客户端与机器人侧继续使用 8D 四元数 XYZW 动作
3. 在发送前执行安全限幅、平滑和四元数连续性处理
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R


CURRENT_DIR = Path(__file__).resolve().parent
REALWORLD_DEPLOY_DIR = CURRENT_DIR.parent
REALWORLD_PI_ROOT = REALWORLD_DEPLOY_DIR.parent
ROBOT_INFERENCE_DIR = REALWORLD_DEPLOY_DIR / "robot_inference"
CONFIG_DIR = ROBOT_INFERENCE_DIR / "configs"
OPENPI_ROOT = REALWORLD_PI_ROOT
OPENPI_SRC_DIR = OPENPI_ROOT / "src"
OPENPI_CLIENT_SRC_DIR = OPENPI_ROOT / "packages" / "openpi-client" / "src"

for path in (CURRENT_DIR, ROBOT_INFERENCE_DIR, CONFIG_DIR, OPENPI_CLIENT_SRC_DIR, OPENPI_SRC_DIR, OPENPI_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from server_config_openpi_pi05 import (  # noqa: E402
    SERVER_IP,
    SERVER_PORT,
    OBSERVATION_PORT,
    ACTION_PORT,
    CONFIG_NAME,
    CHECKPOINT_DIR,
    DEVICE,
    NUM_IMAGES,
    ACTION_DIM,
    TASK_PROMPT,
    INFERENCE_FREQ,
    TARGET_CHUNK_SIZE,
    ACTION_AMPLIFY_POS,
    ACTION_AMPLIFY_ROT,
    ENABLE_ACTION_LIMIT,
    MAX_POS_DELTA_PER_STEP,
    MAX_ROT_DELTA_PER_STEP,
    MAX_POS_ACCELERATION,
    MAX_ROT_ACCELERATION,
    SMOOTH_START_STEPS,
    ACTION_SMOOTHING_ALPHA,
    SAFE_WORKSPACE,
    GRIPPER_CHUNK_CONSISTENCY,
    GRIPPER_THRESHOLD,
    SOCKET_TIMEOUT,
    MAX_CLIENTS,
    VERBOSE,
)

import tcp_binary_protocol as binary_proto  # noqa: E402


try:
    import vlalab

    VLALAB_AVAILABLE = os.environ.get("VLALAB_DISABLED", "").lower() not in ("1", "true", "yes")
except ImportError:
    VLALAB_AVAILABLE = False
    vlalab = None


CLIENT_SERVER_CLOCK_OFFSET_NS = 875000


def _decode_transport_image_bytes(image_payload) -> np.ndarray:
    if isinstance(image_payload, str):
        img_data = base64.b64decode(image_payload)
    else:
        img_data = bytes(image_payload)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("failed to decode image payload")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def quaternion_xyzw_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_rotvec().astype(np.float32)


def canonicalize_quaternion_xyzw(
    quat_xyzw: np.ndarray,
    prev_quat_xyzw: Optional[np.ndarray] = None,
) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(4).copy()
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q /= norm

    if prev_quat_xyzw is not None:
        prev = np.asarray(prev_quat_xyzw, dtype=np.float32).reshape(4).copy()
        prev_norm = float(np.linalg.norm(prev))
        if np.isfinite(prev_norm) and prev_norm > 1e-8:
            prev /= prev_norm
            if float(np.dot(q, prev)) < 0.0:
                q = -q
            return q.astype(np.float32)

    if q[3] < 0.0:
        q = -q
    return q.astype(np.float32)


_TRAIN_ROTVEC_MEAN = np.array([1.88379622, -0.88216197, -0.15260544], dtype=np.float32)


def canonicalize_axisangle_equivalent(
    axisangle: np.ndarray,
    prev_axisangle: Optional[np.ndarray] = None,
) -> np.ndarray:
    v = np.asarray(axisangle, dtype=np.float32).reshape(3)
    theta = float(np.linalg.norm(v))
    if not np.isfinite(theta) or theta < 1e-8:
        return v.astype(np.float32)

    u = v / theta
    v_equiv = (-(2.0 * np.pi - theta) * u).astype(np.float32)
    ref = _TRAIN_ROTVEC_MEAN if prev_axisangle is None else np.asarray(prev_axisangle, dtype=np.float32).reshape(3)

    d_main = float(np.linalg.norm(v - ref))
    d_equiv = float(np.linalg.norm(v_equiv - ref))
    return v_equiv if d_equiv < d_main else v.astype(np.float32)


def axisangle_to_quaternion_xyzw(axisangle: np.ndarray) -> np.ndarray:
    return R.from_rotvec(axisangle).as_quat().astype(np.float32)


def pose8_xyzw_to_state7_axisangle(
    pose8_xyzw: np.ndarray,
    verbose: bool = False,
    prev_quat_xyzw: Optional[np.ndarray] = None,
    prev_axisangle: Optional[np.ndarray] = None,
) -> np.ndarray:
    pose8_xyzw = np.asarray(pose8_xyzw, dtype=np.float32).reshape(8)
    pos = pose8_xyzw[:3]
    quat_xyzw = canonicalize_quaternion_xyzw(pose8_xyzw[3:7], prev_quat_xyzw=prev_quat_xyzw)
    gripper = float(pose8_xyzw[7])

    axisangle = quaternion_xyzw_to_axisangle(quat_xyzw)
    axisangle = canonicalize_axisangle_equivalent(axisangle, prev_axisangle=prev_axisangle)

    state7 = np.concatenate([pos, axisangle, np.array([gripper], dtype=np.float32)], axis=0).astype(np.float32)

    if verbose:
        angle_deg = np.degrees(float(np.linalg.norm(axisangle)))
        print(
            "[OpenPI Pi0.5 推理服务器] 首帧状态转换: "
            f"quat_xyzw={quat_xyzw.tolist()} axisangle={axisangle.tolist()} angle_deg={angle_deg:.1f}"
        )

    return state7


def action7_axisangle_to_pose8_xyzw(action7_aa: np.ndarray) -> np.ndarray:
    action7_aa = np.asarray(action7_aa, dtype=np.float32).reshape(7)
    pos = action7_aa[:3]
    axisangle = action7_aa[3:6]
    gripper = float(action7_aa[6])
    quat_xyzw = axisangle_to_quaternion_xyzw(axisangle)
    return np.concatenate([pos, quat_xyzw, np.array([gripper], dtype=np.float32)], axis=0).astype(np.float32)


class OpenPiPi05InferenceServer:
    """OpenPI Pi0.5 推理服务器。"""

    def __init__(
        self,
        config_name: str = CONFIG_NAME,
        checkpoint_dir: str = CHECKPOINT_DIR,
        device: str = DEVICE,
        task_prompt: str = TASK_PROMPT,
        inference_freq: float = INFERENCE_FREQ,
        server_ip: str = SERVER_IP,
        server_port: int = SERVER_PORT,
        observation_port: int = OBSERVATION_PORT,
        action_port: int = ACTION_PORT,
        max_clients: int = MAX_CLIENTS,
        verbose: bool = VERBOSE,
    ):
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.task_prompt = task_prompt
        self.inference_freq = float(inference_freq)
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.observation_port = int(observation_port)
        self.action_port = int(action_port)
        self.max_clients = int(max_clients)
        self.verbose = bool(verbose)

        self.num_images = int(NUM_IMAGES)
        self.action_dim = int(ACTION_DIM)
        self.target_chunk_size = int(TARGET_CHUNK_SIZE)
        self.action_amplify_pos = float(ACTION_AMPLIFY_POS)
        self.action_amplify_rot = float(ACTION_AMPLIFY_ROT)
        self.enable_action_limit = bool(ENABLE_ACTION_LIMIT)
        self.max_pos_delta_per_step = None if MAX_POS_DELTA_PER_STEP is None else float(MAX_POS_DELTA_PER_STEP)
        self.max_rot_delta_per_step = None if MAX_ROT_DELTA_PER_STEP is None else float(MAX_ROT_DELTA_PER_STEP)
        self.max_pos_acceleration = None if MAX_POS_ACCELERATION is None else float(MAX_POS_ACCELERATION)
        self.max_rot_acceleration = None if MAX_ROT_ACCELERATION is None else float(MAX_ROT_ACCELERATION)
        self.smooth_start_steps = max(0, int(SMOOTH_START_STEPS))
        self.action_smoothing_alpha = float(ACTION_SMOOTHING_ALPHA)
        self.safe_workspace = SAFE_WORKSPACE
        self.gripper_chunk_consistency = bool(GRIPPER_CHUNK_CONSISTENCY)
        self.gripper_threshold = float(GRIPPER_THRESHOLD)

        self.policy = None
        self.running = False

        self.zmq_context = None
        self.obs_socket = None
        self.action_socket = None

        self.episode_start_time = None
        self.last_recv_timestamp = None
        self.last_sent_quat = None
        self.last_obs_quat = None
        self.last_obs_axisangle = None
        self.skipped_obs_count = 0
        self.episode_step_count = 0
        self._view_mapping_logged = False

        self.clock_offset_ns = int(CLIENT_SERVER_CLOCK_OFFSET_NS)
        self.next_chunk_id = 0
        self.chunk_send_timestamps_ns = {}
        self.chunk_send_complete_timestamps_ns = {}
        self.chunk_step_indices = {}
        self.protocol_session_id = 0
        self.camera_ids = tuple(binary_proto.build_camera_ids(["front_view", "wrist_view"]))
        self.prev_action_chunk = None

        self.inference_log = {
            "meta": {
                "config_name": config_name,
                "checkpoint_dir": str(checkpoint_dir),
                "model_type": "openpi_pi05",
                "action_format": "7D_axis_angle",
                "quaternion_order": "XYZW",
                "transport": "zmq_binary_dual_socket",
                "observation_port": self.observation_port,
                "action_port": self.action_port,
                "clock_offset_ns": int(self.clock_offset_ns),
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "steps": [],
        }

        self.vlalab_run = None
        if VLALAB_AVAILABLE:
            ckpt_name = Path(checkpoint_dir).name[:24]
            vlalab_dir = REALWORLD_PI_ROOT / "vlalab_runs"
            try:
                vlalab_dir.mkdir(parents=True, exist_ok=True)
                self.vlalab_run = vlalab.init(
                    project=f"openpi_pi05_{ckpt_name}",
                    config={
                        "model": "openpi_pi05",
                        "config_name": config_name,
                        "checkpoint_dir": str(checkpoint_dir),
                        "task_prompt": task_prompt,
                        "inference_freq": self.inference_freq,
                    },
                    dir=str(vlalab_dir),
                )
                print(f"[OpenPI Pi0.5 推理服务器] VLA-Lab 已启用: {vlalab_dir}")
            except Exception as exc:
                self.vlalab_run = None
                print(f"[OpenPI Pi0.5 推理服务器] VLA-Lab 初始化失败，已禁用: {exc}")

        print("[OpenPI Pi0.5 推理服务器] 初始化...")
        print(f"[OpenPI Pi0.5 推理服务器] config_name={self.config_name}")
        print(f"[OpenPI Pi0.5 推理服务器] checkpoint_dir={self.checkpoint_dir}")
        self._load_policy()
        print("[OpenPI Pi0.5 推理服务器] 图像解码后端: opencv")

    def _make_observation_endpoint(self) -> str:
        return f"tcp://{self.server_ip}:{self.observation_port}"

    def _make_action_endpoint(self) -> str:
        return f"tcp://{self.server_ip}:{self.action_port}"

    def _configure_zmq_socket(self, sock: zmq.Socket):
        sock.setsockopt(zmq.LINGER, 1000)
        sock.setsockopt(zmq.SNDTIMEO, max(1, int(SOCKET_TIMEOUT * 1000)))
        sock.setsockopt(zmq.RCVTIMEO, max(1, int(SOCKET_TIMEOUT * 1000)))
        sock.setsockopt(zmq.SNDHWM, max(2, self.max_clients * 2))
        sock.setsockopt(zmq.RCVHWM, max(2, self.max_clients * 2))
        if hasattr(zmq, "TCP_KEEPALIVE"):
            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
        if hasattr(zmq, "IMMEDIATE"):
            sock.setsockopt(zmq.IMMEDIATE, 1)

    def _send_payload(self, payload: bytes) -> Tuple[int, int, float]:
        if self.action_socket is None:
            raise RuntimeError("action socket is not initialized")
        send_start_timestamp_ns = time.time_ns()
        self.action_socket.send(payload, copy=True)
        send_complete_timestamp_ns = time.time_ns()
        latency_ms = float((send_complete_timestamp_ns - send_start_timestamp_ns) / 1e6)
        return int(send_start_timestamp_ns), int(send_complete_timestamp_ns), latency_ms

    def _load_policy(self):
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        print(f"[OpenPI Pi0.5 推理服务器] 加载训练配置: {self.config_name}")
        train_config = openpi_config.get_config(self.config_name)

        print(f"[OpenPI Pi0.5 推理服务器] 加载策略: {self.checkpoint_dir}")
        self.policy = policy_config.create_trained_policy(
            train_config,
            self.checkpoint_dir,
            default_prompt=self.task_prompt,
            pytorch_device=self.device,
        )
        print("[OpenPI Pi0.5 推理服务器] ✓ 策略加载成功")
        self._warmup()

    def _warmup(self):
        print("[OpenPI Pi0.5 推理服务器] 预热模型...")
        dummy_obs = {
            "observation/front_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.array([0.55, -0.05, 0.4, 0.0, 0.15, -1.0, 1.0], dtype=np.float32),
            "prompt": self.task_prompt,
        }
        result = self.policy.infer(dummy_obs)
        actions = np.asarray(result["actions"])
        print(f"[OpenPI Pi0.5 推理服务器] ✓ 预热完成 (output shape: {actions.shape})")

    def _image_payload_to_log_value(self, image_payload) -> str:
        if isinstance(image_payload, str):
            return image_payload
        return base64.b64encode(bytes(image_payload)).decode("utf-8")

    def _canonical_camera_name(self, camera_name: str) -> str:
        normalized_name = str(camera_name)
        alias = {
            "ego_view": "wrist_view",
            "wrist": "wrist_view",
            "front": "front_view",
        }
        return alias.get(normalized_name, normalized_name)

    def _decode_observation_image_payload(self, image_payload) -> np.ndarray:
        return _decode_transport_image_bytes(image_payload)

    def _resolve_images_for_model(
        self,
        images_payload,
        desired_keys: List[str],
    ) -> Tuple[Dict[str, np.ndarray], List[str], List[str], Dict[str, str], List[str]]:
        decoded_by_source_name: Dict[str, np.ndarray] = {}
        decoded_by_canonical_name: Dict[str, np.ndarray] = {}
        canonical_source_name_map: Dict[str, str] = {}
        received_image_keys: List[str] = []

        if isinstance(images_payload, dict) and len(images_payload) > 0:
            for cam_name, image_payload in images_payload.items():
                source_name = str(cam_name)
                received_image_keys.append(source_name)
                try:
                    image = self._decode_observation_image_payload(image_payload)
                except Exception:
                    continue
                decoded_by_source_name[source_name] = image
                canonical_name = self._canonical_camera_name(source_name)
                decoded_by_canonical_name[canonical_name] = image
                canonical_source_name_map[canonical_name] = source_name

        images_for_model: Dict[str, np.ndarray] = {}
        decoded_image_keys: List[str] = []
        matched_image_source_keys: Dict[str, str] = {}
        missing_image_keys: List[str] = []

        for desired_key in desired_keys:
            if desired_key in decoded_by_source_name:
                images_for_model[desired_key] = decoded_by_source_name[desired_key]
                decoded_image_keys.append(desired_key)
                matched_image_source_keys[desired_key] = desired_key
                continue

            canonical_key = self._canonical_camera_name(desired_key)
            if canonical_key in decoded_by_canonical_name:
                images_for_model[desired_key] = decoded_by_canonical_name[canonical_key]
                decoded_image_keys.append(desired_key)
                matched_image_source_keys[desired_key] = canonical_source_name_map.get(canonical_key, canonical_key)
                continue

            missing_image_keys.append(desired_key)

        self.inference_log["meta"]["image_key_mapping"] = {
            "received_image_keys": received_image_keys,
            "decoded_image_keys": decoded_image_keys,
            "matched_image_source_keys": matched_image_source_keys,
            "missing_image_keys": missing_image_keys,
            "model_video_keys": desired_keys,
        }
        return (
            images_for_model,
            received_image_keys,
            decoded_image_keys,
            matched_image_source_keys,
            missing_image_keys,
        )

    def _limit_and_smooth_action(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        current_state = np.asarray(current_state, dtype=np.float32).reshape(7)

        if not self.enable_action_limit:
            self.prev_action_chunk = action.copy()
            return action

        n_steps = len(action)

        if self.safe_workspace:
            action[:, 0] = np.clip(action[:, 0], self.safe_workspace["x"][0], self.safe_workspace["x"][1])
            action[:, 1] = np.clip(action[:, 1], self.safe_workspace["y"][0], self.safe_workspace["y"][1])
            action[:, 2] = np.clip(action[:, 2], self.safe_workspace["z"][0], self.safe_workspace["z"][1])

        pos_limit = self.max_pos_delta_per_step if self.max_pos_delta_per_step else 1e9
        rot_limit = self.max_rot_delta_per_step if self.max_rot_delta_per_step else 1e9

        if self.smooth_start_steps > 0 and self.episode_step_count < self.smooth_start_steps:
            scale = 0.1 + (self.episode_step_count / self.smooth_start_steps) * 0.9
            pos_limit *= scale
            rot_limit *= scale
            if self.verbose and self.episode_step_count < 5:
                print(
                    f"[SmoothStart] step={self.episode_step_count} "
                    f"scale={scale:.2f} max_pos={pos_limit*100:.2f}cm max_rot={np.degrees(rot_limit):.1f}deg"
                )

        ref_pos = current_state[:3].copy()
        ref_rot = current_state[3:6].copy()
        prev_pos_vel = np.zeros(3, dtype=np.float32)
        prev_rot_vel = np.zeros(3, dtype=np.float32)

        for i in range(n_steps):
            pos_delta = action[i, :3] - ref_pos
            pos_speed = float(np.linalg.norm(pos_delta))
            if pos_speed > pos_limit:
                pos_delta = pos_delta * (pos_limit / pos_speed)
            if self.max_pos_acceleration and self.max_pos_acceleration > 0:
                acc = pos_delta - prev_pos_vel
                acc_norm = float(np.linalg.norm(acc))
                if acc_norm > self.max_pos_acceleration:
                    acc = acc * (self.max_pos_acceleration / acc_norm)
                    pos_delta = prev_pos_vel + acc
                    vel_norm = float(np.linalg.norm(pos_delta))
                    if vel_norm > pos_limit:
                        pos_delta = pos_delta * (pos_limit / vel_norm)
            action[i, :3] = ref_pos + pos_delta
            prev_pos_vel = pos_delta.copy()
            ref_pos = action[i, :3].copy()

            rot_delta = action[i, 3:6] - ref_rot
            rot_speed = float(np.linalg.norm(rot_delta))
            if rot_speed > rot_limit:
                rot_delta = rot_delta * (rot_limit / rot_speed)
            if self.max_rot_acceleration and self.max_rot_acceleration > 0:
                rot_acc = rot_delta - prev_rot_vel
                rot_acc_norm = float(np.linalg.norm(rot_acc))
                if rot_acc_norm > self.max_rot_acceleration:
                    rot_acc = rot_acc * (self.max_rot_acceleration / rot_acc_norm)
                    rot_delta = prev_rot_vel + rot_acc
                    vel_norm = float(np.linalg.norm(rot_delta))
                    if vel_norm > rot_limit:
                        rot_delta = rot_delta * (rot_limit / vel_norm)
            action[i, 3:6] = ref_rot + rot_delta
            prev_rot_vel = rot_delta.copy()
            ref_rot = action[i, 3:6].copy()

        alpha = self.action_smoothing_alpha
        if alpha > 0 and self.prev_action_chunk is not None:
            prev = self.prev_action_chunk
            action[0, :6] = alpha * prev[0, :6] + (1.0 - alpha) * action[0, :6]
            for i in range(1, min(n_steps, 3)):
                decay = alpha * (0.5 ** i)
                if i < len(prev):
                    action[i, :6] = decay * prev[i, :6] + (1.0 - decay) * action[i, :6]

        self.prev_action_chunk = action.copy()
        return action

    def start(self):
        try:
            self.zmq_context = zmq.Context()
            self.obs_socket = self.zmq_context.socket(zmq.PULL)
            self.action_socket = self.zmq_context.socket(zmq.PUSH)
            self._configure_zmq_socket(self.obs_socket)
            self._configure_zmq_socket(self.action_socket)
            self.obs_socket.bind(self._make_observation_endpoint())
            self.action_socket.bind(self._make_action_endpoint())

            poller = zmq.Poller()
            poller.register(self.obs_socket, zmq.POLLIN)

            print(
                "[OpenPI Pi0.5 推理服务器] ✓ ZeroMQ 监听 "
                f"upload={self._make_observation_endpoint()}, download={self._make_action_endpoint()}"
            )
            print(f"[OpenPI Pi0.5 推理服务器] 任务指令: {self.task_prompt}")
            self.running = True

            while self.running:
                try:
                    events = dict(poller.poll(100))
                    if self.obs_socket not in events:
                        continue

                    recv_start_timestamp_ns = time.time_ns()
                    payload = self.obs_socket.recv(copy=True)
                    recv_timestamp_ns = time.time_ns()
                    frames = binary_proto.unpack_framed_payload(payload)
                    recv_metadata = {
                        "transport": "zmq",
                        "recv_start_timestamp_ns": int(recv_start_timestamp_ns),
                        "header_recv_timestamp_ns": int(recv_timestamp_ns),
                        "body_recv_timestamp_ns": int(recv_timestamp_ns),
                        "body_length_bytes": int(len(payload)),
                        "body_recv_chunk_count": 1,
                        "body_recv_chunk_sizes_bytes": [int(len(payload))],
                        "body_recv_chunk_intervals_ns": [],
                        "payload_bytes": int(len(payload)),
                        "framed_message_count": len(frames),
                    }
                    camera_names = binary_proto.camera_names_from_ids(self.camera_ids)
                    parsed = binary_proto.decode_message(frames, camera_names=camera_names)
                    parsed["_recv_payload_bytes"] = int(len(payload))
                    self._process_message(parsed, recv_timestamp_ns, recv_metadata)
                except KeyboardInterrupt:
                    break
                except zmq.Again:
                    continue
                except Exception as exc:
                    print(f"[OpenPI Pi0.5 推理服务器] 循环错误: {exc}")

        except Exception as exc:
            print(f"[OpenPI Pi0.5 推理服务器] 启动失败: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            if self.obs_socket is not None:
                self.obs_socket.close(0)
                self.obs_socket = None
            if self.action_socket is not None:
                self.action_socket.close(0)
                self.action_socket = None
            if self.zmq_context is not None:
                self.zmq_context.term()
                self.zmq_context = None
            self._save_inference_log()

    def _process_message(self, data: dict, recv_timestamp_ns: int, recv_metadata: Optional[Dict] = None):
        try:
            msg_type = data.get("type")

            if msg_type == "reset":
                self.episode_start_time = time.time()
                self.last_recv_timestamp = None
                self.last_sent_quat = None
                self.last_obs_quat = None
                self.last_obs_axisangle = None
                self.skipped_obs_count = 0
                self.episode_step_count = 0
                self._view_mapping_logged = False
                self.prev_action_chunk = None
                self.next_chunk_id = 0
                self.chunk_send_timestamps_ns = {}
                self.chunk_send_complete_timestamps_ns = {}
                self.chunk_step_indices = {}
                self.protocol_session_id = int(data.get("session_id", 0))

                camera_ids = data.get("camera_ids", [])
                if camera_ids:
                    self.camera_ids = tuple(int(camera_id) for camera_id in camera_ids)

                self.inference_log["meta"]["reset_camera_names"] = binary_proto.camera_names_from_ids(self.camera_ids)
                self.inference_log["meta"].pop("image_key_mapping", None)

                response, _ = binary_proto.encode_reset_ack_message(
                    session_id=self.protocol_session_id,
                    clock_offset_ns=self.clock_offset_ns,
                )
                self._send_payload(response)
                if self.verbose:
                    print("[OpenPI Pi0.5 推理服务器] ✓ Episode 重置")
                return

            if msg_type == "heartbeat":
                heartbeat_session_id = int(data.get("session_id", self.protocol_session_id))
                if heartbeat_session_id > 0:
                    self.protocol_session_id = heartbeat_session_id
                heartbeat_seq = int(data.get("heartbeat_seq", 0))
                client_heartbeat_send_timestamp_ns = int(data.get("client_heartbeat_send_timestamp_ns", 0))
                heartbeat_ack_msg, _ = binary_proto.encode_heartbeat_ack_message(
                    session_id=self.protocol_session_id,
                    heartbeat_seq=heartbeat_seq,
                    client_heartbeat_send_timestamp_ns=client_heartbeat_send_timestamp_ns,
                    server_heartbeat_recv_timestamp_ns=int(recv_timestamp_ns),
                )
                self._send_payload(heartbeat_ack_msg)
                return

            if msg_type != "observation":
                if self.verbose:
                    print(f"[OpenPI Pi0.5 推理服务器] 忽略未知消息类型: {msg_type}")
                return

            process_start_timestamp_ns = time.time_ns()

            images_payload = data.get("images", {})
            poses_list = data.get("poses", [])
            grippers_list = data.get("grippers", [])
            client_timestamps = data.get("timestamps", [])
            client_send_timestamp = data.get("send_timestamp")
            client_obs_send_timestamp_ns = data.get("client_obs_send_timestamp_ns")
            client_last_chunk_id = data.get("client_last_chunk_id")
            client_last_chunk_recv_timestamp_ns = data.get("client_last_chunk_recv_timestamp_ns")

            if client_obs_send_timestamp_ns is None and client_send_timestamp is not None:
                client_obs_send_timestamp_ns = int(float(client_send_timestamp) * 1e9)
            elif client_obs_send_timestamp_ns is not None:
                client_obs_send_timestamp_ns = int(client_obs_send_timestamp_ns)

            acked_chunk_id = None if client_last_chunk_id is None else int(client_last_chunk_id)
            acked_chunk_client_recv_timestamp_ns = (
                None if client_last_chunk_recv_timestamp_ns is None else int(client_last_chunk_recv_timestamp_ns)
            )

            self.last_recv_timestamp = recv_timestamp_ns
            recv_start_timestamp_ns = None if recv_metadata is None else recv_metadata.get("recv_start_timestamp_ns")
            header_recv_timestamp_ns = None if recv_metadata is None else recv_metadata.get("header_recv_timestamp_ns")
            body_recv_timestamp_ns = None if recv_metadata is None else recv_metadata.get("body_recv_timestamp_ns")
            recv_body_length_bytes = None if recv_metadata is None else recv_metadata.get("body_length_bytes")
            recv_body_chunk_count = None if recv_metadata is None else recv_metadata.get("body_recv_chunk_count")
            recv_body_chunk_sizes_bytes = None if recv_metadata is None else recv_metadata.get("body_recv_chunk_sizes_bytes")
            recv_body_chunk_intervals_ns = (
                None if recv_metadata is None else recv_metadata.get("body_recv_chunk_intervals_ns")
            )
            recv_payload_bytes = None if recv_metadata is None else recv_metadata.get("payload_bytes")
            recv_frame_count = None if recv_metadata is None else recv_metadata.get("framed_message_count")
            recv_transport = None if recv_metadata is None else recv_metadata.get("transport")

            recv_body_chunk_intervals_ms = None
            recv_body_interval_avg_ms = None
            recv_body_interval_max_ms = None
            recv_body_avg_chunk_bytes = None
            if isinstance(recv_body_chunk_intervals_ns, list):
                recv_body_chunk_intervals_ms = [float(interval_ns) / 1e6 for interval_ns in recv_body_chunk_intervals_ns]
                if recv_body_chunk_intervals_ms:
                    recv_body_interval_avg_ms = float(np.mean(recv_body_chunk_intervals_ms))
                    recv_body_interval_max_ms = float(np.max(recv_body_chunk_intervals_ms))
            if isinstance(recv_body_chunk_sizes_bytes, list) and recv_body_chunk_sizes_bytes:
                recv_body_avg_chunk_bytes = float(np.mean(recv_body_chunk_sizes_bytes))

            desired_keys = ["front_view", "wrist_view"][: self.num_images]
            (
                images_for_model,
                received_image_keys,
                decoded_image_keys,
                matched_image_source_keys,
                missing_image_keys,
            ) = self._resolve_images_for_model(images_payload, desired_keys)

            if self.verbose and not self._view_mapping_logged:
                print(
                    "[OpenPI Pi0.5 推理服务器] 图像键映射: "
                    f"received={received_image_keys} matched={matched_image_source_keys}"
                )
                self._view_mapping_logged = True

            if missing_image_keys:
                raise ValueError(
                    f"missing observation images for model: missing={missing_image_keys}, "
                    f"received={received_image_keys}"
                )

            if not poses_list or not grippers_list:
                raise ValueError("observation missing poses or grippers")

            poses = np.asarray(poses_list, dtype=np.float32).reshape(-1, 7)
            grippers = np.asarray(grippers_list, dtype=np.float32).reshape(-1, 1)
            last_pose7 = poses[-1]
            last_gripper_value = float(grippers[-1].reshape(-1)[0]) if grippers.size > 0 else 1.0
            state8 = np.concatenate(
                [last_pose7, np.array([last_gripper_value], dtype=np.float32)],
                axis=0,
            ).astype(np.float32)

            state7_aa = pose8_xyzw_to_state7_axisangle(
                state8,
                verbose=(self.verbose and self.episode_step_count == 0),
                prev_quat_xyzw=self.last_obs_quat,
                prev_axisangle=self.last_obs_axisangle,
            )
            self.last_obs_quat = canonicalize_quaternion_xyzw(state8[3:7], prev_quat_xyzw=self.last_obs_quat)
            self.last_obs_axisangle = state7_aa[3:6].copy()

            obs = {
                "observation/front_image": images_for_model["front_view"],
                "observation/state": state7_aa,
                "prompt": self.task_prompt,
            }
            if "wrist_view" in images_for_model:
                obs["observation/wrist_image"] = images_for_model["wrist_view"]

            infer_start_timestamp_ns = time.time_ns()
            result = self.policy.infer(obs)
            infer_end_timestamp_ns = time.time_ns()

            new_chunk_7d = np.asarray(result["actions"], dtype=np.float32).copy()
            if new_chunk_7d.ndim == 1:
                new_chunk_7d = new_chunk_7d.reshape(1, -1)
            if len(new_chunk_7d) > self.target_chunk_size:
                new_chunk_7d = new_chunk_7d[: self.target_chunk_size]

            if self.action_amplify_pos != 1.0 or self.action_amplify_rot != 1.0:
                for i in range(len(new_chunk_7d)):
                    if self.action_amplify_pos != 1.0:
                        delta_pos = new_chunk_7d[i, :3] - state7_aa[:3]
                        new_chunk_7d[i, :3] = state7_aa[:3] + delta_pos * self.action_amplify_pos
                    if self.action_amplify_rot != 1.0:
                        delta_rot = new_chunk_7d[i, 3:6] - state7_aa[3:6]
                        new_chunk_7d[i, 3:6] = state7_aa[3:6] + delta_rot * self.action_amplify_rot

            execute_chunk_7d = self._limit_and_smooth_action(new_chunk_7d, state7_aa)

            if self.gripper_chunk_consistency and len(execute_chunk_7d) > 0:
                grippers_arr = execute_chunk_7d[:, 6]
                binary_state = (grippers_arr < self.gripper_threshold).astype(np.int32)
                first_state = int(binary_state[0])
                new_state_count = 0
                for idx in range(len(binary_state)):
                    if int(binary_state[idx]) != first_state:
                        new_state_count += 1
                    elif new_state_count > 0:
                        break
                min_switch = len(execute_chunk_7d) // 2
                if 0 < new_state_count < min_switch:
                    execute_chunk_7d[:, 6] = execute_chunk_7d[0, 6]
                    if self.verbose:
                        print(
                            "[GripperConsistency] "
                            f"new_state_count={new_state_count} < {min_switch}, 使用首动作夹爪值"
                        )

            execute_chunk_7d_raw = execute_chunk_7d.copy()

            action_chunk_8d = []
            prev_quat = None
            for i in range(len(execute_chunk_7d)):
                pose8 = action7_axisangle_to_pose8_xyzw(execute_chunk_7d[i])
                curr_quat = pose8[3:7]
                if prev_quat is not None and float(np.dot(prev_quat, curr_quat)) < 0.0:
                    pose8[3:7] = -curr_quat
                    curr_quat = -curr_quat
                action_chunk_8d.append(pose8)
                prev_quat = curr_quat

            action_chunk_8d = np.asarray(action_chunk_8d, dtype=np.float32)

            if len(action_chunk_8d) > 0:
                current_quat = axisangle_to_quaternion_xyzw(state7_aa[3:6])
                if float(np.dot(current_quat, action_chunk_8d[0, 3:7])) < 0.0:
                    action_chunk_8d[:, 3:7] = -action_chunk_8d[:, 3:7]

            if len(action_chunk_8d) > 0 and self.last_sent_quat is not None:
                if float(np.dot(self.last_sent_quat, action_chunk_8d[0, 3:7])) < 0.0:
                    action_chunk_8d[:, 3:7] = -action_chunk_8d[:, 3:7]

            if len(action_chunk_8d) > 0:
                self.last_sent_quat = action_chunk_8d[-1, 3:7].copy()

            chunk_id = int(self.next_chunk_id)
            self.next_chunk_id += 1
            msg, _ = binary_proto.encode_action_message(
                chunk_id=chunk_id,
                action=action_chunk_8d,
                session_id=self.protocol_session_id,
                obs_seq=int(data.get("obs_seq", 0)),
                infer_latency_us=int((infer_end_timestamp_ns - infer_start_timestamp_ns) / 1000),
            )
            (
                server_chunk_send_timestamp_ns,
                send_complete_timestamp_ns,
                chunk_send_block_latency_ms,
            ) = self._send_payload(msg)

            if self.verbose:
                inference_time_ms = float((infer_end_timestamp_ns - infer_start_timestamp_ns) / 1e6)
                total_time_ms = float((send_complete_timestamp_ns - process_start_timestamp_ns) / 1e6)
                print(
                    "[OpenPI Pi0.5 推理服务器] "
                    f"step={self.episode_step_count} infer={inference_time_ms:.1f}ms "
                    f"total={total_time_ms:.1f}ms action_shape={tuple(action_chunk_8d.shape)}"
                )

            step_idx = len(self.inference_log["steps"])
            self.episode_step_count += 1

            obs_upload_network_latency_ms = None
            obs_upload_recv_start_latency_ms = None
            obs_upload_header_recv_latency_ms = None
            obs_upload_body_only_latency_ms = None
            obs_upload_recv_start_to_header_recv_latency_ms = None
            obs_upload_header_to_body_recv_latency_ms = None
            total_latency_ms = None
            if client_obs_send_timestamp_ns is not None:
                if recv_start_timestamp_ns is not None:
                    obs_upload_recv_start_latency_ms = (
                        recv_start_timestamp_ns - client_obs_send_timestamp_ns - self.clock_offset_ns
                    ) / 1e6
                if header_recv_timestamp_ns is not None:
                    obs_upload_header_recv_latency_ms = (
                        header_recv_timestamp_ns - client_obs_send_timestamp_ns - self.clock_offset_ns
                    ) / 1e6
                obs_upload_network_latency_ms = (
                    recv_timestamp_ns - client_obs_send_timestamp_ns - self.clock_offset_ns
                ) / 1e6
                if body_recv_timestamp_ns is not None and header_recv_timestamp_ns is not None:
                    obs_upload_header_to_body_recv_latency_ms = (
                        body_recv_timestamp_ns - header_recv_timestamp_ns
                    ) / 1e6
                if body_recv_timestamp_ns is not None and recv_start_timestamp_ns is not None:
                    obs_upload_body_only_latency_ms = (body_recv_timestamp_ns - recv_start_timestamp_ns) / 1e6
                if header_recv_timestamp_ns is not None and recv_start_timestamp_ns is not None:
                    obs_upload_recv_start_to_header_recv_latency_ms = (
                        header_recv_timestamp_ns - recv_start_timestamp_ns
                    ) / 1e6
                total_latency_ms = (
                    send_complete_timestamp_ns - client_obs_send_timestamp_ns - self.clock_offset_ns
                ) / 1e6

            acked_chunk_download_latency_ms = None
            acked_chunk_download_from_send_start_latency_ms = None
            if acked_chunk_id is not None and acked_chunk_client_recv_timestamp_ns is not None:
                acked_server_send_timestamp_ns = self.chunk_send_timestamps_ns.get(acked_chunk_id)
                acked_server_send_complete_timestamp_ns = self.chunk_send_complete_timestamps_ns.get(acked_chunk_id)
                acked_chunk_step_idx = self.chunk_step_indices.get(acked_chunk_id)
                if acked_server_send_timestamp_ns is not None:
                    acked_reference_send_timestamp_ns = (
                        int(acked_server_send_complete_timestamp_ns)
                        if acked_server_send_complete_timestamp_ns is not None
                        else int(acked_server_send_timestamp_ns)
                    )
                    acked_chunk_download_latency_ms = (
                        acked_chunk_client_recv_timestamp_ns
                        + self.clock_offset_ns
                        - acked_reference_send_timestamp_ns
                    ) / 1e6
                    acked_chunk_download_from_send_start_latency_ms = (
                        acked_chunk_client_recv_timestamp_ns
                        + self.clock_offset_ns
                        - int(acked_server_send_timestamp_ns)
                    ) / 1e6
                    if acked_chunk_step_idx is not None and 0 <= acked_chunk_step_idx < len(self.inference_log["steps"]):
                        acked_step = self.inference_log["steps"][acked_chunk_step_idx]
                        acked_timing = acked_step.setdefault("timing", {})
                        acked_timing["client_chunk_recv_timestamp_ns"] = int(acked_chunk_client_recv_timestamp_ns)
                        acked_timing["server_chunk_send_complete_timestamp_ns"] = (
                            None
                            if acked_server_send_complete_timestamp_ns is None
                            else int(acked_server_send_complete_timestamp_ns)
                        )
                        acked_timing["chunk_download_latency_ms"] = float(acked_chunk_download_latency_ms)
                        acked_timing["chunk_download_latency_from_send_start_ms"] = float(
                            acked_chunk_download_from_send_start_latency_ms
                        )
                        acked_timing["chunk_download_acked_in_step"] = int(step_idx)

            timing_dict = {
                "transport": recv_transport,
                "clock_offset_ns": int(self.clock_offset_ns),
                "client_obs_send_timestamp_ns": (
                    int(client_obs_send_timestamp_ns) if client_obs_send_timestamp_ns is not None else None
                ),
                "server_obs_recv_start_timestamp_ns": (
                    int(recv_start_timestamp_ns) if recv_start_timestamp_ns is not None else None
                ),
                "server_obs_header_recv_timestamp_ns": (
                    int(header_recv_timestamp_ns) if header_recv_timestamp_ns is not None else None
                ),
                "server_obs_recv_timestamp_ns": int(recv_timestamp_ns),
                "server_obs_payload_bytes": int(recv_payload_bytes) if recv_payload_bytes is not None else None,
                "server_obs_frame_count": int(recv_frame_count) if recv_frame_count is not None else None,
                "server_obs_body_length_bytes": (
                    int(recv_body_length_bytes) if recv_body_length_bytes is not None else None
                ),
                "server_obs_body_recv_chunk_count": (
                    int(recv_body_chunk_count) if recv_body_chunk_count is not None else None
                ),
                "server_obs_body_recv_chunk_sizes_bytes": recv_body_chunk_sizes_bytes,
                "server_obs_body_recv_avg_chunk_bytes": recv_body_avg_chunk_bytes,
                "server_obs_body_recv_chunk_intervals_ms": recv_body_chunk_intervals_ms,
                "server_obs_body_recv_chunk_interval_avg_ms": recv_body_interval_avg_ms,
                "server_obs_body_recv_chunk_interval_max_ms": recv_body_interval_max_ms,
                "infer_start_timestamp_ns": int(infer_start_timestamp_ns),
                "infer_end_timestamp_ns": int(infer_end_timestamp_ns),
                "server_chunk_send_timestamp_ns": int(server_chunk_send_timestamp_ns),
                "server_chunk_send_complete_timestamp_ns": int(send_complete_timestamp_ns),
                "chunk_send_block_latency_ms": float(chunk_send_block_latency_ms),
                "inference_latency_ms": float((infer_end_timestamp_ns - infer_start_timestamp_ns) / 1e6),
                "obs_upload_recv_start_latency_ms": obs_upload_recv_start_latency_ms,
                "obs_upload_header_recv_latency_ms": obs_upload_header_recv_latency_ms,
                "obs_upload_body_only_latency_ms": obs_upload_body_only_latency_ms,
                "obs_upload_recv_start_to_header_recv_latency_ms": obs_upload_recv_start_to_header_recv_latency_ms,
                "obs_upload_header_to_body_recv_latency_ms": obs_upload_header_to_body_recv_latency_ms,
                "obs_upload_network_latency_ms": obs_upload_network_latency_ms,
                "total_latency_ms": total_latency_ms,
                "acked_chunk_id": int(acked_chunk_id) if acked_chunk_id is not None else None,
                "acked_chunk_client_recv_timestamp_ns": (
                    int(acked_chunk_client_recv_timestamp_ns)
                    if acked_chunk_client_recv_timestamp_ns is not None
                    else None
                ),
                "acked_chunk_download_latency_ms": acked_chunk_download_latency_ms,
                "acked_chunk_download_latency_from_send_start_ms": acked_chunk_download_from_send_start_latency_ms,
            }

            current_step = {
                "step": step_idx,
                "timing": timing_dict,
                "input": {
                    "state8_xyzw": state8.tolist(),
                    "state7_axisangle": state7_aa.tolist(),
                    "prompt": self.task_prompt,
                    "obs_timestamps": client_timestamps,
                    "received_image_keys": received_image_keys,
                    "decoded_image_keys": decoded_image_keys,
                    "matched_image_source_keys": matched_image_source_keys,
                },
                "action": {
                    "chunk_id": int(chunk_id),
                    "action7_axisangle_model": new_chunk_7d.tolist(),
                    "action7_axisangle_sent": execute_chunk_7d_raw.tolist(),
                    "action8_xyzw_sent": action_chunk_8d.tolist(),
                },
            }
            if missing_image_keys:
                current_step["input"]["missing_model_image_keys"] = missing_image_keys

            self.inference_log["steps"].append(current_step)
            self.chunk_send_timestamps_ns[chunk_id] = int(server_chunk_send_timestamp_ns)
            self.chunk_send_complete_timestamps_ns[chunk_id] = int(send_complete_timestamp_ns)
            self.chunk_step_indices[chunk_id] = int(step_idx)

            if self.vlalab_run is not None:
                images_dict = None
                if isinstance(images_payload, dict) and images_payload:
                    images_dict = {}
                    for model_key, source_key in matched_image_source_keys.items():
                        if source_key in images_payload:
                            images_dict[model_key] = self._image_payload_to_log_value(images_payload[source_key])
                    if not images_dict:
                        images_dict = {k: self._image_payload_to_log_value(v) for k, v in images_payload.items()}

                vlalab.log(
                    {
                        "state": state7_aa.tolist(),
                        "action": execute_chunk_7d_raw.tolist(),
                        "images": images_dict if images_dict else None,
                        "inference_latency_ms": timing_dict.get("inference_latency_ms"),
                        "obs_upload_network_latency_ms": timing_dict.get("obs_upload_network_latency_ms"),
                    },
                    step=step_idx,
                )

        except Exception as exc:
            print(f"[OpenPI Pi0.5 推理服务器] 处理错误: {exc}")
            import traceback

            traceback.print_exc()

    def _save_inference_log(self):
        try:
            log_dir = CURRENT_DIR / "log"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"inference_log_openpi_pi05_{timestamp}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(self.inference_log, f, indent=2, ensure_ascii=False)
            print(f"[OpenPI Pi0.5 推理服务器] 日志已保存: {log_file}")
            if self.vlalab_run is not None:
                vlalab.finish()
        except Exception as exc:
            print(f"[OpenPI Pi0.5 推理服务器] 保存日志失败: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenPI Pi0.5 inference server")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="OpenPI checkpoint 目录")
    parser.add_argument("--config_name", type=str, default=CONFIG_NAME, help="OpenPI 训练配置名")
    parser.add_argument("--device", type=str, default=DEVICE, help="PyTorch 推理设备")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="观测上传端口")
    parser.add_argument("--observation_port", type=int, default=None, help="观测上传 ZeroMQ 端口")
    parser.add_argument("--action_port", type=int, default=None, help="动作下发 ZeroMQ 端口")
    parser.add_argument("--prompt", type=str, default=TASK_PROMPT, help="任务指令")
    parser.add_argument("--inference_freq", type=float, default=INFERENCE_FREQ, help="推理频率")
    args = parser.parse_args()

    observation_port = args.port if args.observation_port is None else int(args.observation_port)
    action_port = (observation_port + 1) if args.action_port is None else int(args.action_port)

    server = OpenPiPi05InferenceServer(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        task_prompt=args.prompt,
        inference_freq=args.inference_freq,
        server_port=observation_port,
        observation_port=observation_port,
        action_port=action_port,
    )
    server.start()
