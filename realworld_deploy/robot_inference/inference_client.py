#!/usr/bin/env python3
"""
Polymetis 推理客户端（统一版本）
支持本地直连和 SSH 隧道两种模式
"""

import socket
import json
import time
import os
from collections import deque
import numpy as np
import cv2
import threading
import subprocess
import argparse
import torch
import zmq
from queue import Empty, Queue
from pathlib import Path
from typing import Optional, Tuple, Dict
from threading import Thread, Lock, Event

try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_420
except Exception as e:
    raise RuntimeError(
        "TurboJPEG is required for transport image encoding. Install python turbojpeg bindings and libturbojpeg before running inference_client.py"
    ) from e

# 设置路径（使代码可在任意目录运行）
import _path_setup
from async_chunk_scheduler import AsyncActionChunkScheduler
import tcp_binary_protocol as binary_proto

try:
    from polymetis import RobotInterface, GripperInterface
    print("✓ Polymetis 库导入成功")
except ImportError as e:
    print(f"✗ 无法导入 Polymetis 库: {e}")
    import sys
    sys.exit(1)


def _is_finite_array(x: np.ndarray) -> bool:
    x = np.asarray(x)
    return bool(np.all(np.isfinite(x)))


def _normalize_quat_xyzw(q: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Normalize quaternion in (qx,qy,qz,qw) order. Returns (q_unit, ok).
    
    四元数格式: XYZW = [qx, qy, qz, qw]
    单位四元数 (无旋转): [0, 0, 0, 1]
    """
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4 or not _is_finite_array(q):
        # XYZW Identity: [0, 0, 0, 1] (qx=0, qy=0, qz=0, qw=1)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), False
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), False
    return (q / n).astype(np.float32), True


def _quat_ensure_continuity_xyzw(q: np.ndarray, q_ref: Optional[np.ndarray]) -> np.ndarray:
    """Ensure q is in same hemisphere as q_ref (to avoid sign flips)."""
    if q_ref is None:
        return q
    q = np.asarray(q, dtype=np.float32).reshape(4)
    q_ref = np.asarray(q_ref, dtype=np.float32).reshape(4)
    if float(np.dot(q, q_ref)) < 0.0:
        return (-q).astype(np.float32)
    return q


def _quat_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between two unit quaternions (qx,qy,qz,qw)."""
    t = float(np.clip(t, 0.0, 1.0))
    q0, _ = _normalize_quat_xyzw(q0)
    q1, _ = _normalize_quat_xyzw(q1)
    q1 = _quat_ensure_continuity_xyzw(q1, q0)

    dot = float(np.dot(q0, q1))
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        # near-linear
        q = q0 + t * (q1 - q0)
        q, _ = _normalize_quat_xyzw(q)
        return q

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    sin_theta = float(np.sin(theta))

    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    q = (s0 * q0 + s1 * q1).astype(np.float32)
    q, _ = _normalize_quat_xyzw(q)
    return q


def _encode_json_line(
    data: Dict,
    timestamp_ns_field: Optional[str] = None,
    timestamp_s_field: Optional[str] = None,
) -> Tuple[str, Optional[int]]:
    payload = dict(data)
    timestamp_ns_placeholder = "__CASCADE_TS_NS__"
    timestamp_s_placeholder = "__CASCADE_TS_S__"

    if timestamp_ns_field is not None:
        payload[timestamp_ns_field] = timestamp_ns_placeholder
    if timestamp_s_field is not None:
        payload[timestamp_s_field] = timestamp_s_placeholder

    msg = json.dumps(payload)
    send_timestamp_ns = None

    if timestamp_ns_field is not None or timestamp_s_field is not None:
        send_timestamp_ns = time.time_ns()
        if timestamp_ns_field is not None:
            msg = msg.replace(f'"{timestamp_ns_placeholder}"', str(send_timestamp_ns), 1)
        if timestamp_s_field is not None:
            msg = msg.replace(f'"{timestamp_s_placeholder}"', f"{send_timestamp_ns / 1e9:.9f}", 1)

    return msg + '\n', send_timestamp_ns


_FAST_JPEG_ENCODER = None
_FAST_JPEG_BACKEND = 'turbojpeg'
try:
    _FAST_JPEG_ENCODER = TurboJPEG()
except Exception as e:
    raise RuntimeError(
        "Failed to initialize TurboJPEG encoder. Ensure libturbojpeg is installed and visible to the runtime linker."
    ) from e


def _encode_transport_image(image: np.ndarray, quality: int) -> bytes:
    image = np.ascontiguousarray(image)
    quality = int(quality)
    return _FAST_JPEG_ENCODER.encode(
        image,
        quality=quality,
        pixel_format=TJPF_BGR,
        jpeg_subsample=TJSAMP_420,
    )


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return int(parsed)


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return float(parsed)


TCP_TUNE_SNDBUF_BYTES = _get_env_int("GROOT_TCP_SNDBUF_BYTES", 4 * 1024 * 1024)
TCP_TUNE_RCVBUF_BYTES = _get_env_int("GROOT_TCP_RCVBUF_BYTES", 4 * 1024 * 1024)
TCP_TUNE_ENABLE_QUICKACK = _get_env_int("GROOT_TCP_ENABLE_QUICKACK", 1)
TCP_TUNE_ENABLE_KEEPALIVE = _get_env_int("GROOT_TCP_ENABLE_KEEPALIVE", 1)
TCP_TUNE_KEEPIDLE_SEC = _get_env_int("GROOT_TCP_KEEPIDLE_SEC", 30)
TCP_TUNE_KEEPINTVL_SEC = _get_env_int("GROOT_TCP_KEEPINTVL_SEC", 10)
TCP_TUNE_KEEPCNT = _get_env_int("GROOT_TCP_KEEPCNT", 3)
HEARTBEAT_ENABLE = _get_env_int("GROOT_HEARTBEAT_ENABLE", 1)
HEARTBEAT_INTERVAL_MS = _get_env_int("GROOT_HEARTBEAT_INTERVAL_MS", 250)
HEARTBEAT_LOG_INTERVAL_SEC = _get_env_float("GROOT_HEARTBEAT_LOG_INTERVAL_SEC", 2.0)
HEARTBEAT_OFFSET_SMOOTHING_ALPHA = _get_env_float("GROOT_HEARTBEAT_OFFSET_ALPHA", 0.2)


def _tune_tcp_socket(sock: socket.socket):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if TCP_TUNE_SNDBUF_BYTES > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(TCP_TUNE_SNDBUF_BYTES))
    if TCP_TUNE_RCVBUF_BYTES > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(TCP_TUNE_RCVBUF_BYTES))
    if TCP_TUNE_ENABLE_KEEPALIVE:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, int(TCP_TUNE_KEEPIDLE_SEC))
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, int(TCP_TUNE_KEEPINTVL_SEC))
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, int(TCP_TUNE_KEEPCNT))
    if TCP_TUNE_ENABLE_QUICKACK and hasattr(socket, "TCP_QUICKACK"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        except OSError:
            pass


class ObservationBuffer:
    """观测缓冲区管理器（实现时间对齐）"""
    
    def __init__(self, num_cameras: int = 1):
        self.num_cameras = num_cameras
        
        self.latest_images = [None for _ in range(num_cameras)]
        self.latest_image_timestamp = None
        self.latest_pose = None
        self.latest_gripper = None
        self.latest_state_timestamp = None
        self.lock = Lock()
    
    def add_image(self, image: np.ndarray, timestamp: float, camera_idx: int = 0):
        """添加图像到指定相机的缓冲区"""
        with self.lock:
            if camera_idx < len(self.latest_images):
                self.latest_images[camera_idx] = image
                if camera_idx == 0:
                    self.latest_image_timestamp = timestamp
    
    def add_state(self, pose_7d: np.ndarray, gripper_1d: np.ndarray, timestamp: float):
        """添加状态（拆分为7D位姿 + 1D夹爪）"""
        with self.lock:
            self.latest_pose = pose_7d
            self.latest_gripper = gripper_1d
            self.latest_state_timestamp = timestamp

    def get_latest_state(self) -> Optional[Dict]:
        with self.lock:
            if self.latest_pose is None or self.latest_gripper is None or self.latest_state_timestamp is None:
                return None

            return {
                'pose': np.asarray(self.latest_pose),
                'gripper': np.asarray(self.latest_gripper),
                'timestamp': float(self.latest_state_timestamp)
            }

    def get_aligned_obs(self) -> Optional[Dict]:
        with self.lock:
            for image in self.latest_images:
                if image is None:
                    return None
            
            if self.latest_pose is None or self.latest_gripper is None:
                return None
            
            last_image_timestamp = self.latest_image_timestamp
            last_state_timestamp = self.latest_state_timestamp
            last_timestamp = max(last_image_timestamp, last_state_timestamp)
            
            return {
                'images': np.stack(self.latest_images, axis=0),
                'pose': np.asarray(self.latest_pose),
                'gripper': np.asarray(self.latest_gripper),
                'timestamp': float(last_timestamp)
            }


class DPFormatConverter:
    """DP 格式转换器
    
    配置文件格式：
    - robot_eef_pose: [7] = [x, y, z, qx, qy, qz, qw]
    - robot_gripper_state: [1] = [gripper]
    - action: [7] = [x, y, z, qx, qy, qz, qw] (gripper单独处理)
    """
    
    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSED = 0.0
    GRIPPER_THRESHOLD = 0.5
    
    @staticmethod
    def polymetis_to_dp_state(ee_pos: np.ndarray, ee_quat: np.ndarray, gripper_open: bool) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (7D位姿, 1D夹爪) 以匹配配置文件"""
        pose_7d = np.concatenate([ee_pos, ee_quat]).astype(np.float32)  # [x,y,z,qx,qy,qz,qw]
        gripper_value = DPFormatConverter.GRIPPER_OPEN if gripper_open else DPFormatConverter.GRIPPER_CLOSED
        gripper_1d = np.array([gripper_value], dtype=np.float32)
        return pose_7d, gripper_1d
    
    @staticmethod
    def dp_to_polymetis_action(pose_7d: np.ndarray, gripper_1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """从 (7D位姿, 1D夹爪) 转换为 Polymetis 格式"""
        pose_7d = np.array(pose_7d).flatten()
        ee_pos = pose_7d[:3]  # [x, y, z]
        ee_quat = pose_7d[3:7]  # [qx, qy, qz, qw]
        gripper_value = float(gripper_1d[0]) if isinstance(gripper_1d, np.ndarray) else float(gripper_1d)
        gripper_open = gripper_value > DPFormatConverter.GRIPPER_THRESHOLD
        return ee_pos.astype(np.float32), ee_quat.astype(np.float32), gripper_open


class LocalSocketClient:
    """本地 Socket 客户端（直接连接，无需SSH隧道）"""
    
    def __init__(self, server_ip: str, server_port: int, action_port: Optional[int] = None, timeout_s: float = 5.0, buffer_size: int = 4096, encoding: str = 'utf-8'):
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.action_port = int(action_port if action_port is not None else (self.server_port + 1))
        self.timeout_s = float(timeout_s)
        self.buffer_size = buffer_size
        self.encoding = encoding
        self.context = None
        self.upload_socket = None
        self.download_socket = None
        self.protocol_session_id = 0
        self.protocol_camera_names = []
        self.protocol_obs_seq = 0
        self.running = False
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self.sender_thread = None
        self.receiver_thread = None
        self.transport_ready = Event()

    def configure_protocol(self, session_id: int, camera_names):
        self.protocol_session_id = int(session_id)
        self.protocol_camera_names = list(camera_names)
        self.protocol_obs_seq = 0

    def peek_next_obs_seq(self) -> int:
        return int(self.protocol_obs_seq)

    def _make_upload_endpoint(self) -> str:
        return f"tcp://{self.server_ip}:{self.server_port}"

    def _make_download_endpoint(self) -> str:
        return f"tcp://{self.server_ip}:{self.action_port}"

    def _configure_zmq_socket(self, sock: zmq.Socket):
        sock.setsockopt(zmq.LINGER, 1000)
        sock.setsockopt(zmq.SNDTIMEO, max(1, int(self.timeout_s * 1000)))
        sock.setsockopt(zmq.RCVTIMEO, max(1, int(self.timeout_s * 1000)))
        sock.setsockopt(zmq.SNDHWM, 8)
        sock.setsockopt(zmq.RCVHWM, 8)
        if hasattr(zmq, 'TCP_KEEPALIVE'):
            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
        if hasattr(zmq, 'IMMEDIATE'):
            sock.setsockopt(zmq.IMMEDIATE, 1)

    def _sender_loop(self):
        try:
            sock = self.context.socket(zmq.PUSH)
            self._configure_zmq_socket(sock)
            sock.connect(self._make_upload_endpoint())
            self.upload_socket = sock
            self.transport_ready.set()
            while self.running:
                try:
                    payload, result_queue = self.send_queue.get(timeout=0.1)
                except Empty:
                    continue
                if payload is None:
                    break
                try:
                    send_start_ns = time.time_ns()
                    sock.send(payload, copy=True)
                    send_end_ns = time.time_ns()
                    result_queue.put((True, int(send_start_ns), float((send_end_ns - send_start_ns) / 1e6)))
                except Exception as exc:
                    result_queue.put((False, exc, None))
        except Exception as exc:
            self.transport_ready.set()
            self.recv_queue.put({"type": "_transport_error", "message": str(exc)})
        finally:
            if self.upload_socket is not None:
                try:
                    self.upload_socket.close(0)
                except Exception:
                    pass
                self.upload_socket = None

    def _receiver_loop(self):
        try:
            sock = self.context.socket(zmq.PULL)
            self._configure_zmq_socket(sock)
            sock.connect(self._make_download_endpoint())
            self.download_socket = sock
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            self.transport_ready.set()
            while self.running:
                events = dict(poller.poll(100))
                if sock not in events:
                    continue
                try:
                    payload = sock.recv(copy=True)
                except Exception as exc:
                    if self.running:
                        self.recv_queue.put({"type": "_transport_error", "message": str(exc)})
                    continue
                recv_timestamp_ns = time.time_ns()
                frames = binary_proto.unpack_framed_payload(payload)
                parsed = binary_proto.decode_message(frames, camera_names=self.protocol_camera_names)
                parsed['_socket_recv_timestamp_ns'] = int(recv_timestamp_ns)
                parsed['_recv_payload_bytes'] = int(len(payload))
                self.recv_queue.put(parsed)
        except Exception as exc:
            self.transport_ready.set()
            self.recv_queue.put({"type": "_transport_error", "message": str(exc)})
        finally:
            if self.download_socket is not None:
                try:
                    self.download_socket.close(0)
                except Exception:
                    pass
                self.download_socket = None

    def _send_payload(self, payload: bytes) -> Tuple[bool, Optional[int], Optional[float]]:
        if not self.running:
            return False, None, None
        result_queue = Queue(maxsize=1)
        self.send_queue.put((payload, result_queue))
        try:
            ok, detail, send_latency_ms = result_queue.get(timeout=max(1.0, self.timeout_s + 1.0))
        except Empty:
            return False, None, None
        if not ok:
            print(f"[本地连接] 发送错误: {detail}")
            return False, None, None
        return True, int(detail), send_latency_ms
    
    def connect(self, timeout: float = 5.0) -> bool:
        """连接到服务器"""
        try:
            del timeout
            self.context = zmq.Context()
            self.running = True
            self.transport_ready.clear()
            self.sender_thread = Thread(target=self._sender_loop, daemon=True)
            self.receiver_thread = Thread(target=self._receiver_loop, daemon=True)
            self.sender_thread.start()
            self.receiver_thread.start()
            if not self.transport_ready.wait(timeout=2.0):
                raise TimeoutError("ZeroMQ transport thread startup timed out")
            print(f"[本地连接] ✓ ZeroMQ 已连接, upload={self._make_upload_endpoint()}, download={self._make_download_endpoint()}")
            return True
        except Exception as e:
            print(f"[本地连接] ✗ 连接失败: {e}")
            self.close()
            return False
    
    def send_data(self, data: Dict) -> bool:
        """发送数据"""
        try:
            if data.get('type') != 'reset':
                raise ValueError(f"unsupported rawbytes message type: {data.get('type')}")
            payload, _ = binary_proto.encode_reset_message(
                session_id=self.protocol_session_id,
                camera_names=self.protocol_camera_names,
            )
            ok, _send_timestamp_ns, _send_latency_ms = self._send_payload(payload)
            return bool(ok)
        except Exception as e:
            print(f"[本地连接] 发送错误: {e}")
            return False

    def send_data_with_timestamp(
        self,
        data: Dict,
        timestamp_ns_field: str,
        timestamp_s_field: Optional[str] = None,
    ) -> Tuple[bool, Optional[int], Optional[float], Optional[float]]:
        try:
            if data.get('type') != 'observation':
                raise ValueError(f"unsupported rawbytes message type: {data.get('type')}")
            pack_start_ns = time.time_ns()
            payload, timestamp_offset = binary_proto.prepare_observation_message(
                data=data,
                session_id=self.protocol_session_id,
                obs_seq=self.protocol_obs_seq,
                camera_names=self.protocol_camera_names,
            )
            pack_end_ns = time.time_ns()
            send_timestamp_ns = time.time_ns()
            binary_proto.set_observation_send_timestamp(payload, timestamp_offset, send_timestamp_ns)
            ok, actual_send_timestamp_ns, send_latency_ms = self._send_payload(bytes(payload))
            if not ok or actual_send_timestamp_ns is None:
                return False, None, None, None
            self.protocol_obs_seq += 1
            return True, int(send_timestamp_ns), (pack_end_ns - pack_start_ns) / 1e6, send_latency_ms
        except Exception as e:
            print(f"[本地连接] 发送错误: {e}")
            return False, None, None, None

    def send_heartbeat(
        self,
        heartbeat_seq: int,
        client_clock_offset_ns: Optional[int] = None,
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        try:
            payload, heartbeat_send_timestamp_ns = binary_proto.encode_heartbeat_message(
                session_id=self.protocol_session_id,
                heartbeat_seq=int(heartbeat_seq),
                client_clock_offset_ns=client_clock_offset_ns,
            )
            ok, _actual_send_timestamp_ns, send_latency_ms = self._send_payload(payload)
            if not ok:
                return False, None, None
            return True, int(heartbeat_send_timestamp_ns), send_latency_ms
        except Exception as e:
            print(f"[本地连接] 心跳发送错误: {e}")
            return False, None, None
    
    def recv_data(self, timeout: float = 5.0) -> Optional[Dict]:
        """接收数据"""
        try:
            if not self.running:
                return None
            message = self.recv_queue.get(timeout=timeout)
            if message.get('type') == '_transport_error':
                print(f"[本地连接] 接收错误: {message.get('message')}")
                return None
            return message
        except Empty:
            return None
        except Exception as e:
            print(f"[本地连接] 接收错误: {e}")
            return None
    
    def close(self):
        """关闭连接"""
        self.running = False
        try:
            self.send_queue.put_nowait((None, None))
        except Exception:
            pass
        if self.sender_thread and self.sender_thread.is_alive():
            self.sender_thread.join(timeout=2.0)
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=2.0)
        if self.context is not None:
            try:
                self.context.term()
            except Exception:
                pass
            self.context = None
        while not self.recv_queue.empty():
            try:
                self.recv_queue.get_nowait()
            except Empty:
                break
        while not self.send_queue.empty():
            try:
                self.send_queue.get_nowait()
            except Empty:
                break
        if self.sender_thread or self.receiver_thread:
            print(f"[本地连接] ✓ 连接已关闭")
        self.sender_thread = None
        self.receiver_thread = None
    
    def stop_tunnel(self):
        """兼容接口（本地模式无需隧道）"""
        pass


class SSHTunnelClient:
    """SSH 隧道客户端"""
    
    def __init__(self, ssh_host: str, ssh_user: str, ssh_key: str, ssh_port: int, 
                 remote_port: int, local_port: int = 8007, action_port: Optional[int] = None, local_action_port: Optional[int] = None, timeout_s: float = 5.0, buffer_size: int = 4096, encoding: str = 'utf-8'):
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.ssh_port = ssh_port
        self.remote_port = int(remote_port)
        self.local_port = int(local_port)
        self.remote_action_port = int(action_port if action_port is not None else (self.remote_port + 1))
        self.local_action_port = int(local_action_port if local_action_port is not None else (self.local_port + 1))
        self.upload_tunnel_process = None
        self.download_tunnel_process = None
        self._transport = LocalSocketClient(
            server_ip='127.0.0.1',
            server_port=self.local_port,
            action_port=self.local_action_port,
            timeout_s=timeout_s,
            buffer_size=buffer_size,
            encoding=encoding,
        )

    def configure_protocol(self, session_id: int, camera_names):
        self._transport.configure_protocol(session_id, camera_names)

    def peek_next_obs_seq(self) -> int:
        return self._transport.peek_next_obs_seq()
        
    def start_tunnel(self):
        print(
            f"[SSH隧道] 启动隧道: {self.ssh_host}:"
            f"{self.remote_port}/{self.remote_action_port} -> localhost:{self.local_port}/{self.local_action_port}"
        )
        ssh_common_args = [
            'ssh', '-i', self.ssh_key, '-p', str(self.ssh_port),
            '-o', 'IPQoS=lowdelay',
            '-o', 'TCPKeepAlive=yes',
            '-o', 'ExitOnForwardFailure=yes',
        ]
        upload_cmd = ssh_common_args + [
                      '-L', f'{self.local_port}:127.0.0.1:{self.remote_port}',
                      '-N', f'{self.ssh_user}@{self.ssh_host}']
        download_cmd = ssh_common_args + [
                        '-L', f'{self.local_action_port}:127.0.0.1:{self.remote_action_port}',
                        '-N', f'{self.ssh_user}@{self.ssh_host}']
        
        try:
            self.upload_tunnel_process = subprocess.Popen(upload_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.download_tunnel_process = subprocess.Popen(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)
            if self.upload_tunnel_process.poll() is not None:
                raise RuntimeError("upload tunnel exited unexpectedly")
            if self.download_tunnel_process.poll() is not None:
                raise RuntimeError("download tunnel exited unexpectedly")
            print(f"[SSH隧道] ✓ 独立双隧道已建立")
            return True
        except Exception as e:
            self.stop_tunnel()
            print(f"[SSH隧道] ✗ 启动失败: {e}")
            return False
    
    def stop_tunnel(self):
        closed_any = False
        for process in (self.upload_tunnel_process, self.download_tunnel_process):
            if process:
                process.terminate()
                process.wait()
                closed_any = True
        self.upload_tunnel_process = None
        self.download_tunnel_process = None
        if closed_any:
            print(f"[SSH隧道] ✓ 隧道已关闭")
    
    def connect(self, timeout: float = 5.0) -> bool:
        ok = self._transport.connect(timeout=timeout)
        if ok:
            print(f"[SSH隧道] ✓ 已连接")
        else:
            print(f"[SSH隧道] ✗ 连接失败")
        return ok
    
    def send_data(self, data: Dict) -> bool:
        return self._transport.send_data(data)

    def send_data_with_timestamp(
        self,
        data: Dict,
        timestamp_ns_field: str,
        timestamp_s_field: Optional[str] = None,
    ) -> Tuple[bool, Optional[int], Optional[float], Optional[float]]:
        return self._transport.send_data_with_timestamp(data, timestamp_ns_field, timestamp_s_field)

    def send_heartbeat(
        self,
        heartbeat_seq: int,
        client_clock_offset_ns: Optional[int] = None,
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        return self._transport.send_heartbeat(heartbeat_seq, client_clock_offset_ns)
    
    def recv_data(self, timeout: float = 5.0) -> Optional[Dict]:
        return self._transport.recv_data(timeout)
    
    def close(self):
        self._transport.close()


class PolymetisInferenceClient:
    """Polymetis 推理客户端（统一版本）"""
    
    def __init__(self, mode: str = 'local', config_module=None):
        """
        初始化客户端
        
        Args:
            mode: 连接模式 ('local' 或 'ssh')
            config_module: 配置模块（如果为 None，根据 mode 自动加载）
        """
        self.mode = mode
        
        # 加载配置
        if config_module is None:
            if mode == 'local':
                from inference_config_local import (
                    SERVER_IP, SERVER_PORT, OBSERVATION_PORT, ACTION_PORT,
                    ROBOT_IP, ROBOT_PORT, GRIPPER_PORT,
                    IMAGE_QUALITY,
                    INFERENCE_FREQ,
                    COLLECT_OBS_FREQ, EXECUTION_FREQ,
                    EXECUTION_MODE, EXECUTE_HORIZON,
                    DELAY_ESTIMATE_ALPHA, DELAY_ESTIMATE_INIT_STEPS, MAX_DEADLINE_OVERRUN_STEPS,
                    ARRIVAL_SETTLE_POS_DELTA_THRESHOLD,
                    ARRIVAL_SETTLE_STABLE_COUNT, ARRIVAL_CHECK_FREQ,
                    ARRIVAL_LOG_INTERVAL_SEC, ARRIVAL_MAX_WAIT_SEC,
                    CARTESIAN_KX, CARTESIAN_KXD,
                    SOCKET_TIMEOUT, BUFFER_SIZE, ENCODING,
                    CAMERA_CONFIG_PATH,
                    ENABLE_IK_PROJECTION, IK_LINE_SEARCH_STEPS, IK_TOL,
                )
                self.config = {
                    'server_ip': SERVER_IP,
                    'server_port': SERVER_PORT,
                    'observation_port': OBSERVATION_PORT,
                    'action_port': ACTION_PORT,
                    'robot_ip': ROBOT_IP, 'robot_port': ROBOT_PORT, 'gripper_port': GRIPPER_PORT,
                    'image_quality': IMAGE_QUALITY,
                    'camera_config_path': CAMERA_CONFIG_PATH,
                    'inference_freq': INFERENCE_FREQ,
                    'collect_obs_freq': COLLECT_OBS_FREQ,
                    'execution_freq': EXECUTION_FREQ,
                    'execution_mode': EXECUTION_MODE,
                    'execute_horizon': EXECUTE_HORIZON,
                    'delay_estimate_alpha': DELAY_ESTIMATE_ALPHA,
                    'delay_estimate_init_steps': DELAY_ESTIMATE_INIT_STEPS,
                    'max_deadline_overrun_steps': MAX_DEADLINE_OVERRUN_STEPS,
                    'arrival_settle_pos_delta_threshold': ARRIVAL_SETTLE_POS_DELTA_THRESHOLD,
                    'arrival_settle_stable_count': ARRIVAL_SETTLE_STABLE_COUNT,
                    'arrival_check_freq': ARRIVAL_CHECK_FREQ,
                    'arrival_log_interval_sec': ARRIVAL_LOG_INTERVAL_SEC,
                    'arrival_max_wait_sec': ARRIVAL_MAX_WAIT_SEC,
                    'cartesian_kx': CARTESIAN_KX, 'cartesian_kxd': CARTESIAN_KXD,
                    'socket_timeout': SOCKET_TIMEOUT, 'buffer_size': BUFFER_SIZE, 'encoding': ENCODING,
                    'enable_ik_projection': ENABLE_IK_PROJECTION,
                    'ik_line_search_steps': IK_LINE_SEARCH_STEPS,
                    'ik_tol': IK_TOL,
                }
            else:  # ssh mode
                from inference_config_ssh import (
                    SSH_HOST, SSH_USER, SSH_PORT,
                    SERVER_PORT, LOCAL_PORT, OBSERVATION_PORT, ACTION_PORT,
                    LOCAL_OBSERVATION_PORT, LOCAL_ACTION_PORT,
                    ROBOT_IP, ROBOT_PORT, GRIPPER_PORT,
                    IMAGE_QUALITY,
                    INFERENCE_FREQ,
                    COLLECT_OBS_FREQ, EXECUTION_FREQ,
                    EXECUTION_MODE, EXECUTE_HORIZON,
                    DELAY_ESTIMATE_ALPHA, DELAY_ESTIMATE_INIT_STEPS, MAX_DEADLINE_OVERRUN_STEPS,
                    ARRIVAL_SETTLE_POS_DELTA_THRESHOLD,
                    ARRIVAL_SETTLE_STABLE_COUNT, ARRIVAL_CHECK_FREQ,
                    ARRIVAL_LOG_INTERVAL_SEC, ARRIVAL_MAX_WAIT_SEC,
                    CARTESIAN_KX, CARTESIAN_KXD,
                    SOCKET_TIMEOUT, BUFFER_SIZE, ENCODING,
                    CAMERA_CONFIG_PATH,
                    ENABLE_IK_PROJECTION, IK_LINE_SEARCH_STEPS, IK_TOL,
                )
                self.config = {
                    'ssh_host': SSH_HOST, 'ssh_user': SSH_USER, 'ssh_port': SSH_PORT,
                    'server_port': SERVER_PORT,
                    'observation_port': OBSERVATION_PORT,
                    'action_port': ACTION_PORT,
                    'local_port': LOCAL_PORT,
                    'local_observation_port': LOCAL_OBSERVATION_PORT,
                    'local_action_port': LOCAL_ACTION_PORT,
                    'robot_ip': ROBOT_IP, 'robot_port': ROBOT_PORT, 'gripper_port': GRIPPER_PORT,
                    'image_quality': IMAGE_QUALITY,
                    'camera_config_path': CAMERA_CONFIG_PATH,
                    'inference_freq': INFERENCE_FREQ,
                    'collect_obs_freq': COLLECT_OBS_FREQ,
                    'execution_freq': EXECUTION_FREQ,
                    'execution_mode': EXECUTION_MODE,
                    'execute_horizon': EXECUTE_HORIZON,
                    'delay_estimate_alpha': DELAY_ESTIMATE_ALPHA,
                    'delay_estimate_init_steps': DELAY_ESTIMATE_INIT_STEPS,
                    'max_deadline_overrun_steps': MAX_DEADLINE_OVERRUN_STEPS,
                    'arrival_settle_pos_delta_threshold': ARRIVAL_SETTLE_POS_DELTA_THRESHOLD,
                    'arrival_settle_stable_count': ARRIVAL_SETTLE_STABLE_COUNT,
                    'arrival_check_freq': ARRIVAL_CHECK_FREQ,
                    'arrival_log_interval_sec': ARRIVAL_LOG_INTERVAL_SEC,
                    'arrival_max_wait_sec': ARRIVAL_MAX_WAIT_SEC,
                    'cartesian_kx': CARTESIAN_KX, 'cartesian_kxd': CARTESIAN_KXD,
                    'socket_timeout': SOCKET_TIMEOUT, 'buffer_size': BUFFER_SIZE, 'encoding': ENCODING,
                    'enable_ik_projection': ENABLE_IK_PROJECTION,
                    'ik_line_search_steps': IK_LINE_SEARCH_STEPS,
                    'ik_tol': IK_TOL,
                }
        else:
            self.config = config_module
        
        # 创建连接客户端
        if mode == 'local':
            self.client = LocalSocketClient(
                server_ip=self.config['server_ip'],
                server_port=self.config.get('observation_port', self.config['server_port']),
                action_port=self.config.get('action_port'),
                timeout_s=self.config['socket_timeout'],
                buffer_size=self.config['buffer_size'],
                encoding=self.config['encoding']
            )
        else:  # ssh mode
            ssh_key = _path_setup.get_ssh_key_path('id_server')
            self.client = SSHTunnelClient(
                ssh_host=self.config['ssh_host'],
                ssh_user=self.config['ssh_user'],
                ssh_key=ssh_key,
                ssh_port=self.config['ssh_port'],
                remote_port=self.config.get('observation_port', self.config['server_port']),
                local_port=self.config.get('local_observation_port', self.config['local_port']),
                action_port=self.config.get('action_port'),
                local_action_port=self.config.get('local_action_port'),
                timeout_s=self.config['socket_timeout'],
                buffer_size=self.config['buffer_size'],
                encoding=self.config['encoding']
            )
        
        # 机器人配置
        self.robot_ip = self.config['robot_ip']
        self.robot_port = self.config['robot_port']
        self.gripper_port = self.config['gripper_port']
        self.robot = None
        self.gripper = None
        
        # 创建相机实例
        self.cameras = None
        self.camera_names = []

        from cameras import CameraManager
        import os

        config_path = self.config['camera_config_path']
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)

        self.cameras = CameraManager(config_path=config_path)
        self.camera_names = self.cameras.camera_names
        print(f"[客户端] 已从配置文件加载 {len(self.camera_names)} 个相机配置: {self.camera_names}")
        
        # 推理配置
        self.inference_freq = self.config['inference_freq']
        self.inference_interval = 1.0 / self.inference_freq
        self.collect_obs_freq = float(self.config.get('collect_obs_freq', 30.0))
        self.execution_freq = float(self.config.get('execution_freq', 30.0))
        self.execution_mode = str(self.config.get('execution_mode', 'naive_async'))
        self.execute_horizon = max(1, int(self.config.get('execute_horizon', 4)))
        self.delay_estimate_alpha = float(self.config.get('delay_estimate_alpha', 0.5))
        self.delay_estimate_init_steps = max(1, int(self.config.get('delay_estimate_init_steps', 2)))
        self.max_deadline_overrun_steps = max(0, int(self.config.get('max_deadline_overrun_steps', 2)))
        self.image_quality = self.config['image_quality']
        self.image_encode_backend = _FAST_JPEG_BACKEND
        self.arrival_settle_pos_delta_threshold = float(self.config.get('arrival_settle_pos_delta_threshold', 0.001))
        self.arrival_settle_stable_count = max(1, int(self.config.get('arrival_settle_stable_count', 3)))
        self.arrival_check_freq = float(self.config.get('arrival_check_freq', max(self.collect_obs_freq, self.execution_freq, 60.0)))
        self.arrival_log_interval_sec = float(self.config.get('arrival_log_interval_sec', 0.5))
        arrival_max_wait_sec = self.config.get('arrival_max_wait_sec', None)
        self.arrival_max_wait_sec = None if arrival_max_wait_sec is None else float(arrival_max_wait_sec)
        self.cartesian_kx = self.config['cartesian_kx']
        self.cartesian_kxd = self.config['cartesian_kxd']
        
        self.obs_buffer = ObservationBuffer(num_cameras=len(self.camera_names))
        
        # 状态变量
        self.running = False
        self.data_lock = Lock()
        self.action_received = Event()
        
        self.actions_received = 0
        self.observations_sent = 0
        self.gripper_open = True
        self.last_gripper_state = True
        
        self.control_thread = None
        self.arrival_monitor_thread = None
        self.heartbeat_thread = None
        self.current_target_pos = None
        self.current_target_quat = None
        self.target_lock = Lock()
        self.arrival_state_lock = Lock()
        self.arrival_event = Event()
        self.arrival_event.set()
        self.arrival_armed = False
        self.arrival_target_chunk_id = None
        self.latest_arrival_pos_delta = None
        self.latest_arrival_stable_count = 0
        self.arrival_last_sample_pos = None
        
        # IK Check & Projection
        self.robot_api_lock = Lock()
        self.ik_tol = float(self.config.get('ik_tol', 1e-3))
        self.enable_ik_projection = bool(self.config.get('enable_ik_projection', True))
        self.ik_line_search_steps = max(0, int(self.config.get('ik_line_search_steps', 10)))
        
        # 时间戳管理
        self.eval_t_start = None
        self.iter_idx = 0
        self.last_executed_chunk_id = None
        self.next_action_timestamp = None
        self.pending_action_chunks = deque()
        self.pending_action_responses = deque()
        self.pending_chunk_receipt_ack = None
        self.action_response_event = Event()
        
        # 回退点过滤
        self.last_action_output = None
        self.backtrack_threshold = 0.00
        self.filtered_backtrack_count = 0
        self.protocol_session_id = int(time.time_ns() & ((1 << 63) - 1))
        self.clock_offset_ns = None
        self.enable_heartbeat = bool(HEARTBEAT_ENABLE)
        self.heartbeat_interval_sec = max(0.05, float(HEARTBEAT_INTERVAL_MS) / 1000.0)
        self.heartbeat_log_interval_sec = max(0.5, float(HEARTBEAT_LOG_INTERVAL_SEC))
        self.next_heartbeat_seq = 0
        self.last_heartbeat_log_monotonic = 0.0
        self.latest_heartbeat_rtt_ms = None
        
        self.trajectory_log = {
            'observations': [],
            'actions': [],
            'executed': [],
            'heartbeat': [],
            'scheduler': [],
        }

    def _uses_async_execution(self) -> bool:
        return self.execution_mode in ('naive_async', 'rtc')

    def _reset_remote_episode(self) -> bool:
        self.client.configure_protocol(self.protocol_session_id, self.camera_names)
        if not self.client.send_data({'type': 'reset'}):
            print("[客户端] 发送 reset 失败")
            return False
        response = self.client.recv_data(timeout=5.0)
        if not response:
            print("[客户端] 等待 reset_ack 超时")
            return False
        if response.get('type') != 'reset_ack':
            print(f"[客户端] reset_ack 类型错误: {response.get('type')}")
            return False
        status_code = int(response.get('status_code', 0))
        if status_code != 0:
            print(f"[客户端] reset_ack 返回错误状态: {status_code}")
            return False
        self.clock_offset_ns = int(response.get('clock_offset_ns', 0))
        self.next_heartbeat_seq = 0
        print(f"[客户端] ✓ Episode 已重置, clock_offset_ns={self.clock_offset_ns}")
        return True

    def run(self):
        """运行客户端"""
        mode_name = "本地直连" if self.mode == 'local' else "SSH 隧道"
        print("\n" + "="*70)
        print(f"Polymetis 推理客户端 ({mode_name}模式)")
        print("="*70)
        
        try:
            # SSH 模式需要先启动隧道
            if self.mode == 'ssh':
                if not self.client.start_tunnel():
                    print("SSH 隧道启动失败")
                    return
                time.sleep(2)
            
            if not self.client.connect():
                print("连接服务器失败")
                if self.mode == 'ssh':
                    self.client.stop_tunnel()
                return

            if not self._reset_remote_episode():
                print("远端 Episode 重置失败")
                if self.mode == 'ssh':
                    self.client.stop_tunnel()
                return
            
            print("\n[客户端] 初始化机械臂...")
            self._initialize_robot()
            
            print("[客户端] 初始化摄像头...")
            camera_start_success = True
            
            results = self.cameras.start_all()
            if not any(results.values()):
                print("⚠️  所有摄像头启动失败，将不记录图像")
                camera_start_success = False
            else:
                success_count = sum(results.values())
                print(f"✓ 成功启动 {success_count}/{len(results)} 个摄像头")
            
            print(f"[客户端] 图像编码后端: {self.image_encode_backend}")
            if self.enable_heartbeat:
                print(f"[客户端] 心跳保活: interval={self.heartbeat_interval_sec:.3f}s")
            
            self.running = True
            self.eval_t_start = time.monotonic()
            
            # 启动观测收集线程
            obs_thread = Thread(target=self._collect_observations, daemon=True)
            obs_thread.start()
            image_thread = Thread(target=self._collect_images, daemon=True)
            image_thread.start()
            
            recv_thread = Thread(target=self._receive_loop, daemon=True)
            recv_thread.start()

            if self.enable_heartbeat:
                self.heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
                self.heartbeat_thread.start()
            
            self.control_thread = Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

            if not self._uses_async_execution():
                self.arrival_monitor_thread = Thread(target=self._arrival_monitor_loop, daemon=True)
                self.arrival_monitor_thread.start()
            
            self._inference_loop()
            
        except KeyboardInterrupt:
            print("\n[客户端] 检测到 Ctrl+C，正在停止...")
        except Exception as e:
            print(f"程序出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def _initialize_robot(self):
        """初始化机器人"""
        try:
            self.robot = RobotInterface(ip_address=self.robot_ip, port=self.robot_port)
            print(f"已连接到机器人")
            
            try:
                self.gripper = GripperInterface(ip_address=self.robot_ip, port=self.gripper_port)
                print(f"已连接到夹爪")
            except:
                self.gripper = None
                print(f"夹爪服务器未启动")
            
            print("启动笛卡尔阻抗控制...")
            print(f"  刚度 Kx: {self.cartesian_kx}")
            print(f"  阻尼 Kxd: {self.cartesian_kxd}")
            self.robot.start_cartesian_impedance(
                Kx=torch.Tensor(self.cartesian_kx),
                Kxd=torch.Tensor(self.cartesian_kxd)
            )
            
            ee_pos, ee_quat = self.robot.get_ee_pose()
            with self.target_lock:
                self.current_target_pos = ee_pos.cpu().numpy()
                self.current_target_quat = ee_quat.cpu().numpy()
            
            print("笛卡尔阻抗控制已启动")
        except Exception as e:
            print(f"初始化机器人失败: {e}")
            raise

    def _filter_backtracking_actions(self, action_sequence_filtered, verbose=False):
        """过滤回退的动作点，只保留沿轨迹前进的点"""
        if self.last_action_output is None or len(action_sequence_filtered) == 0:
            return action_sequence_filtered, 0
        
        last_pos, last_quat = self.last_action_output
        
        # 计算每个动作与上次执行动作的距离
        distances = []
        for single_action in action_sequence_filtered:
            if isinstance(single_action, dict):
                pose_7d = np.array(single_action['pose'])
            else:
                single_action_arr = np.array(single_action).flatten()
                if len(single_action_arr) >= 7:
                    pose_7d = single_action_arr[:7]
                else:
                    distances.append(float('inf'))
                    continue
            
            target_pos, target_quat, _ = DPFormatConverter.dp_to_polymetis_action(pose_7d, np.array([1.0]))
            pos_dist = np.linalg.norm(target_pos - last_pos)
            distances.append(pos_dist)
        
        if len(distances) == 0:
            return action_sequence_filtered, 0
        
        # 找到第一个距离大于阈值的点
        start_idx = 0
        min_distance_threshold = self.backtrack_threshold
        
        for i, dist in enumerate(distances):
            if dist > min_distance_threshold:
                start_idx = i
                break
        
        if start_idx == 0 and distances[0] <= min_distance_threshold:
            if verbose:
                print(f"[过滤] 所有动作都太接近上次位置，过滤整个chunk")
            return [], len(action_sequence_filtered)
        
        filtered_actions = action_sequence_filtered[start_idx:]
        num_filtered = start_idx
        
        if verbose and num_filtered > 0:
            print(f"[过滤] 过滤掉前 {num_filtered} 个回退点，保留 {len(filtered_actions)} 个前进点")
        
        return filtered_actions, num_filtered

    def _get_relative_time(self):
        return 0.0 if self.eval_t_start is None else max(0.0, time.monotonic() - self.eval_t_start)

    def _set_pending_chunk_receipt_ack(self, chunk_id: Optional[int], client_chunk_recv_timestamp_ns: int):
        if chunk_id is None:
            return
        with self.data_lock:
            self.pending_chunk_receipt_ack = {
                'chunk_id': int(chunk_id),
                'client_chunk_recv_timestamp_ns': int(client_chunk_recv_timestamp_ns)
            }

    def _get_pending_chunk_receipt_ack(self) -> Optional[Dict]:
        with self.data_lock:
            if self.pending_chunk_receipt_ack is None:
                return None
            return dict(self.pending_chunk_receipt_ack)

    def _clear_pending_chunk_receipt_ack(self, ack: Optional[Dict]):
        if ack is None:
            return
        with self.data_lock:
            if self.pending_chunk_receipt_ack == ack:
                self.pending_chunk_receipt_ack = None

    def _enqueue_action_response(self, response: Dict):
        with self.data_lock:
            self.pending_action_responses.append(dict(response))
            self.action_response_event.set()

    def _pop_matching_action_response(self, obs_seq: Optional[int]) -> Optional[Dict]:
        with self.data_lock:
            if not self.pending_action_responses:
                self.action_response_event.clear()
                return None

            matched = None
            retained = deque()
            while self.pending_action_responses:
                item = self.pending_action_responses.popleft()
                item_obs_seq = item.get('obs_seq')
                if matched is None and (
                    obs_seq is None
                    or item_obs_seq is None
                    or int(item_obs_seq) == int(obs_seq)
                ):
                    matched = item
                else:
                    retained.append(item)
            self.pending_action_responses = retained
            if not self.pending_action_responses:
                self.action_response_event.clear()
            return matched

    def _drain_action_responses(self) -> list[Dict]:
        with self.data_lock:
            if not self.pending_action_responses:
                self.action_response_event.clear()
                return []
            responses = list(self.pending_action_responses)
            self.pending_action_responses.clear()
            self.action_response_event.clear()
            return responses

    def _wait_for_action_response(self, obs_seq: Optional[int], timeout: float) -> Optional[Dict]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while self.running and time.monotonic() < deadline:
            response = self._pop_matching_action_response(obs_seq)
            if response is not None:
                return response
            remaining = max(0.0, deadline - time.monotonic())
            wait_timeout = min(0.05, remaining)
            if wait_timeout <= 0:
                break
            self.action_response_event.wait(timeout=wait_timeout)
        return None

    def _wait_until_action_time(self, target_time: float, scheduler: AsyncActionChunkScheduler):
        while self.running:
            self._apply_pending_async_responses(scheduler)
            remaining = float(target_time) - self._get_relative_time()
            if remaining <= 0:
                break
            self.action_response_event.wait(timeout=min(0.01, remaining))

    def _send_observation_request(
        self,
        aligned_obs: Dict,
        *,
        cycle_id: int,
        scheduler: Optional[AsyncActionChunkScheduler] = None,
    ) -> Optional[Dict]:
        last_pose = aligned_obs['pose']
        last_gripper = aligned_obs['gripper']
        observation_log_entry = {
            'step': self.observations_sent,
            'cycle_id': int(cycle_id),
            'pose': last_pose.tolist(),
            'gripper': last_gripper.tolist(),
            'timestamp': aligned_obs['timestamp'],
        }
        self.trajectory_log['observations'].append(observation_log_entry)

        latest_obs_images = aligned_obs['images']
        obs_prepare_start_ns = time.time_ns()

        images_payload = {}
        for cam_idx in range(latest_obs_images.shape[0]):
            img = latest_obs_images[cam_idx]
            cam_name = self.camera_names[cam_idx] if cam_idx < len(self.camera_names) else f"camera_{cam_idx}"
            images_payload[cam_name] = _encode_transport_image(img, self.image_quality)

        pending_chunk_receipt_ack = self._get_pending_chunk_receipt_ack()
        obs_msg = {
            'type': 'observation',
            'images': images_payload,
            'poses': [np.asarray(aligned_obs['pose'], dtype=np.float32).tolist()],
            'grippers': [np.asarray(aligned_obs['gripper'], dtype=np.float32).tolist()],
            'timestamps': [float(aligned_obs['timestamp'])],
        }
        if pending_chunk_receipt_ack is not None:
            obs_msg['client_last_chunk_id'] = int(pending_chunk_receipt_ack['chunk_id'])
            obs_msg['client_last_chunk_recv_timestamp_ns'] = int(pending_chunk_receipt_ack['client_chunk_recv_timestamp_ns'])
        if self.execution_mode == 'rtc' and scheduler is not None and scheduler.has_chunk:
            delay_estimate_steps = scheduler.rounded_delay_estimate_steps
            obs_msg['rtc_metadata'] = {
                'cycle_id': int(cycle_id),
                'execute_horizon': int(self.execute_horizon),
                'delay_estimate_steps': int(0 if delay_estimate_steps is None else max(0, delay_estimate_steps)),
            }
            obs_msg['rtc_action_prefix'] = scheduler.current_chunk()

        obs_prepare_end_ns = time.time_ns()
        obs_seq = self.client.peek_next_obs_seq() if hasattr(self.client, 'peek_next_obs_seq') else self.observations_sent
        send_ok, client_obs_send_timestamp_ns, protocol_pack_latency_ms, sendall_block_latency_ms = self.client.send_data_with_timestamp(
            obs_msg,
            timestamp_ns_field='client_obs_send_timestamp_ns',
            timestamp_s_field='send_timestamp',
        )
        observation_log_entry['obs_seq'] = int(obs_seq)
        observation_log_entry['local_obs_prepare_latency_ms'] = (obs_prepare_end_ns - obs_prepare_start_ns) / 1e6
        observation_log_entry['local_protocol_pack_latency_ms'] = protocol_pack_latency_ms
        observation_log_entry['local_sendall_block_latency_ms'] = sendall_block_latency_ms
        if protocol_pack_latency_ms is not None:
            observation_log_entry['local_obs_prepare_and_pack_latency_ms'] = (
                observation_log_entry['local_obs_prepare_latency_ms'] + protocol_pack_latency_ms
            )
        if protocol_pack_latency_ms is not None and sendall_block_latency_ms is not None:
            observation_log_entry['local_obs_prepare_pack_and_sendall_latency_ms'] = (
                observation_log_entry['local_obs_prepare_latency_ms'] + protocol_pack_latency_ms + sendall_block_latency_ms
            )
        if not send_ok or client_obs_send_timestamp_ns is None:
            observation_log_entry['send_failed'] = True
            return None

        observation_log_entry['client_obs_send_timestamp_ns'] = int(client_obs_send_timestamp_ns)
        self._clear_pending_chunk_receipt_ack(pending_chunk_receipt_ack)
        self.observations_sent += 1
        print(
            f"[客户端][Async] 发送观测 #{self.observations_sent} "
            f"(obs_seq={int(obs_seq)}, cycle={int(cycle_id)}, mode={self.execution_mode})"
        )
        return {
            'obs_seq': int(obs_seq),
            'send_timestamp_ns': int(client_obs_send_timestamp_ns),
            'cycle_id': int(cycle_id),
        }

    def _apply_pending_async_responses(self, scheduler: AsyncActionChunkScheduler):
        for response in self._drain_action_responses():
            obs_seq = response.get('obs_seq')
            if obs_seq is None:
                scheduler_event = {
                    'obs_seq': None,
                    'chunk_id': response.get('chunk_id'),
                    'applied': False,
                    'dropped_reason': 'missing_obs_seq',
                }
                self.trajectory_log['scheduler'].append(scheduler_event)
                print("[客户端][Async] 丢弃无 obs_seq 的动作响应")
                continue

            integration = scheduler.integrate_response(
                obs_seq=int(obs_seq),
                chunk=response['action'],
                recv_timestamp_ns=int(response['received_timestamp_ns']),
                chunk_id=response.get('chunk_id'),
            )
            scheduler_event = {
                'obs_seq': int(integration.obs_seq),
                'chunk_id': integration.chunk_id,
                'send_step_index': integration.send_step_index,
                'elapsed_steps': integration.elapsed_steps,
                'actual_delay_steps': integration.actual_delay_steps,
                'remaining_prefix_steps': integration.remaining_prefix_steps,
                'deadline_overrun_steps': integration.deadline_overrun_steps,
                'delay_estimate_steps': integration.delay_estimate_steps,
                'applied': bool(integration.applied),
                'dropped_reason': integration.dropped_reason,
                'pending_request_count': scheduler.pending_request_count,
            }
            self.trajectory_log['scheduler'].append(scheduler_event)

            if integration.applied:
                print(
                    f"[客户端][Async] 合并 obs_seq={integration.obs_seq}, "
                    f"chunk_id={integration.chunk_id}, delay={integration.actual_delay_steps}, "
                    f"remaining_prefix={integration.remaining_prefix_steps}, "
                    f"delay_est={integration.delay_estimate_steps}"
                )
            else:
                print(
                    f"[客户端][Async] 丢弃 obs_seq={integration.obs_seq}, "
                    f"chunk_id={integration.chunk_id}, reason={integration.dropped_reason}"
                )

    def _execute_async_action_step(self, scheduler: AsyncActionChunkScheduler):
        action_step = scheduler.current_action()
        source_obs_seq, source_chunk_id = scheduler.current_action_source()
        current_time = self._get_relative_time()
        if self.next_action_timestamp is None:
            target_time = current_time
        else:
            target_time = max(float(self.next_action_timestamp), current_time)

        self._wait_until_action_time(target_time, scheduler)
        if not self.running:
            return

        single_action = np.asarray(action_step, dtype=np.float32).reshape(-1)
        if len(single_action) == 8:
            pose_7d = single_action[:7]
            gripper_1d = single_action[7:8]
        elif len(single_action) == 7:
            pose_7d = single_action
            gripper_1d = np.array([1.0], dtype=np.float32)
        else:
            print(f"[客户端][Async] 警告: 动作维度不匹配: {len(single_action)}，保持当前目标")
            scheduler.advance(1)
            self.iter_idx += 1
            self.next_action_timestamp = float(target_time) + self.inference_interval
            return

        target_pos, target_quat, target_gripper_open = DPFormatConverter.dp_to_polymetis_action(pose_7d, gripper_1d)

        current_pos, current_quat = self.robot.get_ee_pose()
        if current_pos is not None and current_quat is not None:
            current_pos_np = current_pos.cpu().numpy()
            current_quat_np = current_quat.cpu().numpy()
            target_pos, target_quat, _ok = self._project_to_feasible(
                current_pos_np,
                current_quat_np,
                target_pos,
                target_quat,
            )

        with self.target_lock:
            self.current_target_pos = target_pos
            self.current_target_quat = target_quat

        if self.gripper is not None and target_gripper_open != self.last_gripper_state:
            action_name = "打开" if target_gripper_open else "关闭"
            print(f"[客户端] 夹爪{action_name}...")
            try:
                if target_gripper_open:
                    self.gripper.goto(
                        width=0.09,
                        speed=0.3,
                        force=1.0,
                        blocking=True,
                    )
                else:
                    self.gripper.grasp(
                        speed=0.2,
                        force=1.0,
                        grasp_width=0.0,
                        epsilon_inner=0.1,
                        epsilon_outer=0.1,
                        blocking=True,
                    )
                actual_width = self.gripper.get_state().width
                print(f"✓ 夹爪已{action_name} (实际宽度: {actual_width:.4f}m)")
                self.last_gripper_state = target_gripper_open
            except Exception as e:
                print(f"✗ 夹爪控制失败: {e}")

        self.gripper_open = target_gripper_open
        self.last_action_output = (target_pos.copy(), target_quat.copy())
        self.trajectory_log['executed'].append({
            'step': int(scheduler.executed_steps),
            'cycle_id': int(scheduler.executed_steps // self.execute_horizon),
            'source_obs_seq': source_obs_seq,
            'source_chunk_id': source_chunk_id,
            'delay_estimate_steps': scheduler.delay_estimate_steps,
            'pending_request_count': scheduler.pending_request_count,
            'pos': target_pos.tolist(),
            'quat': target_quat.tolist(),
            'gripper_open': target_gripper_open,
            'timestamp': float(target_time),
        })
        print(
            f"[客户端][Async] 执行动作 step={scheduler.executed_steps} "
            f"source_obs_seq={source_obs_seq} source_chunk_id={source_chunk_id}"
        )

        scheduler.advance(1)
        self.iter_idx += 1
        self.next_action_timestamp = float(target_time) + self.inference_interval

    def _run_naive_async_loop(self):
        print(f"[客户端] 推理循环已启动")
        print(f"  基础频率: {self.inference_freq:.1f}Hz (dt={self.inference_interval:.3f}s)")
        print(
            f"  调度模式: {self.execution_mode}, execute_horizon={self.execute_horizon}, "
            f"delay_alpha={self.delay_estimate_alpha}, init_steps={self.delay_estimate_init_steps}"
        )
        print("[客户端] 等待观测缓冲区填充...")
        time.sleep(1.0)

        initial_obs = None
        while self.running and initial_obs is None:
            initial_obs = self.obs_buffer.get_aligned_obs()
            if initial_obs is None:
                time.sleep(0.01)
        if not self.running:
            return

        initial_request = self._send_observation_request(initial_obs, cycle_id=-1)
        if initial_request is None:
            print("[客户端][Async] 初始观测发送失败")
            return

        initial_response = self._wait_for_action_response(
            initial_request['obs_seq'],
            timeout=max(5.0, self.inference_interval * 20.0),
        )
        if initial_response is None:
            print(
                f"[客户端][Async] 等待初始 action chunk 超时, obs_seq={initial_request['obs_seq']}"
            )
            return

        initial_chunk = np.asarray(initial_response['action'], dtype=np.float32)
        if initial_chunk.ndim == 1:
            initial_chunk = initial_chunk.reshape(1, -1)
        action_horizon = int(initial_chunk.shape[0])
        action_dim = int(initial_chunk.shape[1])
        scheduler = AsyncActionChunkScheduler(
            action_horizon=action_horizon,
            action_dim=action_dim,
            execute_horizon=self.execute_horizon,
            dt_exec=self.inference_interval,
            delay_estimate_alpha=self.delay_estimate_alpha,
            delay_estimate_init_steps=self.delay_estimate_init_steps,
            max_deadline_overrun_steps=self.max_deadline_overrun_steps,
        )
        scheduler.set_initial_chunk(
            initial_chunk,
            obs_seq=initial_response.get('obs_seq'),
            chunk_id=initial_response.get('chunk_id'),
        )
        self.trajectory_log['scheduler'].append({
            'event': 'initial_chunk',
            'obs_seq': initial_response.get('obs_seq'),
            'chunk_id': initial_response.get('chunk_id'),
            'action_horizon': action_horizon,
            'execute_horizon': self.execute_horizon,
        })
        self.next_action_timestamp = None
        print(
            f"[客户端][Async] ✓ 初始 chunk 就绪: obs_seq={initial_response.get('obs_seq')}, "
            f"chunk_id={initial_response.get('chunk_id')}, horizon={action_horizon}"
        )

        while self.running:
            if scheduler.executed_steps % self.execute_horizon == 0:
                cycle_id = scheduler.executed_steps // self.execute_horizon
                aligned_obs = self.obs_buffer.get_aligned_obs()
                if aligned_obs is not None:
                    request_info = self._send_observation_request(
                        aligned_obs,
                        cycle_id=int(cycle_id),
                        scheduler=scheduler,
                    )
                    if request_info is not None:
                        scheduler.register_request(
                            obs_seq=request_info['obs_seq'],
                            send_timestamp_ns=request_info['send_timestamp_ns'],
                        )
                else:
                    print(
                        f"[客户端][Async] 警告: cycle={cycle_id} 缺少对齐观测，继续执行已承诺 chunk"
                    )

            self._apply_pending_async_responses(scheduler)
            self._execute_async_action_step(scheduler)

    def _arm_arrival_detection(self, chunk_id: Optional[int]):
        with self.arrival_state_lock:
            self.arrival_target_chunk_id = chunk_id
            self.latest_arrival_pos_delta = None
            self.latest_arrival_stable_count = 0
            self.arrival_last_sample_pos = None
            self.arrival_armed = True
        self.arrival_event.clear()

    def _arrival_monitor_loop(self):
        interval = 1.0 / max(self.arrival_check_freq, 1.0)

        while self.running:
            try:
                with self.arrival_state_lock:
                    if not self.arrival_armed:
                        target_chunk_id = None
                    else:
                        target_chunk_id = self.arrival_target_chunk_id

                if target_chunk_id is None:
                    time.sleep(interval)
                    continue

                current_pos, _ = self.robot.get_ee_pose()
                if current_pos is None:
                    time.sleep(interval)
                    continue

                current_pos_np = current_pos.cpu().numpy()
                pos_delta = None
                stable_count = 0
                reached = False

                with self.arrival_state_lock:
                    if self.arrival_armed and self.arrival_target_chunk_id == target_chunk_id:
                        if self.arrival_last_sample_pos is not None:
                            pos_delta = float(np.linalg.norm(current_pos_np - self.arrival_last_sample_pos))

                        if pos_delta is not None and pos_delta <= self.arrival_settle_pos_delta_threshold:
                            self.latest_arrival_stable_count += 1
                        else:
                            self.latest_arrival_stable_count = 0

                        self.latest_arrival_pos_delta = pos_delta
                        self.arrival_last_sample_pos = current_pos_np.copy()
                        stable_count = self.latest_arrival_stable_count

                        if stable_count >= self.arrival_settle_stable_count:
                            self.arrival_armed = False
                            reached = True

                if reached:
                    self.arrival_event.set()
                    delta_text = "N/A" if pos_delta is None else f"{pos_delta:.4f}m"
                    print(
                        f"[客户端] 到达检测通过: chunk_id={target_chunk_id}, "
                        f"pos_delta={delta_text}, stable={stable_count}/{self.arrival_settle_stable_count}"
                    )
                else:
                    time.sleep(interval)
            except Exception as e:
                if self.running:
                    print(f"[客户端] 到达检测错误: {e}")
                time.sleep(interval)

    def _get_pending_chunk_count(self):
        with self.data_lock:
            return len(self.pending_action_chunks)

    def _pop_pending_action_chunk(self):
        with self.data_lock:
            if not self.pending_action_chunks:
                return None
            return self.pending_action_chunks.popleft()

    def _control_loop(self):
        """控制循环"""
        print(f"[控制线程] 已启动，频率: {self.execution_freq:.1f} Hz")
        rate = 1.0 / self.execution_freq
        
        while self.running:
            try:
                loop_start = time.monotonic()
                
                with self.target_lock:
                    if self.current_target_pos is not None and self.current_target_quat is not None:
                        target_pos = torch.from_numpy(self.current_target_pos).float()
                        target_quat = torch.from_numpy(self.current_target_quat).float()
                        
                        # Use robot_api_lock to prevent conflict with IK check in inference loop
                        with self.robot_api_lock:
                            self.robot.update_desired_ee_pose(position=target_pos, orientation=target_quat)
                
                elapsed = time.monotonic() - loop_start
                sleep_time = rate - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                if not self.running:
                    break
                print(f"[控制线程] 错误: {e}")
                time.sleep(0.1)

    def _collect_observations(self):
        """收集观测数据"""
        print(f"[客户端] 观测收集线程已启动 ({self.collect_obs_freq:.1f}Hz)")
        
        while self.running:
            try:
                current_time = self._get_relative_time()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                if ee_pos is not None and ee_quat is not None:
                    ee_pos_np = ee_pos.cpu().numpy()
                    ee_quat_np = ee_quat.cpu().numpy()
                    pose_7d, gripper_1d = DPFormatConverter.polymetis_to_dp_state(ee_pos_np, ee_quat_np, self.gripper_open)
                    self.obs_buffer.add_state(pose_7d, gripper_1d, current_time)
                
                time.sleep(1.0 / self.collect_obs_freq)
            except Exception as e:
                if self.running:
                    print(f"[客户端] 观测收集错误: {e}")

    def _collect_images(self):
        print(f"[客户端] 图像采集线程已启动 ({len(self.camera_names)} cameras)")

        while self.running:
            try:
                frames_dict = self.cameras.read_all_frames(parallel=True)
                current_time = self._get_relative_time()
                for cam_idx, cam_name in enumerate(self.camera_names):
                    frame_data = frames_dict.get(cam_name, None)
                    if frame_data is not None and frame_data.get('color') is not None:
                        self.obs_buffer.add_image(frame_data['color'], current_time, camera_idx=cam_idx)
            except Exception as e:
                if self.running:
                    print(f"[客户端] 图像采集错误: {e}")

    def _update_clock_offset_from_heartbeat_ack(self, data: Dict, client_recv_timestamp_ns: int):
        client_send_timestamp_ns = data.get('client_heartbeat_send_timestamp_ns')
        server_recv_timestamp_ns = data.get('server_heartbeat_recv_timestamp_ns')
        server_send_timestamp_ns = data.get('server_heartbeat_send_timestamp_ns')
        if client_send_timestamp_ns is None or server_recv_timestamp_ns is None or server_send_timestamp_ns is None:
            return
        t1 = int(client_send_timestamp_ns)
        t2 = int(server_recv_timestamp_ns)
        t3 = int(server_send_timestamp_ns)
        t4 = int(client_recv_timestamp_ns)
        rtt_ns = int((t4 - t1) - (t3 - t2))
        if rtt_ns < 0:
            return
        self.latest_heartbeat_rtt_ms = float(rtt_ns) / 1e6
        heartbeat_seq = int(data.get('heartbeat_seq', -1))
        if heartbeat_seq % 5 == 0:
            self.trajectory_log['heartbeat'].append({
                'type': 'heartbeat_ack',
                'heartbeat_seq': heartbeat_seq,
                'client_recv_timestamp_ns': int(t4),
                'rtt_ms': self.latest_heartbeat_rtt_ms,
            })
        now_mono = time.monotonic()
        if (now_mono - self.last_heartbeat_log_monotonic) >= self.heartbeat_log_interval_sec:
            self.last_heartbeat_log_monotonic = now_mono
            print(
                f"[客户端] 心跳保活: seq={int(data.get('heartbeat_seq', -1))}, "
                f"rtt={self.latest_heartbeat_rtt_ms:.2f}ms"
            )

    def _heartbeat_loop(self):
        if not self.enable_heartbeat:
            return
        print(f"[客户端] 心跳线程已启动 ({1.0 / self.heartbeat_interval_sec:.2f}Hz)")
        while self.running:
            try:
                ok, _heartbeat_send_timestamp_ns, _heartbeat_send_block_latency_ms = self.client.send_heartbeat(
                    heartbeat_seq=self.next_heartbeat_seq,
                    client_clock_offset_ns=None,
                )
                if ok:
                    self.next_heartbeat_seq += 1
                time.sleep(self.heartbeat_interval_sec)
            except Exception as e:
                if self.running:
                    print(f"[客户端] 心跳线程错误: {e}")
                time.sleep(self.heartbeat_interval_sec)

    def _ik_feasible(self, target_pos_np: np.ndarray, target_quat_np: np.ndarray) -> bool:
        """检查目标 EE 位姿是否 IK 可行。"""
        try:
            with self.robot_api_lock:
                joint_pos_current = self.robot.get_joint_positions()
                pos = torch.from_numpy(np.asarray(target_pos_np, dtype=np.float32)).float()
                quat = torch.from_numpy(np.asarray(target_quat_np, dtype=np.float32)).float()
                _, success = self.robot.solve_inverse_kinematics(pos, quat, joint_pos_current, tol=self.ik_tol)
            return bool(success)
        except Exception:
            # 如果 IK 接口不可用/异常，则不阻塞控制（返回 True）
            return True

    def _project_to_feasible(self, cur_pos: np.ndarray, cur_quat: np.ndarray,
                             des_pos: np.ndarray, des_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """若 IK 不可行，沿 cur->des 回退搜索到可行点。返回 (pos, quat, ok)。"""
        des_pos = np.asarray(des_pos, dtype=np.float32).reshape(3)
        des_quat, _ = _normalize_quat_xyzw(des_quat)
        des_quat = _quat_ensure_continuity_xyzw(des_quat, cur_quat)

        if not self.enable_ik_projection or self.ik_line_search_steps <= 0:
            return des_pos, des_quat, True

        if self._ik_feasible(des_pos, des_quat):
            return des_pos, des_quat, True

        # 逐步回退（更靠近当前位姿）
        cand_pos = des_pos.copy()
        cand_quat = des_quat.copy()
        for _ in range(self.ik_line_search_steps):
            cand_pos = (0.5 * (cur_pos + cand_pos)).astype(np.float32)
            cand_quat = _quat_slerp_xyzw(cur_quat, cand_quat, 0.5)
            if self._ik_feasible(cand_pos, cand_quat):
                return cand_pos, cand_quat, True

        return cur_pos.astype(np.float32), cur_quat.astype(np.float32), False

    def _receive_loop(self):
        """接收动作循环"""
        while self.running:
            try:
                data = self.client.recv_data(timeout=1.0)
                if data:
                    if data.get('type') == 'heartbeat_ack':
                        received_timestamp_ns = data.get('_socket_recv_timestamp_ns')
                        if received_timestamp_ns is None:
                            received_timestamp_ns = time.time_ns()
                        self._update_clock_offset_from_heartbeat_ack(data, int(received_timestamp_ns))
                        continue
                    # 支持两种类型：'action' 和 'action_sequence'
                    if data.get('type') == 'action':
                        action = np.array(data.get('action'), dtype=np.float32)
                        received_time = self._get_relative_time()
                        received_timestamp_ns = data.get('_socket_recv_timestamp_ns')
                        if received_timestamp_ns is None:
                            received_timestamp_ns = time.time_ns()
                        raw_chunk_id = data.get('chunk_id')
                        server_chunk_send_timestamp_ns = data.get('server_chunk_send_timestamp_ns')
                        recv_payload_bytes = data.get('_recv_payload_bytes')
                        downlink_latency_ms = None
                        if server_chunk_send_timestamp_ns is not None and self.clock_offset_ns is not None:
                            downlink_latency_ms = (
                                int(received_timestamp_ns) + int(self.clock_offset_ns) - int(server_chunk_send_timestamp_ns)
                            ) / 1e6

                        chunk_id = int(raw_chunk_id) if raw_chunk_id is not None else self.actions_received
                        response_entry = {
                            'chunk_id': chunk_id,
                            'obs_seq': None if data.get('obs_seq') is None else int(data.get('obs_seq')),
                            'action': action,
                            'received_time': received_time,
                            'received_timestamp_ns': int(received_timestamp_ns),
                            'server_chunk_send_timestamp_ns': None if server_chunk_send_timestamp_ns is None else int(server_chunk_send_timestamp_ns),
                            'downlink_latency_ms': downlink_latency_ms,
                            'recv_payload_bytes': None if recv_payload_bytes is None else int(recv_payload_bytes),
                        }

                        with self.data_lock:
                            if self._uses_async_execution():
                                self.pending_action_responses.append(dict(response_entry))
                                self.action_response_event.set()
                            else:
                                self.pending_action_chunks.append(dict(response_entry))
                            self.actions_received += 1
                        self._set_pending_chunk_receipt_ack(chunk_id, received_timestamp_ns)
                        self.trajectory_log['actions'].append({
                            'step': chunk_id,
                            'chunk_id': chunk_id,
                            'obs_seq': None if data.get('obs_seq') is None else int(data.get('obs_seq')),
                            'action': action.tolist(),
                            'shape': list(action.shape),
                            'received_time': received_time,
                            'received_timestamp_ns': int(received_timestamp_ns),
                            'server_chunk_send_timestamp_ns': None if server_chunk_send_timestamp_ns is None else int(server_chunk_send_timestamp_ns),
                            'downlink_latency_ms': downlink_latency_ms,
                            'recv_payload_bytes': None if recv_payload_bytes is None else int(recv_payload_bytes)
                        })
                        if not self._uses_async_execution():
                            self.action_received.set()
                    elif data.get('type') == 'action_sequence':
                        # 接收动作序列
                        actions = np.array(data.get('actions'), dtype=np.float32)
                        received_time = self._get_relative_time()
                        received_timestamp_ns = data.get('_socket_recv_timestamp_ns')
                        if received_timestamp_ns is None:
                            received_timestamp_ns = time.time_ns()
                        raw_chunk_id = data.get('chunk_id')
                        server_chunk_send_timestamp_ns = data.get('server_chunk_send_timestamp_ns')
                        recv_payload_bytes = data.get('_recv_payload_bytes')
                        downlink_latency_ms = None
                        if server_chunk_send_timestamp_ns is not None and self.clock_offset_ns is not None:
                            downlink_latency_ms = (
                                int(received_timestamp_ns) + int(self.clock_offset_ns) - int(server_chunk_send_timestamp_ns)
                            ) / 1e6

                        chunk_id = int(raw_chunk_id) if raw_chunk_id is not None else self.actions_received
                        response_entry = {
                            'chunk_id': chunk_id,
                            'obs_seq': None if data.get('obs_seq') is None else int(data.get('obs_seq')),
                            'action': actions,
                            'received_time': received_time,
                            'received_timestamp_ns': int(received_timestamp_ns),
                            'server_chunk_send_timestamp_ns': None if server_chunk_send_timestamp_ns is None else int(server_chunk_send_timestamp_ns),
                            'downlink_latency_ms': downlink_latency_ms,
                            'recv_payload_bytes': None if recv_payload_bytes is None else int(recv_payload_bytes),
                        }

                        with self.data_lock:
                            if self._uses_async_execution():
                                self.pending_action_responses.append(dict(response_entry))
                                self.action_response_event.set()
                            else:
                                self.pending_action_chunks.append(dict(response_entry))
                            self.actions_received += 1
                        self._set_pending_chunk_receipt_ack(chunk_id, received_timestamp_ns)
                        self.trajectory_log['actions'].append({
                            'step': chunk_id,
                            'chunk_id': chunk_id,
                            'obs_seq': None if data.get('obs_seq') is None else int(data.get('obs_seq')),
                            'actions': actions.tolist(),
                            'shape': list(actions.shape),
                            'received_time': received_time,
                            'received_timestamp_ns': int(received_timestamp_ns),
                            'server_chunk_send_timestamp_ns': None if server_chunk_send_timestamp_ns is None else int(server_chunk_send_timestamp_ns),
                            'downlink_latency_ms': downlink_latency_ms,
                            'recv_payload_bytes': None if recv_payload_bytes is None else int(recv_payload_bytes)
                        })
                        if not self._uses_async_execution():
                            self.action_received.set()
                        print(f"[客户端] 收到动作序列: chunk_id={chunk_id}, shape={actions.shape}")
            except Exception as e:
                if self.running:
                    print(f"[客户端] 接收错误: {e}")

    def _inference_loop(self):
        """推理循环"""
        if self._uses_async_execution():
            self._run_naive_async_loop()
            return

        dt = self.inference_interval
        print(f"[客户端] 推理循环已启动")
        print(f"  基础频率: {1/dt:.1f}Hz (dt={dt:.3f}s)")
        print(
            f"  调度模式: 状态驱动 (位置变化阈值 {self.arrival_settle_pos_delta_threshold:.4f}m, "
            f"连续 {self.arrival_settle_stable_count} 次, "
            f"检测频率 {self.arrival_check_freq:.1f}Hz)"
        )
        if self.arrival_max_wait_sec is not None:
            print(
                f"  到达兜底: 最长等待 {self.arrival_max_wait_sec:.2f}s, "
                f"日志间隔 {self.arrival_log_interval_sec:.2f}s"
            )
        else:
            print(f"  到达日志间隔: {self.arrival_log_interval_sec:.2f}s")
        
        # 等待观测缓冲区填充
        print("[客户端] 等待观测缓冲区填充...")
        time.sleep(1.0)
        # start_delay = 0.5
        # time.sleep(start_delay)
        self.next_action_timestamp = None
        self.arrival_event.set()
        print("[客户端] ✓ 开始推理控制")
        
        while self.running:
            try:
                if self.observations_sent > 0:
                    wait_start = time.monotonic()
                    next_wait_log_time = wait_start + self.arrival_log_interval_sec
                    while self.running and not self.arrival_event.wait(timeout=0.05):
                        now = time.monotonic()

                        if now >= next_wait_log_time:
                            with self.arrival_state_lock:
                                waiting_chunk_id = self.arrival_target_chunk_id
                                latest_arrival_pos_delta = self.latest_arrival_pos_delta
                                latest_arrival_stable_count = self.latest_arrival_stable_count
                            waited_sec = now - wait_start
                            if latest_arrival_pos_delta is None:
                                print(
                                    f"[客户端] 等待到达中: chunk_id={waiting_chunk_id}, "
                                    f"已等待 {waited_sec:.2f}s, 尚无位置变化量"
                                )
                            else:
                                print(
                                    f"[客户端] 等待到达中: chunk_id={waiting_chunk_id}, "
                                    f"已等待 {waited_sec:.2f}s, pos_delta={latest_arrival_pos_delta:.4f}m, "
                                    f"stable={latest_arrival_stable_count}/{self.arrival_settle_stable_count}, "
                                    f"threshold={self.arrival_settle_pos_delta_threshold:.4f}m"
                                )
                            next_wait_log_time = now + self.arrival_log_interval_sec

                        if self.arrival_max_wait_sec is not None and (now - wait_start) >= self.arrival_max_wait_sec:
                            with self.arrival_state_lock:
                                timeout_chunk_id = self.arrival_target_chunk_id
                                timeout_arrival_pos_delta = self.latest_arrival_pos_delta
                                timeout_arrival_stable_count = self.latest_arrival_stable_count
                                self.arrival_armed = False
                            self.arrival_event.set()
                            if timeout_arrival_pos_delta is None:
                                print(
                                    f"[客户端] 警告: chunk_id={timeout_chunk_id} 到达等待超时 "
                                    f"({self.arrival_max_wait_sec:.2f}s)，强制继续下一轮"
                                )
                            else:
                                print(
                                    f"[客户端] 警告: chunk_id={timeout_chunk_id} 到达等待超时 "
                                    f"({self.arrival_max_wait_sec:.2f}s), pos_delta={timeout_arrival_pos_delta:.4f}m, "
                                    f"stable={timeout_arrival_stable_count}/{self.arrival_settle_stable_count}，强制继续下一轮"
                                )
                            break
                    if not self.running:
                        break

                aligned_obs = self.obs_buffer.get_aligned_obs()
                if aligned_obs is None:
                    time.sleep(0.01)
                    continue
                
                # 记录观测
                last_pose = aligned_obs['pose']
                last_gripper = aligned_obs['gripper']
                observation_log_entry = {
                    'step': self.observations_sent,
                    'pose': last_pose.tolist(),
                    'gripper': last_gripper.tolist(),
                    'timestamp': aligned_obs['timestamp']
                }
                self.trajectory_log['observations'].append(observation_log_entry)
                
                latest_obs_images = aligned_obs['images']
                obs_prepare_start_ns = time.time_ns()
                
                images_payload = {}
                for cam_idx in range(latest_obs_images.shape[0]):
                    img = latest_obs_images[cam_idx]
                    cam_name = self.camera_names[cam_idx] if cam_idx < len(self.camera_names) else f"camera_{cam_idx}"
                    images_payload[cam_name] = _encode_transport_image(img, self.image_quality)
                
                pending_chunk_receipt_ack = self._get_pending_chunk_receipt_ack()
                obs_msg = {
                    'type': 'observation',
                    'images': images_payload,
                    'poses': [np.asarray(aligned_obs['pose'], dtype=np.float32).tolist()],
                    'grippers': [np.asarray(aligned_obs['gripper'], dtype=np.float32).tolist()],
                    'timestamps': [float(aligned_obs['timestamp'])]
                }
                if pending_chunk_receipt_ack is not None:
                    obs_msg['client_last_chunk_id'] = int(pending_chunk_receipt_ack['chunk_id'])
                    obs_msg['client_last_chunk_recv_timestamp_ns'] = int(pending_chunk_receipt_ack['client_chunk_recv_timestamp_ns'])
                
                obs_prepare_end_ns = time.time_ns()
                send_ok, client_obs_send_timestamp_ns, protocol_pack_latency_ms, sendall_block_latency_ms = self.client.send_data_with_timestamp(
                    obs_msg,
                    timestamp_ns_field='client_obs_send_timestamp_ns',
                    timestamp_s_field='send_timestamp',
                )
                observation_log_entry['local_obs_prepare_latency_ms'] = (obs_prepare_end_ns - obs_prepare_start_ns) / 1e6
                observation_log_entry['local_protocol_pack_latency_ms'] = protocol_pack_latency_ms
                observation_log_entry['local_sendall_block_latency_ms'] = sendall_block_latency_ms
                if protocol_pack_latency_ms is not None:
                    observation_log_entry['local_obs_prepare_and_pack_latency_ms'] = observation_log_entry['local_obs_prepare_latency_ms'] + protocol_pack_latency_ms
                if protocol_pack_latency_ms is not None and sendall_block_latency_ms is not None:
                    observation_log_entry['local_obs_prepare_pack_and_sendall_latency_ms'] = (
                        observation_log_entry['local_obs_prepare_latency_ms'] + protocol_pack_latency_ms + sendall_block_latency_ms
                    )
                if not send_ok or client_obs_send_timestamp_ns is None:
                    time.sleep(0.05)
                    continue

                observation_log_entry['client_obs_send_timestamp_ns'] = int(client_obs_send_timestamp_ns)

                self._clear_pending_chunk_receipt_ack(pending_chunk_receipt_ack)

                self.observations_sent += 1
                print(f"[客户端] 发送观测 #{self.observations_sent} (iter={self.iter_idx})")

                while self.running and self._get_pending_chunk_count() == 0:
                    if self.action_received.wait(timeout=0.05):
                        self.action_received.clear()
                if not self.running:
                    break

                action_chunk = self._pop_pending_action_chunk()
                if action_chunk is not None:
                    chunk_receive_time = action_chunk['received_time']
                    action = action_chunk['action']
                    action_chunk_id = action_chunk['chunk_id']
                    
                    if action is not None:
                        if action.ndim == 1:
                            action_sequence = [action]
                        else:
                            action_sequence = action
 
                        chunk_len = len(action_sequence)
                        chunk_horizon_sec = chunk_len * dt
                        current_time = self._get_relative_time()
                        if self.next_action_timestamp is None:
                            action_start_time = current_time
                        else:
                            action_start_time = max(float(self.next_action_timestamp), current_time)
                        action_timestamps = (
                            np.arange(len(action_sequence), dtype=np.float64) * dt
                            + float(action_start_time)
                        )
                        
                        action_sequence_filtered = action_sequence
                        action_timestamps_filtered = action_timestamps
                        
                        action_sequence_final, num_backtrack_filtered = self._filter_backtracking_actions(
                            action_sequence_filtered, verbose=True
                        )
                        
                        if num_backtrack_filtered > 0:
                            self.filtered_backtrack_count += num_backtrack_filtered
                        
                        if len(action_sequence_final) == 0:
                            if self.filtered_backtrack_count % 10 == 0:
                                print(f"[过滤] 已过滤 {self.filtered_backtrack_count} 个回退点")
                        else:
                            action_timestamps_final = action_timestamps_filtered[num_backtrack_filtered:]

                            skipped_chunk_count = None
                            if action_chunk_id is not None and self.last_executed_chunk_id is not None:
                                skipped_chunk_count = action_chunk_id - self.last_executed_chunk_id - 1
                                if skipped_chunk_count > 0:
                                    print(
                                        f"[客户端] 警告: 执行 chunk_id={action_chunk_id} 前跳过了 {skipped_chunk_count} 个 chunk "
                                        f"(last_executed={self.last_executed_chunk_id})"
                                    )
                            
                            print(
                                f"[客户端] 执行 chunk_id={action_chunk_id}: 收到 {len(action_sequence)} 个动作，"
                                f"移除 {num_backtrack_filtered} 个回退点，执行 {len(action_sequence_final)} 个 "
                                f"(chunk时长 {chunk_horizon_sec:.3f}s)"
                            )
                            
                            for step_idx, single_action in enumerate(action_sequence_final):
                                if isinstance(single_action, dict):
                                    pose_7d = np.array(single_action['pose'])
                                    gripper_1d = np.array(single_action['gripper'])
                                else:
                                    single_action = np.array(single_action).flatten()
                                    if len(single_action) == 8:
                                        pose_7d = single_action[:7]
                                        gripper_1d = single_action[7:8]
                                    elif len(single_action) == 7:
                                        pose_7d = single_action
                                        gripper_1d = np.array([1.0])
                                    else:
                                        print(f"[客户端] 警告: 动作维度不匹配: {len(single_action)}")
                                        continue
                                
                                target_pos, target_quat, target_gripper_open = DPFormatConverter.dp_to_polymetis_action(pose_7d, gripper_1d)

                                target_time = action_timestamps_final[step_idx]
                                target_mono = self.eval_t_start + float(target_time)
                                wait_time = target_mono - time.monotonic()
                                if wait_time > 0:
                                    time.sleep(wait_time)
                                
                                current_pos, current_quat = self.robot.get_ee_pose()
                                if current_pos is None or current_quat is None:
                                    continue
                                
                                current_pos_np = current_pos.cpu().numpy()
                                scaled_target_pos = target_pos
                                scaled_target_quat = target_quat
                                
                                current_quat_np = current_quat.cpu().numpy()
                                scaled_target_pos, scaled_target_quat, ok = self._project_to_feasible(
                                    current_pos_np, current_quat_np,
                                    scaled_target_pos, scaled_target_quat
                                )
                                
                                with self.target_lock:
                                    self.current_target_pos = scaled_target_pos
                                    self.current_target_quat = scaled_target_quat
                                
                                if self.gripper is not None and target_gripper_open != self.last_gripper_state:
                                    action_name = "打开" if target_gripper_open else "关闭"
                                    print(f"[客户端] 夹爪{action_name}...")
                                    try:
                                        if target_gripper_open:
                                            self.gripper.goto(
                                                width=0.09,
                                                speed=0.3,
                                                force=1.0,
                                                blocking=True
                                            )
                                        else:
                                            self.gripper.grasp(
                                                speed=0.2,
                                                force=1.0,
                                                grasp_width=0.0,
                                                epsilon_inner=0.1,
                                                epsilon_outer=0.1,
                                                blocking=True
                                            )
                                        actual_width = self.gripper.get_state().width
                                        print(f"✓ 夹爪已{action_name} (实际宽度: {actual_width:.4f}m)")
                                        self.last_gripper_state = target_gripper_open
                                    except Exception as e:
                                        print(f"✗ 夹爪控制失败: {e}")
                                
                                self.gripper_open = target_gripper_open
                                
                                self.trajectory_log['executed'].append({
                                    'step': self.actions_received,
                                    'chunk_id': action_chunk_id,
                                    'action_step': step_idx,
                                    'pos': scaled_target_pos.tolist(),
                                    'quat': scaled_target_quat.tolist(),
                                    'gripper_open': target_gripper_open,
                                    'timestamp': action_timestamps_final[step_idx],
                                    'chunk_receive_time': chunk_receive_time,
                                    'skipped_chunk_count': skipped_chunk_count
                                })
                                
                                print(
                                    f"[客户端] 执行 chunk_id={action_chunk_id} 动作 #{self.actions_received} "
                                    f"步 {step_idx+1}/{len(action_sequence_final)}"
                                )
                                
                                if step_idx == len(action_sequence_final) - 1:
                                    self._arm_arrival_detection(action_chunk_id)
                                    self.last_action_output = (target_pos.copy(), target_quat.copy())
                                    self.last_executed_chunk_id = action_chunk_id
                                    self.next_action_timestamp = float(action_timestamps_final[step_idx]) + dt
                            
                            self.iter_idx += len(action_sequence_final)
                else:
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"[客户端] 推理循环错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _save_trajectory_log(self):
        """保存轨迹日志"""
        try:
            from datetime import datetime
            
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                    return obj.item()
                else:
                    return obj
            
            serializable_log = convert_to_serializable(self.trajectory_log)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存到 log 目录
            log_dir = _path_setup.get_log_dir()
            log_file = log_dir / f"trajectory_log_{timestamp}.json"
            
            with open(log_file, 'w') as f:
                json.dump(serializable_log, f, indent=2)
            
            print(f"[客户端] 轨迹日志已保存: {log_file}")
        except Exception as e:
            print(f"[客户端] 保存轨迹日志失败: {e}")

    def stop(self):
        """停止客户端"""
        self.running = False
        self.arrival_event.set()
        self.action_received.set()
        self.action_response_event.set()
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        if self.arrival_monitor_thread and self.arrival_monitor_thread.is_alive():
            self.arrival_monitor_thread.join(timeout=2.0)
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)
        
        # 停止相机
        if self.cameras is not None:
            self.cameras.stop_all()
            self.cameras.print_all_stats()
        
        self.client.close()
        self.client.stop_tunnel()
        
        print(f"\n[客户端] ✓ 已停止")
        print(f"[客户端] 总共发送 {self.observations_sent} 个观测")
        print(f"[客户端] 总共接收 {self.actions_received} 个动作")
        print(f"[客户端] 回退点过滤: {self.filtered_backtrack_count} 个 (阈值: {self.backtrack_threshold*1000:.1f}mm)")
        print(f"[客户端] 轨迹连续性: 移除了chunk中的回退点，保留前进点")
        
        self._save_trajectory_log()


def main():
    parser = argparse.ArgumentParser(description='Polymetis 推理客户端')
    parser.add_argument('--mode', '-m', type=str, choices=['local', 'ssh'], default='local',
                        help='连接模式: local (本地直连) 或 ssh (SSH隧道)')
    args = parser.parse_args()
    
    client = PolymetisInferenceClient(mode=args.mode)
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[客户端] 检测到 Ctrl+C，正在停止...")
        client.stop()


if __name__ == "__main__":
    main()
