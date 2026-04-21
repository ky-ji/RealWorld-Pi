import base64
from decimal import Decimal, InvalidOperation
import os
import socket
import struct
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import zmq_binary_protocol_config as proto


TCP_MESSAGE_LENGTH = struct.Struct("<I")
TCP_FRAME_COUNT = struct.Struct("<I")
TCP_FRAME_LENGTH = struct.Struct("<I")
UINT64_FIELD = struct.Struct(proto.PROTOCOL_ENDIAN + "Q")
OBSERVATION_HEADER_CLIENT_SEND_TIMESTAMP_OFFSET = struct.calcsize(proto.PROTOCOL_ENDIAN + "HHQQ")
MSG_HEARTBEAT = b"HBT1"
MSG_HEARTBEAT_ACK = b"HAK1"
HEARTBEAT_HEADER = struct.Struct(proto.PROTOCOL_ENDIAN + "HHQQQq")
HEARTBEAT_ACK_HEADER = struct.Struct(proto.PROTOCOL_ENDIAN + "HHQQQQQ")
HEARTBEAT_HEADER_FIELDS = (
    "version",
    "flags",
    "session_id",
    "heartbeat_seq",
    "client_heartbeat_send_timestamp_ns",
    "client_clock_offset_ns",
)
HEARTBEAT_ACK_HEADER_FIELDS = (
    "version",
    "flags",
    "session_id",
    "heartbeat_seq",
    "client_heartbeat_send_timestamp_ns",
    "server_heartbeat_recv_timestamp_ns",
    "server_heartbeat_send_timestamp_ns",
)
FLAG_CLOCK_OFFSET_VALID = 1 << 3


def _protocol_numpy_byteorder() -> str:
    if proto.PROTOCOL_ENDIAN == "<":
        return "<"
    if proto.PROTOCOL_ENDIAN in (">", "!"):
        return ">"
    raise ValueError(f"unsupported PROTOCOL_ENDIAN={proto.PROTOCOL_ENDIAN!r}; use '<', '>', or '!' for an explicit protocol byte order")


PROTOCOL_NUMPY_BYTEORDER = _protocol_numpy_byteorder()
PROTOCOL_FLOAT32_DTYPE = np.dtype(PROTOCOL_NUMPY_BYTEORDER + "f4")
PROTOCOL_UINT16_DTYPE = np.dtype(PROTOCOL_NUMPY_BYTEORDER + "u2")
ENABLE_PROTOCOL_DEBUG = os.environ.get("GROOT_TCP_PROTOCOL_DEBUG", "").lower() in ("1", "true", "yes")


def _as_protocol_array(value, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=dtype), dtype=dtype)


def _timestamp_to_ns(value) -> int:
    if isinstance(value, (np.integer, int)):
        int_value = int(value)
        if abs(int_value) >= 1_000_000_000_000:
            return int_value
        return int_value * 1_000_000_000
    if isinstance(value, (np.floating, float)):
        decimal_value = Decimal(str(value))
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("timestamp string is empty")
        try:
            decimal_value = Decimal(stripped)
        except InvalidOperation as exc:
            raise ValueError(f"invalid timestamp value: {value!r}") from exc
        if all(ch not in stripped.lower() for ch in (".", "e")):
            int_value = int(decimal_value)
            if abs(int_value) >= 1_000_000_000_000:
                return int_value
    else:
        try:
            decimal_value = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"invalid timestamp value: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"invalid non-finite timestamp value: {value!r}")
    return int(decimal_value * Decimal("1000000000"))


def build_camera_ids(camera_names: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(proto.CAMERA_NAME_TO_ID[name]) for name in camera_names)


def camera_names_from_ids(camera_ids: Sequence[int]) -> List[str]:
    return [proto.CAMERA_ID_TO_NAME.get(int(camera_id), f"camera_{int(camera_id)}") for camera_id in camera_ids]


def _ensure_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError(f"unsupported bytes payload type: {type(value)!r}")


def _pack_frames(frames: Iterable[bytes]) -> bytes:
    normalized_frames = [_ensure_bytes(frame) for frame in frames]
    body_parts = [TCP_FRAME_COUNT.pack(len(normalized_frames))]
    for frame in normalized_frames:
        body_parts.append(TCP_FRAME_LENGTH.pack(len(frame)))
        body_parts.append(frame)
    body = b"".join(body_parts)
    return TCP_MESSAGE_LENGTH.pack(len(body)) + body


def _pack_frames_mutable(frames: Iterable[bytes]) -> Tuple[bytearray, List[int]]:
    normalized_frames = [_ensure_bytes(frame) for frame in frames]
    body_length = TCP_FRAME_COUNT.size
    for frame in normalized_frames:
        body_length += TCP_FRAME_LENGTH.size + len(frame)
    payload = bytearray(TCP_MESSAGE_LENGTH.size + body_length)
    TCP_MESSAGE_LENGTH.pack_into(payload, 0, body_length)
    offset = TCP_MESSAGE_LENGTH.size
    TCP_FRAME_COUNT.pack_into(payload, offset, len(normalized_frames))
    offset += TCP_FRAME_COUNT.size
    frame_offsets = []
    for frame in normalized_frames:
        TCP_FRAME_LENGTH.pack_into(payload, offset, len(frame))
        offset += TCP_FRAME_LENGTH.size
        frame_offsets.append(offset)
        payload[offset: offset + len(frame)] = frame
        offset += len(frame)
    return payload, frame_offsets


def _unpack_frames_from_body(body: bytes) -> List[bytes]:
    frame_count = TCP_FRAME_COUNT.unpack_from(body, 0)[0]
    offset = TCP_FRAME_COUNT.size
    frames = []
    for _ in range(frame_count):
        frame_length = TCP_FRAME_LENGTH.unpack_from(body, offset)[0]
        offset += TCP_FRAME_LENGTH.size
        frame = body[offset: offset + frame_length]
        if len(frame) != frame_length:
            raise ValueError("incomplete frame payload")
        frames.append(bytes(frame))
        offset += frame_length
    if offset != len(body):
        raise ValueError("unexpected trailing bytes in framed message")
    return frames


def unpack_framed_payload(payload: bytes) -> List[bytes]:
    payload_bytes = _ensure_bytes(payload)
    if len(payload_bytes) < TCP_MESSAGE_LENGTH.size:
        raise ValueError("framed payload too short")
    body_length = TCP_MESSAGE_LENGTH.unpack_from(payload_bytes, 0)[0]
    expected_total_length = TCP_MESSAGE_LENGTH.size + int(body_length)
    if len(payload_bytes) != expected_total_length:
        raise ValueError(
            f"framed payload length mismatch: expected {expected_total_length} bytes, got {len(payload_bytes)}"
        )
    body = payload_bytes[TCP_MESSAGE_LENGTH.size:]
    return _unpack_frames_from_body(body)


def _recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
    remaining = int(num_bytes)
    buf = bytearray(remaining)
    view = memoryview(buf)
    offset = 0
    while remaining > 0:
        recv_size = sock.recv_into(view[offset:], remaining)
        if recv_size == 0:
            return None
        offset += recv_size
        remaining -= recv_size
    return bytes(buf)


def _recv_exact_with_chunk_metrics(sock: socket.socket, num_bytes: int) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
    remaining = int(num_bytes)
    buf = bytearray(remaining)
    view = memoryview(buf)
    offset = 0
    first_chunk_timestamp_ns = None
    last_chunk_timestamp_ns = None
    chunk_sizes_bytes: List[int] = []
    chunk_intervals_ns: List[int] = []
    while remaining > 0:
        recv_size = sock.recv_into(view[offset:], remaining)
        if recv_size == 0:
            return None, None
        now_ns = time.time_ns()
        if first_chunk_timestamp_ns is None:
            first_chunk_timestamp_ns = now_ns
        elif last_chunk_timestamp_ns is not None:
            chunk_intervals_ns.append(int(now_ns - last_chunk_timestamp_ns))
        last_chunk_timestamp_ns = now_ns
        chunk_sizes_bytes.append(int(recv_size))
        offset += recv_size
        remaining -= recv_size
    metrics = {
        "first_chunk_timestamp_ns": int(first_chunk_timestamp_ns) if first_chunk_timestamp_ns is not None else None,
        "last_chunk_timestamp_ns": int(last_chunk_timestamp_ns) if last_chunk_timestamp_ns is not None else None,
        "chunk_count": len(chunk_sizes_bytes),
        "chunk_sizes_bytes": chunk_sizes_bytes,
        "chunk_intervals_ns": chunk_intervals_ns,
    }
    return bytes(buf), metrics


def _recv_exact_with_first_chunk_timestamp(sock: socket.socket, num_bytes: int) -> Tuple[Optional[bytes], Optional[int]]:
    remaining = int(num_bytes)
    buf = bytearray(remaining)
    view = memoryview(buf)
    offset = 0
    first_chunk_timestamp_ns = None
    while remaining > 0:
        recv_size = sock.recv_into(view[offset:], remaining)
        if recv_size == 0:
            return None, None
        if first_chunk_timestamp_ns is None:
            first_chunk_timestamp_ns = time.time_ns()
        offset += recv_size
        remaining -= recv_size
    return bytes(buf), first_chunk_timestamp_ns


def recv_framed_message(sock: socket.socket) -> Tuple[Optional[List[bytes]], Optional[int], Optional[Dict[str, Any]]]:
    header, recv_start_timestamp_ns = _recv_exact_with_first_chunk_timestamp(sock, TCP_MESSAGE_LENGTH.size)
    if header is None:
        return None, None, None
    header_recv_timestamp_ns = time.time_ns()
    body_length = TCP_MESSAGE_LENGTH.unpack(header)[0]
    body, body_recv_metrics = _recv_exact_with_chunk_metrics(sock, body_length)
    if body is None:
        return None, None, None
    if body_recv_metrics is None:
        recv_timestamp_ns = time.time_ns()
        body_recv_timestamp_ns = recv_timestamp_ns
        body_chunk_count = None
        body_chunk_sizes = None
        body_chunk_intervals_ns = None
    else:
        body_recv_timestamp_ns = body_recv_metrics.get("last_chunk_timestamp_ns")
        recv_timestamp_ns = int(body_recv_timestamp_ns) if body_recv_timestamp_ns is not None else time.time_ns()
        body_chunk_count = body_recv_metrics.get("chunk_count")
        body_chunk_sizes = body_recv_metrics.get("chunk_sizes_bytes")
        body_chunk_intervals_ns = body_recv_metrics.get("chunk_intervals_ns")
    recv_metadata = {
        "recv_start_timestamp_ns": int(recv_start_timestamp_ns),
        "header_recv_timestamp_ns": int(header_recv_timestamp_ns),
        "body_recv_timestamp_ns": int(body_recv_timestamp_ns) if body_recv_timestamp_ns is not None else int(recv_timestamp_ns),
        "body_length_bytes": int(body_length),
        "body_recv_chunk_count": body_chunk_count,
        "body_recv_chunk_sizes_bytes": body_chunk_sizes,
        "body_recv_chunk_intervals_ns": body_chunk_intervals_ns,
    }
    frames = _unpack_frames_from_body(body)
    return frames, recv_timestamp_ns, recv_metadata


def _coerce_image_bytes(image_value) -> bytes:
    if isinstance(image_value, str):
        return base64.b64decode(image_value)
    return _ensure_bytes(image_value)


def _build_state8(poses_list, grippers_list) -> np.ndarray:
    poses = np.asarray(poses_list, dtype=np.float32).reshape(-1, proto.POSE_DIM)
    grippers = np.asarray(grippers_list, dtype=np.float32).reshape(-1, proto.GRIPPER_DIM)
    last_pose7 = poses[-1]
    last_gripper1 = grippers[-1].reshape(-1)
    gripper_value = float(last_gripper1[0]) if last_gripper1.size > 0 else 1.0
    return _as_protocol_array(np.concatenate([last_pose7, np.array([gripper_value], dtype=np.float32)], axis=0), PROTOCOL_FLOAT32_DTYPE)


def encode_reset_message(session_id: int, camera_names: Sequence[str], reset_timestamp_ns: Optional[int] = None) -> Tuple[bytes, int]:
    send_timestamp_ns = int(reset_timestamp_ns if reset_timestamp_ns is not None else time.time_ns())
    camera_ids = _as_protocol_array(build_camera_ids(camera_names), PROTOCOL_UINT16_DTYPE)
    header = proto.RESET_HEADER.pack(
        proto.PROTOCOL_VERSION,
        proto.FLAG_NONE,
        int(session_id),
        send_timestamp_ns,
        len(camera_ids),
        proto.DEFAULT_IMAGE_CODEC,
        proto.STATE_DIM,
        proto.ACTION_DIM,
        0,
    )
    return _pack_frames([proto.MSG_RESET, header, camera_ids.tobytes()]), send_timestamp_ns


def encode_reset_ack_message(session_id: int, clock_offset_ns: int, send_timestamp_ns: Optional[int] = None) -> Tuple[bytes, int]:
    ack_timestamp_ns = int(send_timestamp_ns if send_timestamp_ns is not None else time.time_ns())
    header = proto.RESET_ACK_HEADER.pack(
        proto.PROTOCOL_VERSION,
        proto.FLAG_NONE,
        int(session_id),
        ack_timestamp_ns,
        int(clock_offset_ns),
        proto.STATUS_OK,
        0,
    )
    return _pack_frames([proto.MSG_RESET_ACK, header]), ack_timestamp_ns


def encode_heartbeat_message(
    session_id: int,
    heartbeat_seq: int,
    client_clock_offset_ns: Optional[int] = None,
    send_timestamp_ns: Optional[int] = None,
) -> Tuple[bytes, int]:
    heartbeat_send_timestamp_ns = int(send_timestamp_ns if send_timestamp_ns is not None else time.time_ns())
    flags = proto.FLAG_NONE
    client_clock_offset_value = 0
    if client_clock_offset_ns is not None:
        flags |= FLAG_CLOCK_OFFSET_VALID
        client_clock_offset_value = int(client_clock_offset_ns)
    header = HEARTBEAT_HEADER.pack(
        proto.PROTOCOL_VERSION,
        flags,
        int(session_id),
        int(heartbeat_seq),
        int(heartbeat_send_timestamp_ns),
        int(client_clock_offset_value),
    )
    return _pack_frames([MSG_HEARTBEAT, header]), heartbeat_send_timestamp_ns


def encode_heartbeat_ack_message(
    session_id: int,
    heartbeat_seq: int,
    client_heartbeat_send_timestamp_ns: int,
    server_heartbeat_recv_timestamp_ns: int,
    send_timestamp_ns: Optional[int] = None,
) -> Tuple[bytes, int]:
    server_heartbeat_send_timestamp_ns = int(send_timestamp_ns if send_timestamp_ns is not None else time.time_ns())
    header = HEARTBEAT_ACK_HEADER.pack(
        proto.PROTOCOL_VERSION,
        proto.FLAG_NONE,
        int(session_id),
        int(heartbeat_seq),
        int(client_heartbeat_send_timestamp_ns),
        int(server_heartbeat_recv_timestamp_ns),
        int(server_heartbeat_send_timestamp_ns),
    )
    return _pack_frames([MSG_HEARTBEAT_ACK, header]), server_heartbeat_send_timestamp_ns


def encode_observation_message(
    data: Dict,
    session_id: int,
    obs_seq: int,
    camera_names: Sequence[str],
    send_timestamp_ns: Optional[int] = None,
) -> Tuple[bytes, int]:
    payload, timestamp_offset = prepare_observation_message(
        data=data,
        session_id=session_id,
        obs_seq=obs_seq,
        camera_names=camera_names,
    )
    client_obs_send_timestamp_ns = int(send_timestamp_ns if send_timestamp_ns is not None else time.time_ns())
    set_observation_send_timestamp(payload, timestamp_offset, client_obs_send_timestamp_ns)
    return bytes(payload), client_obs_send_timestamp_ns


def prepare_observation_message(
    data: Dict,
    session_id: int,
    obs_seq: int,
    camera_names: Sequence[str],
) -> Tuple[bytearray, int]:
    images = data.get("images", {})
    poses_list = data.get("poses", [])
    grippers_list = data.get("grippers", [])
    timestamps = data.get("timestamps", [])
    if not poses_list or not grippers_list:
        raise ValueError("observation missing poses or grippers")
    image_frames = []
    for camera_name in camera_names:
        if camera_name not in images:
            raise KeyError(f"missing image for camera {camera_name}")
        image_frames.append(_coerce_image_bytes(images[camera_name]))
    state8 = _build_state8(poses_list, grippers_list)
    if timestamps:
        obs_capture_timestamp_ns = max(_timestamp_to_ns(ts) for ts in timestamps)
    else:
        obs_capture_timestamp_ns = int(time.time_ns())
    acked_chunk_id = data.get("client_last_chunk_id")
    acked_chunk_recv_timestamp_ns = data.get("client_last_chunk_recv_timestamp_ns")
    flags = proto.FLAG_NONE
    if acked_chunk_id is None or acked_chunk_recv_timestamp_ns is None:
        acked_chunk_id_value = -1
        acked_chunk_recv_timestamp_value = 0
    else:
        flags |= proto.FLAG_ACK_VALID
        acked_chunk_id_value = int(acked_chunk_id)
        acked_chunk_recv_timestamp_value = int(acked_chunk_recv_timestamp_ns)
    header = proto.OBSERVATION_HEADER.pack(
        proto.PROTOCOL_VERSION,
        flags,
        int(session_id),
        int(obs_seq),
        0,
        int(obs_capture_timestamp_ns),
        int(acked_chunk_id_value),
        int(acked_chunk_recv_timestamp_value),
        len(image_frames),
        proto.DEFAULT_IMAGE_CODEC,
        proto.STATE_DIM,
        proto.DTYPE_FLOAT32,
    )
    frames = [proto.MSG_OBSERVATION, header, _as_protocol_array(state8, PROTOCOL_FLOAT32_DTYPE).tobytes()]
    frames.extend(image_frames)
    payload, frame_offsets = _pack_frames_mutable(frames)
    header_offset = frame_offsets[proto.OBS_FRAME_HEADER_INDEX]
    timestamp_offset = header_offset + OBSERVATION_HEADER_CLIENT_SEND_TIMESTAMP_OFFSET
    image_frame_sizes = [len(frame) for frame in image_frames]
    if ENABLE_PROTOCOL_DEBUG:
        print(
            f"[rawbytes][observation] payload_bytes={len(payload)} "
            f"state_bytes={len(frames[proto.OBS_FRAME_STATE_INDEX])} "
            f"image_total_bytes={sum(image_frame_sizes)} "
            f"image_frame_bytes={image_frame_sizes}"
        )
    return payload, timestamp_offset


def set_observation_send_timestamp(payload: bytearray, timestamp_offset: int, send_timestamp_ns: int) -> int:
    UINT64_FIELD.pack_into(payload, timestamp_offset, int(send_timestamp_ns))
    return int(send_timestamp_ns)


def encode_action_message(
    chunk_id: int,
    action,
    session_id: int = 0,
    obs_seq: int = 0,
    infer_latency_us: int = 0,
    send_timestamp_ns: Optional[int] = None,
) -> Tuple[bytes, int]:
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.ndim == 1:
        action_array = action_array.reshape(1, -1)
    action_array = _as_protocol_array(action_array, PROTOCOL_FLOAT32_DTYPE)
    action_timestamp_ns = int(send_timestamp_ns if send_timestamp_ns is not None else time.time_ns())
    header = proto.ACTION_HEADER.pack(
        proto.PROTOCOL_VERSION,
        proto.FLAG_NONE,
        int(session_id),
        int(obs_seq),
        int(chunk_id),
        action_timestamp_ns,
        int(action_array.shape[0]),
        int(action_array.shape[1]),
        proto.DTYPE_FLOAT32,
        0,
        int(infer_latency_us),
    )
    return _pack_frames([proto.MSG_ACTION, header, action_array.tobytes()]), action_timestamp_ns


def encode_error_message(session_id: int, status_code: int, message: str) -> bytes:
    error_message = message.encode("utf-8", errors="replace")
    header = proto.ERROR_HEADER.pack(
        proto.PROTOCOL_VERSION,
        proto.FLAG_NONE,
        int(session_id),
        int(status_code),
    )
    return _pack_frames([proto.MSG_ERROR, header, error_message])


def decode_message(frames: Sequence[bytes], camera_names: Optional[Sequence[str]] = None) -> Dict:
    if not frames:
        raise ValueError("empty framed message")
    msg_type = frames[0]
    if msg_type == proto.MSG_RESET:
        if len(frames) < 3:
            raise ValueError("reset message too short")
        header_values = proto.RESET_HEADER.unpack(frames[proto.RESET_FRAME_HEADER_INDEX])
        header = dict(zip(proto.RESET_HEADER_FIELDS, header_values))
        camera_ids = np.frombuffer(frames[proto.RESET_FRAME_CAMERA_IDS_INDEX], dtype=PROTOCOL_UINT16_DTYPE).astype(np.int32).tolist()
        return {
            "type": "reset",
            "session_id": int(header["session_id"]),
            "camera_ids": camera_ids,
            "camera_names": camera_names_from_ids(camera_ids),
            "client_reset_timestamp_ns": int(header["client_reset_timestamp_ns"]),
            "_header": header,
        }
    if msg_type == proto.MSG_RESET_ACK:
        if len(frames) < 2:
            raise ValueError("reset ack message too short")
        header_values = proto.RESET_ACK_HEADER.unpack(frames[proto.RESET_ACK_FRAME_HEADER_INDEX])
        header = dict(zip(proto.RESET_ACK_HEADER_FIELDS, header_values))
        return {
            "type": "reset_ack",
            "session_id": int(header["session_id"]),
            "server_reset_ack_timestamp_ns": int(header["server_reset_ack_timestamp_ns"]),
            "clock_offset_ns": int(header["clock_offset_ns"]),
            "status_code": int(header["status_code"]),
            "_header": header,
        }
    if msg_type == MSG_HEARTBEAT:
        if len(frames) < 2:
            raise ValueError("heartbeat message too short")
        header_values = HEARTBEAT_HEADER.unpack(frames[1])
        header = dict(zip(HEARTBEAT_HEADER_FIELDS, header_values))
        flags = int(header["flags"])
        client_clock_offset_ns = int(header["client_clock_offset_ns"])
        return {
            "type": "heartbeat",
            "session_id": int(header["session_id"]),
            "heartbeat_seq": int(header["heartbeat_seq"]),
            "client_heartbeat_send_timestamp_ns": int(header["client_heartbeat_send_timestamp_ns"]),
            "client_clock_offset_ns": None if (flags & FLAG_CLOCK_OFFSET_VALID) == 0 else client_clock_offset_ns,
            "_header": header,
        }
    if msg_type == MSG_HEARTBEAT_ACK:
        if len(frames) < 2:
            raise ValueError("heartbeat ack message too short")
        header_values = HEARTBEAT_ACK_HEADER.unpack(frames[1])
        header = dict(zip(HEARTBEAT_ACK_HEADER_FIELDS, header_values))
        return {
            "type": "heartbeat_ack",
            "session_id": int(header["session_id"]),
            "heartbeat_seq": int(header["heartbeat_seq"]),
            "client_heartbeat_send_timestamp_ns": int(header["client_heartbeat_send_timestamp_ns"]),
            "server_heartbeat_recv_timestamp_ns": int(header["server_heartbeat_recv_timestamp_ns"]),
            "server_heartbeat_send_timestamp_ns": int(header["server_heartbeat_send_timestamp_ns"]),
            "_header": header,
        }
    if msg_type == proto.MSG_OBSERVATION:
        if len(frames) < 3:
            raise ValueError("observation message too short")
        header_values = proto.OBSERVATION_HEADER.unpack(frames[proto.OBS_FRAME_HEADER_INDEX])
        header = dict(zip(proto.OBSERVATION_HEADER_FIELDS, header_values))
        state8 = np.frombuffer(frames[proto.OBS_FRAME_STATE_INDEX], dtype=PROTOCOL_FLOAT32_DTYPE).reshape(proto.STATE_DIM).astype(np.float32, copy=True)
        inferred_camera_names = list(camera_names) if camera_names is not None else camera_names_from_ids(proto.DEFAULT_CAMERA_ORDER)
        image_frames = frames[proto.OBS_FRAME_FIRST_IMAGE_INDEX:]
        images = OrderedDict()
        for index, image_bytes in enumerate(image_frames):
            camera_name = inferred_camera_names[index] if index < len(inferred_camera_names) else f"camera_{index}"
            images[camera_name] = bytes(image_bytes)
        acked_chunk_id = int(header["acked_chunk_id"])
        acked_chunk_recv_timestamp_ns = int(header["acked_chunk_recv_timestamp_ns"])
        return {
            "type": "observation",
            "session_id": int(header["session_id"]),
            "obs_seq": int(header["obs_seq"]),
            "images": images,
            "poses": [state8[:proto.POSE_DIM].astype(np.float32).tolist()],
            "grippers": [[float(state8[proto.POSE_DIM])]],
            "timestamps": [float(header["obs_capture_timestamp_ns"]) / 1e9],
            "send_timestamp": float(header["client_obs_send_timestamp_ns"]) / 1e9,
            "client_obs_send_timestamp_ns": int(header["client_obs_send_timestamp_ns"]),
            "client_last_chunk_id": None if acked_chunk_id < 0 else acked_chunk_id,
            "client_last_chunk_recv_timestamp_ns": None if acked_chunk_recv_timestamp_ns <= 0 else acked_chunk_recv_timestamp_ns,
            "state8": state8.tolist(),
            "_header": header,
        }
    if msg_type == proto.MSG_ACTION:
        if len(frames) < 3:
            raise ValueError("action message too short")
        header_values = proto.ACTION_HEADER.unpack(frames[proto.ACTION_FRAME_HEADER_INDEX])
        header = dict(zip(proto.ACTION_HEADER_FIELDS, header_values))
        payload = np.frombuffer(frames[proto.ACTION_FRAME_PAYLOAD_INDEX], dtype=PROTOCOL_FLOAT32_DTYPE)
        action = payload.reshape(int(header["n_steps"]), int(header["action_dim"])).astype(np.float32, copy=True)
        return {
            "type": "action",
            "session_id": int(header["session_id"]),
            "obs_seq": int(header["obs_seq"]),
            "chunk_id": int(header["chunk_id"]),
            "action": action.tolist(),
            "server_chunk_send_timestamp_ns": int(header["server_chunk_send_timestamp_ns"]),
            "infer_latency_us": int(header["infer_latency_us"]),
            "_header": header,
        }
    if msg_type == proto.MSG_ERROR:
        if len(frames) < 2:
            raise ValueError("error message too short")
        header_values = proto.ERROR_HEADER.unpack(frames[proto.ERROR_FRAME_HEADER_INDEX])
        header = dict(zip(proto.ERROR_HEADER_FIELDS, header_values))
        message = frames[proto.ERROR_FRAME_MESSAGE_INDEX].decode("utf-8", errors="replace") if len(frames) > 2 else ""
        return {
            "type": "error",
            "session_id": int(header["session_id"]),
            "status_code": int(header["status_code"]),
            "message": message,
            "_header": header,
        }
    raise ValueError(f"unknown message type: {msg_type!r}")
