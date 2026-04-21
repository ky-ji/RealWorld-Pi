"""Binary protocol constants used by `tcp_binary_protocol.py`.

This replaces the old `zmq_binary_protocol_config_example` dependency with a
real source module so the robot inference stack can run without a missing
example-only import.
"""

from __future__ import annotations

import struct

PROTOCOL_NAME = "zmq_rawbytes_v1"
PROTOCOL_VERSION = 1
PROTOCOL_ENDIAN = "<"

MSG_RESET = b"RST1"
MSG_RESET_ACK = b"RAK1"
MSG_OBSERVATION = b"OBS1"
MSG_ACTION = b"ACT1"
MSG_ERROR = b"ERR1"

TRANSPORT_MODE = "zmq_dual_channel"
ZMQ_OBSERVATION_PATTERN = "push_pull"
ZMQ_ACTION_PATTERN = "push_pull"
ZMQ_LINGER_MS = 1000
ZMQ_SNDTIMEO_MS = 2000
ZMQ_RCVTIMEO_MS = 2000
ZMQ_SNDHWM = 3
ZMQ_RCVHWM = 4

OBSERVATION_PORT = 8007
ACTION_PORT = 8008
LOCAL_OBSERVATION_PORT = 8007
LOCAL_ACTION_PORT = 8008

IMAGE_CODEC_JPEG = 1
IMAGE_CODEC_PNG = 2
IMAGE_CODEC_RAW_RGB = 3
DEFAULT_IMAGE_CODEC = IMAGE_CODEC_JPEG

DTYPE_FLOAT32 = 1
DTYPE_UINT8 = 2
STATE_DTYPE = "float32"
ACTION_DTYPE = "float32"
IMAGE_DTYPE = "uint8"

CAMERA_ID_FRONT = 1
CAMERA_ID_WRIST = 2
CAMERA_ID_LEFT = 3
CAMERA_ID_RIGHT = 4
CAMERA_ID_TOP = 5

CAMERA_NAME_TO_ID = {
    "front_view": CAMERA_ID_FRONT,
    "front": CAMERA_ID_FRONT,
    "wrist_view": CAMERA_ID_WRIST,
    "wrist": CAMERA_ID_WRIST,
    "ego_view": CAMERA_ID_WRIST,
    "left_view": CAMERA_ID_LEFT,
    "right_view": CAMERA_ID_RIGHT,
    "top_view": CAMERA_ID_TOP,
}
CAMERA_ID_TO_NAME = {
    CAMERA_ID_FRONT: "front_view",
    CAMERA_ID_WRIST: "wrist_view",
    CAMERA_ID_LEFT: "left_view",
    CAMERA_ID_RIGHT: "right_view",
    CAMERA_ID_TOP: "top_view",
}
DEFAULT_CAMERA_ORDER = (CAMERA_ID_FRONT, CAMERA_ID_WRIST)

STATE_DIM = 8
POSE_DIM = 7
GRIPPER_DIM = 1
ACTION_DIM = 8

OBSERVATION_HEADER_FORMAT = "HHQQQQqQHHHH"
ACTION_HEADER_FORMAT = "HHQQQQHHHHI"
RESET_HEADER_FORMAT = "HHQQHHHHI"
RESET_ACK_HEADER_FORMAT = "HHQQqII"
ERROR_HEADER_FORMAT = "HHQI"

OBSERVATION_HEADER = struct.Struct(PROTOCOL_ENDIAN + OBSERVATION_HEADER_FORMAT)
ACTION_HEADER = struct.Struct(PROTOCOL_ENDIAN + ACTION_HEADER_FORMAT)
RESET_HEADER = struct.Struct(PROTOCOL_ENDIAN + RESET_HEADER_FORMAT)
RESET_ACK_HEADER = struct.Struct(PROTOCOL_ENDIAN + RESET_ACK_HEADER_FORMAT)
ERROR_HEADER = struct.Struct(PROTOCOL_ENDIAN + ERROR_HEADER_FORMAT)

OBSERVATION_HEADER_FIELDS = (
    "version",
    "flags",
    "session_id",
    "obs_seq",
    "client_obs_send_timestamp_ns",
    "obs_capture_timestamp_ns",
    "acked_chunk_id",
    "acked_chunk_recv_timestamp_ns",
    "num_images",
    "image_codec",
    "state_dim",
    "dtype_code",
)
ACTION_HEADER_FIELDS = (
    "version",
    "flags",
    "session_id",
    "obs_seq",
    "chunk_id",
    "server_chunk_send_timestamp_ns",
    "n_steps",
    "action_dim",
    "dtype_code",
    "reserved",
    "infer_latency_us",
)
RESET_HEADER_FIELDS = (
    "version",
    "flags",
    "session_id",
    "client_reset_timestamp_ns",
    "num_cameras",
    "image_codec",
    "state_dim",
    "action_dim",
    "reserved",
)
RESET_ACK_HEADER_FIELDS = (
    "version",
    "flags",
    "session_id",
    "server_reset_ack_timestamp_ns",
    "clock_offset_ns",
    "status_code",
    "reserved",
)
ERROR_HEADER_FIELDS = (
    "version",
    "flags",
    "session_id",
    "status_code",
)

OBS_FRAME_TYPE_INDEX = 0
OBS_FRAME_HEADER_INDEX = 1
OBS_FRAME_STATE_INDEX = 2
OBS_FRAME_FIRST_IMAGE_INDEX = 3

ACTION_FRAME_TYPE_INDEX = 0
ACTION_FRAME_HEADER_INDEX = 1
ACTION_FRAME_PAYLOAD_INDEX = 2

RESET_FRAME_TYPE_INDEX = 0
RESET_FRAME_HEADER_INDEX = 1
RESET_FRAME_CAMERA_IDS_INDEX = 2

RESET_ACK_FRAME_TYPE_INDEX = 0
RESET_ACK_FRAME_HEADER_INDEX = 1

ERROR_FRAME_TYPE_INDEX = 0
ERROR_FRAME_HEADER_INDEX = 1
ERROR_FRAME_MESSAGE_INDEX = 2

STATUS_OK = 0
STATUS_BAD_VERSION = 1
STATUS_BAD_SESSION = 2
STATUS_BAD_MESSAGE = 3
STATUS_BAD_PAYLOAD = 4
STATUS_INTERNAL_ERROR = 5

FLAG_NONE = 0
FLAG_ACK_VALID = 1 << 0
FLAG_MORE_FRAMES = 1 << 1
FLAG_DROP_OLD = 1 << 2

STATE_BYTES = STATE_DIM * 4
ACTION_STEP_BYTES = ACTION_DIM * 4
CAMERA_ID_DTYPE = "uint16"


def header_size_map() -> dict[str, int]:
    return {
        "observation": OBSERVATION_HEADER.size,
        "action": ACTION_HEADER.size,
        "reset": RESET_HEADER.size,
        "reset_ack": RESET_ACK_HEADER.size,
        "error": ERROR_HEADER.size,
    }
