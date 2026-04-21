from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import tcp_binary_protocol as binary_proto


class TcpBinaryProtocolTest(unittest.TestCase):
    def test_legacy_observation_round_trip_without_rtc_extensions(self):
        payload, _ = binary_proto.encode_observation_message(
            data={
                "type": "observation",
                "images": {
                    "front_view": b"front-bytes",
                    "wrist_view": b"wrist-bytes",
                },
                "poses": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]],
                "grippers": [[1.0]],
                "timestamps": [1.25],
            },
            session_id=7,
            obs_seq=3,
            camera_names=["front_view", "wrist_view"],
            send_timestamp_ns=123456789,
        )
        frames = binary_proto.unpack_framed_payload(payload)
        decoded = binary_proto.decode_message(frames, camera_names=["front_view", "wrist_view"])

        self.assertEqual(decoded["type"], "observation")
        self.assertEqual(decoded["session_id"], 7)
        self.assertEqual(decoded["obs_seq"], 3)
        self.assertEqual(decoded["rtc_metadata"], None)
        self.assertEqual(decoded["rtc_action_prefix"], None)
        self.assertEqual(decoded["images"]["front_view"], b"front-bytes")
        self.assertEqual(decoded["images"]["wrist_view"], b"wrist-bytes")

    def test_observation_round_trip_with_rtc_metadata_and_prefix(self):
        prefix = np.arange(80, dtype=np.float32).reshape(10, 8)
        payload, _ = binary_proto.encode_observation_message(
            data={
                "type": "observation",
                "images": {
                    "front_view": b"front-bytes",
                    "wrist_view": b"wrist-bytes",
                },
                "poses": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]],
                "grippers": [[1.0]],
                "timestamps": [1.5],
                "rtc_metadata": {
                    "cycle_id": 11,
                    "execute_horizon": 4,
                    "delay_estimate_steps": 2,
                },
                "rtc_action_prefix": prefix,
            },
            session_id=9,
            obs_seq=5,
            camera_names=["front_view", "wrist_view"],
            send_timestamp_ns=333,
        )
        frames = binary_proto.unpack_framed_payload(payload)
        decoded = binary_proto.decode_message(frames, camera_names=["front_view", "wrist_view"])

        self.assertEqual(decoded["rtc_metadata"]["cycle_id"], 11)
        self.assertEqual(decoded["rtc_metadata"]["execute_horizon"], 4)
        self.assertEqual(decoded["rtc_metadata"]["delay_estimate_steps"], 2)
        np.testing.assert_allclose(np.asarray(decoded["rtc_action_prefix"], dtype=np.float32), prefix)

    def test_reset_heartbeat_and_action_messages_remain_compatible(self):
        reset_payload, _ = binary_proto.encode_reset_message(
            session_id=42,
            camera_names=["front_view", "wrist_view"],
            reset_timestamp_ns=100,
        )
        reset_frames = binary_proto.unpack_framed_payload(reset_payload)
        reset_decoded = binary_proto.decode_message(reset_frames)
        self.assertEqual(reset_decoded["type"], "reset")
        self.assertEqual(reset_decoded["session_id"], 42)

        heartbeat_payload, _ = binary_proto.encode_heartbeat_message(
            session_id=42,
            heartbeat_seq=7,
            send_timestamp_ns=200,
        )
        heartbeat_frames = binary_proto.unpack_framed_payload(heartbeat_payload)
        heartbeat_decoded = binary_proto.decode_message(heartbeat_frames)
        self.assertEqual(heartbeat_decoded["type"], "heartbeat")
        self.assertEqual(heartbeat_decoded["heartbeat_seq"], 7)

        action_payload, _ = binary_proto.encode_action_message(
            chunk_id=5,
            action=np.arange(80, dtype=np.float32).reshape(10, 8),
            session_id=42,
            obs_seq=8,
            infer_latency_us=1234,
            send_timestamp_ns=300,
        )
        action_frames = binary_proto.unpack_framed_payload(action_payload)
        action_decoded = binary_proto.decode_message(action_frames)
        self.assertEqual(action_decoded["type"], "action")
        self.assertEqual(action_decoded["chunk_id"], 5)
        self.assertEqual(action_decoded["obs_seq"], 8)
        self.assertEqual(np.asarray(action_decoded["action"], dtype=np.float32).shape, (10, 8))


if __name__ == "__main__":
    unittest.main()
