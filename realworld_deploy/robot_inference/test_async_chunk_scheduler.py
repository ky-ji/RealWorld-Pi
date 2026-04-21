from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from async_chunk_scheduler import AsyncActionChunkScheduler


def _chunk(start: int, horizon: int = 10) -> np.ndarray:
    return np.arange(start, start + horizon, dtype=np.float32).reshape(horizon, 1)


def _shift_hold_last(chunk: np.ndarray, steps: int) -> np.ndarray:
    base = np.asarray(chunk, dtype=np.float32)
    if steps <= 0:
        return base.copy()
    if steps >= len(base):
        return np.repeat(base[-1:], len(base), axis=0)
    return np.concatenate([base[steps:], np.repeat(base[-1:], steps, axis=0)], axis=0)


class AsyncChunkSchedulerTest(unittest.TestCase):
    def _make_scheduler(self) -> AsyncActionChunkScheduler:
        return AsyncActionChunkScheduler(
            action_horizon=10,
            action_dim=1,
            execute_horizon=4,
            dt_exec=0.1,
        )

    def test_integrate_response_matches_shifted_future_for_prefix_delays(self):
        for delay_steps in range(4):
            with self.subTest(delay_steps=delay_steps):
                scheduler = self._make_scheduler()
                scheduler.set_initial_chunk(_chunk(0), obs_seq=0, chunk_id=0)
                scheduler.register_request(obs_seq=1, send_timestamp_ns=0)

                scheduler.advance(delay_steps)
                result = scheduler.integrate_response(
                    obs_seq=1,
                    chunk=_chunk(100),
                    recv_timestamp_ns=int(delay_steps * 0.1 * 1e9),
                    chunk_id=10,
                )

                self.assertTrue(result.applied)
                self.assertEqual(result.actual_delay_steps, delay_steps)
                expected = _shift_hold_last(_chunk(100), delay_steps)
                np.testing.assert_allclose(scheduler.current_chunk(), expected)

    def test_late_response_preserves_remaining_old_prefix_across_cycle_boundary(self):
        scheduler = self._make_scheduler()
        scheduler.set_initial_chunk(_chunk(0), obs_seq=0, chunk_id=0)
        scheduler.register_request(obs_seq=1, send_timestamp_ns=0)

        scheduler.advance(4)
        result = scheduler.integrate_response(
            obs_seq=1,
            chunk=_chunk(100),
            recv_timestamp_ns=int(0.5 * 1e9),
            chunk_id=10,
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.actual_delay_steps, 5)
        self.assertEqual(result.remaining_prefix_steps, 1)
        expected = np.array([4, 105, 106, 107, 108, 109, 109, 109, 109, 109], dtype=np.float32).reshape(10, 1)
        np.testing.assert_allclose(scheduler.current_chunk(), expected)

    def test_consecutive_late_responses_keep_committed_prefix_and_accept_newer_suffix(self):
        scheduler = self._make_scheduler()
        scheduler.set_initial_chunk(_chunk(0), obs_seq=0, chunk_id=0)

        scheduler.register_request(obs_seq=1, send_timestamp_ns=0)
        scheduler.advance(4)
        first = scheduler.integrate_response(
            obs_seq=1,
            chunk=_chunk(100),
            recv_timestamp_ns=int(0.5 * 1e9),
            chunk_id=10,
        )
        self.assertTrue(first.applied)

        scheduler.register_request(obs_seq=2, send_timestamp_ns=int(0.4 * 1e9))
        scheduler.advance(4)
        second = scheduler.integrate_response(
            obs_seq=2,
            chunk=_chunk(200),
            recv_timestamp_ns=int(1.0 * 1e9),
            chunk_id=20,
        )

        self.assertTrue(second.applied)
        self.assertEqual(second.actual_delay_steps, 6)
        self.assertEqual(second.remaining_prefix_steps, 2)
        expected = np.array([108, 109, 206, 207, 208, 209, 209, 209, 209, 209], dtype=np.float32).reshape(10, 1)
        np.testing.assert_allclose(scheduler.current_chunk(), expected)

    def test_shift_holds_last_target_when_no_fresh_response_arrives(self):
        scheduler = self._make_scheduler()
        scheduler.set_initial_chunk(_chunk(0), obs_seq=0, chunk_id=0)
        scheduler.advance(12)
        expected = np.full((10, 1), 9.0, dtype=np.float32)
        np.testing.assert_allclose(scheduler.current_chunk(), expected)

    def test_stale_response_is_dropped_after_newer_response_has_been_applied(self):
        scheduler = self._make_scheduler()
        scheduler.set_initial_chunk(_chunk(0), obs_seq=0, chunk_id=0)

        scheduler.register_request(obs_seq=1, send_timestamp_ns=0)
        scheduler.register_request(obs_seq=2, send_timestamp_ns=0)
        applied = scheduler.integrate_response(
            obs_seq=2,
            chunk=_chunk(200),
            recv_timestamp_ns=int(0.1 * 1e9),
            chunk_id=20,
        )
        dropped = scheduler.integrate_response(
            obs_seq=1,
            chunk=_chunk(100),
            recv_timestamp_ns=int(0.2 * 1e9),
            chunk_id=10,
        )

        self.assertTrue(applied.applied)
        self.assertFalse(dropped.applied)
        self.assertEqual(dropped.dropped_reason, "unknown_obs_seq")


if __name__ == "__main__":
    unittest.main()
