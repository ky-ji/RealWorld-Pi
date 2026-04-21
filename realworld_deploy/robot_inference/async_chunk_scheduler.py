from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from typing import Optional

import numpy as np


def _shift_hold_last(array: np.ndarray, steps: int) -> np.ndarray:
    arr = np.asarray(array).copy()
    if arr.ndim == 0:
        raise ValueError("array must have at least 1 dimension")
    if arr.shape[0] == 0:
        raise ValueError("array must have a non-empty leading dimension")

    shift_steps = max(0, int(steps))
    if shift_steps == 0:
        return arr

    if shift_steps >= arr.shape[0]:
        if arr.ndim == 1:
            return np.full(arr.shape, arr[-1], dtype=arr.dtype)
        return np.repeat(arr[-1:, ...], arr.shape[0], axis=0)

    if arr.ndim == 1:
        tail = np.full((shift_steps,), arr[-1], dtype=arr.dtype)
    else:
        tail = np.repeat(arr[-1:, ...], shift_steps, axis=0)
    return np.concatenate([arr[shift_steps:], tail], axis=0)


@dataclasses.dataclass(frozen=True)
class InflightRequest:
    obs_seq: int
    send_timestamp_ns: int
    send_step_index: int


@dataclasses.dataclass(frozen=True)
class ResponseIntegration:
    obs_seq: int
    chunk_id: Optional[int]
    send_step_index: Optional[int]
    elapsed_steps: int
    actual_delay_steps: Optional[int]
    remaining_prefix_steps: Optional[int]
    deadline_overrun_steps: Optional[int]
    delay_estimate_steps: Optional[float]
    applied: bool
    dropped_reason: Optional[str] = None


class AsyncActionChunkScheduler:
    """Maintains a committed future action chunk for fixed-cycle async execution.

    The scheduler stores the currently committed future chunk aligned to "now".
    As each action step is executed, the chunk is shifted left by one and padded
    with the final target to guarantee continuous control even when inference is
    late. When a new action chunk arrives, we splice it into the current chunk
    using the naive-async rule from the plan:

    - preserve the prefix that is already guaranteed to execute
    - replace the remaining suffix with the newly predicted suffix
    """

    def __init__(
        self,
        *,
        action_horizon: int,
        action_dim: int,
        execute_horizon: int,
        dt_exec: float,
        delay_estimate_alpha: float = 0.5,
        delay_estimate_init_steps: int = 2,
        max_deadline_overrun_steps: int = 2,
    ):
        self.action_horizon = max(1, int(action_horizon))
        self.action_dim = max(1, int(action_dim))
        self.execute_horizon = max(1, int(execute_horizon))
        if self.execute_horizon > self.action_horizon:
            raise ValueError(
                f"execute_horizon ({self.execute_horizon}) must be <= action_horizon ({self.action_horizon})"
            )

        self.dt_exec = float(dt_exec)
        if self.dt_exec <= 0:
            raise ValueError(f"dt_exec must be > 0, got {self.dt_exec}")
        self.dt_exec_ns = max(1, int(round(self.dt_exec * 1e9)))
        self.delay_estimate_alpha = float(delay_estimate_alpha)
        self.delay_estimate_init_steps = max(1, int(delay_estimate_init_steps))
        self.max_deadline_overrun_steps = max(0, int(max_deadline_overrun_steps))

        self._chunk: Optional[np.ndarray] = None
        self._source_obs_seq: Optional[np.ndarray] = None
        self._source_chunk_id: Optional[np.ndarray] = None
        self._executed_steps = 0
        self._latest_applied_obs_seq: Optional[int] = None
        self._inflight_requests: "OrderedDict[int, InflightRequest]" = OrderedDict()
        self._delay_observation_count = 0
        self._delay_estimate_steps: Optional[float] = None

    @property
    def has_chunk(self) -> bool:
        return self._chunk is not None

    @property
    def executed_steps(self) -> int:
        return int(self._executed_steps)

    @property
    def pending_request_count(self) -> int:
        return len(self._inflight_requests)

    @property
    def delay_estimate_steps(self) -> Optional[float]:
        return None if self._delay_estimate_steps is None else float(self._delay_estimate_steps)

    @property
    def rounded_delay_estimate_steps(self) -> Optional[int]:
        if self._delay_estimate_steps is None:
            return None
        return int(round(self._delay_estimate_steps))

    def current_chunk(self) -> np.ndarray:
        if self._chunk is None:
            raise RuntimeError("scheduler has not been initialized with an action chunk")
        return self._chunk.copy()

    def current_action(self) -> np.ndarray:
        if self._chunk is None:
            raise RuntimeError("scheduler has not been initialized with an action chunk")
        return self._chunk[0].copy()

    def current_action_source(self) -> tuple[Optional[int], Optional[int]]:
        if self._source_obs_seq is None or self._source_chunk_id is None:
            return None, None
        obs_seq = int(self._source_obs_seq[0])
        chunk_id = int(self._source_chunk_id[0])
        return (None if obs_seq < 0 else obs_seq, None if chunk_id < 0 else chunk_id)

    def set_initial_chunk(self, chunk, *, obs_seq: Optional[int] = None, chunk_id: Optional[int] = None) -> np.ndarray:
        normalized_chunk = self._normalize_chunk(chunk)
        self._chunk = normalized_chunk
        source_obs_seq = -1 if obs_seq is None else int(obs_seq)
        source_chunk_id = -1 if chunk_id is None else int(chunk_id)
        self._source_obs_seq = np.full((self.action_horizon,), source_obs_seq, dtype=np.int32)
        self._source_chunk_id = np.full((self.action_horizon,), source_chunk_id, dtype=np.int32)
        self._latest_applied_obs_seq = None if obs_seq is None else int(obs_seq)
        return self.current_chunk()

    def register_request(self, obs_seq: int, send_timestamp_ns: int) -> InflightRequest:
        request = InflightRequest(
            obs_seq=int(obs_seq),
            send_timestamp_ns=int(send_timestamp_ns),
            send_step_index=int(self._executed_steps),
        )
        self._inflight_requests[int(obs_seq)] = request
        return request

    def advance(self, steps: int = 1) -> np.ndarray:
        if self._chunk is None or self._source_obs_seq is None or self._source_chunk_id is None:
            raise RuntimeError("scheduler has not been initialized with an action chunk")

        shift_steps = max(0, int(steps))
        if shift_steps == 0:
            return self.current_chunk()

        self._chunk = _shift_hold_last(self._chunk, shift_steps)
        self._source_obs_seq = _shift_hold_last(self._source_obs_seq, shift_steps)
        self._source_chunk_id = _shift_hold_last(self._source_chunk_id, shift_steps)
        self._executed_steps += shift_steps
        return self.current_chunk()

    def integrate_response(
        self,
        *,
        obs_seq: int,
        chunk,
        recv_timestamp_ns: int,
        chunk_id: Optional[int] = None,
    ) -> ResponseIntegration:
        request = self._inflight_requests.pop(int(obs_seq), None)
        if request is None:
            return ResponseIntegration(
                obs_seq=int(obs_seq),
                chunk_id=chunk_id,
                send_step_index=None,
                elapsed_steps=0,
                actual_delay_steps=None,
                remaining_prefix_steps=None,
                deadline_overrun_steps=None,
                delay_estimate_steps=self.delay_estimate_steps,
                applied=False,
                dropped_reason="unknown_obs_seq",
            )

        if self._chunk is None or self._source_obs_seq is None or self._source_chunk_id is None:
            raise RuntimeError("scheduler has not been initialized with an action chunk")

        if self._latest_applied_obs_seq is not None and int(obs_seq) <= int(self._latest_applied_obs_seq):
            return ResponseIntegration(
                obs_seq=int(obs_seq),
                chunk_id=chunk_id,
                send_step_index=int(request.send_step_index),
                elapsed_steps=max(0, int(self._executed_steps - request.send_step_index)),
                actual_delay_steps=None,
                remaining_prefix_steps=None,
                deadline_overrun_steps=None,
                delay_estimate_steps=self.delay_estimate_steps,
                applied=False,
                dropped_reason="stale_obs_seq",
            )

        normalized_chunk = self._normalize_chunk(chunk)
        elapsed_steps = max(0, int(self._executed_steps - request.send_step_index))
        actual_delay_steps = max(
            0,
            int(math.ceil(max(0, int(recv_timestamp_ns) - int(request.send_timestamp_ns)) / self.dt_exec_ns)),
        )
        remaining_prefix_steps = max(0, int(actual_delay_steps - elapsed_steps))
        deadline_overrun_steps = max(0, int(actual_delay_steps - self.execute_horizon))

        shifted_chunk = _shift_hold_last(normalized_chunk, elapsed_steps)
        shifted_source_obs_seq = np.full((self.action_horizon,), int(obs_seq), dtype=np.int32)
        shifted_source_chunk_id = np.full(
            (self.action_horizon,),
            -1 if chunk_id is None else int(chunk_id),
            dtype=np.int32,
        )
        shifted_source_obs_seq = _shift_hold_last(shifted_source_obs_seq, elapsed_steps)
        shifted_source_chunk_id = _shift_hold_last(shifted_source_chunk_id, elapsed_steps)

        if remaining_prefix_steps < self.action_horizon:
            self._chunk[remaining_prefix_steps:] = shifted_chunk[remaining_prefix_steps:]
            self._source_obs_seq[remaining_prefix_steps:] = shifted_source_obs_seq[remaining_prefix_steps:]
            self._source_chunk_id[remaining_prefix_steps:] = shifted_source_chunk_id[remaining_prefix_steps:]

        self._latest_applied_obs_seq = int(obs_seq)
        self._drop_inflight_requests_older_than(int(obs_seq))
        self._update_delay_estimate(actual_delay_steps)

        dropped_reason = None
        if deadline_overrun_steps > self.max_deadline_overrun_steps:
            dropped_reason = "deadline_overrun"

        return ResponseIntegration(
            obs_seq=int(obs_seq),
            chunk_id=chunk_id,
            send_step_index=int(request.send_step_index),
            elapsed_steps=elapsed_steps,
            actual_delay_steps=int(actual_delay_steps),
            remaining_prefix_steps=int(remaining_prefix_steps),
            deadline_overrun_steps=int(deadline_overrun_steps),
            delay_estimate_steps=self.delay_estimate_steps,
            applied=True,
            dropped_reason=dropped_reason,
        )

    def _normalize_chunk(self, chunk) -> np.ndarray:
        normalized = np.asarray(chunk, dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized[None, :]
        if normalized.ndim != 2:
            raise ValueError(f"chunk must be rank-2 after normalization, got shape {normalized.shape}")
        if normalized.shape[1] != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {normalized.shape[1]}")
        if normalized.shape[0] >= self.action_horizon:
            return normalized[: self.action_horizon].copy()
        pad = np.repeat(normalized[-1:, :], self.action_horizon - normalized.shape[0], axis=0)
        return np.concatenate([normalized, pad], axis=0)

    def _drop_inflight_requests_older_than(self, obs_seq: int):
        stale_keys = [key for key in self._inflight_requests if int(key) <= int(obs_seq)]
        for key in stale_keys:
            self._inflight_requests.pop(key, None)

    def _update_delay_estimate(self, observed_delay_steps: int):
        observed = float(observed_delay_steps)
        self._delay_observation_count += 1
        if self._delay_estimate_steps is None:
            self._delay_estimate_steps = observed
            return
        if self._delay_observation_count <= self.delay_estimate_init_steps:
            prev_count = self._delay_observation_count - 1
            self._delay_estimate_steps = (
                (self._delay_estimate_steps * prev_count) + observed
            ) / self._delay_observation_count
            return
        alpha = float(np.clip(self.delay_estimate_alpha, 0.0, 1.0))
        self._delay_estimate_steps = alpha * observed + (1.0 - alpha) * self._delay_estimate_steps
