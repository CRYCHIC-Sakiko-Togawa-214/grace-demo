"""Actual-execution alignment and stability-based prefix selection."""

from collections import deque
from numbers import Integral

import numpy as np

from config import (
    ACTION_HORIZON,
    MODEL_ACTION_DIM,
    ODE_STEPS,
    PHYSICAL_ACTION_DIM,
    SEAM_GUIDED_WINDOW,
    STABILITY_HISTORY_CALLS,
    STABILITY_TAIL_NODES,
    THRESHOLD_MULTIPLIER,
)


def integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside [{low}, {high}]: {value}")
    return value


def finite_array(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite {name} with shape {shape}, got {value.shape}")
    return value


def aligned_reference(previous_actions, executed):
    """Use the first min(5, H-executed) unexecuted actions, zero-pad storage only."""
    reference = np.zeros((SEAM_GUIDED_WINDOW, MODEL_ACTION_DIM), dtype=np.float32)
    if previous_actions is None:
        integer(executed, "previous_executed", 0, 0)
        return reference, 0
    previous = finite_array(
        previous_actions, (ACTION_HORIZON, MODEL_ACTION_DIM), "previous_actions"
    )
    executed = integer(executed, "previous_executed", 1, ACTION_HORIZON)
    length = min(SEAM_GUIDED_WINDOW, ACTION_HORIZON - executed)
    reference[:length] = previous[executed : executed + length]
    return reference, length


def variance_scores(clean_estimates):
    clean = finite_array(
        clean_estimates, (ODE_STEPS, ACTION_HORIZON, PHYSICAL_ACTION_DIM), "clean_estimates"
    )
    tail = clean[-STABILITY_TAIL_NODES:]
    centered = tail - tail.mean(axis=0, dtype=np.float32, keepdims=True)
    return np.sum(np.mean(centered**2, axis=0, dtype=np.float32), axis=-1, dtype=np.float32)


class ExecutionScheduler:
    def __init__(self):
        self.history = deque(maxlen=STABILITY_HISTORY_CALLS)

    def decide(self, clean_estimates):
        """Use preceding score profiles, or current scores for the first call."""
        scores = variance_scores(clean_estimates)
        window = np.concatenate(list(self.history)) if self.history else scores
        threshold = float(
            window.mean(dtype=np.float64) + THRESHOLD_MULTIPLIER * window.std(dtype=np.float64)
        )
        crossing = np.flatnonzero(scores > threshold)
        selected = max(1, int(crossing[0])) if len(crossing) else ACTION_HORIZON
        return dict(
            execution_horizon=selected,
            variance_scores=scores,
            variance_threshold=threshold,
        )

    def commit(self, scores):
        self.history.append(finite_array(scores, (ACTION_HORIZON,), "scores").copy())
