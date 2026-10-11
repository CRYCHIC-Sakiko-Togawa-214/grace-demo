"""GRACE: geometric memory, guided generation, and adaptive execution."""

import jax
import jax.numpy as jnp
import numpy as np

from config import ACTION_HORIZON, MODEL_ACTION_DIM, PATH_DIM
from execution import ExecutionScheduler, aligned_reference, finite_array, integer
from geometry import curvature_resolvent_metric, denoising_curvature_information
from sampler import solve


class GRACE:
    """One controller per episode. All stored actions are model-normalized."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous_actions = None
        self._curvature = np.zeros((PATH_DIM, PATH_DIM), dtype=np.float32)
        self._last_selected = 0
        self._scheduler = ExecutionScheduler()

    def plan(self, velocity, noise, *, executed=0):
        """Generate a chunk from the current observation-conditioned velocity.

        velocity(x, t) receives a JAX array of shape (1, H, D) and a scalar
        flow time; it returns an array of the same shape. noise has shape
        (H, D). executed acknowledges how many actions of the preceding
        selected prefix were actually applied; use zero on the first call.
        """
        if self._previous_actions is None:
            integer(executed, "executed", 0, 0)
        else:
            integer(executed, "executed", 1, self._last_selected)
        noise = finite_array(noise, (ACTION_HORIZON, MODEL_ACTION_DIM), "noise")
        reference, length = aligned_reference(self._previous_actions, executed)
        metric = curvature_resolvent_metric(self._curvature)
        sampled = solve(
            velocity,
            jnp.asarray(noise)[None],
            jnp.asarray(reference)[None],
            jnp.asarray(metric)[None],
            jnp.asarray(length),
        )
        sampled = jax.tree.map(lambda value: np.asarray(jax.device_get(value))[0], sampled)
        actions = finite_array(sampled["actions"], (ACTION_HORIZON, MODEL_ACTION_DIM), "actions")
        decision = self._scheduler.decide(sampled["clean_estimates"])
        curvature = denoising_curvature_information(sampled["candidate_prefix"])

        # Replace geometry after generation; use only this solve at the next call.
        self._scheduler.commit(decision["variance_scores"])
        self._previous_actions = actions.copy()
        self._curvature = curvature.copy()
        self._last_selected = decision["execution_horizon"]
        return dict(
            **sampled,
            **decision,
            history_length=length,
            normalized_reference=reference,
            curvature_metric=metric,
            curvature_increment=curvature,
        )
