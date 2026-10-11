"""CPU sanity check using an analytic velocity field; no model or simulator."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from config import ACTION_HORIZON, DT, MODEL_ACTION_DIM, ODE_STEPS, SEAM_GUIDED_WINDOW
from controller import GRACE
from geometry import curvature_resolvent_metric


def velocity(state, time):
    return jnp.float32(0.2) * jnp.tanh(state) + jnp.float32(0.1) * time


def main():
    rng = np.random.default_rng(0)
    controller = GRACE()
    noise = rng.standard_normal((ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32)
    first = controller.plan(velocity, noise)
    assert first["history_length"] == 0
    assert 1 <= first["execution_horizon"] <= ACTION_HORIZON

    # Empty history reduces to ordinary Euler generation.
    baseline = jnp.asarray(noise)[None]
    time = jnp.asarray(1.0, dtype=jnp.float32)
    for _ in range(ODE_STEPS):
        baseline = baseline + DT * velocity(baseline, time)
        time = time + DT
    np.testing.assert_allclose(first["actions"], np.asarray(baseline)[0], atol=2e-6, rtol=2e-6)

    executed = min(3, first["execution_horizon"])
    second = controller.plan(
        velocity,
        rng.standard_normal(noise.shape, dtype=np.float32),
        executed=executed,
    )
    length = min(SEAM_GUIDED_WINDOW, ACTION_HORIZON - executed)
    assert second["history_length"] == length
    np.testing.assert_array_equal(
        second["normalized_reference"][:length], first["actions"][executed : executed + length]
    )
    np.testing.assert_allclose(
        second["curvature_metric"], curvature_resolvent_metric(first["curvature_increment"])
    )
    assert np.isfinite(second["actions"]).all()
    controller.reset()
    restarted = controller.plan(velocity, noise)
    np.testing.assert_array_equal(restarted["actions"], first["actions"])
    print(
        "PASS: empty-history Euler solve, aligned continuation, geometric memory, and episode reset."
    )


if __name__ == "__main__":
    main()
