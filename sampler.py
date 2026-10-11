"""Euler generation with angular guidance and historical half-space projection."""

import jax
import jax.numpy as jnp

from config import (
    ACTION_HORIZON,
    DT,
    MODEL_ACTION_DIM,
    MOTION_ACTION_DIM,
    ODE_STEPS,
    PATH_DIM,
    PHYSICAL_ACTION_DIM,
    SEAM_GUIDED_WINDOW,
    SEAM_STRENGTH,
)


def solve(velocity, noise, reference, curvature_metric, history_length):
    """Mask cumulative paths and both metric axes; L=0 is the base solve."""
    if (
        noise.shape != (1, ACTION_HORIZON, MODEL_ACTION_DIM)
        or reference.shape != (1, SEAM_GUIDED_WINDOW, MODEL_ACTION_DIM)
        or curvature_metric.shape != (1, PATH_DIM, PATH_DIM)
    ):
        raise ValueError("Unexpected solver input shapes")
    mask = (jnp.arange(SEAM_GUIDED_WINDOW) < history_length).astype(noise.dtype)[None, :, None]
    length = jnp.maximum(history_length, 1).astype(noise.dtype)
    history_path = jnp.cumsum(reference[:, :, :MOTION_ACTION_DIM] * mask, axis=1) * mask
    covector = jnp.flip(jnp.cumsum(jnp.flip(history_path, axis=1), axis=1), axis=1) * mask
    covector_norm = jnp.sum(covector**2, axis=(1, 2))

    def step(index, carry):
        state, time, clean_log, candidate_log, guided_log, input_log = carry
        velocity_value = velocity(state, time)
        clean = state - time * velocity_value
        candidate = state + DT * velocity_value
        post_time = time + DT
        path = jnp.cumsum(candidate[:, :SEAM_GUIDED_WINDOW, :MOTION_ACTION_DIM], axis=1) * mask
        target = (1.0 - post_time) * history_path
        current_energy = jnp.sum(path**2, axis=(1, 2)) / (length * MOTION_ACTION_DIM)
        target_energy = jnp.sum(target**2, axis=(1, 2)) / (length * MOTION_ACTION_DIM)
        ratio = jnp.sqrt(target_energy / jnp.where(current_energy > 0, current_energy, 1.0))
        gradient = (
            jnp.where(
                (current_energy > 0)[:, None, None], ratio[:, None, None] * path - target, -target
            )
            * mask
        )
        weighted = (
            jnp.einsum("bij,bj->bi", curvature_metric, gradient.reshape(1, PATH_DIM)).reshape(
                1, SEAM_GUIDED_WINDOW, MOTION_ACTION_DIM
            )
            * mask
        )
        pullback = jnp.flip(jnp.cumsum(jnp.flip(weighted, axis=1), axis=1), axis=1) * mask
        angular_motion = SEAM_STRENGTH * (1.0 - post_time) * (-2.0 / length) * pullback
        angular = (
            jnp.zeros_like(candidate)
            .at[:, :SEAM_GUIDED_WINDOW, :MOTION_ACTION_DIM]
            .set(angular_motion)
        )
        proximal_input = clean + (time / -DT) * angular
        proximal_path = (
            jnp.cumsum(proximal_input[:, :SEAM_GUIDED_WINDOW, :MOTION_ACTION_DIM], axis=1) * mask
        )
        inner = jnp.sum(proximal_path * history_path, axis=(1, 2))
        coefficient = jnp.where(
            (inner < 0) & (covector_norm > 0),
            -inner / jnp.where(covector_norm > 0, covector_norm, 1.0),
            0.0,
        )
        projection = (-DT / time) * coefficient[:, None, None] * covector
        guided = (
            (candidate + angular).at[:, :SEAM_GUIDED_WINDOW, :MOTION_ACTION_DIM].add(projection)
        )
        return (
            guided,
            post_time,
            clean_log.at[index].set(clean[:, :, :PHYSICAL_ACTION_DIM]),
            candidate_log.at[index].set(candidate[:, :SEAM_GUIDED_WINDOW, :PHYSICAL_ACTION_DIM]),
            guided_log.at[index].set(guided[:, :SEAM_GUIDED_WINDOW, :PHYSICAL_ACTION_DIM]),
            input_log.at[index].set(proximal_input[:, :SEAM_GUIDED_WINDOW, :PHYSICAL_ACTION_DIM]),
        )

    prefix_shape = (ODE_STEPS, 1, SEAM_GUIDED_WINDOW, PHYSICAL_ACTION_DIM)
    values = jax.lax.fori_loop(
        0,
        ODE_STEPS,
        step,
        (
            noise,
            jnp.asarray(1.0, noise.dtype),
            jnp.zeros((ODE_STEPS, 1, ACTION_HORIZON, PHYSICAL_ACTION_DIM), noise.dtype),
            jnp.zeros(prefix_shape, noise.dtype),
            jnp.zeros(prefix_shape, noise.dtype),
            jnp.zeros(prefix_shape, noise.dtype),
        ),
    )
    return dict(
        actions=values[0],
        clean_estimates=jnp.swapaxes(values[2], 0, 1),
        candidate_prefix=jnp.swapaxes(values[3], 0, 1),
        guided_prefix=jnp.swapaxes(values[4], 0, 1),
        proximal_input_prefix=jnp.swapaxes(values[5], 0, 1),
    )
