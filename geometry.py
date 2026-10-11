"""Denoising-path curvature and its bounded directional metric."""

import numpy as np

from config import DT, ODE_STEPS, PATH_DIM, PHYSICAL_ACTION_DIM, SEAM_ACTION_DIM, SEAM_GUIDED_WINDOW


def denoising_curvature_information(candidate_trace: np.ndarray) -> np.ndarray:
    """Return dimensionless path-curvature information from one Euler solve.

    Rows of the trace are causal Euler candidates.  In cumulative-action path
    coordinates, the numerator is discrete acceleration covariance and the
    denominator is discrete velocity energy.  The ``DT**2`` factor converts
    the finite differences to derivatives in flow time.
    """
    value = np.asarray(candidate_trace, dtype=np.float32)
    expected = (ODE_STEPS, SEAM_GUIDED_WINDOW, PHYSICAL_ACTION_DIM)
    if value.shape != expected:
        raise ValueError(f"Expected candidate trace {expected}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Candidate trace contains non-finite values")
    paths = np.cumsum(value[:, :, :SEAM_ACTION_DIM], axis=1, dtype=np.float32).reshape(
        ODE_STEPS, PATH_DIM
    )
    velocity_differences = np.diff(paths, axis=0)
    acceleration_differences = np.diff(paths, n=2, axis=0)
    denominator = float(
        np.float32(DT * DT) * np.sum(np.square(velocity_differences), dtype=np.float64)
    )
    if denominator <= 0.0:
        return np.zeros((PATH_DIM, PATH_DIM), dtype=np.float32)
    information = (acceleration_differences.T @ acceleration_differences) / np.float32(denominator)
    return np.asarray(0.5 * (information + information.T), dtype=np.float32)


def curvature_resolvent_metric(information: np.ndarray) -> np.ndarray:
    """Map the preceding generation's curvature F to F(I+F)^-1."""
    value = np.asarray(information, dtype=np.float32)
    expected = (PATH_DIM, PATH_DIM)
    if value.shape != expected:
        raise ValueError(f"Expected curvature information {expected}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Curvature information contains non-finite values")
    value = np.asarray(0.5 * (value + value.T), dtype=np.float32)
    metric = np.linalg.solve(np.eye(PATH_DIM, dtype=np.float32) + value, value)
    return np.asarray(0.5 * (metric + metric.T), dtype=np.float32)
