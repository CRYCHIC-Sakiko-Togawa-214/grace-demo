# GRACE

### Between Glances: Geometric Memory for Adaptive Action Chunking

[Project page](https://crychic-sakiko-togawa-214.github.io/grace-demo/) ·
[Code](https://github.com/CRYCHIC-Sakiko-Togawa-214/grace-demo)

**GRACE — Geometric Replanning with Adaptive Chunk Execution** combines geometric
memory from the preceding generation, history-guided denoising, and stability-based
action-prefix execution.

This repository contains the method core supplied with the paper's supplementary
material. The numerical kernels are extracted from the evaluated implementation;
the controller interface omits experiment management and ablation branches.

## Installation

Use Python 3.11. The method core requires only NumPy and JAX:

```bash
git clone --depth 1 https://github.com/CRYCHIC-Sakiko-Togawa-214/grace-demo.git
cd grace-demo
conda create -n grace python=3.11 -y
conda activate grace
pip install -r requirements.txt
python check.py
```

The pinned dependencies are `numpy==1.26.4` and `jax==0.5.3`. `check.py` runs on CPU
with an analytic velocity field; no policy checkpoint or simulator is needed.
It verifies empty-history Euler generation, continuation alignment, geometric
memory, and episode reset. The same check runs in CI.

## Connect a pretrained policy

Supply a JAX-compatible `velocity(x, t)` conditioned on the **current** observation
and instruction. Input and output shapes are `(1, 10, 32)`; `t` is a scalar flow
time descending from 1 to 0. The callback must expose the denoising field rather
than an already sampled action chunk.

```python
import numpy as np
from controller import GRACE

controller = GRACE()  # One controller per episode.
rng = np.random.default_rng(0)
noise = rng.standard_normal((10, 32), dtype=np.float32)

# velocity is your pretrained policy's current-observation velocity callback.
result = controller.plan(velocity, noise, executed=0)
k = result["execution_horizon"]
normalized_prefix = result["actions"][:k]
```

Noise, stored references, and returned actions use the policy's **normalized** action
coordinates. Coordinates 0–5 represent translation and rotation, coordinate 6 is
the gripper, and the remaining coordinates retain the pretrained policy's padding.
Apply the policy's output transforms before executing the selected prefix.

At the next observation, provide fresh noise and an updated velocity callback.
Pass the **actual** number of applied actions as `executed` (from 1 to the previously
selected horizon). Use zero only on the first call. Call `controller.reset()` at
every episode boundary.

## Configuration

The supplementary code uses the evaluated pi0.5-LIBERO settings in `config.py`:

| Setting | Value |
| --- | --- |
| Action horizon / model dimensions | `10 / 32` |
| Physical / motion dimensions | `7 / 6` |
| Euler steps | `10` |
| Maximum history support | `5` |
| Angular guidance strength | `0.1` |
| Stability tail / preceding score profiles | `5 / 5` |
| Threshold | Mean + `2` standard deviations |

Curvature uses normalized motion coordinates. Stability also includes the gripper.
The curvature state is replaced after each completed solve and used at the next
planning call.

## Benchmark integration

LIBERO evaluation additionally requires the pretrained **pi0.5-LIBERO** policy,
its observation and action transforms, and the LIBERO environment. The evaluated
upstream revisions recorded in the supplement are:

| Dependency | Revision |
| --- | --- |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |

The provided code is a controller core, not a complete benchmark runner. The archive
does not contain model weights, rollout data, training scripts, or evaluation logs.
The CPU sanity check validates method algebra; it does not reproduce benchmark scores.

## Code structure

```text
config.py       # Evaluated pi0.5-LIBERO settings
geometry.py     # Denoising curvature and bounded directional metric
sampler.py      # Euler solve, angular guidance, and half-space projection
execution.py    # Actual-execution alignment and adaptive prefix selection
controller.py   # Stateful GRACE planning cycle
check.py        # CPU sanity check without a model or simulator
requirements.txt
```

## Acknowledgements

GRACE integrates with the pretrained policy from [OpenPI](https://github.com/Physical-Intelligence/openpi)
and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark.
Publication metadata and a BibTeX entry will be added when the paper is publicly available.
