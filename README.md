# Between Glances — GRACE in action

**GRACE — Geometric Replanning with Adaptive Chunk Execution**

[![Repository checks](https://github.com/CRYCHIC-Sakiko-Togawa-214/grace-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/CRYCHIC-Sakiko-Togawa-214/grace-demo/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-blue)](https://crychic-sakiko-togawa-214.github.io/grace-demo/)

A static demonstration for **Between Glances: Geometric Memory for Adaptive Action Chunking**.
A robot reaches, grasps, and carries on between observations: geometric memory guides
movement, while stability determines when it looks again.

**[Live demo](https://crychic-sakiko-togawa-214.github.io/grace-demo/)** ·
**[Featured tasks](MAIN_SHOWCASE.md)** · **[More examples](SUPPLEMENTAL_SHOWCASE.md)**

## Quick start

Python 3.9+ and a modern browser are sufficient. No npm install, build step, robotics
simulator, model checkpoint, or inference service is needed.

```sh
git clone --depth 1 https://github.com/CRYCHIC-Sakiko-Togawa-214/grace-demo.git
cd grace-demo
python3 serve.py
```

Open <http://127.0.0.1:8765/>. Use `--port 8080` for another port. The local server supports
byte-range requests for video seeking and binds to loopback by default. The clone includes
large media files; use a sparse checkout for source-only maintenance.

## Explore the results

The catalog contains **20 examples across 10 tasks**, split into 10 featured and 10
supplemental cases. Every case provides an eight-method comparison and individual clips
for Native, SEAM, RTC, POTR, BID-Backward, DVAC, PACE, and GRACE: **180 evaluation videos**,
plus **3 presentation highlights**.

Choose a task, compare recorded outcomes and completion steps, jump to the GRACE completion
frame, or follow one method. Links preserve the selected example and method, and players
provide video downloads. For example:

[Two pots, GRACE](https://crychic-sakiko-togawa-214.github.io/grace-demo/?case=two-pots-01&method=GRACE)

These are recorded demonstration outcomes. The repository contains presentation code and
media, rather than a robotics training or evaluation implementation. The featured examples
should not be interpreted as aggregate benchmark statistics.

| Path | Purpose |
| --- | --- |
| `index.html`, `assets/site.css`, `assets/site.js` | Static page and interactions |
| `catalog.json` | Canonical case catalog, outcomes, metadata, and media paths |
| `assets/data.js` | Generated browser payload |
| `main_showcase.json`, `supplemental_showcase.json` | Generated catalog subsets |
| `MAIN_SHOWCASE.md`, `SUPPLEMENTAL_SHOWCASE.md` | Generated human-readable outcome tables |
| `single_method_videos/`, `comparison_videos/` | Recorded evaluation clips |
| `assets/highlights/`, `assets/posters/` | Presentation clips and preview images |
| `serve.py`, `scripts/`, `tests/` | Preview, synchronization, validation, and behavior tests |

## Maintain and validate

Edit `catalog.json`, then regenerate its five derived files:

```sh
python3 scripts/sync_catalog.py
python3 scripts/sync_catalog.py --check
python3 scripts/validate.py
python3 -m unittest discover -s tests -v
node --check assets/site.js
node --check assets/data.js
```

Node.js 22+ is needed only for JavaScript syntax checks. The validator checks local links,
HTML anchors, case IDs, task/method membership, video counts, outcome summaries, and stale
catalog copies. CI also detects stale Markdown reports. Tests exercise real HTTP range
requests and rejection of inconsistent data; no external service is needed.

For a sparse checkout, `python3 scripts/validate.py --tracked-assets` verifies omitted
media paths against the Git index. Default validation verifies files on disk. Neither mode
replaces playing new clips or inspecting binary video metadata.

See [the data contract](docs/DATA.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for the update
workflow. Preserve recorded outcomes and comparator attribution when updating the catalog.

## Deployment and reuse

GitHub Pages currently publishes `main` from the repository root. Keep `.nojekyll` and
relative asset paths. Any static HTTP host can serve the same files without a build step.
CI validates the repository and does not change deployment configuration.

There is no repository-wide reuse license yet; see [license status](LICENSE_STATUS.md).
Research video rights and comparator implementation licenses need separate verification.
Citation metadata can be added when a public paper and author information are available.
