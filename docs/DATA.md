# GRACE data contract

`catalog.json` is the single source of truth for the site. It contains the schema version, task list, method order, media paths, outcomes, and completion-step measurements.

After editing it, regenerate the browser payload and both reports:

```sh
python3 scripts/sync_catalog.py
python3 scripts/validate.py
python3 -m unittest discover -s tests -v
```

Generated files are `assets/data.js`, `main_showcase.json`, `supplemental_showcase.json`, `MAIN_SHOWCASE.md`, and `SUPPLEMENTAL_SHOWCASE.md`. Each case must list the eight methods in catalog order. `success` is the measured outcome; `recorded_action_steps` is the first successful action step or the run limit for an unsuccessful method; `rendered_frames` describes the complete rendered clip. GRACE's `advantage` summary and `failed_comparators` must agree with the method rows.

Keep paths relative to the repository root so the GitHub Pages project URL continues to work. Do not edit generated files by hand. Media is intentionally checked by path and metadata in CI without downloading every large video.
