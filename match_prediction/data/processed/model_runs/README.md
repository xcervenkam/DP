# model_runs

This folder contains detailed artifacts for each model run. Every `run_key` has its own subfolder with tables, diagnostics, and metadata.

## Available Runs

- `ml_multiclass`
- `ml_binary`
- `ml_betting_binary`
- `double_poisson_multiclass`
- `double_poisson_binary`

## What a Typical Run Folder Contains

- `run_metadata.json`
- tuning and screening summaries,
- prediction tables,
- class-level or matchday-level metrics,
- feature documentation,
- fit diagnostics or calibration outputs when relevant.

## Why This Folder Matters

This is the main audit trail of the project. If you want to document which model was selected, with which features, and with what performance, start here.
