# deployment

This folder contains the final deployment artifacts for each `run_key`.

## Available Model Packages

- `ml_multiclass`
- `ml_binary`
- `ml_betting_binary`
- `double_poisson_multiclass`
- `double_poisson_binary`

## What a Typical Subfolder Contains

- `best_model.pkl`  
  The saved final model.

- `best_model_metadata.json`  
  Metadata about the selected model, feature set, and fit summary.

## When to Use This Folder

- when you want to generate new predictions without retraining,
- when you want to verify which model was selected as final,
- when you want to connect the saved artifacts to `10_next_matchday_predictions.ipynb`.
