# interim

This folder contains intermediate outputs between `raw` and `processed`.

## Files

- `base_matches.parquet`  
  The integrated base match table created after combining the main data sources. It acts as the stable input for downstream feature engineering.

## Role in the Pipeline

This layer sits between building the base match dataset and generating the richer pre-match modeling features.
