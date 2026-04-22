# raw

This folder contains the raw source data for the prediction project.

## Providers

- [`understat/`](understat/README.md)  
  Match-level and expected-metric data from Understat.

- [`fbref/`](fbref/README.md)  
  Supporting match metadata and season-level team statistics from FBref.

- [`football_data/`](football_data/README.md)  
  Odds data used for the market benchmark.

## Working Principle

Files in this folder should be treated as source inputs. Downstream steps should not manually edit them, but instead create new tables in `interim` or `processed`.
