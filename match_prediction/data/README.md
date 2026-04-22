# data

This folder contains the full data pipeline for the prediction section.

## Structure

- [`raw/`](raw/README.md)  
  Raw source data downloaded or collected from external providers.

- [`interim/`](interim/README.md)  
  Intermediate tables between raw inputs and final feature-ready data.

- [`processed/`](processed/README.md)  
  Cleaned and modeling-ready data, run outputs, and support tables for evaluation and reporting.

## How to Read This Folder

The project follows a standard pipeline logic:

1. `raw` stores source data.
2. `interim` stores the stabilized intermediate match table.
3. `processed` stores the main analytical working layer used by models and reports.

If it is not clear where a table comes from, trace it backward in this order: `outputs` or `processed` -> `interim` -> `raw`.
