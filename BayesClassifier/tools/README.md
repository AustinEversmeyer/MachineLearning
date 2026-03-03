# Tools

This folder contains helper scripts for data generation, analysis, and plotting.

## Layout

- `tools/scripts/`
  - `generate_synthetic_predictions.py`: Creates synthetic prediction CSVs and synthetic inputs.
  - `plot_feature_pdfs.py`: Plots theoretical PDFs with data overlays from prediction CSVs (`feature_*` columns).
- `tools/tests/`
  - `run_all.py`: Runs tool-related tests.
  - `run_wine_pdfs.py`: Runs the wine PDF plotting workflow.
  - `test_*.py`: Unit-style tests for preprocessing and analysis utilities.

## Notes

- Generated outputs should go under `output/`.
- Scripts use paths relative to the repo root; run them from the repo root for consistency.
- PDF plotting supports both runtime and CLI prediction CSV outputs as long as feature columns are present as `feature_<name>`.
