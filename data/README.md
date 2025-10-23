# Data Directory

This directory contains all data files for the Ethereum price prediction project.

## Structure

- **`raw/`** - Original, immutable data dump
  - `ethereum_combined.csv` - Combined Ethereum historical data (2015-2025)
  - Individual yearly CSV files for reference

- **`interim/`** - Intermediate data that has been transformed
  - Place partially processed data here

- **`processed/`** - Final, canonical data sets for modeling
  - Feature-engineered datasets ready for ML models

- **`external/`** - Data from third party sources
  - API responses from TokenMetrics, CoinDesk, etc.

## Data Processing Pipeline

1. **Raw → Interim:** Data cleaning, missing value handling, outlier detection
2. **Interim → Processed:** Feature engineering, scaling, encoding
3. **Processed → Models:** Training and evaluation

## Notes

- Never modify files in `raw/` directory
- Document all transformations in notebooks
- Keep processed datasets with version timestamps if needed

