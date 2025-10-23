# Ethereum Price Prediction - Experimentation Repository

**Course:** 36120 - Advanced Machine Learning  
**Assignment:** AT3 - Data Product with Machine Learning  
**Group:** <number>  
**Student ID:** <student_id>  
**Cryptocurrency:** Ethereum

---

## Project Overview

This repository contains experiments for building machine learning models to predict the **next day's HIGH price** of Ethereum. The models will be deployed via FastAPI and integrated into a Streamlit application for cryptocurrency investment analysis.

## Repository Structure

This project follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template:

```
experiments/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── .gitignore            # Git ignore rules
├── github.txt            # Link to GitHub repository
│
├── data/
│   ├── raw/              # Original, immutable data
│   ├── interim/          # Intermediate transformed data
│   ├── processed/        # Final feature sets for modeling
│   └── external/         # Data from third party sources
│
├── models/               # Trained model artifacts (.joblib, .pkl)
│
├── notebooks/            # Jupyter notebooks for experimentation
│   └── 36120-25SP-group<number>-<student_id>-AT3-experiment-1.ipynb
│
├── references/           # Data dictionaries, manuals, explanatory materials
│
├── reports/              # Generated analysis and figures
│   └── figures/          # Graphics and figures for reporting
│
└── src/                  # Source code for use in this project
    ├── __init__.py
    ├── data/             # Scripts to download or generate data
    ├── features/         # Scripts to turn raw data into features
    ├── models/           # Scripts to train models and make predictions
    └── visualization/    # Scripts to create exploratory and results visualizations
```

## Setup Instructions

### Prerequisites
- Python 3.11.4
- pip or conda package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd experiments
   ```

2. **Create a virtual environment:**
   ```bash
   # Using venv
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n crypto-pred python=3.11.4
   conda activate crypto-pred
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

## Data

### Dataset Information
- **Source:** CoinMarketCap historical data (2015-2025)
- **Cryptocurrency:** Ethereum
- **Format:** Daily OHLCV (Open, High, Low, Close, Volume) + Market Cap
- **Location:** `data/raw/ethereum_combined.csv`

### Data Dictionary
| Column | Description |
|--------|-------------|
| `timeOpen` | Timestamp (ISO 8601) of the start of this time series interval |
| `timeClose` | Timestamp (ISO 8601) of the end of this time series interval |
| `timeHigh` | Timestamp (ISO 8601) of the high of this time series interval |
| `timeLow` | Timestamp (ISO 8601) of the low of this time series interval |
| `name` | The CoinMarketCap cryptocurrency ID |
| `open` | Opening price for time series interval |
| `high` | Highest price during this time series interval |
| `low` | Lowest price during this time series interval |
| `close` | Closing price for this time series interval |
| `volume` | Adjusted volume for this time series interval |
| `marketCap` | Market cap by circulating supply for this time series interval |
| `timestamp` | Timestamp (ISO 8601) of when the conversion currency's current value was referenced |

## Running the Notebooks

1. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Navigate to the notebooks directory and open your experiment notebook**

3. **Run cells sequentially to:**
   - Load and explore data
   - Perform feature engineering
   - Train and evaluate models
   - Save best model artifacts

## Model Development Guidelines

### Target Variable
- Predict **next day's HIGH price** (day +1)
- Created as: `target_high_next_day = df['high'].shift(-1)`

### Algorithm Selection
Choose ONE algorithm (ensure it's different from team members):
- ✅ Linear Regression / Ridge / Lasso
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ XGBoost
- ✅ LightGBM
- ✅ Neural Network (e.g., LSTM, GRU)

### Key Considerations
1. **Time Series Split:** Use chronological splitting (not random) for train/test
2. **Feature Engineering:** Create lag features, rolling statistics, technical indicators
3. **Hyperparameter Tuning:** Use Hyperopt for optimization
4. **Model Interpretation:** Use LIME for explainability
5. **Cross-validation:** Use TimeSeriesSplit for proper validation

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## Model Artifacts

Trained models should be saved in the `models/` directory:
```
models/
├── ethereum_<algorithm>_model.joblib    # Trained model
├── scaler.joblib                         # Feature scaler (if used)
└── feature_names.json                    # Feature names for inference
```

## Experiment Tracking (Optional)

You can use Weights & Biases (wandb) for experiment tracking:
```python
import wandb
wandb.init(project="ethereum-price-prediction", name="experiment-1")
wandb.log({"mae": mae, "rmse": rmse, "r2": r2})
```

## Package Versions

All package versions are specified in `requirements.txt` as per assignment requirements:
- Python 3.11.4
- scikit-learn 1.5.1
- pandas 2.2.2
- xgboost 2.1.0
- lightgbm 4.4.0
- hyperopt 0.2.7
- lime 0.2.0.1
- wandb 0.17.4
- And more...

## Next Steps

After completing experimentation:
1. ✅ Save best model to `models/` folder
2. ✅ Document model performance and parameters
3. ✅ Prepare for FastAPI deployment
4. ✅ Integrate with Streamlit application
5. ✅ Deploy to Render

## Contributing

This is an individual contribution to a group project. Each student:
- Works on their own algorithm
- Maintains their own experiment notebooks
- Contributes their best model to the final application

## License

This project is for educational purposes as part of UTS course 36120.

## Contact

**Student:** <your_name>  
**Email:** <your_email>@student.uts.edu.au  
**GitHub:** <your_github_username>

---

*Last Updated: October 2025*

