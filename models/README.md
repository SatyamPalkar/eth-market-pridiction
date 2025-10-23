# Models Directory

This directory contains trained model artifacts.

## Model Naming Convention

Use the following naming convention for your models:
```
ethereum_<algorithm>_<version>.joblib
```

Examples:
- `ethereum_xgboost_v1.joblib`
- `ethereum_lightgbm_v2.joblib`
- `ethereum_randomforest_final.joblib`

## Required Files

When saving a model, include:
1. **Model file** - `.joblib` or `.pkl` file
2. **Scaler** - `scaler.joblib` (if features were scaled)
3. **Feature list** - `feature_names.json` (for inference)
4. **Metadata** - `model_metadata.json` (hyperparameters, metrics)

## Model Metadata Example

```json
{
  "model_name": "ethereum_xgboost_v1",
  "algorithm": "XGBoost",
  "created_date": "2025-10-13",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.01
  },
  "performance": {
    "train_rmse": 45.32,
    "test_rmse": 52.18,
    "train_r2": 0.92,
    "test_r2": 0.88
  },
  "features": ["open", "close", "volume", "lag_1", "ma_7"]
}
```

## Loading Models

```python
import joblib

# Load model
model = joblib.load('models/ethereum_xgboost_v1.joblib')

# Load scaler
scaler = joblib.load('models/scaler.joblib')

# Make prediction
prediction = model.predict(scaled_features)
```

## Best Practices

- Save only your best performing models
- Keep a backup of models before overwriting
- Document model performance in experiment notebook
- Include model version in filename
- Update this README with model details

