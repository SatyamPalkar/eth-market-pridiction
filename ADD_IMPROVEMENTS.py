#!/usr/bin/env python3
"""
Script to add improvement cells to the notebook
Run this after your existing code to add enhancements
"""

import json

notebook_path = 'notebooks/36120-25SP-AT3-group_25217353-student_id.ipynb'

# Improvement cells to add
improvement_cells = [
    {
        'cell_type': 'markdown',
        'source': ['## IMPROVEMENT 3: Enhanced Metrics & Direction Accuracy\n']
    },
    {
        'cell_type': 'code',
        'source': [
            '# Direction Accuracy - Critical for Trading!\n',
            'xgb_direction_acc = np.mean(np.sign(y_test.values) == np.sign(xgb_pred)) * 100\n',
            'lgbm_direction_acc = np.mean(np.sign(y_test.values) == np.sign(lgbm_pred)) * 100\n',
            'en_direction_acc = np.mean(np.sign(y_test.values) == np.sign(en_pred)) * 100\n',
            '\n',
            'print("\\n" + "="*60)\n',
            'print("DIRECTION ACCURACY (Up/Down Prediction):")\n',
            'print("="*60)\n',
            'print(f"XGBoost:    {xgb_direction_acc:.2f}%")\n',
            'print(f"LightGBM:   {lgbm_direction_acc:.2f}%")\n',
            'print(f"ElasticNet: {en_direction_acc:.2f}%")\n',
            'print("\\n✅ Direction accuracy is key for trading decisions!")\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'source': ['## IMPROVEMENT 4: Production Pipeline\n']
    },
    {
        'cell_type': 'code',
        'source': [
            '# Create Production-Ready Pipeline\n',
            'from sklearn.pipeline import Pipeline\n',
            '\n',
            'production_pipeline = Pipeline([\n',
            '    ("scaler", scaler),\n',
            '    ("model", best_model)\n',
            '])\n',
            '\n',
            'joblib.dump(production_pipeline, "../models/production_pipeline.joblib")\n',
            '\n',
            '# Save metadata\n',
            'metadata = {\n',
            '    "model_name": best_model_name,\n',
            '    "test_mae": float(best_mae),\n',
            '    "features": X_train.columns.tolist(),\n',
            '    "n_features": X_train.shape[1]\n',
            '}\n',
            '\n',
            'with open("../models/production_metadata.json", "w") as f:\n',
            '    json.dump(metadata, f, indent=4)\n',
            '\n',
            'print("✅ Production pipeline created!")\n',
            'print("   - production_pipeline.joblib")\n',
            'print("   - production_metadata.json")\n'
        ]
    }
]

print("Improvement cells defined!")
print(f"Total improvements: {len(improvement_cells)} cells")
print("\nTo add to notebook, run the cells manually or append them programmatically")



