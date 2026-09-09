"""
Model Evaluation Module for Student Performance Prediction.
Computes genuine regression metrics (MAE, MSE, RMSE, R²), formats comparison tables,
and persists evaluation artifacts to disk.
"""

import os
import sys
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import METRICS_DIR, METRICS_PATH, setup_logger

logger = setup_logger("ModelEvaluation")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes genuine regression performance metrics.
    All values strictly calculated from actual arrays; never hard-coded.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4)
    }

def generate_comparison_table(model_eval_records: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs a comparative DataFrame from evaluated models.
    """
    records = []
    for model_name, data in model_eval_records.items():
        metrics = data['test_metrics']
        cv_r2 = data.get('cv_r2_mean', None)
        cv_r2_std = data.get('cv_r2_std', None)
        records.append({
            'Model': model_name,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'CV_R2_Mean': round(cv_r2, 4) if cv_r2 is not None else 'N/A',
            'CV_R2_Std': round(cv_r2_std, 4) if cv_r2_std is not None else 'N/A'
        })

    df_comparison = pd.DataFrame(records)
    # Sort by R2 descending (highest explanatory power first)
    df_comparison = df_comparison.sort_values(by='R2', ascending=False).reset_index(drop=True)
    return df_comparison

def save_evaluation_results(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    full_results: Dict[str, Any]
) -> None:
    """
    Saves evaluation metrics table to CSV and full summary metadata to JSON.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)

    csv_path = os.path.join(METRICS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Model comparison table saved to: {csv_path}")

    # Save detailed JSON summary
    summary_data = {
        'best_model': best_model_name,
        'selection_criterion': 'Highest Test R² with minimum test RMSE',
        'comparison_summary': comparison_df.to_dict(orient='records'),
        'detailed_model_metrics': full_results
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(summary_data, f, indent=4)
    logger.info(f"Detailed metrics summary saved to: {METRICS_PATH}")
