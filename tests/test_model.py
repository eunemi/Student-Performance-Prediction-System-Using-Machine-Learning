import pytest
"""
Unit tests for model training, metrics calculation, and serialization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.model_evaluation import calculate_metrics, generate_comparison_table
from src.model_training import train_and_evaluate_all
from src.utils import MODEL_PATH, METRICS_PATH

def test_calculate_metrics_accuracy():
    """Verify regression metrics formulas."""
    y_true = np.array([50.0, 70.0, 90.0])
    y_pred = np.array([52.0, 68.0, 91.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert 'MAE' in metrics
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
    assert metrics['MAE'] == pytest.approx(1.6667, abs=1e-3)
    assert metrics['RMSE'] > 0

def test_model_training_and_persistence():
    """Verify end-to-end training generates model artifact and metrics."""
    best_name, pipeline, comparison_df, results = train_and_evaluate_all()
    assert best_name in ['Linear Regression', 'Gradient Boosting', 'Random Forest', 'Decision Tree']
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(METRICS_PATH)
    assert isinstance(comparison_df, pd.DataFrame)
    assert len(comparison_df) == 4

    # Test loading serialized model
    loaded_pipe = joblib.load(MODEL_PATH)
    assert hasattr(loaded_pipe, "predict")
