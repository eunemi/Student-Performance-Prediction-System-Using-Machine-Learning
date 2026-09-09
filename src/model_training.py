"""
Model Training Module for Student Performance Prediction.
Trains, cross-validates, evaluates, and persists candidate regression models
using encapsulated, leak-free scikit-learn Pipelines.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    RANDOM_STATE,
    MODELS_DIR,
    MODEL_PATH,
    ensure_directories,
    setup_logger,
    print_banner
)
from src.data_loader import load_raw_data, split_data
from src.preprocessing import build_full_pipeline
from src.model_evaluation import (
    calculate_metrics,
    generate_comparison_table,
    save_evaluation_results
)

logger = setup_logger("ModelTraining")

def get_candidate_models() -> Dict[str, Any]:
    """Returns candidate regression models configured with fixed random states."""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            min_samples_split=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        )
    }

def train_and_evaluate_all(
    file_path: str = None
) -> Tuple[str, Any, pd.DataFrame, Dict[str, Any]]:
    """
    Executes end-to-end model training, cross-validation, and holdout evaluation.
    
    Returns:
        best_model_name, best_pipeline, comparison_df, detailed_results
    """
    ensure_directories()
    
    # 1. Load and Partition Data
    raw_df = load_raw_data(file_path)
    X_train, X_test, y_train, y_test = split_data(raw_df)

    candidates = get_candidate_models()
    evaluation_records = {}
    fitted_pipelines = {}

    print_banner("MODEL TRAINING & CROSS-VALIDATION PIPELINE")
    logger.info(f"Evaluating {len(candidates)} regression candidate architectures...")

    for name, estimator in candidates.items():
        logger.info(f"--> Training candidate: {name}")
        pipeline = build_full_pipeline(estimator)

        # 5-fold cross-validation on training partition
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        logger.info(f"    5-Fold CV R² Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Fit complete pipeline on entire training partition
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on holdout test partition
        y_pred = pipeline.predict(X_test)
        test_metrics = calculate_metrics(y_test.values, y_pred)
        logger.info(f"    Holdout Test Metrics: {test_metrics}")

        evaluation_records[name] = {
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'test_metrics': test_metrics
        }

    # Generate comparison DataFrame
    comparison_df = generate_comparison_table(evaluation_records)
    print("\n" + "=" * 70)
    print(" ACTUAL EXPERIMENTAL MODEL EVALUATION RESULTS (HOLDOUT TEST SET)")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70 + "\n")

    # Select best model: highest R² on holdout test set
    best_model_name = comparison_df.iloc[0]['Model']
    best_pipeline = fitted_pipelines[best_model_name]
    best_r2 = comparison_df.iloc[0]['R2']
    best_rmse = comparison_df.iloc[0]['RMSE']

    logger.info(
        f"Selected Best Model: '{best_model_name}' (Holdout R²={best_r2:.4f}, RMSE={best_rmse:.4f})"
    )

    # Persist Best Model using Joblib
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info(f"Serialized production model saved to: {MODEL_PATH}")

    # Persist Model Metadata
    metadata = {
        'best_model_name': best_model_name,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'holdout_metrics': evaluation_records[best_model_name]['test_metrics'],
        'cv_r2_mean': evaluation_records[best_model_name]['cv_r2_mean'],
        'cv_r2_std': evaluation_records[best_model_name]['cv_r2_std'],
        'candidate_comparison': comparison_df.to_dict(orient='records')
    }
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to: {meta_path}")

    # Save Metrics Tables
    save_evaluation_results(comparison_df, best_model_name, evaluation_records)

    return best_model_name, best_pipeline, comparison_df, evaluation_records

if __name__ == "__main__":
    train_and_evaluate_all()
