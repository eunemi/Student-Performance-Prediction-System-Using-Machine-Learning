"""
Analytics Module for Student Performance Prediction.
Extracts model-level feature importances, performs residual error analysis,
visualizes comparative algorithm performance, and saves plots to disk.
"""

import os
import sys
import json
import matplotlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import PLOTS_DIR, METRICS_PATH, ensure_directories, setup_logger

ensure_directories()
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import load_raw_data, split_data
from src.feature_engineering import AcademicFeatureEngineer
from src.preprocessing import build_preprocessor

logger = setup_logger("Analytics")

def generate_feature_importance_plot(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Fits a Random Forest model to extract and plot feature importance scores.
    
    ACADEMIC DISCLAIMER:
    Feature importance represents statistical association within the trained model
    and does NOT establish causal relationships in student learning.
    """
    if df is None:
        df = load_raw_data()

    X_train, _, y_train, _ = split_data(df)

    # 1. Apply feature engineering
    fe = AcademicFeatureEngineer()
    X_fe = fe.transform(X_train)

    # 2. Fit preprocessor to get transformed feature names
    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(X_fe)

    raw_names = preprocessor.get_feature_names_out()
    clean_names = [name.replace('num__', '').replace('cat__', '') for name in raw_names]

    # 3. Fit Random Forest to extract Gini importance
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_proc, y_train)

    importances = rf.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': clean_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # 4. Plot Feature Importances
    plt.figure(figsize=(11, 7))
    top_imp = imp_df.head(10)
    palette = sns.color_palette("Blues_r", n_colors=len(top_imp))
    sns.barplot(data=top_imp, x='Importance', y='Feature', palette=palette)

    plt.title(
        "Top 10 Feature Importances (Random Forest Regressor)\n"
        "[Note: Model associations indicate statistical predictive value, not direct causality]",
        fontsize=12, fontweight='bold', pad=12
    )
    plt.xlabel("Gini Feature Importance Score")
    plt.ylabel("Academic Feature")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Feature importance plot saved to: {out_path}")

    return imp_df

def generate_residual_analysis_plot(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Generates residual distribution and actual vs predicted scatter plots.
    """
    y_pred = pipeline.predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='#1f77b4', edgecolors='none')
    min_val = min(y_test.min(), y_pred.min()) - 2
    max_val = max(y_test.max(), y_pred.max()) + 2
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y = x)')
    axes[0].set_title("Actual vs. Predicted Final Exam Scores", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Actual Exam Score")
    axes[0].set_ylabel("Predicted Exam Score")
    axes[0].legend()

    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='#2ca02c', bins=20)
    axes[1].axvline(0, color='red', linestyle='--', label='Zero Error Baseline')
    axes[1].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Residual Error")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'residual_analysis.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Residual analysis plot saved to: {out_path}")

def generate_model_comparison_plot() -> None:
    """
    Plots comparative R² and RMSE from the stored evaluation metrics.
    """
    if not os.path.exists(METRICS_PATH):
        logger.warning("Metrics file not found. Run model_training.py first.")
        return

    with open(METRICS_PATH, 'r') as f:
        data = json.load(f)

    comp_list = data.get('comparison_summary', [])
    if not comp_list:
        return

    comp_df = pd.DataFrame(comp_list)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R2 Comparison
    sns.barplot(data=comp_df, x='Model', y='R2', ax=axes[0], palette='crest')
    axes[0].set_title("Holdout Test R² Score by Model (Higher is Better)", fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("R² Score")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                         ha='center', va='center', color='white', fontweight='bold')

    # RMSE Comparison
    sns.barplot(data=comp_df, x='Model', y='RMSE', ax=axes[1], palette='flare')
    axes[1].set_title("Holdout Test RMSE by Model (Lower is Better)", fontweight='bold')
    axes[1].set_ylabel("RMSE (Score Points)")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                         ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Model comparison bar plot saved to: {out_path}")

def run_all_analytics():
    """Generates the full suite of diagnostic analytics and plots."""
    ensure_directories()
    df = load_raw_data()
    imp_df = generate_feature_importance_plot(df)
    generate_model_comparison_plot()

    # If model exists, run residual analysis
    from src.prediction import load_trained_model
    try:
        model = load_trained_model()
        _, X_test, _, y_test = split_data(df)
        generate_residual_analysis_plot(model, X_test, y_test)
    except Exception as e:
        logger.warning(f"Could not run residual analysis: {e}")

    logger.info("All analytics generation tasks completed.")

if __name__ == "__main__":
    run_all_analytics()
