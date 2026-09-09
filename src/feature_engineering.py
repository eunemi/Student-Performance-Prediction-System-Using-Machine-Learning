"""
Feature Engineering Module for Student Performance Prediction.
Constructs domain-specific, academically justified indicators to enhance
predictive power while preserving model interpretability.
"""

import sys
import os
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logger

logger = setup_logger("FeatureEngineering")

class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that calculates derived academic indicators:
    
    1. Academic_Engagement_Index:
       Synthesis of Attendance Rate and Assignment Completion Rate.
       Rationale: Attendance and assignment completion collectively measure behavioral
       engagement and academic persistence. Combining them provides a robust composite signal.
       
    2. Assessment_Momentum:
       Difference between Internal Assessment Score and Previous Score.
       Rationale: Represents academic growth trajectory. A positive value indicates positive
       momentum; a negative delta signals recent academic decline.
       
    3. Study_Efficiency_Ratio:
       Ratio of Previous Score to (Study Hours + 1).
       Rationale: Measures academic yield per hour invested, capturing baseline competence
       and potential inefficiencies in study techniques.
    """
    def __init__(self, add_engagement: bool = True, add_momentum: bool = True, add_efficiency: bool = True):
        self.add_engagement = add_engagement
        self.add_momentum = add_momentum
        self.add_efficiency = add_efficiency

    def fit(self, X, y=None):
        # Feature engineering is stateless; no parameters learned across samples
        return self

    def transform(self, X):
        """Generates derived features on input DataFrame."""
        if isinstance(X, np.ndarray):
            # If array, pass through unchanged
            return X

        df = X.copy()
        
        if self.add_engagement and 'Attendance_Rate' in df.columns and 'Assignment_Completion_Rate' in df.columns:
            # 0.5 * Attendance + 0.5 * Assignment Completion
            df['Academic_Engagement_Index'] = (
                0.5 * df['Attendance_Rate'].fillna(df['Attendance_Rate'].median()) +
                0.5 * df['Assignment_Completion_Rate'].fillna(df['Assignment_Completion_Rate'].median())
            ).round(2)

        if self.add_momentum and 'Internal_Assessment_Score' in df.columns and 'Previous_Score' in df.columns:
            # Trajectory: Internal Assessment - Previous Score
            df['Assessment_Momentum'] = (
                df['Internal_Assessment_Score'].fillna(df['Internal_Assessment_Score'].median()) -
                df['Previous_Score'].fillna(df['Previous_Score'].median())
            ).round(2)

        if self.add_efficiency and 'Previous_Score' in df.columns and 'Study_Hours_Per_Week' in df.columns:
            # Study efficiency ratio
            hours = df['Study_Hours_Per_Week'].fillna(df['Study_Hours_Per_Week'].median())
            prev = df['Previous_Score'].fillna(df['Previous_Score'].median())
            df['Study_Efficiency_Ratio'] = (prev / (hours + 1.0)).round(2)

        return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience standalone function for feature engineering."""
    transformer = AcademicFeatureEngineer()
    engineered_df = transformer.transform(df)
    logger.info(f"Engineered 3 features. New column count: {engineered_df.shape[1]}")
    return engineered_df

if __name__ == "__main__":
    from src.data_loader import load_raw_data, split_data
    raw = load_raw_data()
    X_tr, _, _, _ = split_data(raw)
    feat_df = engineer_features(X_tr)
    print("New features sample:")
    print(feat_df[['Academic_Engagement_Index', 'Assessment_Momentum', 'Study_Efficiency_Ratio']].head())
