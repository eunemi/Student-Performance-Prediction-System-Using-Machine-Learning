"""
Data Preprocessing Module for Student Performance Prediction.
Builds scikit-learn ColumnTransformer and Pipeline architectures
to guarantee zero data leakage between training and evaluation splits.
"""

import sys
import os
from typing import List, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import AcademicFeatureEngineer
from src.utils import setup_logger

logger = setup_logger("Preprocessing")

NUMERICAL_FEATURES: List[str] = [
    'Attendance_Rate',
    'Study_Hours_Per_Week',
    'Previous_Score',
    'Assignment_Completion_Rate',
    'Internal_Assessment_Score',
    'Class_Participation',
    'Academic_Engagement_Index',
    'Assessment_Momentum',
    'Study_Efficiency_Ratio'
]

CATEGORICAL_FEATURES: List[str] = [
    'Parental_Education_Level',
    'Internet_Access',
    'Extra_Curricular'
]

def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an unfitted Scikit-learn ColumnTransformer for numerical
    and categorical features.
    
    Preprocessing design:
    - Numerical: Median imputation (outlier-resilient) -> StandardScaler (zero mean, unit variance).
    - Categorical: Most Frequent imputation -> OneHotEncoder (handle_unknown='ignore').
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def build_full_pipeline(model_estimator) -> Pipeline:
    """
    Combines feature engineering, preprocessing, and the regression estimator
    into a single encapsulated scikit-learn Pipeline.
    
    This ensures:
    1. Zero data leakage (parameters fitted ONLY on train data during fit()).
    2. Seamless inference on raw inputs without manual pre-transformation.
    """
    full_pipeline = Pipeline([
        ('feature_engineer', AcademicFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', model_estimator)
    ])
    return full_pipeline

if __name__ == "__main__":
    from src.data_loader import load_raw_data, split_data
    from sklearn.linear_model import LinearRegression

    df = load_raw_data()
    X_tr, X_te, y_tr, y_te = split_data(df)

    # Test full pipeline fitting
    pipe = build_full_pipeline(LinearRegression())
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    print("Pipeline fit and prediction successful!")
    print(f"Sample test predictions: {preds[:5].round(2)}")
