"""
Unit tests for preprocessing and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import AcademicFeatureEngineer
from src.preprocessing import build_preprocessor, build_full_pipeline
from sklearn.linear_model import LinearRegression

@pytest.fixture
def sample_raw_batch():
    return pd.DataFrame({
        'Attendance_Rate': [80.0, 90.0, np.nan],
        'Study_Hours_Per_Week': [15.0, np.nan, 20.0],
        'Previous_Score': [70.0, 85.0, 60.0],
        'Assignment_Completion_Rate': [75.0, 95.0, 50.0],
        'Internal_Assessment_Score': [65.0, 88.0, 55.0],
        'Class_Participation': [3, 5, 2],
        'Parental_Education_Level': ['Bachelor', np.nan, 'High School'],
        'Internet_Access': ['Yes', 'Yes', 'No'],
        'Extra_Curricular': ['No', 'Yes', 'No']
    })

def test_academic_feature_engineer(sample_raw_batch):
    """Verify engineered columns are correctly appended."""
    fe = AcademicFeatureEngineer()
    transformed = fe.transform(sample_raw_batch)
    assert 'Academic_Engagement_Index' in transformed.columns
    assert 'Assessment_Momentum' in transformed.columns
    assert 'Study_Efficiency_Ratio' in transformed.columns
    assert transformed.shape[1] == sample_raw_batch.shape[1] + 3

def test_preprocessor_imputation_and_scaling(sample_raw_batch):
    """Verify ColumnTransformer handles NaNs and transforms features without errors."""
    fe = AcademicFeatureEngineer()
    df_fe = fe.transform(sample_raw_batch)
    preprocessor = build_preprocessor()
    
    # Fit and transform
    X_proc = preprocessor.fit_transform(df_fe)
    assert isinstance(X_proc, np.ndarray)
    assert not np.isnan(X_proc).any()  # Imputation must resolve all NaNs
    assert X_proc.shape[0] == 3

def test_full_pipeline_encapsulation(sample_raw_batch):
    """Verify complete Pipeline fits and predicts without manual preprocessing."""
    y = np.array([72.0, 89.0, 56.0])
    pipeline = build_full_pipeline(LinearRegression())
    pipeline.fit(sample_raw_batch, y)
    preds = pipeline.predict(sample_raw_batch)
    assert len(preds) == 3
    assert not np.isnan(preds).any()
