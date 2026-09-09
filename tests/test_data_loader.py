"""
Unit tests for data loading and dataset validation.
"""

import pytest
import pandas as pd
from src.data_loader import load_raw_data, split_data, validate_dataset_ranges

def test_load_raw_data_success():
    """Verify standard raw dataset loads correctly."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1000
    assert df.shape[1] == 11
    assert 'Final_Exam_Score' in df.columns

def test_load_raw_data_missing_file():
    """Verify FileNotFoundError on nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("nonexistent_path/student_data.csv")

def test_validate_dataset_ranges_invalid_attendance():
    """Verify ValueError when attendance exceeds 100%."""
    bad_df = pd.DataFrame({
        'Attendance_Rate': [105.0],
        'Study_Hours_Per_Week': [10.0],
        'Previous_Score': [70.0],
        'Assignment_Completion_Rate': [80.0],
        'Internal_Assessment_Score': [75.0],
        'Final_Exam_Score': [72.0]
    })
    with pytest.raises(ValueError, match="Attendance_Rate"):
        validate_dataset_ranges(bad_df)

def test_validate_dataset_ranges_negative_study_hours():
    """Verify ValueError when study hours are negative."""
    bad_df = pd.DataFrame({
        'Attendance_Rate': [80.0],
        'Study_Hours_Per_Week': [-3.0],
        'Previous_Score': [70.0],
        'Assignment_Completion_Rate': [80.0],
        'Internal_Assessment_Score': [75.0],
        'Final_Exam_Score': [72.0]
    })
    with pytest.raises(ValueError, match="Study_Hours_Per_Week"):
        validate_dataset_ranges(bad_df)

def test_split_data_proportions():
    """Verify 80/20 train/test partition sizes."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(y_train) == 800
    assert len(y_test) == 200
    assert 'Student_ID' not in X_train.columns
    assert 'Final_Exam_Score' not in X_train.columns
