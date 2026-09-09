"""
Data Loader Module for Student Performance Prediction.
Handles dataset ingestion, structural validation, domain range checks,
and train-test splitting before preprocessing to prevent data leakage.
"""

import os
import sys
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    IDENTIFIER_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    setup_logger
)

logger = setup_logger("DataLoader")

REQUIRED_COLUMNS = [
    'Student_ID',
    'Attendance_Rate',
    'Study_Hours_Per_Week',
    'Previous_Score',
    'Assignment_Completion_Rate',
    'Internal_Assessment_Score',
    'Class_Participation',
    'Parental_Education_Level',
    'Internet_Access',
    'Extra_Curricular',
    'Final_Exam_Score'
]

def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates the raw student performance CSV dataset.
    
    Args:
        file_path: Path to the raw CSV file. Defaults to project standard path.
        
    Returns:
        pd.DataFrame containing the validated student dataset.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or data contains invalid ranges.
    """
    path = file_path or RAW_DATA_PATH
    if not os.path.exists(path):
        logger.error(f"Dataset file not found at: {path}")
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please ensure 'data/raw/student_performance_data.csv' exists."
        )

    logger.info(f"Loading raw dataset from: {path}")
    df = pd.read_csv(path)

    # 1. Column Integrity Check
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # 2. Domain Range Sanity Validation
    validate_dataset_ranges(df)

    logger.info(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def validate_dataset_ranges(df: pd.DataFrame) -> None:
    """
    Verifies that numerical attributes satisfy academic domain bounds.
    """
    # Non-null values checked for ranges
    if (df['Attendance_Rate'].dropna() < 0).any() or (df['Attendance_Rate'].dropna() > 100).any():
        raise ValueError("Attendance_Rate contains values outside [0, 100].")

    if (df['Study_Hours_Per_Week'].dropna() < 0).any():
        raise ValueError("Study_Hours_Per_Week contains negative values.")

    if (df['Previous_Score'].dropna() < 0).any() or (df['Previous_Score'].dropna() > 100).any():
        raise ValueError("Previous_Score contains values outside [0, 100].")

    if (df['Assignment_Completion_Rate'].dropna() < 0).any() or (df['Assignment_Completion_Rate'].dropna() > 100).any():
        raise ValueError("Assignment_Completion_Rate contains values outside [0, 100].")

    if (df['Internal_Assessment_Score'].dropna() < 0).any() or (df['Internal_Assessment_Score'].dropna() > 100).any():
        raise ValueError("Internal_Assessment_Score contains values outside [0, 100].")

    if (df['Final_Exam_Score'].dropna() < 0).any() or (df['Final_Exam_Score'].dropna() > 100).any():
        raise ValueError("Final_Exam_Score contains values outside [0, 100].")

def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing partitions BEFORE any transformations.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Drop identifier if present
    feature_df = df.drop(columns=[IDENTIFIER_COLUMN]) if IDENTIFIER_COLUMN in df.columns else df.copy()

    if target_col not in feature_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Data split completed: Training samples={len(X_train)} ({100*(1-test_size):.0f}%), "
        f"Testing samples={len(X_test)} ({100*test_size:.0f}%)."
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_raw_data()
    X_tr, X_te, y_tr, y_te = split_data(df)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
