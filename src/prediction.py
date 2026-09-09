"""
Student Prediction Module for Student Performance Prediction.
Validates input features, loads the persisted model pipeline without retraining,
performs real-time dynamic inference, and generates risk categorization and recommendations.
"""

import os
import sys
from typing import Dict, Any, Union
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import MODEL_PATH, setup_logger
from src.risk_analysis import categorize_performance
from src.recommendation_engine import generate_recommendations

logger = setup_logger("Prediction")

VALID_PARENTAL_EDU = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
VALID_BINARY = ['Yes', 'No']

def load_trained_model(model_path: str = MODEL_PATH):
    """
    Loads the serialized model pipeline from disk.
    
    Raises:
        FileNotFoundError: If the model has not yet been trained and saved.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by selecting 'Train Models' from the menu."
        )
    logger.info(f"Loading trained pipeline from: {model_path}")
    return joblib.load(model_path)

def validate_student_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates single student input against academic and physical domain constraints.
    
    Raises:
        ValueError: If any attribute is missing, out-of-range, or has an invalid type.
    """
    cleaned: Dict[str, Any] = {}

    # 1. Attendance
    try:
        att = float(data['Attendance_Rate'])
        if not (0.0 <= att <= 100.0):
            raise ValueError(f"Attendance Rate must be between 0.0 and 100.0%, received: {att}")
        cleaned['Attendance_Rate'] = att
    except (KeyError, TypeError):
        raise ValueError("Missing or invalid numerical field: Attendance_Rate")

    # 2. Study Hours
    try:
        hours = float(data['Study_Hours_Per_Week'])
        if hours < 0.0:
            raise ValueError(f"Study Hours cannot be negative, received: {hours}")
        if hours > 100.0:
            raise ValueError(f"Study Hours per week exceeds physical limit (100 hrs/wk), received: {hours}")
        cleaned['Study_Hours_Per_Week'] = hours
    except (KeyError, TypeError):
        raise ValueError("Missing or invalid numerical field: Study_Hours_Per_Week")

    # 3. Previous Score
    try:
        prev = float(data['Previous_Score'])
        if not (0.0 <= prev <= 100.0):
            raise ValueError(f"Previous Score must be between 0.0 and 100.0, received: {prev}")
        cleaned['Previous_Score'] = prev
    except (KeyError, TypeError):
        raise ValueError("Missing or invalid numerical field: Previous_Score")

    # 4. Assignment Completion Rate
    try:
        assign = float(data['Assignment_Completion_Rate'])
        if not (0.0 <= assign <= 100.0):
            raise ValueError(f"Assignment Completion Rate must be between 0.0 and 100.0%, received: {assign}")
        cleaned['Assignment_Completion_Rate'] = assign
    except (KeyError, TypeError):
        raise ValueError("Missing or invalid numerical field: Assignment_Completion_Rate")

    # 5. Internal Assessment Score
    try:
        internal = float(data['Internal_Assessment_Score'])
        if not (0.0 <= internal <= 100.0):
            raise ValueError(f"Internal Assessment Score must be between 0.0 and 100.0, received: {internal}")
        cleaned['Internal_Assessment_Score'] = internal
    except (KeyError, TypeError):
        raise ValueError("Missing or invalid numerical field: Internal_Assessment_Score")

    # 6. Class Participation
    try:
        part = int(data['Class_Participation'])
        if not (1 <= part <= 5):
            raise ValueError(f"Class Participation rating must be an integer between 1 and 5, received: {part}")
        cleaned['Class_Participation'] = part
    except (KeyError, TypeError):
        raise ValueError("Missing or invalid field: Class_Participation")

    # 7. Parental Education Level
    edu = str(data.get('Parental_Education_Level', 'Bachelor')).strip()
    matched_edu = next((e for e in VALID_PARENTAL_EDU if e.lower() == edu.lower()), None)
    if not matched_edu:
        raise ValueError(f"Parental Education must be one of {VALID_PARENTAL_EDU}, received: '{edu}'")
    cleaned['Parental_Education_Level'] = matched_edu

    # 8. Internet Access
    net = str(data.get('Internet_Access', 'Yes')).strip().capitalize()
    if net not in VALID_BINARY:
        raise ValueError(f"Internet Access must be 'Yes' or 'No', received: '{net}'")
    cleaned['Internet_Access'] = net

    # 9. Extra Curricular
    extra = str(data.get('Extra_Curricular', 'No')).strip().capitalize()
    if extra not in VALID_BINARY:
        raise ValueError(f"Extra Curricular must be 'Yes' or 'No', received: '{extra}'")
    cleaned['Extra_Curricular'] = extra

    return cleaned

def predict_single_student(
    student_data: Dict[str, Any],
    pipeline=None
) -> Dict[str, Any]:
    """
    Performs end-to-end inference for a single student record.
    
    Args:
        student_data: Raw student inputs.
        pipeline: Optional preloaded scikit-learn pipeline.
        
    Returns:
        Dict containing prediction score, category metadata, and recommendations.
    """
    validated = validate_student_input(student_data)

    if pipeline is None:
        pipeline = load_trained_model()

    input_df = pd.DataFrame([validated])
    raw_pred = pipeline.predict(input_df)[0]
    bounded_pred = max(0.0, min(100.0, float(raw_pred)))

    perf_meta = categorize_performance(bounded_pred)
    recommendations = generate_recommendations(validated, perf_meta)

    return {
        'student_input': validated,
        'predicted_score': round(bounded_pred, 2),
        'performance_category': perf_meta['category'].upper(),
        'risk_level': perf_meta['risk_level'],
        'description': perf_meta['description'],
        'recommendations': recommendations
    }

def format_prediction_output(result: Dict[str, Any]) -> str:
    """Formats prediction results for display."""
    score = result['predicted_score']
    category = result['performance_category']
    risk = result['risk_level']
    recs = result['recommendations']

    output = []
    output.append("-" * 50)
    output.append("PREDICTION RESULT")
    output.append("-" * 50)
    output.append(f"Predicted Final Score : {score:.2f}")
    output.append(f"Performance Category  : {category}")
    output.append(f"Risk Tier             : {risk}")
    output.append("-" * 50)
    output.append("\nACADEMIC ADVISORY & RECOMMENDATIONS:")
    for i, r in enumerate(recs, 1):
        output.append(f" {i}. {r}")
    output.append("-" * 50)

    return "\n".join(output)

if __name__ == "__main__":
    sample = {
        'Attendance_Rate': 85.0,
        'Study_Hours_Per_Week': 14.0,
        'Previous_Score': 78.0,
        'Assignment_Completion_Rate': 88.0,
        'Internal_Assessment_Score': 72.0,
        'Class_Participation': 4,
        'Parental_Education_Level': 'Bachelor',
        'Internet_Access': 'Yes',
        'Extra_Curricular': 'Yes'
    }
    res = predict_single_student(sample)
    print(format_prediction_output(res))
