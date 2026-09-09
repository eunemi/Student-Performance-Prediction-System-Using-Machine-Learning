"""
Unit tests for student performance prediction and input validation guards.
"""

import pytest
from src.prediction import (
    validate_student_input,
    predict_single_student,
    load_trained_model
)

@pytest.fixture
def valid_student():
    return {
        'Attendance_Rate': 88.0,
        'Study_Hours_Per_Week': 15.0,
        'Previous_Score': 75.0,
        'Assignment_Completion_Rate': 85.0,
        'Internal_Assessment_Score': 70.0,
        'Class_Participation': 4,
        'Parental_Education_Level': 'Bachelor',
        'Internet_Access': 'Yes',
        'Extra_Curricular': 'Yes'
    }

def test_validate_student_input_valid(valid_student):
    """Verify clean inputs pass validation untouched."""
    cleaned = validate_student_input(valid_student)
    assert cleaned['Attendance_Rate'] == 88.0
    assert cleaned['Class_Participation'] == 4
    assert cleaned['Parental_Education_Level'] == 'Bachelor'

def test_validate_student_input_negative_study_hours(valid_student):
    """Verify negative study hours are rejected."""
    bad_data = valid_student.copy()
    bad_data['Study_Hours_Per_Week'] = -5.0
    with pytest.raises(ValueError, match="negative"):
        validate_student_input(bad_data)

def test_validate_student_input_out_of_bounds_attendance(valid_student):
    """Verify attendance > 100% is rejected."""
    bad_data = valid_student.copy()
    bad_data['Attendance_Rate'] = 105.0
    with pytest.raises(ValueError, match="Attendance Rate must be between"):
        validate_student_input(bad_data)

def test_validate_student_input_invalid_participation(valid_student):
    """Verify rubric score outside 1-5 is rejected."""
    bad_data = valid_student.copy()
    bad_data['Class_Participation'] = 7
    with pytest.raises(ValueError, match="Class Participation rating"):
        validate_student_input(bad_data)

def test_predict_single_student_success(valid_student):
    """Verify prediction produces expected structure and bounded score."""
    model = load_trained_model()
    res = predict_single_student(valid_student, pipeline=model)
    assert 'predicted_score' in res
    assert 'performance_category' in res
    assert 'recommendations' in res
    assert 0.0 <= res['predicted_score'] <= 100.0
    assert isinstance(res['recommendations'], list)
