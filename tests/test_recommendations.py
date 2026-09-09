"""
Unit tests for risk categorization tiers and rule-based recommendation logic.
"""

from src.risk_analysis import categorize_performance
from src.recommendation_engine import generate_recommendations

def test_categorize_performance_tiers():
    """Verify correct academic band mappings."""
    assert categorize_performance(95.0)['category'] == 'Excellent'
    assert categorize_performance(85.0)['category'] == 'Excellent'
    assert categorize_performance(78.0)['category'] == 'Good'
    assert categorize_performance(70.0)['category'] == 'Good'
    assert categorize_performance(62.0)['category'] == 'Average'
    assert categorize_performance(50.0)['category'] == 'Average'
    assert categorize_performance(45.0)['category'] == 'Needs Improvement'
    assert categorize_performance(15.0)['category'] == 'Needs Improvement'

def test_generate_recommendations_low_attendance():
    """Verify low attendance triggers attendance alert."""
    student = {
        'Attendance_Rate': 65.0,
        'Study_Hours_Per_Week': 20.0,
        'Previous_Score': 80.0,
        'Assignment_Completion_Rate': 90.0,
        'Internal_Assessment_Score': 80.0,
        'Class_Participation': 4
    }
    meta = {'score': 72.0, 'category': 'Good'}
    recs = generate_recommendations(student, meta)
    assert any("Attendance Alert" in r for r in recs)

def test_generate_recommendations_low_study_hours():
    """Verify low study hours triggers study hours alert."""
    student = {
        'Attendance_Rate': 90.0,
        'Study_Hours_Per_Week': 4.0,
        'Previous_Score': 80.0,
        'Assignment_Completion_Rate': 90.0,
        'Internal_Assessment_Score': 80.0,
        'Class_Participation': 4
    }
    meta = {'score': 74.0, 'category': 'Good'}
    recs = generate_recommendations(student, meta)
    assert any("Study Hours Alert" in r for r in recs)
