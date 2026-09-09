"""
Recommendation Engine Module for Student Performance Prediction.
Generates transparent, rule-based academic advisory interventions
based on student behavioral indicators and projected performance tier.

NOTE: This is an interpretable, deterministic rule-based advisory system,
NOT a machine learning model. Machine learning predicts the continuous score,
while domain rules translate input deficiencies into pedagogical guidance.
"""

from typing import Dict, List, Any

# Academic benchmark criteria
CRITICAL_ATTENDANCE_THRESHOLD = 75.0      # Statutory minimum in many universities
MIN_STUDY_HOURS_THRESHOLD = 10.0          # Recommended minimum weekly hours
MIN_ASSIGNMENT_COMPLETION = 70.0          # Satisfactory assignment submission
MIN_INTERNAL_ASSESSMENT = 55.0            # Passing internal benchmark
MIN_PREVIOUS_SCORE = 55.0                 # Prerequisite baseline mastery
MIN_CLASS_PARTICIPATION = 2               # Active participation threshold

def generate_recommendations(student_data: Dict[str, Any], performance_meta: Dict[str, Any]) -> List[str]:
    """
    Evaluates individual student inputs and risk tier to synthesize actionable recommendations.
    
    Args:
        student_data: Dictionary containing input attributes (attendance, study hours, etc.)
        performance_meta: Dictionary from risk_analysis containing predicted score and category.
        
    Returns:
        List of formatted recommendation strings.
    """
    recommendations: List[str] = []
    
    attendance = float(student_data.get('Attendance_Rate', 100))
    study_hours = float(student_data.get('Study_Hours_Per_Week', 15))
    prev_score = float(student_data.get('Previous_Score', 70))
    assignment_rate = float(student_data.get('Assignment_Completion_Rate', 80))
    internal_assessment = float(student_data.get('Internal_Assessment_Score', 65))
    participation = int(student_data.get('Class_Participation', 3))
    category = performance_meta.get('category', 'Average')

    # 1. Attendance Check
    if attendance < CRITICAL_ATTENDANCE_THRESHOLD:
        recommendations.append(
            f"[Attendance Alert] Current attendance is {attendance:.1f}%, below the institutional 75% threshold. "
            "Prioritize physical lecture attendance to prevent missing foundational explanations."
        )

    # 2. Study Hours Check
    if study_hours < MIN_STUDY_HOURS_THRESHOLD:
        recommendations.append(
            f"[Study Hours Alert] Weekly self-study ({study_hours:.1f} hrs) is insufficient. "
            "Allocate at least 2 additional hours per weekday for revision and problem-solving."
        )

    # 3. Assignment Completion Check
    if assignment_rate < MIN_ASSIGNMENT_COMPLETION:
        recommendations.append(
            f"[Assignment Alert] Assignment completion rate is {assignment_rate:.1f}%. "
            "Completing formative homework exercises is directly correlated with summative exam readiness."
        )

    # 4. Internal Assessment Check
    if internal_assessment < MIN_INTERNAL_ASSESSMENT:
        recommendations.append(
            f"[Midterm Performance] Internal assessment score ({internal_assessment:.1f}) indicates conceptual gaps. "
            "Schedule instructor office hours or peer mentoring sessions to review midterm mistakes."
        )

    # 5. Previous Academic Foundation
    if prev_score < MIN_PREVIOUS_SCORE:
        recommendations.append(
            f"[Prerequisite Knowledge] Prior academic score ({prev_score:.1f}) suggests underlying weaknesses. "
            "Review foundational prerequisite topics to support advanced coursework comprehension."
        )

    # 6. Participation Check
    if participation <= MIN_CLASS_PARTICIPATION:
        recommendations.append(
            "[Engagement Opportunity] Classroom participation is low. "
            "Engage in active discussions, ask questions during lectures, and participate in peer study groups."
        )

    # 7. Category-Specific Holistic Guidance
    if category == 'Needs Improvement':
        recommendations.append(
            "[Urgent Intervention] High academic risk detected. "
            "An academic advising appointment should be scheduled immediately for personalized remediation."
        )
    elif category == 'Average':
        recommendations.append(
            "[Targeted Improvement] Moderate performance band. Focus on turning weak subjects into strengths "
            "by increasing practice exam attempts and time-management discipline."
        )
    elif category == 'Good':
        recommendations.append(
            "[Consistency Note] On-track trajectory. Maintain current study discipline and explore challenging "
            "supplementary problem sets to elevate performance into distinction."
        )
    elif category == 'Excellent':
        recommendations.append(
            "[Enrichment Pathway] Outstanding performance projected. Consider taking leadership in study groups, "
            "participating in academic competitions, or assisting faculty with research."
        )

    return recommendations

if __name__ == "__main__":
    sample_student = {
        'Attendance_Rate': 68.0,
        'Study_Hours_Per_Week': 6.5,
        'Previous_Score': 52.0,
        'Assignment_Completion_Rate': 60.0,
        'Internal_Assessment_Score': 48.0,
        'Class_Participation': 2
    }
    sample_meta = {'score': 47.5, 'category': 'Needs Improvement'}
    recs = generate_recommendations(sample_student, sample_meta)
    print(f"Generated {len(recs)} recommendations for at-risk sample:")
    for r in recs:
        print(f" - {r}")
