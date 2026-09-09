"""
Performance Risk Categorization Module for Student Performance Prediction.
Maps continuous predicted scores to interpretable academic performance tiers.
"""

from typing import Dict, Any

# Project-defined academic reference thresholds
TIER_EXCELLENT_MIN = 85.0
TIER_GOOD_MIN = 70.0
TIER_AVERAGE_MIN = 50.0

CATEGORIES = {
    'EXCELLENT': 'Excellent',
    'GOOD': 'Good',
    'AVERAGE': 'Average',
    'NEEDS_IMPROVEMENT': 'Needs Improvement'
}

def categorize_performance(score: float) -> Dict[str, Any]:
    """
    Categorizes a continuous predicted score into project-defined academic tiers.
    
    Academic Disclaimer:
    These categories are project-defined pedagogical tiers created for early academic
    guidance. They do not constitute standardized institutional grades or rigid
    psychometric classifications.
    
    Args:
        score: Continuous predicted final score (0 - 100).
        
    Returns:
        Dict containing category label, description, and risk tier.
    """
    bounded_score = max(0.0, min(100.0, float(score)))

    if bounded_score >= TIER_EXCELLENT_MIN:
        category = CATEGORIES['EXCELLENT']
        risk_level = "Low Risk / High Achiever"
        description = "Student demonstrates strong mastery and high probability of academic distinction."
    elif bounded_score >= TIER_GOOD_MIN:
        category = CATEGORIES['GOOD']
        risk_level = "Moderate / On Track"
        description = "Student demonstrates solid competence with stable academic trajectory."
    elif bounded_score >= TIER_AVERAGE_MIN:
        category = CATEGORIES['AVERAGE']
        risk_level = "Moderate Risk / Marginal Pass"
        description = "Student performance is borderline. Targeted effort is needed to secure high marks."
    else:
        category = CATEGORIES['NEEDS_IMPROVEMENT']
        risk_level = "High Academic Risk"
        description = "Student is in significant danger of failing or severe underperformance without intervention."

    return {
        'score': round(bounded_score, 2),
        'category': category,
        'risk_level': risk_level,
        'description': description
    }

if __name__ == "__main__":
    for s in [92.5, 78.4, 58.2, 42.1]:
        res = categorize_performance(s)
        print(f"Score {s:5.1f} -> Category: {res['category']:<18} | Risk: {res['risk_level']}")
