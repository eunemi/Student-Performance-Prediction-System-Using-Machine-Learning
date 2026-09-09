import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.mpl_cache')
"""
Student Performance Prediction System - Main CLI Application
VITyarthi – Fundamentals of AI and ML
Autonomous Multi-Agent Engineering Team
"""

import os
import sys
import json
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    MODEL_PATH,
    METRICS_PATH,
    RAW_DATA_PATH,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_directories,
    print_banner
)
from src.data_loader import load_raw_data
from src.model_training import train_and_evaluate_all
from src.prediction import (
    load_trained_model,
    predict_single_student,
    format_prediction_output,
    VALID_PARENTAL_EDU
)
from src.analytics import run_all_analytics
from src.reporting import generate_project_report

MENU_HEADER = """
==================================================
       STUDENT PERFORMANCE PREDICTION SYSTEM
==================================================

1. Train Models
2. Evaluate Models
3. Predict Student Performance
4. View Analytics
5. Generate Report
6. Exit
"""

def prompt_bounded_float(prompt_text: str, min_val: float, max_val: float = None) -> float:
    """Safely prompts the user for a float within [min_val, max_val]."""
    while True:
        user_input = input(prompt_text).strip()
        try:
            val = float(user_input)
            if val < min_val:
                print(f" [Error] Value cannot be less than {min_val}. Please try again.")
                continue
            if max_val is not None and val > max_val:
                print(f" [Error] Value cannot exceed {max_val}. Please try again.")
                continue
            return val
        except ValueError:
            print(" [Error] Please enter a valid numerical value.")

def prompt_bounded_int(prompt_text: str, min_val: int, max_val: int) -> int:
    """Safely prompts the user for an integer within [min_val, max_val]."""
    while True:
        user_input = input(prompt_text).strip()
        try:
            val = int(user_input)
            if not (min_val <= val <= max_val):
                print(f" [Error] Please enter an integer between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print(f" [Error] Invalid input. Please enter an integer between {min_val} and {max_val}.")

def prompt_choice(prompt_text: str, options: list) -> str:
    """Prompts user to select from a list of options by index."""
    print(f"\n{prompt_text}")
    for idx, opt in enumerate(options, 1):
        print(f"  {idx}. {opt}")
    choice_idx = prompt_bounded_int(" Select option number: ", 1, len(options))
    return options[choice_idx - 1]

def handle_train_models():
    """Option 1: Trains and benchmarks all candidate models."""
    print_banner("OPTION 1: TRAIN & CROSS-VALIDATE MODELS")
    try:
        best_name, _, comparison_df, _ = train_and_evaluate_all()
        print("\n [Success] Model training and evaluation completed successfully!")
        print(f" [Selected Model] {best_name}")
        print(" [Model Checkpoint] Saved to models/student_performance_model.joblib")
    except Exception as e:
        print(f"\n [Error] Training failed: {e}")

def handle_evaluate_models():
    """Option 2: Displays model evaluation comparison table."""
    print_banner("OPTION 2: MODEL EVALUATION BENCHMARKS")
    if not os.path.exists(METRICS_PATH):
        print(" [Notice] No trained models found yet.")
        train_now = input(" Would you like to train models now? (y/n): ").strip().lower()
        if train_now == 'y':
            handle_train_models()
        return

    try:
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)

        records = data.get('comparison_summary', [])
        best = data.get('best_model', 'Unknown')
        criterion = data.get('selection_criterion', 'Highest Test R²')

        df = pd.DataFrame(records)
        print(" HOLDOUT TEST SET PERFORMANCE COMPARISON:")
        print("=" * 65)
        print(df.to_string(index=False))
        print("=" * 65)
        print(f"\n Best Selected Model : {best}")
        print(f" Selection Criterion : {criterion}")
    except Exception as e:
        print(f" [Error] Failed to read evaluation metrics: {e}")

def handle_predict_student():
    """Option 3: Interactive single-student performance prediction."""
    print_banner("OPTION 3: PREDICT STUDENT PERFORMANCE")

    if not os.path.exists(MODEL_PATH):
        print(" [Warning] Trained model not found at models/student_performance_model.joblib")
        print(" Machine learning prediction requires a pre-trained model checkpoint.")
        train_choice = input(" Would you like to train the model now? (y/n): ").strip().lower()
        if train_choice == 'y':
            handle_train_models()
        else:
            return

    try:
        model = load_trained_model()
    except Exception as e:
        print(f" [Error] Could not load model: {e}")
        return

    print("\n Please enter the student's academic and behavioral indicators:")
    print("-" * 60)

    attendance = prompt_bounded_float(" Attendance Rate (0.0 - 100.0%): ", 0.0, 100.0)
    study_hours = prompt_bounded_float(" Weekly Self-Study Hours (0 - 40 hrs): ", 0.0, 100.0)
    prev_score = prompt_bounded_float(" Previous Semester / Prerequisite Score (0 - 100): ", 0.0, 100.0)
    assignment = prompt_bounded_float(" Assignment Completion Rate (0.0 - 100.0%): ", 0.0, 100.0)
    internal = prompt_bounded_float(" Internal / Midterm Assessment Score (0 - 100): ", 0.0, 100.0)
    participation = prompt_bounded_int(" Classroom Participation Rating (1 [Low] to 5 [High]): ", 1, 5)

    parental_edu = prompt_choice("Parental Education Level:", VALID_PARENTAL_EDU)
    internet = prompt_choice("Home Internet Access:", ["Yes", "No"])
    extra = prompt_choice("Extra-Curricular Participation:", ["Yes", "No"])

    student_data = {
        'Attendance_Rate': attendance,
        'Study_Hours_Per_Week': study_hours,
        'Previous_Score': prev_score,
        'Assignment_Completion_Rate': assignment,
        'Internal_Assessment_Score': internal,
        'Class_Participation': participation,
        'Parental_Education_Level': parental_edu,
        'Internet_Access': internet,
        'Extra_Curricular': extra
    }

    print("\n Running predictive inference through trained pipeline...")
    try:
        prediction_result = predict_single_student(student_data, pipeline=model)
        print("\n" + format_prediction_output(prediction_result))
    except Exception as e:
        print(f" [Error] Prediction failed: {e}")

def handle_view_analytics():
    """Option 4: Generates and displays diagnostic analytics and plot paths."""
    print_banner("OPTION 4: VIEW ANALYTICS & DIAGNOSTICS")
    print(" Generating diagnostic plots (Feature Importance, Residuals, Correlations)...")
    try:
        run_all_analytics()
        print("\n [Success] Analytics plots updated successfully under 'outputs/plots/':")
        for f in os.listdir(PLOTS_DIR):
            if f.endswith('.png'):
                print(f"   • outputs/plots/{f}")

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                data = json.load(f)
            best = data.get('best_model', 'Unknown')
            summary = data.get('comparison_summary', [])
            print(f"\n Current Production Model: {best}")
            print(f" Evaluated Architectures : {len(summary)} algorithms")
    except Exception as e:
        print(f" [Error] Analytics generation failed: {e}")

def handle_generate_report():
    """Option 5: Generates comprehensive Markdown and JSON reports."""
    print_banner("OPTION 5: GENERATE COMPREHENSIVE REPORT")
    print(" Compiling experimental report from empirical execution data...")
    try:
        report_content = generate_project_report()
        report_file = os.path.join(REPORTS_DIR, 'project_report.md')
        print(f"\n [Success] Formal academic report generated at: {report_file}")
        print(f" [Success] Machine-readable JSON saved at: {os.path.join(REPORTS_DIR, 'project_summary.json')}")
        print("\n Executive Summary Preview:")
        print("-" * 60)
        lines = report_content.split('\n')
        # Display first 25 lines of report
        print("\n".join(lines[:25]))
        print("...\n [Full report available in outputs/reports/project_report.md]")
    except Exception as e:
        print(f" [Error] Report generation failed: {e}")

def main():
    """Main application loop."""
    ensure_directories()

    while True:
        print(MENU_HEADER)
        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            handle_train_models()
        elif choice == '2':
            handle_evaluate_models()
        elif choice == '3':
            handle_predict_student()
        elif choice == '4':
            handle_view_analytics()
        elif choice == '5':
            handle_generate_report()
        elif choice == '6':
            print("\nExiting Student Performance Prediction System. Goodbye!\n")
            break
        else:
            print("\n [Error] Invalid choice. Please enter a number between 1 and 6.")
        
        input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    main()
