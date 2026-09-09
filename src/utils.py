"""
Utility functions, project paths, and shared constants.
"""

import os
import sys
import logging

# Shared Configuration Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = 'Final_Exam_Score'
IDENTIFIER_COLUMN = 'Student_ID'

# Base Paths (Relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUTS_DIR, 'metrics')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, 'student_performance_data.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'student_performance_model.joblib')
METRICS_PATH = os.path.join(METRICS_DIR, 'evaluation_metrics.json')
REPORT_PATH = os.path.join(REPORTS_DIR, 'project_report.md')

def ensure_directories():
    """Ensure all critical output and model directories exist."""
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)
        
    # Configure matplotlib cache directory locally to avoid sandbox write issues
    mpl_cache = os.path.join(PROJECT_ROOT, '.mpl_cache')
    os.makedirs(mpl_cache, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = mpl_cache

def setup_logger(name: str = "StudentPerformance") -> logging.Logger:
    """Configures a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def print_banner(title: str):
    """Prints a styled CLI section banner."""
    line = "=" * 60
    print(f"\n{line}")
    print(f" {title.center(58)} ")
    print(f"{line}\n")
