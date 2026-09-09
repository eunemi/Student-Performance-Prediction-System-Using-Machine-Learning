# Component & Class Diagram Documentation

## 1. Architectural Components
The software architecture is modularized into distinct functional components conforming to single-responsibility principles.

---

## 2. Component Diagram (Mermaid)

```mermaid
classDiagram
    class DataLoader {
        +load_raw_data(file_path) DataFrame
        +validate_dataset_ranges(df) void
        +split_data(df, target_col, test_size, random_state) Tuple
    }

    class AcademicFeatureEngineer {
        +add_engagement bool
        +add_momentum bool
        +add_efficiency bool
        +fit(X, y) AcademicFeatureEngineer
        +transform(X) DataFrame
    }

    class PreprocessingBuilder {
        +build_preprocessor() ColumnTransformer
        +build_full_pipeline(estimator) Pipeline
    }

    class ModelTrainer {
        +get_candidate_models() Dict
        +train_and_evaluate_all(file_path) Tuple
    }

    class ModelEvaluator {
        +calculate_metrics(y_true, y_pred) Dict
        +generate_comparison_table(results) DataFrame
        +save_evaluation_results(comp_df, best_name, full_res) void
    }

    class StudentPredictor {
        +load_trained_model(path) Pipeline
        +validate_student_input(data) Dict
        +predict_single_student(data, pipeline) Dict
        +format_prediction_output(result) str
    }

    class RiskAnalyzer {
        +categorize_performance(score) Dict
    }

    class RecommendationEngine {
        +generate_recommendations(student_data, performance_meta) List~str~
    }

    class AnalyticsEngine {
        +generate_feature_importance_plot(df) DataFrame
        +generate_residual_analysis_plot(pipeline, X_test, y_test) void
        +generate_model_comparison_plot() void
        +run_all_analytics() void
    }

    class ReportGenerator {
        +dataframe_to_markdown(df) str
        +generate_project_report() str
    }

    DataLoader ..> AcademicFeatureEngineer : feeds raw splits
    AcademicFeatureEngineer ..> PreprocessingBuilder : embedded in pipeline
    PreprocessingBuilder ..> ModelTrainer : provides full pipeline
    ModelTrainer ..> ModelEvaluator : passes predictions
    ModelTrainer ..> StudentPredictor : serializes Joblib model
    StudentPredictor ..> RiskAnalyzer : passes predicted score
    StudentPredictor ..> RecommendationEngine : passes inputs & category
    ModelEvaluator ..> AnalyticsEngine : provides metrics
    ModelEvaluator ..> ReportGenerator : provides comparison tables
```
