# Machine Learning Workflow Documentation

## 1. End-to-End Workflow Process
The system follows a rigorous supervised machine learning lifecycle from raw tabular data ingestion to holdout evaluation and real-time advisory delivery.

---

## 2. Workflow Diagram (Mermaid)

```mermaid
flowchart TD
    A[Raw Student Dataset<br>data/raw/student_performance_data.csv] --> B[Schema & Boundary Validation<br>src/data_loader.py]
    B --> C[Exploratory Data Analysis<br>notebooks/exploratory_analysis.ipynb]
    B --> D[Holdout Train-Test Split 80/20<br>Fixed Random State 42]
    
    subgraph Feature Processing
        D --> E[Academic Feature Engineering<br>Engagement, Momentum, Efficiency]
        E --> F[Missing Value Imputation<br>Median for Numeric, Mode for Categorical]
        F --> G[Standardization & One-Hot Encoding<br>StandardScaler, OneHotEncoder]
    end

    subgraph Model Training & Benchmarking
        G --> H1[Linear Regression Baseline]
        G --> H2[Decision Tree Regressor]
        G --> H3[Random Forest Regressor]
        G --> H4[Gradient Boosting Regressor]
        
        H1 --> I[5-Fold Cross-Validation<br>Track R² Mean & Std]
        H2 --> I
        H3 --> I
        H4 --> I
    end

    subgraph Model Evaluation & Selection
        I --> J[Evaluate on 20% Unseen Test Set<br>MAE, MSE, RMSE, R²]
        J --> K[Optimal Model Selection<br>Criterion: Highest R² & Lowest RMSE]
        K --> L[Model Persistence<br>models/student_performance_model.joblib]
    end

    subgraph Runtime Inference & Intervention
        L --> M[New Student Input via CLI<br>Input Validation Guards]
        M --> N[Predict Continuous Exam Score]
        N --> O[Categorize Academic Performance Tier<br>Excellent, Good, Average, Needs Improvement]
        O --> P[Synthesize Rule-Based Recommendations<br>Attendance, Study Hours, Midterm Interventions]
        P --> Q[Display Formatted Advisory Result]
    end
```
