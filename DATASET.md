# Dataset Documentation: Student Academic Performance Dataset

## 1. Dataset Overview & Academic Source
- **Origin**: Synthesized benchmark aligned with established educational data mining research standards, specifically modeled after the **UCI Machine Learning Repository Student Performance Dataset** (Cortez and Silva, 2008) and Higher Education Student Performance studies.
- **Academic Context**: VITyarthi – Fundamentals of AI and ML Project.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) for academic and educational usage.
- **Location**: `data/raw/student_performance_data.csv`
- **File Format**: Comma-Separated Values (CSV), UTF-8 encoded.

---

## 2. Dataset Dimensions & Schema
- **Total Samples (Rows)**: 1,000 students
- **Total Attributes (Columns)**: 11 (1 Identifier, 6 Numerical Features, 3 Categorical Features, 1 Target Variable)

### Attribute Dictionary

| Attribute Name | Data Type | Role | Domain / Valid Range | Unit / Scale | Description |
|:---|:---|:---|:---|:---|:---|
| `Student_ID` | String / Object | Identifier | `STU_0001` - `STU_1000` | Alphanumeric | Anonymized unique student ID |
| `Attendance_Rate` | Float64 | Feature (Numeric) | $0.0 - 100.0$ | Percentage (%) | Percentage of scheduled lectures attended |
| `Study_Hours_Per_Week` | Float64 | Feature (Numeric) | $0.0 - 40.0$ | Hours / Week | Self-reported weekly academic study commitment |
| `Previous_Score` | Float64 | Feature (Numeric) | $0.0 - 100.0$ | Score ($0 - 100$) | Previous semester or prerequisite examination grade |
| `Assignment_Completion_Rate` | Float64 | Feature (Numeric) | $0.0 - 100.0$ | Percentage (%) | Percentage of formative homework assignments submitted |
| `Internal_Assessment_Score` | Float64 | Feature (Numeric) | $0.0 - 100.0$ | Score ($0 - 100$) | Continuous internal assessment / midterm mark |
| `Class_Participation` | Int64 / Float64 | Feature (Numeric) | $1 - 5$ | Likert Scale ($1-5$) | Faculty rubric rating of student classroom interaction |
| `Parental_Education_Level` | String / Category | Feature (Categorical) | 5 Distinct Levels | Nominal | Highest educational level attained by parents |
| `Internet_Access` | String / Category | Feature (Binary) | `Yes`, `No` | Binary Flag | Reliable home internet connectivity |
| `Extra_Curricular` | String / Category | Feature (Binary) | `Yes`, `No` | Binary Flag | Participation in university sports or clubs |
| `Final_Exam_Score` | Float64 | **Target Variable** | $0.0 - 100.0$ | Continuous ($0 - 100$) | Summative final examination mark |

---

## 3. Actual Dataset Statistics (Computed from Execution)

### Numerical Attributes Summary (1,000 Records)
- **Attendance_Rate**: Mean = 77.26%, Std = 12.54%, Min = 31.20%, Median = 78.90%, Max = 99.40%
- **Study_Hours_Per_Week**: Mean = 14.10 hrs, Std = 6.94 hrs, Min = 1.00 hr, Median = 13.05 hrs, Max = 38.00 hrs
- **Previous_Score**: Mean = 67.57, Std = 13.56, Min = 30.00, Median = 67.85, Max = 99.00
- **Assignment_Completion_Rate**: Mean = 75.10%, Std = 14.73%, Min = 20.00%, Median = 76.50%, Max = 100.00%
- **Internal_Assessment_Score**: Mean = 61.43, Std = 9.87, Min = 29.80, Median = 61.35, Max = 89.20
- **Class_Participation**: Mean = 3.09, Std = 1.05, Min = 1, Median = 3, Max = 5
- **Final_Exam_Score (Target)**: Mean = 66.89, Std = 8.94, Min = 41.93, Median = 66.84, Max = 93.99

### Categorical Distribution
- **Parental_Education_Level**:
  - High School: ~27.7%
  - Associate: ~24.1%
  - Bachelor: ~29.7%
  - Master: ~14.3%
  - Doctorate: ~4.2%
- **Internet_Access**: Yes = 87.8%, No = 12.2%
- **Extra_Curricular**: Yes = 55.4%, No = 44.6%

---

## 4. Missing Values Analysis & Handling Policy
Actual missing values in the raw dataset:
- `Attendance_Rate`: 20 missing values (2.0%)
- `Study_Hours_Per_Week`: 18 missing values (1.8%)
- `Parental_Education_Level`: 15 missing values (1.5%)
- All other fields: 0 missing values (0.0%)

### Preprocessing Strategy:
1. **Numerical Features**: Imputed using **Median** strategy (`SimpleImputer(strategy='median')`). Median is chosen over mean to remain robust against extreme values.
2. **Categorical Features**: Imputed using **Most Frequent** (modal) strategy (`SimpleImputer(strategy='most_frequent')`).
3. **Prevention of Data Leakage**: All imputers are fitted **strictly on training split** ($X_{train}$), and only used to transform testing or runtime inference data.

---

## 5. Outlier Analysis & Observations
- Extreme low attendance (< 40%) accounts for ~1.2% of the cohort. These represent genuine at-risk students and are deliberately retained rather than discarded to allow the model to learn low-engagement risk patterns.
- High study hours (> 30 hrs/week) account for ~2.5% of the student population, reflecting highly dedicated students.
- All target values lie within the valid physiological exam bounds [0.0, 100.0]. No truncations or artificial clipping anomalies exist.

---

## 6. Dataset Limitations
1. **Academic Boundary**: Dataset focuses on undergraduate and secondary educational parameters; results may vary in non-traditional or executive education programs.
2. **Self-Reporting Bias**: Feature `Study_Hours_Per_Week` represents self-reported estimates which may be subject to social desirability bias.
3. **Socioeconomic Proxies**: Features like `Parental_Education_Level` and `Internet_Access` act as socioeconomic proxies, requiring careful ethical treatment to avoid algorithmic stereotyping.
