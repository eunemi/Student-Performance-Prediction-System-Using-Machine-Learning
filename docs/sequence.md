# Sequence Diagram Documentation

## 1. Inference & Advisory Sequence
Illustrates the exact temporal message flow during an interactive student prediction session.

---

## 2. Sequence Diagram (Mermaid)

```mermaid
sequenceDiagram
    autonumber
    actor User as Student / Faculty
    participant CLI as main.py (CLI)
    participant Validator as src/prediction.py (Validator)
    participant Joblib as models/ (Joblib Store)
    participant Model as Pipeline (Model Estimator)
    participant Risk as src/risk_analysis.py
    participant Rec as src/recommendation_engine.py

    User ->> CLI: Select Option 3: Predict Student Performance
    CLI ->> Joblib: Check if model checkpoint exists
    Joblib -->> CLI: Confirm model file present
    
    loop Input Collection & Validation
        User ->> CLI: Provide student attributes (Attendance, Study Hours, etc.)
        CLI ->> Validator: validate_student_input(data)
        alt Invalid Input (e.g. Attendance > 100%)
            Validator -->> CLI: Raise ValueError
            CLI -->> User: Display error and re-prompt for valid input
        else Valid Input
            Validator -->> CLI: Return sanitized input dictionary
        end
    end

    CLI ->> Joblib: load_trained_model()
    Joblib -->> CLI: Return deserialized Pipeline
    CLI ->> Model: predict(input_df)
    Model -->> CLI: Return raw continuous score (e.g. 78.42)
    
    CLI ->> Risk: categorize_performance(score)
    Risk -->> CLI: Return tier metadata (e.g. GOOD, On Track)
    
    CLI ->> Rec: generate_recommendations(student_data, risk_meta)
    Rec -->> CLI: Return tailored pedagogical advice list
    
    CLI -->> User: Render formatted Prediction Result banner & Advisory
```
