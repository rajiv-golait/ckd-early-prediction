# Early Prediction for Chronic Kidney Disease Detection: A Progressive Approach to Health Management

## Project Description
Chronic Kidney Disease (CKD) is a major medical problem that can be managed or cured if detected early. Many routine medical tests contain valuable information that can help in the early detection of CKD. This project leverages machine learning to analyze such test results and predict the likelihood of CKD, supporting timely intervention and better health outcomes.

This project is part of an internship at [SmartInternz](https://nsel.smartinternz.com/).

## Real-World Scenarios
- **Scenario 1: Early Detection through Routine Tests**
  - A patient visits a clinic for a routine check-up. Abnormal creatinine and albumin levels in standard tests help detect CKD early, enabling timely treatment.
- **Scenario 2: Predicting Disease Severity and Survival**
  - A hospital uses AI to analyze patient data and predict disease severity and survival outcomes, helping doctors plan targeted treatments.
- **Scenario 3: Monitoring Disease Progression**
  - The system tracks key biomarkers over time for CKD patients, alerting physicians to worsening patterns and supporting proactive care.

## Project Structure
- `Flask/` - Flask web app for CKD prediction
- `Training/` - Jupyter notebook for data analysis and model training
- `dataset/` - CKD dataset
- `test_ckd.py` - Automated test script for the Flask app

## Setup Instructions
1. **Clone the repository and navigate to the project directory.**
2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
### 1. Train the Model
- Open `Training/Chronic_kidney_disease_analysis.ipynb` in Jupyter Notebook or JupyterLab.
- Run all cells to preprocess data, train, and save the model (`CKD.pkl`) and scaler (`scaler.pkl`) to the `Flask/` directory.

### 2. Start the Flask Web App
```bash
cd Flask
python app.py
```
- Open your browser and go to `http://127.0.0.1:5000/` to use the CKD prediction interface.

### 3. Automated Testing
- In the project root, run:
```bash
python test_ckd.py
```
- This will send multiple test cases to the Flask app and print predictions and probabilities for each.

## Working
- The web app collects 24 medical test features from the user.
- The input is preprocessed and scaled using the same pipeline as model training.
- The trained Random Forest model predicts the probability of CKD.
- The result page displays the prediction and probability, with a warning for uncertain cases.
- The test script automates validation using both real and synthetic test cases.

## Output Analysis
- The model achieves perfect accuracy on the notebook's test set, but real-world test cases show some false positives and false negatives.
- This highlights the importance of validating with real, unseen data and the limitations of small or imbalanced datasets.

## Improvement Suggestions
- Collect more diverse and balanced data for better generalization.
- Explore advanced feature engineering and domain-specific attributes.
- Consider more robust evaluation metrics and cross-validation.
- If allowed, try ensemble methods or advanced models (e.g., XGBoost) and oversampling techniques (e.g., SMOTE).
- Regularly update the model with new patient data.

## Internship Note
This project was developed as part of an experiential learning internship at [SmartInternz](https://nsel.smartinternz.com/). 