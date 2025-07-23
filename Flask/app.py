from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os

# Load the saved model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'CKD.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

app = Flask(__name__)

# Feature order must match the notebook
FEATURES = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium',
    'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
    'hypertension', 'diabetesmellitus', 'coronary_artery_disease', 'appetite',
    'pedal_edema', 'anemia'
]

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure all features are present and in correct order
        input_features = []
        for feat in FEATURES:
            val = request.form.get(feat)
            if val is None:
                return render_template('result.html', prediction_text=f"Missing value for {feat}", probability="-")
            input_features.append(float(val))
        features_value = [np.array(input_features)]
        df = pd.DataFrame(features_value, columns=FEATURES)
        df.to_csv("debug_input.csv", index=False)  # For debugging
        df_scaled = scaler.transform(df)
        proba = model.predict_proba(df_scaled)[0, 1]
        if proba >= 0.5:
            prediction_text = f"Oops! You have Chronic Kidney Disease. (Probability: {proba:.2f})"
        elif 0.4 <= proba < 0.5:
            prediction_text = f"Warning: The prediction is uncertain. Please consult a doctor. (Probability: {proba:.2f})"
        else:
            prediction_text = f"Great! You DON'T have Chronic Kidney Disease. (Probability: {proba:.2f})"
        return render_template('result.html', prediction_text=prediction_text, probability=f"{proba:.2f}")
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}", probability="-")

if __name__ == '__main__':
    app.run(debug=True)