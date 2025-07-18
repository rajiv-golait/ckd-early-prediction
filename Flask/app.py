from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

# Load the saved model
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'CKD.pkl'), 'rb'))
# Load the scaler used during training
scaler = pickle.load(open(os.path.join(os.path.dirname(__file__), 'scaler.pkl'), 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read all 24 input values from the form in the correct order
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = [
        'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
        'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
        'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium',
        'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
        'hypertension', 'diabetesmellitus', 'coronary_artery_disease', 'appetite',
        'pedal_edema', 'anemia'
    ]
    df = pd.DataFrame(features_value, columns=features_name)
    print('Input DataFrame for prediction:')
    print(df)
    df_scaled = scaler.transform(df)
    proba = model.predict_proba(df_scaled)[0, 1]
    print('CKD probability:', proba)
    if proba >= 0.5:
        prediction_text = f"Oops! You have Chronic Kidney Disease. (Probability: {proba:.2f})"
    elif 0.4 <= proba < 0.5:
        prediction_text = f"Warning: The prediction is uncertain. Please consult a doctor. (Probability: {proba:.2f})"
    else:
        prediction_text = f"Great! You DON'T have Chronic Kidney Disease. (Probability: {proba:.2f})"
    return render_template('result.html', prediction_text=prediction_text, probability=f"{proba:.2f}")

if __name__ == '__main__':
    app.run(debug=True)