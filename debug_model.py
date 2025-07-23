import pickle
import numpy as np
import pandas as pd
import os

# Load the saved model and scaler
model_path = os.path.join('Flask', 'CKD.pkl')
scaler_path = os.path.join('Flask', 'scaler.pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

# Feature order must match the notebook
FEATURES = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium',
    'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
    'hypertension', 'diabetesmellitus', 'coronary_artery_disease', 'appetite',
    'pedal_edema', 'anemia'
]

# Test data from test_ckd.py (should indicate CKD)
test_data = {
    "age": 60,
    "blood_pressure": 150,
    "specific_gravity": 1.01,
    "albumin": 3,
    "sugar": 2,
    "red_blood_cells": 0,  # 0 = abnormal, 1 = normal
    "pus_cell": 0,         # 0 = abnormal, 1 = normal
    "pus_cell_clumps": 1,  # 1 = present, 0 = notpresent
    "bacteria": 1,         # 1 = present, 0 = notpresent
    "blood_glucose_random": 180,
    "blood_urea": 80,
    "serum_creatinine": 5.2,
    "sodium": 132,
    "potassium": 5.8,
    "hemoglobin": 8.5,
    "packed_cell_volume": 28,
    "white_blood_cell_count": 11200,
    "red_blood_cell_count": 3.2,
    "hypertension": 1,     # 1 = yes, 0 = no
    "diabetesmellitus": 1, # 1 = yes, 0 = no
    "coronary_artery_disease": 1, # 1 = yes, 0 = no
    "appetite": 0,         # 0 = poor, 1 = good
    "pedal_edema": 1,      # 1 = yes, 0 = no
    "anemia": 1            # 1 = yes, 0 = no
}

# Create input array in correct order
input_features = []
for feat in FEATURES:
    input_features.append(float(test_data[feat]))

print("Input features:", input_features)

# Create DataFrame and scale
features_value = [np.array(input_features)]
df = pd.DataFrame(features_value, columns=FEATURES)
print("DataFrame shape:", df.shape)
print("DataFrame:\n", df)

# Scale the features
df_scaled = scaler.transform(df)
print("Scaled features shape:", df_scaled.shape)
print("Scaled features:", df_scaled)

# Get prediction
prediction = model.predict(df_scaled)
print("Prediction:", prediction)

# Get probability
proba = model.predict_proba(df_scaled)
print("Prediction probabilities:", proba)
print("CKD probability (class 1):", proba[0, 1])

# Also test with a healthy case
print("\n--- Testing with healthy case ---")
healthy_data = {
    "age": 30,
    "blood_pressure": 80,
    "specific_gravity": 1.025,
    "albumin": 0,
    "sugar": 0,
    "red_blood_cells": 1,  # 1 = normal
    "pus_cell": 1,         # 1 = normal
    "pus_cell_clumps": 0,  # 0 = notpresent
    "bacteria": 0,         # 0 = notpresent
    "blood_glucose_random": 120,
    "blood_urea": 36,
    "serum_creatinine": 1.2,
    "sodium": 137,
    "potassium": 4.6,
    "hemoglobin": 15.4,
    "packed_cell_volume": 44,
    "white_blood_cell_count": 7800,
    "red_blood_cell_count": 5.2,
    "hypertension": 0,     # 0 = no
    "diabetesmellitus": 0, # 0 = no
    "coronary_artery_disease": 0, # 0 = no
    "appetite": 1,         # 1 = good
    "pedal_edema": 0,      # 0 = no
    "anemia": 0            # 0 = no
}

healthy_features = []
for feat in FEATURES:
    healthy_features.append(float(healthy_data[feat]))

healthy_df = pd.DataFrame([np.array(healthy_features)], columns=FEATURES)
healthy_scaled = scaler.transform(healthy_df)
healthy_proba = model.predict_proba(healthy_scaled)
print("Healthy case CKD probability:", healthy_proba[0, 1])
