import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

print("Starting model retraining...")

# Load dataset
data = pd.read_csv('dataset/kidney_disease.csv')
print(f"Dataset loaded. Shape: {data.shape}")

# Rename columns
data.columns = [
    'id', 'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium',
    'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
    'hypertension', 'diabetesmellitus', 'coronary_artery_disease', 'appetite',
    'pedal_edema', 'anemia', 'class'
]
data = data.drop('id', axis=1)

# Convert numeric columns
numeric_cols = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium',
    'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count'
]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].mean())

# Fill categorical columns with mode
categorical_cols = [
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'hypertension',
    'diabetesmellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia', 'class'
]
for col in categorical_cols:
    data[col] = data[col].astype(str).str.strip().str.lower()
    data[col] = data[col].fillna(data[col].mode()[0])

# Standardize class values
print("Class distribution before cleaning:")
print(data['class'].value_counts())

# Clean up class values
data['class'] = data['class'].replace({'ckd\t': 'ckd', 'notckd': 'notckd'})
print("Class distribution after cleaning:")
print(data['class'].value_counts())

# Properly encode binary features with meaningful mappings
# For categorical features, we need to be explicit about the encoding

# Binary mappings that make sense
binary_mappings = {
    'red_blood_cells': {'normal': 1, 'abnormal': 0},
    'pus_cell': {'normal': 1, 'abnormal': 0}, 
    'pus_cell_clumps': {'notpresent': 0, 'present': 1},
    'bacteria': {'notpresent': 0, 'present': 1},
    'hypertension': {'no': 0, 'yes': 1},
    'diabetesmellitus': {'no': 0, 'yes': 1},
    'coronary_artery_disease': {'no': 0, 'yes': 1},
    'appetite': {'poor': 0, 'good': 1},
    'pedal_edema': {'no': 0, 'yes': 1},
    'anemia': {'no': 0, 'yes': 1},
    'class': {'notckd': 0, 'ckd': 1}  # 0 = healthy, 1 = CKD
}

# Apply mappings
for col, mapping in binary_mappings.items():
    if col in data.columns:
        data[col] = data[col].map(mapping)
        # Fill any unmapped values with mode
        if data[col].isnull().any():
            mode_val = data[col].mode()[0] if not data[col].mode().empty else 0
            data[col] = data[col].fillna(mode_val)

print("Final class distribution:")
print(data['class'].value_counts())
print(f"CKD cases (class=1): {sum(data['class'] == 1)}")
print(f"Healthy cases (class=0): {sum(data['class'] == 0)}")

# Define features
features = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium',
    'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
    'hypertension', 'diabetesmellitus', 'coronary_artery_disease', 'appetite',
    'pedal_edema', 'anemia'
]

X = data[features]
y = data['class']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train model with balanced class weights
rfc = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
rfc.fit(X_train, y_train)

# Evaluate model
y_pred = rfc.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test with known CKD case
print("\n--- Testing with known cases ---")
ckd_cases = data[data['class'] == 1]
if len(ckd_cases) > 0:
    ckd_sample = ckd_cases.iloc[0][features]
    ckd_scaled = scaler.transform([ckd_sample])
    ckd_proba = rfc.predict_proba(ckd_scaled)[0, 1]
    print(f"Known CKD case probability: {ckd_proba:.3f}")

# Test with known healthy case  
healthy_cases = data[data['class'] == 0]
if len(healthy_cases) > 0:
    healthy_sample = healthy_cases.iloc[0][features]
    healthy_scaled = scaler.transform([healthy_sample])
    healthy_proba = rfc.predict_proba(healthy_scaled)[0, 1]
    print(f"Known healthy case probability: {healthy_proba:.3f}")

# Test with our test data
print("\n--- Testing with our specific test case ---")
test_data = {
    "age": 60,
    "blood_pressure": 150,
    "specific_gravity": 1.01,
    "albumin": 3,
    "sugar": 2,
    "red_blood_cells": 0,  # abnormal
    "pus_cell": 0,         # abnormal
    "pus_cell_clumps": 1,  # present
    "bacteria": 1,         # present
    "blood_glucose_random": 180,
    "blood_urea": 80,
    "serum_creatinine": 5.2,
    "sodium": 132,
    "potassium": 5.8,
    "hemoglobin": 8.5,
    "packed_cell_volume": 28,
    "white_blood_cell_count": 11200,
    "red_blood_cell_count": 3.2,
    "hypertension": 1,     # yes
    "diabetesmellitus": 1, # yes
    "coronary_artery_disease": 1, # yes
    "appetite": 0,         # poor
    "pedal_edema": 1,      # yes
    "anemia": 1            # yes
}

input_features = [test_data[feat] for feat in features]
test_df = pd.DataFrame([input_features], columns=features)
test_scaled = scaler.transform(test_df)
test_proba = rfc.predict_proba(test_scaled)[0, 1]
print(f"Test case CKD probability: {test_proba:.3f}")

# Save model and scaler
flask_dir = 'Flask'
if not os.path.exists(flask_dir):
    os.makedirs(flask_dir)
    
pickle.dump(rfc, open(os.path.join(flask_dir, 'CKD.pkl'), 'wb'))
pickle.dump(scaler, open(os.path.join(flask_dir, 'scaler.pkl'), 'wb'))
print(f"\nModel and scaler saved to {flask_dir} directory.")
print("Retraining completed successfully!")
