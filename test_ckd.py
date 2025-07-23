import requests

url = "http://127.0.0.1:5000/predict"

# Example CKD-positive test case (matches new model's feature order and encoding)
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

try:
    response = requests.post(url, data=test_data)
    print("Status Code:", response.status_code)
    print("Response:")
    print(response.text)
except Exception as e:
    print(f"Error during test: {e}") 