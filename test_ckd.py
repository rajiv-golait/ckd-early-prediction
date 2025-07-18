import requests
from bs4 import BeautifulSoup

# URL of your Flask app
url = "http://127.0.0.1:5000/predict"

# List of test cases (both healthy and CKD)
test_cases = [
    # Real CKD case (row 0)
    {
        "age": 48.0, "blood_pressure": 80.0, "specific_gravity": 1.02, "albumin": 1.0, "sugar": 0.0,
        "red_blood_cells": 1, "pus_cell": 1, "pus_cell_clumps": 0, "bacteria": 0,
        "blood_glucose_random": 121.0, "blood_urea": 36.0, "serum_creatinine": 1.2, "sodium": 138.0, "potassium": 4.5,
        "hemoglobin": 15.4, "packed_cell_volume": 44, "white_blood_cell_count": 7800, "red_blood_cell_count": 5.2,
        "hypertension": 1, "diabetesmellitus": 1, "coronary_artery_disease": 0, "appetite": 0, "pedal_edema": 0, "anemia": 0
    },
    # Real non-CKD case (row 250)
    {
        "age": 40.0, "blood_pressure": 80.0, "specific_gravity": 1.025, "albumin": 0.0, "sugar": 0.0,
        "red_blood_cells": 1, "pus_cell": 1, "pus_cell_clumps": 0, "bacteria": 0,
        "blood_glucose_random": 140.0, "blood_urea": 10.0, "serum_creatinine": 1.2, "sodium": 135.0, "potassium": 5.0,
        "hemoglobin": 15.0, "packed_cell_volume": 48, "white_blood_cell_count": 10400, "red_blood_cell_count": 4.5,
        "hypertension": 0, "diabetesmellitus": 0, "coronary_artery_disease": 0, "appetite": 0, "pedal_edema": 0, "anemia": 0
    },
    # Real non-CKD case (row 251)
    {
        "age": 23.0, "blood_pressure": 80.0, "specific_gravity": 1.025, "albumin": 0.0, "sugar": 0.0,
        "red_blood_cells": 1, "pus_cell": 1, "pus_cell_clumps": 0, "bacteria": 0,
        "blood_glucose_random": 70.0, "blood_urea": 36.0, "serum_creatinine": 1.0, "sodium": 150.0, "potassium": 4.6,
        "hemoglobin": 17.0, "packed_cell_volume": 52, "white_blood_cell_count": 9800, "red_blood_cell_count": 5.0,
        "hypertension": 0, "diabetesmellitus": 0, "coronary_artery_disease": 0, "appetite": 0, "pedal_edema": 0, "anemia": 0
    },
    # Healthy case
    {
        "age": 50, "blood_pressure": 80, "specific_gravity": 1.02, "albumin": 1, "sugar": 0,
        "red_blood_cells": 1, "pus_cell": 1, "pus_cell_clumps": 0, "bacteria": 0,
        "blood_glucose_random": 100, "blood_urea": 30, "serum_creatinine": 1.2, "sodium": 140, "potassium": 4.5,
        "hemoglobin": 15, "packed_cell_volume": 40, "white_blood_cell_count": 8000, "red_blood_cell_count": 5,
        "hypertension": 0, "diabetesmellitus": 0, "coronary_artery_disease": 0, "appetite": 1, "pedal_edema": 0, "anemia": 0
    },
    # CKD case (high risk)
    {
        "age": 65, "blood_pressure": 150, "specific_gravity": 1.005, "albumin": 4, "sugar": 3,
        "red_blood_cells": 0, "pus_cell": 0, "pus_cell_clumps": 1, "bacteria": 1,
        "blood_glucose_random": 350, "blood_urea": 120, "serum_creatinine": 8.5, "sodium": 130, "potassium": 6.2,
        "hemoglobin": 7, "packed_cell_volume": 22, "white_blood_cell_count": 18000, "red_blood_cell_count": 2.5,
        "hypertension": 1, "diabetesmellitus": 1, "coronary_artery_disease": 1, "appetite": 0, "pedal_edema": 1, "anemia": 1
    },
    # Borderline case
    {
        "age": 45, "blood_pressure": 90, "specific_gravity": 1.015, "albumin": 2, "sugar": 1,
        "red_blood_cells": 1, "pus_cell": 1, "pus_cell_clumps": 0, "bacteria": 0,
        "blood_glucose_random": 140, "blood_urea": 45, "serum_creatinine": 1.8, "sodium": 137, "potassium": 5.0,
        "hemoglobin": 12, "packed_cell_volume": 36, "white_blood_cell_count": 9500, "red_blood_cell_count": 4.2,
        "hypertension": 0, "diabetesmellitus": 1, "coronary_artery_disease": 0, "appetite": 1, "pedal_edema": 0, "anemia": 0
    },
    # CKD case (moderate risk)
    {
        "age": 60, "blood_pressure": 140, "specific_gravity": 1.01, "albumin": 3, "sugar": 2,
        "red_blood_cells": 0, "pus_cell": 0, "pus_cell_clumps": 1, "bacteria": 1,
        "blood_glucose_random": 250, "blood_urea": 80, "serum_creatinine": 4.0, "sodium": 132, "potassium": 5.8,
        "hemoglobin": 9, "packed_cell_volume": 28, "white_blood_cell_count": 12000, "red_blood_cell_count": 3.0,
        "hypertension": 1, "diabetesmellitus": 1, "coronary_artery_disease": 0, "appetite": 0, "pedal_edema": 1, "anemia": 1
    },
    # Healthy case (young)
    {
        "age": 30, "blood_pressure": 75, "specific_gravity": 1.025, "albumin": 0, "sugar": 0,
        "red_blood_cells": 1, "pus_cell": 1, "pus_cell_clumps": 0, "bacteria": 0,
        "blood_glucose_random": 90, "blood_urea": 18, "serum_creatinine": 0.9, "sodium": 142, "potassium": 4.2,
        "hemoglobin": 16, "packed_cell_volume": 44, "white_blood_cell_count": 7000, "red_blood_cell_count": 5.2,
        "hypertension": 0, "diabetesmellitus": 0, "coronary_artery_disease": 0, "appetite": 1, "pedal_edema": 0, "anemia": 0
    },
]

for idx, data in enumerate(test_cases, 1):
    response = requests.post(url, data=data)
    soup = BeautifulSoup(response.text, "html.parser")
    prediction = soup.find("p", class_="positive") or soup.find("p", class_="negative")
    prob_div = soup.find("div", class_="prob")
    prob = None
    if prob_div and "CKD Probability:" in prob_div.text:
        try:
            prob = float(prob_div.text.split(":")[-1].strip())
        except:
            pass
    print(f"\nTest Case {idx}:")
    print("Input:", data)
    print("Prediction:", prediction.text.strip() if prediction else "Not found")
    print("Probability:", prob_div.text.strip() if prob_div else "Not found")
    if prob is not None and 0.4 <= prob < 0.6:
        print("[Warning] The model is uncertain about this prediction. Please consult a doctor.") 