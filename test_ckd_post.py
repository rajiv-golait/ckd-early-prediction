import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "age": 60,
    "blood_pressure": 150,
    "specific_gravity": 1.01,
    "albumin": 3,
    "sugar": 2,
    "red_blood_cells": 0,
    "pus_cell": 0,
    "pus_cell_clumps": 1,
    "bacteria": 1,
    "blood_glucose_random": 180,
    "blood_urea": 80,
    "serum_creatinine": 5.2,
    "sodium": 132,
    "potassium": 5.8,
    "hemoglobin": 8.5,
    "packed_cell_volume": 28,
    "white_blood_cell_count": 11200,
    "red_blood_cell_count": 3.2,
    "hypertension": 1,
    "diabetesmellitus": 1,
    "coronary_artery_disease": 1,
    "appetite": 0,
    "pedal_edema": 1,
    "anemia": 1
}
response = requests.post(url, data=data)
print(response.text)