import requests

sample = {
  "age": 79,
  "sex": "Female",
  "region": "North",
  "urban_rural": "Urban",
  "income": 12800.0,
  "education": "No HS",
  "marital_status": "Married",
  "employment_status": "Employed",
  "household_size": 3,
  "dependents": 1,
  "bmi": 26.6,
  "smoker": "Never",
  "alcohol_freq": "Weekly",
  "visits_last_year": 2,
  "hospitalizations_last_3yrs": 0,
  "medication_count": 3,
  "systolic_bp": 131.0,
  "diastolic_bp": 79.0,
  "ldl": 97.3,
  "hba1c": 4.82,
  "plan_type": "POS",
  "network_tier": "Gold",
  "deductible": 1000,
  "copay": 10,
  "policy_term_years": 1,
  "provider_quality": 3.1,
  "risk_score": 1.0,
  "chronic_count": 2,
  "hypertension": 0,
  "diabetes": 0,
  "asthma": 0,
  "cardiovascular_disease": 0,
  "mental_health": 1
}

url = "http://localhost:9696/predict"
response = requests.post(url, json=sample)
print(response.json())
