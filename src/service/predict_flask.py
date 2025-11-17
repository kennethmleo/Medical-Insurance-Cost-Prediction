#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import os

# --- Load from MLflow registry ---
# --- Load from MLflow registry ---
MODEL_NAME = "best RF model"
MODEL_ALIAS = "champion"

# ✅ Detect local vs remote tracking URI
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
else:
    # Default to local tracking
    mlflow.set_tracking_uri("file:./mlruns")

print(f"🔍 Loading model '{MODEL_NAME}' ({MODEL_ALIAS}) from MLflow at {mlflow.get_tracking_uri()}...")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

app = Flask("insurance-predictor")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()

        # --- Accept either single record or list of records ---
        if isinstance(data, dict):
            df = pd.DataFrame([data])  # single record
        elif isinstance(data, list):
            df = pd.DataFrame(data)  # batch prediction
        else:
            return jsonify({"error": "Input must be a JSON object or list"}), 400

        preds = model.predict(df)

        # --- Return one or many ---
        results = [
            {"predicted_monthly_premium": round(float(p), 2)} for p in preds
        ]
        return jsonify(results if len(results) > 1 else results[0])

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
