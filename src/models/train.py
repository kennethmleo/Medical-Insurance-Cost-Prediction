#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import mlflow
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_pipeline(
    data_path: str = "data/cleaned_insurance.csv",
    model_name: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train and log a model with MLflow."""

    # --- Load data ---
    df = pd.read_csv(data_path)
    target = "monthly_premium"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Preprocessing ---
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )

    # --- Choose model ---
    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        )
    elif model_name == "linreg":
        model = LinearRegression()
    else:
        raise ValueError("model_name must be 'rf' or 'linreg'")

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # --- Train ---
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    # --- Log to MLflow ---
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("insurance-cost-prediction")

    with mlflow.start_run(run_name=f"{model_name}_run"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

    print(f"{model_name.upper()} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f}")


# --- CLI entry point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train medical insurance cost model")
    parser.add_argument("--data_path", type=str, default="data/cleaned_insurance.csv")
    parser.add_argument("--model_name", type=str, default="rf", help="rf or linreg")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    train_pipeline(
        data_path=args.data_path,
        model_name=args.model_name,
        test_size=args.test_size,
        random_state=args.random_state,
    )
