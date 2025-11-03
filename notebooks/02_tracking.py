#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import mlflow
from math import sqrt


# In[2]:


df = pd.read_csv("../data/cleaned_insurance.csv")
df.head()


# In[3]:


target = "monthly_premium"


# In[4]:


X = df.drop(columns=[target])
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()


# In[6]:


from sklearn.pipeline import Pipeline

numeric = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ]
)


# In[7]:


from sklearn.linear_model import LinearRegression

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

model.fit(X_train, y_train)


# In[8]:


preds = model.predict(X_val)

rmse = sqrt(mean_squared_error(y_val, preds))
mae = mean_absolute_error(y_val, preds)
r2 = r2_score(y_val, preds)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.3f}")


# In[9]:


mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("insurance-cost-prediction")

with mlflow.start_run(run_name="baseline_linreg"):
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # log model artifact
    mlflow.sklearn.log_model(model, name="model")

print("Run logged to MLflow.")


# In[11]:


from sklearn.ensemble import RandomForestRegressor

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)

rmse = sqrt(mean_squared_error(y_val, rf_preds))
mae = mean_absolute_error(y_val, rf_preds)
r2 = r2_score(y_val, rf_preds)

with mlflow.start_run(run_name="random_forest"):
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(rf_model, name="model")

print(f"RF RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")


# In[17]:


def run_experiment(
    X_train, X_val, y_train, y_val,
    preprocessor,
    n_estimators, max_depth, min_samples_leaf, random_state=42
):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state
        ))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    with mlflow.start_run(run_name=f"RF_ne{n_estimators}_md{max_depth}_ml{min_samples_leaf}"):
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, name="model")

    print(f"→ n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_leaf={min_samples_leaf} | RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")


# In[14]:


import itertools

n_estimators_list = [100, 200, 300]
max_depth_list = [8, 12, None]
min_samples_leaf_list = [1, 3, 5]

param_grid = itertools.product(n_estimators_list, max_depth_list, min_samples_leaf_list)


# In[18]:


for n_estimators, max_depth, min_samples_leaf in param_grid:
    run_experiment(
        X_train, X_val, y_train, y_val,
        preprocessor,
        n_estimators, max_depth, min_samples_leaf
    )