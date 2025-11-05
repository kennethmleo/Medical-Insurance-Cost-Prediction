#!/usr/bin/env python
# coding: utf-8

from prefect import flow, task
from src.models.train import train_pipeline


@task
def load_data():
    print("✅ Data ready: data/cleaned_insurance.csv")
    return "data/cleaned_insurance.csv"


@task
def train_model(data_path: str):
    print("🚀 Starting model training...")
    train_pipeline(data_path=data_path, model_name="rf")
    print("✅ Training complete!")


@flow(name="insurance_training_flow")
def main_flow():
    """Main Prefect flow for training insurance cost prediction model."""
    data_path = load_data()
    train_model(data_path)


if __name__ == "__main__":
    main_flow()
