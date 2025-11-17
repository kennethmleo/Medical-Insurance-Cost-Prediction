#!/usr/bin/env python
# coding: utf-8

import mlflow
from mlflow.exceptions import MlflowException

def check_registry(uri="sqlite:///mlflow.db"):
    print(f"🔍 Checking MLflow model registry at: {uri}\n")
    mlflow.set_tracking_uri(uri)

    try:
        client = mlflow.tracking.MlflowClient()
        models = client.get_registered_model('best RF model')

        if not models:
            print("⚠️  No registered models found.")
            return

        for m in models:
            print(f"📦 Model: {m.name}")
            for v in m.latest_versions:
                aliases = getattr(v, "aliases", [])
                print(f"   - Version: {v.version}")
                print(f"     Aliases: {aliases if aliases else 'None'}")
                print(f"     Source:  {v.source}")
                print()
        print("✅ Registry check complete!\n")

    except MlflowException as e:
        print(f"❌ Error accessing MLflow registry: {e}")

if __name__ == "__main__":
    check_registry("sqlite:///mlflow.db")
