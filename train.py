from __future__ import annotations
import argparse
import os
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import DataConfig, load_dataset
from .pipeline import build_model, split_xy

def train(data_path: str, artifacts_dir: str = "artifacts", test_size: float = 0.2, random_state: int = 42):
    os.makedirs(artifacts_dir, exist_ok=True)

    cfg = DataConfig(path=data_path)
    df = load_dataset(cfg)

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    print("Evaluation:", metrics)

    # Save artifacts
    model_path = os.path.join(artifacts_dir, "model.joblib")
    dump(model, model_path)
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        import json; json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample.csv", help="Path to CSV dataset")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Directory to save artifacts")
    args = parser.parse_args()
    train(args.data, args.artifacts)
