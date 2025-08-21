from __future__ import annotations
import argparse, json
from joblib import load

EXAMPLE = {
    "hours_studied": 2.5,
    "attendance_rate": 0.92,
    "assignments_completed": 10,
    "past_grade": 78.0,
    "social_activity_hours": 1.5,
    "sleep_hours": 7.0,
    "school_type": "public",
    "parent_education": "bachelors"
}

def predict(payload_path: str, model_path: str = "artifacts/model.joblib"):
    model = load(model_path)
    with open(payload_path, "r") as f:
        payload = json.load(f)
    import pandas as pd
    X = pd.DataFrame([payload])
    y_pred = model.predict(X)[0]
    print({"prediction": float(y_pred)})
    return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, default="examples/payload.json", help="Path to JSON payload")
    parser.add_argument("--model", type=str, default="artifacts/model.joblib", help="Path to trained model")
    args = parser.parse_args()
    predict(args.payload, args.model)
