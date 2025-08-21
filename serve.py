from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat
from joblib import load
import pandas as pd

app = FastAPI(title="Student Performance Predictor")

class Features(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24)
    attendance_rate: confloat(ge=0, le=1)  # 0..1
    assignments_completed: conint(ge=0, le=100)
    past_grade: confloat(ge=0, le=100)
    social_activity_hours: confloat(ge=0, le=24)
    sleep_hours: confloat(ge=0, le=24)
    school_type: str  # "public" or "private"
    parent_education: str  # "hs" | "bachelors" | "masters" | "phd"

def get_model():
    return load("artifacts/model.joblib")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: Features):
    model = get_model()
    df = pd.DataFrame([features.dict()])
    y_pred = float(model.predict(df)[0])
    return {"prediction": y_pred}
