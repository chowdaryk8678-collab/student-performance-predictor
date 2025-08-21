from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

FEATURES_NUM = [
    "hours_studied",
    "attendance_rate",
    "assignments_completed",
    "past_grade",
    "social_activity_hours",
    "sleep_hours",
]

FEATURES_CAT = [
    "school_type",
    "parent_education",
]

TARGET = "final_grade"

def build_preprocessor() -> ColumnTransformer:
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, FEATURES_NUM),
            ("cat", cat_transformer, FEATURES_CAT),
        ]
    )
    return preprocessor

def build_model(random_state: int = 42) -> Pipeline:
    pre = build_preprocessor()
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("preprocessor", pre), ("model", rf)])
    return pipe

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].copy()
    return X, y
