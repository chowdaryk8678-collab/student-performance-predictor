from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

TARGET_COL = "final_grade"

@dataclass
class DataConfig:
    path: str

def load_dataset(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")
    return df
