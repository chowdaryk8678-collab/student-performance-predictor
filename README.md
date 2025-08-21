# Student Performance Predictor

A clean, production-ready starter project to **predict student final grades** with a scikit‑learn pipeline and serve predictions via **FastAPI**.

## ✨ Features
- End‑to‑end ML pipeline (train ➜ evaluate ➜ save).
- Robust preprocessing with `ColumnTransformer` (numeric scaling + categorical encoding).
- Model: `RandomForestRegressor` (easy to swap for XGBoost, LightGBM, etc.).
- CLI scripts:
  - `python src/train.py --data data/sample.csv` → trains and saves `artifacts/model.joblib` and `artifacts/preprocessor.joblib`
  - `python src/predict.py --payload examples/payload.json` → loads artifacts and prints prediction
  - `uvicorn src.serve:app --reload` → REST API for online inference
- Tiny offline **sample dataset** included (no internet required).
- Simple tests and lint-friendly structure.

## 🧰 Project Structure
```
student-performance-predictor/
├─ data/
│  └─ sample.csv
├─ src/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ pipeline.py
│  ├─ train.py
│  ├─ predict.py
│  └─ serve.py
├─ tests/
│  └─ test_smoke.py
├─ examples/
│  └─ payload.json
├─ artifacts/            # created after training
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

## 🚀 Quickstart
```bash
# 1) Create and activate a virtual environment (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Train (uses included sample dataset)
python src/train.py --data data/sample.csv

# 4) Local prediction from JSON payload
python src/predict.py --payload examples/payload.json

# 5) Serve an API
uvicorn src.serve:app --reload
# Then POST to http://127.0.0.1:8000/predict with JSON body like examples/payload.json
```

## 📊 Dataset
This starter includes a small synthetic dataset at `data/sample.csv`. Replace it with your real data (same column names) or add mapping in `src/data.py`.

### Expected Columns (features)
- `hours_studied` (float)
- `attendance_rate` (float 0–1)
- `assignments_completed` (int)
- `past_grade` (float 0–100)
- `social_activity_hours` (float)
- `sleep_hours` (float)
- `school_type` (categorical: "public" or "private")
- `parent_education` (categorical: "hs" | "bachelors" | "masters" | "phd")
- `final_grade` (target, 0–100)

## 🧪 Tests
```bash
pytest -q
```

## 📝 License
[MIT](LICENSE)
