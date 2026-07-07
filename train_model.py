
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

APP_DIR = Path(__file__).parent
BASE_REFERENCE_YEAR = 2024
BASE_REFERENCE_PERIOD = "2024 Q3 observed market reference"
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

market = pd.read_csv(DATA_DIR / "market_price_clean.csv")

FEATURE_COLS = [
    "brand",
    "model_name",
    "product_type",
    "car_year",
    "vehicle_age",
    "brand_origin",
    "segment",
    "brand_tier",
    "rv_grade",
    "risk_grade",
    "profit_grade",
]
CAT_COLS = ["brand","model_name","product_type","brand_origin","segment","brand_tier","rv_grade","risk_grade","profit_grade"]
NUM_COLS = ["car_year","vehicle_age"]

df = market.dropna(subset=["car_price","car_year","vehicle_age"]).copy()
X = df[FEATURE_COLS].copy()
for c in CAT_COLS:
    X[c] = X[c].fillna("Unknown").astype(str)
y_log = np.log1p(df["car_price"].astype(float))

X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
    X, y_log, df["car_price"].astype(float), test_size=0.2, random_state=42
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), CAT_COLS),
    ("num", "passthrough", NUM_COLS)
])

model = RandomForestRegressor(
    n_estimators=260,
    random_state=42,
    max_depth=18,
    min_samples_leaf=2,
    n_jobs=-1,
)

pipe = Pipeline([("preprocess", preprocess), ("model", model)])
pipe.fit(X_train, y_train)

pred_log = pipe.predict(X_test)
pred_price = np.expm1(pred_log)

metrics = {
    "training_rows": int(len(df)),
    "test_rows": int(len(X_test)),
    "r2_log_price": float(r2_score(y_test, pred_log)),
    "mae_price_usd": float(mean_absolute_error(price_test, pred_price)),
    "median_ae_price_usd": float(median_absolute_error(price_test, pred_price)),
    "model_note": "RandomForestRegressor trained on log(car_price) using 2024 Q3 observed-reference market prices. Future-year values are projected in app.py, not directly predicted by this model.",
    "base_reference_year": BASE_REFERENCE_YEAR,
    "base_reference_period": BASE_REFERENCE_PERIOD,
}

joblib.dump(pipe, MODEL_DIR / "residual_price_model.pkl")
with open(MODEL_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(json.dumps(metrics, indent=2))
