"""
train_regression.py
---------------------------------
MLflow-based Regression Training for Maximum Monthly EMI Prediction

Models:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Metrics:
- RMSE
- MAE
- R2 Score
"""

# ==================================================
# PYTHON PATH FIX (CRITICAL)
# ==================================================
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ==================================================
# IMPORTS
# ==================================================
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from scripts.feature_engineering import FeatureEngineer


# ==================================================
# CONFIG
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "EMI_dataset_clean.csv"
MODEL_DIR = BASE_DIR / "models"

TARGET_COL = "max_monthly_emi"
CLASS_TARGET = "emi_eligibility"

MODEL_DIR.mkdir(exist_ok=True)

# ==================================================
# MLFLOW CONFIG
# ==================================================
MLFLOW_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("EMIPredict_Regression")

print("🔗 MLflow Tracking URI:", mlflow.get_tracking_uri())


# ==================================================
# LOAD DATA
# ==================================================
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

y = df[TARGET_COL]


# ==================================================
# FEATURE ENGINEERING
# ==================================================
print("🧠 Running feature engineering...")
fe = FeatureEngineer(df)
X = fe.run_full_pipeline(target_cols=[CLASS_TARGET, TARGET_COL])

# 🔐 Save pipeline ONCE (used by Streamlit)
PIPELINE_PATH = MODEL_DIR / "feature_pipeline.pkl"
joblib.dump(fe.pipeline, PIPELINE_PATH)


# ==================================================
# TRAIN–TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# MODELS
# ==================================================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoostRegressor": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
}


# ==================================================
# TRAIN & LOG TO MLFLOW
# ==================================================
best_model = None
best_rmse = float("inf")

for model_name, model in models.items():
    print(f"\n🚀 Training {model_name}...")

    with mlflow.start_run(run_name=model_name):

        # ---------- TAGS ----------
        mlflow.set_tag("problem_type", "regression")
        mlflow.set_tag("target", TARGET_COL)
        mlflow.set_tag("model_name", model_name)

        # ---------- PARAMS ----------
        mlflow.log_params(model.get_params())

        # ---------- TRAIN ----------
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---------- METRICS ----------
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        })

        # ---------- REGISTER MODEL ----------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="EMIPredict_Regressor"
        )

        # ---------- LOG PREPROCESSING ----------
        mlflow.log_artifact(
            PIPELINE_PATH,
            artifact_path="preprocessing"
        )

        print(
            f"📊 {model_name} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.4f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model


# ==================================================
# SAVE BEST MODEL LOCALLY
# ==================================================
BEST_MODEL_PATH = MODEL_DIR / "best_regressor.pkl"
joblib.dump(best_model, BEST_MODEL_PATH)

print("\n🏆 Best Regression Model Saved!")
print(f"📁 Path: {BEST_MODEL_PATH}")
print(f"⭐ Best RMSE: {best_rmse:.2f}")
