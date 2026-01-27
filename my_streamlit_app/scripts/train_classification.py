"""
train_classification.py
---------------------------------
MLflow-based Classification Training
EMI Eligibility Prediction (Production Ready)
"""

"""
train_classification.py
---------------------------------
MLflow-based Classification Training
"""

import sys
from pathlib import Path

# ✅ FIX PYTHON PATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from scripts.feature_engineering import FeatureEngineer



# ==================================================
# PATHS & CONFIG
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "EMI_dataset_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGET_COL = "emi_eligibility"
REG_TARGET = "max_monthly_emi"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==================================================
# MLFLOW CONFIG
# ==================================================
MLFLOW_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("EMIPredict_Classification")

print("🔗 MLflow Tracking URI:", mlflow.get_tracking_uri())


# ==================================================
# LOAD DATA
# ==================================================
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

y_raw = df[TARGET_COL]


# ==================================================
# LABEL ENCODING (CRITICAL FOR STREAMLIT)
# ==================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
joblib.dump(label_encoder, LABEL_ENCODER_PATH)


# ==================================================
# FEATURE ENGINEERING
# ==================================================
print("🧠 Running feature engineering...")
fe = FeatureEngineer(df)
X = fe.run_full_pipeline(target_cols=[TARGET_COL, REG_TARGET])

PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_pipeline.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

joblib.dump(fe.pipeline, PIPELINE_PATH)
joblib.dump(fe.feature_names_, FEATURE_COLS_PATH)


# ==================================================
# TRAIN–TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# MODELS
# ==================================================
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1
    ),

    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoostClassifier": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
}


# ==================================================
# TRAIN & LOG TO MLFLOW
# ==================================================
best_model = None
best_f1 = -1.0

for model_name, model in models.items():
    print(f"\n🚀 Training {model_name}...")

    with mlflow.start_run(run_name=model_name):

        # ---------- TAGS ----------
        mlflow.set_tag("problem_type", "classification")
        mlflow.set_tag("target", TARGET_COL)
        mlflow.set_tag("model_name", model_name)

        # ---------- TRAIN ----------
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---------- METRICS ----------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # ---------- LOG MODEL ----------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="EMIPredict_Classifier"
        )

        # ---------- LOG PREPROCESSING ----------
        mlflow.log_artifact(LABEL_ENCODER_PATH, artifact_path="preprocessing")
        mlflow.log_artifact(PIPELINE_PATH, artifact_path="preprocessing")
        mlflow.log_artifact(FEATURE_COLS_PATH, artifact_path="preprocessing")

        print(
            f"📊 {model_name} | "
            f"Accuracy={acc:.4f} | "
            f"Precision={prec:.4f} | "
            f"Recall={rec:.4f} | "
            f"F1={f1:.4f}"
        )

        # ---------- BEST MODEL ----------
        if f1 > best_f1:
            best_f1 = f1
            best_model = model


# ==================================================
# SAVE BEST MODEL LOCALLY
# ==================================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_classifier.pkl")
joblib.dump(best_model, BEST_MODEL_PATH)

print("\n🏆 Best Classification Model Saved!")
print(f"📁 Path: {BEST_MODEL_PATH}")
print(f"⭐ Best F1 Score: {best_f1:.4f}")
