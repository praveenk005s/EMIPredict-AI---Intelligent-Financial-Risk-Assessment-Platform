"""
data_preprocessing.py
---------------------------------
Data loading and cleaning pipeline for EMIPredict AI
"""

import pandas as pd
import numpy as np
import os


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded successfully | Shape: {df.shape}")
    return df


# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Remove duplicates
    - Fix inconsistent values
    - Handle missing values
    - Standardize categories
    """

    # Remove duplicates
    df = df.drop_duplicates()

    # -----------------------------
    # AGE CLEANING
    # -----------------------------
    age_map = {
        "58.0.0": "58",
        "38.0.0": "38",
        "32.0.0": "32"
    }

    if "age" in df.columns:
        df["age"] = df["age"].replace(age_map)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].astype("Int64")

    # -----------------------------
    # GENDER STANDARDIZATION
    # -----------------------------
    gender_map = {
        "female": "Female",
        "Female": "Female",
        "F": "Female",
        "FEMALE": "Female",
        "male": "Male",
        "Male": "Male",
        "M": "Male",
        "MALE": "Male"
    }

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace(gender_map)

    # -----------------------------
    # NUMERIC TYPE FIXES
    # -----------------------------
    numeric_fix_cols = ["bank_balance", "monthly_salary"]

    for col in numeric_fix_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # -----------------------------
    # CATEGORICAL CLEANING (FIX)
    # -----------------------------
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = (
            df[col]
            .replace(["nan", "NaN", "None", ""], np.nan)  # 🔥 critical
            .fillna("Unknown")                           # 🔥 single category
            .astype(str)
            .str.strip()
        )

    # -----------------------------
    # NUMERIC MISSING VALUES
    # -----------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # -----------------------------
    # STRIP SPACES IN CATEGORICAL
    # -----------------------------
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df[categorical_cols] = df[categorical_cols].fillna(
        df[categorical_cols].mode().iloc[0]
    )
    

    print("🧹 Data cleaning completed successfully.")
    return df



   
# --------------------------------------------------
# SAVE CLEAN DATA
# --------------------------------------------------
def save_clean_data(df: pd.DataFrame, output_path: str):
    """Save cleaned dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned dataset saved at: {output_path}")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":

    INPUT_PATH = r"E:/Project_03/notes/EMIPredict-AI/data/emi_prediction_dataset.csv"
    OUTPUT_PATH = r"E:/Project_03/notes/EMIPredict-AI/data/EMI_dataset_clean.csv"

    df_raw = load_data(INPUT_PATH)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean, OUTPUT_PATH)
