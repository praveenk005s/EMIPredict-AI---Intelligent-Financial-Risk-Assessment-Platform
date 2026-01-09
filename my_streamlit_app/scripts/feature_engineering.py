import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FeatureEngineer:
    """
    Production-grade Feature Engineering Pipeline
    for EMI Eligibility & Risk Prediction
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.pipeline = None
        self.feature_names_ = None



    # --------------------------------------------------
    # FEATURE CREATION
    # --------------------------------------------------
    def create_financial_ratios(self):

        self.df["debt_to_income"] = (
            self.df["current_emi_amount"] /
            (self.df["monthly_salary"] + 1e-6)
        )

        self.df["expense_to_income"] = (
            self.df["school_fees"]
            + self.df["college_fees"]
            + self.df["travel_expenses"]
            + self.df["groceries_utilities"]
            + self.df["other_monthly_expenses"]
        ) / (self.df["monthly_salary"] + 1e-6)

        self.df["affordability_ratio"] = (
            self.df["requested_amount"] /
            ((self.df["monthly_salary"] *
              self.df["requested_tenure"]) + 1e-6)
        )

        # Clip (banking safe)
        self.df["debt_to_income"] = self.df["debt_to_income"].clip(0, 5)
        self.df["expense_to_income"] = self.df["expense_to_income"].clip(0, 5)
        self.df["affordability_ratio"] = self.df["affordability_ratio"].clip(0, 10)

        return self

    def create_risk_features(self):

        self.df["credit_risk_score"] = self.df["credit_score"] / 850

        self.df["employment_stability"] = np.where(
            self.df["years_of_employment"] > 5, "Stable",
            np.where(self.df["years_of_employment"] > 2, "Moderate", "Unstable")
        )

        return self

    def create_interaction_features(self):

        self.df["income_x_credit"] = (
            self.df["monthly_salary"] * self.df["credit_score"]
        )

        self.df["dependents_ratio"] = (
            self.df["dependents"] /
            (self.df["family_size"] + 1e-6)
        )

        return self

    # --------------------------------------------------
    # PREPARE FOR MODELING
    # --------------------------------------------------
    def prepare_for_modeling(self, target_cols=None):

        if target_cols is None:
            target_cols = []

        X = self.df.drop(columns=target_cols, errors="ignore")

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        num_cols = X.select_dtypes(include=np.number).columns.tolist()

        # 🔥 FINAL GUARANTEE: categoricals are strings
        for col in cat_cols:
            X[col] = X[col].astype(str)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    cat_cols,
                ),
                (
                    "num",
                    StandardScaler(),
                    num_cols,
                ),
            ]
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor)]
        )

        X_transformed = self.pipeline.fit_transform(X)

        # Feature names
        cat_names = (
            self.pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .get_feature_names_out(cat_cols)
        )

        self.feature_names_ = list(cat_names) + num_cols

        return pd.DataFrame(X_transformed, columns=self.feature_names_)

    # --------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------
    def run_full_pipeline(self, target_cols=None):
        (
            self.clean_data()
            .create_financial_ratios()
            .create_risk_features()
            .create_interaction_features()
        )
        return self.prepare_for_modeling(target_cols)


# --------------------------------------------------
# MAIN (TRAINING ONLY)
# --------------------------------------------------
if __name__ == "__main__":

    input_path = r"E:/Project_03/notes/EMIPredict-AI/data/EMI_dataset_clean.csv"
    output_dir = r"E:/Project_03/notes/EMIPredict-AI/models"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    TARGET_COLS = ["emi_eligibility", "max_monthly_emi"]

    fe = FeatureEngineer(df)
    X = fe.run_full_pipeline(target_cols=TARGET_COLS)

    joblib.dump(fe.pipeline, os.path.join(output_dir, "feature_pipeline.pkl"))
    joblib.dump(fe.feature_names_, os.path.join(output_dir, "feature_columns.pkl"))

    print("✅ Feature pipeline & feature columns saved successfully")
    print("🔢 Total features:", len(fe.feature_names_))
