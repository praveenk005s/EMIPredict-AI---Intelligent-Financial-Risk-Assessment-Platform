import sys
from pathlib import Path

# ==================================================
# PATH SETUP (CRITICAL FOR STREAMLIT CLOUD)
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scripts.business_rules import business_rule_eligibility

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide"
)

# ==================================================
# PATHS
# ==================================================
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

PIPELINE_PATH = MODEL_DIR / "feature_pipeline.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

CLASSIFIER_PATH = BASE_DIR / "best_classifier.pkl"
REGRESSOR_PATH = BASE_DIR / "best_regressor.pkl"

ADMIN_DATA_PATH = DATA_DIR / "admin_applications.csv"
DATA_DIR.mkdir(exist_ok=True)

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    return clf, reg

@st.cache_resource
def load_assets():
    pipeline = joblib.load(PIPELINE_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return pipeline, feature_cols, label_encoder

classifier, regressor = load_models()
pipeline, FEATURE_COLUMNS, label_encoder = load_assets()

# ==================================================
# SIDEBAR
# ==================================================
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Single Prediction",
        "📂 Batch Prediction",
        "📈 EDA",
        "🛠 Admin Panel",
        "🧠 Model Info"
    ]
)

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div style="background:#111827;padding:15px;border-radius:10px">
<h2 style="color:white">💳 EMIPredict AI</h2>
<p style="color:#9CA3AF">Production-Ready EMI Eligibility & Financial Risk Platform</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SINGLE PREDICTION
# ==================================================
if page == "🏠 Single Prediction":

    st.subheader("🔍 Applicant Information")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        age = c1.number_input("Age", 18, 80, 30)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        marital_status = c3.selectbox("Marital Status", ["Single", "Married"])

        monthly_salary = c1.number_input("Monthly Salary (₹)", 0, 500000, 60000)
        employment_type = c2.selectbox("Employment Type", ["Private", "Government", "Self-Employed"])
        years_of_employment = c3.number_input("Years of Employment", 0.0, 40.0, 5.0)

        requested_amount = c1.number_input("Requested Loan (₹)", 0, 5000000, 500000)
        requested_tenure = c2.number_input("Tenure (Months)", 1, 240, 24)
        credit_score = c3.number_input("CIBIL Score", 300, 900, 720)

        education = c1.selectbox("Education", ["Professional", "Graduate", "High School", "Post Graduate"])
        company_type = c2.selectbox("Company Type", ["Mid-size", "MNC", "Startup", "Large Indian", "Small"])
        house_type = c3.selectbox("House Type", ["Rented", "Family", "Own"])

        monthly_rent = c1.number_input("Monthly Rent", 0, 1000000, 0)
        family_size = c2.number_input("Family Size", 1, 10, 3)
        dependents = c3.number_input("Dependents", 0, 10, 0)

        school_fees = c1.number_input("School Fees", 0, 1000000, 0)
        college_fees = c2.number_input("College Fees", 0, 1000000, 0)
        travel_expenses = c3.number_input("Travel Expenses", 0, 1000000, 0)

        groceries_utilities = c1.number_input("Groceries & Utilities", 0, 1000000, 0)
        other_monthly_expenses = c2.number_input("Other Monthly Expenses", 0, 1000000, 0)
        current_emi_amount = c3.number_input("Current EMI Amount", 0, 1000000, 0)

        existing_loans = c1.selectbox("Existing Loans", ["No", "Yes"])

        submit = st.form_submit_button("🚀 Predict")

    if submit:
        # ---------------- BUSINESS RULES ----------------
        rule_input = {
            "credit_score": credit_score,
            "existing_loans": existing_loans,
            "monthly_salary": monthly_salary,
            "monthly_rent": monthly_rent,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "current_emi_amount": current_emi_amount,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure
        }

        rule_result = business_rule_eligibility(rule_input)

        if not rule_result["approved"]:
            st.error("❌ Loan Rejected (Banking Rules)")
            st.warning(rule_result["reason"])
            st.stop()

        # ---------------- ML FEATURES ----------------
        df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "credit_score": credit_score,
            "education": education,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "bank_balance": 50000,
            "emergency_fund": 20000,
            "emi_scenario": "Personal Loan EMI"
        }])

        # Derived features
        df["debt_to_income"] = df["current_emi_amount"] / (df["monthly_salary"] + 1e-6)
        df["expense_to_income"] = (
            df["school_fees"] + df["college_fees"] +
            df["travel_expenses"] + df["groceries_utilities"] +
            df["other_monthly_expenses"]
        ) / (df["monthly_salary"] + 1e-6)

        df["affordability_ratio"] = df["requested_amount"] / (
            df["monthly_salary"] * df["requested_tenure"] + 1e-6
        )

        df["credit_risk_score"] = df["credit_score"] / 900
        df["income_x_credit"] = df["monthly_salary"] * df["credit_score"]

        df["employment_stability"] = np.where(
            df["years_of_employment"] > 5, "Stable",
            np.where(df["years_of_employment"] > 2, "Moderate", "Unstable")
        )

        df["dependents_ratio"] = df["dependents"] / (df["family_size"] + 1e-6)

        X = pipeline.transform(df)
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        # ---------------- PREDICTION ----------------
        class_id = int(classifier.predict(X)[0])
        eligibility = label_encoder.inverse_transform([class_id])[0]
        max_emi = float(regressor.predict(X)[0])

        col1, col2 = st.columns(2)
        col1.success("✅ Loan Approved (Rules Passed)")
        col1.info(f"ML Risk Decision: **{eligibility}**")

        col2.success(f"💰 Max Safe EMI: ₹{max_emi:,.2f}")
        col2.info(f"Requested EMI: ₹{rule_result['requested_emi']:,}")

# ==================================================
# ADMIN PANEL (CRUD)
# ==================================================
elif page == "🛠 Admin Panel":

    st.subheader("🛠 Admin Panel – Loan Applications")

    if ADMIN_DATA_PATH.exists():
        admin_df = pd.read_csv(ADMIN_DATA_PATH)
    else:
        admin_df = pd.DataFrame(columns=[
            "application_id", "monthly_salary", "credit_score",
            "requested_amount", "status"
        ])

    st.dataframe(admin_df, use_container_width=True)

    with st.form("add_app"):
        app_id = st.text_input("Application ID")
        salary = st.number_input("Monthly Salary", 0, 500000, 60000)
        credit = st.number_input("CIBIL Score", 300, 900, 720)
        loan_amt = st.number_input("Requested Amount", 0, 5000000, 500000)

        if st.form_submit_button("➕ Add"):
            admin_df.loc[len(admin_df)] = [
                app_id, salary, credit, loan_amt, "New"
            ]
            admin_df.to_csv(ADMIN_DATA_PATH, index=False)
            st.success("Application Added")

# ==================================================
# MODEL INFO
# ==================================================
elif page == "🧠 Model Info":
    st.success("✔ Banking Rules First")
    st.success("✔ ML for Risk Refinement")
    st.success("✔ Cloud-Ready Streamlit App")
