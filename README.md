💳 EMIPredict AI
Intelligent EMI Eligibility & Financial Risk Assessment Platform

🚀 Production-ready FinTech platform that combines Indian banking business rules, 
CIBIL-based eligibility, machine learning risk assessment, MLflow model governance, and an enterprise Streamlit web application.

📌 Problem Statement

Banks and NBFCs face challenges in:

Assessing loan eligibility accurately

Managing risk & defaults

Enforcing hard banking rules (CIBIL, income, obligations)

Providing real-time EMI decisions

EMIPredict AI solves this by:

Applying banking rules first

Using ML only for risk refinement

Delivering transparent, explainable decisions

🏦 Core Philosophy (Banking-Grade)

Rules First → ML Second → Decision Final

❌ Hard Reject if banking rules fail

✅ ML models refine risk only after rules pass

🔐 Ensures regulatory compliance & trust

🧠 End-to-End Architecture
Dataset (400K Records)
        ↓
Data Quality Assessment & Preprocessing
        ↓
Feature Engineering & Exploratory Analysis
        ↓
Business Rules Engine (CIBIL + Banking Rules)
        ↓
ML Model Training & MLflow Tracking
        ↓
Model Evaluation & Selection
        ↓
Streamlit Application
        ↓
Cloud Deployment & Performance Testing
        ↓
Production-Ready Financial Platform

📊 Dataset

Size: ~400,000 records

Domain: Indian Banking / EMI / Loans

Key Features:

Demographics

Income & Expenses

Credit Score (CIBIL)

Loan Details

Existing EMIs

Risk Indicators

🧹 Data Preprocessing

✔ Duplicate removal
✔ Numeric normalization
✔ Categorical standardization
✔ Missing value handling
✔ Financial consistency checks

⚙ Feature Engineering
Financial Ratios

Debt-to-Income Ratio

Expense-to-Income Ratio

Affordability Ratio

Risk Features

Credit Risk Score

Employment Stability

Dependents Ratio

Income × Credit Interaction

🏦 Business Rules Engine (Hard Rules)
🔒 Mandatory Banking Rules (Before ML)
Rule	Description
CIBIL Score	< 650 → ❌ Reject
Existing Loans	> 2 → ❌ Reject
EMI Burden	> 80% salary → ❌ Reject
Negative Cash Flow	→ ❌ Reject
Affordability Check	EMI + Expenses ≤ Salary

📌 ML is NEVER used if rules fail

🤖 Machine Learning Models
🎯 Classification – EMI Eligibility

Logistic Regression

Random Forest

XGBoost (Production)

📈 Regression – Max Monthly EMI

Linear Regression

Random Forest Regressor

XGBoost Regressor (Production)

📦 MLflow Integration

✔ Experiment Tracking
✔ Metric Logging
✔ Model Registry
✔ Version Control
✔ Production Staging

mlflow ui


📍 http://127.0.0.1:5000

🖥 Streamlit Application
Features

🔍 Single EMI Prediction

📂 Batch CSV Prediction

📊 Model Monitoring

📈 Exploratory Data Analysis

🧠 Model Information

🔐 Banking Rules Explanation

Decision Flow
User Input
   ↓
Banking Rules Check
   ↓
(If Passed)
ML Prediction
   ↓
Final Decision

🧑‍💼 Admin / CRUD (Planned)

✔ Upload datasets
✔ Update business rules
✔ Model version management
✔ Audit logs
✔ Access control

☁ Cloud Deployment

Docker-ready architecture

Streamlit Cloud

Scalable inference

Secure artifact storage

⚡ Performance & Scalability

Handles 100K+ predictions

Optimized feature pipeline

Cached ML assets

Stateless inference


🏆 Why This Project Is Enterprise-Ready

✔ Banking-compliant rules
✔ ML governance with MLflow
✔ Explainable decisions
✔ Scalable architecture
✔ Production deployment ready

👨‍💻 Author

Praveen Kumar
📌 Data Scientist | Machine Learning Engineer | FinTech
📍 India
