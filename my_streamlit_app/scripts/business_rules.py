import numpy as np


# ==================================================
# EMI CALCULATION
# ==================================================
def calculate_emi(principal, tenure_months, annual_rate=12):
    if principal <= 0 or tenure_months <= 0:
        return 0.0

    r = annual_rate / 12 / 100
    n = tenure_months
    emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return round(float(emi), 2)


# ==================================================
# BUSINESS RULE ENGINE (BANK FIRST, ML SECOND)
# ==================================================
def business_rule_eligibility(row: dict):
    """
    REAL INDIAN BANKING HARD RULES
    (These override ML decisions)
    """

    # --------------------------------------------------
    # SAFE INPUT EXTRACTION
    # --------------------------------------------------
    monthly_salary = float(row.get("monthly_salary", 0))
    credit_score = int(row.get("credit_score", 0))
    requested_amount = float(row.get("requested_amount", 0))
    requested_tenure = int(row.get("requested_tenure", 0))

    monthly_rent = float(row.get("monthly_rent", 0))
    school_fees = float(row.get("school_fees", 0))
    college_fees = float(row.get("college_fees", 0))
    travel_expenses = float(row.get("travel_expenses", 0))
    groceries_utilities = float(row.get("groceries_utilities", 0))
    other_monthly_expenses = float(row.get("other_monthly_expenses", 0))
    current_emi_amount = float(row.get("current_emi_amount", 0))

    existing_loans_raw = row.get("existing_loans", 0)

    if isinstance(existing_loans_raw, str):
        existing_loans = 1 if existing_loans_raw.strip().lower() == "yes" else 0
    elif isinstance(existing_loans_raw, (int, float)):
        existing_loans = int(existing_loans_raw)
    else:
        existing_loans = 0


    # --------------------------------------------------
    # RULE 1: CIBIL SCORE (HARD STOP)
    # --------------------------------------------------
    if credit_score < 650:
        return {
            "approved": False,
            "reason": f"CIBIL score too low ({credit_score})",
            "risk_level": "Very High"
        }

    # --------------------------------------------------
    # RULE 2: EXISTING LOANS LIMIT
    # --------------------------------------------------
    if existing_loans > 2:
        return {
            "approved": False,
            "reason": "More than 2 existing loans",
            "risk_level": "High"
        }

    # --------------------------------------------------
    # RULE 3: TOTAL MONTHLY EXPENSES
    # --------------------------------------------------
    total_expenses = (
        monthly_rent +
        school_fees +
        college_fees +
        travel_expenses +
        groceries_utilities +
        other_monthly_expenses +
        current_emi_amount
    )

    # --------------------------------------------------
    # RULE 4: REQUESTED EMI
    # --------------------------------------------------
    requested_emi = calculate_emi(
        principal=requested_amount,
        tenure_months=requested_tenure
    )

    total_outflow = total_expenses + requested_emi

    # --------------------------------------------------
    # RULE 5: EMI-TO-INCOME RATIO (BANKING GOLD RULE)
    # --------------------------------------------------
    emi_ratio = total_outflow / (monthly_salary + 1e-6)

    # 🔒 Indian Banks:
    # < 40% → Safe
    # 40–50% → Risky
    # > 50% → Reject
    if emi_ratio > 0.50:
        return {
            "approved": False,
            "reason": "EMI exceeds 50% of monthly salary",
            "requested_emi": requested_emi,
            "total_outflow": round(total_outflow, 2),
            "risk_level": "High"
        }

    # --------------------------------------------------
    # RULE 6: CASH FLOW BUFFER
    # --------------------------------------------------
    if monthly_salary - total_outflow < monthly_salary * 0.2:
        return {
            "approved": False,
            "reason": "Insufficient monthly cash buffer",
            "requested_emi": requested_emi,
            "total_outflow": round(total_outflow, 2),
            "risk_level": "Medium"
        }

    # --------------------------------------------------
    # PASSED ALL BANK RULES
    # --------------------------------------------------
    return {
        "approved": True,
        "reason": "Approved as per Indian banking rules",
        "requested_emi": requested_emi,
        "total_outflow": round(total_outflow, 2),
        "risk_level": "Low"
    }
