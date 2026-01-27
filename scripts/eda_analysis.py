"""
eda_analysis.py
---------------------------------
Exploratory Data Analysis (EDA) for EMIPredict AI
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", palette="pastel")


class EDAAnalysis:

    def __init__(self, data_path: str, report_path: str = None):
        self.data_path = data_path
        self.report_path = report_path
        self.df = None

    # ------------------------------
    # Load Data
    # ------------------------------
    def load_data(self):
        print(f"📂 Loading dataset: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded | Shape: {self.df.shape}")
        return self.df

    # ------------------------------
    # Data Overview
    # ------------------------------
    def data_overview(self):
        print("\n🔍 Dataset Info:")
        self.df.info()

        print("\n📊 Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])

        print("\n📈 Summary Statistics:")
        return self.df.describe(include="all").T

    # ------------------------------
    # Univariate Analysis
    # ------------------------------
    def univariate_analysis(self, save_plots=False, max_categories=10):
        print("\n📊 Univariate Analysis")

        num_cols = self.df.select_dtypes(include=np.number).columns
        cat_cols = self.df.select_dtypes(include="object").columns

        for col in num_cols:
            plt.figure(figsize=(6, 3))
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            if save_plots and self.report_path:
                plt.savefig(os.path.join(self.report_path, f"dist_{col}.png"))
            plt.close()

        for col in cat_cols:
            top_categories = self.df[col].value_counts().head(max_categories).index
            plt.figure(figsize=(6, 3))
            sns.countplot(
                y=self.df[col],
                order=top_categories
            )
            plt.title(f"Top {max_categories} Categories - {col}")
            plt.tight_layout()
            if save_plots and self.report_path:
                plt.savefig(os.path.join(self.report_path, f"cat_{col}.png"))
            plt.close()

        print("✅ Univariate analysis completed")

    # ------------------------------
    # EMI Eligibility Insights
    # ------------------------------
    def emi_eligibility_insights(self, save_plots=False):
        print("\n💡 EMI Eligibility Insights")

        plt.figure(figsize=(6, 4))
        sns.countplot(x="emi_eligibility", data=self.df)
        plt.title("EMI Eligibility Distribution")
        plt.tight_layout()
        if save_plots and self.report_path:
            plt.savefig(os.path.join(self.report_path, "emi_eligibility_dist.png"))
        plt.close()

        for feature in ["gender", "employment_type", "emi_scenario"]:
            if feature in self.df.columns:
                plt.figure(figsize=(6, 4))
                sns.countplot(x=feature, hue="emi_eligibility", data=self.df)
                plt.title(f"Eligibility by {feature}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                if save_plots and self.report_path:
                    plt.savefig(os.path.join(self.report_path, f"eligibility_by_{feature}.png"))
                plt.close()

        print("✅ EMI eligibility analysis completed")

    # ------------------------------
    # Correlation Analysis
    # ------------------------------
    def correlation_analysis(self, target_col="max_monthly_emi", save_plots=False):
        print("\n📉 Correlation Analysis")

        num_df = self.df.select_dtypes(include=np.number)
        corr = num_df.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, cmap="coolwarm", linewidths=0.3)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        if save_plots and self.report_path:
            plt.savefig(os.path.join(self.report_path, "correlation_heatmap.png"))
        plt.close()

        if target_col in corr.columns:
            top_corr = corr[target_col].sort_values(ascending=False).head(10)
            print(f"\n🔗 Top correlations with {target_col}:")
            print(top_corr)

        return corr

    # ------------------------------
    # Outlier Detection
    # ------------------------------
    def outlier_detection(self, cols=None, save_plots=False):
        print("\n🚨 Outlier Detection")

        if cols is None:
            cols = ["monthly_salary", "requested_amount", "current_emi_amount"]

        for col in cols:
            if col in self.df.columns:
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=self.df[col])
                plt.title(f"Outliers - {col}")
                plt.tight_layout()
                if save_plots and self.report_path:
                    plt.savefig(os.path.join(self.report_path, f"outliers_{col}.png"))
                plt.close()

        print("✅ Outlier detection completed")

    # ------------------------------
    # Business Insights
    # ------------------------------
    def business_insights(self, save_plots=False):
        print("\n📈 Business Insights")

        if "emi_scenario" in self.df.columns:
            summary = (
                self.df.groupby("emi_scenario")
                .agg(
                    avg_salary=("monthly_salary", "mean"),
                    avg_requested_amount=("requested_amount", "mean"),
                    dominant_eligibility=("emi_eligibility", lambda x: x.mode()[0])
                )
                .reset_index()
            )

            print(summary)

            plt.figure(figsize=(8, 4))
            sns.barplot(x="emi_scenario", y="avg_salary", data=summary)
            plt.title("Average Salary by EMI Scenario")
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_plots and self.report_path:
                plt.savefig(os.path.join(self.report_path, "avg_salary_by_scenario.png"))
            plt.close()

            return summary

    # ------------------------------
    # Save Summary
    # ------------------------------
    def save_summary(self):
        if self.report_path:
            path = os.path.join(self.report_path, "eda_summary.csv")
            self.df.describe(include="all").T.to_csv(path)
            print(f"📁 EDA summary saved at: {path}")


# ------------------------------
# Script Execution
# ------------------------------
if __name__ == "__main__":
    data_path = r"E:/Project_03/notes/EMIPredict-AI/data/EMI_dataset_clean.csv"
    report_path = r"E:/Project_03/notes/EMIPredict-AI/eda/reports"

    os.makedirs(report_path, exist_ok=True)

    eda = EDAAnalysis(data_path, report_path)
    eda.load_data()
    eda.data_overview()
    eda.univariate_analysis(save_plots=True)
    eda.emi_eligibility_insights(save_plots=True)
    eda.correlation_analysis(save_plots=True)
    eda.outlier_detection(save_plots=True)
    eda.business_insights(save_plots=True)
    eda.save_summary()
