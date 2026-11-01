"""
train_compas_model.py - Train Risk Assessment Model & Extract Fairness Metrics from ProPublica COMPAS Dataset

Performs:
1. ProPublica standard data cleaning on compas-scores-two-years.csv (6,172 filtered records).
2. Calculation of empirical algorithmic bias metrics:
   - False Positive Rate disparity (African-American vs Caucasian).
   - False Negative Rate disparity.
   - Disparate Impact Ratio (EEOC 80% Four-Fifths compliance).
3. Supervised Machine Learning model training:
   - COMPAS Risk Score Model (Ridge regression predicting 0-100 risk decile scale).
   - Recidivism Probability Classifier (Logistic Regression predicting two-year recidivism).
4. Model serialization to compas_model.pkl and compas_stats.json.
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split


def load_and_preprocess_compas(data_path: str = "compas-scores-two-years.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"COMPAS dataset file not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Standard ProPublica filtering criteria:
    clean_df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    # Save cleaned subset
    clean_cols = [
        "id", "name", "sex", "age", "age_cat", "race",
        "priors_count", "c_charge_degree", "c_charge_desc",
        "decile_score", "score_text", "two_year_recid"
    ]
    clean_df = clean_df[clean_cols].copy()
    clean_df.to_csv("compas_clean.csv", index=False)
    print(f"[+] Cleaned COMPAS dataset saved to compas_clean.csv: {len(clean_df)} records.")
    return clean_df


def compute_fairness_disparities(clean_df: pd.DataFrame):
    """
    Computes empirical ProPublica fairness disparities:
    - High risk definition: decile_score >= 5 (Medium / High)
    - False Positive Rate (FPR): Non-recidivists misclassified as high risk
    - False Negative Rate (FNR): Recidivists misclassified as low risk
    """
    clean_df = clean_df.copy()
    clean_df["predicted_high_risk"] = clean_df["decile_score"] >= 5

    non_recid = clean_df[clean_df["two_year_recid"] == 0]
    recid = clean_df[clean_df["two_year_recid"] == 1]

    # FPR by race
    fpr_aa = float(non_recid[non_recid["race"] == "African-American"]["predicted_high_risk"].mean())
    fpr_cau = float(non_recid[non_recid["race"] == "Caucasian"]["predicted_high_risk"].mean())
    fpr_hisp = float(non_recid[non_recid["race"] == "Hispanic"]["predicted_high_risk"].mean())

    # FNR by race
    fnr_aa = float(1.0 - recid[recid["race"] == "African-American"]["predicted_high_risk"].mean())
    fnr_cau = float(1.0 - recid[recid["race"] == "Caucasian"]["predicted_high_risk"].mean())
    fnr_hisp = float(1.0 - recid[recid["race"] == "Hispanic"]["predicted_high_risk"].mean())

    # Disparate impact ratio (favorable outcome = low risk / diversion)
    fav_aa = float((clean_df[clean_df["race"] == "African-American"]["decile_score"] < 5).mean())
    fav_cau = float((clean_df[clean_df["race"] == "Caucasian"]["decile_score"] < 5).mean())
    disparate_impact_ratio = float(fav_aa / fav_cau) if fav_cau > 0 else 0.0

    stats = {
        "total_records": int(len(clean_df)),
        "race_distribution": {k: int(v) for k, v in clean_df["race"].value_counts().to_dict().items()},
        "fpr_african_american": round(fpr_aa * 100, 2),
        "fpr_caucasian": round(fpr_cau * 100, 2),
        "fpr_hispanic": round(fpr_hisp * 100, 2),
        "fpr_disparity_ratio": round(fpr_aa / fpr_cau, 2) if fpr_cau > 0 else 0.0,
        "fnr_african_american": round(fnr_aa * 100, 2),
        "fnr_caucasian": round(fnr_cau * 100, 2),
        "fnr_hispanic": round(fnr_hisp * 100, 2),
        "disparate_impact_ratio": round(disparate_impact_ratio, 3),
        "four_fifths_compliant": bool(disparate_impact_ratio >= 0.80),
    }

    print("\n--- ProPublica Empirical Fairness Audit ---")
    print(f"African-American False Positive Rate: {stats['fpr_african_american']}%")
    print(f"Caucasian False Positive Rate:        {stats['fpr_caucasian']}%")
    print(f"Disparity Gap: African-Americans are {stats['fpr_disparity_ratio']}x more likely to be falsely flagged as High Risk!")
    print(f"Disparate Impact Ratio: {stats['disparate_impact_ratio']} (Four-Fifths compliant: {stats['four_fifths_compliant']})")
    return stats


def train_model(clean_df: pd.DataFrame, stats: dict):
    """
    Trains:
    1. COMPAS Risk Score Model (Ridge regression predicting decile_score * 10 in [0, 100])
    2. Recidivism Binary Classifier (LogisticRegression predicting two_year_recid)
    """
    features = ["age", "priors_count", "c_charge_degree", "sex", "race"]
    X_raw = clean_df[features].copy()
    y_score = clean_df["decile_score"].values * 10.0  # 10 to 100 scale
    y_recid = clean_df["two_year_recid"].values

    # One-hot encode categoricals
    X = pd.get_dummies(X_raw, columns=["c_charge_degree", "sex", "race"], drop_first=False)
    feature_columns = list(X.columns)

    X_train, X_test, y_score_train, y_score_test, y_recid_train, y_recid_test = train_test_split(
        X, y_score, y_recid, test_size=0.20, random_state=42, stratify=y_recid
    )

    # 1. Train Risk Score Ridge Model
    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X_train, y_score_train)
    y_score_pred = reg.predict(X_test)
    r2 = float(r2_score(y_score_test, y_score_pred))

    # 2. Train Recidivism Classifier
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_recid_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_recid_test, clf.predict(X_test)))
    auc = float(roc_auc_score(y_recid_test, y_prob))

    print(f"\n[+] Risk Score Model: R² = {r2:.4f}")
    print(f"[+] Recidivism Classifier: Accuracy = {acc:.4f}, ROC-AUC = {auc:.4f}")

    coef_dict = {feat: float(round(coef, 4)) for feat, coef in zip(feature_columns, reg.coef_)}
    intercept = float(round(reg.intercept_, 4))

    print("\nCOMPAS Risk Decile Model Coefficients:")
    for feat, val in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:30s}: {val:+.4f} pts")

    baseline_profile = {
        "age": float(clean_df["age"].median()),
        "priors_count": 0.0,
        "c_charge_degree": "M",
        "sex": "Male",
        "race": "Caucasian",
    }

    model_artifacts = {
        "score_model": reg,
        "clf_model": clf,
        "feature_columns": feature_columns,
        "coef_dict": coef_dict,
        "intercept": intercept,
        "r2_score": round(r2, 4),
        "train_accuracy": round(acc * 100, 2),
        "roc_auc": round(auc, 4),
        "baseline_profile": baseline_profile,
        "disparity_metrics": stats,
    }

    with open("compas_model.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)
    print("\n[+] Model artifact serialized to compas_model.pkl")

    stats["model_r2"] = round(r2, 4)
    stats["model_accuracy"] = round(acc * 100, 2)
    stats["model_roc_auc"] = round(auc, 4)
    stats["intercept"] = intercept
    stats["coefficients"] = coef_dict
    with open("compas_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("[+] Fairness stats saved to compas_stats.json")

    return model_artifacts


if __name__ == "__main__":
    df = load_and_preprocess_compas()
    stats = compute_fairness_disparities(df)
    train_model(df, stats)
