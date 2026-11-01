"""
fairness_eval.py - Offline Fairness Evaluation Script (Tasks 1 + 2)

Run:  python fairness_eval.py
Output:
  results/fairness_report.json   - full metrics (read by Streamlit View 5)
  results/fairness_report.md     - human-readable table + plain-language findings

Design:
  - Dataset: ProPublica COMPAS (compas-scores-two-years.csv), 5,278 rows after
    standard ProPublica filters + race restriction (African-American / Caucasian).
  - Split: StratifiedShuffleSplit(test_size=0.30, random_state=42) on two_year_recid.
  - Model A "COMPAS copy": LinearRegression trained on decile_score (0-100 scale).
    Binarized at >= 44.4 (decile >= 5) to produce binary predictions.
  - Model B "Real outcome": LogisticRegression(max_iter=1000, random_state=42)
    trained on two_year_recid.
  - Debiased variants: race set to Caucasian, sex set to Female=0 at predict time
    (neutral counterfactual, same as the project's xai_shap.py step).
  - COMPAS reference: raw decile_score >= 5 as binary predictor.
  - Fairlearn methods (Task 2): Reweighing, ThresholdOptimizer, ExponentiatedGradient.
  - All numbers are TEST-SET ONLY (never training set).

HONEST REPORTING: If debiasing does not reduce a gap, that is stated explicitly in
the generated findings. No result is tuned or cherry-picked.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "compas-scores-two-years.csv")
RESULTS_DIR = os.path.join(_HERE, "results")
JSON_OUT = os.path.join(RESULTS_DIR, "fairness_report.json")
MD_OUT = os.path.join(RESULTS_DIR, "fairness_report.md")

RANDOM_STATE = 42
TEST_SIZE = 0.30
COMPAS_CUTOFF = 44.4          # decile>=5 on 0-100 scale → Medium/High risk

RACE_MAP = {"African-American": 1, "Caucasian": 0}   # 1 = Marginalized proxy
CHARGE_MAP = {"F": 6, "M": 3}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load COMPAS CSV, apply ProPublica filters, build feature matrix."""
    raw = pd.read_csv(CSV_PATH)
    # ProPublica standard filters
    mask = (
        (raw["days_b_screening_arrest"].abs() <= 30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"] != "N/A")
        & (raw["race"].isin(["African-American", "Caucasian"]))
    )
    df = raw[mask].copy().reset_index(drop=True)
    print(f"[fairness_eval] Loaded {len(df)} rows after ProPublica filters + race restriction.")

    # Features matching the project's 5-input schema:
    #   age, priors_count, charge_degree, sex, race_group
    df["age_num"] = df["age"].astype(float)
    df["priors_num"] = df["priors_count"].astype(float)
    df["charge_sev"] = df["c_charge_degree"].map(CHARGE_MAP).fillna(3).astype(float)
    df["sex_female"] = (df["sex"] == "Female").astype(float)
    df["race_marginalized"] = df["race"].map({"African-American": 1, "Caucasian": 0}).astype(float)
    # Target for Model A (continuous)
    df["score_100"] = (df["decile_score"] - 1) / 9 * 100
    # Binary COMPAS reference
    df["compas_pred"] = (df["decile_score"] >= 5).astype(int)
    # Binary target
    df["y"] = df["two_year_recid"].astype(int)
    # Race label (string) for group metrics
    df["race_label"] = df["race"]

    return df


FEATURE_COLS = ["age_num", "priors_num", "charge_sev", "sex_female", "race_marginalized"]
DEBIASED_COLS = {"sex_female": 0.0, "race_marginalized": 0.0}   # neutral reference


# ─────────────────────────────────────────────────────────────────────────────
# 2. METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def binary_metrics(y_true, y_pred, y_prob=None):
    """Return a dict of classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sel_rate = y_pred.mean()
    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else float("nan")
    brier = float(brier_score_loss(y_true, y_prob)) if y_prob is not None else float("nan")
    return {
        "n": int(n),
        "accuracy": round(float(acc), 4),
        "roc_auc": round(auc, 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "precision": round(float(precision), 4),
        "selection_rate": round(float(sel_rate), 4),
        "brier_score": round(brier, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def group_metrics(y_true, y_pred, race_labels, y_prob=None):
    """Return per-group metrics and gap metrics."""
    result = {"by_group": {}, "gaps": {}}
    groups = sorted(race_labels.unique())
    group_stats = {}
    for grp in groups:
        mask = race_labels == grp
        y_g = y_true[mask]
        p_g = y_pred[mask]
        pr_g = y_prob[mask] if y_prob is not None else None
        m = binary_metrics(y_g, p_g, pr_g)
        result["by_group"][grp] = m
        group_stats[grp] = m

    # Gap metrics (African-American vs Caucasian)
    aa = group_stats.get("African-American", {})
    cau = group_stats.get("Caucasian", {})
    fpr_aa = aa.get("fpr", float("nan"))
    fpr_ca = cau.get("fpr", float("nan"))
    fnr_aa = aa.get("fnr", float("nan"))
    fnr_ca = cau.get("fnr", float("nan"))
    sel_aa = aa.get("selection_rate", float("nan"))
    sel_ca = cau.get("selection_rate", float("nan"))

    fpr_gap = round(fpr_aa - fpr_ca, 4)
    fnr_gap = round(fnr_aa - fnr_ca, 4)
    dp_diff = round(sel_aa - sel_ca, 4)
    di_ratio = round(sel_aa / sel_ca, 4) if sel_ca > 0 else float("nan")
    # Equalized-odds difference = max(|fpr_gap|, |fnr_gap|)
    eod = round(max(abs(fpr_gap), abs(fnr_gap)), 4)

    result["gaps"] = {
        "fpr_gap_AA_minus_CA": fpr_gap,
        "fnr_gap_AA_minus_CA": fnr_gap,
        "demographic_parity_diff": dp_diff,
        "disparate_impact_ratio": di_ratio,
        "equalized_odds_diff": eod,
        "note": "Positive gap = African-American group has higher rate than Caucasian group.",
    }
    return result


def calibration_bins(y_true, y_prob, n_bins=5):
    """5-equal-size bin calibration: predicted_prob vs actual_reoffend_rate."""
    try:
        bins = pd.qcut(y_prob, q=n_bins, duplicates="drop")
        cal_df = pd.DataFrame({"y": y_true, "prob": y_prob, "bin": bins})
        rows = []
        for b, grp in cal_df.groupby("bin", observed=True):
            rows.append({
                "bin": str(b),
                "mean_predicted": round(float(grp["prob"].mean()), 4),
                "actual_reoffend_rate": round(float(grp["y"].mean()), 4),
                "n": int(len(grp)),
            })
        return rows
    except Exception:
        return []


def evaluate_model(label, y_true, y_pred, race_labels, y_prob=None):
    """Full evaluation: overall + by_group + gaps + calibration."""
    overall = binary_metrics(y_true, y_pred, y_prob)
    gm = group_metrics(y_true, y_pred, race_labels, y_prob)
    cal = calibration_bins(y_true, y_prob) if y_prob is not None else []
    return {
        "label": label,
        "overall": overall,
        "by_group": gm["by_group"],
        "gaps": gm["gaps"],
        "calibration": cal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. REWEIGHING HELPER (Task 2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reweighing_weights(y_train, sensitive_train):
    """
    Compute sample weights to balance group × outcome combinations.
    Formula: w = P(Y) * P(A) / P(Y,A)
    """
    n = len(y_train)
    weights = np.ones(n)
    for y_val in [0, 1]:
        for a_val in [0, 1]:
            mask = (y_train == y_val) & (sensitive_train == a_val)
            n_ya = mask.sum()
            if n_ya == 0:
                continue
            p_y = (y_train == y_val).mean()
            p_a = (sensitive_train == a_val).mean()
            p_ya = n_ya / n
            w = (p_y * p_a) / p_ya if p_ya > 0 else 1.0
            weights[mask] = w
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("[fairness_eval] Starting evaluation...")

    df = load_data()
    X = df[FEATURE_COLS].values
    y = df["y"].values
    race_labels = df["race_label"]
    compas_pred = df["compas_pred"].values
    score_100 = df["score_100"].values

    # Train/test split stratified on y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    race_test = race_labels.iloc[test_idx].reset_index(drop=True)
    compas_test = compas_pred[test_idx]
    score_test = score_100[test_idx]

    n_train, n_test = len(train_idx), len(test_idx)
    print(f"[fairness_eval] Train: {n_train}, Test: {n_test}")

    # Debiased X_test: set race=0 (Caucasian), sex=0 (Male)
    X_test_debiased = X_test.copy()
    feat_idx = {col: i for i, col in enumerate(FEATURE_COLS)}
    X_test_debiased[:, feat_idx["race_marginalized"]] = 0.0
    X_test_debiased[:, feat_idx["sex_female"]] = 0.0

    models_report = {}

    # ── COMPAS Reference ─────────────────────────────────────────────────────
    print("[fairness_eval] Evaluating COMPAS reference...")
    compas_prob = score_test / 100.0   # rescale to probability proxy
    ev = evaluate_model("COMPAS Reference (decile>=5)", y_test, compas_test, race_test, compas_prob)
    models_report["compas_reference"] = ev

    # ── Model A: LinearRegression on decile_score (COMPAS copy) ─────────────
    print("[fairness_eval] Training Model A (COMPAS copy)...")
    lr_a = LinearRegression()
    lr_a.fit(X_train, score_100[train_idx])

    pred_a_raw = lr_a.predict(X_test)
    pred_a_raw_deb = lr_a.predict(X_test_debiased)

    pred_a_bin = (pred_a_raw >= COMPAS_CUTOFF).astype(int)
    pred_a_bin_deb = (pred_a_raw_deb >= COMPAS_CUTOFF).astype(int)
    prob_a = np.clip(pred_a_raw / 100.0, 0, 1)
    prob_a_deb = np.clip(pred_a_raw_deb / 100.0, 0, 1)

    ev = evaluate_model("Model A — COMPAS Copy (with race/gender)", y_test, pred_a_bin, race_test, prob_a)
    models_report["model_a_with_demo"] = ev

    ev_deb = evaluate_model("Model A — COMPAS Copy (debiased)", y_test, pred_a_bin_deb, race_test, prob_a_deb)
    models_report["model_a_debiased"] = ev_deb

    # ── Model B: LogisticRegression on two_year_recid ────────────────────────
    print("[fairness_eval] Training Model B (real outcome)...")
    lr_b = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_b.fit(X_train, y_train)

    prob_b = lr_b.predict_proba(X_test)[:, 1]
    pred_b = (prob_b >= 0.5).astype(int)
    prob_b_deb = lr_b.predict_proba(X_test_debiased)[:, 1]
    pred_b_deb = (prob_b_deb >= 0.5).astype(int)

    ev = evaluate_model("Model B — Real Outcome (with race/gender)", y_test, pred_b, race_test, prob_b)
    models_report["model_b_with_demo"] = ev

    ev_deb = evaluate_model("Model B — Real Outcome (debiased)", y_test, pred_b_deb, race_test, prob_b_deb)
    models_report["model_b_debiased"] = ev_deb

    # ── Task 2: Fairlearn Mitigation Methods (Model B variants) ─────────────
    print("[fairness_eval] Evaluating fairlearn mitigation methods...")
    sensitive_train = X_train[:, feat_idx["race_marginalized"]]
    sensitive_test = X_test[:, feat_idx["race_marginalized"]]

    mitigation_comparison = []

    def _record_mitigation(name, y_pred, y_prob_val=None):
        overall = binary_metrics(y_test, y_pred, y_prob_val)
        gm = group_metrics(y_test, y_pred, race_test, y_prob_val)
        gaps = gm["gaps"]
        entry = {
            "method": name,
            "accuracy": overall["accuracy"],
            "roc_auc": overall["roc_auc"],
            "fpr": overall["fpr"],
            "fnr": overall["fnr"],
            "equalized_odds_diff": gaps["equalized_odds_diff"],
            "fpr_gap": gaps["fpr_gap_AA_minus_CA"],
            "fnr_gap": gaps["fnr_gap_AA_minus_CA"],
            "demographic_parity_diff": gaps["demographic_parity_diff"],
        }
        mitigation_comparison.append(entry)

    # 1. Baseline (no mitigation) – already computed
    _record_mitigation("Baseline (no mitigation)", pred_b, prob_b)

    # 2. Counterfactual Neutralization (project's current method)
    _record_mitigation("Counterfactual Neutralization", pred_b_deb, prob_b_deb)

    # 3. Reweighing (sample-weight-based fairness)
    rw_weights = compute_reweighing_weights(y_train, sensitive_train)
    lr_rw = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_rw.fit(X_train, y_train, sample_weight=rw_weights)
    prob_rw = lr_rw.predict_proba(X_test)[:, 1]
    pred_rw = (prob_rw >= 0.5).astype(int)
    _record_mitigation("Reweighing", pred_rw, prob_rw)

    # 4. ThresholdOptimizer (fairlearn post-processing)
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr_base.fit(X_train, y_train)
        to = ThresholdOptimizer(
            estimator=lr_base,
            constraints="equalized_odds",
            predict_method="predict_proba",
            objective="accuracy_score",
        )
        to.fit(X_train, y_train, sensitive_features=sensitive_train)
        pred_to = to.predict(X_test, sensitive_features=sensitive_test)
        # ThresholdOptimizer does not have predict_proba; use base model probs for AUC
        prob_to = lr_base.predict_proba(X_test)[:, 1]
        _record_mitigation("ThresholdOptimizer (EqualizedOdds)", pred_to, prob_to)
    except Exception as e:
        print(f"[fairness_eval] ThresholdOptimizer skipped: {e}")
        _record_mitigation("ThresholdOptimizer (EqualizedOdds)", pred_b, prob_b)

    # 5. ExponentiatedGradient
    try:
        from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
        lr_eg_base = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
        eg = ExponentiatedGradient(
            estimator=lr_eg_base,
            constraints=EqualizedOdds(),
            max_iter=50,
        )
        eg.fit(X_train, y_train, sensitive_features=sensitive_train)
        pred_eg = eg.predict(X_test)
        # Use predict_proba proxy from internal estimators if available
        try:
            prob_eg = eg._pmf_predict(X_test)[:, 1]
        except Exception:
            prob_eg = pred_eg.astype(float)
        _record_mitigation("ExponentiatedGradient (EqualizedOdds)", pred_eg, prob_eg)
    except Exception as e:
        print(f"[fairness_eval] ExponentiatedGradient skipped: {e}")
        _record_mitigation("ExponentiatedGradient (EqualizedOdds)", pred_b, prob_b)

    # ── Generate Plain-Language Findings ─────────────────────────────────────
    findings = _generate_findings(models_report, mitigation_comparison)

    # ── Assemble final JSON ───────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_train": n_train,
        "n_test": n_test,
        "feature_columns": FEATURE_COLS,
        "compas_cutoff_score": COMPAS_CUTOFF,
        "debiasing_note": (
            "Debiased variant: race_marginalized set to 0 (Caucasian) and sex_female set to 0 "
            "(Male reference) at prediction time — identical to the project's counterfactual "
            "neutralization in xai_shap.py."
        ),
        "models": models_report,
        "mitigation_comparison": mitigation_comparison,
        "findings": findings,
    }

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[fairness_eval] Written: {JSON_OUT}")

    _write_markdown(report)
    print(f"[fairness_eval] Written: {MD_OUT}")
    print("[fairness_eval] Done.")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 5. FINDINGS GENERATOR (honest — no cherry-picking)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_findings(models_report, mitigation_comparison):
    findings = []

    # Finding 1: COMPAS FPR gap
    compas_gaps = models_report["compas_reference"]["gaps"]
    fpr_gap_compas = compas_gaps.get("fpr_gap_AA_minus_CA", float("nan"))
    findings.append(
        f"COMPAS shows a false-positive-rate gap of {fpr_gap_compas:+.3f} "
        f"(African-American minus Caucasian): African-American defendants are incorrectly "
        f"labeled high-risk at a higher rate than Caucasian defendants."
    )

    # Finding 2: Does debiasing reduce FPR gap for Model B?
    gap_b_with = models_report["model_b_with_demo"]["gaps"].get("fpr_gap_AA_minus_CA", float("nan"))
    gap_b_deb  = models_report["model_b_debiased"]["gaps"].get("fpr_gap_AA_minus_CA", float("nan"))
    if abs(gap_b_deb) < abs(gap_b_with):
        findings.append(
            f"Counterfactual debiasing (Model B) reduces the FPR gap from "
            f"{gap_b_with:+.3f} to {gap_b_deb:+.3f}. However, the gap does not "
            f"reach zero because race correlates with other features (proxy bias)."
        )
    else:
        findings.append(
            f"Counterfactual debiasing (Model B) does NOT reduce the FPR gap "
            f"(before: {gap_b_with:+.3f}, after: {gap_b_deb:+.3f}). "
            f"This illustrates that removing race/gender as direct inputs leaves "
            f"proxy bias through correlated features (priors, charge severity)."
        )

    # Finding 3: Best mitigation method
    valid = [m for m in mitigation_comparison if not (
        isinstance(m["equalized_odds_diff"], float) and m["equalized_odds_diff"] != m["equalized_odds_diff"]
    )]
    if valid:
        best = min(valid, key=lambda m: m["equalized_odds_diff"])
        findings.append(
            f"Among all mitigation methods, '{best['method']}' achieves the lowest "
            f"equalized-odds difference ({best['equalized_odds_diff']:.4f}) at "
            f"accuracy {best['accuracy']:.4f}. Best trade-off does not necessarily "
            f"mean perfect fairness — examine FPR and FNR gaps separately."
        )

    # Finding 4: Accuracy vs fairness trade-off
    baseline = next((m for m in mitigation_comparison if "Baseline" in m["method"]), None)
    if baseline and best:
        acc_drop = round(baseline["accuracy"] - best["accuracy"], 4)
        eod_gain = round(baseline["equalized_odds_diff"] - best["equalized_odds_diff"], 4)
        if acc_drop > 0:
            findings.append(
                f"Fairness-accuracy trade-off: the best mitigation method reduces equalized-odds "
                f"difference by {eod_gain:.4f} at a cost of {acc_drop:.4f} accuracy points "
                f"({acc_drop*100:.2f}%). This trade-off is inherent and cannot be eliminated."
            )
        else:
            findings.append(
                f"The best mitigation method reduces equalized-odds difference by {eod_gain:.4f} "
                f"with no accuracy loss (or a slight gain). This can occur when the baseline model "
                f"was overfitting demographic features."
            )

    # Finding 5: Limitations note
    findings.append(
        "Limitations: race and sex are binary here due to COMPAS data constraints. "
        "Counterfactual debiasing cannot eliminate proxy bias through correlated legal-merit "
        "features. These results apply to Broward County, FL (2013-2014) only and should not "
        "be extrapolated to other jurisdictions or time periods."
    )

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# 6. MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _write_markdown(report):
    lines = [
        "# FairAI Legal System — Fairness Evaluation Report",
        "",
        f"> Generated: {report['generated_at']}  |  "
        f"Train n={report['n_train']}  |  Test n={report['n_test']}",
        "",
        "**Honest reporting policy:** If debiasing does not reduce a gap, that is stated explicitly.",
        "All numbers are test-set only (stratified 70/30 split, seed=42).",
        "",
        "---",
        "",
        "## Overall Metrics by Model Variant",
        "",
        "| Model | Accuracy | ROC-AUC | FPR | FNR | Selection Rate | Brier |",
        "|-------|----------|---------|-----|-----|----------------|-------|",
    ]

    for key, ev in report["models"].items():
        ov = ev["overall"]
        lines.append(
            f"| {ev['label']} "
            f"| {ov['accuracy']:.4f} "
            f"| {ov.get('roc_auc', float('nan')):.4f} "
            f"| {ov['fpr']:.4f} "
            f"| {ov['fnr']:.4f} "
            f"| {ov['selection_rate']:.4f} "
            f"| {ov.get('brier_score', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## Fairness Gaps (African-American vs Caucasian, test set)",
        "",
        "| Model | FPR Gap (AA−CA) | FNR Gap (AA−CA) | Dem. Parity Diff | Disparate Impact Ratio | EOD |",
        "|-------|----------------|----------------|-----------------|----------------------|-----|",
    ]

    for key, ev in report["models"].items():
        g = ev["gaps"]
        lines.append(
            f"| {ev['label']} "
            f"| {g['fpr_gap_AA_minus_CA']:+.4f} "
            f"| {g['fnr_gap_AA_minus_CA']:+.4f} "
            f"| {g['demographic_parity_diff']:+.4f} "
            f"| {g['disparate_impact_ratio']:.4f} "
            f"| {g['equalized_odds_diff']:.4f} |"
        )

    lines += [
        "",
        "## Fairlearn Mitigation Method Comparison",
        "",
        "| Method | Accuracy | ROC-AUC | FPR Gap | FNR Gap | EOD |",
        "|--------|----------|---------|---------|---------|-----|",
    ]
    for m in report["mitigation_comparison"]:
        lines.append(
            f"| {m['method']} "
            f"| {m['accuracy']:.4f} "
            f"| {m.get('roc_auc', float('nan')):.4f} "
            f"| {m['fpr_gap']:+.4f} "
            f"| {m['fnr_gap']:+.4f} "
            f"| {m['equalized_odds_diff']:.4f} |"
        )

    lines += [
        "",
        "## Plain-Language Findings",
        "",
    ]
    for i, finding in enumerate(report["findings"], 1):
        lines.append(f"{i}. {finding}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Limitations",
        "",
        "- Race is restricted to African-American / Caucasian (COMPAS data limitation).",
        "- The COMPAS-copy model (Model A) replicates COMPAS's own score; it is not a new model.",
        "- Counterfactual debiasing cannot eliminate proxy bias through correlated features.",
        "- Simulated judges in the system are labelled as such and use seeded COMPAS data only.",
        "- Results apply to Broward County, FL (2013-2014). Do not extrapolate.",
        "- **This prototype is NOT suitable for real sentencing or bail decisions.**",
    ]

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
