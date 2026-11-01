"""
xai_shap.py - Explainable AI (SHAP) & Counterfactual Debiasing Engine

Computes exact Shapley feature attributions for the 3 active feature groups:
  - Legal merits:  prior_felonies, offense_severity, age_at_arrest
  - Race / caste:  race_caste (Marginalized vs General)
  - Gender:        gender (Male / Female / Non-binary)

Removed features (weight = 0, never in COMPAS): prior_misdemeanors,
bail_failure_history, community_ties_score, income_level.
"""

from typing import Any, Dict, List, Tuple
from model import LegalRiskModel


class LegalSHAPExplainer:
    """Shapley Value Explainer & Counterfactual Debiasing Engine."""

    SENSITIVE_ATTRS    = ["race_caste", "gender"]
    LEGAL_MERIT_ATTRS  = ["prior_felonies", "offense_severity", "age_at_arrest"]

    def __init__(self, model: LegalRiskModel):
        self.model = model
        self.baseline_data  = LegalRiskModel.POPULATION_BASELINE.copy()
        self.baseline_score = self.model.predict_risk_score(self.baseline_data)
        # Bias threshold: 5.0 for COMPAS-fitted weights, 10.0 for hand-coded fallback
        self.BIAS_THRESHOLD = 5.0 if "COMPAS" in model.WEIGHT_SOURCE else 10.0

    def compute_shap_values(self, accused_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Exact Shapley values for an additive linear model:
            phi_i = weight_i * (x_i - baseline_i)
        """
        shap: Dict[str, float] = {}
        cur  = self.model._extract_feature_vector(accused_data)
        base = self.model._extract_feature_vector(self.baseline_data)
        W    = self.model.WEIGHTS

        # Legal merits
        for feat in self.LEGAL_MERIT_ATTRS:
            shap[feat] = round((cur[feat] - base[feat]) * W.get(feat, 0.0), 2)

        # Race / caste: contribution of Marginalized dummy relative to baseline
        shap["race_caste"] = round(
            cur["race_caste_Marginalized"] * W.get("race_caste_Marginalized", 0.0)
            - base["race_caste_Marginalized"] * W.get("race_caste_Marginalized", 0.0), 2)

        # Gender: contribution of Female + Non-binary dummies relative to baseline
        shap["gender"] = round(
            (cur["gender_Female"]     * W.get("gender_Female",     0.0)
             + cur["gender_Non-binary"] * W.get("gender_Non-binary", 0.0))
            - (base["gender_Female"]     * W.get("gender_Female",     0.0)
               + base["gender_Non-binary"] * W.get("gender_Non-binary", 0.0)), 2)

        return shap

    def perform_counterfactual_debiasing(
        self, accused_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str, str]:
        """Neutralise race_caste → General, gender → Male; keep merits fixed."""
        cf = accused_data.copy()
        cf["race_caste"] = "General"
        cf["gender"]     = "Male"
        score = self.model.predict_risk_score(cf)
        verdict, reason = self.model.get_recommendation(score)
        return cf, score, verdict, reason

    def explain_and_evaluate(self, accused_data: Dict[str, Any]) -> Dict[str, Any]:
        """End-to-end SHAP analysis, counterfactual debiasing, and bias-gap scoring."""
        original_score = self.model.predict_risk_score(accused_data)
        orig_verdict, orig_reason = self.model.get_recommendation(original_score)

        shap_values = self.compute_shap_values(accused_data)
        sorted_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factor, top_factor_impact = sorted_factors[0]

        sensitive_attribution = sum(
            shap_values[a] for a in self.SENSITIVE_ATTRS if a in shap_values)

        cf, debiased_score, debiased_verdict, debiased_reason = (
            self.perform_counterfactual_debiasing(accused_data))

        bias_gap    = round(abs(original_score - debiased_score), 2)
        is_high_bias = bias_gap >= self.BIAS_THRESHOLD

        return {
            "original_score":          original_score,
            "original_verdict":        orig_verdict,
            "original_reasoning":      orig_reason,
            "shap_values":             shap_values,
            "top_contributing_factor": top_factor,
            "top_factor_impact":       top_factor_impact,
            "sensitive_attribution":   round(sensitive_attribution, 2),
            "debiased_score":          debiased_score,
            "debiased_verdict":        debiased_verdict,
            "debiased_reasoning":      debiased_reason,
            "bias_gap":                bias_gap,
            "bias_flag":               "HIGH_BIAS_DETECTED" if is_high_bias else "LOW_BIAS_PASS",
            "requires_strict_human_scrutiny": is_high_bias,
            "debiased_profile":        cf,
        }
