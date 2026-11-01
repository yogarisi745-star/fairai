"""
xai_shap.py - Explainable AI (SHAP) & Counterfactual Debiasing Engine

Computes exact Shapley feature attributions to inspect how demographic factors
(race/caste, income, gender) and legitimate legal merits influence AI predictions.
Performs counterfactual debiasing by neutralizing sensitive demographic attributes,
and flags cases where AI demographic disparity exceeds tolerable legal thresholds.
"""

from typing import Any, Dict, List, Tuple
from model import LegalRiskModel


class LegalSHAPExplainer:
    """
    Shapley Value Explainer & Counterfactual Debiasing Engine for legal risk models.
    """

    SENSITIVE_ATTRS = ["race_caste", "income_level", "gender"]
    LEGAL_MERIT_ATTRS = [
        "prior_felonies",
        "prior_misdemeanors",
        "offense_severity",
        "age_at_arrest",
        "bail_failure_history",
        "community_ties_score",
    ]

    # Bias disparity threshold (in risk score points)
    # A gap >= 10.0 indicates significant AI discrimination requiring intensive human judge review.
    BIAS_THRESHOLD = 10.0

    def __init__(self, model: LegalRiskModel):
        self.model = model
        self.baseline_data = LegalRiskModel.POPULATION_BASELINE.copy()
        self.baseline_score = self.model.predict_risk_score(self.baseline_data)

    def compute_shap_values(self, accused_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes exact Shapley values for each individual attribute
        relative to the baseline reference population.

        For additive models, Shapley value of feature i equals its marginal contribution:
        phi_i = f(x with feature i) - f(x with baseline feature i)
        """
        shap_values: Dict[str, float] = {}
        all_keys = self.LEGAL_MERIT_ATTRS + self.SENSITIVE_ATTRS

        current_feats = self.model._extract_feature_vector(accused_data)
        base_feats = self.model._extract_feature_vector(self.baseline_data)

        # 1. Legal Merits
        for feat in self.LEGAL_MERIT_ATTRS:
            diff = current_feats[feat] - base_feats[feat]
            weight = self.model.WEIGHTS.get(feat, 0.0)
            shap_values[feat] = round(diff * weight, 2)

        # 2. Race / Caste
        val_race = current_feats["race_caste_Marginalized"] * self.model.WEIGHTS["race_caste_Marginalized"]
        base_val_race = base_feats["race_caste_Marginalized"] * self.model.WEIGHTS["race_caste_Marginalized"]
        shap_values["race_caste"] = round(val_race - base_val_race, 2)

        # 3. Income Level
        val_inc = (
            current_feats["income_level_Low"] * self.model.WEIGHTS["income_level_Low"]
            + current_feats["income_level_High"] * self.model.WEIGHTS["income_level_High"]
        )
        base_val_inc = (
            base_feats["income_level_Low"] * self.model.WEIGHTS["income_level_Low"]
            + base_feats["income_level_High"] * self.model.WEIGHTS["income_level_High"]
        )
        shap_values["income_level"] = round(val_inc - base_val_inc, 2)

        # 4. Gender
        val_gen = (
            current_feats["gender_Male"] * self.model.WEIGHTS["gender_Male"]
            + current_feats["gender_Female"] * self.model.WEIGHTS["gender_Female"]
            + current_feats["gender_Non-binary"] * self.model.WEIGHTS["gender_Non-binary"]
        )
        base_val_gen = (
            base_feats["gender_Male"] * self.model.WEIGHTS["gender_Male"]
            + base_feats["gender_Female"] * self.model.WEIGHTS["gender_Female"]
            + base_feats["gender_Non-binary"] * self.model.WEIGHTS["gender_Non-binary"]
        )
        shap_values["gender"] = round(val_gen - base_val_gen, 2)

        return shap_values

    def perform_counterfactual_debiasing(
        self, accused_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str, str]:
        """
        Creates a counterfactual clone where sensitive protected attributes
        (caste/race, income bracket, gender) are set to neutral baseline levels,
        holding legal criminal history strictly constant.

        Returns:
            (debiased_profile, debiased_score, debiased_verdict, debiased_reasoning)
        """
        debiased_data = accused_data.copy()

        # Counterfactual intervention: neutralize protected attributes
        debiased_data["race_caste"] = "General"  # baseline parity
        debiased_data["income_level"] = "Medium"  # neutral economic baseline
        debiased_data["gender"] = "Male"  # reference gender baseline

        debiased_score = self.model.predict_risk_score(debiased_data)
        verdict, reason = self.model.get_recommendation(debiased_score)
        return debiased_data, debiased_score, verdict, reason

    def explain_and_evaluate(self, accused_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        End-to-end SHAP analysis, counterfactual debiasing, and bias-gap scoring.
        """
        original_score = self.model.predict_risk_score(accused_data)
        orig_verdict, orig_reason = self.model.get_recommendation(original_score)

        shap_values = self.compute_shap_values(accused_data)

        # Find top contributing factor overall
        sorted_factors = sorted(
            shap_values.items(), key=lambda item: abs(item[1]), reverse=True
        )
        top_factor, top_factor_impact = sorted_factors[0]

        # Calculate demographic sensitive attribution
        sensitive_attribution = sum(
            shap_values[attr] for attr in self.SENSITIVE_ATTRS if attr in shap_values
        )

        # Counterfactual debiasing
        (
            debiased_profile,
            debiased_score,
            debiased_verdict,
            debiased_reason,
        ) = self.perform_counterfactual_debiasing(accused_data)

        bias_gap = round(abs(original_score - debiased_score), 2)
        is_high_bias = bias_gap >= self.BIAS_THRESHOLD

        return {
            "original_score": original_score,
            "original_verdict": orig_verdict,
            "original_reasoning": orig_reason,
            "shap_values": shap_values,
            "top_contributing_factor": top_factor,
            "top_factor_impact": top_factor_impact,
            "sensitive_attribution": round(sensitive_attribution, 2),
            "debiased_score": debiased_score,
            "debiased_verdict": debiased_verdict,
            "debiased_reasoning": debiased_reason,
            "bias_gap": bias_gap,
            "bias_flag": "HIGH_BIAS_DETECTED" if is_high_bias else "LOW_BIAS_PASS",
            "requires_strict_human_scrutiny": is_high_bias,
            "debiased_profile": debiased_profile,
        }
