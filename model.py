"""
model.py - AI Criminal Risk Assessment Model

Implements a risk scoring algorithm trained on historical legal data.
Crucially captures the real-world dilemma: historical training datasets
often incorporate systemic societal disparities (penalizing marginalized
caste/race or lower income individuals for equivalent criminal history).
"""

from typing import Any, Dict, Tuple
import numpy as np


class LegalRiskModel:
    """
    Predicts a recidivism / criminal risk score between 0 and 100
    based on accused merits and demographic background.
    """

    # Baseline reference for neutral population
    POPULATION_BASELINE = {
        "prior_felonies": 1.0,
        "prior_misdemeanors": 1.5,
        "offense_severity": 4.0,  # 1 (low) to 10 (violent/severe)
        "age_at_arrest": 32.0,
        "bail_failure_history": 0.0,  # 0: no, 1: yes
        "community_ties_score": 5.0,  # 1 (isolated) to 10 (strong employment/family)
        "race_caste": "General",  # "General" vs "Marginalized"
        "income_level": "Medium",  # "Low", "Medium", "High"
        "gender": "Male",  # "Male", "Female", "Non-binary"
    }

    # Model Weights (reflecting historical bias present in unmitigated criminal justice data)
    WEIGHTS = {
        # Legitimate legal merits
        "prior_felonies": 7.5,
        "prior_misdemeanors": 2.5,
        "offense_severity": 5.0,
        "age_at_arrest": -0.4,  # older defendants statistically lower recidivism
        "bail_failure_history": 12.0,
        "community_ties_score": -3.0,  # strong ties lower risk
        # Sensitive / Protected demographic factors (SYSTEMIC BIAS SOURCE)
        "race_caste_Marginalized": 18.0,  # +18 points purely due to historical bias
        "race_caste_General": 0.0,
        "income_level_Low": 14.0,  # +14 points penalizing poverty
        "income_level_Medium": 0.0,
        "income_level_High": -6.0,  # favors wealthy defendants
        "gender_Male": 4.0,
        "gender_Female": -4.0,
        "gender_Non-binary": 2.0,
    }

    BASE_INTERCEPT = 22.0

    def __init__(self):
        pass

    def _extract_feature_vector(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Encodes raw dictionary into numerical feature mapping."""
        feats = {
            "prior_felonies": float(data.get("prior_felonies", 0)),
            "prior_misdemeanors": float(data.get("prior_misdemeanors", 0)),
            "offense_severity": float(data.get("offense_severity", 5)),
            "age_at_arrest": float(data.get("age_at_arrest", 30)),
            "bail_failure_history": float(data.get("bail_failure_history", 0)),
            "community_ties_score": float(data.get("community_ties_score", 5)),
            "race_caste_Marginalized": 1.0 if data.get("race_caste") == "Marginalized" else 0.0,
            "race_caste_General": 1.0 if data.get("race_caste") == "General" else 0.0,
            "income_level_Low": 1.0 if data.get("income_level") == "Low" else 0.0,
            "income_level_Medium": 1.0 if data.get("income_level") == "Medium" else 0.0,
            "income_level_High": 1.0 if data.get("income_level") == "High" else 0.0,
            "gender_Male": 1.0 if data.get("gender") == "Male" else 0.0,
            "gender_Female": 1.0 if data.get("gender") == "Female" else 0.0,
            "gender_Non-binary": 1.0 if data.get("gender") == "Non-binary" else 0.0,
        }
        return feats

    def predict_risk_score(self, accused_data: Dict[str, Any]) -> float:
        """
        Calculates predicted risk score (0 to 100).
        Higher score = higher perceived recidivism risk.
        """
        feats = self._extract_feature_vector(accused_data)
        raw_score = self.BASE_INTERCEPT

        for k, v in feats.items():
            if k in self.WEIGHTS:
                raw_score += self.WEIGHTS[k] * v

        # Apply a mild sigmoid-like boundary compression to keep within [0, 100]
        score = max(0.0, min(100.0, raw_score))
        return float(round(score, 1))

    @staticmethod
    def get_recommendation(risk_score: float) -> Tuple[str, str]:
        """
        Translates risk score to recommended legal disposition and guideline.
        """
        if risk_score <= 35.0:
            return (
                "Bail Granted (Unconditional / Personal Recognizance)",
                "Low recidivism risk. Pre-trial release recommended with routine check-in.",
            )
        elif risk_score <= 65.0:
            return (
                "Supervised Probation / Conditional Bail with Monitoring",
                "Moderate risk. Conditional release with bail bond, community service or curfew.",
            )
        else:
            return (
                "Custodial Remand / Incarceration Sentencing",
                "High recidivism risk. Pre-trial detention or custodial sentence recommended.",
            )
