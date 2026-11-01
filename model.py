"""
model.py - AI Criminal Risk Assessment Model

Implements a risk scoring algorithm.  When compas-scores-two-years.csv is
present next to this file, the weights are fitted from the real ProPublica
COMPAS dataset via linear regression (see compas_loader.py).  If the CSV is
absent the model falls back silently to the original hand-coded weights so
the application never crashes on import.

Fitted weights are cached to compas_weights.json in the same directory so
the Streamlit app does NOT refit on every page reload.

PUBLIC API (must not change — consumed by xai_shap.py and the rest of the
pipeline):
  LegalRiskModel()
  .predict_risk_score(dict) -> float in [0, 100] rounded to 1 decimal
  .get_recommendation(score) -> (verdict, reason)
  ._extract_feature_vector(dict) -> dict[str, float]
  .WEIGHTS          dict with the 14 feature keys listed below
  .BASE_INTERCEPT   float
  .POPULATION_BASELINE dict
"""

from typing import Any, Dict, Tuple
import json
import os
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH    = os.path.join(_HERE, "compas-scores-two-years.csv")
_CACHE_PATH  = os.path.join(_HERE, "compas_weights.json")

# ── Original hand-coded weights (fallback when CSV is absent) ──────────────
_ORIGINAL_WEIGHTS: Dict[str, float] = {
    "prior_felonies":          7.5,
    "prior_misdemeanors":      2.5,
    "offense_severity":        5.0,
    "age_at_arrest":          -0.4,
    "bail_failure_history":   12.0,
    "community_ties_score":   -3.0,
    "race_caste_Marginalized": 18.0,
    "race_caste_General":       0.0,
    "income_level_Low":        14.0,
    "income_level_Medium":      0.0,
    "income_level_High":       -6.0,
    "gender_Male":              4.0,
    "gender_Female":           -4.0,
    "gender_Non-binary":        2.0,
}
_ORIGINAL_INTERCEPT: float = 22.0
_ORIGINAL_BASELINE: Dict[str, Any] = {
    "prior_felonies":       1.0,
    "prior_misdemeanors":   1.5,
    "offense_severity":     4.0,
    "age_at_arrest":       32.0,
    "bail_failure_history": 0.0,
    "community_ties_score": 5.0,
    "race_caste":   "General",
    "income_level": "Medium",
    "gender":       "Male",
}


def _load_or_fit() -> Tuple[Dict[str, float], float, Dict[str, Any], str]:
    """
    Returns (WEIGHTS, BASE_INTERCEPT, POPULATION_BASELINE, source_label).
    Tries the cache first, then fits from CSV, then falls back to hand-coded.
    Never raises — guaranteed safe to call at import time.
    """
    # 1. Try the weight cache
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r") as f:
                cached = json.load(f)
            weights   = cached["weights"]
            intercept = float(cached["intercept"])
            baseline  = cached["population_baseline"]
            print("[model] Loaded COMPAS-fitted weights from cache (compas_weights.json).")
            return weights, intercept, baseline, "COMPAS-fitted (cached)"
        except Exception as e:
            print(f"[model] Cache read failed ({e}), will refit.")

    # 2. Fit from CSV if present
    if os.path.exists(_CSV_PATH):
        try:
            from compas_loader import load_compas, fit_weights, compute_population_baseline
            import pandas as pd

            df, dropped = load_compas(_CSV_PATH)
            weights, intercept = fit_weights(df)
            baseline = compute_population_baseline(df)

            # Persist cache
            cache_payload = {
                "weights": weights,
                "intercept": intercept,
                "population_baseline": baseline,
            }
            with open(_CACHE_PATH, "w") as f:
                json.dump(cache_payload, f, indent=2)
            print(f"[model] Fitted weights from COMPAS CSV ({len(df)} rows after filtering). "
                  f"Cache written to compas_weights.json.")
            return weights, intercept, baseline, "COMPAS-fitted"
        except Exception as e:
            print(f"[model] COMPAS fitting failed ({e}). Falling back to hand-coded weights.")

    # 3. Fallback
    print("[model] compas-scores-two-years.csv not found. Using original hand-coded weights.")
    return (
        _ORIGINAL_WEIGHTS.copy(),
        _ORIGINAL_INTERCEPT,
        _ORIGINAL_BASELINE.copy(),
        "hand-coded (fallback)",
    )


# ── Module-level initialisation (runs once at import) ─────────────────────
_weights, _intercept, _baseline, _weight_source = _load_or_fit()


class LegalRiskModel:
    """
    Predicts a recidivism / criminal risk score between 0 and 100
    based on accused merits and demographic background.

    When compas-scores-two-years.csv is present, the weights are fitted
    from real COMPAS data via OLS linear regression.  See compas_loader.py
    for full feature mapping and limitation notes.
    """

    # ── Public class attributes (consumed by xai_shap.py) ─────────────────
    # These are set from the module-level fitted values so that the class
    # attributes are always consistent with what predict_risk_score uses.
    WEIGHTS:             Dict[str, float] = _weights
    BASE_INTERCEPT:      float            = _intercept
    POPULATION_BASELINE: Dict[str, Any]  = _baseline

    #: Human-readable label indicating where the weights came from.
    WEIGHT_SOURCE: str = _weight_source

    def __init__(self):
        # Instance attributes mirror class attributes so both
        # instance.WEIGHTS and LegalRiskModel.WEIGHTS work.
        self.WEIGHTS             = LegalRiskModel.WEIGHTS
        self.BASE_INTERCEPT      = LegalRiskModel.BASE_INTERCEPT
        self.POPULATION_BASELINE = LegalRiskModel.POPULATION_BASELINE
        self.WEIGHT_SOURCE       = LegalRiskModel.WEIGHT_SOURCE

    def _extract_feature_vector(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Encodes raw dictionary into numerical feature mapping (14 keys)."""
        feats = {
            "prior_felonies":          float(data.get("prior_felonies", 0)),
            "prior_misdemeanors":      float(data.get("prior_misdemeanors", 0)),
            "offense_severity":        float(data.get("offense_severity", 5)),
            "age_at_arrest":           float(data.get("age_at_arrest", 30)),
            "bail_failure_history":    float(data.get("bail_failure_history", 0)),
            "community_ties_score":    float(data.get("community_ties_score", 5)),
            "race_caste_Marginalized": 1.0 if data.get("race_caste") == "Marginalized" else 0.0,
            "race_caste_General":      1.0 if data.get("race_caste") == "General"      else 0.0,
            "income_level_Low":        1.0 if data.get("income_level") == "Low"        else 0.0,
            "income_level_Medium":     1.0 if data.get("income_level") == "Medium"     else 0.0,
            "income_level_High":       1.0 if data.get("income_level") == "High"       else 0.0,
            "gender_Male":             1.0 if data.get("gender") == "Male"             else 0.0,
            "gender_Female":           1.0 if data.get("gender") == "Female"           else 0.0,
            "gender_Non-binary":       1.0 if data.get("gender") == "Non-binary"       else 0.0,
        }
        return feats

    def predict_risk_score(self, accused_data: Dict[str, Any]) -> float:
        """
        Calculates predicted risk score (0 to 100).
        Higher score = higher perceived recidivism risk.
        Always clipped to [0, 100] and rounded to 1 decimal.
        """
        feats = self._extract_feature_vector(accused_data)
        raw_score = self.BASE_INTERCEPT
        for k, v in feats.items():
            if k in self.WEIGHTS:
                raw_score += self.WEIGHTS[k] * v
        score = max(0.0, min(100.0, raw_score))
        return float(round(score, 1))

    @staticmethod
    def get_recommendation(risk_score: float) -> Tuple[str, str]:
        """
        Translates risk score to recommended legal disposition and guideline.
        Cut-offs are unchanged: <= 35 low, <= 65 moderate, > 65 high.
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
