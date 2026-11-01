"""
model.py - AI Criminal Risk Assessment Model

Weights are fitted from the real ProPublica COMPAS dataset via OLS regression.
If compas-scores-two-years.csv is absent the model falls back to hand-coded weights.

Active features (10 keys):
  prior_felonies, offense_severity, age_at_arrest
  race_caste_Marginalized, race_caste_General
  gender_Male, gender_Female, gender_Non-binary

Removed (weight = 0.0 in COMPAS mode — no effect on score):
  prior_misdemeanors    — COMPAS has no felony/misdemeanor split
  bail_failure_history  — not in COMPAS
  community_ties_score  — not in COMPAS
  income_level_*        — not in COMPAS
"""

from typing import Any, Dict, Tuple
import json
import os
import numpy as np

_HERE       = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH   = os.path.join(_HERE, "compas-scores-two-years.csv")
_CACHE_PATH = os.path.join(_HERE, "compas_weights.json")

# ── Original hand-coded fallback weights (10 active keys) ─────────────────────
_ORIGINAL_WEIGHTS: Dict[str, float] = {
    "prior_felonies":          7.5,
    "offense_severity":        5.0,
    "age_at_arrest":          -0.4,
    "race_caste_Marginalized": 18.0,
    "race_caste_General":       0.0,
    "gender_Male":              4.0,
    "gender_Female":           -4.0,
    "gender_Non-binary":        2.0,
}
_ORIGINAL_INTERCEPT: float = 22.0
_ORIGINAL_BASELINE: Dict[str, Any] = {
    "prior_felonies":   1.0,
    "offense_severity": 4.0,
    "age_at_arrest":   32.0,
    "race_caste":      "General",
    "gender":          "Male",
}


def _load_or_fit() -> Tuple[Dict[str, float], float, Dict[str, Any], str]:
    """Returns (WEIGHTS, BASE_INTERCEPT, POPULATION_BASELINE, source_label). Never raises."""
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH) as f:
                cached = json.load(f)
            print("[model] Loaded COMPAS-fitted weights from cache (compas_weights.json).")
            return cached["weights"], float(cached["intercept"]), cached["population_baseline"], "COMPAS-fitted (cached)"
        except Exception as e:
            print(f"[model] Cache read failed ({e}), will refit.")

    if os.path.exists(_CSV_PATH):
        try:
            from compas_loader import load_compas, fit_weights, compute_population_baseline
            df, _ = load_compas(_CSV_PATH)
            weights, intercept = fit_weights(df)
            baseline = compute_population_baseline(df)
            with open(_CACHE_PATH, "w") as f:
                json.dump({"weights": weights, "intercept": intercept, "population_baseline": baseline}, f, indent=2)
            print(f"[model] Fitted weights from COMPAS CSV ({len(df)} rows). Cache written.")
            return weights, intercept, baseline, "COMPAS-fitted"
        except Exception as e:
            print(f"[model] COMPAS fitting failed ({e}). Using hand-coded fallback.")

    print("[model] CSV not found. Using hand-coded weights.")
    return _ORIGINAL_WEIGHTS.copy(), _ORIGINAL_INTERCEPT, _ORIGINAL_BASELINE.copy(), "hand-coded (fallback)"


_weights, _intercept, _baseline, _weight_source = _load_or_fit()


class LegalRiskModel:
    """
    Predicts a recidivism / criminal risk score between 0 and 100.
    Uses OLS-fitted weights from the ProPublica COMPAS dataset when available.
    """

    WEIGHTS:             Dict[str, float] = _weights
    BASE_INTERCEPT:      float            = _intercept
    POPULATION_BASELINE: Dict[str, Any]  = _baseline
    WEIGHT_SOURCE:       str             = _weight_source

    def __init__(self):
        self.WEIGHTS             = LegalRiskModel.WEIGHTS
        self.BASE_INTERCEPT      = LegalRiskModel.BASE_INTERCEPT
        self.POPULATION_BASELINE = LegalRiskModel.POPULATION_BASELINE
        self.WEIGHT_SOURCE       = LegalRiskModel.WEIGHT_SOURCE

    def _extract_feature_vector(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Encodes raw dict into the 8 active numerical feature keys."""
        return {
            "prior_felonies":          float(data.get("prior_felonies", 0)),
            "offense_severity":        float(data.get("offense_severity", 5)),
            "age_at_arrest":           float(data.get("age_at_arrest", 30)),
            "race_caste_Marginalized": 1.0 if data.get("race_caste") == "Marginalized" else 0.0,
            "race_caste_General":      1.0 if data.get("race_caste") == "General"      else 0.0,
            "gender_Male":             1.0 if data.get("gender") == "Male"             else 0.0,
            "gender_Female":           1.0 if data.get("gender") == "Female"           else 0.0,
            "gender_Non-binary":       1.0 if data.get("gender") == "Non-binary"       else 0.0,
        }

    def predict_risk_score(self, accused_data: Dict[str, Any]) -> float:
        """Returns risk score in [0, 100] rounded to 1 decimal."""
        feats = self._extract_feature_vector(accused_data)
        raw = self.BASE_INTERCEPT + sum(self.WEIGHTS.get(k, 0.0) * v for k, v in feats.items())
        return float(round(max(0.0, min(100.0, raw)), 1))

    @staticmethod
    def get_recommendation(risk_score: float) -> Tuple[str, str]:
        if risk_score <= 35.0:
            return ("Bail Granted (Unconditional / Personal Recognizance)",
                    "Low recidivism risk. Pre-trial release recommended with routine check-in.")
        elif risk_score <= 65.0:
            return ("Supervised Probation / Conditional Bail with Monitoring",
                    "Moderate risk. Conditional release with bail bond, community service or curfew.")
        else:
            return ("Custodial Remand / Incarceration Sentencing",
                    "High recidivism risk. Pre-trial detention or custodial sentence recommended.")
