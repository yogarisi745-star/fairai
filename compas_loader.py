"""
compas_loader.py - ProPublica COMPAS Dataset Loader & Feature Mapper

Loads and filters the real COMPAS dataset (compas-scores-two-years.csv),
applies the standard ProPublica filters, and maps COMPAS columns to the
project's feature schema.

LIMITATIONS (documented):
  - race is restricted to African-American and Caucasian only (the two
    groups with sufficient sample size in this dataset).
  - income_level, community_ties_score, bail_failure_history are NOT
    present in COMPAS. Their project feature keys are kept with constant
    baseline values during fitting so their fitted weights come out 0.0.
  - COMPAS only provides total priors_count, not a felony/misdemeanor
    split. We set prior_felonies = priors_count and prior_misdemeanors = 0
    throughout. This means the fitted weight for prior_felonies absorbs
    the full prior-record effect.
  - gender Non-binary does not exist in COMPAS. The weight for
    gender_Non-binary is carried over from the original hard-coded value
    (2.0) and is NOT data-derived.

ProPublica data source:
  https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
"""

from typing import Dict, Tuple
import pandas as pd

# ── Mapping constants ────────────────────────────────────────────────────────

# Change these strings to reclassify COMPAS race labels without touching code.
RACE_MAP = {
    "African-American": "Marginalized",
    "Caucasian": "General",
}

# c_charge_degree -> offense_severity on project's 1-10 scale
CHARGE_SEVERITY_MAP = {
    "F": 6,   # Felony
    "M": 3,   # Misdemeanor
}

# Baseline constants for features absent from COMPAS.
# These are held fixed for every row so their regression coefficients -> 0.
INCOME_BASELINE = "Medium"
COMMUNITY_TIES_BASELINE = 5.0   # mid-point of 1-10 scale
BAIL_FAILURE_BASELINE = 0.0     # no bail failure history


def load_compas(path: str) -> Tuple[pd.DataFrame, int]:
    """
    Load the ProPublica COMPAS CSV, apply standard filters, and map columns
    to project features.

    ProPublica filter criteria (matching original analysis):
      - abs(days_b_screening_arrest) <= 30
      - is_recid != -1
      - c_charge_degree != 'O'  (Ordinance violations)
      - score_text != 'N/A'
      - race in {African-American, Caucasian}

    Returns:
        (df, rows_dropped_total)  where df has all project feature columns
        and rows_dropped_total is the total rows removed by all filters.
    """
    raw = pd.read_csv(path)
    original_count = len(raw)
    print(f"[compas_loader] Raw rows loaded      : {original_count}")

    # ── Step 1: ProPublica standard filters ──────────────────────────────────
    mask_days   = raw["days_b_screening_arrest"].abs() <= 30
    mask_recid  = raw["is_recid"] != -1
    mask_charge = raw["c_charge_degree"] != "O"
    mask_score  = raw["score_text"] != "N/A"

    df = raw[mask_days & mask_recid & mask_charge & mask_score].copy()
    after_propublica = len(df)
    dropped_propublica = original_count - after_propublica
    print(f"[compas_loader] After ProPublica filters : {after_propublica} rows "
          f"({dropped_propublica} dropped)")

    # ── Step 2: Race restriction ─────────────────────────────────────────────
    mask_race = df["race"].isin(RACE_MAP.keys())
    dropped_race = (~mask_race).sum()
    df = df[mask_race].copy()
    print(f"[compas_loader] After race filter        : {len(df)} rows "
          f"({dropped_race} dropped — races other than African-American / Caucasian)")

    total_dropped = original_count - len(df)
    print(f"[compas_loader] Total rows dropped       : {total_dropped} "
          f"(kept {len(df)} rows)")

    # ── Step 3: Map COMPAS columns to project feature schema ─────────────────

    # Race / caste
    df["race_caste"] = df["race"].map(RACE_MAP)

    # Gender  (COMPAS uses Male / Female only)
    df["gender"] = df["sex"]

    # Age at arrest
    df["age_at_arrest"] = df["age"]

    # Prior record.
    # NOTE: COMPAS provides only total priors_count, not a felony/misdemeanor
    # breakdown. We assign all priors to prior_felonies and set
    # prior_misdemeanors = 0. The fitted weight for prior_felonies therefore
    # captures the combined prior-record effect.
    df["prior_felonies"]      = df["priors_count"]
    df["prior_misdemeanors"]  = 0

    # Offense severity from charge degree
    df["offense_severity"] = df["c_charge_degree"].map(CHARGE_SEVERITY_MAP).fillna(3)

    # Features absent from COMPAS — held at baseline constants so regression
    # gives them 0.0 weight (no fabricated data).
    df["income_level"]          = INCOME_BASELINE
    df["community_ties_score"]  = COMMUNITY_TIES_BASELINE
    df["bail_failure_history"]  = BAIL_FAILURE_BASELINE

    # Target: rescale decile_score (1-10) to 0-100
    df["target_score"] = (df["decile_score"] - 1) / 9 * 100

    # Keep only project-relevant columns
    keep_cols = [
        "race_caste", "gender",
        "age_at_arrest", "prior_felonies", "prior_misdemeanors",
        "offense_severity", "income_level",
        "community_ties_score", "bail_failure_history",
        "target_score",
        # keep original for diagnostics
        "race", "sex", "decile_score", "c_charge_degree", "priors_count",
        # used to build demo cases from real rows (names are never kept)
        "id", "c_charge_desc", "two_year_recid",
    ]
    df = df[keep_cols].reset_index(drop=True)

    return df, total_dropped


def fit_weights(df: pd.DataFrame):
    """
    Fit a linear regression on the COMPAS-mapped features and return
    (WEIGHTS dict, BASE_INTERCEPT float).

    Design choices:
      - Only columns with non-zero variance are included in the regression
        matrix. community_ties_score, bail_failure_history and
        prior_misdemeanors are all held at constant baseline values across
        every row (see compas_loader constants), so including them in the
        matrix would be collinear with the intercept and produce arbitrary
        large coefficients from lstsq. They are excluded and set to 0.0.
      - One-hot encode race_caste and gender with drop_first=True so the
        reference categories (General, Male) get weight 0.0.
      - income_level is constant ("Medium") — dropped from matrix, 0.0.
      - gender_Non-binary does not appear in COMPAS; its weight is carried
        over from the original hand-coded value (2.0) and is NOT data-derived.
    """
    import numpy as np

    # ── Features with actual variance in the COMPAS-mapped data ──────────────
    # constant cols: prior_misdemeanors=0, community_ties_score=5,
    #                bail_failure_history=0, income_level="Medium"
    active_num_cols = ["prior_felonies", "offense_severity", "age_at_arrest"]

    # One-hot encode race_caste: drop "General" (reference) -> keep "Marginalized"
    # One-hot encode gender: drop "Male" (reference) -> keep "Female"
    # We do NOT use drop_first=True because the alphabetical first level may
    # not be the desired reference category.
    race_dummies   = pd.get_dummies(df["race_caste"], prefix="race_caste")
    if "race_caste_General" in race_dummies.columns:
        race_dummies = race_dummies.drop(columns=["race_caste_General"])

    gender_dummies = pd.get_dummies(df["gender"], prefix="gender")
    if "gender_Male" in gender_dummies.columns:
        gender_dummies = gender_dummies.drop(columns=["gender_Male"])

    X_df = pd.concat([df[active_num_cols], race_dummies, gender_dummies], axis=1)
    X    = X_df.values.astype(float)
    y    = df["target_score"].values.astype(float)

    # Add intercept column
    ones  = np.ones((len(X), 1))
    X_aug = np.hstack([ones, X])

    # OLS via numpy least-squares
    coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept   = float(coeffs[0])
    feat_coeffs = coeffs[1:]

    # ── Assemble WEIGHTS dict (all 14 project keys must be present) ───────────
    WEIGHTS: Dict[str, float] = {}

    # Active numerical / merit weights (data-derived)
    for i, col in enumerate(active_num_cols):
        WEIGHTS[col] = float(feat_coeffs[i])

    # Inactive numerical features — constant in data → regression weight 0.0
    WEIGHTS["prior_misdemeanors"]   = 0.0
    WEIGHTS["bail_failure_history"] = 0.0
    WEIGHTS["community_ties_score"] = 0.0

    # Race dummy: drop_first drops "race_caste_General"
    race_cols  = list(race_dummies.columns)   # ["race_caste_Marginalized"]
    race_offset = len(active_num_cols)
    for i, col in enumerate(race_cols):
        WEIGHTS[col] = float(feat_coeffs[race_offset + i])
    WEIGHTS["race_caste_General"] = 0.0       # reference category

    # Income — constant in data → 0.0
    WEIGHTS["income_level_Medium"] = 0.0
    WEIGHTS["income_level_Low"]    = 0.0
    WEIGHTS["income_level_High"]   = 0.0

    # Gender dummy: drop_first drops "gender_Male"
    gender_cols   = list(gender_dummies.columns)  # ["gender_Female"]
    gender_offset = race_offset + len(race_cols)
    for i, col in enumerate(gender_cols):
        WEIGHTS[col] = float(feat_coeffs[gender_offset + i])
    WEIGHTS["gender_Male"]      = 0.0  # reference category
    # Non-binary not in COMPAS — carry original hand-coded value (NOT data-derived)
    WEIGHTS["gender_Non-binary"] = 2.0

    return WEIGHTS, intercept


def compute_population_baseline(df: pd.DataFrame) -> dict:
    """
    Compute data-driven POPULATION_BASELINE means/modes from the fitted DataFrame.
    Categorical fields use the project's "neutral" reference values
    (General / Medium / Male) rather than the statistical mode, so that
    the counterfactual debiasing in xai_shap.py continues to work correctly.
    """
    baseline = {
        # Numeric: data-driven means
        "prior_felonies":       float(round(df["prior_felonies"].mean(), 1)),
        "prior_misdemeanors":   0.0,   # constant in fitted data
        "offense_severity":     float(round(df["offense_severity"].mean(), 1)),
        "age_at_arrest":        float(round(df["age_at_arrest"].mean(), 1)),
        "bail_failure_history": 0.0,   # constant baseline
        "community_ties_score": 5.0,   # constant baseline mid-point

        # Categorical: project reference levels (NOT statistical mode)
        # These must match the counterfactual debiasing neutralisation in xai_shap.py
        "race_caste":   "General",
        "income_level": "Medium",
        "gender":       "Male",
    }
    return baseline
