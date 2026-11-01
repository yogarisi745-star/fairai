"""
compas_cases.py - Demo cases and judicial history built ONLY from real COMPAS rows.

Every defendant here is a real row of ProPublica's compas-scores-two-years.csv,
identified only by its COMPAS `id`.  Defendant names are never read or shown.

What COMPAS cannot provide (so it is simulated and labelled as such):
  - Judges, judicial decisions and written reasoning do not exist in COMPAS.
    The two simulated judges below act on real COMPAS defendants:
      Judge A accepts the debiased AI recommendation.
      Judge B overrides it to custody for Marginalized defendants.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "compas-scores-two-years.csv")

JUDGE_FAIR = "JD-101 (Judge A)"
JUDGE_BIASED = "JD-202 (Judge B)"
JUDGE_OTHER = "JD-303 (Judge C)"
ALL_JUDGES = [JUDGE_FAIR, JUDGE_BIASED, JUDGE_OTHER]

_BAIL = "Bail Granted (Unconditional / Personal Recognizance)"
_PROBATION = "Supervised Probation / Conditional Bail with Monitoring"
_CUSTODY = "Custodial Remand / Incarceration Sentencing"


def csv_available() -> bool:
    return os.path.exists(CSV_PATH)


def _row_to_case(row: pd.Series, description: str) -> Dict[str, Any]:
    rid = int(row["id"])
    return {
        "case_id": f"COMPAS-{rid}",
        "accused_id": f"COMPAS-ID-{rid}",
        "accused_name": f"COMPAS Defendant #{rid}",
        "description": description,
        "demographics": {
            "race_caste": row["race_caste"],
            "gender": row["gender"],
        },
        "merits": {
            "prior_felonies":   int(row["prior_felonies"]),
            "offense_severity": int(row["offense_severity"]),
            "age_at_arrest":    int(row["age_at_arrest"]),
        },
        "compas_decile_score": int(row["decile_score"]),
        "compas_race_label": row["race"],
    }


def _describe(row: pd.Series, lead: str) -> str:
    charge = str(row["c_charge_desc"]).strip() if pd.notna(row["c_charge_desc"]) else "charge not recorded"
    outcome = "reoffended within 2 years" if int(row["two_year_recid"]) == 1 else "did NOT reoffend within 2 years"
    degree = "felony" if row["c_charge_degree"] == "F" else "misdemeanor"
    return (
        f"{lead} Real COMPAS record: {row['race']}, {row['sex']}, age {int(row['age_at_arrest'])}, "
        f"{int(row['priors_count'])} prior offense(s), {degree} charge ({charge}). "
        f"COMPAS decile score {int(row['decile_score'])}/10; the defendant {outcome}."
    )


def _pick(df: pd.DataFrame, mask) -> Optional[pd.Series]:
    sub = df[mask].sort_values("id")
    return sub.iloc[0] if len(sub) else None


def build_benchmark_cases(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Four deterministic real-row cases (order matters to the tests/UI):
      0 Marginalized male, young, few priors, scored Medium/High but did not reoffend
      1 General male with many priors who did reoffend (merit-driven)
      2 Marginalized female, at most 1 prior, mid-range COMPAS score
      3 'Twin' of case 0: General male, same age/priors/charge degree
    """
    c0 = _pick(df, (df.race_caste == "Marginalized") & (df.gender == "Male")
               & (df.age_at_arrest.between(21, 25)) & (df.priors_count <= 1)
               & (df.c_charge_degree == "F") & (df.decile_score >= 5) & (df.two_year_recid == 0))
    c1 = _pick(df, (df.race_caste == "General") & (df.gender == "Male")
               & (df.priors_count >= 10) & (df.decile_score >= 8) & (df.two_year_recid == 1))
    c2 = _pick(df, (df.id == 203))  # Real COMPAS row: Raw 41.6 (Probation) -> Debiased 34.9 (Bail Granted)
    c3 = _pick(df, (df.race_caste == "General") & (df.gender == "Male")
               & (df.age_at_arrest == c0["age_at_arrest"]) & (df.priors_count == c0["priors_count"])
               & (df.c_charge_degree == c0["c_charge_degree"]))
    if any(c is None for c in (c0, c1, c2, c3)):
        raise ValueError("Could not find all benchmark rows in the COMPAS data.")
    return [
        _row_to_case(c0, _describe(c0, "HIGH-BIAS EXAMPLE: demographic group inflates the risk score.")),
        _row_to_case(c1, _describe(c1, "MERIT-DRIVEN EXAMPLE: score is driven by a long prior record.")),
        _row_to_case(c2, _describe(c2, "HIGH-BIAS EXAMPLE: low-level first offense, Marginalized group.")),
        _row_to_case(c3, _describe(c3, "COUNTERFACTUAL TWIN of the first case: same age, priors and "
                                        "charge degree, but General group.")),
    ]


def build_history(df: pd.DataFrame, explainer, exclude_ids: set) -> List[Dict[str, Any]]:
    """
    Real COMPAS defendants assigned to two simulated judges.
    Judge A: accepts the debiased AI recommendation (equitable).
    Judge B: overrides to custody for Marginalized defendants (biased pattern).
    """
    def assess(row):
        c = _row_to_case(row, "")
        data = {**c["demographics"], **c["merits"]}
        return c, explainer.explain_and_evaluate(data)

    pool = df[~df["id"].isin(exclude_ids)].sort_values("id")
    records: List[Dict[str, Any]] = []

    def take(mask, n, skip=0):
        return pool[mask].iloc[skip:skip + n]

    # ---- Judge A: 3 Marginalized + 3 General, one 'high' score in each group
    fair_rows = []
    for grp in ("Marginalized", "General"):
        g = pool.race_caste == grp
        high = take(g & (pool.decile_score >= 8), 1)
        rest = take(g & (pool.decile_score <= 6), 2)
        fair_rows += list(high.itertuples(index=False)) + list(rest.itertuples(index=False))
    for i, tup in enumerate(fair_rows, 1):
        row = pd.Series(tup._asdict())
        case, ev = assess(row)
        records.append({
            "case_id": f"HIST-A-{i:02d}", "accused_id": case["accused_id"],
            "demographics": case["demographics"], "merits": case["merits"],
            "judge_id": JUDGE_FAIR, "judge_action": "ACCEPTED",
            "final_verdict": ev["debiased_verdict"],
            "reasoning": "Accepted the debiased AI assessment, which is based on the prior record and charge only.",
        })

    # ---- Judge B: 4 Marginalized (3 overridden to custody) + 3 General (accepted)
    used = {int(r["accused_id"].split("-")[-1]) for r in records}
    pool_b = pool[~pool["id"].isin(used)]
    marg = pool_b[(pool_b.race_caste == "Marginalized") & (pool_b.decile_score <= 6)
                  & (pool_b.two_year_recid == 0)].iloc[:4]
    gen = pool_b[(pool_b.race_caste == "General") & (pool_b.decile_score <= 6)].iloc[:3]
    for i, (_, row) in enumerate(marg.iterrows(), 1):
        case, ev = assess(row)
        override = i <= 3
        records.append({
            "case_id": f"HIST-B-{i:02d}", "accused_id": case["accused_id"],
            "demographics": case["demographics"], "merits": case["merits"],
            "judge_id": JUDGE_BIASED,
            "judge_action": "OVERRIDDEN" if override else "ACCEPTED",
            "final_verdict": _CUSTODY if override else ev["debiased_verdict"],
            "reasoning": ("Overrode the AI recommendation; custody ordered for deterrence."
                          if override else "Conditional release under supervision."),
        })
    for j, (_, row) in enumerate(gen.iterrows(), 1):
        case, ev = assess(row)
        records.append({
            "case_id": f"HIST-B-{4 + j:02d}", "accused_id": case["accused_id"],
            "demographics": case["demographics"], "merits": case["merits"],
            "judge_id": JUDGE_BIASED, "judge_action": "ACCEPTED",
            "final_verdict": ev["debiased_verdict"],
            "reasoning": "Routine release on the AI-recommended conditions.",
        })
    return records
