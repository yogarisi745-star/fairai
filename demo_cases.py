"""
demo_cases.py - Curated Benchmark Legal Cases & Historical Seeding Data

DATA SOURCE: when compas-scores-two-years.csv is present, every case and every
historical record comes from real COMPAS rows (see compas_cases.py; judges are
simulated because COMPAS contains no judge data).  The hand-written profiles
below are only a fallback used when the CSV is missing.

Contains realistic case profiles illustrating:
1. High AI Bias cases where demographics unfairly inflate recidivism risk.
2. Low AI Bias cases driven purely by legitimate criminal history merits.
3. Historical seed data demonstrating judicial bias trends across multiple judges.

COMPAS data mode note:
  With real COMPAS-fitted weights (race ~5.93 pts, gender_Female ~0.82 pts,
  income 0.0 pts) the maximum demographic bias gap is ~6.75 pts.
  The HIGH_BIAS_DETECTED threshold is therefore 5.0 pts in COMPAS mode
  (10.0 pts for the hand-coded fallback), so Marginalized cases are flagged
  and General cases are not.
"""

from typing import Any, Dict, List

_FALLBACK_BENCHMARK_CASES = [
    {
        "case_id": "CRIM-2026-7041",
        "accused_id": "ACC-7041",
        "accused_name": "Demo Defendant 1",
        "description": (
            "Non-violent petty retail theft. Demonstrates demographic bias: "
            "Marginalized + Female inflates the score relative to the debiased baseline."
        ),
        "demographics": {"race_caste": "Marginalized", "gender": "Female"},
        "merits": {"prior_felonies": 0, "offense_severity": 3, "age_at_arrest": 24},
    },
    {
        "case_id": "CRIM-2026-8102",
        "accused_id": "ACC-8102",
        "accused_name": "Demo Defendant 2",
        "description": "Armed commercial burglary with weapon possession and violent assault threat.",
        "demographics": {"race_caste": "General", "gender": "Male"},
        "merits": {"prior_felonies": 3, "offense_severity": 9, "age_at_arrest": 38},
    },
    {
        "case_id": "CRIM-2026-9215",
        "accused_id": "ACC-9215",
        "accused_name": "Demo Defendant 3",
        "description": "First-time check fraud under financial duress.",
        "demographics": {"race_caste": "Marginalized", "gender": "Female"},
        "merits": {"prior_felonies": 0, "offense_severity": 2, "age_at_arrest": 29},
    },
    {
        "case_id": "CRIM-2026-5534",
        "accused_id": "ACC-5534",
        "accused_name": "Demo Defendant 4",
        "description": "Corporate embezzlement and falsification of financial accounts.",
        "demographics": {"race_caste": "General", "gender": "Male"},
        "merits": {"prior_felonies": 1, "offense_severity": 6, "age_at_arrest": 45},
    },
]


def _load_compas_cases():
    """Real COMPAS rows when the CSV is present, otherwise None."""
    try:
        import compas_cases
        if not compas_cases.csv_available():
            return None, None
        from compas_loader import load_compas
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = load_compas(compas_cases.CSV_PATH)
        return compas_cases, df
    except Exception as exc:  # never break import
        print(f"[demo_cases] COMPAS cases unavailable ({exc}); using hand-written fallback cases.")
        return None, None


_cc, _df = _load_compas_cases()
DATA_SOURCE = "COMPAS (real rows)" if _cc is not None else "hand-written fallback"
BENCHMARK_CASES = _cc.build_benchmark_cases(_df) if _cc is not None else _FALLBACK_BENCHMARK_CASES


def seed_historical_blockchain(blockchain_instance, explainer_instance) -> None:
    """Seed the ledger from real COMPAS defendants (simulated judges), or the hand-written fallback."""
    if _cc is None:
        return _seed_handwritten(blockchain_instance, explainer_instance)
    exclude = {int(c["case_id"].split("-")[-1]) for c in BENCHMARK_CASES}
    for c in _cc.build_history(_df, explainer_instance, exclude):
        accused_data = {**c["demographics"], **c["merits"]}
        ev = explainer_instance.explain_and_evaluate(accused_data)
        blockchain_instance.record_verdict(
            case_id=c["case_id"], accused_id=c["accused_id"],
            demographics=c["demographics"], merits=c["merits"],
            ai_original_score=ev["original_score"], ai_debiased_score=ev["debiased_score"],
            bias_gap=ev["bias_gap"], shap_summary=ev["shap_values"],
            ai_recommended_verdict=ev["debiased_verdict"],
            judge_id=c["judge_id"], judge_action=c["judge_action"],
            final_verdict=c["final_verdict"], judicial_reasoning=c["reasoning"],
        )


def _seed_handwritten(blockchain_instance, explainer_instance) -> None:
    """
    Populates historical cases into the blockchain ledger to simulate real-world
    judicial history across two judges:
    - Judge JD-101 (Judge A): Balanced, follows debiased guidance, equitable.
    - Judge JD-202 (Judge B): Demonstrates systemic bias (systematically
      overriding AI towards incarceration for Marginalized defendants and leniency
      for High Income / General defendants).
    """
    historical_cases = [
        # Judge A (Fair / Equitable Judge)
        {
            "case_id": "HIST-VERMA-01",
            "accused_id": "ACC-H101",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 25, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-101 (Judge A)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Bail Granted (Unconditional / Personal Recognizance)",
            "reasoning": "Accepted debiased AI assessment; non-violent offense and strong local roots.",
        },
        {
            "case_id": "HIST-VERMA-02",
            "accused_id": "ACC-H102",
            "demographics": {"race_caste": "General", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 2, "offense_severity": 4, "age_at_arrest": 31, "bail_failure_history": 0, "community_ties_score": 6},
            "judge_id": "JD-101 (Judge A)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Standard conditional probation per objective criminal record.",
        },
        {
            "case_id": "HIST-VERMA-03",
            "accused_id": "ACC-H103",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Female"},
            "merits": {"prior_felonies": 1, "prior_misdemeanors": 1, "offense_severity": 4, "age_at_arrest": 28, "bail_failure_history": 0, "community_ties_score": 6},
            "judge_id": "JD-101 (Judge A)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Proportional probation following debiased assessment.",
        },
        {
            "case_id": "HIST-VERMA-04",
            "accused_id": "ACC-H104",
            "demographics": {"race_caste": "General", "income_level": "High", "gender": "Male"},
            "merits": {"prior_felonies": 3, "prior_misdemeanors": 3, "offense_severity": 8, "age_at_arrest": 40, "bail_failure_history": 1, "community_ties_score": 3},
            "judge_id": "JD-101 (Judge A)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Severe violent recidivism risk based on prior felonies and gun offense.",
        },
        {
            "case_id": "HIST-VERMA-05",
            "accused_id": "ACC-H105",
            "demographics": {"race_caste": "Marginalized", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 2, "prior_misdemeanors": 2, "offense_severity": 8, "age_at_arrest": 35, "bail_failure_history": 1, "community_ties_score": 4},
            "judge_id": "JD-101 (Judge A)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "High offense severity and repeat flight risk warrant remand.",
        },
        {
            "case_id": "HIST-VERMA-06",
            "accused_id": "ACC-H106",
            "demographics": {"race_caste": "General", "income_level": "Low", "gender": "Female"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 0, "offense_severity": 2, "age_at_arrest": 22, "bail_failure_history": 0, "community_ties_score": 8},
            "judge_id": "JD-101 (Judge A)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Bail Granted (Unconditional / Personal Recognizance)",
            "reasoning": "First time petty violation; unconditional release granted.",
        },

        # Judge B (Biased Judicial Pattern - High Asymmetric Overrides against Marginalized)
        {
            "case_id": "HIST-SINGH-01",
            "accused_id": "ACC-H201",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 23, "bail_failure_history": 0, "community_ties_score": 6},
            "judge_id": "JD-202 (Judge B)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Overriding AI leniency recommendation; deemed deterrence necessary.",
        },
        {
            "case_id": "HIST-SINGH-02",
            "accused_id": "ACC-H202",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Female"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 27, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-202 (Judge B)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Overrode AI bail recommendation; imposing custodial deterrence.",
        },
        {
            "case_id": "HIST-SINGH-03",
            "accused_id": "ACC-H203",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Male"},
            "merits": {"prior_felonies": 1, "prior_misdemeanors": 2, "offense_severity": 5, "age_at_arrest": 29, "bail_failure_history": 0, "community_ties_score": 5},
            "judge_id": "JD-202 (Judge B)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Rejecting debiased probation proposal; ordering immediate prison sentence.",
        },
        {
            "case_id": "HIST-SINGH-04",
            "accused_id": "ACC-H204",
            "demographics": {"race_caste": "Marginalized", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 2, "offense_severity": 4, "age_at_arrest": 30, "bail_failure_history": 0, "community_ties_score": 5},
            "judge_id": "JD-202 (Judge B)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Conditional bail granted under strict surveillance.",
        },
        {
            "case_id": "HIST-SINGH-05",
            "accused_id": "ACC-H205",
            "demographics": {"race_caste": "General", "income_level": "High", "gender": "Male"},
            "merits": {"prior_felonies": 1, "prior_misdemeanors": 2, "offense_severity": 6, "age_at_arrest": 42, "bail_failure_history": 0, "community_ties_score": 8},
            "judge_id": "JD-202 (Judge B)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Bail Granted (Unconditional / Personal Recognizance)",
            "reasoning": "High standing in business community; flight risk negligible.",
        },
        {
            "case_id": "HIST-SINGH-06",
            "accused_id": "ACC-H206",
            "demographics": {"race_caste": "General", "income_level": "High", "gender": "Female"},
            "merits": {"prior_felonies": 2, "prior_misdemeanors": 1, "offense_severity": 7, "age_at_arrest": 39, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-202 (Judge B)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Overrode AI incarceration suggestion; defendant given diversion option.",
        },
        {
            "case_id": "HIST-SINGH-07",
            "accused_id": "ACC-H207",
            "demographics": {"race_caste": "General", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 26, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-202 (Judge B)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Bail Granted (Unconditional / Personal Recognizance)",
            "reasoning": "Routine bail for low-level property damage.",
        },
    ]

    for c in historical_cases:
        accused_data = {**c["demographics"], **c["merits"]}
        eval_result = explainer_instance.explain_and_evaluate(accused_data)

        blockchain_instance.record_verdict(
            case_id=c["case_id"],
            accused_id=c["accused_id"],
            demographics=c["demographics"],
            merits=c["merits"],
            ai_original_score=eval_result["original_score"],
            ai_debiased_score=eval_result["debiased_score"],
            bias_gap=eval_result["bias_gap"],
            shap_summary=eval_result["shap_values"],
            ai_recommended_verdict=eval_result["debiased_verdict"],
            judge_id=c["judge_id"],
            judge_action=c["judge_action"],
            final_verdict=c["final_verdict"],
            judicial_reasoning=c["reasoning"],
        )
