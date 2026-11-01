"""
demo_cases.py - Curated Benchmark Legal Cases & Historical Seeding Data

Contains realistic case profiles illustrating:
1. High AI Bias cases where demographics unfairly inflate recidivism risk.
2. Low AI Bias cases driven purely by legitimate criminal history merits.
3. Historical seed data demonstrating judicial bias trends across multiple judges.
"""

from typing import Any, Dict, List

BENCHMARK_CASES = [
    {
        "case_id": "CRIM-2026-7041",
        "accused_id": "ACC-7041",
        "accused_name": "Arjun Kumar",
        "description": "Accused of non-violent petty retail theft (store merchandise valued at $120). No weapons involved.",
        "demographics": {
            "race_caste": "Marginalized",
            "income_level": "Low",
            "gender": "Male",
        },
        "merits": {
            "prior_felonies": 0,
            "prior_misdemeanors": 1,
            "offense_severity": 3,  # Scale 1-10
            "age_at_arrest": 24,
            "bail_failure_history": 0,
            "community_ties_score": 7,  # Local resident, primary family breadwinner
        },
    },
    {
        "case_id": "CRIM-2026-8102",
        "accused_id": "ACC-8102",
        "accused_name": "Vikram Seth",
        "description": "Arrested for armed commercial burglary with weapon possession and violent assault threat.",
        "demographics": {
            "race_caste": "General",
            "income_level": "Medium",
            "gender": "Male",
        },
        "merits": {
            "prior_felonies": 3,
            "prior_misdemeanors": 4,
            "offense_severity": 9,
            "age_at_arrest": 38,
            "bail_failure_history": 1,
            "community_ties_score": 2,
        },
    },
    {
        "case_id": "CRIM-2026-9215",
        "accused_id": "ACC-9215",
        "accused_name": "Pooja Devi",
        "description": "First-time offense, accused of check fraud under financial duress to pay medical bills.",
        "demographics": {
            "race_caste": "Marginalized",
            "income_level": "Low",
            "gender": "Female",
        },
        "merits": {
            "prior_felonies": 0,
            "prior_misdemeanors": 0,
            "offense_severity": 2,
            "age_at_arrest": 29,
            "bail_failure_history": 0,
            "community_ties_score": 8,
        },
    },
    {
        "case_id": "CRIM-2026-5534",
        "accused_id": "ACC-5534",
        "accused_name": "Rohan Mehra",
        "description": "Corporate embezzlement and falsification of financial accounts.",
        "demographics": {
            "race_caste": "General",
            "income_level": "High",
            "gender": "Male",
        },
        "merits": {
            "prior_felonies": 1,
            "prior_misdemeanors": 2,
            "offense_severity": 6,
            "age_at_arrest": 45,
            "bail_failure_history": 0,
            "community_ties_score": 6,
        },
    },
]


def seed_historical_blockchain(blockchain_instance, explainer_instance) -> None:
    """
    Populates historical cases into the blockchain ledger to simulate real-world
    judicial history across two judges:
    - Judge JD-101 (Judge Verma): Balanced, follows debiased guidance, equitable.
    - Judge JD-202 (Judge Singhania): Demonstrates systemic bias (systematically
      overriding AI towards incarceration for Marginalized defendants and leniency
      for High Income / General defendants).
    """
    historical_cases = [
        # Judge Verma (Fair / Equitable Judge)
        {
            "case_id": "HIST-VERMA-01",
            "accused_id": "ACC-H101",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 25, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-101 (Judge Verma)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Bail Granted (Unconditional / Personal Recognizance)",
            "reasoning": "Accepted debiased AI assessment; non-violent offense and strong local roots.",
        },
        {
            "case_id": "HIST-VERMA-02",
            "accused_id": "ACC-H102",
            "demographics": {"race_caste": "General", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 2, "offense_severity": 4, "age_at_arrest": 31, "bail_failure_history": 0, "community_ties_score": 6},
            "judge_id": "JD-101 (Judge Verma)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Standard conditional probation per objective criminal record.",
        },
        {
            "case_id": "HIST-VERMA-03",
            "accused_id": "ACC-H103",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Female"},
            "merits": {"prior_felonies": 1, "prior_misdemeanors": 1, "offense_severity": 4, "age_at_arrest": 28, "bail_failure_history": 0, "community_ties_score": 6},
            "judge_id": "JD-101 (Judge Verma)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Proportional probation following debiased assessment.",
        },
        {
            "case_id": "HIST-VERMA-04",
            "accused_id": "ACC-H104",
            "demographics": {"race_caste": "General", "income_level": "High", "gender": "Male"},
            "merits": {"prior_felonies": 3, "prior_misdemeanors": 3, "offense_severity": 8, "age_at_arrest": 40, "bail_failure_history": 1, "community_ties_score": 3},
            "judge_id": "JD-101 (Judge Verma)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Severe violent recidivism risk based on prior felonies and gun offense.",
        },
        {
            "case_id": "HIST-VERMA-05",
            "accused_id": "ACC-H105",
            "demographics": {"race_caste": "Marginalized", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 2, "prior_misdemeanors": 2, "offense_severity": 8, "age_at_arrest": 35, "bail_failure_history": 1, "community_ties_score": 4},
            "judge_id": "JD-101 (Judge Verma)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "High offense severity and repeat flight risk warrant remand.",
        },
        {
            "case_id": "HIST-VERMA-06",
            "accused_id": "ACC-H106",
            "demographics": {"race_caste": "General", "income_level": "Low", "gender": "Female"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 0, "offense_severity": 2, "age_at_arrest": 22, "bail_failure_history": 0, "community_ties_score": 8},
            "judge_id": "JD-101 (Judge Verma)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Bail Granted (Unconditional / Personal Recognizance)",
            "reasoning": "First time petty violation; unconditional release granted.",
        },

        # Judge Singhania (Biased Judicial Pattern - High Asymmetric Overrides against Marginalized)
        {
            "case_id": "HIST-SINGH-01",
            "accused_id": "ACC-H201",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 23, "bail_failure_history": 0, "community_ties_score": 6},
            "judge_id": "JD-202 (Judge Singhania)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Overriding AI leniency recommendation; deemed deterrence necessary.",
        },
        {
            "case_id": "HIST-SINGH-02",
            "accused_id": "ACC-H202",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Female"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 27, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-202 (Judge Singhania)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Overrode AI bail recommendation; imposing custodial deterrence.",
        },
        {
            "case_id": "HIST-SINGH-03",
            "accused_id": "ACC-H203",
            "demographics": {"race_caste": "Marginalized", "income_level": "Low", "gender": "Male"},
            "merits": {"prior_felonies": 1, "prior_misdemeanors": 2, "offense_severity": 5, "age_at_arrest": 29, "bail_failure_history": 0, "community_ties_score": 5},
            "judge_id": "JD-202 (Judge Singhania)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Custodial Remand / Incarceration Sentencing",
            "reasoning": "Rejecting debiased probation proposal; ordering immediate prison sentence.",
        },
        {
            "case_id": "HIST-SINGH-04",
            "accused_id": "ACC-H204",
            "demographics": {"race_caste": "Marginalized", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 2, "offense_severity": 4, "age_at_arrest": 30, "bail_failure_history": 0, "community_ties_score": 5},
            "judge_id": "JD-202 (Judge Singhania)",
            "judge_action": "ACCEPTED",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Conditional bail granted under strict surveillance.",
        },
        {
            "case_id": "HIST-SINGH-05",
            "accused_id": "ACC-H205",
            "demographics": {"race_caste": "General", "income_level": "High", "gender": "Male"},
            "merits": {"prior_felonies": 1, "prior_misdemeanors": 2, "offense_severity": 6, "age_at_arrest": 42, "bail_failure_history": 0, "community_ties_score": 8},
            "judge_id": "JD-202 (Judge Singhania)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Bail Granted (Unconditional / Personal Recognizance)",
            "reasoning": "High standing in business community; flight risk negligible.",
        },
        {
            "case_id": "HIST-SINGH-06",
            "accused_id": "ACC-H206",
            "demographics": {"race_caste": "General", "income_level": "High", "gender": "Female"},
            "merits": {"prior_felonies": 2, "prior_misdemeanors": 1, "offense_severity": 7, "age_at_arrest": 39, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-202 (Judge Singhania)",
            "judge_action": "OVERRIDDEN",
            "final_verdict": "Supervised Probation / Conditional Bail with Monitoring",
            "reasoning": "Overrode AI incarceration suggestion; defendant given diversion option.",
        },
        {
            "case_id": "HIST-SINGH-07",
            "accused_id": "ACC-H207",
            "demographics": {"race_caste": "General", "income_level": "Medium", "gender": "Male"},
            "merits": {"prior_felonies": 0, "prior_misdemeanors": 1, "offense_severity": 3, "age_at_arrest": 26, "bail_failure_history": 0, "community_ties_score": 7},
            "judge_id": "JD-202 (Judge Singhania)",
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
