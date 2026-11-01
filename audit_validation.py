"""
audit_validation.py - Offline Judicial Bias Monitor Statistical Power Analysis (Task 3)

Run:  python audit_validation.py
Output: results/audit_validation.json

Method:
  - Uses the project's real JudicialBiasMonitor and JudicialBlockchain (imported, not reimplemented).
  - Simulates judges deciding on real COMPAS defendants (sampled with fixed seed=42).
  - "Fair" judge: always accepts the debiased recommendation (no override).
  - "Biased" judge: if defendant is Marginalized (African-American), overrides to
    custodial/harsh verdict with probability p. Otherwise accepts.
  - For each (n_cases, bias_probability) pair, runs 500 Monte Carlo simulations.
  - Reports:
      detection_rate   — fraction of biased-judge sims correctly flagged FLAGGED_FOR_INVESTIGATION
      false_flag_rate  — fraction of fair-judge sims incorrectly flagged
  - Conclusion: minimum n for reliable detection (detection_rate > 0.80, false_flag_rate < 0.05).

If false_flag_rate is high at small n, a minimum-cases guard is added to judicial_monitor.py
(MIN_CASES_BEFORE_FLAGGING = 20) and this file documents the rationale.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# Ensure project root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from blockchain import JudicialBlockchain
from judicial_monitor import JudicialBiasMonitor
from compas_loader import load_compas

RANDOM_STATE = 42
N_SIMULATIONS = 500
N_CASES_LIST = [10, 20, 50, 100, 200]
BIAS_PROBS = [0.1, 0.2, 0.3, 0.5]

CSV_PATH = os.path.join(_HERE, "compas-scores-two-years.csv")
RESULTS_DIR = os.path.join(_HERE, "results")
JSON_OUT = os.path.join(RESULTS_DIR, "audit_validation.json")

FAIR_VERDICT = "Bail Granted (Unconditional / Personal Recognizance)"
HARSH_VERDICT = "Custodial Remand / Incarceration Sentencing"

JUDGE_BIASED = "SIM-BIASED-JUDGE"
JUDGE_FAIR   = "SIM-FAIR-JUDGE"


def _make_blockchain_with_verdicts(rows, rng, judge_id, is_biased, bias_prob):
    """
    Create a fresh JudicialBlockchain and record 'n' verdicts from the given rows.
    A biased judge overrides Marginalized defendants to HARSH with probability bias_prob.
    A fair judge always accepts the debiased recommendation (FAIR_VERDICT).
    Returns a JudicialBiasMonitor built on that chain.
    """
    bc = JudicialBlockchain(difficulty=1)
    for _, row in rows.iterrows():
        race_caste = row["race_caste"]      # "Marginalized" or "General"
        is_marg = race_caste == "Marginalized"

        # Fair recommendation (debiased = neutral reference)
        debiased_verdict = FAIR_VERDICT

        if is_biased and is_marg and rng.random() < bias_prob:
            final_verdict = HARSH_VERDICT
            action = "OVERRIDDEN"
        else:
            final_verdict = debiased_verdict
            action = "ACCEPTED"

        bc.record_verdict(
            case_id=f"SIM-{row.name}",
            accused_id=f"COMPAS-ID-{row.get('id', row.name)}",
            demographics={"race_caste": race_caste, "gender": str(row.get("gender", "Male"))},
            merits={
                "prior_felonies": int(row.get("prior_felonies", 0)),
                "offense_severity": int(row.get("offense_severity", 3)),
                "age_at_arrest": int(row.get("age_at_arrest", 30)),
            },
            ai_original_score=float(row.get("target_score", 50.0)),
            ai_debiased_score=float(row.get("target_score", 50.0)),
            bias_gap=0.0,
            shap_summary={"race_caste": 5.0 if is_marg else 0.0},
            ai_recommended_verdict=debiased_verdict,
            judge_id=judge_id,
            judge_action=action,
            final_verdict=final_verdict,
            judicial_reasoning="[Simulation]",
        )
    monitor = JudicialBiasMonitor(bc)
    return monitor


def run_simulation(df, n_cases, bias_prob, n_sims, rng_master):
    """Run n_sims simulations for both fair and biased judges. Return detection/false-flag rates."""
    detection_count = 0
    false_flag_count = 0

    for sim_i in range(n_sims):
        # Sample n_cases rows (with replacement for small n)
        sampled = df.sample(n=n_cases, replace=(n_cases > len(df)), random_state=int(rng_master.integers(0, 2**31)))

        # Biased judge
        rng_b = np.random.default_rng(int(rng_master.integers(0, 2**31)))
        monitor_b = _make_blockchain_with_verdicts(sampled, rng_b, JUDGE_BIASED, True, bias_prob)
        result_b = monitor_b.analyze_judge_trends(JUDGE_BIASED)
        if result_b.get("status") == "FLAGGED_FOR_INVESTIGATION":
            detection_count += 1

        # Fair judge
        rng_f = np.random.default_rng(int(rng_master.integers(0, 2**31)))
        monitor_f = _make_blockchain_with_verdicts(sampled, rng_f, JUDGE_FAIR, False, 0.0)
        result_f = monitor_f.analyze_judge_trends(JUDGE_FAIR)
        if result_f.get("status") == "FLAGGED_FOR_INVESTIGATION":
            false_flag_count += 1

    detection_rate = detection_count / n_sims
    false_flag_rate = false_flag_count / n_sims
    return round(detection_rate, 4), round(false_flag_rate, 4)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("[audit_validation] Loading COMPAS data...")
    df, _ = load_compas(CSV_PATH)
    print(f"[audit_validation] {len(df)} rows loaded.")

    rng_master = np.random.default_rng(RANDOM_STATE)

    results_grid = []
    for n in N_CASES_LIST:
        for p in BIAS_PROBS:
            print(f"[audit_validation] n={n:3d}, p={p:.1f} — running {N_SIMULATIONS} sims...")
            det, ff = run_simulation(df, n, p, N_SIMULATIONS, rng_master)
            results_grid.append({
                "n_cases": n,
                "bias_prob": p,
                "detection_rate": det,
                "false_flag_rate": ff,
            })
            print(f"  detection_rate={det:.3f}, false_flag_rate={ff:.3f}")

    # Determine minimum n for reliable detection per bias_prob
    reliability_threshold = {"min_detection_rate": 0.80, "max_false_flag_rate": 0.05}
    min_n_by_p = {}
    for p in BIAS_PROBS:
        rows_for_p = [r for r in results_grid if r["bias_prob"] == p]
        reliable = [r for r in rows_for_p
                    if r["detection_rate"] >= reliability_threshold["min_detection_rate"]
                    and r["false_flag_rate"] <= reliability_threshold["max_false_flag_rate"]]
        if reliable:
            min_n = min(r["n_cases"] for r in reliable)
        else:
            min_n = None  # never achieved
        min_n_by_p[str(p)] = min_n

    # Check if false_flag_rate is high at small n
    ff_at_small_n = [r["false_flag_rate"] for r in results_grid if r["n_cases"] <= 10]
    high_ff_at_small_n = any(ff > 0.05 for ff in ff_at_small_n)

    # Determine overall min n (across all p>=0.2)
    reliable_overall = [r for r in results_grid
                        if r["bias_prob"] >= 0.2
                        and r["detection_rate"] >= 0.80
                        and r["false_flag_rate"] <= 0.05]
    overall_min_n = min(r["n_cases"] for r in reliable_overall) if reliable_overall else None

    # Plain-language conclusion
    conclusion_parts = []
    if overall_min_n is not None:
        conclusion_parts.append(
            f"For bias levels of 20% or more, reliable detection (detection rate >= 80%, "
            f"false-flag rate <= 5%) requires at least {overall_min_n} cases per judge."
        )
    else:
        conclusion_parts.append(
            "Reliable detection was not achieved within the tested case counts (up to 200) "
            "for the given thresholds. Larger samples or stronger bias signals are needed."
        )

    conclusion_parts.append(
        "The 6-7 case seeded histories in the demo system are demonstration-only and "
        "do not have sufficient statistical power for reliable bias detection. "
        "They illustrate the mechanism, not a validated audit."
    )

    if high_ff_at_small_n:
        conclusion_parts.append(
            f"False-flag rate for fair judges is elevated at n<=10. "
            f"The judicial_monitor.py has been updated with MIN_CASES_BEFORE_FLAGGING=20 "
            f"to prevent premature flagging on insufficient data."
        )
    else:
        conclusion_parts.append(
            "False-flag rate for fair judges remains at or below 5% even at small n in these "
            "simulations; however, the MIN_CASES_BEFORE_FLAGGING guard is still recommended "
            "as a conservative safeguard for real deployments."
        )

    conclusion = " ".join(conclusion_parts)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_simulations": N_SIMULATIONS,
        "n_cases_tested": N_CASES_LIST,
        "bias_probs_tested": BIAS_PROBS,
        "reliability_threshold": reliability_threshold,
        "high_false_flag_at_small_n": high_ff_at_small_n,
        "min_n_for_reliable_detection_by_bias_prob": min_n_by_p,
        "overall_min_n_at_bias_0_2_plus": overall_min_n,
        "results_grid": results_grid,
        "conclusion": conclusion,
        "simulation_design": (
            "Fair judge: always accepts debiased recommendation. "
            "Biased judge: overrides Marginalized defendants to custodial with probability p. "
            f"500 simulations per (n, p) pair. Seed={RANDOM_STATE}. "
            "Data: real COMPAS rows, ProPublica-filtered (5,278 rows)."
        ),
    }

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[audit_validation] Written: {JSON_OUT}")
    print(f"[audit_validation] Conclusion: {conclusion}")
    return report, high_ff_at_small_n


if __name__ == "__main__":
    main()
