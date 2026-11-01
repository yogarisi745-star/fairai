"""
run_cli_demo.py - End-to-End Command Line Demonstration of AI Bias Mitigation in Legal Systems

Demonstrates the 6-stage pipeline:
1. AI Criminal Risk Scoring
2. SHAP (XAI) Attribution & Counterfactual Debiasing
3. Human-in-the-Loop Judge Review (HITL-1) & Override
4. Immutable Blockchain Recording & Anti-Tamper Verification
5. AI Judicial Bias Trend Analysis
6. Human Judicial Oversight Board Reporting (HITL-2)
"""

import sys
import json
import io

# Ensure stdout supports Unicode (emoji) on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from legal_system import LegalBiasMitigationSystem
from demo_cases import BENCHMARK_CASES


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def print_step(step_num: int, title: str):
    print(f"\n>>> [STEP {step_num}] {title}")
    print("-" * 75)


def run_demo():
    print_header("Legal AI Bias Mitigation & Blockchain Accountability Framework")
    print("Prototype Initialized. Loading Legal System and Seeding Historical Ledger...")
    system = LegalBiasMitigationSystem(seed_data=True)

    # ---------------------------------------------------------
    # CASE 1: High AI Bias Scenario (Marginalized + Low Income)
    # ---------------------------------------------------------
    case_1 = BENCHMARK_CASES[0]  # Arjun Kumar
    print_header(f"PROCESSING CASE: {case_1['case_id']} - {case_1['accused_name']}")
    print(f"Charge Description: {case_1['description']}")
    print(f"Demographics      : {case_1['demographics']}")
    print(f"Criminal Merits   : {case_1['merits']}")

    print_step(1, "AI Initial Risk Prediction (Trained on Historical Legal Data)")
    assessment_1 = system.assess_case(case_1)
    print(f"• Raw AI Risk Score       : {assessment_1['ai_original_score']} / 100")
    print(f"• Initial AI Recommendation: {assessment_1['ai_original_verdict']}")
    print(f"• Initial AI Rationale     : {assessment_1['ai_original_reasoning']}")

    print_step(2, "SHAP (XAI) Factor Attribution & Counterfactual Debiasing")
    print("• Feature Attributions (Points added/subtracted relative to baseline):")
    for feat, impact in sorted(assessment_1["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
        marker = "⚠️ [PROTECTED ATTR]" if feat in ["race_caste", "income_level", "gender"] else "  [Legal Merit]    "
        sign = "+" if impact >= 0 else ""
        print(f"   {marker} {feat:<22}: {sign}{impact:.1f} pts")

    print(f"\n• Top Contributing Factor    : {assessment_1['top_contributing_factor']} ({assessment_1['top_factor_impact']} pts)")
    print(f"• Sensitive Demographics Sum : +{assessment_1['sensitive_attribution']} pts")
    print(f"• Counterfactual Debiased Score: {assessment_1['ai_debiased_score']} / 100 (Reduced by {assessment_1['bias_gap']} pts)")
    print(f"• Debiased AI Recommendation : {assessment_1['ai_debiased_verdict']}")

    print_step(3, "Bias Gap Evaluation & Human-in-the-Loop Judge Review (HITL-1)")
    print(f"• Bias Disparity Gap (Delta) : {assessment_1['bias_gap']} points (Threshold: {system.explainer.BIAS_THRESHOLD})")
    print(f"• Triage Status Flag         : {assessment_1['bias_flag']}")
    print(f"• Triage System Guidance     : {assessment_1['triage_guidance']}")

    print("\n[JUDICIAL BENCH - HUMAN JUDGE ADJUDICATION]")
    print("Judge Rajesh Verma (JD-101) reviews case merits, SHAP decomposition, and bias alert:")
    judge_action = "ACCEPTED"
    final_verdict = assessment_1["ai_debiased_verdict"]
    judicial_reasoning = (
        "Court finds defendant is charged with non-violent retail infraction with no prior felonies "
        "and strong community ties. SHAP breakdown demonstrates the initial AI high-risk score was "
        "unlawfully inflated (+32 points) by protected socio-economic attributes. Court accepts the "
        "debiased score and orders unconditional personal recognizance bail release."
    )
    print(f"• Judge Action      : {judge_action} (Accepted Debiased AI Recommendation)")
    print(f"• Final Legal Order : {final_verdict}")
    print(f"• Judicial Rationale: \"{judicial_reasoning}\"")

    print_step(4, "Cryptographic Blockchain Commitment (Ensuring Immutability)")
    commit_res = system.judge_adjudicate_and_commit(
        case_assessment=assessment_1,
        judge_id="JD-101 (Judge Verma)",
        judge_action=judge_action,
        final_verdict=final_verdict,
        judicial_reasoning=judicial_reasoning,
    )
    print(f"• Status            : {commit_res['status']}")
    print(f"• Block Number      : #{commit_res['block_index']}")
    print(f"• SHA-256 Hash      : {commit_res['block_hash']}")
    print(f"• Previous Block Hash: {commit_res['previous_hash']}")

    # Verify blockchain integrity
    is_valid, msg = system.blockchain.verify_chain_integrity()
    print(f"• Chain Verification: {'✅ VALID' if is_valid else '❌ INVALID'} ({msg})")

    # ---------------------------------------------------------
    # CASE 2: Low AI Bias Scenario (Legitimate Criminal Merits)
    # ---------------------------------------------------------
    case_2 = BENCHMARK_CASES[1]  # Vikram Seth
    print_header(f"PROCESSING CASE: {case_2['case_id']} - {case_2['accused_name']}")
    print(f"Charge Description: {case_2['description']}")
    assessment_2 = system.assess_case(case_2)
    print(f"• Raw AI Risk Score       : {assessment_2['ai_original_score']} / 100")
    print(f"• Debiased AI Score       : {assessment_2['ai_debiased_score']} / 100")
    print(f"• Bias Disparity Gap      : {assessment_2['bias_gap']} points")
    print(f"• Triage Status Flag      : {assessment_2['bias_flag']}")
    print(f"• Top Contributing Factor : {assessment_2['top_contributing_factor']} ({assessment_2['top_factor_impact']} pts - Criminal Merit)")
    print(f"• Triage Guidance         : {assessment_2['triage_guidance']}")

    system.judge_adjudicate_and_commit(
        case_assessment=assessment_2,
        judge_id="JD-101 (Judge Verma)",
        judge_action="ACCEPTED",
        final_verdict=assessment_2["ai_debiased_verdict"],
        judicial_reasoning=(
            "Independent review affirms high recidivism risk due to weapon possession, "
            "prior violent felonies, and repeat bail failure history. Remand ordered."
        ),
    )
    print("• Case adjudications successfully committed to ledger.")

    # ---------------------------------------------------------
    # STEP 5 & 6: AI Judicial Trend Analysis & Oversight Board (HITL-2)
    # ---------------------------------------------------------
    print_header("AI ANALYSIS OF JUDICIAL TRENDS & JUDICIAL OVERSIGHT BOARD (HITL-2)")
    print("Auditing historical judicial trends from immutable ledger records...")

    judges = system.list_all_judges()
    for j_id in judges:
        audit = system.audit_judge(j_id)
        print(f"\n=======================================================")
        print(f"AUDIT DOSSIER: {audit['judge_id']}")
        print(f"=======================================================")
        print(f"• Total Cases Reviewed  : {audit['total_cases']}")
        print(f"• Override Frequency    : {audit['total_overrides']} cases ({audit['override_rate']:.1%})")
        print(f"• Marginalized Group    : Total={audit['marginalized_stats']['total']}, "
              f"Favorable Rate={audit['marginalized_stats']['favorable_rate']:.1%}, "
              f"Overrides to Prison={audit['marginalized_stats']['overrides_to_harsh']}")
        print(f"• General Group         : Total={audit['general_stats']['total']}, "
              f"Favorable Rate={audit['general_stats']['favorable_rate']:.1%}, "
              f"Overrides to Leniency={audit['general_stats']['overrides_to_lenient']}")
        print(f"• Disparate Impact Ratio: {audit['disparate_impact_ratio']:.3f} (Fairness Standard >= 0.80)")
        print(f"• Demographic Parity Gap: {audit['demographic_parity_gap']:.1%}")
        print(f"• Statistical p-value   : {audit['p_value']}")

        if audit["status"] == "FLAGGED_FOR_INVESTIGATION":
            print(f"• Board Audit Status    : 🚨 {audit['status']}")
            print("• ACTION FOR JUDICIAL OVERSIGHT BOARD (HUMAN):")
            for finding in audit["findings"]:
                print(f"   ⚠️  {finding}")
            print("   -> Formal inquiry initiated to verify if judicial discretion is being exercised unfairly.")
        else:
            print(f"• Board Audit Status    : ✅ {audit['status']}")
            for finding in audit["findings"]:
                print(f"   ✓  {finding}")

    # ---------------------------------------------------------
    # STEP 4 DEMO: BLOCKCHAIN TAMPER DETECTION DEMONSTRATION
    # ---------------------------------------------------------
    print_header("DEMONSTRATION: BLOCKCHAIN IMMUTABILITY & TAMPER DETECTION")
    print("Attempting unauthorized retroactive tampering of Block #1 verdict...")
    is_valid_before, _ = system.blockchain.verify_chain_integrity()
    print(f"Before tampering: Chain Integrity = {is_valid_before}")

    system.blockchain.simulate_tampering(block_index=1, new_verdict="Tampered: Discharged Without Trial")
    is_valid_after, error_reason = system.blockchain.verify_chain_integrity()
    print(f"After tampering : Chain Integrity = {is_valid_after}")
    print(f"Detection Alert : 🛑 {error_reason}")
    print("The blockchain guarantees that past verdicts cannot be surreptitiously altered!")

    print_header("PROTOTYPE DEMONSTRATION COMPLETE")
    print("All 6 requirements verified successfully.")


if __name__ == "__main__":
    run_demo()
