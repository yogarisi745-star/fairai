"""
legal_system.py - Integrated AI Legal Bias Mitigation System

Coordinates the 2-level Human-in-the-Loop pipeline:
1. AI Criminal Risk Prediction
2. SHAP (XAI) Attribution & Counterfactual Debiasing
3. Disparity Gap Check & Judge (Human) Adjudication (HITL-1)
4. Cryptographic Blockchain Immutable Recording
5. AI Judicial Trend Analysis & Judicial Oversight Board Audit (HITL-2)
"""

from typing import Any, Dict, List, Optional
from model import LegalRiskModel
from xai_shap import LegalSHAPExplainer
from blockchain import JudicialBlockchain
from judicial_monitor import JudicialBiasMonitor
from demo_cases import BENCHMARK_CASES, seed_historical_blockchain


class LegalBiasMitigationSystem:
    """Master orchestrator for the dual-level human-in-the-loop legal framework."""

    def __init__(self, seed_data: bool = True):
        self.model = LegalRiskModel()
        self.explainer = LegalSHAPExplainer(self.model)
        self.blockchain = JudicialBlockchain(difficulty=2)
        self.monitor = JudicialBiasMonitor(self.blockchain)

        if seed_data:
            seed_historical_blockchain(self.blockchain, self.explainer)

    def assess_case(self, case_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1 & 2:
        - Runs AI prediction on accused profile.
        - Runs SHAP to find top contributing factors (merits vs sensitive factors).
        - Executes counterfactual debiasing and computes bias gap Delta.
        - Emits human-in-the-loop triage recommendation.
        """
        accused_data = {**case_input["demographics"], **case_input["merits"]}
        explanation = self.explainer.explain_and_evaluate(accused_data)

        bias_gap = explanation["bias_gap"]
        is_high_bias = explanation["requires_strict_human_scrutiny"]

        # Human-in-the-loop triage flag
        if is_high_bias:
            triage_guidance = (
                f"🚨 HIGH AI BIAS WARNING (Bias Gap: +{bias_gap} pts). "
                f"The AI model heavily penalized sensitive demographic attributes "
                f"({explanation['top_contributing_factor']}: {explanation['top_factor_impact']} pts). "
                f"Judge scrutiny MANDATORY: Consider overriding AI original output with debiased recommendation."
            )
        else:
            triage_guidance = (
                f"✅ LOW AI BIAS (Bias Gap: {bias_gap} pts). "
                f"Prediction is overwhelmingly grounded in legitimate criminal history merits. "
                f"Eligible for judicial fast-tracking with standard review."
            )

        return {
            "case_id": case_input["case_id"],
            "accused_id": case_input["accused_id"],
            "accused_name": case_input.get("accused_name", "Anonymous"),
            "description": case_input.get("description", ""),
            "demographics": case_input["demographics"],
            "merits": case_input["merits"],
            "ai_original_score": explanation["original_score"],
            "ai_original_verdict": explanation["original_verdict"],
            "ai_original_reasoning": explanation["original_reasoning"],
            "shap_values": explanation["shap_values"],
            "top_contributing_factor": explanation["top_contributing_factor"],
            "top_factor_impact": explanation["top_factor_impact"],
            "sensitive_attribution": explanation["sensitive_attribution"],
            "ai_debiased_score": explanation["debiased_score"],
            "ai_debiased_verdict": explanation["debiased_verdict"],
            "ai_debiased_reasoning": explanation["debiased_reasoning"],
            "bias_gap": bias_gap,
            "bias_flag": explanation["bias_flag"],
            "is_high_bias": is_high_bias,
            "triage_guidance": triage_guidance,
        }

    def judge_adjudicate_and_commit(
        self,
        case_assessment: Dict[str, Any],
        judge_id: str,
        judge_action: str,  # "ACCEPTED" or "OVERRIDDEN"
        final_verdict: str,
        judicial_reasoning: str,
    ) -> Dict[str, Any]:
        """
        Step 3 & 4:
        - Human Judge renders decision (HITL-1) with legal reasoning.
        - Decision and full audit trail are cryptographically committed to blockchain.
        """
        if not judicial_reasoning or len(judicial_reasoning.strip()) < 10:
            raise ValueError("Judicial due process requires substantive written legal reasoning.")

        committed_block = self.blockchain.record_verdict(
            case_id=case_assessment["case_id"],
            accused_id=case_assessment["accused_id"],
            demographics=case_assessment["demographics"],
            merits=case_assessment["merits"],
            ai_original_score=case_assessment["ai_original_score"],
            ai_debiased_score=case_assessment["ai_debiased_score"],
            bias_gap=case_assessment["bias_gap"],
            shap_summary=case_assessment["shap_values"],
            ai_recommended_verdict=case_assessment["ai_debiased_verdict"],
            judge_id=judge_id,
            judge_action=judge_action,
            final_verdict=final_verdict,
            judicial_reasoning=judicial_reasoning,
        )

        return {
            "status": "RECORDED_ON_BLOCKCHAIN",
            "block_index": committed_block.index,
            "block_hash": committed_block.hash,
            "previous_hash": committed_block.previous_hash,
            "timestamp": committed_block.timestamp,
            "case_id": case_assessment["case_id"],
            "judge_id": judge_id,
            "judge_action": judge_action,
            "final_verdict": final_verdict,
        }

    def audit_judge(self, judge_id: str) -> Dict[str, Any]:
        """
        Step 5:
        AI analyzes historical trends of a judge to check for bias or favoritism
        and produces an audit report for the Judicial Oversight Board.
        """
        return self.monitor.analyze_judge_trends(judge_id)

    def list_all_judges(self) -> List[str]:
        return self.monitor.get_all_judges()
