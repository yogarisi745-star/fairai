"""
test_pipeline.py - Verification & Unit Test Suite for AI Bias Mitigation System
"""

import unittest
from model import LegalRiskModel
from xai_shap import LegalSHAPExplainer
from blockchain import JudicialBlockchain
from judicial_monitor import JudicialBiasMonitor
from legal_system import LegalBiasMitigationSystem
from demo_cases import BENCHMARK_CASES


class TestLegalBiasMitigation(unittest.TestCase):

    def setUp(self):
        self.model = LegalRiskModel()
        self.explainer = LegalSHAPExplainer(self.model)
        self.blockchain = JudicialBlockchain(difficulty=1)
        self.monitor = JudicialBiasMonitor(self.blockchain)
        self.system = LegalBiasMitigationSystem(seed_data=True)

    def test_model_prediction_bounds(self):
        """Verify model output is always bounded within [0, 100]."""
        high_risk_profile = {
            "prior_felonies": 10,
            "prior_misdemeanors": 10,
            "offense_severity": 10,
            "age_at_arrest": 20,
            "bail_failure_history": 1,
            "community_ties_score": 1,
            "race_caste": "Marginalized",
            "income_level": "Low",
            "gender": "Male",
        }
        score = self.model.predict_risk_score(high_risk_profile)
        self.assertTrue(0.0 <= score <= 100.0)

        low_risk_profile = {
            "prior_felonies": 0,
            "prior_misdemeanors": 0,
            "offense_severity": 1,
            "age_at_arrest": 60,
            "bail_failure_history": 0,
            "community_ties_score": 10,
            "race_caste": "General",
            "income_level": "High",
            "gender": "Female",
        }
        score_low = self.model.predict_risk_score(low_risk_profile)
        self.assertTrue(0.0 <= score_low <= 100.0)
        self.assertTrue(score > score_low)

    def test_shap_and_counterfactual_debiasing(self):
        """Test SHAP explains sensitive features and counterfactual debiasing reduces score."""
        marginalized_case = BENCHMARK_CASES[0]  # Arjun Kumar
        accused_data = {**marginalized_case["demographics"], **marginalized_case["merits"]}

        res = self.explainer.explain_and_evaluate(accused_data)
        self.assertIn("shap_values", res)
        self.assertIn("race_caste", res["shap_values"])
        self.assertIn("income_level", res["shap_values"])

        # Protected attributes should have positive attribution on unmitigated model
        self.assertGreater(res["shap_values"]["race_caste"], 0)
        self.assertGreater(res["shap_values"]["income_level"], 0)

        # Debiased score should be lower than original score
        self.assertLess(res["debiased_score"], res["original_score"])
        self.assertTrue(res["requires_strict_human_scrutiny"])
        self.assertEqual(res["bias_flag"], "HIGH_BIAS_DETECTED")

    def test_blockchain_immutability_and_tampering(self):
        """Verify blockchain detects unauthorized retroactive data tampering."""
        is_valid, msg = self.blockchain.verify_chain_integrity()
        self.assertTrue(is_valid)

        # Add a test block
        self.blockchain.record_verdict(
            case_id="TEST-CASE-01",
            accused_id="TEST-ACC-01",
            demographics={"race_caste": "General", "income_level": "Medium", "gender": "Male"},
            merits={"prior_felonies": 0, "prior_misdemeanors": 0, "offense_severity": 1, "age_at_arrest": 30, "bail_failure_history": 0, "community_ties_score": 8},
            ai_original_score=20.0,
            ai_debiased_score=20.0,
            bias_gap=0.0,
            shap_summary={"race_caste": 0.0},
            ai_recommended_verdict="Bail Granted",
            judge_id="JD-TEST",
            judge_action="ACCEPTED",
            final_verdict="Bail Granted",
            judicial_reasoning="Defendant is a first-time non-violent offender.",
        )

        is_valid, _ = self.blockchain.verify_chain_integrity()
        self.assertTrue(is_valid)

        # Tamper with block
        self.blockchain.simulate_tampering(1, "Tampered Verdict")
        is_tampered_valid, reason = self.blockchain.verify_chain_integrity()
        self.assertFalse(is_tampered_valid)
        self.assertIn("hash mismatch", reason)

    def test_judicial_bias_monitoring_and_board_alerts(self):
        """Verify judicial bias monitoring catches biased Judge Singhania and clears Judge Verma."""
        # Judge Verma should be compliant
        audit_verma = self.system.audit_judge("JD-101 (Judge Verma)")
        self.assertEqual(audit_verma["status"], "EQUITABLE_COMPLIANT")
        self.assertFalse(audit_verma["requires_board_review"])

        # Judge Singhania should be flagged for investigation
        audit_singhania = self.system.audit_judge("JD-202 (Judge Singhania)")
        self.assertEqual(audit_singhania["status"], "FLAGGED_FOR_INVESTIGATION")
        self.assertTrue(audit_singhania["requires_board_review"])
        self.assertLess(audit_singhania["disparate_impact_ratio"], 0.80)


if __name__ == "__main__":
    unittest.main()
