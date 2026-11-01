"""
test_pipeline.py - Verification & Unit Test Suite for AI Bias Mitigation System

Active features (8 keys): prior_felonies, offense_severity, age_at_arrest,
race_caste_Marginalized, race_caste_General, gender_Male, gender_Female, gender_Non-binary.

Removed (weight=0, not in COMPAS): prior_misdemeanors, bail_failure_history,
community_ties_score, income_level_*.
"""

import os
import sys
import unittest

from model import LegalRiskModel
from xai_shap import LegalSHAPExplainer
from blockchain import JudicialBlockchain
from judicial_monitor import JudicialBiasMonitor
from legal_system import LegalBiasMitigationSystem
from demo_cases import BENCHMARK_CASES


class TestModelAndWeights(unittest.TestCase):

    def setUp(self):
        self.model = LegalRiskModel()

    def test_weights_has_all_8_keys(self):
        """WEIGHTS must contain exactly the 8 active feature keys."""
        expected_keys = {
            "prior_felonies", "offense_severity", "age_at_arrest",
            "race_caste_Marginalized", "race_caste_General",
            "gender_Male", "gender_Female", "gender_Non-binary",
        }
        self.assertEqual(set(self.model.WEIGHTS.keys()), expected_keys)

    def test_model_prediction_bounds(self):
        """Score is always in [0, 100] and high-risk > low-risk."""
        high_risk = {"prior_felonies": 10, "offense_severity": 10,
                     "age_at_arrest": 20, "race_caste": "Marginalized", "gender": "Male"}
        low_risk  = {"prior_felonies": 0,  "offense_severity": 1,
                     "age_at_arrest": 60, "race_caste": "General", "gender": "Female"}
        s_high = self.model.predict_risk_score(high_risk)
        s_low  = self.model.predict_risk_score(low_risk)
        self.assertTrue(0.0 <= s_high <= 100.0)
        self.assertTrue(0.0 <= s_low  <= 100.0)
        self.assertGreater(s_high, s_low)

    def test_fallback_weights_has_8_keys(self):
        """Original fallback WEIGHTS must also define the 8 active keys."""
        import model as m
        self.assertEqual(len(m._ORIGINAL_WEIGHTS), 8)
        self.assertIsInstance(m._ORIGINAL_INTERCEPT, float)

    def test_weight_source_is_set(self):
        self.assertIsInstance(self.model.WEIGHT_SOURCE, str)
        self.assertGreater(len(self.model.WEIGHT_SOURCE), 0)


class TestCompasLoader(unittest.TestCase):

    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "compas-scores-two-years.csv")

    def setUp(self):
        if not os.path.exists(self.CSV_PATH):
            self.skipTest("compas-scores-two-years.csv not present.")

    def test_loader_row_count_and_filters(self):
        from compas_loader import load_compas
        df, total_dropped = load_compas(self.CSV_PATH)
        self.assertGreaterEqual(len(df), 5000)
        self.assertLessEqual(len(df), 6500)
        self.assertGreater(total_dropped, 0)
        # Active columns present
        for col in ["race_caste", "gender", "age_at_arrest",
                    "prior_felonies", "offense_severity", "target_score"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        # Removed columns must NOT be in the dataframe
        for col in ["prior_misdemeanors", "income_level",
                    "community_ties_score", "bail_failure_history"]:
            self.assertNotIn(col, df.columns, f"Removed column still present: {col}")
        self.assertTrue(df["race_caste"].isin(["General", "Marginalized"]).all())
        self.assertTrue((df["target_score"] >= 0).all())
        self.assertTrue((df["target_score"] <= 100).all())


class TestSHAPAndDebiasing(unittest.TestCase):

    def setUp(self):
        self.model     = LegalRiskModel()
        self.explainer = LegalSHAPExplainer(self.model)

    def test_shap_additivity(self):
        """sum(SHAP values) == score - baseline_score within 0.1 rounding tolerance."""
        profiles = [
            BENCHMARK_CASES[0],
            BENCHMARK_CASES[1],
            {"demographics": {"race_caste": "General", "gender": "Female"},
             "merits":       {"prior_felonies": 2, "offense_severity": 7, "age_at_arrest": 45}},
        ]
        for case in profiles:
            accused_data = {**case["demographics"], **case["merits"]}
            result   = self.explainer.explain_and_evaluate(accused_data)
            shap_sum = sum(result["shap_values"].values())
            expected = result["original_score"] - self.explainer.baseline_score
            self.assertAlmostEqual(shap_sum, expected, delta=0.1,
                msg=f"SHAP additivity failed: sum={shap_sum:.3f} expected={expected:.3f}")

    def test_shap_keys_are_race_caste_and_gender_and_merits(self):
        """SHAP result must contain race_caste, gender — NOT income_level."""
        accused_data = {**BENCHMARK_CASES[0]["demographics"], **BENCHMARK_CASES[0]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertIn("race_caste", result["shap_values"])
        self.assertIn("gender",     result["shap_values"])
        self.assertNotIn("income_level",          result["shap_values"])
        self.assertNotIn("bail_failure_history",  result["shap_values"])
        self.assertNotIn("community_ties_score",  result["shap_values"])
        self.assertNotIn("prior_misdemeanors",    result["shap_values"])

    def test_race_caste_shap_positive_for_marginalized(self):
        accused_data = {**BENCHMARK_CASES[0]["demographics"], **BENCHMARK_CASES[0]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertGreater(result["shap_values"]["race_caste"], 0)

    def test_counterfactual_debiasing_no_keyerror(self):
        for case in BENCHMARK_CASES:
            accused_data = {**case["demographics"], **case["merits"]}
            _, score, verdict, _ = self.explainer.perform_counterfactual_debiasing(accused_data)
            self.assertIsInstance(score, float)
            self.assertTrue(0.0 <= score <= 100.0)

    def test_high_bias_demo_case_flags_scrutiny(self):
        """BENCHMARK_CASES[0] (Marginalized+Female) must trigger HIGH_BIAS_DETECTED."""
        accused_data = {**BENCHMARK_CASES[0]["demographics"], **BENCHMARK_CASES[0]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertGreater(result["shap_values"]["race_caste"], 0)
        self.assertLess(result["debiased_score"], result["original_score"])
        self.assertGreaterEqual(result["bias_gap"], self.explainer.BIAS_THRESHOLD)
        self.assertEqual(result["bias_flag"], "HIGH_BIAS_DETECTED")
        self.assertTrue(result["requires_strict_human_scrutiny"])

    def test_low_bias_demo_case_has_zero_race_attribution(self):
        """BENCHMARK_CASES[1] (General/Male) — race SHAP = 0, bias_gap = 0."""
        accused_data = {**BENCHMARK_CASES[1]["demographics"], **BENCHMARK_CASES[1]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertEqual(result["shap_values"]["race_caste"], 0.0)
        self.assertEqual(result["bias_gap"], 0.0)
        self.assertEqual(result["bias_flag"], "LOW_BIAS_PASS")

    def test_threshold_follows_weight_source(self):
        expected = 10.0 if self.model.WEIGHT_SOURCE.startswith("hand-coded") else 5.0
        self.assertEqual(self.explainer.BIAS_THRESHOLD, expected)


class TestCompasOnlyDemo(unittest.TestCase):

    def setUp(self):
        import demo_cases
        if demo_cases.DATA_SOURCE != "COMPAS (real rows)":
            self.skipTest("COMPAS CSV not present.")
        self.dc = demo_cases

    def test_cases_are_real_compas_rows(self):
        for c in self.dc.BENCHMARK_CASES:
            self.assertTrue(c["case_id"].startswith("COMPAS-"))
            self.assertIn("COMPAS Defendant #", c["accused_name"])
            self.assertIn("compas_decile_score", c)

    def test_twin_case_has_same_merits_different_group(self):
        a, b = self.dc.BENCHMARK_CASES[0], self.dc.BENCHMARK_CASES[3]
        self.assertEqual(a["merits"], b["merits"])
        self.assertNotEqual(a["demographics"]["race_caste"], b["demographics"]["race_caste"])

    def test_seeded_history_uses_compas_ids(self):
        system = LegalBiasMitigationSystem(seed_data=True)
        ids = [blk.case_data.get("accused_id", "") for blk in system.blockchain.chain[1:]]
        ids = [i for i in ids if i]
        self.assertTrue(ids and all(i.startswith("COMPAS-ID-") for i in ids))


class TestBlockchain(unittest.TestCase):

    def setUp(self):
        self.blockchain = JudicialBlockchain(difficulty=1)

    def test_blockchain_immutability_and_tampering(self):
        is_valid, _ = self.blockchain.verify_chain_integrity()
        self.assertTrue(is_valid)
        self.blockchain.record_verdict(
            case_id="TEST-01", accused_id="TEST-ACC-01",
            demographics={"race_caste": "General", "gender": "Male"},
            merits={"prior_felonies": 0, "offense_severity": 1, "age_at_arrest": 30},
            ai_original_score=20.0, ai_debiased_score=20.0, bias_gap=0.0,
            shap_summary={"race_caste": 0.0},
            ai_recommended_verdict="Bail Granted",
            judge_id="JD-TEST", judge_action="ACCEPTED",
            final_verdict="Bail Granted",
            judicial_reasoning="First-time non-violent offender.",
        )
        is_valid, _ = self.blockchain.verify_chain_integrity()
        self.assertTrue(is_valid)
        self.blockchain.simulate_tampering(1, "Tampered Verdict")
        is_tampered, reason = self.blockchain.verify_chain_integrity()
        self.assertFalse(is_tampered)
        self.assertIn("hash mismatch", reason)


class TestJudicialMonitoring(unittest.TestCase):

    def setUp(self):
        self.system = LegalBiasMitigationSystem(seed_data=True)

    def test_judicial_bias_monitoring_and_board_alerts(self):
        audit_a = self.system.audit_judge("JD-101 (Judge A)")
        self.assertEqual(audit_a["status"], "EQUITABLE_COMPLIANT")
        self.assertFalse(audit_a["requires_board_review"])

        audit_b = self.system.audit_judge("JD-202 (Judge B)")
        self.assertEqual(audit_b["status"], "FLAGGED_FOR_INVESTIGATION")
        self.assertTrue(audit_b["requires_board_review"])
        self.assertLess(audit_b["disparate_impact_ratio"], 0.80)


class TestEndToEnd(unittest.TestCase):

    def test_cli_demo_runs_end_to_end(self):
        """run_cli_demo.run_demo() must complete without exceptions."""
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        import run_cli_demo
        try:
            run_cli_demo.run_demo()
        except Exception as exc:
            self.fail(f"run_cli_demo.run_demo() raised: {exc}")

    def test_app_import_no_crash(self):
        """Importing app.py must not raise."""
        try:
            import importlib
            if "app" in sys.modules:
                importlib.reload(sys.modules["app"])
            else:
                import app  # noqa: F401
        except Exception as exc:
            self.fail(f"import app raised: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
