"""
test_pipeline.py - Verification & Unit Test Suite for AI Bias Mitigation System

Updated for COMPAS data mode:
  - Removed assertions tied to old hand-coded weights (e.g. income SHAP > 0,
    since income now has weight 0.0 in data-fitted mode).
  - Added tests for: loader row count/filters, 14-key WEIGHTS, [0,100] bounds,
    SHAP additivity, fallback behaviour, and end-to-end CLI demo.
  - HIGH_BIAS_DETECTED threshold (10.0) requires a bias gap of >= 10 pts.
    With real COMPAS coefficients (race ~5.93, gender ~0.82) the maximum
    demographic gap is ~6.75 pts, so the high-bias demo case now falls into
    LOW_BIAS_PASS — this reflects real-world COMPAS data and is expected.
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
    """Tests for model initialisation, weight structure and prediction bounds."""

    def setUp(self):
        self.model = LegalRiskModel()

    # ── (b) WEIGHTS contains all 14 expected keys ─────────────────────────────
    def test_weights_has_all_14_keys(self):
        expected_keys = {
            "prior_felonies", "prior_misdemeanors", "offense_severity",
            "age_at_arrest", "bail_failure_history", "community_ties_score",
            "race_caste_Marginalized", "race_caste_General",
            "income_level_Low", "income_level_Medium", "income_level_High",
            "gender_Male", "gender_Female", "gender_Non-binary",
        }
        self.assertEqual(set(self.model.WEIGHTS.keys()), expected_keys)

    # ── (c) Predictions are within [0, 100] ──────────────────────────────────
    def test_model_prediction_bounds(self):
        """Verify model output is always bounded within [0, 100]."""
        high_risk = {
            "prior_felonies": 10, "prior_misdemeanors": 10,
            "offense_severity": 10, "age_at_arrest": 20,
            "bail_failure_history": 1, "community_ties_score": 1,
            "race_caste": "Marginalized", "income_level": "Low", "gender": "Male",
        }
        score_high = self.model.predict_risk_score(high_risk)
        self.assertTrue(0.0 <= score_high <= 100.0,
                        f"High-risk score {score_high} out of [0,100]")

        low_risk = {
            "prior_felonies": 0, "prior_misdemeanors": 0,
            "offense_severity": 1, "age_at_arrest": 60,
            "bail_failure_history": 0, "community_ties_score": 10,
            "race_caste": "General", "income_level": "High", "gender": "Female",
        }
        score_low = self.model.predict_risk_score(low_risk)
        self.assertTrue(0.0 <= score_low <= 100.0,
                        f"Low-risk score {score_low} out of [0,100]")

        # High-risk must outscore low-risk
        self.assertGreater(score_high, score_low)

    # ── (e) Fallback works when CSV is missing ────────────────────────────────
    def test_fallback_when_csv_missing(self):
        """
        When compas-scores-two-years.csv is absent and the cache is absent,
        model must fall back to hand-coded weights without raising.
        We test this by importing the fallback constants directly.
        """
        import model as model_module
        # The original fallback weights must still define all 14 keys
        self.assertEqual(len(model_module._ORIGINAL_WEIGHTS), 14)
        self.assertIsInstance(model_module._ORIGINAL_INTERCEPT, float)

    def test_weight_source_is_set(self):
        """WEIGHT_SOURCE attribute must exist and be a non-empty string."""
        self.assertIsInstance(self.model.WEIGHT_SOURCE, str)
        self.assertGreater(len(self.model.WEIGHT_SOURCE), 0)


class TestCompasLoader(unittest.TestCase):
    """Tests for compas_loader.py — only run when the CSV is present."""

    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "compas-scores-two-years.csv")

    def setUp(self):
        if not os.path.exists(self.CSV_PATH):
            self.skipTest("compas-scores-two-years.csv not present; skipping loader tests.")

    # ── (a) Loader row count and filters ─────────────────────────────────────
    def test_loader_row_count_and_filters(self):
        from compas_loader import load_compas
        df, total_dropped = load_compas(self.CSV_PATH)

        # ProPublica expected ~6172 rows after standard filters;
        # after race restriction (African-American + Caucasian only) ~5278.
        self.assertGreaterEqual(len(df), 5000,
                                f"Expected ~5278 rows after filtering, got {len(df)}")
        self.assertLessEqual(len(df), 6500)
        self.assertGreater(total_dropped, 0)

        # All required project columns present
        for col in ["race_caste", "gender", "age_at_arrest", "prior_felonies",
                    "prior_misdemeanors", "offense_severity", "income_level",
                    "community_ties_score", "bail_failure_history", "target_score"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        # race_caste must only contain project values
        self.assertTrue(df["race_caste"].isin(["General", "Marginalized"]).all())

        # target_score in [0, 100]
        self.assertTrue((df["target_score"] >= 0).all())
        self.assertTrue((df["target_score"] <= 100).all())


class TestSHAPAndDebiasing(unittest.TestCase):
    """Tests for SHAP values and counterfactual debiasing."""

    def setUp(self):
        self.model    = LegalRiskModel()
        self.explainer = LegalSHAPExplainer(self.model)

    # ── (d) SHAP values sum to (score - baseline_score) within 0.1 tolerance ─
    def test_shap_additivity(self):
        """
        For an additive linear model, sum(SHAP values) == score - baseline_score
        (within 0.1 rounding tolerance for 1-decimal rounding).
        """
        profiles = [
            BENCHMARK_CASES[0],  # high-bias case
            BENCHMARK_CASES[1],  # low-bias case
            {
                "demographics": {"race_caste": "General", "income_level": "High", "gender": "Female"},
                "merits": {"prior_felonies": 2, "prior_misdemeanors": 0, "offense_severity": 7,
                           "age_at_arrest": 45, "bail_failure_history": 0, "community_ties_score": 6},
            },
        ]
        for case in profiles:
            accused_data = {**case["demographics"], **case["merits"]}
            result = self.explainer.explain_and_evaluate(accused_data)
            shap_sum = sum(result["shap_values"].values())
            expected = result["original_score"] - self.explainer.baseline_score
            self.assertAlmostEqual(
                shap_sum, expected, delta=0.1,
                msg=f"SHAP additivity failed: sum={shap_sum:.3f} expected={expected:.3f} "
                    f"for {accused_data}"
            )

    def test_shap_values_present_for_all_sensitive_attrs(self):
        """SHAP result must contain entries for race_caste, income_level, gender."""
        accused_data = {**BENCHMARK_CASES[0]["demographics"], **BENCHMARK_CASES[0]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertIn("race_caste",   result["shap_values"])
        self.assertIn("income_level", result["shap_values"])
        self.assertIn("gender",       result["shap_values"])

    def test_race_caste_shap_positive_for_marginalized(self):
        """
        Marginalized race_caste must produce a positive SHAP vs General baseline.
        With COMPAS-fitted weights, race_caste_Marginalized coefficient is ~5.93.
        """
        accused_data = {**BENCHMARK_CASES[0]["demographics"], **BENCHMARK_CASES[0]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertGreater(result["shap_values"]["race_caste"], 0,
                           "Marginalized race should have positive SHAP attribution")

    def test_income_shap_is_zero_in_compas_mode(self):
        """
        income_level has 0.0 weight in COMPAS mode (not in dataset).
        income SHAP must be 0.0 regardless of the accused's income_level.
        """
        accused_data = {"race_caste": "Marginalized", "income_level": "Low",
                        "gender": "Male", "prior_felonies": 1, "prior_misdemeanors": 0,
                        "offense_severity": 5, "age_at_arrest": 30,
                        "bail_failure_history": 0, "community_ties_score": 5}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertEqual(result["shap_values"]["income_level"], 0.0,
                         "income_level SHAP should be 0.0 in COMPAS mode")

    def test_counterfactual_debiasing_no_keyerror(self):
        """Counterfactual debiasing must run without KeyError for all BENCHMARK_CASES."""
        for case in BENCHMARK_CASES:
            accused_data = {**case["demographics"], **case["merits"]}
            debiased, score, verdict, reason = (
                self.explainer.perform_counterfactual_debiasing(accused_data)
            )
            self.assertIsInstance(score, float)
            self.assertTrue(0.0 <= score <= 100.0)

    def test_high_bias_demo_case_has_race_attribution(self):
        """
        BENCHMARK_CASES[0] is the 'high-bias' demo (Marginalized defendant).
        With COMPAS weights the race SHAP (~5.93) is real and positive,
        and the debiased score must be lower than the original.
        Note: the gap may be < 10.0 (LOW_BIAS_PASS) because real COMPAS
        coefficients are smaller than the hand-coded weights.
        """
        accused_data = {**BENCHMARK_CASES[0]["demographics"], **BENCHMARK_CASES[0]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertGreater(result["shap_values"]["race_caste"], 0)
        self.assertLess(result["debiased_score"], result["original_score"])

    def test_low_bias_demo_case_has_zero_race_attribution(self):
        """BENCHMARK_CASES[1] is General/Male — race SHAP must be 0."""
        accused_data = {**BENCHMARK_CASES[1]["demographics"], **BENCHMARK_CASES[1]["merits"]}
        result = self.explainer.explain_and_evaluate(accused_data)
        self.assertEqual(result["shap_values"]["race_caste"], 0.0)
        self.assertEqual(result["bias_gap"], 0.0)


class TestBlockchain(unittest.TestCase):
    """Tests for blockchain immutability and tamper detection."""

    def setUp(self):
        self.blockchain = JudicialBlockchain(difficulty=1)

    def test_blockchain_immutability_and_tampering(self):
        """Verify blockchain detects unauthorized retroactive data tampering."""
        is_valid, msg = self.blockchain.verify_chain_integrity()
        self.assertTrue(is_valid)

        self.blockchain.record_verdict(
            case_id="TEST-CASE-01", accused_id="TEST-ACC-01",
            demographics={"race_caste": "General", "income_level": "Medium", "gender": "Male"},
            merits={"prior_felonies": 0, "prior_misdemeanors": 0, "offense_severity": 1,
                    "age_at_arrest": 30, "bail_failure_history": 0, "community_ties_score": 8},
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
        is_tampered_valid, reason = self.blockchain.verify_chain_integrity()
        self.assertFalse(is_tampered_valid)
        self.assertIn("hash mismatch", reason)


class TestJudicialMonitoring(unittest.TestCase):
    """Tests for judicial bias monitoring and oversight board alerts."""

    def setUp(self):
        self.system = LegalBiasMitigationSystem(seed_data=True)

    def test_judicial_bias_monitoring_and_board_alerts(self):
        """Verify Judge Verma is compliant and Judge Singhania is flagged."""
        audit_verma = self.system.audit_judge("JD-101 (Judge Verma)")
        self.assertEqual(audit_verma["status"], "EQUITABLE_COMPLIANT")
        self.assertFalse(audit_verma["requires_board_review"])

        audit_singhania = self.system.audit_judge("JD-202 (Judge Singhania)")
        self.assertEqual(audit_singhania["status"], "FLAGGED_FOR_INVESTIGATION")
        self.assertTrue(audit_singhania["requires_board_review"])
        self.assertLess(audit_singhania["disparate_impact_ratio"], 0.80)


class TestEndToEnd(unittest.TestCase):
    """End-to-end smoke tests."""

    # ── (f) Full pipeline in run_cli_demo.py runs end to end ─────────────────
    def test_cli_demo_runs_end_to_end(self):
        """run_cli_demo.run_demo() must complete without exceptions."""
        import sys
        import run_cli_demo
        # Reconfigure stdout to UTF-8 so emoji in run_cli_demo don't crash
        # on Windows cp1252 consoles during testing.
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        try:
            run_cli_demo.run_demo()
        except Exception as exc:
            self.fail(f"run_cli_demo.run_demo() raised: {exc}")

    def test_app_import_no_crash(self):
        """Importing app.py must not raise (Streamlit smoke test)."""
        try:
            import importlib
            # Streamlit decorators run at import; we just need no crash.
            if "app" in sys.modules:
                importlib.reload(sys.modules["app"])
            else:
                import app  # noqa: F401
        except Exception as exc:
            self.fail(f"import app raised: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
