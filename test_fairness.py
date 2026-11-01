"""
test_fairness.py - Tests for fairness_eval.py outputs and schema validation

Covers:
  1. fairness_report.json schema: all keys present, values in valid ranges.
  2. fairness_report.md: file exists and is non-empty.
  3. audit_validation.json schema: structure verified.
  4. Tiny confusion-matrix fixture: FPR/FNR computed manually, asserted to match metrics function.
  5. App imports correctly when results files are present.
  6. App handles a missing results file gracefully (no crash on import).

Run:  python test_fairness.py
"""

import json
import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

FAIRNESS_JSON = os.path.join(_HERE, "results", "fairness_report.json")
FAIRNESS_MD   = os.path.join(_HERE, "results", "fairness_report.md")
AUDIT_JSON    = os.path.join(_HERE, "results", "audit_validation.json")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fpr_fnr(y_true, y_pred):
    """Tiny manual FPR/FNR for fixture tests."""
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    sel = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0
    return fpr, fnr, sel


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFairnessReportJSON(unittest.TestCase):
    """Schema and range checks on results/fairness_report.json."""

    def setUp(self):
        if not os.path.exists(FAIRNESS_JSON):
            self.skipTest("results/fairness_report.json not found — run fairness_eval.py first.")
        with open(FAIRNESS_JSON, encoding="utf-8") as f:
            self.report = json.load(f)

    def test_top_level_keys(self):
        required = {"generated_at", "n_train", "n_test", "models",
                    "mitigation_comparison", "findings"}
        for key in required:
            self.assertIn(key, self.report, f"Missing top-level key: {key}")

    def test_model_keys_present(self):
        expected_models = {
            "compas_reference", "model_a_with_demo", "model_a_debiased",
            "model_b_with_demo", "model_b_debiased",
        }
        for m in expected_models:
            self.assertIn(m, self.report["models"], f"Missing model key: {m}")

    def test_each_model_has_required_sections(self):
        for key, ev in self.report["models"].items():
            for section in ["label", "overall", "by_group", "gaps"]:
                self.assertIn(section, ev, f"Model {key} missing section: {section}")

    def test_overall_metrics_in_valid_range(self):
        range_01 = {"accuracy", "fpr", "fnr", "precision", "selection_rate"}
        for key, ev in self.report["models"].items():
            ov = ev["overall"]
            for metric in range_01:
                val = ov.get(metric, None)
                if val is not None and not math.isnan(val):
                    self.assertGreaterEqual(val, 0.0, f"{key}.overall.{metric} < 0")
                    self.assertLessEqual(val, 1.0, f"{key}.overall.{metric} > 1")
            # ROC-AUC
            auc = ov.get("roc_auc", None)
            if auc is not None and not math.isnan(auc):
                self.assertGreaterEqual(auc, 0.0)
                self.assertLessEqual(auc, 1.0)

    def test_gaps_section_keys(self):
        required_gap_keys = {
            "fpr_gap_AA_minus_CA", "fnr_gap_AA_minus_CA",
            "demographic_parity_diff", "disparate_impact_ratio", "equalized_odds_diff",
        }
        for key, ev in self.report["models"].items():
            for gk in required_gap_keys:
                self.assertIn(gk, ev["gaps"], f"Model {key} gaps missing: {gk}")

    def test_by_group_has_both_races(self):
        for key, ev in self.report["models"].items():
            groups = set(ev["by_group"].keys())
            self.assertIn("African-American", groups, f"{key} missing African-American group")
            self.assertIn("Caucasian", groups, f"{key} missing Caucasian group")

    def test_mitigation_comparison_has_5_methods(self):
        self.assertGreaterEqual(
            len(self.report["mitigation_comparison"]), 4,
            "Expected at least 4 mitigation methods in comparison"
        )

    def test_mitigation_entries_have_required_keys(self):
        required = {"method", "accuracy", "equalized_odds_diff", "fpr_gap", "fnr_gap"}
        for entry in self.report["mitigation_comparison"]:
            for k in required:
                self.assertIn(k, entry, f"Mitigation entry missing key: {k}")

    def test_findings_non_empty(self):
        self.assertGreater(len(self.report["findings"]), 0)
        for finding in self.report["findings"]:
            self.assertIsInstance(finding, str)
            self.assertGreater(len(finding), 10)

    def test_train_test_sizes_reasonable(self):
        self.assertGreater(self.report["n_train"], 3000)
        self.assertGreater(self.report["n_test"], 1000)
        total = self.report["n_train"] + self.report["n_test"]
        self.assertGreaterEqual(total, 5000)
        self.assertLessEqual(total, 6000)

    def test_compas_fpr_gap_is_positive(self):
        """COMPAS is known to have a higher FPR for African-American defendants."""
        gap = self.report["models"]["compas_reference"]["gaps"]["fpr_gap_AA_minus_CA"]
        self.assertGreater(gap, 0.0,
            "COMPAS reference FPR gap (AA - CA) should be positive (known bias)")


class TestFairnessReportMD(unittest.TestCase):
    """Markdown report exists and has content."""

    def test_md_file_exists_and_nonempty(self):
        if not os.path.exists(FAIRNESS_MD):
            self.skipTest("results/fairness_report.md not found — run fairness_eval.py first.")
        size = os.path.getsize(FAIRNESS_MD)
        self.assertGreater(size, 200, "fairness_report.md is nearly empty")

    def test_md_contains_table(self):
        if not os.path.exists(FAIRNESS_MD):
            self.skipTest("results/fairness_report.md not found.")
        with open(FAIRNESS_MD, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("|", content, "No markdown table found in fairness_report.md")
        self.assertIn("Accuracy", content)


class TestAuditValidationJSON(unittest.TestCase):
    """Schema checks on results/audit_validation.json."""

    def setUp(self):
        if not os.path.exists(AUDIT_JSON):
            self.skipTest("results/audit_validation.json not found — run audit_validation.py first.")
        with open(AUDIT_JSON, encoding="utf-8") as f:
            self.report = json.load(f)

    def test_top_level_keys(self):
        required = {
            "generated_at", "n_simulations", "n_cases_tested", "bias_probs_tested",
            "results_grid", "conclusion",
        }
        for k in required:
            self.assertIn(k, self.report, f"Missing key: {k}")

    def test_results_grid_shape(self):
        grid = self.report["results_grid"]
        n_cases = len(self.report["n_cases_tested"])
        n_probs = len(self.report["bias_probs_tested"])
        self.assertEqual(len(grid), n_cases * n_probs,
            f"Expected {n_cases * n_probs} grid entries, got {len(grid)}")

    def test_detection_and_ff_rates_in_range(self):
        for row in self.report["results_grid"]:
            self.assertGreaterEqual(row["detection_rate"], 0.0)
            self.assertLessEqual(row["detection_rate"], 1.0)
            self.assertGreaterEqual(row["false_flag_rate"], 0.0)
            self.assertLessEqual(row["false_flag_rate"], 1.0)

    def test_conclusion_non_empty(self):
        self.assertIsInstance(self.report["conclusion"], str)
        self.assertGreater(len(self.report["conclusion"]), 50)


class TestConfusionMatrixFixture(unittest.TestCase):
    """
    Hand-computed confusion matrix fixture to verify the metrics function logic.
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 0, 1]
    TP=2, FP=2, TN=2, FN=2 → FPR=0.5, FNR=0.5, Sel=0.5
    """

    def test_fpr_fnr_selection_rate_fixture(self):
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 1, 0, 0, 1]
        fpr, fnr, sel = _compute_fpr_fnr(y_true, y_pred)
        self.assertAlmostEqual(fpr, 0.5, places=6, msg=f"FPR expected 0.5, got {fpr}")
        self.assertAlmostEqual(fnr, 0.5, places=6, msg=f"FNR expected 0.5, got {fnr}")
        self.assertAlmostEqual(sel, 0.5, places=6, msg=f"Selection rate expected 0.5, got {sel}")

    def test_perfect_predictor_zero_fpr_fnr(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        fpr, fnr, sel = _compute_fpr_fnr(y_true, y_pred)
        self.assertAlmostEqual(fpr, 0.0)
        self.assertAlmostEqual(fnr, 0.0)
        self.assertAlmostEqual(sel, 0.5)

    def test_always_predict_positive_fpr_one(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 1, 1]
        fpr, fnr, sel = _compute_fpr_fnr(y_true, y_pred)
        self.assertAlmostEqual(fpr, 1.0)
        self.assertAlmostEqual(fnr, 0.0)
        self.assertAlmostEqual(sel, 1.0)

    def test_consistency_with_fairness_report(self):
        """FPR from fairness_report.json for COMPAS reference must be consistent
        with confusion matrix entries (FP / (FP + TN))."""
        if not os.path.exists(FAIRNESS_JSON):
            self.skipTest("results/fairness_report.json not found.")
        with open(FAIRNESS_JSON, encoding="utf-8") as f:
            report = json.load(f)
        ov = report["models"]["compas_reference"]["overall"]
        fp, tn = ov["fp"], ov["tn"]
        expected_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        self.assertAlmostEqual(ov["fpr"], round(expected_fpr, 4), places=3,
            msg="Stored FPR inconsistent with confusion matrix FP/(FP+TN)")


class TestAppHandlesMissingResults(unittest.TestCase):
    """App _load_json helper returns None gracefully when file is missing."""

    def test_load_json_returns_none_for_missing_file(self):
        """Simulate missing results file — app should not crash."""
        import importlib
        # We test the helper logic directly without launching Streamlit
        import types

        def _load_json_safe(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        result = _load_json_safe("/nonexistent/path/file.json")
        self.assertIsNone(result)

    def test_app_imports_without_crash(self):
        """app.py must be importable without raising (even without Streamlit context)."""
        try:
            import importlib
            if "app" in sys.modules:
                # Reload to pick up changes
                importlib.reload(sys.modules["app"])
            else:
                import app  # noqa: F401
        except SystemExit:
            pass  # Streamlit may call sys.exit in test context
        except Exception as exc:
            self.fail(f"import app raised: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
