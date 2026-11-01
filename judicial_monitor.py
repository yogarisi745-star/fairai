"""
judicial_monitor.py - AI Judicial Bias & Trend Analyzer (Human-in-the-Loop Level 2)

Audits judges by analyzing historical decisions committed to the blockchain.
Detects whether a judge systematically disfavors or favors specific demographic groups
(e.g., disproportionate override rates, harsh sentencing for marginalized groups,
or leniency for high-income defendants), and compiles audit reports for the
Judicial Oversight Board.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats
from blockchain import JudicialBlockchain


class JudicialBiasMonitor:
    """
    Analyzes historical verdicts to identify judicial bias trends
    and reports findings to the Human Judicial Oversight Board.
    """

    def __init__(self, blockchain: JudicialBlockchain):
        self.blockchain = blockchain

    def extract_judge_records(self, judge_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves verified case records from blockchain, optionally filtered by judge_id."""
        records = []
        for block in self.blockchain.chain:
            if block.index == 0:
                continue  # skip genesis block
            data = block.case_data
            if judge_id is None or data.get("judge_id") == judge_id:
                records.append(data)
        return records

    def get_all_judges(self) -> List[str]:
        """Returns list of distinct judge IDs present in the ledger."""
        judges = set()
        for block in self.blockchain.chain:
            if block.index > 0:
                j_id = block.case_data.get("judge_id")
                if j_id:
                    judges.add(j_id)
        return sorted(list(judges))

    def analyze_judge_trends(self, judge_id: str) -> Dict[str, Any]:
        """
        Calculates demographic sentencing disparities, override behaviors,
        and statistical significance tests for a specific judge.
        """
        records = self.extract_judge_records(judge_id)
        total_cases = len(records)

        if total_cases == 0:
            return {
                "judge_id": judge_id,
                "total_cases": 0,
                "status": "INSUFFICIENT_DATA",
                "message": f"No historical cases recorded for Judge {judge_id}.",
            }

        # 1. Tally outcomes across Race / Caste
        marginalized_favorable = 0  # Bail or Probation
        marginalized_harsh = 0  # Incarceration
        general_favorable = 0
        general_harsh = 0

        # Override trends
        marginalized_overrides_to_harsh = 0
        general_overrides_to_lenient = 0
        total_overrides = 0
        total_acceptances = 0

        for r in records:
            verdict = r.get("final_verdict", "").lower()
            is_harsh = "custodial" in verdict or "incarceration" in verdict
            race_caste = r.get("demographics", {}).get("race_caste", "General")
            action = r.get("judge_action", "")
            orig_ai = r.get("ai_recommended_verdict", "").lower()

            if action == "OVERRIDDEN":
                total_overrides += 1
                if race_caste == "Marginalized" and is_harsh and "custodial" not in orig_ai:
                    marginalized_overrides_to_harsh += 1
                elif race_caste == "General" and not is_harsh and "custodial" in orig_ai:
                    general_overrides_to_lenient += 1
            else:
                total_acceptances += 1

            if race_caste == "Marginalized":
                if is_harsh:
                    marginalized_harsh += 1
                else:
                    marginalized_favorable += 1
            else:
                if is_harsh:
                    general_harsh += 1
                else:
                    general_favorable += 1

        total_marginalized = marginalized_favorable + marginalized_harsh
        total_general = general_favorable + general_harsh

        rate_marginalized_fav = (
            marginalized_favorable / total_marginalized if total_marginalized > 0 else 0.0
        )
        rate_general_fav = (
            general_favorable / total_general if total_general > 0 else 0.0
        )

        rate_marginalized_harsh = (
            marginalized_harsh / total_marginalized if total_marginalized > 0 else 0.0
        )
        rate_general_harsh = (
            general_harsh / total_general if total_general > 0 else 0.0
        )

        # 2. Disparate Impact Ratio (Four-Fifths Rule): Favorable rate protected / Favorable rate control
        if rate_general_fav > 0:
            disparate_impact_ratio = round(rate_marginalized_fav / rate_general_fav, 3)
        else:
            disparate_impact_ratio = 1.0

        # Demographic Parity Difference
        demographic_parity_diff = round(abs(rate_marginalized_fav - rate_general_fav), 3)

        # 3. Statistical Significance (Fisher's exact or Chi-Square Test)
        table = [
            [marginalized_favorable, marginalized_harsh],
            [general_favorable, general_harsh],
        ]

        # Use Fisher exact if any cell < 5, else Chi2
        min_cell = min(
            marginalized_favorable,
            marginalized_harsh,
            general_favorable,
            general_harsh,
        )
        p_value = 1.0
        if total_marginalized > 0 and total_general > 0:
            try:
                if min_cell < 5:
                    _, p_value = stats.fisher_exact(table)
                else:
                    res = stats.chi2_contingency(table)
                    p_value = res.pvalue
            except Exception:
                p_value = 1.0

        p_value = round(float(p_value), 4)

        # 4. Evaluation Criteria & Board Alert Status
        # Flag if Disparate Impact Ratio < 0.80 (standard 4/5ths threshold) AND difference > 15%
        is_disparate = (
            disparate_impact_ratio < 0.80 and demographic_parity_diff >= 0.15
        ) or (p_value < 0.05 and demographic_parity_diff >= 0.20)

        flagged = is_disparate and total_cases >= 4

        status = "FLAGGED_FOR_INVESTIGATION" if flagged else "EQUITABLE_COMPLIANT"

        # Construct actionable recommendation for Judicial Oversight Board
        findings: List[str] = []
        if flagged:
            findings.append(
                f"Statistically significant disparity detected: Favorable disposition rate for Marginalized defendants is {rate_marginalized_fav:.1%} vs {rate_general_fav:.1%} for General group."
            )
            findings.append(
                f"Disparate Impact Ratio = {disparate_impact_ratio:.2f} (Breaches Four-Fifths legal fairness benchmark < 0.80)."
            )
            if marginalized_overrides_to_harsh > 0:
                findings.append(
                    f"Judge overrode AI recommendation towards harsher custodial punishment in {marginalized_overrides_to_harsh} Marginalized accused cases."
                )
        else:
            findings.append(
                f"Sentencing distribution across demographic categories adheres to equity thresholds (DIR: {disparate_impact_ratio:.2f}, Parity Gap: {demographic_parity_diff:.1%})."
            )

        return {
            "judge_id": judge_id,
            "total_cases": total_cases,
            "total_overrides": total_overrides,
            "total_acceptances": total_acceptances,
            "override_rate": round(total_overrides / total_cases, 2) if total_cases > 0 else 0.0,
            "marginalized_stats": {
                "total": total_marginalized,
                "favorable": marginalized_favorable,
                "harsh": marginalized_harsh,
                "favorable_rate": round(rate_marginalized_fav, 3),
                "harsh_rate": round(rate_marginalized_harsh, 3),
                "overrides_to_harsh": marginalized_overrides_to_harsh,
            },
            "general_stats": {
                "total": total_general,
                "favorable": general_favorable,
                "harsh": general_harsh,
                "favorable_rate": round(rate_general_fav, 3),
                "harsh_rate": round(rate_general_harsh, 3),
                "overrides_to_lenient": general_overrides_to_lenient,
            },
            "disparate_impact_ratio": disparate_impact_ratio,
            "demographic_parity_gap": demographic_parity_diff,
            "p_value": p_value,
            "status": status,
            "findings": findings,
            "requires_board_review": flagged,
        }
