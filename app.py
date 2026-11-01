"""
app.py - Interactive Web Prototype: AI Legal Bias Mitigation & Blockchain Accountability

Institutional Dashboard:
- View 1: Accused Intake & AI Risk Assessment with SHAP (XAI) Attribution & Debiasing.
- View 2: Judicial Bench (HITL Level-1): Judge review, fast-track / override with reasoning, and blockchain commitment.
- View 3: Cryptographic Blockchain Ledger Explorer & Tamper Integrity Verifier.
- View 4: Judicial Oversight Board (HITL Level-2): AI trend analysis, demographic disparity, and judicial bias flags.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from legal_system import LegalBiasMitigationSystem
from demo_cases import BENCHMARK_CASES

# Page Configuration
st.set_page_config(
    page_title="AI Legal Bias Mitigation & Judicial Oversight Ledger",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Institutional Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 16px;
        border-left: 4px solid #2563eb;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .bias-alert {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #d97706;
        color: #92400e;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        line-height: 1.5;
    }
    .fast-track-pass {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        color: #166534;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        line-height: 1.5;
    }
    .board-flag {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        color: #991b1b;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        line-height: 1.5;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "legal_system" not in st.session_state:
    st.session_state.legal_system = LegalBiasMitigationSystem(seed_data=True)

if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = None

system = st.session_state.legal_system

# Sidebar Navigation
st.sidebar.title("Judicial AI Oversight")
st.sidebar.caption("Dual-Level Human-in-the-Loop & Blockchain Architecture")

navigation = st.sidebar.radio(
    "Navigation Module",
    [
        "1. Accused Assessment & SHAP Debiasing",
        "2. Judicial Bench Review (HITL-1)",
        "3. Cryptographic Ledger & Audit Guard",
        "4. Judicial Oversight Board (HITL-2)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Architecture")
st.sidebar.markdown(
    """
    - **Stage 1:** Baseline Risk Estimation
    - **Stage 2:** SHAP Factor Attribution & Debiasing
    - **Stage 3:** Bias Gap Analysis & Judicial Adjudication (HITL-1)
    - **Stage 4:** SHA-256 Cryptographic Ledger
    - **Stage 5:** Aggregate Disparity Analysis (Four-Fifths Rule)
    - **Stage 6:** Judicial Oversight Board Review (HITL-2)
    """
)

# Show model weight source in sidebar
from model import LegalRiskModel as _LRM
_weight_src = _LRM.WEIGHT_SOURCE
if "COMPAS" in _weight_src:
    st.sidebar.info(f"Model Configuration: **{_weight_src}**")
else:
    st.sidebar.warning(f"Model Configuration: **{_weight_src}** (CSV not found)")

# ==============================================================================
# TAB 1: ACCUSED ASSESSMENT & SHAP DEBIASING
# ==============================================================================
if navigation == "1. Accused Assessment & SHAP Debiasing":
    st.title("Accused Assessment & Algorithmic Debiasing")
    st.write(
        "Evaluate an accused individual's recidivism risk score, isolate protected demographic bias using "
        "exact Shapley values, and generate counterfactual debiased recommendations."
    )

    col_bench, col_manual = st.columns([1, 2])

    with col_bench:
        st.subheader("Select Benchmark Case")
        case_options = {f"{c['case_id']} - {c.get('case_label', c['accused_name'])}": c for c in BENCHMARK_CASES}
        selected_case_name = st.selectbox("Pre-configured Cases", list(case_options.keys()))
        selected_case = case_options[selected_case_name]

        use_selected = st.button("Load Selected Profile", use_container_width=True)

    with col_manual:
        st.subheader("Accused Profile Details")

        if use_selected or "case_input_data" not in st.session_state:
            st.session_state.case_input_data = selected_case

        active_data = st.session_state.case_input_data

        c1, c2, c3 = st.columns(3)
        with c1:
            case_id = st.text_input("Case ID", value=active_data.get("case_id", "CRIM-2026-CUSTOM"))
            accused_id = st.text_input("Accused ID", value=active_data.get("accused_id", "ACC-CUSTOM"))
            accused_name = st.text_input("Accused Name", value=active_data.get("accused_name", "Defendant"))
        with c2:
            st.markdown("**Protected Demographics (Bias Audit)**")
            race_caste = st.selectbox(
                "Race Group (Marginalized = African-American, General = Caucasian)",
                ["Marginalized", "General"],
                index=0 if active_data["demographics"].get("race_caste") == "Marginalized" else 1,
            )
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Non-binary"],
                index=["Male", "Female", "Non-binary"].index(active_data["demographics"].get("gender", "Male")),
            )
        with c3:
            st.markdown("**Objective Criminal Merits**")
            prior_felonies = st.number_input(
                "Prior Offenses (total)", min_value=0, max_value=20,
                value=int(active_data["merits"].get("prior_felonies", 0))
            )
            offense_severity = st.slider(
                "Offense Severity (1-10)", min_value=1, max_value=10,
                value=int(active_data["merits"].get("offense_severity", 5)),
            )
            age_at_arrest = st.number_input(
                "Age at Arrest", min_value=18, max_value=90,
                value=int(active_data["merits"].get("age_at_arrest", 28))
            )

        charge_desc = st.text_area(
            "Charge Description",
            value=active_data.get("description", "Non-violent allegation."),
            height=68,
        )

        run_assessment = st.button("Run Algorithmic Scoring & SHAP Debiasing", type="primary", use_container_width=True)

    if run_assessment:
        case_payload = {
            "case_id": case_id,
            "accused_id": accused_id,
            "accused_name": accused_name,
            "description": charge_desc,
            "demographics": {
                "race_caste": race_caste,
                "gender": gender,
            },
            "merits": {
                "prior_felonies":   prior_felonies,
                "offense_severity": offense_severity,
                "age_at_arrest":    age_at_arrest,
            },
        }
        assessment = system.assess_case(case_payload)
        st.session_state.current_assessment = assessment

    # Display Results if Available
    if st.session_state.current_assessment:
        res = st.session_state.current_assessment
        st.markdown("---")
        st.subheader(f"Assessment Output: {res['case_id']} ({res['accused_name']})")

        # Metric Overview Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Raw AI Risk Score", f"{res['ai_original_score']} / 100")
        with m2:
            delta_val = -res["bias_gap"] if res["bias_gap"] > 0 else 0
            st.metric("Debiased Risk Score", f"{res['ai_debiased_score']} / 100", delta=f"{delta_val} pts")
        with m3:
            st.metric("Bias Disparity Gap", f"{res['bias_gap']} pts")
        with m4:
            st.metric("Triage Classification", res["bias_flag"])

        # Bias Alert Banner
        if res["is_high_bias"]:
            st.markdown(
                f"""
                <div class="bias-alert">
                    <span class="status-badge badge-critical">Scrutiny Required</span><br>
                    <strong>High Demographic Disparity Detected (Bias Gap: +{res['bias_gap']} points)</strong><br>
                    {res['triage_guidance']}<br>
                    <strong>Mandatory Procedure:</strong> Case flagged for substantive human review. Fast-tracking prohibited.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="fast-track-pass">
                    <span class="status-badge badge-success">Clearance</span><br>
                    <strong>Low Demographic Disparity (Bias Gap: {res['bias_gap']} points)</strong><br>
                    {res['triage_guidance']}<br>
                    <strong>Standard Procedure:</strong> Eligible for standard expedited judicial review.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Disposition Comparison
        col_disp1, col_disp2 = st.columns(2)
        with col_disp1:
            st.markdown("**Initial Baseline AI Recommendation:**")
            st.error(f"**Verdict:** {res['ai_original_verdict']}")
            st.caption(f"Rationale: {res['ai_original_reasoning']}")
        with col_disp2:
            st.markdown("**Debiased Counterfactual AI Recommendation:**")
            st.success(f"**Verdict:** {res['ai_debiased_verdict']}")
            st.caption(f"Rationale: {res['ai_debiased_reasoning']}")

        # SHAP Feature Attribution Chart
        st.subheader("Explainable AI (SHAP) Factor Attribution")
        st.write(
            "Points added (+) or subtracted (-) relative to population baseline. "
            "Sensitive demographic factors are categorized separately from verified criminal history merits."
        )

        shap_df = pd.DataFrame(
            [
                {
                    "Feature": k,
                    "Attribution (Points)": v,
                    "Category": (
                        "Protected Demographic (Bias)"
                        if k in ["race_caste", "gender"]
                        else "Legitimate Criminal Merit"
                    ),
                }
                for k, v in res["shap_values"].items()
            ]
        ).sort_values(by="Attribution (Points)", key=abs, ascending=True)

        st.bar_chart(
            shap_df,
            x="Attribution (Points)",
            y="Feature",
            color="Category",
            horizontal=True,
        )

        st.info("Proceed to the **'2. Judicial Bench Review (HITL-1)'** tab in the sidebar to adjudicate this case.")


# ==============================================================================
# TAB 2: JUDICIAL BENCH (HUMAN-IN-THE-LOOP LEVEL 1)
# ==============================================================================
elif navigation == "2. Judicial Bench Review (HITL-1)":
    st.title("Judicial Bench: Human-in-the-Loop Adjudication (HITL-1)")
    st.write(
        "Presiding judicial officers review case files, inspect algorithmic disparity decompositions, "
        "and exercise constitutional authority to accept or override recommendations."
    )

    if not st.session_state.current_assessment:
        st.warning("No active case loaded. Please run assessment in Tab 1 or load a sample case below.")
        if st.button("Load Sample High-Bias Case (Real COMPAS Record)"):
            st.session_state.current_assessment = system.assess_case(BENCHMARK_CASES[0])
            st.rerun()
    else:
        case = st.session_state.current_assessment

        st.subheader(f"Case Dossier: {case['case_id']} - {case['accused_name']}")

        c_info1, c_info2 = st.columns(2)
        with c_info1:
            st.markdown("**Defendant & Case Details:**")
            st.write(f"- **Allegation:** {case['description']}")
            st.write(f"- **Demographics:** {case['demographics']}")
            st.write(f"- **Criminal Record:** {case['merits']}")

        with c_info2:
            st.markdown("**Algorithmic & Debiasing Audit:**")
            st.write(f"- **Raw Baseline Risk Score:** {case['ai_original_score']} / 100")
            st.write(f"- **Debiased Risk Score:** {case['ai_debiased_score']} / 100")
            st.write(f"- **Demographic Disparity Gap:** {case['bias_gap']} pts")
            st.write(f"- **Primary Disparity Factor:** `{case['top_contributing_factor']}` ({case['top_factor_impact']} pts)")

        st.markdown("---")
        st.subheader("Official Judicial Disposition Entry")

        judge_id = st.selectbox(
            "Presiding Judicial Officer",
            ["JD-101 (Judge A)", "JD-202 (Judge B)", "JD-303 (Judge C)"],
        )

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            decision_type = st.radio(
                "Judicial Action",
                ["Accept Debiased AI Recommendation", "Override AI Recommendation"],
                index=0,
            )
            judge_action = "ACCEPTED" if "Accept" in decision_type else "OVERRIDDEN"

        with col_act2:
            verdict_choices = [
                "Bail Granted (Unconditional / Personal Recognizance)",
                "Supervised Probation / Conditional Bail with Monitoring",
                "Custodial Remand / Incarceration Sentencing",
                "Case Dismissed / Deferred Prosecution",
            ]
            default_index = 0
            for idx, choice in enumerate(verdict_choices):
                if choice.startswith(case["ai_debiased_verdict"][:10]):
                    default_index = idx
                    break

            final_verdict = st.selectbox(
                "Final Judicial Order / Disposition",
                verdict_choices,
                index=default_index if judge_action == "ACCEPTED" else 0,
            )

        default_reasoning = (
            f"Judicial review of case {case['case_id']} confirms that the raw baseline calculation "
            f"incorporated protected demographic disparity ({case['top_contributing_factor']}). "
            f"Upon counterfactual debiasing and evaluating defendant's verified merits, "
            f"the court orders {final_verdict}."
            if case["is_high_bias"]
            else f"Independent judicial assessment affirms legitimate criminal history warrants {final_verdict}."
        )

        judicial_reasoning = st.text_area(
            "Written Legal Rationale (Mandatory for Due Process & Audit Accountability)",
            value=default_reasoning,
            height=120,
        )

        commit_btn = st.button("Commit Final Verdict to Cryptographic Ledger", type="primary", use_container_width=True)

        if commit_btn:
            try:
                commit_res = system.judge_adjudicate_and_commit(
                    case_assessment=case,
                    judge_id=judge_id,
                    judge_action=judge_action,
                    final_verdict=final_verdict,
                    judicial_reasoning=judicial_reasoning,
                )
                st.success(
                    f"Case {case['case_id']} successfully committed to Cryptographic Ledger in Block #{commit_res['block_index']}."
                )
                st.markdown(
                    f"""
                    - **Block Index:** `#{commit_res['block_index']}`
                    - **SHA-256 Block Hash:** `{commit_res['block_hash']}`
                    - **Previous Hash Link:** `{commit_res['previous_hash']}`
                    - **Judicial Action:** `{commit_res['judge_action']}` -> `{commit_res['final_verdict']}`
                    """
                )
            except Exception as e:
                st.error(f"Error committing to blockchain: {e}")


# ==============================================================================
# TAB 3: CRYPTOGRAPHIC LEDGER & AUDIT GUARD
# ==============================================================================
elif navigation == "3. Cryptographic Ledger & Audit Guard":
    st.title("Cryptographic Ledger & Tamper Guard")
    st.write(
        "Every judicial decision, algorithmic prediction, SHAP attribution summary, and legal rationale is "
        "cryptographically committed into immutable SHA-256 blocks to prevent retrospective tampering."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        verify_chain = st.button("Verify Cryptographic Ledger Integrity", use_container_width=True)
    with col_btn2:
        tamper_demo = st.button("Simulate Unauthorized Database Modification (Block #1)", use_container_width=True)

    if verify_chain:
        is_valid, msg = system.blockchain.verify_chain_integrity()
        if is_valid:
            st.success(f"Ledger Integrity Verified: {msg}")
        else:
            st.error(f"CRYPTOGRAPHIC AUDIT ALERT: {msg}")

    if tamper_demo:
        system.blockchain.simulate_tampering(block_index=1, new_verdict="Tampered: Complete Dismissal")
        st.warning("Simulated unauthorized edit on Block #1: verdict modified without valid cryptographic proof-of-work.")
        is_valid, msg = system.blockchain.verify_chain_integrity()
        st.error(f"CRITICAL AUDIT ALERT: {msg}")

    st.subheader(f"Committed Blocks ({len(system.blockchain.chain)} Blocks Total)")

    blocks_data = []
    for blk in reversed(system.blockchain.chain):
        cdata = blk.case_data
        blocks_data.append(
            {
                "Block #": blk.index,
                "Case ID": cdata.get("case_id", "N/A"),
                "Judicial Officer": cdata.get("judge_id", "System Genesis"),
                "Raw Score": cdata.get("ai_original_score", "N/A"),
                "Debiased Score": cdata.get("ai_debiased_score", "N/A"),
                "Action": cdata.get("judge_action", "N/A"),
                "Final Verdict": cdata.get("final_verdict", cdata.get("message", "N/A")),
                "Block Hash": blk.hash[:16] + "...",
                "Previous Hash": blk.previous_hash[:16] + "...",
            }
        )

    st.dataframe(pd.DataFrame(blocks_data), use_container_width=True)

    with st.expander("Inspect Raw Cryptographic Block Payloads"):
        selected_blk_idx = st.selectbox(
            "Select Block Index",
            range(len(system.blockchain.chain)),
            index=len(system.blockchain.chain) - 1,
        )
        st.json(system.blockchain.chain[selected_blk_idx].to_dict())


# ==============================================================================
# TAB 4: JUDICIAL OVERSIGHT BOARD (HUMAN-IN-THE-LOOP LEVEL 2)
# ==============================================================================
elif navigation == "4. Judicial Oversight Board (HITL-2)":
    st.title("Judicial Oversight Board: Disparity Audit (HITL-2)")
    st.write(
        "Level-2 statistical monitoring audits presiding judicial officers across the immutable ledger. "
        "Applies the EEOC Four-Fifths compliance standard and Fisher's exact test to detect systematic disparities."
    )

    judges_list = system.list_all_judges()
    if not judges_list:
        st.info("No judicial cases recorded yet.")
    else:
        selected_judge = st.selectbox("Select Judicial Officer to Audit", judges_list)
        audit = system.audit_judge(selected_judge)

        st.subheader(f"Equity Compliance Dossier: {audit['judge_id']}")

        # Status Banner
        if audit["status"] == "FLAGGED_FOR_INVESTIGATION":
            st.markdown(
                f"""
                <div class="board-flag">
                    <span class="status-badge badge-critical">Investigation Recommended</span><br>
                    <strong>Discretionary Sentencing Disparity Detected</strong><br>
                    Sentencing audit detected statistically significant demographic variance in override patterns.<br>
                    <strong>Disparate Impact Ratio:</strong> {audit['disparate_impact_ratio']} (EEOC Benchmark Standard ≥ 0.80) |
                    <strong>Demographic Parity Gap:</strong> {audit['demographic_parity_gap']:.1%}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="fast-track-pass">
                    <span class="status-badge badge-success">Compliant</span><br>
                    <strong>Statutory Equity Clearance</strong><br>
                    Sentencing patterns and discretionary override distribution adhere to constitutional standards.<br>
                    <strong>Disparate Impact Ratio:</strong> {audit['disparate_impact_ratio']} |
                    <strong>Demographic Parity Gap:</strong> {audit['demographic_parity_gap']:.1%}
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        with col_st1:
            st.metric("Total Cases Adjudicated", audit["total_cases"])
        with col_st2:
            st.metric("Discretionary Override Rate", f"{audit['override_rate']:.1%}")
        with col_st3:
            st.metric("Disparate Impact Ratio", f"{audit['disparate_impact_ratio']:.2f}")
        with col_st4:
            st.metric("Statistical p-value", f"{audit['p_value']}")

        st.markdown("---")
        c_comp1, c_comp2 = st.columns(2)

        with c_comp1:
            st.markdown("#### Marginalized Demographic Cohort")
            m_stats = audit["marginalized_stats"]
            st.write(f"- **Total Cases:** {m_stats['total']}")
            st.write(f"- **Favorable Disposition Rate (Bail/Probation):** {m_stats['favorable_rate']:.1%}")
            st.write(f"- **Custodial Detention Rate:** {m_stats['harsh_rate']:.1%}")
            st.write(f"- **Discretionary Overrides to Harsher Disposition:** {m_stats['overrides_to_harsh']}")

        with c_comp2:
            st.markdown("#### General / Control Demographic Cohort")
            g_stats = audit["general_stats"]
            st.write(f"- **Total Cases:** {g_stats['total']}")
            st.write(f"- **Favorable Disposition Rate (Bail/Probation):** {g_stats['favorable_rate']:.1%}")
            st.write(f"- **Custodial Detention Rate:** {g_stats['harsh_rate']:.1%}")
            st.write(f"- **Discretionary Overrides to Lenient Disposition:** {g_stats['overrides_to_lenient']}")

        # Findings and Actionable Next Steps
        st.subheader("Statistical Audit Findings")
        for finding in audit["findings"]:
            st.markdown(f"- {finding}")

        if audit["requires_board_review"]:
            st.error(
                "Board Directive: Initiate formal administrative review to examine case files "
                "where discretionary judicial overrides were exercised against marginalized defendants."
            )
        else:
            st.success("Board Directive: Routine compliance clearance; sentencing variance remains within acceptable bounds.")
