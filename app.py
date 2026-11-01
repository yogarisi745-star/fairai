"""
app.py - Interactive Web Prototype: AI Legal Bias Mitigation & Blockchain Accountability

Features:
- View 1: Accused Intake & AI Risk Assessment with SHAP (XAI) Attribution & Debiasing.
- View 2: Judicial Bench (HITL Level-1): Judge review, fast-track / override with reasoning, and blockchain commitment.
- View 3: Immutable Blockchain Ledger Explorer & Tamper Integrity Verifier.
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
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bias-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .fast-track-pass {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .board-flag {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 14px;
        border-radius: 6px;
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
st.sidebar.title("⚖️ Judicial AI Oversight")
st.sidebar.caption("Dual-Level Human-in-the-Loop & Blockchain Framework")

navigation = st.sidebar.radio(
    "Navigation Mode",
    [
        "1. Accused Assessment & SHAP Debiasing",
        "2. Judicial Bench (HITL-1)",
        "3. Blockchain Ledger & Tamper Guard",
        "4. Judicial Oversight Board (HITL-2)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Summary")
st.sidebar.markdown(
    """
    - **Step 1:** AI Risk Score (Historical Data)
    - **Step 2:** SHAP Factor Attribution & Debiasing
    - **Step 3:** Bias Gap & Judge Adjudication (HITL-1)
    - **Step 4:** Blockchain Ledger (SHA-256)
    - **Step 5:** Judicial Trend Audit & Alerts
    - **Step 6:** Judicial Oversight Board (HITL-2)
    """
)

# Show model weight source in sidebar
from model import LegalRiskModel as _LRM
_weight_src = _LRM.WEIGHT_SOURCE
if "COMPAS" in _weight_src:
    st.sidebar.success(f"🔬 Model weights: **{_weight_src}**")
else:
    st.sidebar.warning(f"⚠️ Model weights: **{_weight_src}** (CSV not found)")

# ==============================================================================
# TAB 1: ACCUSED ASSESSMENT & SHAP DEBIASING
# ==============================================================================
if navigation == "1. Accused Assessment & SHAP Debiasing":
    st.title("Case Assessment & Explainable AI (SHAP) Debiasing")
    st.write(
        "Evaluate an accused individual's recidivism risk score, isolate protected demographic bias using SHAP, "
        "and perform counterfactual debiasing."
    )

    col_bench, col_manual = st.columns([1, 2])

    with col_bench:
        st.subheader("Select Benchmark Case")
        case_options = {f"{c['case_id']} - {c.get('case_label', c['accused_name'])}": c for c in BENCHMARK_CASES}
        selected_case_name = st.selectbox("Pre-configured Cases", list(case_options.keys()))
        selected_case = case_options[selected_case_name]

        use_selected = st.button("Load Selected Case Details", use_container_width=True)

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
            st.markdown("**Protected Demographics (Bias Scrutiny)**")
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

        run_assessment = st.button("Run AI Risk Scoring & SHAP Debiasing", type="primary", use_container_width=True)

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
            st.metric("Triage Status", res["bias_flag"])

        # Bias Alert Banner
        if res["is_high_bias"]:
            st.markdown(
                f"""
                <div class="bias-alert">
                    <strong>🚨 HIGH AI BIAS DETECTED (Bias Gap: +{res['bias_gap']} points)</strong><br>
                    {res['triage_guidance']}<br>
                    <strong>Action Required:</strong> Case flagged for mandatory Human Judge scrutiny. Fast-tracking prohibited.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="fast-track-pass">
                    <strong>✅ LOW AI BIAS (Disparity Gap: {res['bias_gap']} points)</strong><br>
                    {res['triage_guidance']}<br>
                    <strong>Action:</strong> Eligible for judicial fast-tracking with standard review.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Disposition Comparison
        col_disp1, col_disp2 = st.columns(2)
        with col_disp1:
            st.markdown("**Initial Biased AI Recommendation:**")
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
            "Sensitive demographic factors are categorized separately from legitimate criminal merits."
        )

        shap_df = pd.DataFrame(
            [
                {
                    "Feature": k,
                    "Attribution (Points)": v,
                    "Category": (
                        "Protected Demographic (Bias)"
                        if k in ["race_caste", "income_level", "gender"]
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

        st.info("👉 Switch to the **'2. Judicial Bench (HITL-1)'** tab in the sidebar to adjudicate this case.")


# ==============================================================================
# TAB 2: JUDICIAL BENCH (HUMAN-IN-THE-LOOP LEVEL 1)
# ==============================================================================
elif navigation == "2. Judicial Bench (HITL-1)":
    st.title("Judicial Bench: Human-in-the-Loop Adjudication (HITL-1)")
    st.write(
        "Human judges review the case, inspect AI bias flags and SHAP explanations, "
        "and exercise constitutional authority to accept or override AI recommendations."
    )

    if not st.session_state.current_assessment:
        st.warning("⚠️ No active case loaded. Please run assessment in Tab 1 or select a demo case below.")
        if st.button("Load Sample High-Bias Case (real COMPAS record)"):
            st.session_state.current_assessment = system.assess_case(BENCHMARK_CASES[0])
            st.rerun()
    else:
        case = st.session_state.current_assessment

        st.subheader(f"Dossier: Case #{case['case_id']} - {case['accused_name']}")

        c_info1, c_info2 = st.columns(2)
        with c_info1:
            st.markdown("**Defendant & Case Details:**")
            st.write(f"- **Allegation:** {case['description']}")
            st.write(f"- **Demographics:** {case['demographics']}")
            st.write(f"- **Criminal Record:** {case['merits']}")

        with c_info2:
            st.markdown("**AI Model & Debiasing Audit:**")
            st.write(f"- **Raw AI Risk Score:** {case['ai_original_score']} / 100")
            st.write(f"- **Debiased Risk Score:** {case['ai_debiased_score']} / 100")
            st.write(f"- **Demographic Disparity Gap:** {case['bias_gap']} pts")
            st.write(f"- **Top Bias Factor:** `{case['top_contributing_factor']}` ({case['top_factor_impact']} pts)")

        st.markdown("---")
        st.subheader("Judge's Official Adjudication Form")

        judge_id = st.selectbox(
            "Presiding Judge ID",
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
            f"Court review of case {case['case_id']} affirms that the initial AI risk calculation "
            f"was inflated by protected demographic attributes ({case['top_contributing_factor']}). "
            f"Upon counterfactual evaluation and considering defendant's non-violent background, "
            f"the court orders {final_verdict}."
            if case["is_high_bias"]
            else f"Independent judicial assessment affirms legitimate criminal history warrants {final_verdict}."
        )

        judicial_reasoning = st.text_area(
            "Written Judicial Rationale (Mandatory for Due Process & Legal Accountability)",
            value=default_reasoning,
            height=120,
        )

        commit_btn = st.button("Commit Final Verdict to Blockchain Ledger", type="primary", use_container_width=True)

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
                    f"✅ Case {case['case_id']} successfully committed to Blockchain Ledger in Block #{commit_res['block_index']}!"
                )
                st.markdown(
                    f"""
                    - **Block Index:** `#{commit_res['block_index']}`
                    - **SHA-256 Hash:** `{commit_res['block_hash']}`
                    - **Previous Hash:** `{commit_res['previous_hash']}`
                    - **Action:** `{commit_res['judge_action']}` -> `{commit_res['final_verdict']}`
                    """
                )
            except Exception as e:
                st.error(f"Error committing to blockchain: {e}")


# ==============================================================================
# TAB 3: BLOCKCHAIN LEDGER & TAMPER GUARD
# ==============================================================================
elif navigation == "3. Blockchain Ledger & Tamper Guard":
    st.title("Cryptographic Blockchain Ledger & Immutability Guard")
    st.write(
        "Every judicial decision, AI score, SHAP explanation, and judge's reasoning is "
        "permanently sealed into SHA-256 blocks to prevent post-hoc reputational tampering or corruption."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        verify_chain = st.button("🔍 Verify Entire Blockchain Integrity", use_container_width=True)
    with col_btn2:
        tamper_demo = st.button("⚠️ Simulate Unauthorized Database Tampering (Block #1)", use_container_width=True)

    if verify_chain:
        is_valid, msg = system.blockchain.verify_chain_integrity()
        if is_valid:
            st.success(f"✅ {msg}")
        else:
            st.error(f"🛑 INTEGRITY VIOLATION: {msg}")

    if tamper_demo:
        system.blockchain.simulate_tampering(block_index=1, new_verdict="Tampered: Complete Dismissal")
        st.warning("Simulated unauthorized edit on Block #1: verdict modified without recomputing cryptographic hash.")
        is_valid, msg = system.blockchain.verify_chain_integrity()
        st.error(f"🛑 AUDIT ALERT: {msg}")

    st.subheader(f"Ledger Blocks ({len(system.blockchain.chain)} Blocks Total)")

    blocks_data = []
    for blk in reversed(system.blockchain.chain):
        cdata = blk.case_data
        blocks_data.append(
            {
                "Block #": blk.index,
                "Case ID": cdata.get("case_id", "N/A"),
                "Judge": cdata.get("judge_id", "System Genesis"),
                "AI Score": cdata.get("ai_original_score", "N/A"),
                "Debiased Score": cdata.get("ai_debiased_score", "N/A"),
                "Judge Action": cdata.get("judge_action", "N/A"),
                "Final Verdict": cdata.get("final_verdict", cdata.get("message", "N/A")),
                "Block Hash": blk.hash[:16] + "...",
                "Prev Hash": blk.previous_hash[:16] + "...",
            }
        )

    st.dataframe(pd.DataFrame(blocks_data), use_container_width=True)

    with st.expander("Inspect Raw Block Cryptographic Payloads"):
        selected_blk_idx = st.selectbox(
            "Select Block to Inspect",
            range(len(system.blockchain.chain)),
            index=len(system.blockchain.chain) - 1,
        )
        st.json(system.blockchain.chain[selected_blk_idx].to_dict())


# ==============================================================================
# TAB 4: JUDICIAL OVERSIGHT BOARD (HUMAN-IN-THE-LOOP LEVEL 2)
# ==============================================================================
elif navigation == "4. Judicial Oversight Board (HITL-2)":
    st.title("Judicial Oversight Board: Aggregate Bias Audit (HITL-2)")
    st.write(
        "While Level-1 Human-in-the-Loop allows judges to correct AI bias, Level-2 AI monitoring "
        "audits the judges themselves across the blockchain ledger. Detects systematic bias, favoritism, "
        "or disparate impact, and submits automated dossiers to the Judicial Oversight Board."
    )

    judges_list = system.list_all_judges()
    if not judges_list:
        st.info("No judicial cases recorded yet.")
    else:
        selected_judge = st.selectbox("Select Judge to Audit", judges_list)
        audit = system.audit_judge(selected_judge)

        st.subheader(f"Judicial Equity Dossier: {audit['judge_id']}")

        # Status Banner
        if audit["status"] == "FLAGGED_FOR_INVESTIGATION":
            st.markdown(
                f"""
                <div class="board-flag">
                    <strong>🚨 FLAGGED FOR JUDICIAL OVERSIGHT BOARD INVESTIGATION</strong><br>
                    AI judicial trend monitoring detected statistically significant disparity in sentencing or override behavior.<br>
                    <strong>Disparate Impact Ratio:</strong> {audit['disparate_impact_ratio']} (Legal Standard ≥ 0.80) |
                    <strong>Demographic Parity Gap:</strong> {audit['demographic_parity_gap']:.1%}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="fast-track-pass">
                    <strong>✅ EQUITABLE_COMPLIANT</strong><br>
                    Sentencing patterns and override rates adhere to constitutional equity standards.<br>
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
            st.metric("Override Rate", f"{audit['override_rate']:.1%}")
        with col_st3:
            st.metric("Disparate Impact Ratio", f"{audit['disparate_impact_ratio']:.2f}")
        with col_st4:
            st.metric("Statistical p-value", f"{audit['p_value']}")

        st.markdown("---")
        c_comp1, c_comp2 = st.columns(2)

        with c_comp1:
            st.markdown("#### Marginalized Demographic Group")
            m_stats = audit["marginalized_stats"]
            st.write(f"- **Total Cases:** {m_stats['total']}")
            st.write(f"- **Favorable Rate (Bail/Probation):** {m_stats['favorable_rate']:.1%}")
            st.write(f"- **Severe Rate (Incarceration):** {m_stats['harsh_rate']:.1%}")
            st.write(f"- **Overrides to Harsher Sentence:** {m_stats['overrides_to_harsh']}")

        with c_comp2:
            st.markdown("#### General / Control Demographic Group")
            g_stats = audit["general_stats"]
            st.write(f"- **Total Cases:** {g_stats['total']}")
            st.write(f"- **Favorable Rate (Bail/Probation):** {g_stats['favorable_rate']:.1%}")
            st.write(f"- **Severe Rate (Incarceration):** {g_stats['harsh_rate']:.1%}")
            st.write(f"- **Overrides to Lenient Sentence:** {g_stats['overrides_to_lenient']}")

        # Findings and Actionable Next Steps
        st.subheader("Automated Judicial Oversight Findings")
        for finding in audit["findings"]:
            st.markdown(f"- {finding}")

        if audit["requires_board_review"]:
            st.error(
                "⚖️ **Board Action Recommended:** Initiate an independent review panel to scrutinize "
                "case files where discretionary overrides were exercised against marginalized defendants."
            )
        else:
            st.success("⚖️ **Board Action:** Routine clearance; no anomalies detected.")
