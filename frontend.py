import streamlit as st
import pandas as pd
import json
from datetime import datetime

from src.utils.guardrails import validate_claim_input
from src.rag.hybrid_retriever import hybrid_retriever
from src.services.ml_models import claims_ml_service
from src.services.feedback_service import feedback_service
from src.agents.multi_agent_workflow import multi_agent_workflow
from src.agents.classification_agent import classification_agent
from src.utils.data_loader import load_claims_dataset

# Set Page Config
st.set_page_config(
    page_title="ClaimLens — AI-Powered Insurance Claims Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-green { background-color: #DEF7EC; color: #03543F; }
    .badge-yellow { background-color: #FEF08A; color: #854D0E; }
    .badge-red { background-color: #FDE8E8; color: #9B1C1C; }
</style>
""", unsafe_allow_html=True)

# Pre-warm services
@st.cache_resource
def init_services():
    hybrid_retriever.initialize()
    claims_ml_service.train_or_load()
    return True

init_services()

# Sidebar: System Status & Sample Loader
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=64)
st.sidebar.title("System Status")
st.sidebar.success(f"⚡ Vector DB: {hybrid_retriever.vector_store.count()} indexed")
st.sidebar.info("🤖 Models: Hybrid RAG + Random Forest + Multi-Agent")
st.sidebar.markdown("---")

df_sample = load_claims_dataset()

st.sidebar.subheader("Quick Test Data")
if st.sidebar.button("🎲 Load Random Claim Sample"):
    rand_row = df_sample.sample(1).iloc[0].to_dict()
    st.session_state["claim_id"] = str(rand_row.get("claim_id", "CLM-NEW-001"))
    st.session_state["policy_id"] = str(rand_row.get("policy_id", "POL-100234"))
    st.session_state["claim_type"] = str(rand_row.get("claim_type", "Auto")).strip().capitalize()
    st.session_state["state"] = str(rand_row.get("state", "CA")).strip().upper()
    st.session_state["tenure"] = float(rand_row.get("policyholder_tenure_years", 3.0))
    st.session_state["prev_claims"] = int(rand_row.get("previous_claims_count", 0))
    st.session_state["amount"] = float(rand_row.get("claim_amount", 3500.0))
    st.session_state["deductible"] = float(rand_row.get("deductible", 500.0))
    st.session_state["incident_date"] = str(rand_row.get("incident_date", "2024-01-10"))
    st.session_state["claim_filed_date"] = str(rand_row.get("claim_filed_date", "2024-01-14"))
    st.sidebar.success(f"Loaded: {st.session_state['claim_id']}")

# Title Banner
st.markdown('<div class="main-title">🛡️ ClaimLens — AI-Powered Insurance Claims Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Triage, Hybrid RAG Historical Precedents, Multi-Agent Workflow, and Explainable Risk Prediction</div>', unsafe_allow_html=True)

# Top Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Multi-Agent Analysis & Triage",
    "🔍 Hybrid Search & Precedents",
    "📊 ML Prediction & Explainability",
    "🔄 Feedback Loop & Learning"
])

# ================= TAB 1: Multi-Agent Analysis & Triage =================
with tab1:
    st.subheader("1. Claim Intake & Details")
    
    CLAIM_TYPES = ["Auto", "Home", "Renters", "Property", "Business"]
    US_STATES = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]

    curr_type = str(st.session_state.get("claim_type", "Auto")).strip().capitalize()
    type_idx = CLAIM_TYPES.index(curr_type) if curr_type in CLAIM_TYPES else 0

    curr_state = str(st.session_state.get("state", "CA")).strip().upper()
    state_idx = US_STATES.index(curr_state) if curr_state in US_STATES else 4  # Default CA

    col1, col2, col3 = st.columns(3)
    with col1:
        cid = st.text_input("Claim ID", value=st.session_state.get("claim_id", "CLM-0000000001"))
        c_type = st.selectbox("Claim Type", CLAIM_TYPES, index=type_idx)
        state = st.selectbox("State", US_STATES, index=state_idx)
    with col2:
        pol_id = st.text_input("Policy ID", value=st.session_state.get("policy_id", "POL-98214"))
        tenure = st.number_input("Policyholder Tenure (Years)", min_value=0.0, max_value=60.0, value=float(st.session_state.get("tenure", 4.2)), step=0.1)
        prev_claims = st.number_input("Previous Claims Count", min_value=0, max_value=20, value=int(st.session_state.get("prev_claims", 0)))
    with col3:
        amount = st.number_input("Claim Amount ($)", min_value=0.0, value=float(st.session_state.get("amount", 4200.0)), step=100.0)
        deductible = st.number_input("Policy Deductible ($)", min_value=0.0, value=float(st.session_state.get("deductible", 500.0)), step=50.0)
        inc_date = st.text_input("Incident Date (YYYY-MM-DD)", value=st.session_state.get("incident_date", "2024-02-01"))
        filed_date = st.text_input("Claim Filed Date (YYYY-MM-DD)", value=st.session_state.get("claim_filed_date", "2024-02-05"))

    desc = st.text_area("Incident Description / Notes", value="Vehicle rear fender damaged in bumper collision at traffic intersection.")

    claim_payload = {
        "claim_id": cid,
        "policy_id": pol_id,
        "claim_type": c_type,
        "state": state,
        "policyholder_tenure_years": tenure,
        "previous_claims_count": prev_claims,
        "incident_date": inc_date,
        "claim_filed_date": filed_date,
        "claim_amount": amount,
        "deductible": deductible,
        "claim_status": "Open",
        "description": desc
    }

    # Guardrails verification
    val = validate_claim_input(claim_payload)
    if not val.is_valid:
        st.error(f"⚠️ Guardrail Violations: {', '.join(val.errors)}")
    if val.warnings:
        for w in val.warnings:
            st.warning(f"ℹ️ Guardrail Notice: {w}")

    if st.button("🚀 Run Multi-Agent Claims Assessment", type="primary", disabled=not val.is_valid):
        with st.spinner("Executing Multi-Agent Workflow (Investigation -> Risk -> Recommendation -> LLM Judge)..."):
            workflow_out = multi_agent_workflow.process_claim(val.sanitized_data)
            
            # Triage & Priority Ribbon
            triage = workflow_out["classification"]
            rec = workflow_out["recommendation"]
            risk = workflow_out["risk_assessment"]
            judge = workflow_out["llm_as_judge_review"]

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Priority Tier", triage["priority_tier"])
            m2.metric("Target SLA", f"{triage['target_sla_resolution_hours']} Hours")
            m3.metric("Fraud Risk Score", f"{risk['fraud_probability_percent']}%")
            m4.metric("LLM-as-Judge Score", f"{judge['overall_quality_score']} / 10.0")

            # Helper to prevent KaTeX LaTeX math mode from corrupting currency text
            def safe_txt(t):
                return str(t).replace("$", r"\$")

            # Agent 1: Investigation
            with st.expander("🕵️ Agent 1: Investigation Findings", expanded=True):
                inv = workflow_out["investigation"]
                st.markdown(safe_txt(inv["findings"]))
                if inv["key_precedents"]:
                    st.markdown("**Historical Precedent Clustered Claims:**")
                    for p in inv["key_precedents"]:
                        st.markdown(f"- `{safe_txt(p)}`")

            # Agent 2: Risk Assessment
            with st.expander("⚠️ Agent 2: Risk & Fraud Assessment", expanded=True):
                st.markdown(safe_txt(risk["risk_narrative"]))
                st.markdown("**Primary Risk Indicators:**")
                for rd in risk["risk_drivers"]:
                    st.markdown(f"- 🔴 {rd}")
                st.caption("Provenance: Scikit-Learn RandomForestClassifier (80 estimators, 5-fold Platt calibration) trained on 5,000 ground-truth labels from Hugging Face FreeInsuranceClaims100M.")

            # Agent 3: Final Handling Recommendation
            with st.expander("📋 Agent 3: Actionable Recommendation", expanded=True):
                decision_badge = "badge-green" if rec["decision"] == "AUTO_APPROVE" else ("badge-yellow" if rec["decision"] == "MANUAL_ADJUSTER_REVIEW" else "badge-red")
                st.markdown(f'**Recommended Pathway:** <span class="badge-pill {decision_badge}">{rec["decision"]}</span>', unsafe_allow_html=True)
                st.info(f"**Action Statement:** {safe_txt(rec['action_statement'])}")
                st.markdown(f"**Recommended Payout:** {safe_txt(rec['recommended_payout'])}")
                max_allowable_calc = max(0.0, claim_payload['claim_amount'] - claim_payload['deductible'])
                st.caption(f"Financial Settlement Rule: Net allowable payout is bounded by max(0, Claim Amount - Deductible) = max(0, \\${claim_payload['claim_amount']:,.2f} - \\${claim_payload['deductible']:,.2f}) = **\\${max_allowable_calc:,.2f}**.")
                st.markdown("**Actionable Settlement Checklist:**")
                for step in rec["actionable_steps"]:
                    st.markdown(f"- [ ] {safe_txt(step)}")
                st.markdown(f"**Executive Synthesis:** *{safe_txt(rec['executive_rationale'])}*")

            # LLM-as-Judge Card
            with st.expander("⚖️ Independent LLM-as-Judge Quality Audit", expanded=True):
                st.write(f"**Verdict:** `{judge['verdict']}` | **Score:** `{judge['overall_quality_score']}/10.0`")
                sc_col1, sc_col2, sc_col3 = st.columns(3)
                sc_col1.metric("Factual Consistency", f"{judge['metric_scores']['factual_consistency']}/10")
                sc_col2.metric("Completeness", f"{judge['metric_scores']['completeness']}/10")
                sc_col3.metric("Policy Compliance", f"{judge['metric_scores']['policy_compliance']}/10")
                st.caption(f"Audit Critique: {judge['audit_critique']}")
                if judge.get("detected_issues"):
                    st.markdown("**Audit Findings / Violations:**")
                    for iss in judge["detected_issues"]:
                        st.warning(f"⚠️ {iss}")


# ================= TAB 2: Hybrid Search & Precedents =================
with tab2:
    st.subheader("Hybrid Claims Retrieval (In-Memory Vector Cosine + BM25 Keyword Search)")
    search_q = st.text_input("Enter natural language query or loss scenario", value="Auto accident rear-end damage collision repair")
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        f_type = st.selectbox("Filter Claim Type", ["All", "Auto", "Home", "Renters", "Property", "Business"])
    with sc2:
        f_state = st.selectbox("Filter State", ["All", "CA", "TX", "NY", "FL", "PA", "IL", "GA", "OH"])
    with sc3:
        f_status = st.selectbox("Filter Claim Status", ["All", "Approved", "Closed", "Open", "Denied", "Under Investigation"])

    if st.button("🔍 Search Historical Claims"):
        with st.spinner("Searching vector index and keyword corpus..."):
            ret_type = None if f_type == "All" else f_type
            ret_state = None if f_state == "All" else f_state
            ret_status = None if f_status == "All" else f_status

            results = hybrid_retriever.search(
                query=search_q,
                top_k=6,
                claim_type=ret_type,
                state=ret_state,
                claim_status=ret_status
            )

            if not results:
                st.warning("No similar claims matched the criteria.")
            else:
                st.success(f"Found {len(results)} relevant historical claims:")
                for r in results:
                    with st.container():
                        st.markdown(f"#### 📄 Claim {r['claim_id']} ({r['claim_type']} - {r['state']})")
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.write(f"**Amount:** ${r['claim_amount']:,.2f}")
                        rc2.write(f"**Status:** `{r['claim_status']}`")
                        rc3.write(f"**Method:** `{r.get('retrieval_method', 'Hybrid')}`")
                        rc4.write(f"**Relevance Score:** `{r['hybrid_score']}`")
                        st.write(f"**Summary:** {r.get('narrative', 'N/A')}")
                        st.markdown("---")


# ================= TAB 3: ML Prediction & Explainability =================
with tab3:
    st.subheader("Machine Learning Predictions & Explainable AI")
    st.caption("Trained on Hugging Face 100M insurance claims sample using Scikit-Learn models.")
    
    if st.button("⚡ Run ML Predictive Analysis for Current Claim"):
        preds = claims_ml_service.predict(claim_payload)
        
        p1, p2, p3 = st.columns(3)
        p1.metric("Peer Loss Benchmark", f"${preds['predicted_claim_amount']:,.2f}", f"Peer Range: {preds['estimated_amount_range']}")
        p2.metric("Predicted Resolution Time", f"{preds['predicted_days_to_resolution']} Days")
        cal_score = preds.get('calibrated_fraud_probability_percent', preds['fraud_probability_percent'])
        p3.metric("Fraud Risk Score", f"{preds['fraud_probability_percent']}%", f"Calibrated: {cal_score}% ({preds['risk_tier']})")

        net_ceiling = max(0.0, claim_payload['claim_amount'] - claim_payload['deductible'])
        st.caption(f"💡 **Actuarial Note:** Peer loss benchmark reflects statistical loss across similar historical claims. Net payable settlement is capped at Claim Amount minus Deductible: **${net_ceiling:,.2f}** | Active Model Version: **{preds.get('model_version', 'v1.0')}**")

        st.markdown("### 🔍 Explainable Feature Drivers & Attribution")
        st.caption("Clearly distinguishes statistical ML model drivers from business underwriting policy flags:")
        
        ed1, ed2 = st.columns(2)
        with ed1:
            st.markdown("#### 🤖 Machine Learning Feature Drivers")
            ml_drivers = [d for d in preds["top_risk_drivers"] if "[ML Model Driver]" in d]
            for md in ml_drivers:
                st.markdown(f"- 📌 {md.replace('[ML Model Driver] ', '')}")
        with ed2:
            st.markdown("#### 📋 Underwriting Policy Rules")
            rule_drivers = [d for d in preds["top_risk_drivers"] if "[Policy Rule]" in d]
            if not rule_drivers:
                st.write("No restrictive policy rules triggered.")
            for rd in rule_drivers:
                st.markdown(f"- ⚠️ {rd.replace('[Policy Rule] ', '')}")

        st.markdown("### 📊 Global Feature Importance Distribution")
        st.caption("ℹ️ Global weights represent Tree Gini split importances. Local sample attributions represent an interpretable Gini-weighted proxy indicator, distinguished from exact TreeSHAP values.")
        imp_df = pd.DataFrame(
            list(preds["feature_importance_summary"].items()),
            columns=["Feature", "Importance Weight"]
        ).sort_values(by="Importance Weight", ascending=False)
        st.bar_chart(imp_df.set_index("Feature"))


# ================= TAB 4: Feedback Loop & Continuous Improvement =================
with tab4:
    st.subheader("Human-in-the-Loop Feedback & Continuous Improvement")
    st.caption("Enables claim adjusters to record audit decisions, override recommendations, and trigger versioned model retraining.")

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("#### Submit Adjuster Review")
        fb_claim_id = st.text_input("Claim ID for Review", value=cid)
        fb_decision = st.selectbox("Adjuster Decision", ["Approved", "Modified Payout", "Sent to SIU", "Denied"])
        fb_agrees = st.checkbox("Agrees with AI Recommendation", value=True)
        fb_adj_amt = st.number_input("Adjusted Payout Amount ($)", value=amount)
        fb_confirmed_fraud = st.selectbox(
            "Ground Truth Fraud Determination",
            ["Unconfirmed / Pending Investigation", "Confirmed Fraud (True)", "Confirmed Legitimate / Exonerated (False)"]
        )
        fb_notes = st.text_area("Adjuster Audit Notes", "Reviewed police report and verified repair quote.")
        
        if st.button("💾 Save Review to Continuous Improvement Store"):
            confirmed_fraud_val = None
            if fb_confirmed_fraud == "Confirmed Fraud (True)":
                confirmed_fraud_val = True
            elif fb_confirmed_fraud == "Confirmed Legitimate / Exonerated (False)":
                confirmed_fraud_val = False

            entry = feedback_service.record_feedback({
                "claim_id": fb_claim_id,
                "adjuster_id": "ADJ-CURRENT-USER",
                "adjuster_decision": fb_decision,
                "agrees_with_ai": fb_agrees,
                "adjusted_amount": fb_adj_amt,
                "confirmed_fraud": confirmed_fraud_val,
                "notes": fb_notes
            })
            st.success(f"Feedback logged successfully (ID: {entry['feedback_id']})")

    with fc2:
        st.markdown("#### Real-Time Feedback Analytics")
        metrics = feedback_service.get_feedback_metrics()
        st.metric("Total Adjuster Reviews", metrics["total_reviews"])
        st.metric("AI Recommendation Agreement Rate", f"{metrics['ai_agreement_rate_percent']}%")
        
        if metrics["decision_distribution"]:
            dist_df = pd.DataFrame(
                list(metrics["decision_distribution"].items()),
                columns=["Decision", "Count"]
            )
            st.dataframe(dist_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔄 MLOps Model Retraining Pipeline")
        st.caption("Triggers controlled retraining incorporating verified adjuster decisions with versioning and validation promotion gates.")
        if st.button("🚀 Trigger Model Retraining with Feedback"):
            with st.spinner("Retraining candidate model bundle with feedback signals..."):
                retrain_res = feedback_service.trigger_retraining(claims_ml_service, min_feedback_count=1)
                if retrain_res["status"] == "success":
                    st.success(f"Retraining Complete! Active Model upgraded to **{retrain_res['active_version']}** (Brier score: {retrain_res.get('candidate_validation_brier')})")
                    st.json(retrain_res)
                elif retrain_res["status"] == "rejected_regression":
                    st.warning(f"⚠️ Promotion Gate Tripped: {retrain_res['message']}")
                    st.json(retrain_res)
                else:
                    st.info(retrain_res["message"])

        if metrics["recent_reviews"]:
            st.markdown("##### Recent Feedback Logs")
            st.dataframe(pd.DataFrame(metrics["recent_reviews"]), use_container_width=True)
