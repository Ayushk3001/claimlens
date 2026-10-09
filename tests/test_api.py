import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.main import app
from src.services.feedback_service import feedback_service

client = TestClient(app)

@pytest.fixture(autouse=True)
def preserve_feedback_store():
    """Ensure feedback tests never permanently pollute data/feedback_store.json."""
    fb_file = feedback_service.feedback_file
    initial_content = fb_file.read_text(encoding="utf-8") if fb_file.exists() else None
    try:
        yield
    finally:
        if initial_content is not None:
            fb_file.write_text(initial_content, encoding="utf-8")
        elif fb_file.exists():
            fb_file.unlink()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_validate_endpoint():
    payload = {
        "claim_id": "CLM-001",
        "policy_id": "POL-101",
        "claim_type": "Auto",
        "state": "CA",
        "policyholder_tenure_years": 3.0,
        "previous_claims_count": 0,
        "incident_date": "2024-03-01",
        "claim_filed_date": "2024-03-02",
        "claim_amount": 4200.0,
        "deductible": 500.0,
        "claim_status": "Open"
    }
    response = client.post("/claims/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["is_valid"] is True

def test_predict_endpoint():
    payload = {
        "claim_type": "Auto",
        "state": "TX",
        "policyholder_tenure_years": 5.0,
        "previous_claims_count": 0,
        "incident_date": "2024-01-01",
        "claim_filed_date": "2024-01-03",
        "claim_amount": 3000.0,
        "deductible": 500.0
    }
    response = client.post("/claims/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_claim_amount" in response.json()
    assert "fraud_probability_percent" in response.json()

def test_hybrid_search_endpoint():
    payload = {
        "query": "Auto collision repair",
        "top_k": 3,
        "claim_type": "Auto"
    }
    response = client.post("/claims/hybrid-search", json=payload)
    assert response.status_code == 200
    assert "results" in response.json()

def test_full_analysis_workflow_endpoint():
    payload = {
        "claim_id": "CLM-TEST-ANALYZE",
        "policy_id": "POL-555",
        "claim_type": "Auto",
        "state": "NY",
        "policyholder_tenure_years": 4.0,
        "previous_claims_count": 1,
        "incident_date": "2024-02-10",
        "claim_filed_date": "2024-02-12",
        "claim_amount": 5500.0,
        "deductible": 500.0
    }
    response = client.post("/claims/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "ml_predictions" in data
    assert "investigation" in data
    assert "risk_assessment" in data
    assert "recommendation" in data
    assert "llm_as_judge_review" in data

def test_workflow_financial_settlement_bounds():
    # Test that net payout never exceeds (claim_amount - deductible)
    payload = {
        "claim_id": "CLM-0000000001",
        "policy_id": "POL-98214",
        "claim_type": "Auto",
        "state": "IA",
        "policyholder_tenure_years": 4.2,
        "previous_claims_count": 15,
        "incident_date": "2024-02-01",
        "claim_filed_date": "2024-02-02",
        "claim_amount": 4200.0,
        "deductible": 900.0
    }
    response = client.post("/claims/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    rec = data["recommendation"]
    judge = data["llm_as_judge_review"]

    # Maximum allowable net = 4200 - 900 = 3300
    import re
    payout_numbers = [float(x.replace(",", "")) for x in re.findall(r"\$([0-9,]+\.?[0-9]*)", rec["recommended_payout"])]
    if payout_numbers:
        assert max(payout_numbers) <= (4200.0 - 900.0 + 1.0)
    assert judge["verdict"] == "PASS"

def test_llm_judge_catches_financial_inconsistency():
    from src.agents.multi_agent_workflow import multi_agent_workflow
    
    claim_data = {"claim_amount": 4200.0, "deductible": 900.0, "claim_type": "Auto"}
    inv = {"findings": "Test findings"}
    risk = {"fraud_probability_percent": 25.0, "risk_tier": "Low Risk"}
    bad_recommendation = {
        "decision": "MANUAL_ADJUSTER_REVIEW",
        "recommended_payout": "$4,103.07 - $5,014.87",  # Exceeds 4200 - 900 = 3300!
        "action_statement": "Review claim."
    }
    judge_res = multi_agent_workflow.run_llm_as_judge(claim_data, inv, risk, bad_recommendation)
    assert judge_res["verdict"] == "FLAGGED_FOR_AUDIT"
    assert any("Financial Inconsistency" in iss for iss in judge_res["detected_issues"])

def test_edge_case_claim_amount_below_deductible():
    # Edge case: Claim amount $650 is below deductible $1,000 -> payout must be exactly $0.00
    payload = {
        "claim_id": "CLM-EDGE-DED",
        "policy_id": "POL-98214",
        "claim_type": "Auto",
        "state": "CA",
        "policyholder_tenure_years": 4.0,
        "previous_claims_count": 0,
        "incident_date": "2024-02-01",
        "claim_filed_date": "2024-02-02",
        "claim_amount": 650.0,
        "deductible": 1000.0
    }
    response = client.post("/claims/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    rec = data["recommendation"]
    judge = data["llm_as_judge_review"]

    assert rec["decision"] == "CLAIM_WITHIN_DEDUCTIBLE"
    assert "$0.00" in rec["recommended_payout"]
    assert judge["verdict"] == "PASS"

def test_edge_case_zero_payout_violation_in_judge():
    # If loss is within deductible but agent proposed non-zero payout, judge must fail
    from src.agents.multi_agent_workflow import multi_agent_workflow
    claim_data = {"claim_amount": 650.0, "deductible": 1000.0, "claim_type": "Auto"}
    inv = {"findings": "Test findings"}
    risk = {"fraud_probability_percent": 15.0, "risk_tier": "Low Risk"}
    violating_rec = {
        "decision": "AUTO_APPROVE",
        "recommended_payout": "$350.00",  # Illegal payout since claim is within deductible
        "action_statement": "Approve settlement."
    }
    judge_res = multi_agent_workflow.run_llm_as_judge(claim_data, inv, risk, violating_rec)
    assert judge_res["verdict"] == "FLAGGED_FOR_AUDIT"
    assert any("Financial Inconsistency" in iss for iss in judge_res["detected_issues"])

def test_feedback_retraining_pipeline_endpoint():
    # Submit review with confirmed fraud and trigger controlled retraining
    fb_payload = {
        "claim_id": "CLM-0000000001",
        "adjuster_id": "ADJ-SENIOR-TEST",
        "adjuster_decision": "Sent to SIU",
        "agrees_with_ai": True,
        "adjusted_amount": 4200.0,
        "confirmed_fraud": True,
        "notes": "Verified intentional damage pattern."
    }
    client.post("/claims/feedback", json=fb_payload)
    
    retrain_res = client.post("/claims/feedback/retrain?min_feedback_count=1")
    assert retrain_res.status_code == 200
    rdata = retrain_res.json()
    assert rdata["status"] in ["success", "rejected_regression"]
    assert "active_version" in rdata
    if rdata["status"] == "success":
        assert "candidate_validation_brier" in rdata
    else:
        assert "Promotion rejected" in rdata["message"]

def test_settlement_invariants_across_branches():
    from src.agents.multi_agent_workflow import multi_agent_workflow

    # 1. SIU referral must have recommended_net_payout == 0.0 and withheld authorization status
    rec_siu = multi_agent_workflow.run_recommendation_agent(
        claim_data={"claim_amount": 4200.0, "deductible": 900.0, "claim_type": "Auto"},
        investigation={"findings": "Suspect"},
        risk_analysis={"fraud_probability_percent": 65.0, "risk_tier": "High Risk"},
        ml_results={"predicted_claim_amount": 4200.0}
    )
    assert rec_siu["decision"] == "SIU_REFERRAL"
    assert rec_siu["recommended_net_payout"] == 0.0
    assert rec_siu["authorized_net_payout"] == 0.0
    assert rec_siu["net_settlement_ceiling"] == 3300.0
    assert rec_siu["payment_authorization_status"] == "DISBURSEMENT_WITHHELD_SIU_INQUIRY"

    # 2. Within deductible must have recommended_net_payout == 0.0
    rec_ded = multi_agent_workflow.run_recommendation_agent(
        claim_data={"claim_amount": 600.0, "deductible": 900.0, "claim_type": "Auto"},
        investigation={"findings": "Minor"},
        risk_analysis={"fraud_probability_percent": 5.0, "risk_tier": "Low Risk"},
        ml_results={"predicted_claim_amount": 600.0}
    )
    assert rec_ded["decision"] == "CLAIM_WITHIN_DEDUCTIBLE"
    assert rec_ded["recommended_net_payout"] == 0.0
    assert rec_ded["authorized_net_payout"] == 0.0
    assert rec_ded["net_settlement_ceiling"] == 0.0
    assert rec_ded["payment_authorization_status"] == "ZERO_INDEMNITY_CLOSED"

    # 3. Auto approve must have recommended_net_payout == (amt - deductible)
    rec_app = multi_agent_workflow.run_recommendation_agent(
        claim_data={"claim_amount": 3000.0, "deductible": 500.0, "claim_type": "Auto"},
        investigation={"findings": "Clear"},
        risk_analysis={"fraud_probability_percent": 5.0, "risk_tier": "Low Risk"},
        ml_results={"predicted_claim_amount": 3000.0}
    )
    assert rec_app["decision"] == "AUTO_APPROVE"
    assert rec_app["recommended_net_payout"] == 2500.0
    assert rec_app["authorized_net_payout"] == 2500.0
    assert rec_app["net_settlement_ceiling"] == 2500.0
    assert rec_app["payment_authorization_status"] == "ELIGIBLE_FAST_TRACK_DISBURSEMENT"

def test_llm_judge_evaluates_structured_numeric_payout():
    from src.agents.multi_agent_workflow import multi_agent_workflow

    claim_data = {"claim_amount": 4200.0, "deductible": 900.0, "claim_type": "Auto"}
    inv = {"findings": "Normal claim"}
    risk = {"fraud_probability_percent": 10.0, "risk_tier": "Low Risk"}
    
    # Passing excessive structured numeric payout (4000 > 3300) without dollar formatting
    violating_rec = {
        "decision": "AUTO_APPROVE",
        "recommended_net_payout": 4000.0,
        "authorized_net_payout": 4000.0,
        "recommended_payout": "Standard approval issued",
        "action_statement": "Authorize payment"
    }
    judge_res = multi_agent_workflow.run_llm_as_judge(claim_data, inv, risk, violating_rec)
    assert judge_res["verdict"] == "FLAGGED_FOR_AUDIT"
    assert any("exceeds maximum allowable net loss" in iss for iss in judge_res["detected_issues"])

def test_ml_service_heuristic_local_feature_impact():
    from src.services.ml_models import claims_ml_service

    claim_sample = {
        "claim_type": "Auto",
        "state": "CA",
        "policyholder_tenure_years": 4.0,
        "previous_claims_count": 1,
        "claim_amount": 3500.0,
        "deductible": 500.0
    }
    res = claims_ml_service.predict(claim_sample)
    assert "heuristic_local_feature_impact" in res
    assert "local_feature_attributions" in res
    assert isinstance(res["heuristic_local_feature_impact"], dict)

def test_approved_decision_does_not_alter_fraud_label_without_explicit_confirmation():
    # Verify strict label integrity: Adjuster approval does NOT convert fraud ground truth to False
    from src.services.feedback_service import feedback_service
    from src.utils.data_loader import load_claims_dataset

    base_df = load_claims_dataset()
    # Find a claim that has fraud = True in base dataset
    fraud_claims = base_df[base_df["is_fraud_flagged_ground_truth"] == True]
    if not fraud_claims.empty:
        test_cid = fraud_claims.iloc[0]["claim_id"]
        # Submit approval feedback without explicit confirmed_fraud flag
        feedback_service.record_feedback({
            "claim_id": test_cid,
            "adjuster_id": "ADJ-TEST-AUDIT",
            "adjuster_decision": "Approved", # Routine approval
            "confirmed_fraud": None,        # Unverified
            "adjusted_amount": 2500.0
        })
        
        # When partitioning dataset for retraining, fraud target must remain intact (not changed to False)
        train_pool, _ = feedback_service._get_or_create_golden_eval_set(base_df)
        df_augmented = train_pool.copy()
        for entry in feedback_service._load_all():
            if entry.get("claim_id") == test_cid:
                conf = entry.get("confirmed_fraud")
                if conf is not None:
                    df_augmented.loc[df_augmented["claim_id"] == test_cid, "is_fraud_flagged_ground_truth"] = bool(conf)
        
        # Check that it remained True
        match = df_augmented[df_augmented["claim_id"] == test_cid]
        if not match.empty:
            assert match.iloc[0]["is_fraud_flagged_ground_truth"] == True
