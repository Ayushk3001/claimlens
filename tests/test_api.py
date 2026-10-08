import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

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
    # Submit review and trigger controlled retraining
    fb_payload = {
        "claim_id": "CLM-0000000001",
        "adjuster_id": "ADJ-SENIOR-TEST",
        "adjuster_decision": "Sent to SIU",
        "agrees_with_ai": True,
        "adjusted_amount": 4200.0,
        "notes": "Verified intentional damage pattern."
    }
    client.post("/claims/feedback", json=fb_payload)
    
    retrain_res = client.post("/claims/feedback/retrain?min_feedback_count=1")
    assert retrain_res.status_code == 200
    rdata = retrain_res.json()
    assert rdata["status"] == "success"
    assert "active_version" in rdata
    assert "candidate_accuracy" in rdata


