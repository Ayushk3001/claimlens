import pytest
from src.services.ml_models import claims_ml_service
from src.agents.classification_agent import classification_agent

def test_ml_model_prediction():
    claims_ml_service.train_or_load()
    sample = {
        "claim_type": "Home",
        "state": "FL",
        "policyholder_tenure_years": 2.0,
        "previous_claims_count": 1,
        "incident_date": "2024-02-01",
        "claim_filed_date": "2024-02-05",
        "claim_amount": 12500.0,
        "deductible": 1000.0
    }
    pred = claims_ml_service.predict(sample)
    assert "predicted_claim_amount" in pred
    assert "predicted_days_to_resolution" in pred
    assert "fraud_probability_percent" in pred
    assert 0.0 <= pred["fraud_probability_percent"] <= 100.0
    assert "top_risk_drivers" in pred
    assert len(pred["top_risk_drivers"]) > 0

def test_classification_and_prioritization():
    sample = {
        "claim_amount": 75000.0,
        "policyholder_tenure_years": 0.3,
        "previous_claims_count": 3,
        "claim_type": "Business"
    }
    triage = classification_agent.classify_and_prioritize(sample)
    assert triage["complexity_level"] == "High Complexity"
    assert "P1" in triage["priority_tier"]
    assert triage["priority_score"] >= 70.0
