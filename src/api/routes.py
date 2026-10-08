from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.utils.guardrails import validate_claim_input
from src.rag.hybrid_retriever import hybrid_retriever
from src.services.ml_models import claims_ml_service
from src.services.feedback_service import feedback_service
from src.agents.classification_agent import classification_agent
from src.agents.multi_agent_workflow import multi_agent_workflow
from src.utils.data_loader import load_claims_dataset

router = APIRouter(prefix="", tags=["Insurance Claims"])

class ClaimInputRequest(BaseModel):
    claim_id: Optional[str] = Field("CLM-NEW-001", description="Unique Claim ID")
    policy_id: Optional[str] = Field("POL-100234", description="Policy ID")
    claim_type: str = Field("Auto", description="Insurance line: Auto, Home, Renters, Property, Business")
    state: str = Field("CA", description="2-letter US State code e.g. CA, NY, TX")
    policyholder_tenure_years: float = Field(3.0, ge=0.0, description="Tenure in years")
    previous_claims_count: int = Field(0, ge=0, description="Number of prior claims filed")
    incident_date: Optional[str] = Field("2024-01-10", description="Incident date (YYYY-MM-DD)")
    claim_filed_date: Optional[str] = Field("2024-01-14", description="Claim filing date (YYYY-MM-DD)")
    claim_amount: float = Field(3500.0, ge=0.0, description="Claimed loss amount in USD")
    deductible: float = Field(500.0, ge=0.0, description="Policy deductible in USD")
    claim_status: Optional[str] = Field("Open", description="Claim status")
    description: Optional[str] = Field(None, description="Optional incident description notes")

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query or description of the incident")
    top_k: int = Field(5, ge=1, le=25, description="Number of similar claims to retrieve")
    claim_type: Optional[str] = None
    state: Optional[str] = None
    claim_status: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    max_prev_claims: Optional[int] = None

class FeedbackRequest(BaseModel):
    claim_id: str
    adjuster_id: Optional[str] = "ADJ-DEFAULT"
    adjuster_decision: str = Field(..., description="Approved, Modified, Rejected, Sent to SIU")
    agrees_with_ai: bool = True
    adjusted_amount: Optional[float] = None
    confirmed_fraud: Optional[bool] = Field(None, description="Explicit confirmation of fraud ground truth: True (confirmed fraud), False (exonerated), None (unconfirmed/pending)")
    notes: Optional[str] = ""

@router.get("/health")
def health_check():
    """System health check and component status."""
    return {
        "status": "healthy",
        "service": "ClaimLens — AI-Powered Insurance Claims Assistant",
        "vector_store_indexed_count": hybrid_retriever.vector_store.count(),
        "ml_models_ready": claims_ml_service.is_trained
    }

@router.post("/claims/validate")
def validate_claim(payload: ClaimInputRequest):
    """Validate claim inputs against schema and business guardrails."""
    val_res = validate_claim_input(payload.model_dump())
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail={"errors": val_res.errors, "warnings": val_res.warnings})
    return val_res.to_dict()

@router.post("/claims/hybrid-search")
def search_similar_claims(payload: HybridSearchRequest):
    """Retrieve similar historical claims using Hybrid Search (ChromaDB Vector + BM25) with metadata filtering."""
    results = hybrid_retriever.search(
        query=payload.query,
        top_k=payload.top_k,
        claim_type=payload.claim_type,
        state=payload.state,
        claim_status=payload.claim_status,
        min_amount=payload.min_amount,
        max_amount=payload.max_amount,
        max_prev_claims=payload.max_prev_claims
    )
    return {
        "query": payload.query,
        "count": len(results),
        "results": results
    }

@router.post("/claims/classify")
def classify_claim(payload: ClaimInputRequest):
    """Classify claim complexity and calculate priority triage score."""
    val_res = validate_claim_input(payload.model_dump())
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail=val_res.errors)
    return classification_agent.classify_and_prioritize(val_res.sanitized_data)

@router.post("/claims/predict")
def predict_claim_metrics(payload: ClaimInputRequest):
    """Predict claim payout amount, resolution days, and fraud probability with explainability."""
    val_res = validate_claim_input(payload.model_dump())
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail=val_res.errors)
    return claims_ml_service.predict(val_res.sanitized_data)

@router.post("/claims/analyze")
def analyze_claim_workflow(payload: ClaimInputRequest):
    """Execute end-to-end multi-agent claims workflow (Investigation, Risk, Recommendation, and LLM Judge)."""
    val_res = validate_claim_input(payload.model_dump())
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail=val_res.errors)
    
    result = multi_agent_workflow.process_claim(val_res.sanitized_data)
    result["guardrails"] = {
        "is_valid": val_res.is_valid,
        "warnings": val_res.warnings
    }
    return result

@router.post("/claims/feedback")
def submit_feedback(payload: FeedbackRequest):
    """Record adjuster decision feedback for continuous learning loop."""
    return feedback_service.record_feedback(payload.model_dump())

@router.get("/claims/feedback/metrics")
def get_feedback_metrics():
    """Retrieve continuous improvement feedback loop metrics."""
    return feedback_service.get_feedback_metrics()

@router.post("/claims/feedback/retrain")
def retrain_model_pipeline(min_feedback_count: int = 1):
    """Trigger controlled model retraining pipeline on logged adjuster feedback."""
    return feedback_service.trigger_retraining(claims_ml_service, min_feedback_count=min_feedback_count)

@router.get("/claims/sample")
def get_sample_claim():
    """Fetch a random historical claim from the loaded 100M sample dataset."""
    df = load_claims_dataset()
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="No claims dataset loaded.")
    sample_row = df.sample(1).iloc[0].to_dict()
    # Clean NaN values
    clean_row = {k: (None if str(v) == "nan" else v) for k, v in sample_row.items()}
    return clean_row
