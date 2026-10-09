import pytest
from src.rag.hybrid_retriever import hybrid_retriever
from src.rag.keyword_search import keyword_search_engine
from src.rag.embeddings import embedding_service

def test_embedding_service_dimensions():
    vec = embedding_service.get_embedding("Auto insurance rear-end collision claim")
    assert isinstance(vec, list)
    assert len(vec) == 1536

def test_hybrid_search_retrieval():
    hybrid_retriever.initialize(max_records=500)
    results = hybrid_retriever.search(
        query="Auto collision damage claim",
        top_k=3,
        claim_type="Auto"
    )
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        assert "claim_id" in results[0]
        assert "hybrid_score" in results[0]
        assert results[0]["claim_type"].lower() == "auto"

def test_keyword_search_with_state_filter():
    results = keyword_search_engine.search(
        query="property damage claim",
        top_k=5,
        state="CA"
    )
    assert isinstance(results, list)
    for r in results:
        assert r["state"].upper() == "CA"

def test_hybrid_search_metadata_amount_and_prev_claims_filtering():
    hybrid_retriever.initialize(max_records=500)
    # Test valid inclusion matching
    results = hybrid_retriever.search(
        query="damage repair loss claim",
        top_k=5,
        min_amount=500.0,
        max_amount=50000.0,
        max_prev_claims=3
    )
    assert isinstance(results, list)
    assert len(results) > 0, "Expected non-empty result set for standard bounds"
    for r in results:
        amt = float(r["claim_amount"])
        assert 500.0 <= amt <= 50000.0, f"Claim amount {amt} violated [500, 50000] bounds"
        prev_count = int(r["previous_claims_count"])
        assert prev_count <= 3, f"Prior claims count {prev_count} exceeded limit of 3"

    # Test strict exclusion: impossibly low maximum amount excludes all records
    excluded_results = hybrid_retriever.search(
        query="severe loss claim",
        top_k=5,
        min_amount=1000000.0  # $1,000,000 floor exceeds all sample claims
    )
    assert len(excluded_results) == 0, "Expected zero results for $1M min_amount exclusion test"

def test_deterministic_fallback_embedding_behavior():
    """Verify that deterministic fallback embeddings produce unit-normalized vectors with reproducible values."""
    import numpy as np

    text_a = "Rear bumper collision in parking lot minor dent paint scrape"
    text_b = "Total commercial warehouse fire loss suspected electrical failure"

    vec_a1 = embedding_service._fallback_embedding(text_a)
    vec_a2 = embedding_service._fallback_embedding(text_a)
    vec_b = embedding_service._fallback_embedding(text_b)

    # 1. Determinism assertion: identical input yields exact bit-level identical vector
    assert vec_a1 == vec_a2, "Fallback embedding must be completely deterministic for identical text"

    # 2. Dimensionality assertion
    assert len(vec_a1) == 1536
    assert len(vec_b) == 1536

    # 3. Unit L2 norm assertion (normalized cosine vector)
    norm_a = float(np.linalg.norm(vec_a1))
    norm_b = float(np.linalg.norm(vec_b))
    assert abs(norm_a - 1.0) < 1e-4, f"Fallback vector norm {norm_a} is not unit normalized"
    assert abs(norm_b - 1.0) < 1e-4, f"Fallback vector norm {norm_b} is not unit normalized"

    # 4. Non-trivial distinction between different texts
    dot_prod = float(np.dot(vec_a1, vec_b))
    assert dot_prod < 0.99, f"Different text inputs produced unexpectedly identical vectors (cosine sim: {dot_prod})"

