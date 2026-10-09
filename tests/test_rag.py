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

def test_vector_store_dimensionality_and_numeric_validation():
    """Verify that search_by_vector strictly validates vector shape, finiteness, and dimensionality."""
    import numpy as np
    from src.rag.vector_store import claim_vector_store

    # Test rejection of non-vector type
    with pytest.raises(TypeError):
        claim_vector_store.search_by_vector("invalid-string-vector")

    # Test rejection of wrong dimension (e.g. 100 instead of 1536)
    with pytest.raises(ValueError, match="dimension mismatch"):
        claim_vector_store.search_by_vector([0.1] * 100)

    # Test rejection of 2D array
    with pytest.raises(ValueError, match="1-dimensional"):
        claim_vector_store.search_by_vector(np.zeros((1, 1536)))

    # Test rejection of NaN values
    nan_vec = [0.0] * 1536
    nan_vec[0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite"):
        claim_vector_store.search_by_vector(nan_vec)

    # Test rejection of zero-magnitude vector
    zero_vec = [0.0] * 1536
    with pytest.raises(ValueError, match="zero-magnitude"):
        claim_vector_store.search_by_vector(zero_vec)

    # Test search() rejects non-string query
    with pytest.raises(TypeError, match="query must be a string"):
        claim_vector_store.search([0.1] * 1536)

def test_deterministic_vector_search_with_known_vector():
    """Verify vector search returns exact known matches and scores using an isolated ClaimVectorStore."""
    import numpy as np
    from src.rag.vector_store import ClaimVectorStore

    test_store = ClaimVectorStore()
    dim = test_store.dimension

    # Construct 3 orthogonal unit vectors in 1536 dimensions
    v0 = np.zeros(dim, dtype=np.float32); v0[0] = 1.0
    v1 = np.zeros(dim, dtype=np.float32); v1[1] = 1.0
    v2 = np.zeros(dim, dtype=np.float32); v2[2] = 1.0

    test_store.ids = ["ID-0", "ID-1", "ID-2"]
    test_store.documents = ["Doc 0", "Doc 1", "Doc 2"]
    test_store.metadatas = [
        {"claim_id": "CLM-000", "claim_type": "Auto"},
        {"claim_id": "CLM-001", "claim_type": "Home"},
        {"claim_id": "CLM-002", "claim_type": "Auto"}
    ]
    test_store.embedding_matrix = np.vstack([v0, v1, v2])

    # Search with v0: top candidate must be ID-0 with perfect similarity
    results = test_store.search_by_vector(v0, top_k=2)
    assert len(results) == 2
    assert results[0]["claim_id"] == "CLM-000"
    assert results[0]["vector_similarity"] == 1.0  # (1.0 + 1.0) / 2.0 = 1.0

    # Search with metadata filter claim_type="Auto": should match CLM-000 and CLM-002, excluding CLM-001
    filtered = test_store.search_by_vector(v1, top_k=3, where_filter={"claim_type": "Auto"})
    assert all(r["claim_type"] == "Auto" for r in filtered)
    assert not any(r["claim_id"] == "CLM-001" for r in filtered)


