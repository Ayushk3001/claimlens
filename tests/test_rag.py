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
