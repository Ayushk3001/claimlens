from typing import List, Dict, Any, Optional
import pandas as pd

from src.rag.vector_store import claim_vector_store
from src.rag.keyword_search import keyword_search_engine
from src.utils.data_loader import load_claims_dataset

class HybridRetriever:
    def __init__(self):
        self.vector_store = claim_vector_store
        self.keyword_search = keyword_search_engine
        self.is_initialized = False

    def initialize(self, max_records: int = 2000):
        """Pre-indexes data into both Vector Store and BM25 search engine."""
        df = load_claims_dataset()
        self.keyword_search.index_dataframe(df)
        self.vector_store.index_dataframe(df, max_records=max_records)
        self.is_initialized = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        claim_type: Optional[str] = None,
        state: Optional[str] = None,
        claim_status: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        max_prev_claims: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            self.initialize()

        # Build vector store metadata filter where possible
        metadata_filter = {}
        if claim_type:
            metadata_filter["claim_type"] = claim_type.capitalize()
        if state:
            metadata_filter["state"] = state.upper()
        if claim_status:
            metadata_filter["claim_status"] = claim_status.capitalize()

        # Retrieve top candidates from both search methods
        candidate_k = max(top_k * 3, 20)
        semantic_results = self.vector_store.search(
            query=query,
            top_k=candidate_k,
            where_filter=metadata_filter if metadata_filter else None
        )
        
        keyword_results = self.keyword_search.search(
            query=query,
            top_k=candidate_k,
            claim_type=claim_type,
            state=state,
            claim_status=claim_status,
            min_amount=min_amount,
            max_amount=max_amount,
            max_prev_claims=max_prev_claims
        )

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}
        source_map: Dict[str, set] = {}

        for rank, item in enumerate(semantic_results):
            cid = item["claim_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (semantic_weight / (60.0 + rank + 1))
            item_map[cid] = item
            source_map.setdefault(cid, set()).add("Semantic")

        for rank, item in enumerate(keyword_results):
            cid = item["claim_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (keyword_weight / (60.0 + rank + 1))
            if cid not in item_map:
                item_map[cid] = item
            source_map.setdefault(cid, set()).add("Keyword")

        # Combine, score, and sort
        combined = []
        max_possible_rrf = (semantic_weight / 61.0) + (keyword_weight / 61.0)  # max theoretical top rank in both (~0.01639)
        
        for cid, score in rrf_scores.items():
            doc = dict(item_map[cid])
            sources = source_map.get(cid, set())
            doc["retrieval_method"] = "Hybrid" if len(sources) > 1 else list(sources)[0]
            doc["rrf_score"] = round(score, 6)
            # Continuous intuitive similarity mapping [0.45, 0.98] preserving relative rank separation
            normalized_ratio = min(1.0, score / max_possible_rrf)
            doc["hybrid_score"] = round(0.45 + 0.53 * normalized_ratio, 4)
            combined.append(doc)

        # Apply strict filtering across all combined candidates to guarantee consistency
        filtered_combined = []
        for doc in combined:
            if claim_type and str(doc.get("claim_type", "")).lower() != claim_type.lower():
                continue
            if state and str(doc.get("state", "")).upper() != state.upper():
                continue
            if claim_status and str(doc.get("claim_status", "")).lower() != claim_status.lower():
                continue
            amt = float(doc.get("claim_amount", 0.0))
            if min_amount is not None and amt < min_amount:
                continue
            if max_amount is not None and amt > max_amount:
                continue
            if max_prev_claims is not None and int(doc.get("previous_claims_count", 0)) > max_prev_claims:
                continue
            filtered_combined.append(doc)

        filtered_combined.sort(key=lambda x: (x["hybrid_score"], x.get("vector_similarity", 0.0), x.get("bm25_score", 0.0)), reverse=True)
        return filtered_combined[:top_k]

hybrid_retriever = HybridRetriever()
