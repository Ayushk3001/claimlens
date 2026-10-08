import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import pandas as pd

from src.utils.data_loader import format_claim_narrative

class KeywordSearchEngine:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_docs: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple, robust regex-based word tokenization."""
        return re.findall(r"\b\w+\b", str(text).lower())

    def index_dataframe(self, df: pd.DataFrame):
        """Build BM25 index from dataframe rows."""
        self.corpus_docs = []
        self.tokenized_corpus = []
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            narrative = format_claim_narrative(row_dict)
            row_dict["narrative"] = narrative
            tokens = self._tokenize(narrative)
            self.corpus_docs.append(row_dict)
            self.tokenized_corpus.append(tokens)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        claim_type: Optional[str] = None,
        state: Optional[str] = None,
        claim_status: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        max_prev_claims: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 search with metadata filtering."""
        if not self.bm25 or not self.corpus_docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            scores = [0.0] * len(self.corpus_docs)
        else:
            scores = self.bm25.get_scores(query_tokens)

        results = []
        for idx, score in enumerate(scores):
            doc = self.corpus_docs[idx]
            
            # Apply metadata filters
            if claim_type and str(doc.get("claim_type", "")).lower() != claim_type.lower():
                continue
            if state and str(doc.get("state", "")).upper() != state.upper():
                continue
            if claim_status and str(doc.get("claim_status", "")).lower() != claim_status.lower():
                continue
            if min_amount is not None and float(doc.get("claim_amount", 0)) < min_amount:
                continue
            if max_amount is not None and float(doc.get("claim_amount", 0)) > max_amount:
                continue
            if max_prev_claims is not None and int(doc.get("previous_claims_count", 0)) > max_prev_claims:
                continue

            doc_copy = dict(doc)
            doc_copy["bm25_score"] = float(score)
            results.append(doc_copy)

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

keyword_search_engine = KeywordSearchEngine()
