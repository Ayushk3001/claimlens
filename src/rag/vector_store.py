import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.data_loader import format_claim_narrative
from src.rag.embeddings import embedding_service

class ClaimVectorStore:
    """High-performance, cross-platform in-memory Vector Database for insurance claims retrieval.
    Computes exact normalized cosine similarity search with rich metadata filtering.
    """
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = str(persist_dir or settings.CHROMA_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = "insurance_claims_100m_sample"
        
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self._id_index: set = set()

    def count(self) -> int:
        return len(self.ids)

    def index_dataframe(self, df: pd.DataFrame, max_records: int = 1500, batch_size: int = 100):
        """Index a subset of dataframe records into vector store with normalized embeddings."""
        if len(self.ids) >= min(len(df), max_records):
            return

        sample_df = df.head(max_records)
        new_ids = []
        new_docs = []
        new_metas = []

        for _, row in sample_df.iterrows():
            row_dict = row.to_dict()
            c_id = str(row_dict["claim_id"])
            if c_id in self._id_index:
                continue

            narrative = format_claim_narrative(row_dict)
            meta = {
                "claim_id": c_id,
                "policy_id": str(row_dict.get("policy_id", "")),
                "claim_type": str(row_dict.get("claim_type", "")),
                "state": str(row_dict.get("state", "")),
                "claim_status": str(row_dict.get("claim_status", "")),
                "claim_amount": float(row_dict.get("claim_amount", 0.0)),
                "deductible": float(row_dict.get("deductible", 0.0)),
                "policyholder_tenure_years": float(row_dict.get("policyholder_tenure_years", 0.0)),
                "previous_claims_count": int(row_dict.get("previous_claims_count", 0)),
                "days_to_resolution": float(row_dict.get("days_to_resolution", -1.0)) if pd.notna(row_dict.get("days_to_resolution")) else -1.0,
                "is_fraud_flagged_ground_truth": bool(row_dict.get("is_fraud_flagged_ground_truth", False))
            }

            new_ids.append(c_id)
            new_docs.append(narrative)
            new_metas.append(meta)
            self._id_index.add(c_id)

        if not new_ids:
            return

        # Generate embeddings in batches
        all_vecs = []
        for i in range(0, len(new_ids), batch_size):
            batch_docs = new_docs[i:i + batch_size]
            batch_embs = embedding_service.get_embeddings(batch_docs)
            all_vecs.extend(batch_embs)

        new_mat = np.array(all_vecs, dtype=np.float32)
        # Ensure row-wise normalization
        norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_mat = new_mat / norms

        if self.embedding_matrix is None or len(self.embedding_matrix) == 0:
            self.embedding_matrix = new_mat
        else:
            self.embedding_matrix = np.vstack([self.embedding_matrix, new_mat])

        self.ids.extend(new_ids)
        self.documents.extend(new_docs)
        self.metadatas.extend(new_metas)

    def search(
        self,
        query: str,
        top_k: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic cosine similarity search with metadata filtering."""
        if self.count() == 0 or self.embedding_matrix is None:
            return []

        # Generate normalized query vector
        query_vec = np.array(embedding_service.get_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        # Cosine similarity matrix multiplication: (N, D) @ (D,) -> (N,)
        cosine_scores = np.dot(self.embedding_matrix, query_vec)

        # Apply where_filter and sort
        scored_candidates = []
        for idx, score in enumerate(cosine_scores):
            meta = self.metadatas[idx]

            # Check metadata filters
            if where_filter:
                match = True
                for k, v in where_filter.items():
                    val = meta.get(k)
                    if isinstance(v, str) and isinstance(val, str):
                        if val.lower() != v.lower():
                            match = False
                            break
                    elif val != v:
                        match = False
                        break
                if not match:
                    continue

            # Normalized cosine similarity mapped to [0.0, 1.0]
            sim = float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
            item = dict(meta)
            item["narrative"] = self.documents[idx]
            item["vector_similarity"] = round(sim, 4)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["vector_similarity"], reverse=True)
        return scored_candidates[:top_k]

claim_vector_store = ClaimVectorStore()
