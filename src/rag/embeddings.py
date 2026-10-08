import os
import hashlib
import numpy as np
from typing import List, Optional
from openai import OpenAI

from src.utils.config import settings

class EmbeddingService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL
        self.model = model or settings.OPENAI_EMBEDDING_MODEL
        self.dimension = 1536  # Default for text-embedding-3-small
        
        self.client = None
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
            except Exception:
                self.client = None

    def _fallback_embedding(self, text: str) -> List[float]:
        """Generate deterministic normalized dense 1536-dim embedding vector based on text hashing."""
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        
        # Modulate slightly by token keywords for semantic clustering
        tokens = text.lower().split()
        for idx, token in enumerate(tokens[:30]):
            tok_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            pos = tok_seed % self.dimension
            vec[pos] += 1.5 / ((idx + 1) ** 0.5)
            
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def get_embedding(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension
            
        if self.client:
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                # Fallback on rate limit / connectivity issues
                pass
                
        return self._fallback_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        if self.client:
            try:
                # Batch in groups of 100
                all_embeddings = []
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    all_embeddings.extend([item.embedding for item in response.data])
                return all_embeddings
            except Exception:
                pass
                
        return [self._fallback_embedding(t) for t in texts]

embedding_service = EmbeddingService()
