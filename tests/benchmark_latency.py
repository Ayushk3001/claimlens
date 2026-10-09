import os
import sys
import time
import json
import platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.vector_store import claim_vector_store
from src.rag.keyword_search import keyword_search_engine
from src.rag.hybrid_retriever import hybrid_retriever
from src.rag.embeddings import embedding_service

def run_latency_benchmarks(num_runs: int = 200) -> dict:
    # Ensure RAG engines are initialized
    if not hybrid_retriever.is_initialized:
        print("Initializing ClaimLens Hybrid RAG engines...")
        hybrid_retriever.initialize()

    dim = claim_vector_store.embedding_matrix.shape[1] if claim_vector_store.embedding_matrix is not None else 1536
    n_records = len(claim_vector_store.ids)

    print("=" * 70)
    print("CLAIMLENS EMPIRICAL LATENCY BENCHMARK SUITE")
    print(f"Sample Size: {num_runs} iterations | Active Indexed Vectors: {n_records}")
    print(f"Vector Dimensions: {dim} | Environment: {platform.system()} ({platform.machine()})")
    print("=" * 70)

    test_queries = [
        "Rear bumper collision in parking lot minor dent paint scrape",
        "Total business inventory destroyed by electrical fire arson",
        "Upstairs plumbing pipe burst damaging hardwood floors ceiling",
        "Hail storm vehicle body damage windshield shattered",
        "Kitchen grease fire cabinet structural damage smoke remediation"
    ]
    
    # 1. Exact Vector Matrix Cosine Similarity Latency
    # Measures raw in-memory matrix multiplication (N, D) @ (D,) + top-k extraction
    vector_latencies_ms = []
    rng = np.random.RandomState(42)
    sample_query_vecs = [
        (rng.randn(dim) / np.linalg.norm(rng.randn(dim))).tolist()
        for _ in range(num_runs)
    ]

    for qvec in sample_query_vecs:
        t0 = time.perf_counter()
        _ = claim_vector_store.search(qvec, top_k=5)
        t1 = time.perf_counter()
        vector_latencies_ms.append((t1 - t0) * 1000.0)

    # 2. BM25Okapi Keyword Search Latency
    bm25_latencies_ms = []
    for i in range(num_runs):
        qtext = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        _ = keyword_search_engine.search(qtext, top_k=5)
        t1 = time.perf_counter()
        bm25_latencies_ms.append((t1 - t0) * 1000.0)

    # 3. Hybrid RRF Fusion Latency (Pre-computed query vector, internal algorithm isolation)
    fusion_latencies_ms = []
    for i in range(num_runs):
        qtext = test_queries[i % len(test_queries)]
        qvec = sample_query_vecs[i]
        t0 = time.perf_counter()
        v_res = claim_vector_store.search(qvec, top_k=10)
        k_res = keyword_search_engine.search(qtext, top_k=10)
        rrf_scores = {}
        for rk, item in enumerate(v_res):
            cid = item["claim_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 0.6 * (1.0 / (60 + rk + 1))
        for rk, item in enumerate(k_res):
            cid = item["claim_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 0.4 * (1.0 / (60 + rk + 1))
        _ = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        t1 = time.perf_counter()
        fusion_latencies_ms.append((t1 - t0) * 1000.0)

    # 4. End-to-End Hybrid Search Latency (including embedding retrieval)
    e2e_latencies_ms = []
    for i in range(15):
        qtext = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        _ = hybrid_retriever.search(query=qtext, top_k=5)
        t1 = time.perf_counter()
        e2e_latencies_ms.append((t1 - t0) * 1000.0)

    def compute_stats(arr):
        return {
            "min_ms": round(float(np.min(arr)), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "max_ms": round(float(np.max(arr)), 3)
        }

    results = {
        "benchmark_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "processor": platform.processor(),
            "active_indexed_vectors": n_records,
            "embedding_dimension": dim
        },
        "benchmarks": {
            "vector_matrix_cosine_search": {
                "description": "In-memory normalized NumPy cosine matrix dot product ((N, D) @ (D,)) + top-k partition",
                "iterations": num_runs,
                "stats": compute_stats(vector_latencies_ms)
            },
            "bm25_keyword_search": {
                "description": "Rank-BM25 tokenization and scoring across claim narrative corpus",
                "iterations": num_runs,
                "stats": compute_stats(bm25_latencies_ms)
            },
            "internal_hybrid_fusion": {
                "description": "Vector dot product + BM25 score merge via Reciprocal Rank Fusion (excluding network)",
                "iterations": num_runs,
                "stats": compute_stats(fusion_latencies_ms)
            },
            "end_to_end_hybrid_retrieval": {
                "description": "Full pipeline: Query string -> Embedding Service -> Vector Search + BM25 -> RRF top-5",
                "iterations": len(e2e_latencies_ms),
                "stats": compute_stats(e2e_latencies_ms)
            }
        }
    }

    # Print clean formatted summary
    v_stat = results["benchmarks"]["vector_matrix_cosine_search"]["stats"]
    b_stat = results["benchmarks"]["bm25_keyword_search"]["stats"]
    h_stat = results["benchmarks"]["internal_hybrid_fusion"]["stats"]
    e_stat = results["benchmarks"]["end_to_end_hybrid_retrieval"]["stats"]

    print("\nBENCHMARK RESULTS (LATENCY IN MILLISECONDS):")
    print(f"{'Component':<32} | {'Min':>7} | {'Median':>7} | {'Mean':>7} | {'P95':>7} | {'Max':>7}")
    print("-" * 75)
    print(f"{'NumPy Vector Cosine Search':<32} | {v_stat['min_ms']:>6.2f}ms | {v_stat['median_ms']:>6.2f}ms | {v_stat['mean_ms']:>6.2f}ms | {v_stat['p95_ms']:>6.2f}ms | {v_stat['max_ms']:>6.2f}ms")
    print(f"{'BM25Okapi Keyword Search':<32} | {b_stat['min_ms']:>6.2f}ms | {b_stat['median_ms']:>6.2f}ms | {b_stat['mean_ms']:>6.2f}ms | {b_stat['p95_ms']:>6.2f}ms | {b_stat['max_ms']:>6.2f}ms")
    print(f"{'Internal Hybrid RRF Fusion':<32} | {h_stat['min_ms']:>6.2f}ms | {h_stat['median_ms']:>6.2f}ms | {h_stat['mean_ms']:>6.2f}ms | {h_stat['p95_ms']:>6.2f}ms | {h_stat['max_ms']:>6.2f}ms")
    print(f"{'End-to-End Hybrid Search (API)':<32} | {e_stat['min_ms']:>6.2f}ms | {e_stat['median_ms']:>6.2f}ms | {e_stat['mean_ms']:>6.2f}ms | {e_stat['p95_ms']:>6.2f}ms | {e_stat['max_ms']:>6.2f}ms")
    print("=" * 75)
    print(f"System: {platform.platform()} | Python: {platform.python_version()} | Indexed Vectors: {n_records}")
    print("=" * 75)

    # Save to docs/latency_benchmark.json
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_file = docs_dir / "latency_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical benchmark report to: {out_file}")

    return results

if __name__ == "__main__":
    run_latency_benchmarks()
