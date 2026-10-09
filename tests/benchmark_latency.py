import os
import sys
import time
import json
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.vector_store import claim_vector_store
from src.rag.keyword_search import keyword_search_engine
from src.rag.hybrid_retriever import hybrid_retriever
from src.rag.embeddings import embedding_service

def compute_stats(arr: List[float]) -> Dict[str, float]:
    """Calculate standard summary latency statistics in milliseconds."""
    return {
        "min_ms": round(float(np.min(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(np.max(arr)), 3)
    }

def run_latency_benchmarks(
    matrix_runs: int = 200,
    warmup_runs: int = 25,
    api_e2e_runs: int = 15
) -> Dict[str, Any]:
    """Run comprehensive, empirical latency benchmarks across all retrieval subsystems.
    
    Strictly separates:
    1. Isolated in-memory vector matrix search (search_by_vector) without network overhead
    2. Local fallback pseudo-embedding generation (hash-based)
    3. BM25Okapi lexical retrieval
    4. Internal Hybrid RRF Fusion ranking
    5. End-to-End Hybrid Retrieval (including live/fallback embedding service)
    """
    if not hybrid_retriever.is_initialized:
        print("Initializing ClaimLens Hybrid RAG engines...")
        hybrid_retriever.initialize(max_records=2000)

    dim = claim_vector_store.dimension
    n_records = claim_vector_store.count()
    if n_records == 0:
        raise RuntimeError("Vector store has zero indexed records. Benchmark cannot proceed.")

    print("=" * 75)
    print("CLAIMLENS EMPIRICAL LATENCY BENCHMARK SUITE")
    print(f"Matrix Iterations: {matrix_runs} (Warm-up: {warmup_runs}) | Indexed Vectors: {n_records}")
    print(f"Embedding Dimension: {dim} | Environment: {platform.system()} ({platform.machine()})")
    print("=" * 75)

    test_queries = [
        "Rear bumper collision in parking lot minor dent paint scrape",
        "Total business inventory destroyed by electrical fire arson",
        "Upstairs plumbing pipe burst damaging hardwood floors ceiling",
        "Hail storm vehicle body damage windshield shattered",
        "Kitchen grease fire cabinet structural damage smoke remediation"
    ]

    # Pre-generate valid, normalized 1536-dimensional query vectors for isolated matrix testing
    rng = np.random.RandomState(42)
    sample_query_vecs = []
    for _ in range(matrix_runs + warmup_runs):
        raw = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(raw)
        sample_query_vecs.append((raw / norm).tolist())

    # --------------------------------------------------------------------------
    # 1. Isolated Vector Matrix Cosine Search (search_by_vector)
    # Measures purely the internal (N, D) @ (D,) matrix dot product & top-k ranking
    # --------------------------------------------------------------------------
    print("\n[1/5] Benchmarking isolated vector matrix cosine similarity search...")
    # Warm-up phase
    for i in range(warmup_runs):
        _ = claim_vector_store.search_by_vector(sample_query_vecs[i], top_k=5)

    vector_latencies_ms = []
    for i in range(warmup_runs, warmup_runs + matrix_runs):
        qvec = sample_query_vecs[i]
        t0 = time.perf_counter()
        _ = claim_vector_store.search_by_vector(qvec, top_k=5)
        t1 = time.perf_counter()
        vector_latencies_ms.append((t1 - t0) * 1000.0)

    # --------------------------------------------------------------------------
    # 2. Local Fallback Embedding Generation Latency (Offline Hash-Based)
    # --------------------------------------------------------------------------
    print("[2/5] Benchmarking local hash-based fallback embedding generation...")
    for i in range(warmup_runs):
        _ = embedding_service._fallback_embedding(test_queries[i % len(test_queries)])

    fallback_latencies_ms = []
    for i in range(matrix_runs):
        qtext = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        _ = embedding_service._fallback_embedding(qtext)
        t1 = time.perf_counter()
        fallback_latencies_ms.append((t1 - t0) * 1000.0)

    # --------------------------------------------------------------------------
    # 3. BM25Okapi Keyword Search Latency
    # --------------------------------------------------------------------------
    print("[3/5] Benchmarking BM25Okapi keyword search...")
    for i in range(warmup_runs):
        _ = keyword_search_engine.search(test_queries[i % len(test_queries)], top_k=5)

    bm25_latencies_ms = []
    for i in range(matrix_runs):
        qtext = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        _ = keyword_search_engine.search(qtext, top_k=5)
        t1 = time.perf_counter()
        bm25_latencies_ms.append((t1 - t0) * 1000.0)

    # --------------------------------------------------------------------------
    # 4. Internal Hybrid RRF Fusion Latency (Precomputed Vector + BM25, no network)
    # --------------------------------------------------------------------------
    print("[4/5] Benchmarking internal hybrid RRF score fusion...")
    fusion_latencies_ms = []
    for i in range(matrix_runs):
        qtext = test_queries[i % len(test_queries)]
        qvec = sample_query_vecs[i + warmup_runs]
        t0 = time.perf_counter()
        v_res = claim_vector_store.search_by_vector(qvec, top_k=10)
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

    # --------------------------------------------------------------------------
    # 5. End-to-End Hybrid Search Latency (hybrid_retriever.search via live/mock)
    # --------------------------------------------------------------------------
    print(f"[5/5] Benchmarking end-to-end hybrid retrieval ({api_e2e_runs} iterations)...")
    # 2 warm-up iterations
    for i in range(min(2, api_e2e_runs)):
        _ = hybrid_retriever.search(query=test_queries[i % len(test_queries)], top_k=5)

    e2e_latencies_ms = []
    for i in range(api_e2e_runs):
        qtext = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        _ = hybrid_retriever.search(query=qtext, top_k=5)
        t1 = time.perf_counter()
        e2e_latencies_ms.append((t1 - t0) * 1000.0)

    is_live_api = bool(embedding_service.client is not None)

    results = {
        "benchmark_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methodology": "Empirical multi-iteration latency measurement with warm-up exclusion and isolated matrix math",
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "processor": platform.processor(),
            "active_indexed_vectors": n_records,
            "embedding_dimension": dim,
            "live_embedding_api_active": is_live_api
        },
        "benchmarks": {
            "vector_matrix_cosine_search": {
                "operation": "In-memory normalized NumPy cosine matrix dot product ((N, D) @ (D,)) + top-k partition via search_by_vector",
                "warmup_runs": warmup_runs,
                "measured_iterations": matrix_runs,
                "stats": compute_stats(vector_latencies_ms)
            },
            "local_fallback_embedding_generation": {
                "operation": "Deterministic SHA256/MD5 text hash pseudo-embedding generation (local CPU)",
                "warmup_runs": warmup_runs,
                "measured_iterations": matrix_runs,
                "stats": compute_stats(fallback_latencies_ms)
            },
            "bm25_keyword_search": {
                "operation": "Rank-BM25 tokenization and scoring across claim narrative corpus",
                "warmup_runs": warmup_runs,
                "measured_iterations": matrix_runs,
                "stats": compute_stats(bm25_latencies_ms)
            },
            "internal_hybrid_fusion": {
                "operation": "Vector search_by_vector + BM25 score merge via Reciprocal Rank Fusion (excluding network)",
                "warmup_runs": warmup_runs,
                "measured_iterations": matrix_runs,
                "stats": compute_stats(fusion_latencies_ms)
            },
            "end_to_end_hybrid_retrieval": {
                "operation": "Full pipeline: Natural language query string -> Embedding generation -> Vector Search + BM25 -> RRF top-5",
                "warmup_runs": 2,
                "measured_iterations": api_e2e_runs,
                "stats": compute_stats(e2e_latencies_ms)
            }
        }
    }

    v_stat = results["benchmarks"]["vector_matrix_cosine_search"]["stats"]
    f_stat = results["benchmarks"]["local_fallback_embedding_generation"]["stats"]
    b_stat = results["benchmarks"]["bm25_keyword_search"]["stats"]
    h_stat = results["benchmarks"]["internal_hybrid_fusion"]["stats"]
    e_stat = results["benchmarks"]["end_to_end_hybrid_retrieval"]["stats"]

    print("\n" + "=" * 78)
    print("EMPIRICAL LATENCY BENCHMARK RESULTS (IN MILLISECONDS):")
    print(f"{'Component / Subsystem':<35} | {'Min':>7} | {'Median':>7} | {'Mean':>7} | {'P95':>7} | {'Max':>7}")
    print("-" * 78)
    print(f"{'NumPy Cosine Matrix Search':<35} | {v_stat['min_ms']:>6.2f}ms | {v_stat['median_ms']:>6.2f}ms | {v_stat['mean_ms']:>6.2f}ms | {v_stat['p95_ms']:>6.2f}ms | {v_stat['max_ms']:>6.2f}ms")
    print(f"{'Local Fallback Embedding (CPU)':<35} | {f_stat['min_ms']:>6.2f}ms | {f_stat['median_ms']:>6.2f}ms | {f_stat['mean_ms']:>6.2f}ms | {f_stat['p95_ms']:>6.2f}ms | {f_stat['max_ms']:>6.2f}ms")
    print(f"{'BM25Okapi Keyword Search':<35} | {b_stat['min_ms']:>6.2f}ms | {b_stat['median_ms']:>6.2f}ms | {b_stat['mean_ms']:>6.2f}ms | {b_stat['p95_ms']:>6.2f}ms | {b_stat['max_ms']:>6.2f}ms")
    print(f"{'Internal Hybrid RRF Fusion':<35} | {h_stat['min_ms']:>6.2f}ms | {h_stat['median_ms']:>6.2f}ms | {h_stat['mean_ms']:>6.2f}ms | {h_stat['p95_ms']:>6.2f}ms | {h_stat['max_ms']:>6.2f}ms")
    print(f"{'End-to-End Hybrid Search':<35} | {e_stat['min_ms']:>6.2f}ms | {e_stat['median_ms']:>6.2f}ms | {e_stat['mean_ms']:>6.2f}ms | {e_stat['p95_ms']:>6.2f}ms | {e_stat['max_ms']:>6.2f}ms")
    print("=" * 78)
    print(f"System: {platform.platform()} | Python: {platform.python_version()} | Indexed Vectors: {n_records}")
    print(f"Embedding Provider: {'Live External API (OpenAI)' if is_live_api else 'Deterministic Local Fallback'}")
    print("=" * 78)

    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_file = docs_dir / "latency_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEmpirical report saved to: {out_file}")

    return results

if __name__ == "__main__":
    run_latency_benchmarks()
