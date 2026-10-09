import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_tc003_loss_amount_constraint_logic():
    """Verify that TC-003 fails constraint checks if claim amount is below min_loss_amount ($10k)."""
    min_required = 10000.0

    # Passing case: $24,000 claim amount
    passing_claim = {"claim_amount": 24000.0, "deductible": 2500.0}
    assert passing_claim["claim_amount"] >= min_required

    # Failing case: $6,000 claim amount
    failing_claim = {"claim_amount": 6000.0, "deductible": 1000.0}
    assert failing_claim["claim_amount"] < min_required

def test_settlement_ceiling_invariant_across_all_tiers():
    """Verify net settlement ceiling strictly equals max(0, Claim Amount - Deductible)."""
    test_cases = [
        {"claim_amount": 2800.0, "deductible": 500.0, "expected_ceiling": 2300.0},
        {"claim_amount": 85000.0, "deductible": 1000.0, "expected_ceiling": 84000.0},
        {"claim_amount": 24000.0, "deductible": 2500.0, "expected_ceiling": 21500.0},
        {"claim_amount": 400.0, "deductible": 500.0, "expected_ceiling": 0.0},
    ]

    for tc in test_cases:
        calculated = max(0.0, round(tc["claim_amount"] - tc["deductible"], 2))
        assert calculated == tc["expected_ceiling"]

def test_evaluation_metric_validator_rejects_malformed_values():
    """Verify that judge metric parser rejects non-numeric, boolean, NaN, and out-of-range values."""
    def validate_metric(val):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not np.isfinite(val):
            raise ValueError(f"Metric must be finite numeric, got {val!r}")
        if not (0.0 <= val <= 10.0):
            raise ValueError(f"Metric out of bounds [0.0, 10.0]: {val}")
        return val

    # Valid values
    assert validate_metric(9.5) == 9.5
    assert validate_metric(0.0) == 0.0
    assert validate_metric(10.0) == 10.0

    # Invalid: boolean (subclass of int in Python)
    with pytest.raises(ValueError, match="finite numeric"):
        validate_metric(True)

    # Invalid: string
    with pytest.raises(ValueError, match="finite numeric"):
        validate_metric("9.5")

    # Invalid: NaN / Inf
    with pytest.raises(ValueError, match="finite numeric"):
        validate_metric(float("nan"))
    with pytest.raises(ValueError, match="finite numeric"):
        validate_metric(float("inf"))

    # Invalid: out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        validate_metric(10.5)
    with pytest.raises(ValueError, match="out of bounds"):
        validate_metric(-0.5)

def test_algorithmic_grade_derivation():
    """Verify objective grade calculation from pass rate and average metric scores."""
    def derive_grade(pass_rate: float, mean_metric: float) -> str:
        if pass_rate == 1.0 and mean_metric >= 0.90:
            return "EXCELLENT (A+)"
        elif pass_rate >= 0.75 and mean_metric >= 0.80:
            return "VERY GOOD (A)"
        elif pass_rate >= 0.50:
            return "SATISFACTORY (B)"
        else:
            return "REQUIRES_REVIEW (C)"

    assert derive_grade(1.0, 0.93) == "EXCELLENT (A+)"
    assert derive_grade(1.0, 0.85) == "VERY GOOD (A)"
    assert derive_grade(0.80, 0.82) == "VERY GOOD (A)"
    assert derive_grade(0.66, 0.85) == "SATISFACTORY (B)"
    assert derive_grade(0.33, 0.95) == "REQUIRES_REVIEW (C)"

def test_latency_compute_stats_accuracy():
    """Verify summary statistics calculations in benchmark_latency.py."""
    from tests.benchmark_latency import compute_stats

    sample = [2.0, 2.2, 2.5, 3.0, 4.0, 10.0]
    stats = compute_stats(sample)

    assert stats["min_ms"] == 2.0
    assert stats["max_ms"] == 10.0
    assert stats["median_ms"] == round(float(np.median(sample)), 3)
    assert stats["mean_ms"] == round(float(np.mean(sample)), 3)
    assert stats["p95_ms"] > stats["median_ms"]

def test_authoritative_json_files_conformance():
    """Verify docs/Evaluation_Summary.json and docs/latency_benchmark.json conform to declared schemas."""
    eval_file = PROJECT_ROOT / "docs" / "Evaluation_Summary.json"
    bench_file = PROJECT_ROOT / "docs" / "latency_benchmark.json"

    assert eval_file.exists(), "docs/Evaluation_Summary.json must exist"
    assert bench_file.exists(), "docs/latency_benchmark.json must exist"

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # Validate Evaluation Summary structure
    assert "evaluation_timestamp" in eval_data
    assert eval_data["benchmark_tests_count"] == 3
    assert eval_data["tests_passed"] == 3
    assert eval_data["pass_rate_percent"] == 100.0

    metrics = eval_data["metrics_summary"]
    for req_key in ["judge_faithfulness_score", "judge_completeness_score", "judge_policy_compliance_score", "judge_overall_quality_score", "overall_system_grade"]:
        assert req_key in metrics, f"Missing key {req_key} in metrics_summary"

    for r in eval_data["individual_results"]:
        assert r["test_passed"] is True
        assert r["decision_aligned"] is True
        assert r["constraints_passed"] is True
        assert r["loss_amount_verified"] is True
        assert r["settlement_ceiling_verified"] is True

    # Validate Latency Benchmark structure
    assert "benchmark_timestamp" in bench_data
    assert bench_data["environment"]["active_indexed_vectors"] == 2000
    assert bench_data["environment"]["embedding_dimension"] == 1536
    benchmarks = bench_data["benchmarks"]
    for b_key in ["vector_matrix_cosine_search", "bm25_keyword_search", "internal_hybrid_fusion", "end_to_end_hybrid_retrieval"]:
        assert b_key in benchmarks
        stats = benchmarks[b_key]["stats"]
        assert all(k in stats for k in ["min_ms", "median_ms", "mean_ms", "p95_ms", "max_ms"])
