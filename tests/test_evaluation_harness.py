import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from tests.run_evaluation import (
    compute_expected_net_settlement_ceiling,
    verify_net_settlement_ceiling,
    verify_benchmark_constraints,
    validate_judge_metrics,
    calculate_test_passed,
    derive_overall_grade,
    EVAL_BENCHMARKS
)
from scripts.generate_all_deliverables import (
    load_authoritative_eval_summary,
    load_authoritative_latency_benchmark
)


# ==============================================================================
# 1. PRODUCTION EVALUATION LOGIC TESTS
# ==============================================================================

def test_production_settlement_ceiling_functions():
    """Verify production settlement ceiling functions correctly enforce the invariant max(0, claim - ded)."""
    # Standard auto claim
    expected = compute_expected_net_settlement_ceiling(claim_amount=2800.0, deductible=500.0)
    assert expected == 2300.0
    assert verify_net_settlement_ceiling(actual_ceiling=2300.0, claim_amount=2800.0, deductible=500.0) is True

    # High-value business claim
    expected_high = compute_expected_net_settlement_ceiling(claim_amount=85000.0, deductible=1000.0)
    assert expected_high == 84000.0
    assert verify_net_settlement_ceiling(actual_ceiling=84000.0, claim_amount=85000.0, deductible=1000.0) is True

    # Claim amount within deductible: ceiling must be strictly 0.0
    expected_zero = compute_expected_net_settlement_ceiling(claim_amount=400.0, deductible=500.0)
    assert expected_zero == 0.0
    assert verify_net_settlement_ceiling(actual_ceiling=0.0, claim_amount=400.0, deductible=500.0) is True

    # Rejection of invalid / exceeded ceiling
    assert verify_net_settlement_ceiling(actual_ceiling=2800.0, claim_amount=2800.0, deductible=500.0) is False
    assert verify_net_settlement_ceiling(actual_ceiling=-1.0, claim_amount=2800.0, deductible=500.0) is False


def test_production_tc003_minimum_loss_constraint():
    """Verify production verify_benchmark_constraints enforces TC-003 min_loss_amount ($10k)."""
    tc003 = next(tc for tc in EVAL_BENCHMARKS if tc["test_id"] == "TC-003")
    assert tc003["min_loss_amount"] == 10000.0

    # Passing case: $24,000 loss
    valid_inp = dict(tc003["input"])
    valid_rec = {"decision": "MANUAL_ADJUSTER_REVIEW", "net_settlement_ceiling": 21500.0}
    valid_risk = {"fraud_probability_percent": 20.0}
    valid_ml = {"calibrated_fraud_probability_percent": 12.0}

    passed, notes = verify_benchmark_constraints(tc003, valid_inp, valid_rec, valid_risk, valid_ml)
    assert passed is True
    assert len(notes) == 0

    # Failing case: Claim amount $8,000 (below $10,000 minimum)
    failing_inp = dict(tc003["input"])
    failing_inp["claim_amount"] = 8000.0
    failing_rec = {"decision": "MANUAL_ADJUSTER_REVIEW", "net_settlement_ceiling": 5500.0}

    passed_fail, notes_fail = verify_benchmark_constraints(tc003, failing_inp, failing_rec, valid_risk, valid_ml)
    assert passed_fail is False
    assert any("min_loss_amount" in n for n in notes_fail)


def test_production_constraints_detect_ceiling_violation():
    """Verify production verify_benchmark_constraints rejects violated net settlement ceilings."""
    tc001 = next(tc for tc in EVAL_BENCHMARKS if tc["test_id"] == "TC-001")
    inp = dict(tc001["input"])  # claim 2800, ded 500 -> expected ceiling 2300
    risk = {"fraud_probability_percent": 25.0}
    ml = {"calibrated_fraud_probability_percent": 6.1}

    # Rec proposes illegal ceiling 2800 (failed to deduct deductible)
    bad_rec = {"decision": "AUTO_APPROVE", "net_settlement_ceiling": 2800.0}
    passed, notes = verify_benchmark_constraints(tc001, inp, bad_rec, risk, ml)
    assert passed is False
    assert any("Net settlement ceiling" in n for n in notes)


def test_production_validate_judge_metrics_accepts_valid():
    """Verify production validate_judge_metrics accepts valid scores and normalizes to [0, 1]."""
    valid_judge = {
        "verdict": "PASS",
        "overall_quality_score": 9.3,
        "metric_scores": {
            "factual_consistency": 9.5,
            "completeness": 9.0,
            "policy_compliance": 9.5
        }
    }
    norm = validate_judge_metrics(valid_judge, test_id="TC-VALID")
    assert norm["judge_faithfulness_score"] == 0.95
    assert norm["judge_completeness_score"] == 0.90
    assert norm["judge_policy_compliance_score"] == 0.95
    assert norm["judge_overall_quality_score"] == 0.93


def test_production_validate_judge_metrics_rejects_malformed_and_out_of_bounds():
    """Verify production validate_judge_metrics rejects missing, non-numeric, boolean, NaN, and out-of-range metrics."""
    # 1. Non-dict input
    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_judge_metrics("invalid_string")

    # 2. Missing metric_scores dict
    with pytest.raises(ValueError, match="missing valid 'metric_scores'"):
        validate_judge_metrics({"verdict": "PASS", "overall_quality_score": 9.0})

    # 3. Missing required metric
    incomplete_judge = {
        "overall_quality_score": 9.0,
        "metric_scores": {"factual_consistency": 9.0, "completeness": 9.0}
    }
    with pytest.raises(ValueError, match="missing required metric 'policy_compliance'"):
        validate_judge_metrics(incomplete_judge)

    # 4. Boolean value (must not be treated as int 1)
    bool_judge = {
        "overall_quality_score": 9.0,
        "metric_scores": {"factual_consistency": True, "completeness": 9.0, "policy_compliance": 9.0}
    }
    with pytest.raises(ValueError, match="must be finite numeric"):
        validate_judge_metrics(bool_judge)

    # 5. String value
    str_judge = {
        "overall_quality_score": 9.0,
        "metric_scores": {"factual_consistency": "9.5", "completeness": 9.0, "policy_compliance": 9.0}
    }
    with pytest.raises(ValueError, match="must be finite numeric"):
        validate_judge_metrics(str_judge)

    # 6. NaN / Inf
    nan_judge = {
        "overall_quality_score": 9.0,
        "metric_scores": {"factual_consistency": float("nan"), "completeness": 9.0, "policy_compliance": 9.0}
    }
    with pytest.raises(ValueError, match="must be finite numeric"):
        validate_judge_metrics(nan_judge)

    # 7. Out of bounds (> 10.0 or < 0.0)
    high_judge = {
        "overall_quality_score": 9.0,
        "metric_scores": {"factual_consistency": 11.5, "completeness": 9.0, "policy_compliance": 9.0}
    }
    with pytest.raises(ValueError, match="out of bounds"):
        validate_judge_metrics(high_judge)

    neg_judge = {
        "overall_quality_score": 9.0,
        "metric_scores": {"factual_consistency": -1.0, "completeness": 9.0, "policy_compliance": 9.0}
    }
    with pytest.raises(ValueError, match="out of bounds"):
        validate_judge_metrics(neg_judge)


def test_production_calculate_test_passed_requires_all_gates():
    """Verify calculate_test_passed passes if and only if decision aligns, constraints pass, and judge passes."""
    good_judge = {"verdict": "PASS", "overall_quality_score": 9.0}
    flagged_judge = {"verdict": "FLAGGED_FOR_AUDIT", "overall_quality_score": 9.0}
    low_score_judge = {"verdict": "PASS", "overall_quality_score": 7.5}

    # All pass
    assert calculate_test_passed(decision_aligned=True, constraints_passed=True, judge=good_judge) is True

    # Failed decision alignment
    assert calculate_test_passed(decision_aligned=False, constraints_passed=True, judge=good_judge) is False

    # Failed constraints
    assert calculate_test_passed(decision_aligned=True, constraints_passed=False, judge=good_judge) is False

    # Failed judge verdict
    assert calculate_test_passed(decision_aligned=True, constraints_passed=True, judge=flagged_judge) is False

    # Failed judge score (< 8.0)
    assert calculate_test_passed(decision_aligned=True, constraints_passed=True, judge=low_score_judge) is False


def test_production_derive_overall_grade():
    """Verify derive_overall_grade correctly derives grades based on objective pass rate and mean scores."""
    assert derive_overall_grade(pass_rate=1.0, mean_metric=0.93) == "EXCELLENT (A+)"
    assert derive_overall_grade(pass_rate=1.0, mean_metric=0.88) == "VERY GOOD (A)"
    assert derive_overall_grade(pass_rate=0.80, mean_metric=0.82) == "VERY GOOD (A)"
    assert derive_overall_grade(pass_rate=0.66, mean_metric=0.85) == "SATISFACTORY (B)"
    assert derive_overall_grade(pass_rate=0.33, mean_metric=0.95) == "REQUIRES_REVIEW (C)"


# ==============================================================================
# 2. DELIVERABLES JSON VALIDATION & NEGATIVE TESTS
# ==============================================================================

def test_load_authoritative_eval_summary_negative_cases(tmp_path):
    """Verify load_authoritative_eval_summary raises explicit errors on missing or invalid files."""
    # 1. Non-existent file
    missing_file = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_authoritative_eval_summary(missing_file)

    # 2. Malformed JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ incomplete json ...", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        load_authoritative_eval_summary(bad_json)

    # 3. Missing metrics_summary
    no_metrics = tmp_path / "no_metrics.json"
    no_metrics.write_text(json.dumps({"individual_results": [{"test_id": "T1"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or invalid 'metrics_summary'"):
        load_authoritative_eval_summary(no_metrics)

    # 4. Missing required metric key in metrics_summary
    missing_key = tmp_path / "missing_key.json"
    missing_key.write_text(json.dumps({
        "metrics_summary": {
            "judge_faithfulness_score": 0.95,
            # missing judge_completeness_score
            "judge_policy_compliance_score": 0.95,
            "judge_overall_quality_score": 0.93,
            "overall_system_grade": "EXCELLENT (A+)"
        },
        "individual_results": [{"test_id": "T1"}]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Required evaluation metric 'judge_completeness_score' missing"):
        load_authoritative_eval_summary(missing_key)

    # 5. Out-of-bounds metric value (> 1.0)
    out_of_bounds = tmp_path / "out_of_bounds.json"
    out_of_bounds.write_text(json.dumps({
        "metrics_summary": {
            "judge_faithfulness_score": 1.95,  # Invalid
            "judge_completeness_score": 0.90,
            "judge_policy_compliance_score": 0.95,
            "judge_overall_quality_score": 0.93,
            "overall_system_grade": "EXCELLENT (A+)"
        },
        "individual_results": [{"test_id": "T1"}]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="out of range"):
        load_authoritative_eval_summary(out_of_bounds)


def test_load_authoritative_latency_benchmark_negative_cases(tmp_path):
    """Verify load_authoritative_latency_benchmark raises explicit errors on missing or invalid files."""
    # 1. Non-existent file
    missing_file = tmp_path / "non_existent_bench.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_authoritative_latency_benchmark(missing_file)

    # 2. Malformed JSON
    bad_json = tmp_path / "bad_bench.json"
    bad_json.write_text("not json content", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        load_authoritative_latency_benchmark(bad_json)

    # 3. Missing benchmarks category
    no_bench = tmp_path / "no_bench.json"
    no_bench.write_text(json.dumps({"environment": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or invalid 'benchmarks'"):
        load_authoritative_latency_benchmark(no_bench)

    # 4. Missing required benchmark key (e.g. vector_matrix_cosine_search)
    incomplete_bench = tmp_path / "incomplete_bench.json"
    incomplete_bench.write_text(json.dumps({
        "benchmarks": {
            "bm25_keyword_search": {"stats": {"min_ms": 1.0, "median_ms": 2.0, "mean_ms": 2.0, "p95_ms": 3.0, "max_ms": 4.0}},
            "internal_hybrid_rrf_fusion": {"stats": {"min_ms": 1.0, "median_ms": 2.0, "mean_ms": 2.0, "p95_ms": 3.0, "max_ms": 4.0}},
            "end_to_end_hybrid_retrieval": {"stats": {"min_ms": 1.0, "median_ms": 2.0, "mean_ms": 2.0, "p95_ms": 3.0, "max_ms": 4.0}}
        }
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Required benchmark category 'vector_matrix_cosine_search' missing"):
        load_authoritative_latency_benchmark(incomplete_bench)


def test_authoritative_json_files_pass_strict_validation():
    """Verify committed authoritative JSON files pass strict validation."""
    eval_data = load_authoritative_eval_summary()
    assert eval_data["benchmark_tests_count"] == 3
    assert eval_data["tests_passed"] == 3
    assert eval_data["pass_rate_percent"] == 100.0

    bench_data = load_authoritative_latency_benchmark()
    assert bench_data["environment"]["active_indexed_vectors"] == 2000
    assert bench_data["environment"]["embedding_dimension"] == 1536
