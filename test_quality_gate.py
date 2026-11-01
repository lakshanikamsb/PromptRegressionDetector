import json

from quality_gate import evaluate_reports


def write_reports(tmp_path, report, invariants):
    report_path = tmp_path / "report.json"
    invariant_path = tmp_path / "invariants.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invariant_path.write_text(json.dumps(invariants), encoding="utf-8")
    return report_path, invariant_path


def valid_invariants(score=0.9):
    return {
        "preservation_score": score,
        "results": [{"behavioral_score": score}],
    }


def valid_report(status):
    return [
        {
            "Test Case": "Check a refund request.",
            "Status": status,
            "Latency Score": 0.5
            if status == "Performance Regression"
            else 1.0,
        }
    ]


def test_performance_regressions_are_reported_but_non_blocking(tmp_path):
    paths = write_reports(
        tmp_path,
        valid_report("Performance Regression"),
        valid_invariants(),
    )

    result = evaluate_reports(*paths)

    assert result["passed"] is True
    assert result["functional_regressions"] == 0
    assert result["performance_regressions"] == 1


def test_functional_regression_fails_quality_gate(tmp_path):
    paths = write_reports(
        tmp_path,
        valid_report("Regression Detected"),
        valid_invariants(),
    )

    result = evaluate_reports(*paths)

    assert result["passed"] is False
    assert result["functional_regressions"] == 1


def test_low_behavioral_preservation_fails_quality_gate(tmp_path):
    paths = write_reports(
        tmp_path,
        valid_report("No Regression"),
        valid_invariants(score=0.7),
    )

    result = evaluate_reports(*paths)

    assert result["passed"] is False
    assert result["preservation_score"] == 0.7


def test_empty_or_malformed_reports_fail_quality_gate(tmp_path):
    paths = write_reports(tmp_path, [], valid_invariants())

    result = evaluate_reports(*paths)

    assert result["passed"] is False
    assert any("non-empty JSON array" in error for error in result["errors"])


def test_non_string_status_is_reported_as_malformed(tmp_path):
    paths = write_reports(
        tmp_path,
        [
            {
                "Test Case": "Check a refund request.",
                "Status": [],
                "Latency Score": 1.0,
            }
        ],
        valid_invariants(),
    )

    result = evaluate_reports(*paths)

    assert result["passed"] is False
    assert any("invalid Status" in error for error in result["errors"])
