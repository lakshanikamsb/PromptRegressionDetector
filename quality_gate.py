import argparse
import json
import math
import sys
from pathlib import Path


MIN_PRESERVATION_SCORE = 0.75
VALID_STATUSES = {
    "Regression Detected",
    "Performance Regression",
    "No Regression",
}


def evaluate_reports(report_path, invariant_path):
    result = {
        "errors": [],
        "functional_regressions": 0,
        "performance_regressions": 0,
        "preservation_score": None,
    }

    try:
        with Path(report_path).open(encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        result["errors"].append(
            f"Could not read evaluation report '{report_path}': {error}"
        )
        report = None

    if report is not None:
        if not isinstance(report, list) or not report:
            result["errors"].append(
                "Evaluation report must be a non-empty JSON array."
            )
        else:
            for index, item in enumerate(report):
                if not isinstance(item, dict):
                    result["errors"].append(
                        f"Evaluation report item {index + 1} must be a JSON object."
                    )
                    continue

                test_case = item.get("Test Case")
                if not isinstance(test_case, str) or not test_case.strip():
                    result["errors"].append(
                        f"Evaluation report item {index + 1} must contain "
                        "a non-empty Test Case."
                    )

                status = item.get("Status")
                if not isinstance(status, str) or status not in VALID_STATUSES:
                    result["errors"].append(
                        f"Evaluation report item {index + 1} has an invalid Status."
                    )
                elif status == "Regression Detected":
                    result["functional_regressions"] += 1
                elif status == "Performance Regression":
                    result["performance_regressions"] += 1

                latency_score = item.get("Latency Score")
                if (
                    isinstance(latency_score, bool)
                    or not isinstance(latency_score, (int, float))
                    or not math.isfinite(latency_score)
                    or not 0 <= latency_score <= 1
                ):
                    result["errors"].append(
                        f"Evaluation report item {index + 1} must contain "
                        "a finite Latency Score between 0 and 1."
                    )

    try:
        with Path(invariant_path).open(encoding="utf-8") as file:
            invariant_data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        result["errors"].append(
            f"Could not read behavioral invariant report "
            f"'{invariant_path}': {error}"
        )
        invariant_data = None

    if invariant_data is not None:
        if not isinstance(invariant_data, dict):
            result["errors"].append(
                "Behavioral invariant report must be a JSON object."
            )
        else:
            preservation_score = invariant_data.get("preservation_score")
            invariant_results = invariant_data.get("results")

            if (
                isinstance(preservation_score, bool)
                or not isinstance(preservation_score, (int, float))
                or not math.isfinite(preservation_score)
                or not 0 <= preservation_score <= 1
            ):
                result["errors"].append(
                    "Behavioral preservation score must be a finite number "
                    "between 0 and 1."
                )
            else:
                result["preservation_score"] = preservation_score

            if (
                not isinstance(invariant_results, list)
                or not invariant_results
                or any(not isinstance(item, dict) for item in invariant_results)
            ):
                result["errors"].append(
                    "Behavioral invariant report must contain a non-empty "
                    "results array of objects."
                )
            else:
                for index, item in enumerate(invariant_results):
                    score = item.get("behavioral_score")
                    if (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(score)
                        or not 0 <= score <= 1
                    ):
                        result["errors"].append(
                            "Behavioral invariant result "
                            f"{index + 1} must contain a finite behavioral_score "
                            "between 0 and 1."
                        )

    result["passed"] = (
        not result["errors"]
        and result["functional_regressions"] == 0
        and result["preservation_score"] >= MIN_PRESERVATION_SCORE
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate fresh prompt evaluation and invariant reports."
    )
    parser.add_argument(
        "--report",
        default="reports/report.json",
        help="Path to the evaluation report JSON.",
    )
    parser.add_argument(
        "--invariants",
        default="reports/behavioral_invariant_results.json",
        help="Path to the behavioral invariant results JSON.",
    )
    args = parser.parse_args(argv)

    result = evaluate_reports(args.report, args.invariants)

    print("PROMPT REGRESSION QUALITY GATE")
    print("--------------------------------")
    print(f"Functional regressions: {result['functional_regressions']}")
    print(
        "Performance regressions: "
        f"{result['performance_regressions']} "
        "(advisory; they do not fail the functional quality gate)"
    )
    if result["preservation_score"] is not None:
        print(
            "Behavioral preservation: "
            f"{result['preservation_score']:.2f} "
            f"(minimum {MIN_PRESERVATION_SCORE:.2f})"
        )

    if result["errors"]:
        for error in result["errors"]:
            print(f"ERROR: {error}")
    if result["functional_regressions"]:
        print("QUALITY GATE FAILED: Functional regressions were detected.")
    if (
        result["preservation_score"] is not None
        and result["preservation_score"] < MIN_PRESERVATION_SCORE
    ):
        print("QUALITY GATE FAILED: Behavioral preservation is too low.")

    if not result["passed"]:
        print("QUALITY GATE FAILED")
        return 1

    print("QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
