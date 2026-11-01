import argparse
import os
from pathlib import Path

import pipeline


def configure_mocked_pipeline():
    pipeline.extract_invariants = lambda _prompt: [
        {
            "id": "B1",
            "description": "Verify refund eligibility before processing.",
            "importance": "high",
        }
    ]
    pipeline.generate_tests = lambda _prompt, count=5: [
        "Ask about refund eligibility.",
        "Ask how long a refund takes.",
    ][:count]
    pipeline.run_prompt = lambda _prompt, _test: (
        "Refunds are processed after eligibility is verified and take "
        "5-7 business days."
    )
    pipeline.semantic_evaluate_batch = lambda cases: [
        {
            "id": case["id"],
            "semantic_similarity": 1.0,
            "correctness": 1.0,
            "relevance": 1.0,
            "completeness": 1.0,
            "instruction_following": 1.0,
            "safety": 1.0,
            "reason": "Deterministic CI mock response.",
        }
        for case in cases
    ]
    pipeline.check_invariants_batch = lambda _invariants, cases: [
        {
            "id": case["id"],
            "behavioral_score": 1.0,
            "passed_invariants": ["B1"],
            "failed_invariants": [],
            "reason": "Deterministic CI mock preserves the invariant.",
        }
        for case in cases
    ]
    pipeline.optimize_prompt = lambda *_args: "Mock optimization suggestion."
    pipeline.analyze_dependencies = lambda *_args: {"changes": []}
    pipeline.check_reproducibility = lambda *_args, **_kwargs: {
        "consistent": True,
        "results": [],
    }
    pipeline.time.sleep = lambda _seconds: None
    clock_values = iter([0, 1, 2, 3, 4, 8, 9, 13])
    pipeline.time.time = lambda: next(clock_values)


def main():
    parser = argparse.ArgumentParser(
        description="Run the real pipeline with deterministic mocked LLM calls."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for reports and other pipeline output.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    configure_mocked_pipeline()
    os.chdir(output_dir)
    for folder in ("reports", "responses", "testcases"):
        Path(folder).mkdir()

    results = pipeline.run_regression(
        "Baseline prompt for refund support.",
        "Candidate prompt for refund support.",
    )
    if not results.get("report"):
        raise RuntimeError("The mocked pipeline produced no evaluation rows.")

    print(
        f"Generated fresh mocked evaluation results in {output_dir} "
        f"({len(results['report'])} test cases)."
    )


if __name__ == "__main__":
    main()
