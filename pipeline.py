import os
import json
import time

from agents.test_generator_agent import generate_tests
from agents.testing_agent import run_prompt
from agents.evaluation_agent import (
    evaluate,
    latency_score,
    semantic_evaluate_batch
)
from agents.regression_agent import detect
from agents.optimization_agent import optimize_prompt
from agents.dependency_impact_agent import analyze_dependencies
from agents.reproducibility_agent import check_reproducibility
from agents.behavioral_invariant_agent import (
    extract_invariants,
    check_invariants_batch,
    calculate_preservation_score
)


# --------------------------------------------------
# DEVELOPMENT SETTINGS
# --------------------------------------------------

MAX_TESTS = 5
MAX_SEMANTIC_TESTS = 5
MAX_INVARIANT_TESTS = 5
REPRODUCIBILITY_TESTS = 3
REPRODUCIBILITY_REPETITIONS = 1


# --------------------------------------------------
# CREATE REQUIRED FOLDERS
# --------------------------------------------------

os.makedirs("testcases", exist_ok=True)
os.makedirs("responses", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# --------------------------------------------------
# RUN REGRESSION PIPELINE
# --------------------------------------------------

def run_regression(prompt_v1, prompt_v2):

    print("\n====================================")
    print("PROMPT REGRESSION PIPELINE")
    print("====================================\n")


    # --------------------------------------------------
    # STEP 1: EXTRACT BEHAVIORAL INVARIANTS
    # --------------------------------------------------

    print("Step 1: Extracting behavioral invariants...")

    invariants = extract_invariants(prompt_v1)

    with open(
        "reports/behavioral_invariants.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            invariants,
            file,
            indent=4
        )

    print(
        f"Extracted {len(invariants)} behavioral invariants."
    )


    # --------------------------------------------------
    # STEP 2: LOAD OR GENERATE TEST CASES
    # --------------------------------------------------

    print("\nStep 2: Loading test cases...")

    # --------------------------------------------------
    # STEP 2: AUTOMATIC TEST CASE GENERATION
    # --------------------------------------------------

    print("\nStep 2: Generating test cases automatically...")

    tests = generate_tests(
        prompt_v1,
        count=5
    )

    print(
        f"Generated {len(tests)} test cases automatically."
    ) 
    # --------------------------------------------------
    # IMPORTANT:
    # LIMIT TESTS TO 5 DURING DEVELOPMENT
    # --------------------------------------------------

    tests = tests[:MAX_TESTS]

    print(
        f"Using {len(tests)} test cases."
    )


    # --------------------------------------------------
    # STEP 3: RUN PROMPT V1
    # --------------------------------------------------

    print("\nStep 3: Testing Prompt V1...")

    results_v1 = []

    for index, test in enumerate(tests):

        print(
            f"V1 test {index + 1}/{len(tests)}"
        )

        start_time = time.time()

        output = run_prompt(
            prompt_v1,
            test
        )

        end_time = time.time()

        latency = end_time - start_time

        results_v1.append(
            {
                "id": index + 1,
                "input": test,
                "output": output,
                "latency": latency
            }
        )

        # Small delay to reduce API pressure
        time.sleep(0.5)


    with open(
        "responses/responses_v1.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results_v1,
            file,
            indent=4
        )


    # --------------------------------------------------
    # STEP 4: RUN PROMPT V2
    # --------------------------------------------------

    print("\nStep 4: Testing Prompt V2...")

    results_v2 = []

    for index, test in enumerate(tests):

        print(
            f"V2 test {index + 1}/{len(tests)}"
        )

        start_time = time.time()

        output = run_prompt(
            prompt_v2,
            test
        )

        end_time = time.time()

        latency = end_time - start_time

        results_v2.append(
            {
                "id": index + 1,
                "input": test,
                "output": output,
                "latency": latency
            }
        )

        time.sleep(0.5)


    with open(
        "responses/responses_v2.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results_v2,
            file,
            indent=4
        )


    # --------------------------------------------------
    # STEP 5: SEMANTIC EVALUATION
    # --------------------------------------------------

    print("\nStep 5: Semantic evaluation...")

    semantic_cases = []

    for old_case, new_case in zip(
        results_v1[:MAX_SEMANTIC_TESTS],
        results_v2[:MAX_SEMANTIC_TESTS]
    ):

        semantic_cases.append(
            {
                "id": old_case["id"],
                "test_case": old_case["input"],
                "old_output": old_case["output"][:1200],
                "new_output": new_case["output"][:1200]
            }
        )


    semantic_results = semantic_evaluate_batch(
        semantic_cases
    )


    # --------------------------------------------------
    # STEP 6: BEHAVIORAL INVARIANT TESTING
    # --------------------------------------------------

    print("\nStep 6: Checking behavioral invariants...")

    invariant_cases = []

    for new_case in results_v2[:MAX_INVARIANT_TESTS]:

        invariant_cases.append(
            {
                "id": new_case["id"],
                "test_case": new_case["input"],
                "output": new_case["output"][:1200]
            }
        )


    invariant_results = check_invariants_batch(
        invariants,
        invariant_cases
    )


    preservation_score = calculate_preservation_score(
        invariant_results
    )


    # --------------------------------------------------
    # STEP 7: COMBINE ALL EVALUATION RESULTS
    # --------------------------------------------------

    print("\nStep 7: Creating evaluation report...")

    report = []


    for index, (old_case, new_case) in enumerate(
        zip(results_v1, results_v2)
    ):

        traditional = evaluate(
            old_case["output"],
            new_case["output"]
        )


        semantic = {}

        for result in semantic_results:

            if result.get("id") == old_case["id"]:

                semantic = result

                break


        invariant = {}

        for result in invariant_results:

            if result.get("id") == old_case["id"]:

                invariant = result

                break


        semantic_similarity = semantic.get(
            "semantic_similarity",
            0
        )

        correctness = semantic.get(
            "correctness",
            0
        )

        relevance = semantic.get(
            "relevance",
            0
        )

        completeness = semantic.get(
            "completeness",
            0
        )

        instruction_following = semantic.get(
            "instruction_following",
            0
        )

        safety = semantic.get(
            "safety",
            0
        )


        behavioral_score = invariant.get(
            "behavioral_score",
            1.0
        )


        failed_invariants = invariant.get(
            "failed_invariants",
            []
        )


        latency = latency_score(
            old_case["latency"],
            new_case["latency"]
        )


        evaluation = {
            **traditional,

            "Semantic Similarity": semantic_similarity,

            "Correctness": correctness,

            "Relevance": relevance,

            "Completeness": completeness,

            "Instruction Following": instruction_following,

            "Safety": safety,

            "Behavioral Score": behavioral_score,

            "Failed Invariants": failed_invariants,

            "Latency Score": latency,

            "Semantic Reason": semantic.get(
                "reason",
                ""
            )
        }


        status = detect(
            evaluation
        )

        report.append(
            {
                "Test Case": old_case["input"],

                "Old Output": old_case["output"],

                "New Output": new_case["output"],

                "V1 Latency": round(
                    old_case["latency"],
                    3
                ),

                "V2 Latency": round(
                    new_case["latency"],
                    3
                ),
                "Latency Increase": round(
                    new_case["latency"] - old_case["latency"],
                    3
                ),

                **evaluation,

                "Status": status
            }
        )


    # --------------------------------------------------
    # STEP 8: SAVE EVALUATION REPORT
    # --------------------------------------------------

    with open(
        "reports/report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )
        

    # --------------------------------------------------
    # STEP 9: OPTIMIZATION
    # --------------------------------------------------

    print("\nStep 9: Generating optimization suggestions...")

    # Send only first 5 cases to reduce token usage.
    optimization_report = report[:5]

    optimization = optimize_prompt(
        prompt_v1,
        prompt_v2,
        optimization_report,
        invariants
    )


    with open(
        "reports/optimization_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "optimization": optimization
            },
            file,
            indent=4
        )


    # --------------------------------------------------
    # STEP 10: DEPENDENCY IMPACT
    # --------------------------------------------------

    print("\nStep 10: Analyzing dependency impact...")

    dependency_report = analyze_dependencies(
        prompt_v1,
        prompt_v2
    )


    with open(
        "reports/dependency_impact_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dependency_report,
            file,
            indent=4
        )


    # --------------------------------------------------
    # STEP 11: REPRODUCIBILITY
    # --------------------------------------------------

    print("\nStep 11: Checking reproducibility...")

    reproducibility_tests = tests[
        :REPRODUCIBILITY_TESTS
    ]


    reproducibility_report = check_reproducibility(
        prompt_v2,
        reproducibility_tests,
        repetitions=REPRODUCIBILITY_REPETITIONS
    )


    with open(
        "reports/reproducibility_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            reproducibility_report,
            file,
            indent=4
        )


    # --------------------------------------------------
    # STEP 12: SAVE INVARIANT RESULTS
    # --------------------------------------------------

    with open(
        "reports/behavioral_invariant_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "preservation_score": preservation_score,
                "results": invariant_results
            },
            file,
            indent=4
        )


    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    print("\n====================================")
    print("PIPELINE COMPLETED")
    print("====================================\n")

    # ----------------------------------------
    # Return Everything
    # ----------------------------------------

    return {
    "report": report,
    "optimization": optimization,
    "dependency": dependency_report,
    "reproducibility": reproducibility_report,
    "invariants": invariants,
    "behavioral_results": invariant_results,
    "behavioral_preservation_score": preservation_score
}