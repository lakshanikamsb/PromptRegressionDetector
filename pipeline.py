import json

from agents.test_generator_agent import generate_tests
from agents.testing_agent import run_prompt
from agents.evaluation_agent import evaluate
from agents.regression_agent import detect
from agents.optimization_agent import optimize_prompt


def run_regression(prompt_v1, prompt_v2):

    print("\nGenerating Test Cases...\n")

    # ----------------------------------------
    # Generate Test Cases
    # ----------------------------------------

    tests = generate_tests(prompt_v1)

    # ----------------------------------------
    # Run Prompt Version 1
    # ----------------------------------------

    print("Running Prompt Version 1...\n")

    results_v1 = []

    for i, test in enumerate(tests, 1):

        print(f"Running Test {i}")

        output = run_prompt(
            prompt_v1,
            test
        )

        results_v1.append({

            "id": i,

            "input": test,

            "output": output

        })

    with open(
        "responses/responses_v1.json",
        "w"
    ) as f:

        json.dump(
            results_v1,
            f,
            indent=4
        )

    print("\nPrompt Version 1 Completed\n")

    # ----------------------------------------
    # Run Prompt Version 2
    # ----------------------------------------

    print("Running Prompt Version 2...\n")

    results_v2 = []

    for i, test in enumerate(tests, 1):

        print(f"Running Test {i}")

        output = run_prompt(
            prompt_v2,
            test
        )

        results_v2.append({

            "id": i,

            "input": test,

            "output": output

        })

    with open(
        "responses/responses_v2.json",
        "w"
    ) as f:

        json.dump(
            results_v2,
            f,
            indent=4
        )

    print("\nPrompt Version 2 Completed\n")

    # ----------------------------------------
    # Evaluation
    # ----------------------------------------

    print("Evaluating...\n")

    report = []

    for old_case, new_case in zip(results_v1, results_v2):

        similarity = evaluate(

            old_case["output"],

            new_case["output"]

        )

        status = detect(similarity)

        report.append({

            "Test Case": old_case["input"],

            "Old Output": old_case["output"],

            "New Output": new_case["output"],

            "Similarity": similarity,

            "Status": status

        })

    with open(
        "reports/evaluation_report.json",
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print("Evaluation Completed\n")

    # ----------------------------------------
    # Optimization
    # ----------------------------------------

    optimization = ""

    regression_found = False

    for item in report:

        if item["Status"] == "Regression Detected":

            regression_found = True

            break

    if regression_found:

        print("Running Optimization Agent...\n")

        optimization = optimize_prompt(

            prompt_v1,

            prompt_v2,

            report

        )

        with open(
            "reports/optimization_report.json",
            "w"
        ) as f:

            json.dump(

                {

                    "Optimization": optimization

                },

                f,

                indent=4

            )

        print("Optimization Completed\n")

    else:

        optimization = "No regression detected."

    # ----------------------------------------
    # Return Everything
    # ----------------------------------------

    return {

        "tests": tests,

        "responses_v1": results_v1,

        "responses_v2": results_v2,

        "report": report,

        "optimization": optimization

    }