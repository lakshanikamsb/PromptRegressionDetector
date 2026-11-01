import os
import json
from difflib import SequenceMatcher

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_response(prompt, user_input):

    full_prompt = f"""
System Prompt:
{prompt}

User:
{user_input}
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


def similarity(text1, text2):

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


def check_reproducibility(prompt, tests, repetitions=3):

    results = []

    # Use only first 3 tests to reduce API usage
    selected_tests = tests[:3]

    for test in selected_tests:

        outputs = []

        for i in range(repetitions):

            output = generate_response(
                prompt,
                test
            )

            outputs.append(output)

        scores = []

        for i in range(len(outputs)):

            for j in range(i + 1, len(outputs)):

                score = similarity(
                    outputs[i],
                    outputs[j]
                )

                scores.append(score)

        if len(scores) > 0:

            average_score = sum(scores) / len(scores)

        else:

            average_score = 1.0

        if average_score >= 0.80:

            status = "Highly Reproducible"

        elif average_score >= 0.60:

            status = "Moderately Reproducible"

        else:

            status = "Low Reproducibility"

        results.append({

            "test": test,

            "outputs": outputs,

            "reproducibility_score": round(
                average_score,
                2
            ),

            "status": status

        })

    if len(results) > 0:

        overall_score = sum(
            item["reproducibility_score"]
            for item in results
        ) / len(results)

    else:

        overall_score = 0

    if overall_score >= 0.80:

        overall_status = "Highly Reproducible"

    elif overall_score >= 0.60:

        overall_status = "Moderately Reproducible"

    else:

        overall_status = "Low Reproducibility"

    report = {

        "overall_reproducibility_score":
            round(overall_score, 2),

        "overall_status":
            overall_status,

        "tests": results

    }

    return report