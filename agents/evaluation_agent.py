import os
import json
import re
import time

from difflib import SequenceMatcher

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "it",
    "as",
    "be",
    "by",
    "from",
    "at"
}


def get_words(text):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def text_similarity(old_output, new_output):
    return round(
        SequenceMatcher(
            None,
            old_output.lower(),
            new_output.lower()
        ).ratio(),
        2
    )


def word_overlap(old_output, new_output):

    old_words = set(get_words(old_output))
    new_words = set(get_words(new_output))

    if len(old_words) == 0:
        return 1.0

    overlap = old_words.intersection(new_words)

    return round(
        len(overlap) / len(old_words),
        2
    )


def length_similarity(old_output, new_output):

    old_length = len(old_output)
    new_length = len(new_output)

    if old_length == 0:
        return 1.0

    difference = abs(old_length - new_length)

    score = 1 - (
        difference / max(old_length, new_length)
    )

    return round(
        max(0, score),
        2
    )


def information_preservation(old_output, new_output):

    old_words = set(get_words(old_output))
    new_words = set(get_words(new_output))

    if len(old_words) == 0:
        return 1.0

    preserved = old_words.intersection(new_words)

    return round(
        len(preserved) / len(old_words),
        2
    )


def important_number_preservation(old_output, new_output):

    old_numbers = re.findall(
        r"\d+(?:\s*[-–]\s*\d+)?",
        old_output
    )

    new_numbers = re.findall(
        r"\d+(?:\s*[-–]\s*\d+)?",
        new_output
    )

    old_numbers = set(
        number.replace(" ", "")
        for number in old_numbers
    )

    new_numbers = set(
        number.replace(" ", "")
        for number in new_numbers
    )

    if len(old_numbers) == 0:
        return 1.0

    preserved = old_numbers.intersection(
        new_numbers
    )

    score = len(preserved) / len(old_numbers)

    return round(score, 2)


def evaluate(old_output, new_output):

    similarity = text_similarity(
        old_output,
        new_output
    )

    overlap = word_overlap(
        old_output,
        new_output
    )

    length = length_similarity(
        old_output,
        new_output
    )

    preservation = information_preservation(
        old_output,
        new_output
    )

    number_preservation = important_number_preservation(
        old_output,
        new_output
    )

    overall_score = (
        similarity * 0.25
        + overlap * 0.20
        + length * 0.15
        + preservation * 0.20
        + number_preservation * 0.20
    )

    return {
        "Similarity": similarity,
        "Word Overlap": overlap,
        "Length Similarity": length,
        "Information Preservation": preservation,
        "Number Preservation": number_preservation,
        "Overall Score": round(
            overall_score,
            2
        )
    }


def latency_score(latency_v1, latency_v2):

    if latency_v1 <= 0:
        return 1.0

    ratio = latency_v2 / latency_v1

    if ratio <= 1:
        return 1.0

    if ratio >= 3:
        return 0.0

    score = 1 - (
        (ratio - 1) / 2
    )

    return round(
        max(0, score),
        2
    )

def semantic_evaluate_batch(test_cases):

    if not test_cases:
        return []

    test_cases = test_cases[:5]

    cases_text = ""

    for i, case in enumerate(test_cases, start=1):
        cases_text += f"""
TEST CASE {i}

ID: {case.get("id", i)}

User Input:
{case.get("test_case", "")}

OLD OUTPUT:
{case.get("old_output", "")[:1000]}

NEW OUTPUT:
{case.get("new_output", "")[:1000]}

"""

    instruction = f"""
You are an expert software testing evaluator.

Compare the OLD OUTPUT and NEW OUTPUT for each test case.

Evaluate BEHAVIOR and MEANING, not exact wording.

Return exactly one JSON object for each test case.

For each case provide:

- id
- semantic_similarity: number from 0 to 1
- correctness: number from 0 to 1
- relevance: number from 0 to 1
- completeness: number from 0 to 1
- instruction_following: number from 0 to 1
- safety: number from 0 to 1
- reason: short explanation

IMPORTANT:

1. Return ONLY valid JSON.
2. Do NOT use markdown.
3. Do NOT use ```json.
4. Keep every "reason" SHORT.
5. Do not put newline characters inside reason.
6. Do not use double quotes inside the reason text.
7. Do not add any text before or after the JSON.
8. Return exactly {len(test_cases)} objects.

Expected format:

[
  {{
    "id": 1,
    "semantic_similarity": 0.8,
    "correctness": 0.9,
    "relevance": 0.9,
    "completeness": 0.8,
    "instruction_following": 0.9,
    "safety": 1.0,
    "reason": "The new response preserves the main behavior."
  }}
]

Test cases:

{cases_text}
"""

    for attempt in range(4):

        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": instruction
                    }
                ],
                temperature=0,
                max_tokens=3000
            )

            result = response.choices[0].message.content.strip()

            print("\nSemantic evaluation response:")
            print(result)

            # Remove markdown fences if the model adds them
            if result.startswith("```"):
                result = result.replace("```json", "")
                result = result.replace("```", "")
                result = result.strip()

            # Find the JSON array
            start = result.find("[")
            end = result.rfind("]")

            if start == -1 or end == -1:
                raise ValueError(
                    "Semantic evaluator did not return a JSON array."
                )

            result = result[start:end + 1]

            try:

                parsed_result = json.loads(result)

            except json.JSONDecodeError as json_error:

                print("\nInvalid JSON returned by semantic evaluator.")
                print("JSON error:")
                print(json_error)

                print("\nRaw model response:")
                print(result)

                if attempt < 3:
                    print("\nRetrying semantic evaluation...")
                    time.sleep(2)
                    continue

                raise Exception(
                    "Semantic evaluator returned invalid JSON "
                    "after multiple attempts."
                )

            if not isinstance(parsed_result, list):
                raise ValueError(
                    "Semantic evaluator returned something other than a list."
                )

            return parsed_result

        except Exception as e:

            error_message = str(e)

            if (
                "429" in error_message
                or "rate_limit_exceeded" in error_message
            ):

                if attempt < 3:

                    wait_time = [5, 10, 20][attempt]

                    print(
                        f"Semantic evaluation rate limit reached. "
                        f"Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                raise Exception(
                    "Semantic evaluation failed because Groq "
                    "rate limit remained active."
                )

            if attempt < 3 and (
                "JSON" in error_message
                or "json" in error_message
                or "Unterminated string" in error_message
            ):

                print("\nSemantic evaluation JSON error.")
                print("Retrying...")
                time.sleep(2)

                continue

            raise e

    raise Exception(
        "Unable to complete semantic evaluation."
    )