import json
import time

from groq_client import create_groq_client

client = create_groq_client()

MODEL = "openai/gpt-oss-20b"


def extract_invariants(prompt):

    instruction = f"""
You are a behavioral invariant extraction agent.

Analyze the following original prompt.

Extract the important behaviors that must remain unchanged
when the prompt is modified.

Only extract behaviors that are clearly supported by the prompt.
Do not invent new requirements.

Return ONLY valid JSON.
Return a JSON array.

Each invariant must contain:
- id
- description
- importance

The importance must be "high", "medium", or "low".

Original prompt:

{prompt}
"""

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": instruction
                    }
                ],
                temperature=0,
                max_tokens=1200
            )

            result = response.choices[0].message.content.strip()

            if result.startswith("```"):
                result = result.replace(
                    "```json", ""
                ).replace(
                    "```", ""
                ).strip()

            return json.loads(result)

        except Exception as e:
            error_message = str(e)

            if (
                "429" in error_message
                or "rate_limit_exceeded" in error_message
            ):
                if attempt < 3:
                    wait_time = [5, 10, 20][attempt]

                    print(
                        "Invariant extraction rate limit. "
                        f"Waiting {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    raise Exception(
                        "Invariant extraction failed because "
                        "the Groq rate limit remained active."
                    ) from e
            else:
                raise


def check_invariants_batch(invariants, test_cases):

    if not test_cases:
        return []

    # Development limit
    test_cases = test_cases[:5]

    instruction = f"""
You are a behavioral invariant testing agent.

The invariants describe behaviors that the original prompt
must preserve.

Evaluate whether each NEW output preserves each invariant.

Focus on behavior and meaning. Do not mark an invariant
as failed merely because the wording is different.
A meaningful behavioral change should fail the invariant.

Return ONLY valid JSON.
Return a JSON array.

Each item must contain:
- id
- behavioral_score
- passed_invariants
- failed_invariants
- reason

behavioral_score must be between 0 and 1.

Use:
1.0 = all important behavior preserved
0.75 = mostly preserved
0.50 = partially preserved
0.25 = mostly violated
0.0 = completely violated

Invariants:

{json.dumps(invariants, indent=2)}

Test cases:

{json.dumps(test_cases, indent=2)}
"""

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": instruction
                    }
                ],
                temperature=0,
                max_tokens=2500
            )

            result = response.choices[0].message.content.strip()

            if result.startswith("```"):
                result = result.replace(
                    "```json", ""
                ).replace(
                    "```", ""
                ).strip()

            return json.loads(result)

        except Exception as e:
            error_message = str(e)

            if (
                "429" in error_message
                or "rate_limit_exceeded" in error_message
            ):
                if attempt < 3:
                    wait_time = [5, 10, 20][attempt]

                    print(
                        "Invariant evaluation rate limit. "
                        f"Waiting {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    raise Exception(
                        "Invariant evaluation failed because "
                        "the Groq rate limit remained active."
                    ) from e
            else:
                raise


def calculate_preservation_score(invariant_results):

    if not invariant_results:
        return 0.0

    scores = []

    for result in invariant_results:
        score = result.get("behavioral_score", 0)
        scores.append(score)

    return round(sum(scores) / len(scores), 2)
