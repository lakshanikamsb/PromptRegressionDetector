import json
import os
import time

from groq_client import create_groq_client

client = create_groq_client()

MODEL = "openai/gpt-oss-20b"

def generate_tests(prompt, count=5):

    instruction = f"""
You are a software testing expert.

Generate exactly {count} different user test cases for the following
system prompt.

The test cases must be realistic and different from each other.

Prompt:
{prompt}

IMPORTANT:
Return ONLY a valid JSON array.
Do not use markdown.
Do not use ```json.
Do not explain anything.

The response MUST look exactly like:

[
  "test case 1",
  "test case 2",
  "test case 3",
  "test case 4",
  "test case 5"
]

Keep every test case short.
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

                temperature=0.2,

                # Give the model enough space to finish JSON
                max_tokens=1500
            )

            result = response.choices[0].message.content.strip()

            print("\nGenerated test cases response:")
            print(result)

            # -----------------------------------------
            # Remove markdown if the model adds it
            # -----------------------------------------

            if result.startswith("```"):

                result = result.replace(
                    "```json",
                    ""
                ).replace(
                    "```",
                    ""
                ).strip()

            # -----------------------------------------
            # Find JSON array
            # -----------------------------------------

            start = result.find("[")

            end = result.rfind("]")

            if start == -1 or end == -1:

                raise ValueError(
                    "Test Generator did not return a JSON array."
                )

            result = result[start:end + 1]

            # -----------------------------------------
            # Parse JSON
            # -----------------------------------------

            tests = json.loads(result)

            # -----------------------------------------
            # Validate result
            # -----------------------------------------

            if not isinstance(tests, list):

                raise ValueError(
                    "Test Generator returned something other "
                    "than a JSON list."
                )

            if len(tests) == 0:

                raise ValueError(
                    "Test Generator returned an empty list."
                )

            # Keep only requested number
            tests = tests[:count]

            # -----------------------------------------
            # Save automatically generated tests
            # -----------------------------------------

            os.makedirs(
                "testcases",
                exist_ok=True
            )

            with open(
                "testcases/generated_tests.json",
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    tests,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            print(
                f"\nSuccessfully generated {len(tests)} test cases."
            )

            return tests

        except Exception as e:

            error_message = str(e)

            # -----------------------------------------
            # Rate limit
            # -----------------------------------------

            if (
                "429" in error_message
                or "rate_limit_exceeded" in error_message
            ):

                if attempt < 3:

                    wait_time = [5, 10, 20][attempt]

                    print(
                        f"Test generation rate limit reached. "
                        f"Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                raise Exception(
                    "Test generation failed because "
                    "Groq rate limit remained active."
                )

            # -----------------------------------------
            # JSON error
            # -----------------------------------------

            if isinstance(e, json.JSONDecodeError):

                print(
                    "\nInvalid JSON received from Test Generator."
                )

                print(
                    "Model response was:"
                )

                print(result)

                if attempt < 3:

                    print(
                        "\nRetrying test generation..."
                    )

                    time.sleep(2)

                    continue

                raise Exception(
                    "Test Generator returned invalid JSON "
                    "after multiple attempts."
                )

            # -----------------------------------------
            # Other errors
            # -----------------------------------------

            raise e

    raise Exception(
        "Unable to generate test cases."
    )