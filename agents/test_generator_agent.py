import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_tests(prompt):

    instruction = f"""
You are a software testing expert.

Generate exactly 10 different user test cases for the following system prompt.

Prompt:
{prompt}

Return ONLY a JSON array like this:

[
  "test1",
  "test2",
  "test3"
]

Do not explain anything.
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": instruction
            }
        ],

        temperature=0.3
    )

    tests = response.choices[0].message.content

    tests = json.loads(tests)

    with open(
        "testcases/generated_tests.json",
        "w"
    ) as f:

        json.dump(
            tests,
            f,
            indent=4
        )

    return tests