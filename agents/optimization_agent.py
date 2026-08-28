import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def optimize_prompt(old_prompt, new_prompt, evaluation_report):

    optimization_prompt = f"""
You are an expert Prompt Engineer.

Two versions of a prompt were tested.

OLD PROMPT:
{old_prompt}

NEW PROMPT:
{new_prompt}

Evaluation Report:
{evaluation_report}

Your task:

1. Identify why regression happened.
2. Explain the weaknesses in the new prompt.
3. Suggest improvements.
4. Write a new optimized prompt.

Return your answer in the following format:

Problems:
- ...

Suggestions:
- ...

Optimized Prompt:
...

Reason:
...
"""

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": optimization_prompt
                }
            ],

            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Optimization failed: {e}"