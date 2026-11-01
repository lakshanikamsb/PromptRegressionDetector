import json
from groq_client import create_groq_client

client = create_groq_client()


def analyze_dependencies(old_prompt, new_prompt):

    instruction = f"""
You are a Prompt Dependency Impact Analyzer.

Compare the OLD PROMPT and NEW PROMPT.

OLD PROMPT:
{old_prompt}

NEW PROMPT:
{new_prompt}

Identify dependencies introduced, removed, or changed by the new prompt.

Look for dependencies such as:

- databases
- APIs
- external tools
- knowledge bases
- files
- user profile information
- order information
- authentication
- external services
- model capabilities
- variables
- policies
- external systems

Also identify which types of test cases may be affected.

Return ONLY valid JSON in this format:

{{
    "old_dependencies": [],
    "new_dependencies": [],
    "added_dependencies": [],
    "removed_dependencies": [],
    "changed_dependencies": [],
    "affected_test_types": [],
    "impact_level": "Low",
    "explanation": ""
}}

Impact level must be one of:

Low
Medium
High
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": instruction
            }
        ],

        temperature=0
    )

    result = response.choices[0].message.content

    # Remove markdown JSON formatting if Gemini/Groq adds it
    result = result.replace("```json", "")
    result = result.replace("```", "")
    result = result.strip()

    report = json.loads(result)

    return report