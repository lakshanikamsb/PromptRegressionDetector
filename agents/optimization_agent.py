from groq_client import create_groq_client

client = create_groq_client()


def optimize_prompt(prompt_v1, prompt_v2, report, invariants):

    failed_cases = []

    for case in report:

        if case["Status"] == "Regression Detected":

            failed_cases.append({
                "test_case": case["Test Case"],
                "old_output": case["Old Output"],
                "new_output": case["New Output"],
                "similarity": case["Similarity"],
                "word_overlap": case["Word Overlap"],
                "information_preservation":
                    case["Information Preservation"],
                "overall_score":
                    case["Overall Score"]
            })

    instruction = f"""
You are a Prompt Optimization Agent.

Your job is to improve Prompt Version 2 so that it behaves
more consistently with Prompt Version 1 and reduces the
number of regression cases during retesting.


BEHAVIORAL INVARIANTS:

{invariants}


PROMPT VERSION 1:
{prompt_v1}

CURRENT PROMPT VERSION 2:
{prompt_v2}

FAILED REGRESSION CASES:
{failed_cases}

IMPORTANT RULES:

1. Preserve the original purpose and domain of Prompt Version 1.

2. Do NOT add unrelated features, topics, or capabilities.

3. Analyze the failed test cases carefully.

4. Identify why Prompt Version 2 produced behavior different
   from Prompt Version 1.

5. Modify Prompt Version 2 only where necessary to fix the
   detected problems.

6. Preserve the useful behavior of Prompt Version 1.

7. Make the optimized prompt concise and clear.

8. The optimized prompt should produce responses that are
   semantically and behaviorally similar to Prompt Version 1.

9. Do not intentionally make the response identical word-for-word.
   Preserve the meaning and required behavior.

10. Give special attention to the regression cases instead of
    adding unrelated instructions.

11. Do not add store hours, subscriptions, discount codes,
    returns, damaged products, or other topics unless those
    topics are actually present in Prompt Version 1 or the
    failed test cases.

12. The goal is to reduce the number of regression cases
    during the next retest.

Return ONLY the complete optimized prompt.

Do not include:
Problems:
Suggestions:
Reason:
Optimized Prompt:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": instruction
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content