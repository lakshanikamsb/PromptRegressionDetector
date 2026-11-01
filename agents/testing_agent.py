from groq_client import create_groq_client

client = create_groq_client()


def run_prompt(prompt, user_input):

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