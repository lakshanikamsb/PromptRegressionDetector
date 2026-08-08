from groq_client import generate_response

# Define reusable agent personas — add as many as you need
AGENT_PROMPTS = {
    "customer_support": "You are a helpful customer support assistant.",
    "coding_assistant": "You are an expert software engineer who writes clean, well-documented code.",
    "sales_agent": "You are a persuasive but honest sales assistant helping customers choose the right product.",
    "hr_assistant": "You are an HR assistant who answers policy questions clearly and professionally.",
}

def build_prompt(agent_type: str, user_input: str) -> str:
    role_description = AGENT_PROMPTS.get(agent_type, "You are a helpful assistant.")
    return f"""
{role_description}

User:
{user_input}
"""

def run_agent(agent_type: str, user_input: str) -> str:
    prompt = build_prompt(agent_type, user_input)
    return generate_response(prompt)

if __name__ == "__main__":
    agent_type = input(f"Choose agent type {list(AGENT_PROMPTS.keys())}: ").strip()
    user_input = input("Ask something: ")

    response = run_agent(agent_type, user_input)

    print("\nAI Response:")
    print(response)