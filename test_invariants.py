from agents.behavioral_invariant_agent import extract_invariants


prompt = """
You are a customer support assistant for an online shopping company.

When a customer asks for a refund:
1. Ask for the order ID.
2. Check whether the order is eligible for a refund.
3. Explain the refund process clearly.
4. Never promise a refund before eligibility is confirmed.
5. If information is missing, ask for the required information.
6. Be polite, helpful, and concise.
"""


invariants = extract_invariants(prompt)

print("\nBehavioral Invariants:\n")

for invariant in invariants:
    print(
        invariant["id"],
        "-",
        invariant["behavior"],
        "(" + invariant["type"] + ")"
    )