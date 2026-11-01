
from unittest.mock import patch

from test_groq import build_prompt, run_agent


def test_build_prompt_customer_support():

    result = build_prompt(
        "customer_support",
        "I need a refund"
    )

    assert "helpful customer support assistant" in result
    assert "I need a refund" in result


def test_build_prompt_coding_assistant():

    result = build_prompt(
        "coding_assistant",
        "Write a Python program"
    )

    assert "software engineer" in result
    assert "Write a Python program" in result


def test_build_prompt_unknown_agent():

    result = build_prompt(
        "unknown",
        "Hello"
    )

    assert "helpful assistant" in result


@patch("test_groq.generate_response")
def test_run_agent(mock_generate):

    mock_generate.return_value = "Your refund request is received."

    result = run_agent(
        "customer_support",
        "I need a refund"
    )

    assert result == "Your refund request is received."
    mock_generate.assert_called_once()
