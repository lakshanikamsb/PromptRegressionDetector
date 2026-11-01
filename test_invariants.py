
import json
from types import SimpleNamespace

from agents import behavioral_invariant_agent as agent


def test_extract_invariants(monkeypatch):

    expected = [
        {
            "id": "B1",
            "description": "Ask for the order ID",
            "importance": "high"
        },
        {
            "id": "B2",
            "description": "Check refund eligibility",
            "importance": "high"
        }
    ]

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(expected)
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return fake_response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=fake_create
            )
        )
    )

    monkeypatch.setattr(agent, "client", fake_client)

    result = agent.extract_invariants("Customer support prompt")

    assert result == expected
    assert len(result) == 2


def test_calculate_preservation_score():

    results = [
        {"behavioral_score": 1.0},
        {"behavioral_score": 0.8}
    ]

    score = agent.calculate_preservation_score(results)

    assert score == 0.9


def test_empty_invariant_results():

    result = agent.calculate_preservation_score([])

    assert result == 0.0


def test_empty_test_cases():

    result = agent.check_invariants_batch(
        [{"id": "B1"}],
        []
    )

    assert result == []
