import streamlit

import groq_client


def test_get_api_key_reads_streamlit_cloud_secret(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        streamlit,
        "secrets",
        {"GROQ_API_KEY": "cloud-test-key"},
    )

    assert groq_client.get_api_key() == "cloud-test-key"


def test_get_api_key_reports_missing_secret_safely(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(streamlit, "secrets", {})

    try:
        groq_client.get_api_key()
    except ValueError as error:
        assert "GROQ_API_KEY is missing" in str(error)
    else:
        raise AssertionError("A missing key should produce a clear error.")
