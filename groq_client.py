import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def get_api_key():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        st = None
        StreamlitSecretNotFoundError = None

    if st is not None:
        try:
            api_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except StreamlitSecretNotFoundError:
            pass

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Set it in the environment, your local "
            ".env file, or Streamlit Community Cloud secrets."
        )
    return api_key

def create_groq_client():
    return Groq(api_key=get_api_key())


client = create_groq_client()


def generate_response(prompt):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content