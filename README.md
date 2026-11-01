Prompt Regression Detector is an Agentic AI application that compares prompt versions, generates test cases, evaluates LLM responses, and detects regressions.
Its unique "Behavioral Invariant Agent" identifies essential behaviors that must remain consistent across prompt updates, helping detect behavioral changes that simple response similarity checks may miss. The system also analyzes regression issues, suggests prompt improvements, and supports retesting before deployment.
Technologies: Python, Generative AI, LLMs, Agentic AI, Prompt Engineering, Automated Testing, MLOps.

## CI/CD

GitHub Actions runs the pytest suite, checks Python syntax, and executes the
real regression pipeline using deterministic mocked LLM responses. The fresh
reports from that run are checked by `quality_gate.py` and uploaded as workflow
artifacts; checked-in reports are not used as CI results. Performance
regressions are reported separately and are advisory for the functional quality
gate. Functional regressions and behavioral preservation below 0.75 fail the
gate.

The separate Groq integration job runs only after a push to `main`. Configure
the `GROQ_API_KEY` repository Actions secret to enable it. Unit tests and the
mocked pipeline do not make real Groq API calls.

## Streamlit Community Cloud deployment

Deployment is not automatic and has not been performed. To deploy this
repository:

1. In Streamlit Community Cloud, create an app from this GitHub repository,
   select the `main` branch, and set the main file to `dashboard.py`.
2. In the app settings, add `GROQ_API_KEY` to the **Secrets** section (for
   example, `GROQ_API_KEY = "your-key"`). Do not commit the key or a local
   `.streamlit/secrets.toml` file.
3. Keep the GitHub Actions Groq secret separate from the Streamlit app secret;
   each is configured in its own settings page.

For controlled releases, enable a GitHub branch protection rule for `main`
that requires the CI test job to pass before merging. Streamlit Community
Cloud deployment, account access, secret setup, and branch protection must be
configured and verified in their respective external services.
