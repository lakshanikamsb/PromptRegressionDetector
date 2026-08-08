from pipeline import run_regression
import json
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Prompt Regression Detector",
    layout="wide"
)

st.title("Prompt Regression Detector")
prompt_v1 = st.text_area(
    "Prompt Version 1",
    height=200
)

prompt_v2 = st.text_area(
    "Prompt Version 2",
    height=200
)
# ----------------------------
# Load Evaluation Report
# ----------------------------

if st.button("Run Regression"):

    with st.spinner("Running Regression..."):

        results = run_regression(
            prompt_v1,
            prompt_v2
        )

    report = results["report"]
    optimization = results["optimization"]

    df = pd.DataFrame(report)

    # ----------------------------
    # Metrics
    # ----------------------------

    total_tests = len(df)

    regressions = len(
        df[df["Status"] == "Regression Detected"]
    )

    average_similarity = round(
        df["Similarity"].mean(),
        2
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Tests", total_tests)
    c2.metric("Regressions", regressions)
    c3.metric("Average Similarity", average_similarity)

    st.divider()

    st.subheader("Evaluation Report")
    st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("Optimization Suggestions")

    st.write(optimization)