import streamlit as st
import pandas as pd
import time

from pipeline import run_regression

from agents.testing_agent import run_prompt
from agents.evaluation_agent import (
    evaluate,
    latency_score,
    semantic_evaluate_batch
)
from agents.regression_agent import detect
from agents.behavioral_invariant_agent import (
    check_invariants_batch,
    calculate_preservation_score
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Prompt Regression Detector",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("Prompt Regression Detector")

st.write(
    "AI-Powered Prompt Testing and MLOps Regression Detection"
)


# =========================================================
# PROMPT INPUT
# =========================================================

st.subheader("Prompt Versions")

prompt_v1 = st.text_area(
    "Prompt Version 1",
    height=200,
    key="prompt_input_v1"
)

prompt_v2 = st.text_area(
    "Prompt Version 2",
    height=200,
    key="prompt_input_v2"
)


# =========================================================
# RUN REGRESSION
# =========================================================

if st.button(
    "Run Regression",
    type="primary"
):

    if not prompt_v1.strip() or not prompt_v2.strip():

        st.warning(
            "Please enter both Prompt Version 1 and Prompt Version 2."
        )

    else:

        with st.spinner(
            "Running complete prompt regression pipeline..."
        ):

            try:

                results = run_regression(
                    prompt_v1,
                    prompt_v2
                )

                # -----------------------------------------
                # Store main results
                # -----------------------------------------

                st.session_state["report"] = (
                    results.get("report", [])
                )

                st.session_state["optimization"] = (
                    results.get("optimization", "")
                )

                st.session_state["dependency"] = (
                    results.get("dependency", {})
                )

                st.session_state["reproducibility"] = (
                    results.get("reproducibility", {})
                )

                # -----------------------------------------
                # Behavioral invariant results
                # -----------------------------------------

                st.session_state["invariants"] = (
                    results.get("invariants", [])
                )

                st.session_state["behavioral_results"] = (
                    results.get("behavioral_results", [])
                )

                st.session_state["behavioral_preservation_score"] = (
                    results.get(
                        "behavioral_preservation_score",
                        0
                    )
                )

                # -----------------------------------------
                # Store prompts
                # -----------------------------------------

                st.session_state["prompt_v1"] = prompt_v1
                st.session_state["prompt_v2"] = prompt_v2

                # -----------------------------------------
                # Clear old retest results
                # -----------------------------------------

                if "retest_results" in st.session_state:
                    del st.session_state["retest_results"]

                st.success(
                    "Regression testing completed successfully."
                )

            except Exception as e:

                st.error(
                    f"Regression pipeline failed: {e}"
                )
                st.exception(e)

# =========================================================
# DISPLAY RESULTS
# =========================================================

if "report" in st.session_state:

    report = st.session_state.get(
        "report",
        []
    )

    optimization = st.session_state.get(
        "optimization",
        ""
    )

    dependency_report = st.session_state.get(
        "dependency",
        {}
    )

    reproducibility_report = st.session_state.get(
        "reproducibility",
        {}
    )

    invariants = st.session_state.get(
        "invariants",
        []
    )

    behavioral_results = st.session_state.get(
        "behavioral_results",
        []
    )

    behavioral_score = st.session_state.get(
        "behavioral_preservation_score",
        0
    )

    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame(report)

    # =====================================================
    # REGRESSION SUMMARY
    # =====================================================

    st.divider()

    st.subheader("Regression Summary")

    total_tests = len(df)

    if "Status" in df.columns:

        regressions = len(
            df[
                df["Status"] == "Regression Detected"
            ]
        )

    else:

        regressions = 0

    if "Similarity" in df.columns:

        average_similarity = round(
            df["Similarity"].mean(),
            2
        )

    else:

        average_similarity = 0

    if "Overall Score" in df.columns:

        average_overall_score = round(
            df["Overall Score"].mean(),
            2
        )

    else:

        average_overall_score = 0

    # =====================================================
    # CONSISTENCY / REPRODUCIBILITY
    # =====================================================

    consistency = 0

    if isinstance(
        reproducibility_report,
        dict
    ):

        if "reproducibility_score" in reproducibility_report:

            consistency = reproducibility_report[
                "reproducibility_score"
            ]

        elif "score" in reproducibility_report:

            consistency = reproducibility_report[
                "score"
            ]

    # =====================================================
    # TOP METRICS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Tests",
        total_tests
    )

    c2.metric(
        "Regressions",
        regressions
    )

    c3.metric(
        "Average Similarity",
        average_similarity
    )

    c4.metric(
        "Overall Score",
        average_overall_score
    )

    c5.metric(
        "Consistency",
        consistency
    )

    # =====================================================
    # BEHAVIORAL INVARIANT ANALYSIS
    # =====================================================

    st.divider()

    st.subheader(
        "Behavioral Invariant Analysis"
    )

    if len(invariants) == 0:

        st.info(
            "No behavioral invariants were extracted."
        )

    else:

        st.write(
            f"**{len(invariants)} behavioral invariant(s) "
            f"were extracted from Prompt V1.**"
        )

        for invariant in invariants:

            invariant_id = invariant.get(
                "id",
                "?"
            )

            behavior = invariant.get(
                "behavior",
                ""
            )

            invariant_type = invariant.get(
                "type",
                "required"
            )

            st.markdown(
                f"### Invariant {invariant_id}"
            )

            st.write(
                behavior
            )

            st.caption(
                f"Type: {invariant_type}"
            )

    # -----------------------------------------------------
    # Behavioral Preservation Score
    # -----------------------------------------------------

    st.subheader(
        "Behavioral Preservation"
    )

    st.metric(
        "Behavioral Preservation Score",
        f"{float(behavioral_score):.2f}"
    )

    if behavioral_score >= 0.75:

        st.success(
            "Important behaviors from Prompt V1 are mostly preserved."
        )

    elif behavioral_score >= 0.50:

        st.warning(
            "Some important behaviors may have changed."
        )

    else:

        st.error(
            "Significant behavioral regression detected."
        )

    # -----------------------------------------------------
    # Individual invariant results
    # -----------------------------------------------------

    if len(behavioral_results) > 0:

        st.subheader(
            "Invariant Verification Results"
        )

        for result_item in behavioral_results:

            test_id = result_item.get(
                "id",
                "?"
            )

            score = float(
                result_item.get(
                    "behavioral_score",
                    0
                )
            )

            passed = result_item.get(
                "passed_invariants",
                []
            )

            failed = result_item.get(
                "failed_invariants",
                []
            )

            reason = result_item.get(
                "reason",
                ""
            )

            st.markdown(
                f"#### Test Case {test_id}"
            )

            if score >= 0.75:

                st.success(
                    f"Behavior Preserved — Score: {score:.2f}"
                )

            elif score >= 0.50:

                st.warning(
                    f"Partially Preserved — Score: {score:.2f}"
                )

            else:

                st.error(
                    f"Behavioral Regression — Score: {score:.2f}"
                )

            st.write(
                f"**Passed Invariants:** {passed}"
            )

            st.write(
                f"**Failed Invariants:** {failed}"
            )

            if reason:

                st.write(
                    f"**Reason:** {reason}"
                )

    # =====================================================
    # LATENCY ANALYSIS
    # =====================================================

    st.divider()

    st.subheader(
        "Latency Analysis"
    )

    if (
        "V1 Latency" in df.columns
        and "V2 Latency" in df.columns
        and "Latency Increase" in df.columns
    ):

        average_v1_latency = round(
            df["V1 Latency"].mean(),
            2
        )

        average_v2_latency = round(
            df["V2 Latency"].mean(),
            2
        )

        average_latency_change = round(
            df["Latency Increase"].mean(),
            2
        )

        l1, l2, l3 = st.columns(3)

        l1.metric(
            "V1 Average Latency",
            f"{average_v1_latency}s"
        )

        l2.metric(
            "V2 Average Latency",
            f"{average_v2_latency}s"
        )

        l3.metric(
            "Latency Change",
            f"{average_latency_change}s"
        )

    else:

        st.info(
            "Latency information is not available."
        )

    # =====================================================
    # EVALUATION REPORT
    # =====================================================

    st.divider()

    st.subheader(
        "Evaluation Report"
    )

    if len(df) > 0:

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "No evaluation results are available."
        )

    # =====================================================
    # OPTIMIZATION
    # =====================================================

    st.divider()

    st.subheader(
        "Optimization Suggestions"
    )

    if optimization:

        st.write(
            optimization
        )

    else:

        st.info(
            "No optimization suggestions are available."
        )

    # =====================================================
    # DEPENDENCY IMPACT
    # =====================================================

    st.divider()

    st.subheader(
        "Prompt Dependency Impact Analysis"
    )

    if isinstance(
        dependency_report,
        dict
    ):

        st.write(
            "**Old Dependencies**"
        )

        st.write(
            dependency_report.get(
                "old_dependencies",
                []
            )
        )

        st.write(
            "**New Dependencies**"
        )

        st.write(
            dependency_report.get(
                "new_dependencies",
                []
            )
        )

        st.write(
            "**Added Dependencies**"
        )

        st.write(
            dependency_report.get(
                "added_dependencies",
                []
            )
        )

        st.write(
            "**Removed Dependencies**"
        )

        st.write(
            dependency_report.get(
                "removed_dependencies",
                []
            )
        )

        if "impact_level" in dependency_report:

            st.write(
                f"**Impact Level:** "
                f"{dependency_report['impact_level']}"
            )

        if "explanation" in dependency_report:

            st.write(
                "**Explanation**"
            )

            st.write(
                dependency_report["explanation"]
            )

        if "affected_test_types" in dependency_report:

            st.write(
                "**Affected Test Types**"
            )

            st.write(
                dependency_report[
                    "affected_test_types"
                ]
            )

    else:

        st.info(
            "Dependency analysis information is not available."
        )

    # =====================================================
    # REPRODUCIBILITY
    # =====================================================

    st.divider()

    st.subheader(
        "Prompt Reproducibility Engine"
    )

    if isinstance(
        reproducibility_report,
        dict
    ):

        reproducibility_score = reproducibility_report.get(
            "overall_reproducibility_score",
            reproducibility_report.get(
                "reproducibility_score",
                reproducibility_report.get(
                    "score",
                    0
                )
            )
        )

        st.metric(
            "Reproducibility Score",
            reproducibility_score
        )

        if "status" in reproducibility_report:

            st.write(
                f"**Reproducibility Status:** "
                f"{reproducibility_report['status']}"
            )

        if "results" in reproducibility_report:

            st.write(
                "**Repeated Execution Results**"
            )

            st.write(
                reproducibility_report[
                    "results"
                ]
            )

    else:

        st.info(
            "Reproducibility information is not available."
        )

    # =====================================================
    # RETEST REGRESSION CASES
    # =====================================================

    st.divider()

    st.subheader(
        "Retest Regression Cases"
    )

    if "Status" in df.columns:

        regression_cases = df[
            df["Status"] == "Regression Detected"
        ]

    else:

        regression_cases = pd.DataFrame()

    if len(regression_cases) == 0:

        st.success(
            "No regression cases found. "
            "Retesting is not required."
        )

    else:

        st.write(
            f"{len(regression_cases)} regression "
            f"test case(s) were detected."
        )

        st.write(
            "Click the button below to run these "
            "regression cases again."
        )

        # =================================================
        # RETEST BUTTON
        # =================================================

        if st.button(
            "Retest Regression Cases"
        ):

            prompt_v1_saved = st.session_state.get(
                "prompt_v1",
                ""
            )

            prompt_v2_saved = st.session_state.get(
                "prompt_v2",
                ""
            )

            saved_invariants = st.session_state.get(
                "invariants",
                []
            )

            optimized_prompt = st.session_state.get(
                "optimized_prompt",
                prompt_v2
            )

            retest_results = []

            with st.spinner(
                "Retesting regression cases..."
            ):

                # -------------------------------------------------
                # Prepare cases for semantic + invariant evaluation
                # -------------------------------------------------

                semantic_cases = []

                raw_retest_data = []

                for _, case in regression_cases.iterrows():

                    test_input = case[
                        "Test Case"
                    ]

                    # ---------------------------------------------
                    # Run V1
                    # ---------------------------------------------

                    start_time = time.perf_counter()

                    old_output = run_prompt(
                        prompt_v1_saved,
                        test_input
                    )

                    end_time = time.perf_counter()

                    old_latency = round(
                        end_time - start_time,
                        2
                    )

                    # ---------------------------------------------
                    # Run V2
                    # ---------------------------------------------

                    start_time = time.perf_counter()
                           
                    new_output = run_prompt(
                        optimized_prompt,
                        test_input
                    )

                    end_time = time.perf_counter()

                    new_latency = round(
                        end_time - start_time,
                        2
                    )

                    raw_retest_data.append({

                        "Test Case":
                            test_input,

                        "Old Output":
                            old_output,

                        "New Output":
                            new_output,

                        "V1 Latency":
                            old_latency,

                        "V2 Latency":
                            new_latency

                    })

                    semantic_cases.append({

                        "id":
                            len(semantic_cases) + 1,

                        "test_case":
                            test_input,

                        "old_output":
                            old_output,

                        "new_output":
                            new_output

                    })

                # -------------------------------------------------
                # Semantic evaluation
                # -------------------------------------------------

                semantic_results = []

                try:

                    semantic_results = (
                        semantic_evaluate_batch(
                            semantic_cases
                        )
                    )

                except Exception as e:

                    st.warning(
                        "Semantic evaluation during retest "
                        f"could not be completed: {e}"
                    )

                # -------------------------------------------------
                # Behavioral invariant evaluation
                # -------------------------------------------------

                invariant_cases = []

                for index, item in enumerate(
                    raw_retest_data
                ):

                    invariant_cases.append({

                        "id":
                            index + 1,

                        "test_case":
                            item["Test Case"],

                        "output":
                            item["New Output"]

                    })

                invariant_results = []

                if len(saved_invariants) > 0:

                    try:

                        invariant_results = (
                            check_invariants_batch(
                                saved_invariants,
                                invariant_cases
                            )
                        )

                    except Exception as e:

                        st.warning(
                            "Behavioral invariant verification "
                            f"during retest could not be completed: {e}"
                        )

                # -------------------------------------------------
                # Build final retest results
                # -------------------------------------------------

                for index, item in enumerate(
                    raw_retest_data
                ):

                    old_output = item[
                        "Old Output"
                    ]

                    new_output = item[
                        "New Output"
                    ]

                    old_latency = item[
                        "V1 Latency"
                    ]

                    new_latency = item[
                        "V2 Latency"
                    ]

                    # ---------------------------------------------
                    # Traditional evaluation
                    # ---------------------------------------------

                    evaluation = evaluate(
                        old_output,
                        new_output
                    )

                    # ---------------------------------------------
                    # Semantic result
                    # ---------------------------------------------

                    semantic_result = {}

                    if index < len(
                        semantic_results
                    ):

                        semantic_result = (
                            semantic_results[index]
                        )

                    semantic_similarity = float(
                        semantic_result.get(
                            "semantic_similarity",
                            evaluation.get(
                                "Similarity",
                                0
                            )
                        )
                    )

                    correctness = float(
                        semantic_result.get(
                            "correctness",
                            1
                        )
                    )

                    relevance = float(
                        semantic_result.get(
                            "relevance",
                            1
                        )
                    )

                    completeness = float(
                        semantic_result.get(
                            "completeness",
                            1
                        )
                    )

                    instruction_following = float(
                        semantic_result.get(
                            "instruction_following",
                            1
                        )
                    )

                    safety = float(
                        semantic_result.get(
                            "safety",
                            1
                        )
                    )

                    semantic_reason = (
                        semantic_result.get(
                            "reason",
                            ""
                        )
                    )

                    # ---------------------------------------------
                    # Behavioral invariant result
                    # ---------------------------------------------

                    invariant_result = {}

                    if index < len(
                        invariant_results
                    ):

                        invariant_result = (
                            invariant_results[index]
                        )

                    behavioral_score = float(
                        invariant_result.get(
                            "behavioral_score",
                            1
                        )
                    )

                    passed_invariants = (
                        invariant_result.get(
                            "passed_invariants",
                            []
                        )
                    )

                    failed_invariants = (
                        invariant_result.get(
                            "failed_invariants",
                            []
                        )
                    )

                    invariant_reason = (
                        invariant_result.get(
                            "reason",
                            ""
                        )
                    )

                    # ---------------------------------------------
                    # Latency
                    # ---------------------------------------------

                    latency_metric = latency_score(
                        old_latency,
                        new_latency
                    )

                    latency_increase = round(
                        new_latency - old_latency,
                        2
                    )

                    # ---------------------------------------------
                    # Calculate final retest score
                    # ---------------------------------------------

                    overall_score = (
                        semantic_similarity * 0.30
                        + correctness * 0.15
                        + relevance * 0.10
                        + completeness * 0.10
                        + instruction_following * 0.10
                        + safety * 0.05
                        + behavioral_score * 0.20
                    )

                    overall_score = round(
                        overall_score,
                        2
                    )

                    # ---------------------------------------------
                    # Prepare detection input
                    # ---------------------------------------------

                    detection_input = {

                        "Similarity":
                            evaluation.get(
                                "Similarity",
                                0
                            ),

                        "Word Overlap":
                            evaluation.get(
                                "Word Overlap",
                                0
                            ),

                        "Length Similarity":
                            evaluation.get(
                                "Length Similarity",
                                0
                            ),

                        "Information Preservation":
                            evaluation.get(
                                "Information Preservation",
                                0
                            ),

                        "Semantic Similarity":
                            semantic_similarity,

                        "Correctness":
                            correctness,

                        "Relevance":
                            relevance,

                        "Completeness":
                            completeness,

                        "Instruction Following":
                            instruction_following,

                        "Safety":
                            safety,

                        "Behavioral Score":
                            behavioral_score,

                        "Latency Score":
                            latency_metric,

                        "Failed Invariants":
                            failed_invariants

                    }

                    # ---------------------------------------------
                    # Regression detection
                    # ---------------------------------------------

                    status = detect(
                        detection_input
                    )

                    # ---------------------------------------------
                    # Store retest result
                    # ---------------------------------------------

                    retest_results.append({

                        "Test Case":
                            item["Test Case"],

                        "Old Output":
                            old_output,

                        "New Output":
                            new_output,

                        "Similarity":
                            evaluation.get(
                                "Similarity",
                                0
                            ),

                        "Semantic Similarity":
                            semantic_similarity,

                        "Word Overlap":
                            evaluation.get(
                                "Word Overlap",
                                0
                            ),

                        "Length Similarity":
                            evaluation.get(
                                "Length Similarity",
                                0
                            ),

                        "Information Preservation":
                            evaluation.get(
                                "Information Preservation",
                                0
                            ),

                        "Correctness":
                            correctness,

                        "Relevance":
                            relevance,

                        "Completeness":
                            completeness,

                        "Instruction Following":
                            instruction_following,

                        "Safety":
                            safety,

                        "Behavioral Score":
                            behavioral_score,

                        "Passed Invariants":
                            passed_invariants,

                        "Failed Invariants":
                            failed_invariants,

                        "Overall Score":
                            overall_score,

                        "V1 Latency":
                            old_latency,

                        "V2 Latency":
                            new_latency,

                        "Latency Increase":
                            latency_increase,

                        "Latency Score":
                            latency_metric,

                        "Semantic Reason":
                            semantic_reason,

                        "Invariant Reason":
                            invariant_reason,

                        "Status":
                            status

                    })

            # =================================================
            # STORE RETEST RESULTS
            # =================================================

            st.session_state[
                "retest_results"
            ] = retest_results

            st.success(
                "Retesting completed successfully."
            )


    # =====================================================
    # DISPLAY RETEST RESULTS
    # =====================================================

    if "retest_results" in st.session_state:

        retest_results = st.session_state[
            "retest_results"
        ]

        if len(retest_results) > 0:

            retest_df = pd.DataFrame(
                retest_results
            )

            st.subheader(
                "Retest Results"
            )

            st.dataframe(
                retest_df,
                use_container_width=True
            )

            # =================================================
            # RETEST SUMMARY
            # =================================================

            retest_regressions = len(
                retest_df[
                    retest_df["Status"]
                    == "Regression Detected"
                ]
            )

            retest_average = round(
                retest_df[
                    "Similarity"
                ].mean(),
                2
            )

            retest_overall = round(
                retest_df[
                    "Overall Score"
                ].mean(),
                2
            )

            retest_behavioral = round(
                retest_df[
                    "Behavioral Score"
                ].mean(),
                2
            )

            r1, r2, r3, r4, r5 = st.columns(5)

            r1.metric(
                "Retested Cases",
                len(retest_df)
            )

            r2.metric(
                "Remaining Regressions",
                retest_regressions
            )

            r3.metric(
                "Retest Average Similarity",
                retest_average
            )

            r4.metric(
                "Retest Overall Score",
                retest_overall
            )

            r5.metric(
                "Retest Behavioral Score",
                retest_behavioral
            )

            # =================================================
            # RETEST MESSAGE
            # =================================================

            if retest_regressions == 0:

                st.success(
                    "All previously detected regression "
                    "cases passed the retest."
                )

            else:

                st.warning(
                    f"{retest_regressions} regression case(s) "
                    "still remain after retesting."
                )