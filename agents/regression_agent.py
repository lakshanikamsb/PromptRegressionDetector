def detect(evaluation):
    """
    Detect prompt regression using semantic and behavioral signals.

    Important principle:
    Different wording alone should NOT cause a regression.

    A regression should mainly be detected when the new response
    changes or violates important behavior.
    """

    semantic_similarity = evaluation.get(
        "Semantic Similarity",
        0
    )

    behavioral_score = evaluation.get(
        "Behavioral Score",
        1
    )

    correctness = evaluation.get(
        "Correctness",
        1
    )

    relevance = evaluation.get(
        "Relevance",
        1
    )

    completeness = evaluation.get(
        "Completeness",
        1
    )

    instruction_following = evaluation.get(
        "Instruction Following",
        1
    )

    safety = evaluation.get(
        "Safety",
        1
    )

    latency_score = evaluation.get(
        "Latency Score",
        1
    )

    word_overlap = evaluation.get(
        "Word Overlap",
        1
    )

    # ---------------------------------------------------------
    # RULE 1
    # Critical behavioral violation
    # ---------------------------------------------------------

    failed_invariants = evaluation.get(
        "Failed Invariants",
        []
    )


    if behavioral_score < 0.50:
        return "Regression Detected"

    if len(failed_invariants) > 0:
            if behavioral_score < 0.75:
                return "Regression Detected"
    

    # ---------------------------------------------------------
    # RULE 2
    # Serious correctness problem
    # ---------------------------------------------------------

    if correctness < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 3
    # Serious safety problem
    # ---------------------------------------------------------

    if safety < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 4
    # Serious instruction-following problem
    # ---------------------------------------------------------

    if instruction_following < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 5
    # Relevance problem
    # ---------------------------------------------------------

    if relevance < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 6
    # Semantic similarity is only considered a regression
    # when behavioral preservation is also weak.
    #
    # This prevents different wording from being incorrectly
    # classified as a regression.
    # ---------------------------------------------------------

    if semantic_similarity < 0.35:

        if behavioral_score < 0.75:
            return "Regression Detected"

        if correctness < 0.75:
            return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 7
    # Very low completeness combined with weak semantic
    # preservation.
    # ---------------------------------------------------------

    if completeness < 0.45:

        if semantic_similarity < 0.50:
            return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 8
    # Latency regression
    # ---------------------------------------------------------

    if latency_score < 0.30:
        return "Regression Detected"

    # ---------------------------------------------------------
    # Otherwise the behavior is considered preserved.
    # ---------------------------------------------------------

    return "No Regression"