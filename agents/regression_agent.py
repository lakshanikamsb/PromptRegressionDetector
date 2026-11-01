
def detect(evaluation):
    """
    Detect functional and performance regressions separately.

    Different wording alone should not cause a regression.
    Latency is retained as a separate performance signal.
    """

    semantic_similarity = evaluation.get("Semantic Similarity", 1)
    behavioral_score = evaluation.get("Behavioral Score", 1)
    correctness = evaluation.get("Correctness", 1)
    relevance = evaluation.get("Relevance", 1)
    completeness = evaluation.get("Completeness", 1)
    instruction_following = evaluation.get("Instruction Following", 1)
    safety = evaluation.get("Safety", 1)

    # Keep latency information for performance analysis
    latency_score = evaluation.get("Latency Score", 1)

    failed_invariants = evaluation.get("Failed Invariants", [])

    # ---------------------------------------------------------
    # RULE 1: Critical behavioral violation
    # ---------------------------------------------------------

    if behavioral_score < 0.50:
        return "Regression Detected"

    if len(failed_invariants) > 0 and behavioral_score < 0.75:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 2: Serious correctness problem
    # ---------------------------------------------------------

    if correctness < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 3: Serious safety problem
    # ---------------------------------------------------------

    if safety < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 4: Serious instruction-following problem
    # ---------------------------------------------------------

    if instruction_following < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 5: Relevance problem
    # ---------------------------------------------------------

    if relevance < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 6: Semantic similarity with weak behavior
    # ---------------------------------------------------------

    if semantic_similarity < 0.35:
        if behavioral_score < 0.75 or correctness < 0.75:
            return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 7: Low completeness with weak semantic preservation
    # ---------------------------------------------------------

    if completeness < 0.45 and semantic_similarity < 0.50:
        return "Regression Detected"

    # ---------------------------------------------------------
    # RULE 8: Performance / latency regression
    # ---------------------------------------------------------

    if latency_score < 0.30:
        return "Performance Regression"

    # ---------------------------------------------------------
    # Otherwise
    # ---------------------------------------------------------

    return "No Regression"
