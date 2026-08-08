def detect(score):

    if score < 0.80:
        return "Regression Detected"

    return "No Regression"