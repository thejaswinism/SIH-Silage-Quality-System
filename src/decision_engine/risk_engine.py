def calculate_risk(sample):
    """
    Calculate an initial risk level from ML, visual,
    and sensor inputs.

    NOTE:
    Thresholds in this prototype are placeholders.
    They must be replaced with scientifically validated
    rules before final deployment.
    """

    risk_score = 0

    # -------------------------
    # ML QUALITY
    # -------------------------
    ml_quality = sample["ml"]["quality"]

    if ml_quality == "GOOD":
        risk_score += 0

    elif ml_quality == "MEDIUM":
        risk_score += 1

    elif ml_quality == "POOR":
        risk_score += 2

    # -------------------------
    # VISUAL INDICATORS
    # -------------------------
    visual = sample["visual"]

    if visual["mould_indicator"] == "possible":
        risk_score += 2

    elif visual["mould_indicator"] == "present":
        risk_score += 3

    if visual["appearance"] == "abnormal":
        risk_score += 1

    # -------------------------
    # SENSOR INDICATORS
    # -------------------------
    sensor = sample["sensor"]

    temperature = sensor["temperature"]

    if temperature > 35:
        risk_score += 2

    elif temperature > 30:
        risk_score += 1

    # -------------------------
    # FINAL RISK
    # -------------------------
    if risk_score <= 1:
        risk = "LOW"

    elif risk_score <= 3:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    return risk_score, risk
