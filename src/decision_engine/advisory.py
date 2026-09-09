def generate_advisory(risk, sample):
    """
    Generate a farmer-friendly advisory based on
    the detected risk and available indicators.

    NOTE:
    Advisory rules are preliminary placeholders.
    They must be validated using the microbiology team's
    research and reliable scientific sources.
    """

    mould = sample["visual"].get("mould_indicator", "none")

    # Possible or visible mould indicator
    if mould in ["possible", "present"]:
        return (
            "Possible visual spoilage indicator detected. "
            "Inspect the silage carefully and consider "
            "laboratory confirmation before feeding."
        )

    # High risk
    if risk == "HIGH":
        return (
            "High-risk indicators detected. "
            "Further inspection or laboratory testing "
            "is recommended before feeding."
        )

    # Medium risk
    if risk == "MEDIUM":
        return (
            "Some quality-risk indicators were detected. "
            "Monitor storage conditions and reassess the silage."
        )

    # Low risk
    return (
        "No major risk indicators detected in this screening. "
        "Continue proper storage and routine monitoring."
    )
