def generate_advisory(risk, sample):
    """
    Generate a farmer-friendly advisory based on
    detected risk and available indicators.

    NOTE:
    These are preliminary prototype rules.
    They must be replaced with scientifically validated
    rules from the microbiology research.
    """

    mould = sample["visual"].get("mould_indicator", "none")

    if mould in ["possible", "present"]:
        return (
            "Possible visual spoilage indicator detected. "
            "Inspect the silage carefully and consider "
            "laboratory confirmation before feeding."
        )

    if risk == "HIGH":
        return (
            "High-risk indicators detected. "
            "Further inspection or laboratory testing "
            "is recommended before feeding."
        )

    if risk == "MEDIUM":
        return (
            "Some quality-risk indicators were detected. "
            "Monitor storage conditions and reassess the silage."
        )

    return (
        "No major risk indicators detected in this screening. "
        "Continue proper storage and routine monitoring."
    )
