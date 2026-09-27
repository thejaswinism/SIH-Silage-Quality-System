
import os
import streamlit as st
import torch
import joblib
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GauChara",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PATHS
# ============================================================

PROJECT = os.path.dirname(os.path.abspath(__file__))

FBSI_MODEL_PATH = os.path.join(
    PROJECT,
    "models",
    "FBSI_MobileNetV3_best.pth"
)

# ============================================================
# FBSI MODEL
# ============================================================

FBSI_CLASSES = [
    "Score 0",
    "Score 1",
    "Score 1-2",
    "Score 2",
    "Score 3",
    "Score 4"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

@st.cache_resource
def load_fbsi_model():

    model = models.mobilenet_v3_small(
        weights=None
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(FBSI_CLASSES)
    )

    model.load_state_dict(
        torch.load(
            FBSI_MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


fbsi_model = load_fbsi_model()

fbsi_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_fbsi(image):

    image = image.convert("RGB")

    tensor = fbsi_transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = fbsi_model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    prediction = FBSI_CLASSES[
        predicted.item()
    ]

    return prediction, float(confidence.item())




# ============================================================
# SENSOR RISK MODEL
# ============================================================

SENSOR_MODEL_PATH = os.path.join(
    PROJECT,
    "models",
    "Sensor_Risk_Best_Model.joblib"
)

SENSOR_FEATURES = [
    "internaltemperature",
    "externaltemperature",
    "internalhumidity",
    "percentagelevel",
    "kglevel",
    "delta_temp",
    "temp_rate_C_per_hr",
    "simulated_pH",
    "simulated_moisture_pct",
    "simulated_CO2_ppm"
]

@st.cache_resource
def load_sensor_model():

    return joblib.load(
        SENSOR_MODEL_PATH
    )


sensor_model = load_sensor_model()


def predict_sensor(sensor_input):

    X = [[
        sensor_input[f]
        for f in SENSOR_FEATURES
    ]]

    prediction = sensor_model.predict(X)[0]

    confidence = None

    if hasattr(sensor_model, "predict_proba"):

        probabilities = sensor_model.predict_proba(X)[0]

        confidence = float(
            max(probabilities)
        )

    return prediction, confidence




# ============================================================
# PADOVA SILAGE QUALITY MODEL
# ============================================================

PADOVA_MODEL_PATH = os.path.join(
    PROJECT,
    "models",
    "Padova_Silage_Risk_Best_Model.joblib"
)

PADOVA_NUMERICAL_FEATURES = [
    "dm.f",
    "ash.f",
    "cp.f",
    "ee.f",
    "ndf.f",
    "adf.f",
    "lignin.f",
    "wsc.f",
    "starch.f"
]

PADOVA_CATEGORICAL_FEATURES = [
    "method",
    "inhoculant",
    "dose",
    "delay",
    "fao.class",
    "maturity",
    "field"
]

PADOVA_FEATURES = (
    PADOVA_NUMERICAL_FEATURES
    + PADOVA_CATEGORICAL_FEATURES
)

@st.cache_resource
def load_padova_model():

    return joblib.load(
        PADOVA_MODEL_PATH
    )


padova_model = load_padova_model()


def predict_padova(padova_input):

    X = [[
        padova_input[f]
        for f in PADOVA_FEATURES
    ]]

    prediction = padova_model.predict(X)[0]

    confidence = None

    if hasattr(
        padova_model,
        "predict_proba"
    ):

        probabilities = padova_model.predict_proba(X)[0]

        confidence = float(
            max(probabilities)
        )

    return prediction, confidence




# ============================================================
# AI EVIDENCE FUSION
# ============================================================

def fuse_results(
    fbsi_result,
    sensor_result,
    padova_result=None
):

    fbsi = fbsi_result["prediction"]

    sensor = sensor_result["prediction"]

    padova = None

    if padova_result is not None:
        padova = padova_result["prediction"]

    # --------------------------------------------------------
    # VISUAL EVIDENCE
    # --------------------------------------------------------

    visual_high = fbsi in [
        "Score 3",
        "Score 4"
    ]

    visual_low = fbsi in [
        "Score 0",
        "Score 1",
        "Score 1-2"
    ]

    # --------------------------------------------------------
    # SENSOR EVIDENCE
    # --------------------------------------------------------

    sensor_high = sensor == (
        "HIGH_DETERIORATION_INDICATORS"
    )

    sensor_early = sensor == (
        "EARLY_DETERIORATION_INDICATORS"
    )

    sensor_low = sensor in [
        "LOW_APPARENT_RISK",
        "NORMAL_SENSOR_BEHAVIOR"
    ]

    # --------------------------------------------------------
    # VOTING
    # --------------------------------------------------------

    high_votes = 0
    medium_votes = 0
    low_votes = 0

    # Visual
    if visual_high:
        high_votes += 1

    elif visual_low:
        low_votes += 1

    else:
        medium_votes += 1

    # Sensor
    if sensor_high:
        high_votes += 1

    elif sensor_early:
        medium_votes += 1

    elif sensor_low:
        low_votes += 1

    # Padova
    if padova == "HIGH_RISK":
        high_votes += 1

    elif padova == "MEDIUM_RISK":
        medium_votes += 1

    elif padova == "LOW_RISK":
        low_votes += 1

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if high_votes >= 2:

        final_risk = "HIGH"

        advisory = (
            "Potential quality or deterioration indicators "
            "were detected. Inspect the silo and feed before "
            "use. Check for heating, air entry, moisture "
            "problems or visible spoilage."
        )

    elif low_votes >= 2:

        final_risk = "LOW"

        advisory = (
            "No strong deterioration indicators were detected "
            "by the available screening models. Continue "
            "routine monitoring and maintain proper storage."
        )

    else:

        final_risk = "MEDIUM"

        advisory = (
            "Mixed or intermediate indicators were detected. "
            "Inspect the feed or silage and continue monitoring "
            "temperature, moisture and storage conditions."
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidences = []

    if fbsi_result.get("confidence") is not None:
        confidences.append(
            fbsi_result["confidence"]
        )

    if sensor_result.get("confidence") is not None:
        confidences.append(
            sensor_result["confidence"]
        )

    if (
        padova_result is not None
        and padova_result.get("confidence") is not None
    ):
        confidences.append(
            padova_result["confidence"]
        )

    overall_confidence = None

    if confidences:
        overall_confidence = (
            sum(confidences)
            / len(confidences)
        )

    return {

        "final_risk": final_risk,

        "fbsi": fbsi,

        "sensor": sensor,

        "padova": padova,

        "high_votes": high_votes,

        "medium_votes": medium_votes,

        "low_votes": low_votes,

        "overall_confidence":
            overall_confidence,

        "advisory": advisory
    }


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap'
);

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F3F0DD;
    color: #173B1B;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-left: 5%;
    padding-right: 5%;
    padding-bottom: 3rem;
}

.brand {
    font-size: 42px;
    font-weight: 600;
    color: #173B1B;
    margin-bottom: 35px;
}

.hero {
    text-align: center;
    margin-bottom: 45px;
}

.hero h1 {
    font-size: 44px;
    color: #173B1B;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    color: #526553;
}

.card {
    background: #E5E2CE;
    border-radius: 22px;
    padding: 30px 24px;
    min-height: 350px;
    margin-bottom: 25px;
    text-align: center;
}

.card-title {
    font-size: 27px;
    font-weight: 600;
    color: #173B1B;
    margin-bottom: 22px;
}

.card-image {
    height: 210px;
    border-radius: 20px;
    border: 5px solid #175A25;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 75px;
    background: #D9D7C3;
}

.card-description {
    margin-top: 14px;
    color: #526553;
    font-size: 15px;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid #175A25;
    color: #173B1B;
    background: #F3F0DD;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #DDE5D1;
    border-color: #175A25;
}

.result-box {
    background: #E5E2CE;
    border-radius: 20px;
    padding: 25px;
    margin-top: 20px;
}

.footer {
    margin-top: 60px;
    padding-top: 35px;
    border-top: 1px solid #D5D1BA;
    color: #526553;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


def navigate(page):
    st.session_state.page = page


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">🌿 GauChara</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">

        <h1>Smarter feed, healthier herds.</h1>

        <p>
        AI-powered rapid feed and silage quality screening
        </p>

    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:

        st.markdown("""
        <div class="card">
            <div class="card-title">Sensors</div>
            <div class="card-image">🌡️</div>
            <div class="card-description">
                Monitor silo storage conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Sensors",
            key="open_sensors",
            use_container_width=True
        ):
            navigate("sensors")
            st.rerun()

    with c2:

        st.markdown("""
        <div class="card">
            <div class="card-title">Upload Image</div>
            <div class="card-image">📸</div>
            <div class="card-description">
                Analyze feed appearance using AI.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Image",
            key="open_image",
            use_container_width=True
        ):
            navigate("image")
            st.rerun()

    with c3:

        st.markdown("""
        <div class="card">
            <div class="card-title">AI Fusion</div>
            <div class="card-image">🤖</div>
            <div class="card-description">
                Combine available AI evidence.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Fusion",
            key="open_fusion",
            use_container_width=True
        ):
            navigate("fusion")
            st.rerun()


    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:

        st.markdown("""
        <div class="card">
            <div class="card-title">Silage Quality</div>
            <div class="card-image">🌾</div>
            <div class="card-description">
                Silage quality screening.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Silage Quality",
            key="open_quality",
            use_container_width=True
        ):
            navigate("quality")
            st.rerun()

    with c2:

        st.markdown("""
        <div class="card">
            <div class="card-title">Feedback</div>
            <div class="card-image">💬</div>
            <div class="card-description">
                Share observations from the farm.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Feedback",
            key="open_feedback",
            use_container_width=True
        ):
            navigate("feedback")
            st.rerun()

    with c3:

        st.markdown("""
        <div class="card">
            <div class="card-title">Cattle Data</div>
            <div class="card-image">🐄</div>
            <div class="card-description">
                Manage cattle and feeding information.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Cattle Data",
            key="open_cattle",
            use_container_width=True
        ):
            navigate("cattle")
            st.rerun()


    # --------------------------------------------------------
    # ROW 3
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:

        st.markdown("""
        <div class="card">
            <div class="card-title">Scan History</div>
            <div class="card-image">📊</div>
            <div class="card-description">
                Review previous screenings.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open History",
            key="open_history",
            use_container_width=True
        ):
            navigate("history")
            st.rerun()

    with c2:

        st.markdown("""
        <div class="card">
            <div class="card-title">Language & Settings</div>
            <div class="card-image">🌐</div>
            <div class="card-description">
                Configure application preferences.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Settings",
            key="open_settings",
            use_container_width=True
        ):
            navigate("settings")
            st.rerun()

    with c3:

        st.markdown("""
        <div class="card">
            <div class="card-title">The GauChara Story</div>
            <div class="card-image">🌱</div>
            <div class="card-description">
                Learn about GauChara.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Story",
            key="open_story",
            use_container_width=True
        ):
            navigate("story")
            st.rerun()


# ============================================================
# IMAGE PAGE — REAL FBSI AI
# ============================================================

elif st.session_state.page == "image":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("📸 Feed Image Analysis")

    st.write(
        "Upload a feed or feed-bunk image for AI-based "
        "visual screening."
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Selected sample",
            width=500
        )

        if st.button(
            "🔍 ANALYZE IMAGE",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Running FBSI MobileNetV3..."
            ):

                prediction, confidence = predict_fbsi(image)

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.success(
                f"Visual Feed Bunk Score: {prediction}"
            )

            st.metric(
                "AI Confidence",
                f"{confidence * 100:.1f}%"
            )

            if prediction in ["Score 3", "Score 4"]:

                st.warning(
                    "Higher visual score detected. "
                    "Further inspection is recommended."
                )

            elif prediction in [
                "Score 0",
                "Score 1",
                "Score 1-2"
            ]:

                st.info(
                    "Lower visual score detected. "
                    "Continue routine monitoring."
                )

            else:

                st.info(
                    "Intermediate visual score detected. "
                    "Additional evidence can be considered."
                )

            st.caption(
                "AI visual screening only — not a laboratory measurement."
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ============================================================
# SENSOR PAGE — REAL SENSOR AI
# ============================================================

elif st.session_state.page == "sensors":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("🌡️ Silo Sensor Monitoring")

    st.write(
        "Enter current silo conditions for AI-based "
        "deterioration screening."
    )

    st.markdown("### Storage conditions")

    c1, c2 = st.columns(2)

    with c1:

        internal_temperature = st.number_input(
            "Internal Temperature (°C)",
            value=25.0
        )

        external_temperature = st.number_input(
            "External Temperature (°C)",
            value=28.0
        )

        internal_humidity = st.number_input(
            "Internal Humidity (%)",
            value=60.0
        )

        percentage_level = st.number_input(
            "Silo Level (%)",
            value=50.0
        )

        kg_level = st.number_input(
            "Quantity (kg)",
            value=100.0
        )

    with c2:

        delta_temp = st.number_input(
            "Temperature Difference (°C)",
            value=0.0
        )

        temp_rate = st.number_input(
            "Temperature Rate (°C/hr)",
            value=0.0
        )

        simulated_ph = st.number_input(
            "pH",
            value=4.0
        )

        simulated_moisture = st.number_input(
            "Moisture (%)",
            value=65.0
        )

        simulated_co2 = st.number_input(
            "CO₂ (ppm)",
            value=500.0
        )

    st.divider()

    if st.button(
        "🤖 ANALYZE SENSOR DATA",
        type="primary",
        use_container_width=True
    ):

        sensor_input = {

            "internaltemperature":
                internal_temperature,

            "externaltemperature":
                external_temperature,

            "internalhumidity":
                internal_humidity,

            "percentagelevel":
                percentage_level,

            "kglevel":
                kg_level,

            "delta_temp":
                delta_temp,

            "temp_rate_C_per_hr":
                temp_rate,

            "simulated_pH":
                simulated_ph,

            "simulated_moisture_pct":
                simulated_moisture,

            "simulated_CO2_ppm":
                simulated_co2
        }

        with st.spinner(
            "Running Sensor Risk Model..."
        ):

            prediction, confidence = predict_sensor(
                sensor_input
            )

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.success(
            f"Sensor Assessment: **{prediction}**"
        )

        if confidence is not None:

            st.metric(
                "AI Confidence",
                f"{confidence * 100:.1f}%"
            )

        if prediction == "HIGH_DETERIORATION_INDICATORS":

            st.error(
                "High deterioration indicators detected. "
                "Inspect the silo and storage conditions."
            )

        elif prediction == "EARLY_DETERIORATION_INDICATORS":

            st.warning(
                "Early deterioration indicators detected. "
                "Continue close monitoring."
            )

        elif prediction == "SENSOR_FAULT":

            st.warning(
                "The model indicates a possible sensor fault. "
                "Check the sensor readings before interpreting "
                "the result."
            )

        elif prediction == "LOW_APPARENT_RISK":

            st.success(
                "Low apparent storage risk according to "
                "the sensor model."
            )

        elif prediction == "NORMAL_SENSOR_BEHAVIOR":

            st.info(
                "Sensor behavior is currently classified "
                "as normal."
            )

        st.caption(
            "Sensor screening is decision support and does "
            "not replace laboratory analysis."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# SILAGE QUALITY PAGE — REAL PADOVA AI
# ============================================================

elif st.session_state.page == "quality":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("🌾 Silage Quality Screening")

    st.write(
        "Enter the fresh-feed measurements and contextual "
        "information used by the Padova silage-quality model."
    )

    st.markdown("### 🧪 Fresh-feed measurements")

    c1, c2, c3 = st.columns(3)

    with c1:

        dm_f = st.number_input(
            "Dry Matter — DM (%)",
            value=35.0
        )

        ash_f = st.number_input(
            "Ash (%)",
            value=5.0
        )

        cp_f = st.number_input(
            "Crude Protein — CP (%)",
            value=8.0
        )

    with c2:

        ee_f = st.number_input(
            "Ether Extract — EE (%)",
            value=3.0
        )

        ndf_f = st.number_input(
            "Neutral Detergent Fibre — NDF (%)",
            value=45.0
        )

        adf_f = st.number_input(
            "Acid Detergent Fibre — ADF (%)",
            value=25.0
        )

    with c3:

        lignin_f = st.number_input(
            "Lignin (%)",
            value=4.0
        )

        wsc_f = st.number_input(
            "Water-Soluble Carbohydrates — WSC (%)",
            value=8.0
        )

        starch_f = st.number_input(
            "Starch (%)",
            value=25.0
        )

    st.markdown("### 📋 Experimental / contextual information")

    q1, q2, q3 = st.columns(3)

    with q1:

        method = st.text_input(
            "Method",
            value=""
        )

        inhoculant = st.text_input(
            "Inoculant",
            value=""
        )

        dose = st.text_input(
            "Dose",
            value=""
        )

    with q2:

        delay = st.text_input(
            "Delay",
            value=""
        )

        fao_class = st.text_input(
            "FAO Class",
            value=""
        )

        maturity = st.text_input(
            "Maturity",
            value=""
        )

    with q3:

        field = st.text_input(
            "Field",
            value=""
        )

    st.divider()

    if st.button(
        "🧪 ANALYZE SILAGE QUALITY",
        type="primary",
        use_container_width=True
    ):

        padova_input = {

            "dm.f": dm_f,
            "ash.f": ash_f,
            "cp.f": cp_f,
            "ee.f": ee_f,
            "ndf.f": ndf_f,
            "adf.f": adf_f,
            "lignin.f": lignin_f,
            "wsc.f": wsc_f,
            "starch.f": starch_f,

            "method": method,
            "inhoculant": inhoculant,
            "dose": dose,
            "delay": delay,
            "fao.class": fao_class,
            "maturity": maturity,
            "field": field
        }

        with st.spinner(
            "Running Padova Silage Risk Model..."
        ):

            prediction, confidence = predict_padova(
                padova_input
            )

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        if prediction == "HIGH_RISK":

            st.error(
                "Silage Risk: HIGH"
            )

            st.warning(
                "The model classified the sample in the "
                "high-risk FQI category. Further assessment "
                "is recommended."
            )

        elif prediction == "MEDIUM_RISK":

            st.warning(
                "Silage Risk: MEDIUM"
            )

            st.info(
                "The model classified the sample in the "
                "medium-risk FQI category."
            )

        else:

            st.success(
                "Silage Risk: LOW"
            )

            st.info(
                "The model classified the sample in the "
                "low-risk FQI category."
            )

        if confidence is not None:

            st.metric(
                "AI Confidence",
                f"{confidence * 100:.1f}%"
            )

        st.caption(
            "Risk categories are based on the FQI-derived "
            "labels used during Padova model training."
        )

        st.caption(
            "This is screening/decision support and does "
            "not replace laboratory analysis."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FUSION PAGE — UNIFIED AI DECISION
# ============================================================

elif st.session_state.page == "fusion":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("🤖 GauChara AI Fusion")

    st.write(
        "Combine visual, sensor and silage-quality evidence "
        "for a unified screening assessment."
    )

    st.info(
        "You can provide the available evidence streams. "
        "The system combines them at the decision layer."
    )

    st.divider()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    st.markdown("### 📸 Visual Evidence")

    fusion_image = st.file_uploader(
        "Upload feed-bunk image",
        type=["jpg", "jpeg", "png"],
        key="fusion_image"
    )

    fbsi_result = None

    if fusion_image:

        image = Image.open(
            fusion_image
        ).convert("RGB")

        st.image(
            image,
            caption="Fusion image",
            width=450
        )

        with st.spinner(
            "Running visual AI..."
        ):

            fbsi_prediction, fbsi_confidence = predict_fbsi(
                image
            )

        fbsi_result = {
            "prediction": fbsi_prediction,
            "confidence": fbsi_confidence
        }

        st.success(
            f"Visual result: **{fbsi_prediction}** "
            f"({fbsi_confidence * 100:.1f}%)"
        )

    # --------------------------------------------------------
    # SENSOR INPUTS
    # --------------------------------------------------------

    st.divider()

    st.markdown("### 🌡️ Sensor Evidence")

    s1, s2 = st.columns(2)

    with s1:

        fusion_internal_temp = st.number_input(
            "Internal Temperature (°C)",
            value=25.0,
            key="fusion_internal_temp"
        )

        fusion_external_temp = st.number_input(
            "External Temperature (°C)",
            value=28.0,
            key="fusion_external_temp"
        )

        fusion_humidity = st.number_input(
            "Internal Humidity (%)",
            value=60.0,
            key="fusion_humidity"
        )

        fusion_level = st.number_input(
            "Silo Level (%)",
            value=50.0,
            key="fusion_level"
        )

        fusion_kg = st.number_input(
            "Quantity (kg)",
            value=100.0,
            key="fusion_kg"
        )

    with s2:

        fusion_delta = st.number_input(
            "Temperature Difference (°C)",
            value=0.0,
            key="fusion_delta"
        )

        fusion_rate = st.number_input(
            "Temperature Rate (°C/hr)",
            value=0.0,
            key="fusion_rate"
        )

        fusion_ph = st.number_input(
            "pH",
            value=4.0,
            key="fusion_ph"
        )

        fusion_moisture = st.number_input(
            "Moisture (%)",
            value=65.0,
            key="fusion_moisture"
        )

        fusion_co2 = st.number_input(
            "CO₂ (ppm)",
            value=500.0,
            key="fusion_co2"
        )

    sensor_input = {

        "internaltemperature":
            fusion_internal_temp,

        "externaltemperature":
            fusion_external_temp,

        "internalhumidity":
            fusion_humidity,

        "percentagelevel":
            fusion_level,

        "kglevel":
            fusion_kg,

        "delta_temp":
            fusion_delta,

        "temp_rate_C_per_hr":
            fusion_rate,

        "simulated_pH":
            fusion_ph,

        "simulated_moisture_pct":
            fusion_moisture,

        "simulated_CO2_ppm":
            fusion_co2
    }

    # --------------------------------------------------------
    # PADOVA
    # --------------------------------------------------------

    st.divider()

    st.markdown("### 🌾 Silage Quality Evidence")

    use_padova = st.checkbox(
        "Include Padova silage-quality analysis",
        value=False
    )

    padova_result = None

    if use_padova:

        p1, p2, p3 = st.columns(3)

        with p1:

            p_dm = st.number_input(
                "DM (%)",
                value=35.0,
                key="fusion_dm"
            )

            p_ash = st.number_input(
                "Ash (%)",
                value=5.0,
                key="fusion_ash"
            )

            p_cp = st.number_input(
                "CP (%)",
                value=8.0,
                key="fusion_cp"
            )

        with p2:

            p_ee = st.number_input(
                "EE (%)",
                value=3.0,
                key="fusion_ee"
            )

            p_ndf = st.number_input(
                "NDF (%)",
                value=45.0,
                key="fusion_ndf"
            )

            p_adf = st.number_input(
                "ADF (%)",
                value=25.0,
                key="fusion_adf"
            )

        with p3:

            p_lignin = st.number_input(
                "Lignin (%)",
                value=4.0,
                key="fusion_lignin"
            )

            p_wsc = st.number_input(
                "WSC (%)",
                value=8.0,
                key="fusion_wsc"
            )

            p_starch = st.number_input(
                "Starch (%)",
                value=25.0,
                key="fusion_starch"
            )

        st.caption(
            "Optional contextual fields"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            p_method = st.text_input(
                "Method",
                key="fusion_method"
            )

            p_inoculant = st.text_input(
                "Inoculant",
                key="fusion_inoculant"
            )

            p_dose = st.text_input(
                "Dose",
                key="fusion_dose"
            )

        with r2:

            p_delay = st.text_input(
                "Delay",
                key="fusion_delay"
            )

            p_fao = st.text_input(
                "FAO Class",
                key="fusion_fao"
            )

            p_maturity = st.text_input(
                "Maturity",
                key="fusion_maturity"
            )

        with r3:

            p_field = st.text_input(
                "Field",
                key="fusion_field"
            )

    # --------------------------------------------------------
    # RUN FUSION
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🚀 RUN UNIFIED AI ANALYSIS",
        type="primary",
        use_container_width=True
    ):

        if fbsi_result is None:

            st.warning(
                "Please upload a feed-bunk image first."
            )

        else:

            with st.spinner(
                "Running sensor AI and evidence fusion..."
            ):

                sensor_prediction, sensor_confidence = (
                    predict_sensor(sensor_input)
                )

                sensor_result = {

                    "prediction":
                        sensor_prediction,

                    "confidence":
                        sensor_confidence
                }

                # Padova is optional
                if use_padova:

                    padova_input = {

                        "dm.f": p_dm,
                        "ash.f": p_ash,
                        "cp.f": p_cp,
                        "ee.f": p_ee,
                        "ndf.f": p_ndf,
                        "adf.f": p_adf,
                        "lignin.f": p_lignin,
                        "wsc.f": p_wsc,
                        "starch.f": p_starch,

                        "method": p_method,
                        "inhoculant": p_inoculant,
                        "dose": p_dose,
                        "delay": p_delay,
                        "fao.class": p_fao,
                        "maturity": p_maturity,
                        "field": p_field
                    }

                    padova_prediction, padova_confidence = (
                        predict_padova(
                            padova_input
                        )
                    )

                    padova_result = {

                        "prediction":
                            padova_prediction,

                        "confidence":
                            padova_confidence
                    }

                final_result = fuse_results(
                    fbsi_result,
                    sensor_result,
                    padova_result
                )

            # ------------------------------------------------
            # FINAL RESULT
            # ------------------------------------------------

            st.divider()

            st.markdown(
                "## 🧠 Unified Screening Result"
            )

            risk = final_result["final_risk"]

            if risk == "HIGH":

                st.error(
                    "🔴 OVERALL SCREENING RISK: HIGH"
                )

            elif risk == "MEDIUM":

                st.warning(
                    "🟡 OVERALL SCREENING RISK: MEDIUM"
                )

            else:

                st.success(
                    "🟢 OVERALL SCREENING RISK: LOW"
                )

            # ------------------------------------------------
            # EVIDENCE TABLE
            # ------------------------------------------------

            st.markdown("### Evidence Summary")

            e1, e2, e3 = st.columns(3)

            with e1:

                st.metric(
                    "📸 Visual",
                    final_result["fbsi"]
                )

            with e2:

                st.metric(
                    "🌡️ Sensor",
                    final_result["sensor"]
                )

            with e3:

                st.metric(
                    "🌾 Padova",
                    final_result["padova"]
                    if final_result["padova"]
                    else "Not included"
                )

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            if final_result["overall_confidence"] is not None:

                st.metric(
                    "Overall Model Confidence",
                    f"{final_result['overall_confidence'] * 100:.1f}%"
                )

            # ------------------------------------------------
            # VOTE BREAKDOWN
            # ------------------------------------------------

            st.markdown("### Evidence Voting")

            v1, v2, v3 = st.columns(3)

            with v1:
                st.metric(
                    "High Evidence",
                    final_result["high_votes"]
                )

            with v2:
                st.metric(
                    "Medium Evidence",
                    final_result["medium_votes"]
                )

            with v3:
                st.metric(
                    "Low Evidence",
                    final_result["low_votes"]
                )

            # ------------------------------------------------
            # ADVISORY
            # ------------------------------------------------

            st.markdown("### 💡 Farmer Advisory")

            st.info(
                final_result["advisory"]
            )

            st.caption(
                "GauChara provides rapid screening and "
                "decision support. It does not replace "
                "laboratory testing or professional assessment."
            )


# ============================================================
# OTHER PAGES
# ============================================================

elif st.session_state.page == "feedback":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("💬 Feedback")

    feedback = st.text_area(
        "Farm observation / feedback"
    )

    if st.button("Submit Feedback"):

        st.success(
            "Thank you for your feedback!"
        )


elif st.session_state.page == "cattle":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("🐄 Cattle Data")

    st.info(
        "Cattle and feeding records can be added here."
    )


elif st.session_state.page == "history":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("📊 Scan History")

    st.info(
        "Previous screening records will appear here."
    )


elif st.session_state.page == "settings":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("🌐 Language & Settings")

    st.selectbox(
        "Language",
        [
            "English",
            "Kannada",
            "Hindi",
            "Tamil",
            "Telugu",
            "Marathi"
        ]
    )

    st.checkbox(
        "Enable notifications"
    )


elif st.session_state.page == "story":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("🌱 The GauChara Story")

    st.write(
        "GauChara is an AI-assisted feed and silage "
        "screening prototype designed to support dairy "
        "farmers with rapid quality and storage assessment."
    )

    st.markdown("""
    **Core technologies**

    - Computer Vision
    - Machine Learning
    - Sensor Analytics
    - Silage Quality Analysis
    - Evidence Fusion
    - Farmer Advisory
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<b>GauChara</b><br>
Smarter feed, healthier herds.

</div>
""", unsafe_allow_html=True)
