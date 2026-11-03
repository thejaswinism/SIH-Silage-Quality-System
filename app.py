
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

# ============================================================
# LOCAL DATABASE (NO SUPABASE REQUIRED)
# ============================================================
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(PROJECT, "gau_chara_data.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS cattle (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT UNIQUE NOT NULL,
            name TEXT,
            breed TEXT,
            age_years REAL,
            notes TEXT,
            created_at TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer TEXT,
            category TEXT,
            message TEXT NOT NULL,
            created_at TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_type TEXT,
            visual_result TEXT,
            sensor_result TEXT,
            padova_result TEXT,
            overall_risk TEXT,
            advisory TEXT,
            created_at TEXT
        )""")

def save_scan(scan_type, visual=None, sensor=None, padova=None,
              risk=None, advisory=None):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO scans
            (scan_type, visual_result, sensor_result, padova_result,
             overall_risk, advisory, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (scan_type, visual, sensor, padova, risk, advisory,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

init_db()


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

    model = joblib.load(
        SENSOR_MODEL_PATH
    )

    # Compatibility fix for models saved with an older
    # scikit-learn version and loaded with a newer version.
    try:
        from sklearn.impute import SimpleImputer

        visited = set()

        def repair_imputers(obj):

            if id(obj) in visited:
                return

            visited.add(id(obj))

            if isinstance(obj, SimpleImputer):

                if (
                    not hasattr(obj, "_fill_dtype")
                    and hasattr(obj, "statistics_")
                ):
                    obj._fill_dtype = obj.statistics_.dtype

            if hasattr(obj, "get_params"):

                try:
                    params = obj.get_params(deep=False)

                    for child in params.values():

                        if child is not None:
                            repair_imputers(child)

                except Exception:
                    pass

            if isinstance(obj, (list, tuple)):

                for child in obj:
                    repair_imputers(child)

            elif isinstance(obj, dict):

                for child in obj.values():
                    repair_imputers(child)

        repair_imputers(model)

    except Exception:
        pass

    return model


sensor_model = load_sensor_model()


def predict_sensor(sensor_input):
    import pandas as pd

    X = pd.DataFrame(
        [[sensor_input[f] for f in SENSOR_FEATURES]],
        columns=SENSOR_FEATURES
    )
    prediction = sensor_model.predict(X)[0]
    confidence = None

    if hasattr(sensor_model, "predict_proba"):
        probabilities = sensor_model.predict_proba(X)[0]
        confidence = float(max(probabilities))

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

padova_model = None

def predict_padova(padova_input):
    global padova_model
    import pandas as pd

    if padova_model is None:
        padova_model = load_padova_model()

    X = pd.DataFrame(
        [[padova_input[f] for f in PADOVA_FEATURES]],
        columns=PADOVA_FEATURES
    )
    prediction = padova_model.predict(X)[0]
    confidence = None

    if hasattr(padova_model, "predict_proba"):
        probabilities = padova_model.predict_proba(X)[0]
        confidence = float(max(probabilities))

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

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

/* GauChara palette and global typography */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
}
.stApp {
    background: #F3F0DD !important;
    color: #173B1B !important;
}
.block-container {
    padding-top: 2rem;
    padding-left: 5%;
    padding-right: 5%;
    padding-bottom: 3rem;
}
#MainMenu, header, footer { visibility: hidden; }

/* Headings and standard text */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp p, .stApp li, .stApp label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stCaption {
    color: #173B1B !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #174D25 !important;
    font-weight: 700 !important;
}

/* Brand, hero and dashboard cards */
.brand { font-size: 42px; font-weight: 700; color: #174D25 !important; margin-bottom: 35px; }
.hero { text-align: center; margin-bottom: 45px; }
.hero h1 { font-size: 44px; color: #174D25 !important; }
.hero p { font-size: 18px; color: #526553 !important; }
.card {
    background: #E5E2CE !important; border-radius: 22px; padding: 30px 24px;
    min-height: 350px; margin-bottom: 25px; text-align: center;
}
.card-title { font-size: 27px; font-weight: 700; color: #174D25 !important; margin-bottom: 22px; }
.card-image {
    height: 210px; border-radius: 20px; border: 5px solid #175A25;
    display: flex; align-items: center; justify-content: center;
    font-size: 75px; background: #D9D7C3 !important;
}
.card-description { margin-top: 14px; color: #526553 !important; font-size: 15px; }
.result-box { background: #E5E2CE !important; border-radius: 20px; padding: 25px; margin-top: 20px; color: #173B1B !important; }
.result-box * { color: #173B1B !important; }
.footer { margin-top: 60px; padding-top: 35px; border-top: 1px solid #D5D1BA; color: #526553 !important; }

/* Buttons */
div.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    border-radius: 10px !important; border: 1px solid #175A25 !important;
    background: #175A25 !important; color: #FFFFFF !important; font-weight: 600 !important;
}
div.stButton > button *, .stFormSubmitButton > button *, .stDownloadButton > button * { color: #FFFFFF !important; }
div.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    background: #26783A !important; border-color: #26783A !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, [data-baseweb="input"] input {
    background: #FFFDF4 !important; color: #173B1B !important;
    -webkit-text-fill-color: #173B1B !important;
    border: 1px solid #9AA98D !important; border-radius: 8px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #718071 !important; -webkit-text-fill-color: #718071 !important; }
[data-baseweb="select"] > div { background: #FFFDF4 !important; border-color: #9AA98D !important; }
[data-baseweb="select"] *, [role="option"] { color: #173B1B !important; }
[data-baseweb="popover"] [role="listbox"] { background: #FFFDF4 !important; }

/* Metrics: dark readable values on light cards */
[data-testid="stMetric"] { background: #E5E2CE !important; padding: 14px !important; border-radius: 12px !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #526553 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #174D25 !important; }

/* Alerts and status messages */
[data-testid="stAlert"] { border-radius: 12px !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] span,
[data-testid="stAlert"] div { color: #173B1B !important; }
[data-testid="stAlert"] { background: #DCEAF0 !important; }
[data-testid="stAlert"][kind="success"] { background: #D9EBD4 !important; }
[data-testid="stAlert"][kind="warning"] { background: #FFF0BD !important; }
[data-testid="stAlert"][kind="error"] { background: #F8D9D5 !important; }

/* Tables and dataframes */
[data-testid="stDataFrame"], [data-testid="stTable"] { background: #FFFDF4 !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] *, [data-testid="stTable"] * { color: #173B1B !important; }

/* Checkbox and radio labels */
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p,
[data-testid="stRadio"] label, [data-testid="stRadio"] p { color: #173B1B !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #173B1B !important; }
.stTabs [aria-selected="true"] { color: #175A25 !important; }

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
# IMAGE PAGE — GAUCHARA AI SCREENING
# ============================================================

elif st.session_state.page == "image":

    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()

    st.title("📸 Check Your Feed")

    st.write(
        "Upload a clear picture of your feed or feed bunk. "
        "GauChara combines visual AI analysis with available "
        "storage sensor evidence."
    )

    uploaded = st.file_uploader(
        "📷 Upload / Take Photo",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Selected feed sample",
            width=500
        )

        if st.button(
            "🔍 ANALYZE WITH GAUCHARA",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "GauChara AI is analyzing your feed..."
            ):

                # --------------------------------------------
                # 1. FBSI IMAGE ANALYSIS
                # --------------------------------------------

                prediction, confidence = predict_fbsi(image)

                # --------------------------------------------
                # 2. SENSOR EVIDENCE
                # --------------------------------------------
                # Prototype sensor stream.
                # Farmer does NOT enter these values.

                sensor_input = {
                    "internaltemperature": 25.0,
                    "externaltemperature": 28.0,
                    "internalhumidity": 60.0,
                    "percentagelevel": 50.0,
                    "kglevel": 100.0,
                    "delta_temp": 0.0,
                    "temp_rate_C_per_hr": 0.0,
                    "simulated_pH": 4.0,
                    "simulated_moisture_pct": 65.0,
                    "simulated_CO2_ppm": 500.0
                }

                sensor_prediction, sensor_confidence = predict_sensor(
                    sensor_input
                )

            # --------------------------------------------
            # 3. COMBINED GAUCHARA RESULT
            # --------------------------------------------

            visual_high = prediction in ["Score 3", "Score 4"]

            sensor_high = sensor_prediction in [
                "HIGH_DETERIORATION_INDICATORS"
            ]

            sensor_early = sensor_prediction in [
                "EARLY_DETERIORATION_INDICATORS"
            ]

            visual_low = prediction in [
                "Score 0",
                "Score 1",
                "Score 1-2"
            ]

            sensor_low = sensor_prediction in [
                "LOW_APPARENT_RISK",
                "NORMAL_SENSOR_BEHAVIOR"
            ]

            high_votes = int(visual_high) + int(sensor_high)
            low_votes = int(visual_low) + int(sensor_low)

            if high_votes >= 2:
                overall_risk = "HIGH"
                risk_message = (
                    "Potential feed or storage deterioration detected. "
                    "Further inspection is recommended."
                )

            elif low_votes >= 2:
                overall_risk = "LOW"
                risk_message = (
                    "No strong deterioration indicators were detected. "
                    "Continue routine monitoring."
                )

            else:
                overall_risk = "MEDIUM"
                risk_message = (
                    "The available evidence is mixed. "
                    "Inspect the feed and continue monitoring."
                )

            confidence_values = [
                value for value in [confidence, sensor_confidence]
                if value is not None
            ]
            combined_confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values else None
            )

            # --------------------------------------------
            # 4. ONE RESULT CARD
            # --------------------------------------------

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                "## 🌾 GauChara AI Assessment"
            )

            st.markdown(
                f"### 📸 Visual Feed Score: **{prediction}**"
            )

            st.metric(
                "Visual AI Confidence",
                f"{confidence * 100:.1f}%"
            )

            st.markdown(
                f"### 🌡️ Storage Evidence: "
                f"**{sensor_prediction.replace('_', ' ').title()}**"
            )

            if combined_confidence is not None:
                st.metric(
                    "Average confidence of available models (not combined accuracy)",
                    f"{combined_confidence * 100:.1f}%"
                )

            st.markdown("---")

            if overall_risk == "HIGH":

                st.error(
                    "🔴 OVERALL RISK: HIGH"
                )

            elif overall_risk == "MEDIUM":

                st.warning(
                    "🟡 OVERALL RISK: MEDIUM"
                )

            else:

                st.success(
                    "🟢 OVERALL RISK: LOW"
                )

            st.markdown("### 💡 Farmer Advisory")

            st.write(risk_message)

            st.caption(
                "Evidence used: FBSI visual analysis + "
                "available storage sensor evidence."
            )

            st.caption(
                "Rapid screening only — not a laboratory measurement."
            )

            save_scan(
                "Photo screening (sensor inputs simulated)",
                visual=prediction,
                sensor=sensor_prediction,
                risk=overall_risk,
                advisory=risk_message
            )
            st.success("Screening record saved to local scan history.")

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

        save_scan(
            "Manual sensor screening",
            sensor=prediction,
            risk=(
                "HIGH" if prediction == "HIGH_DETERIORATION_INDICATORS"
                else "MEDIUM" if prediction == "EARLY_DETERIORATION_INDICATORS"
                else "LOW" if prediction in ["LOW_APPARENT_RISK", "NORMAL_SENSOR_BEHAVIOR"]
                else "REVIEW"
            ),
            advisory="Sensor model screening result; verify readings and inspect the silo."
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

            save_scan(
                "AI fusion",
                visual=final_result["fbsi"],
                sensor=final_result["sensor"],
                padova=final_result["padova"],
                risk=final_result["final_risk"],
                advisory=final_result["advisory"]
            )
            st.success("Fusion result saved to local scan history.")


# ============================================================
# FEEDBACK, CATTLE, HISTORY, SETTINGS AND STORY
# ============================================================

elif st.session_state.page == "feedback":
    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()
    st.title("💬 Farmer Feedback")
    farmer = st.text_input("Your name (optional)")
    category = st.selectbox("Feedback category", ["App experience", "Feed quality observation", "Sensor issue", "Suggestion", "Other"])
    message = st.text_area("Your feedback", height=140)
    if st.button("Submit Feedback", type="primary"):
        if not message.strip():
            st.warning("Please enter feedback before submitting.")
        else:
            with get_db() as conn:
                conn.execute("INSERT INTO feedback (farmer, category, message, created_at) VALUES (?, ?, ?, ?)",
                    (farmer.strip(), category, message.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            st.success("Thank you! Your feedback has been saved.")
            st.rerun()
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    st.caption(f"Feedback submissions on this app instance: {count}")


elif st.session_state.page == "cattle":
    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()
    st.title("🐄 Cattle Records")
    with st.form("add_cattle_form", clear_on_submit=True):
        tag = st.text_input("Cattle ID / Tag *")
        name = st.text_input("Cattle name")
        breed = st.text_input("Breed")
        age = st.number_input("Age (years)", min_value=0.0, max_value=40.0, value=2.0, step=0.5)
        notes = st.text_area("Feeding / health notes")
        submitted = st.form_submit_button("Register Cattle")
    if submitted:
        if not tag.strip():
            st.error("Cattle ID / Tag is required.")
        else:
            try:
                with get_db() as conn:
                    conn.execute("INSERT INTO cattle (tag, name, breed, age_years, notes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (tag.strip(), name.strip(), breed.strip(), age, notes.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                st.success("Cattle record added.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("That cattle ID already exists.")
    st.subheader("Registered cattle")
    with get_db() as conn:
        cattle_rows = conn.execute("SELECT * FROM cattle ORDER BY id DESC").fetchall()
    if cattle_rows:
        import pandas as pd
        st.dataframe([dict(row) for row in cattle_rows], use_container_width=True, hide_index=True)
        choices = {f'{r["tag"]} — {r["name"] or "Unnamed"}': r["id"] for r in cattle_rows}
        selected = st.selectbox("Select record to delete", list(choices.keys()))
        if st.button("Delete selected cattle record"):
            with get_db() as conn:
                conn.execute("DELETE FROM cattle WHERE id = ?", (choices[selected],))
            st.success("Record deleted.")
            st.rerun()
    else:
        st.info("No cattle records added yet.")


elif st.session_state.page == "history":
    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()
    st.title("📊 Screening History")
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()
    if rows:
        import pandas as pd
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download scan history (CSV)", df.to_csv(index=False).encode("utf-8"),
            "gau_chara_scan_history.csv", "text/csv")
        if st.button("Clear all scan history"):
            with get_db() as conn:
                conn.execute("DELETE FROM scans")
            st.success("History cleared.")
            st.rerun()
    else:
        st.info("No screenings saved yet. Complete an analysis first.")


elif st.session_state.page == "settings":
    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()
    st.title("🌐 Language & Settings")
    language = st.selectbox("Select language", ["English", "Kannada", "Hindi", "Tamil", "Telugu", "Marathi"], key="selected_language")
    if language == "English":
        st.success("English interface selected.")
    else:
        st.warning(f"{language} is selected, but full interface translation is not implemented yet. The application currently remains in English.")
    st.checkbox("Enable in-app alerts", value=True, key="in_app_alerts")
    st.caption("In-app preferences only. Push notifications are not configured.")


elif st.session_state.page == "story":
    if st.button("← Back to GauChara"):
        navigate("home")
        st.rerun()
    st.title("🌱 The GauChara Story")
    st.subheader("Smarter feed, healthier herds.")
    st.write("GauChara is an AI-assisted rapid feed and silage screening prototype designed to support dairy farmers in identifying visual and storage-related warning indicators.")
    st.markdown("""
    **How it works**
    1. FBSI computer vision classifies visible feed-bunk condition.
    2. The sensor model evaluates supplied storage-condition measurements.
    3. The Padova model classifies FQI-derived silage risk when its required inputs are available.
    4. The fusion layer combines available classifications into a screening category and advisory.

    **Important limitations**
    - A photo does not directly measure pH, moisture, nutrients, toxins or adulterants.
    - The photo workflow currently uses simulated sensor inputs, not live IoT readings.
    - Padova risk categories are derived from FQI tertiles in its training dataset.
    - GauChara is a screening aid, not a replacement for laboratory testing.
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
