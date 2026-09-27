# GauChara

## Smarter feed, healthier herds.

GauChara is an AI-enabled rapid feed and silage quality screening system designed to support dairy farmers with fast and accessible decision support.

## AI Components

### FBSI Visual Analysis
Uses a MobileNetV3-Small image classification model to analyse feed-bunk visual appearance and predict Feed Bunk Score categories.

### Sensor Risk Analysis
Uses an ExtraTrees-based machine-learning model to analyse silo monitoring variables including temperature, humidity, pH, moisture and CO2.

### Silage Quality Analysis
Uses a Random Forest model trained on the Padova maize silage dataset to classify silage into FQI-derived risk categories.

### Unified AI Fusion
Combines available visual, sensor and silage-quality evidence into a single screening-level LOW / MEDIUM / HIGH risk decision.

## Important

GauChara is a rapid screening and decision-support prototype. It is not intended to replace laboratory feed analysis or professional veterinary/agricultural assessment.

## Run locally

Install the required packages using:

pip install -r requirements.txt

Then run:

streamlit run app.py

## Technology

- Python
- Streamlit
- PyTorch
- Torchvision
- Scikit-learn
- Pandas
- Joblib
