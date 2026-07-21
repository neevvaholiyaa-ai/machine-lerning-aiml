import joblib
import os
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

@st.cache_resource
def load_model(key, variant="best"):
    return joblib.load(os.path.join(MODELS_DIR, f"{key}_{variant}.pkl"))

@st.cache_resource
def load_scaler(key):
    try:
        return joblib.load(os.path.join(MODELS_DIR, f"scaler{key[-1]}.pkl"))
    except FileNotFoundError:
        return None

@st.cache_resource
def load_artifacts(key):
    suffix = key[-1]
    artifacts = {"scaler": load_scaler(key)}
    for variant in ["best", "lr", "dt", "rf", "svr"]:
        try:
            artifacts[variant] = load_model(key, variant)
        except FileNotFoundError:
            artifacts[variant] = None
    return artifacts

@st.cache_resource
def load_encoders():
    import joblib
    enc = {}
    try:
        enc["le6"] = joblib.load(os.path.join(MODELS_DIR, "le6.pkl"))
        enc["le8"] = joblib.load(os.path.join(MODELS_DIR, "le8.pkl"))
        enc["freq_map5_region"] = joblib.load(os.path.join(MODELS_DIR, "freq_map5_region.pkl"))
        enc["freq_map5_country"] = joblib.load(os.path.join(MODELS_DIR, "freq_map5_country.pkl"))
        enc["freq_map6"] = joblib.load(os.path.join(MODELS_DIR, "freq_map6.pkl"))
        enc["freq_map7"] = joblib.load(os.path.join(MODELS_DIR, "freq_map7.pkl"))
    except FileNotFoundError:
        pass
    return enc

MODEL_NAMES = {
    "best": "Best (Tuned)",
    "lr": "Linear Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "svr": "SVR",
}

DATASET_INFO = {
    "df1": {
        "label": "FTAs in India (in million)",
        "target": "FTAs in India (in million)",
        "target_cols": ["FTAs in India (in million)"],
        "features": ["Year"],
        "single_output": True,
        "unit": "million",
        "has_scaler": True,
        "r2": 0.922,
        "best_variant": "lr",
        "note": "⚠️ RandomForest can't extrapolate beyond 2020 (training data max). Using Linear Regression for future forecasts.",
    },
    "df2": {
        "label": "Age Group % Distribution (15-24 & 25-34)",
        "target": "% Age 15-24 & 25-34",
        "target_cols": ["% Age 15-24", "% Age 25-34"],
        "features": ["Year", "FTAs", "% Age 0-14", "% Age 35-44", "% Age 45-54", "% Age 55-64"],
        "single_output": False,
        "unit": "percent",
        "has_scaler": True,
        "r2": 0.655,
    },
    "df3": {
        "label": "Quarterly % Distribution (Q2 & Q4)",
        "target": "% Q2 & Q4",
        "target_cols": ["% Q2 (Apr-Jun)", "% Q4 (Oct-Dec)"],
        "features": ["Year", "Arrivals", "% Q1 (Jan-Mar)"],
        "single_output": False,
        "unit": "percent",
        "has_scaler": True,
        "r2": -0.345,
    },
    "df4": {
        "label": "World & India Tourism (million)",
        "target": "World & India arrivals",
        "target_cols": ["World arrivals (million)", "India arrivals (million)"],
        "features": ["Year", "India's Rank"],
        "single_output": False,
        "unit": "million",
        "has_scaler": True,
        "r2": -0.38,
    },
    "df5": {
        "label": "Purpose of Visit %",
        "target": "Business, Leisure, Medical, Diaspora %",
        "target_cols": ["Business %", "Leisure %", "Medical %", "Diaspora %"],
        "features": ["Arrivals", "Region freq", "Country freq"],
        "single_output": False,
        "unit": "percent",
        "has_scaler": False,
        "r2": 0.102,
    },
    "df6": {
        "label": "Monument Visitation (Domestic & Foreign 2020-21)",
        "target": "Domestic & Foreign 2020-21",
        "target_cols": ["Domestic 2020-21", "Foreign 2020-21"],
        "features": ["Domestic 2019-20", "Foreign 2019-20", "Circle freq", "Monument label"],
        "single_output": False,
        "unit": "visitors",
        "has_scaler": False,
        "r2": 0.284,
    },
    "df7": {
        "label": "Region Arrivals 2019",
        "target": "Number of Arrivals - 2019",
        "target_cols": ["Number of Arrivals - 2019"],
        "features": ["Arrivals 2017", "Arrivals 2018", "Region", "Country freq"],
        "single_output": True,
        "unit": "arrivals",
        "has_scaler": False,
        "r2": 0.994,
        "best_variant": "lr",
        "note": "🌟 Best model in the notebook — R²=0.994. Enter realistic 2017/2018 arrival values."
    },
    "df8": {
        "label": "State-wise Total Tourists 2020",
        "target": "Total_2020",
        "target_cols": ["Total_2020"],
        "features": ["State (encoded)", "Total_2019"],
        "single_output": True,
        "unit": "visitors",
        "has_scaler": True,
        "r2": 0.903,
        "best_variant": "rf",
    },
}

REGION_OPTIONS = [
    "Australasia", "Central and South America", "East Asia", "Eastern Europe",
    "North America", "Not Classified elsewhere", "South Asia", "South East Asia",
    "West Asia", "Western Europe",
]
