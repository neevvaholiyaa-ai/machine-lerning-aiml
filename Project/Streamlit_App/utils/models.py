import joblib
import os
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

@st.cache_resource
def load_model(key, variant="best"):
    path = os.path.join(MODELS_DIR, f"{key}_{variant}.pkl")
    return joblib.load(path)

@st.cache_resource
def load_scaler(key):
    path = os.path.join(MODELS_DIR, f"scaler{key[-1]}.pkl")
    return joblib.load(path)

@st.cache_resource
def load_artifacts(key):
    suffix = key[-1]
    artifacts = {}
    try:
        artifacts["scaler"] = load_scaler(key)
    except FileNotFoundError:
        artifacts["scaler"] = None
    for variant in ["best", "lr", "dt", "rf", "svr"]:
        try:
            artifacts[variant] = load_model(key, variant)
        except FileNotFoundError:
            artifacts[variant] = None
    return artifacts

MODEL_NAMES = {
    "best": "Best (Tuned)",
    "lr": "Linear Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "svr": "SVR",
}

DATASET_INFO = {
    "df1": {
        "label": "FTAs Prediction",
        "target": "FTAs in India (in million)",
        "target_cols": ["FTAs in India (in million)"],
        "features": ["Year"],
        "single_output": True,
    },
    "df2": {
        "label": "Age Group Distribution Prediction",
        "target": "% distribution by Age-Group 15-24 & 25-34",
        "target_cols": ["% distribution by Age-Group (in Year) - 15-24", "% distribution by Age-Group (in Year) - 25-34"],
        "features": ["Year", "FTAs", "% distribution by Age-Group (in Year) - 0-14",
                     "% distribution by Age-Group (in Year) - 35-44", "% distribution by Age-Group (in Year) - 45-54",
                     "% distribution by Age-Group (in Year) - 55-64"],
        "single_output": False,
    },
    "df3": {
        "label": "Quarterly Distribution Prediction",
        "target": "% Quarterly Q2 & Q4",
        "target_cols": ["% Distribution by Quarter - 2nd Quarter(Apr-June)", "% Distribution by Quarter - 4th Quarter (Oct-Dec)"],
        "features": ["Year", "Arrivals", "% Distribution by Quarter - 1st Quarter (Jan-Mar)"],
        "single_output": False,
    },
    "df4": {
        "label": "World vs India Tourism Prediction",
        "target": "World & India Numbers (in million)",
        "target_cols": ["World - Number (in million)", "India - Number (in million)"],
        "features": ["Year", "Rank of India"],
        "single_output": False,
    },
    "df5": {
        "label": "Purpose of Visit Prediction",
        "target": "Purpose % (Business, Leisure, Medical, Diaspora)",
        "target_cols": ["Business and Professional(%)", "Leisure Holiday and Recreation(%)", "Medical(%)", "Indian Diaspora(%)"],
        "features": ["Arrivals (in numbers)", "Region_cat", "Country of Nationality_cat"],
        "single_output": False,
    },
    "df6": {
        "label": "Monument Visitation Prediction",
        "target": "Domestic & Foreign 2020-21",
        "target_cols": ["Domestic-2020-21", "Foreign-2020-21"],
        "features": ["Domestic-2019-20", "Foreign-2019-20", "Circle_new", "Name of the Monument_new"],
        "single_output": False,
    },
    "df7": {
        "label": "Region Arrivals Prediction (2019)",
        "target": "Number of Arrivals - 2019",
        "target_cols": ["Number of Arrivals - 2019"],
        "features": ["Number of Arrivals - 2017", "Number of Arrivals - 2018",
                     "Region flags (12)", "Country of Nationality_new"],
        "single_output": True,
    },
    "df8": {
        "label": "State-wise Tourist Prediction (2020)",
        "target": "Total_2020",
        "target_cols": ["Total_2020"],
        "features": ["States/UTs_new", "Total_2019"],
        "single_output": True,
    },
}
