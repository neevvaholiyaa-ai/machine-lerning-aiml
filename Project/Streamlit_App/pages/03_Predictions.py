import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data_loader import load_raw, DATASET_LABELS
from utils.models import load_artifacts, MODEL_NAMES, DATASET_INFO
from utils.preprocessing import preprocess_input

st.title("ML Predictions")
st.markdown("Use trained models to predict tourism metrics")

dataset_key = st.sidebar.selectbox(
    "Select Prediction Model",
    options=list(DATASET_INFO.keys()),
    format_func=lambda k: f"{k.upper()} - {DATASET_INFO[k]['label']}",
)

info = DATASET_INFO[dataset_key]
artifacts = load_artifacts(dataset_key)

model_variant = st.sidebar.selectbox(
    "Select Model Variant",
    options=[v for v in ["best", "lr", "dt", "rf", "svr"] if artifacts.get(v) is not None],
    format_func=lambda v: MODEL_NAMES[v],
)

model = artifacts[model_variant]
scaler = artifacts.get("scaler")

st.subheader(f"{DATASET_LABELS.get(dataset_key, dataset_key)} — {MODEL_NAMES[model_variant]}")
st.caption(f"Target: {info['target']}")

df = load_raw(dataset_key)

st.markdown("### Enter Input Features")

input_data = {}
feature_cols = info["features"]

if dataset_key == "df1":
    year = st.number_input("Year", min_value=1981, max_value=2030, value=2023, step=1)
    input_data = pd.DataFrame([[year]], columns=["Year"])

elif dataset_key == "df2":
    year = st.number_input("Year", min_value=2001, max_value=2030, value=2023, step=1)
    ftas = st.number_input("FTAs", value=10000000)
    age_0_14 = st.slider("% Age 0-14", 0.0, 20.0, 5.0)
    age_35_44 = st.slider("% Age 35-44", 0.0, 30.0, 15.0)
    age_45_54 = st.slider("% Age 45-54", 0.0, 25.0, 12.0)
    age_55_64 = st.slider("% Age 55-64", 0.0, 20.0, 8.0)
    input_data = pd.DataFrame([[year, ftas, age_0_14, age_35_44, age_45_54, age_55_64]],
                               columns=feature_cols)

elif dataset_key == "df3":
    year = st.number_input("Year", min_value=2001, max_value=2030, value=2023, step=1)
    arrivals = st.number_input("Arrivals", value=10000000)
    q1 = st.slider("% Q1 (Jan-Mar)", 0.0, 50.0, 20.0)
    input_data = pd.DataFrame([[year, arrivals, q1]], columns=feature_cols)

elif dataset_key == "df4":
    year = st.number_input("Year", min_value=2001, max_value=2030, value=2025, step=1)
    rank = st.number_input("India's Rank", min_value=1, max_value=50, value=21, step=1)
    input_data = pd.DataFrame([[year, rank]], columns=feature_cols)

elif dataset_key == "df5":
    arrivals = st.number_input("Arrivals (in numbers)", value=100000)
    region_cat = st.number_input("Region Category (frequency)", value=10)
    country_cat = st.number_input("Country Category (frequency)", value=5)
    input_data = pd.DataFrame([[arrivals, region_cat, country_cat]], columns=feature_cols)

elif dataset_key == "df6":
    domestic_1920 = st.number_input("Domestic 2019-20", value=100000)
    foreign_1920 = st.number_input("Foreign 2019-20", value=10000)
    circle_new = st.number_input("Circle (frequency encoded)", value=5)
    monument_new = st.number_input("Monument (label encoded)", value=50)
    input_data = pd.DataFrame([[domestic_1920, foreign_1920, circle_new, monument_new]],
                               columns=feature_cols)

elif dataset_key == "df7":
    arr_2017 = st.number_input("Arrivals 2017", value=50000)
    arr_2018 = st.number_input("Arrivals 2018", value=55000)
    region_cols = ["Region_Australasia", "Region_Central and South America", "Region_East Asia",
                   "Region_Eastern Europe", "Region_North America", "Region_Not Classified elsewhere",
                   "Region_South Asia", "Region_South East Asia", "Region_West Asia",
                   "Region_Western Europe"]
    region_vals = {}
    for rc in region_cols:
        region_vals[rc] = st.checkbox(rc.replace("Region_", ""), value=(rc == "Region_South Asia"))
    country_cat = st.number_input("Country of Nationality (frequency)", value=10)
    row = [arr_2017, arr_2018] + [1 if region_vals[rc] else 0 for rc in region_cols] + [country_cat]
    input_data = pd.DataFrame([row], columns=feature_cols)

elif dataset_key == "df8":
    state_new = st.number_input("State (label encoded)", value=10)
    total_2019 = st.number_input("Total 2019", value=5000000)
    input_data = pd.DataFrame([[state_new, total_2019]], columns=feature_cols)

if st.button("Predict", type="primary"):
    try:
        X = preprocess_input(dataset_key, input_data)
        if scaler:
            X = scaler.transform(X)
        pred = model.predict(X)

        if info["single_output"]:
            pred_val = float(pred[0])
            st.success(f"### Predicted {info['target']}: **{pred_val:,.0f}**")
        else:
            st.success("### Predicted Values:")
            cols = st.columns(len(info["target_cols"]))
            for i, (c, pv) in enumerate(zip(info["target_cols"], pred[0])):
                with cols[i]:
                    st.metric(c, f"{pv:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
