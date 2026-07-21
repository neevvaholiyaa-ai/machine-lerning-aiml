import streamlit as st
import numpy as np
from utils.data_loader import load_raw, DATASET_LABELS
from utils.models import load_artifacts, MODEL_NAMES, DATASET_INFO, REGION_OPTIONS, load_encoders
from utils.preprocessing import (
    build_features_df1, build_features_df2,
    build_features_df7, build_features_df8,
)

PREDICT_MODELS = ["df1", "df2", "df7", "df8"]

st.title("🤖 ML Predictions")
st.caption("Select a model from the sidebar, enter inputs, and get predictions")

dataset_key = st.sidebar.selectbox(
    "Choose Model",
    options=PREDICT_MODELS,
    format_func=lambda k: f"Model {k.upper()} — {DATASET_INFO[k]['label']}",
)

info = DATASET_INFO[dataset_key]
artifacts = load_artifacts(dataset_key)
encoders = load_encoders()

default_variant = info.get("best_variant", "best")
model_options = [v for v in ["best", "lr", "dt", "rf", "svr"] if artifacts.get(v) is not None]
default_idx = model_options.index(default_variant) if default_variant in model_options else 0

model_variant = st.sidebar.selectbox(
    "Algorithm", options=model_options,
    format_func=lambda v: MODEL_NAMES[v], index=default_idx,
)

model = artifacts[model_variant]
scaler = artifacts["scaler"]

r2 = info["r2"]
r2_color = "green" if r2 > 0.8 else "orange" if r2 > 0.3 else "red"
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Accuracy (R²):** :{r2_color}[**{r2:.3f}**]")
note = info.get("note")
if note:
    st.sidebar.caption(note)

df = load_raw(dataset_key)

with st.container(border=True):
    st.markdown(f"### ✏️ {DATASET_LABELS.get(dataset_key, dataset_key)}")
    st.caption(f"Target: **{info['target']}** | R²: {r2:.3f} | Training data: {df.shape[0]} rows")
    input_df = None

    if dataset_key == "df1":
        c1, c2 = st.columns([1, 2])
        with c1:
            year = st.number_input("Forecast Year", min_value=1981, max_value=2035, value=2025, step=1,
                                    help="Enter a year to predict FTAs")
        with c2:
            st.info(f"📌 Predicts **Foreign Tourist Arrivals (million)** for a given year. Uses LR model (extrapolates beyond training data).", icon="ℹ️")
        if st.button("🔮 Predict FTAs", type="primary", use_container_width=True):
            input_df = build_features_df1(year)

    elif dataset_key == "df2":
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Year", 2001, 2035, 2025)
            ftas = st.number_input("FTAs (millions)", 0.0, 50.0, 10.0, step=0.5)
            age_0_14 = st.slider("% Age Group 0-14", 0.0, 30.0, 5.0)
        with c2:
            age_35_44 = st.slider("% Age Group 35-44", 0.0, 30.0, 15.0)
            age_45_54 = st.slider("% Age Group 45-54", 0.0, 25.0, 12.0)
            age_55_64 = st.slider("% Age Group 55-64", 0.0, 20.0, 8.0)
        st.caption("ℹ️ Predicts % distribution for age groups 15-24 and 25-34 based on other age groups and FTAs.")
        if st.button("🔮 Predict Age Distribution", type="primary", use_container_width=True):
            input_df = build_features_df2(year, ftas, age_0_14, age_35_44, age_45_54, age_55_64)

    elif dataset_key == "df7":
        c1, c2 = st.columns(2)
        with c1:
            year1 = st.number_input("Year 1 (Known Arrivals)", min_value=2000, max_value=2030, value=2017, step=1,
                                     help="First year of known arrival data")
            arr_yr1 = st.number_input(f"Arrivals in {year1}", 1, 5_000_000, 500_000, step=10_000,
                                       help=f"Number of tourist arrivals in {year1}")
            region_label = st.selectbox("Region", REGION_OPTIONS,
                                        help="Geographic region of the country")
        with c2:
            year2 = st.number_input("Year 2 (Known Arrivals)", min_value=2000, max_value=2030, value=2018, step=1,
                                     help="Second year of known arrival data")
            arr_yr2 = st.number_input(f"Arrivals in {year2}", 1, 5_000_000, 550_000, step=10_000,
                                       help=f"Number of tourist arrivals in {year2}")
            country_freq = st.number_input("Country frequency in dataset", 1, 100, 10,
                                            help="How often this country appears in training data")
        st.success("🌟 **R² = 0.994 — Highly accurate!** Enter arrivals from two recent years to predict the next year.", icon="🌟")
        if st.button("🔮 Predict Next Year Arrivals", type="primary", use_container_width=True):
            input_df = build_features_df7(arr_yr1, arr_yr2, region_label, country_freq)

    elif dataset_key == "df8":
        c1, c2 = st.columns(2)
        with c1:
            states_list = sorted([s.strip() for s in df["States/UTs"].dropna().unique().tolist()])
            state_name = st.selectbox("State / UT", states_list,
                                      help="Select an Indian state or union territory")
            le8 = encoders.get("le8")
            if le8:
                state_label = int(le8.transform([state_name])[0])
            else:
                state_label = st.number_input("State (encoded)", 0, 40, 10)
        with c2:
            total_2019 = st.number_input("Total Tourists in 2019", 0, 1_000_000_000, 50_000_000,
                                          step=100_000,
                                          help="Domestic + Foreign combined arrivals in 2019")
        st.caption("ℹ️ Predicts total tourist arrivals for 2020 based on 2019 data and state.")
        if st.button("🔮 Predict 2020 Tourists", type="primary", use_container_width=True):
            input_df = build_features_df8(state_label, total_2019)

if input_df is not None:
    try:
        X = input_df.values
        if scaler is not None:
            X = scaler.transform(X)
        raw_pred = model.predict(X)

        if isinstance(raw_pred, np.ndarray) and raw_pred.ndim > 1:
            pred_vals = raw_pred[0]
        else:
            pred_vals = raw_pred
        if isinstance(pred_vals, np.ndarray):
            pred_vals = pred_vals.flatten()

        st.divider()
        with st.container(border=True):
            st.markdown("### 📊 Prediction Result")

            if info["single_output"]:
                val = float(pred_vals[0] if hasattr(pred_vals, '__iter__') and not isinstance(pred_vals, (str, bytes)) else pred_vals)
                if info["unit"] == "million":
                    st.success(f"## {val:,.2f} million")
                    st.caption(f"≈ {val * 1_000_000:,.0f} visitors")
                elif info["unit"] == "arrivals":
                    st.success(f"## {val:,.0f} arrivals")
                else:
                    st.success(f"## {val:,.0f} visitors")
            else:
                cols = st.columns(min(len(info["target_cols"]), 4))
                labels_map = {
                    "df2": ["% Age Group 15-24", "% Age Group 25-34"],
                }
                labels = labels_map.get(dataset_key, info["target_cols"])
                for i, (label, pv) in enumerate(zip(labels, pred_vals)):
                    with cols[i % len(cols)]:
                        val = float(pv)
                        st.metric(label, f"{val:,.2f}%" if info["unit"] == "percent" else f"{val:,.0f}")

            with st.expander("📋 Input Values Used"):
                st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)
