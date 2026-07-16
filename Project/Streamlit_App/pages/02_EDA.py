import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_raw, DATASET_LABELS

st.title("Exploratory Data Analysis")
st.markdown("Interactive visualizations for all 8 datasets")

dataset_key = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASET_LABELS.keys()),
    format_func=lambda k: DATASET_LABELS[k],
)

df = load_raw(dataset_key)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

st.subheader(f"Dataset: {DATASET_LABELS[dataset_key]}")
st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📈 Trends", "🔗 Correlations"])

with tab1:
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column for histogram", numeric_cols)
        fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    if len(categorical_cols) > 0:
        selected_cat = st.selectbox("Select categorical column", categorical_cols)
        value_counts = df[selected_cat].value_counts().reset_index()
        value_counts.columns = [selected_cat, "Count"]
        fig = px.bar(value_counts.head(20), x=selected_cat, y="Count", title=f"Top values in {selected_cat}")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "Year" in df.columns and len(numeric_cols) > 1:
        year_cols = [c for c in numeric_cols if c != "Year"]
        selected_trend = st.multiselect("Select columns to trend", year_cols, default=year_cols[:min(3, len(year_cols))])
        if selected_trend:
            fig = go.Figure()
            for c in selected_trend:
                fig.add_trace(go.Scatter(x=df["Year"], y=df[c], mode="lines+markers", name=c))
            fig.update_layout(title="Trends over Year", xaxis_title="Year", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
    elif len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
        y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlation Matrix", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Raw Data Preview")
st.dataframe(df.head(100), use_container_width=True)
