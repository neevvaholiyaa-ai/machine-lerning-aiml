import streamlit as st
import pandas as pd

from utils.data_loader import load_raw, DATASET_LABELS

st.title("Data Explorer")
st.markdown("Browse and filter raw datasets")

dataset_key = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASET_LABELS.keys()),
    format_func=lambda k: DATASET_LABELS[k],
)

df = load_raw(dataset_key)

st.subheader(DATASET_LABELS[dataset_key])
st.caption(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

with st.expander("Column Info"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Nulls": df.isnull().sum().values,
        "Unique": [df[c].nunique() for c in df.columns],
    }).reset_index(drop=True)
    st.dataframe(info_df, use_container_width=True)

st.subheader("Filter Data")
filter_col = st.selectbox("Filter by column", df.columns)
if df[filter_col].dtype in ["int64", "float64"]:
    min_v, max_v = float(df[filter_col].min()), float(df[filter_col].max())
    if min_v < max_v:
        range_v = st.slider("Range", min_value=min_v, max_value=max_v, value=(min_v, max_v))
        df = df[(df[filter_col] >= range_v[0]) & (df[filter_col] <= range_v[1])]
else:
    unique_vals = df[filter_col].dropna().unique().tolist()
    selected = st.multiselect("Select values", unique_vals, default=unique_vals[:min(5, len(unique_vals))])
    if selected:
        df = df[df[filter_col].isin(selected)]

row_limit = st.slider("Rows to display", 10, min(1000, len(df)), min(100, len(df)))
st.dataframe(df.head(row_limit), use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download as CSV", data=csv, file_name=f"{dataset_key}.csv", mime="text/csv")
