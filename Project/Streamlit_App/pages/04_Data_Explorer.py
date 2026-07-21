import streamlit as st
import pandas as pd
from utils.data_loader import load_raw, DATASET_LABELS

st.title("🗃️ Data Explorer")
st.caption("Browse, filter, and download any dataset")

dataset_key = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASET_LABELS.keys()),
    format_func=lambda k: DATASET_LABELS[k],
)

df = load_raw(dataset_key)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", DATASET_LABELS[dataset_key])
    c2.metric("Rows", df.shape[0])
    c3.metric("Columns", df.shape[1])

st.divider()

with st.expander("🔍 Column Information", expanded=True):
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Nulls": df.isnull().sum().values,
        "Null %": (df.isnull().sum() / len(df) * 100).round(1).values,
        "Unique": [df[c].nunique() for c in df.columns],
    }).reset_index(drop=True)
    st.dataframe(info, use_container_width=True, hide_index=True)

st.divider()

st.markdown("### 🔎 Filter Data")
c1, c2 = st.columns([1, 1])
with c1:
    filter_col = st.selectbox("Filter by column", df.columns, key="filter_col")
with c2:
    if df[filter_col].dtype in ["int64", "float64", "Int64", "Float64"]:
        min_v, max_v = float(df[filter_col].min()), float(df[filter_col].max())
        if min_v < max_v:
            range_v = st.slider("Range", min_value=min_v, max_value=max_v,
                                value=(min_v, max_v), key="range_slider")
            df_filtered = df[(df[filter_col] >= range_v[0]) & (df[filter_col] <= range_v[1])]
        else:
            df_filtered = df
            st.info("Column has constant value")
    else:
        unique_vals = df[filter_col].dropna().unique().tolist()
        selected = st.multiselect("Select values", sorted(unique_vals, key=str),
                                  default=sorted(unique_vals, key=str)[:min(5, len(unique_vals))],
                                  key="multi_filter")
        if selected:
            df_filtered = df[df[filter_col].isin(selected)]
        else:
            df_filtered = df

st.divider()
st.markdown(f"### 📄 Preview ({len(df_filtered)} rows)")

row_limit = st.slider("Rows to display", 10, min(1000, len(df_filtered)),
                       min(100, len(df_filtered)), key="row_limit")
st.dataframe(df_filtered.head(row_limit), use_container_width=True, hide_index=True)

st.divider()
csv = df_filtered.to_csv(index=False).encode("utf-8")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.download_button("📥 Download CSV", data=csv,
                       file_name=f"{dataset_key}.csv", mime="text/csv",
                       use_container_width=True)
with c2:
    excel = df_filtered.to_excel(index=False) if False else None
    st.download_button("📥 Download Excel", data=csv,
                       file_name=f"{dataset_key}.csv",
                       mime="text/csv", use_container_width=True,
                       disabled=True)
