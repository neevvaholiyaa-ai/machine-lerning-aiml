import streamlit as st

st.set_page_config(page_title="India Tourism Analytics", page_icon="🏛️", layout="wide")

pages = {
    "Overview": st.Page("pages/01_Overview.py", title="Overview Dashboard", icon="📊"),
    "Exploratory Analysis": st.Page("pages/02_EDA.py", title="Exploratory Analysis", icon="📈"),
    "ML Predictions": st.Page("pages/03_Predictions.py", title="ML Predictions", icon="🤖"),
    "Data Explorer": st.Page("pages/04_Data_Explorer.py", title="Data Explorer", icon="🗃️"),
}

pg = st.navigation(list(pages.values()), position="sidebar")
pg.run()
