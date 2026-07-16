import pandas as pd
import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "All_Data_set_Tourism")

FILES = {
    "df1": "India-Tourism-Statistics-1981-2020-fta_nri_ita.csv",
    "df2": "India-Tourism-Statistics-2001-2019-agegroup.csv",
    "df3": "India-Tourism-Statistics-2001-2019-quaterly.csv",
    "df4": "India-Tourism-Statistics-2001-2019-worldvsindia.csv",
    "df5": "India-Tourism-Statistics-2019_region-and-reason.csv",
    "df6": "India-Tourism-Statistics-2021-monuments.csv",
    "df7": "India-Tourism-Statistics-region-2017-2019.csv",
    "df8": "India-Tourism-Statistics-statewise_2019-2020_domestic_foreign.csv",
}

DATASET_LABELS = {
    "df1": "FTAs / NRI / ITA Arrivals (1981-2020)",
    "df2": "Age Group Distribution (2001-2019)",
    "df3": "Quarterly Distribution (2001-2019)",
    "df4": "World vs India Tourism (2001-2019)",
    "df5": "Region & Reason for Travel (2019)",
    "df6": "Monument Visitation (2021)",
    "df7": "Region Arrivals (2017-2019)",
    "df8": "State-wise Domestic/Foreign (2019-2020)",
}

@st.cache_data
def load_raw(key):
    path = os.path.join(DATA_DIR, FILES[key])
    return pd.read_csv(path)

@st.cache_data
def load_all_raw():
    return {k: load_raw(k) for k in FILES}

@st.cache_data
def load_cleaned(key):
    path = os.path.join(BASE_DIR, "..", "Cleaned_Data_Tourism", f"{key}.xlsx")
    return pd.read_excel(path, engine="openpyxl")
