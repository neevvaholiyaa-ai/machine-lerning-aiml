import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_raw

st.title("India Tourism Analytics Dashboard")
st.markdown("Key insights from India's tourism data (1981–2021)")

dfs = {}
for k in ["df1", "df4", "df5", "df6", "df7", "df8"]:
    dfs[k] = load_raw(k)

dfs["df4"]["Rank of India"] = (
    dfs["df4"]["Rank of India"]
    .astype(str).str.extract(r'(\d+)')[0]
    .astype(float)
    .interpolate(method="linear", limit_direction="forward")
    .astype(int)
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_ftas = dfs["df1"]["FTAs in India (in million)"].sum()
    st.metric("Total FTAs (1981-2020)", f"{total_ftas:.1f}M")
with col2:
    total_arrivals = dfs["df8"]["Domestic -2019"].sum() + dfs["df8"]["Foreign - 2019"].sum()
    st.metric("Total Arrivals (2019)", f"{total_arrivals/1e6:.1f}M")
with col3:
    latest_rank = dfs["df4"]["Rank of India"].iloc[-1]
    st.metric("India's Global Rank", f"{int(latest_rank)}")
with col4:
    covid_drop = dfs["df8"]["Domestic -2020"].sum() / dfs["df8"]["Domestic -2019"].sum() - 1
    st.metric("COVID Domestic Drop (2020)", f"{covid_drop*100:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Top Sources", "Monuments", "States"])

with tab1:
    df1 = dfs["df1"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1["Year"], y=df1["FTAs in India (in million)"], mode="lines+markers", name="FTAs"))
    fig.add_trace(go.Scatter(x=df1["Year"], y=df1["NRIs arrivals in India (in million)"], mode="lines+markers", name="NRIs"))
    fig.add_trace(go.Scatter(x=df1["Year"], y=df1["ITAs in India  (in million)"], mode="lines+markers", name="ITAs"))
    fig.update_layout(title="FTA / NRI / ITA Trends (1981-2020)", xaxis_title="Year", yaxis_title="Arrivals (million)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df5 = dfs["df5"]
    top_countries = df5.groupby("Country of Nationality")["Arrivals (in numbers)"].sum().nlargest(10).reset_index()
    fig = px.bar(top_countries, x="Country of Nationality", y="Arrivals (in numbers)",
                 title="Top 10 Source Countries by Arrivals (2019)", color="Arrivals (in numbers)",
                 color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    purpose = df5[["Business and Professional(%)", "Leisure Holiday and Recreation(%)",
                   "Medical(%)", "Indian Diaspora(%)", "Others(%)"]].mean().reset_index()
    purpose.columns = ["Purpose", "Percentage"]
    fig2 = px.pie(purpose, values="Percentage", names="Purpose", title="Average Purpose of Visit (2019)")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    df6 = dfs["df6"]
    top_monuments = df6.nlargest(10, "Domestic-2019-20")
    fig = px.bar(top_monuments, x="Name of the Monument", y=["Domestic-2019-20", "Foreign-2019-20"],
                 title="Top 10 Monuments by Visitation (2019-20)", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    df8 = dfs["df8"]
    df8_plot = df8[df8["S. No."].astype(str).str.match(r"^\d+$")].copy()
    df8_plot["S. No."] = df8_plot["S. No."].astype(int)
    top_states = df8_plot.nlargest(10, "Domestic -2019")
    fig = px.bar(top_states, x="States/UTs", y=["Domestic -2019", "Foreign - 2019"],
                 title="Top 10 States/UTs by Tourism (2019)", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
