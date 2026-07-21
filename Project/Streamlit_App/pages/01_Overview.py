import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_raw

st.title("📊 Dashboard Overview")
st.caption("Key performance indicators and insights from India's tourism data (1981–2021)")

dfs = {}
for k in ["df1", "df4", "df5", "df6", "df7", "df8"]:
    dfs[k] = load_raw(k)

dfs["df4"]["Rank of India"] = (
    dfs["df4"]["Rank of India"].astype(str).str.extract(r'(\d+)')[0]
    .astype(float).interpolate(method="linear", limit_direction="forward").astype(int)
)

dfs["df8"] = dfs["df8"][dfs["df8"]["S. No."].astype(str).str.match(r"^\d+$")].copy()
dfs["df8"]["S. No."] = dfs["df8"]["S. No."].astype(int)

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
    st.metric("COVID Drop (2020)", f"{covid_drop*100:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🌍 Top Sources", "🏛️ Monuments", "🗺️ States"])

with tab1:
    df1 = dfs["df1"]
    fig = go.Figure()
    colors = {"FTAs": "#FF6B35", "NRIs": "#1B1B2F", "ITAs": "#6C757D"}
    for col, name in [("FTAs in India (in million)", "FTAs"),
                      ("NRIs arrivals in India (in million)", "NRIs"),
                      ("ITAs in India  (in million)", "ITAs")]:
        fig.add_trace(go.Scatter(
            x=df1["Year"], y=df1[col],
            mode="lines+markers", name=name,
            line=dict(width=3, color=colors[name]),
            marker=dict(size=8, color=colors[name])
        ))
    fig.update_layout(
        title="FTA / NRI / ITA Trends (1981-2020)",
        xaxis_title="Year", yaxis_title="Arrivals (million)",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df5 = dfs["df5"]
    df5_clean = df5[~df5["Country of Nationality"].str.lower().str.contains("total|grand", na=False)]
    top_countries = df5_clean.groupby("Country of Nationality")["Arrivals (in numbers)"].sum().nlargest(10).reset_index()
    fig = px.bar(top_countries, x="Country of Nationality", y="Arrivals (in numbers)",
                 title="Top 10 Source Countries by Arrivals (2019)",
                 color="Arrivals (in numbers)", color_continuous_scale="Oranges",
                 text_auto=".2s")
    fig.update_layout(xaxis_tickangle=-30, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        purpose = df5[["Business and Professional(%)", "Leisure Holiday and Recreation(%)",
                       "Medical(%)", "Indian Diaspora(%)", "Others(%)"]].mean().reset_index()
        purpose.columns = ["Purpose", "Percentage"]
        fig2 = px.pie(purpose, values="Percentage", names="Purpose",
                      title="Average Purpose of Visit (2019)",
                      color_discrete_sequence=px.colors.sequential.Oranges_r,
                      hole=0.45)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        region_data = df5_clean.groupby("Region")["Arrivals (in numbers)"].sum().nlargest(5).reset_index()
        fig3 = px.bar(region_data, x="Region", y="Arrivals (in numbers)",
                      title="Top 5 Regions by Arrivals (2019)",
                      color="Arrivals (in numbers)", color_continuous_scale="Oranges",
                      text_auto=".2s")
        fig3.update_layout(xaxis_tickangle=-20, template="plotly_white", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    df6 = dfs["df6"]
    df6_clean = df6[~df6["Name of the Monument"].str.lower().str.contains("total|grand", na=False)]
    top_monuments = df6_clean.nlargest(10, "Domestic-2019-20")
    fig = px.bar(top_monuments, x="Name of the Monument",
                 y=["Domestic-2019-20", "Foreign-2019-20"],
                 title="Top 10 Monuments by Visitation (2019-20)",
                 barmode="group",
                 color_discrete_sequence=["#FF6B35", "#1B1B2F"],
                 text_auto=".2s")
    fig.update_layout(xaxis_tickangle=-30, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    df8 = dfs["df8"]
    top_states = df8.nlargest(10, "Domestic -2019")
    fig = px.bar(top_states, x="States/UTs",
                 y=["Domestic -2019", "Foreign - 2019"],
                 title="Top 10 States/UTs by Tourism (2019)",
                 barmode="group",
                 color_discrete_sequence=["#FF6B35", "#1B1B2F"],
                 text_auto=".2s")
    fig.update_layout(xaxis_tickangle=-30, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
