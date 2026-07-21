import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_raw, DATASET_LABELS

st.title("📈 Exploratory Data Analysis")
st.caption("Select a dataset from the sidebar to explore with auto-generated visualizations")

dataset_key = st.sidebar.selectbox(
    "Choose Dataset",
    options=list(DATASET_LABELS.keys()),
    format_func=lambda k: DATASET_LABELS[k],
)

df = load_raw(dataset_key)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(numeric_cols))
    if "Year" in df.columns:
        c4.metric("Period", f"{int(df['Year'].min())}–{int(df['Year'].max())}")

if dataset_key == "df1":
    tab_a, tab_b = st.tabs(["📈 Trends", "📊 Distributions"])
    with tab_a:
        cols_line = [c for c in numeric_cols if c != "Year"]
        fig = go.Figure()
        for c in cols_line:
            fig.add_trace(go.Scatter(x=df["Year"], y=df[c],
                                     mode="lines+markers", name=c.split(" (")[0],
                                     line=dict(width=3)))
        fig.update_layout(title="All Metrics Over Time", template="plotly_white",
                          xaxis_title="Year", yaxis_title="Value", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        col_s = st.selectbox("Select column", [c for c in numeric_cols if c != "Year"], key="h1")
        fig = px.histogram(df, x=col_s, marginal="box", title=col_s,
                           template="plotly_white", color_discrete_sequence=["#FF6B35"])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

elif dataset_key == "df2":
    tab_a, tab_b = st.tabs(["👥 Age Group Trends", "📊 Summary"])
    with tab_a:
        age_cols = [c for c in numeric_cols if "Age-Group" in c]
        fig = go.Figure()
        colors = px.colors.sequential.Oranges[2:]
        for i, c in enumerate(age_cols):
            label = c.split(" - ")[-1].strip() if " - " in c else c
            fig.add_trace(go.Scatter(x=df["Year"], y=df[c], mode="lines+markers",
                                     name=label, line=dict(width=2.5, color=colors[i % len(colors)])))
        fig.update_layout(title="Age Group % Distribution", template="plotly_white",
                          xaxis_title="Year", yaxis_title="Percentage", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        avg_age = df[age_cols].mean().reset_index()
        avg_age.columns = ["Age Group", "Average %"]
        avg_age["Age Group"] = avg_age["Age Group"].str.extract(r'(\d+[\d\-]+)').fillna(avg_age["Age Group"])
        fig = px.bar(avg_age, x="Age Group", y="Average %",
                     title="Average Age Group Distribution",
                     color="Average %", color_continuous_scale="Oranges",
                     text_auto=".2f")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

elif dataset_key == "df3":
    tab_a, tab_b = st.tabs(["📅 Quarterly Trends", "📊 Data"])
    with tab_a:
        q_cols = [c for c in numeric_cols if "Quarter" in c]
        fig = go.Figure()
        for c in q_cols:
            label = c.split(" - ")[-1] if " - " in c else c
            fig.add_trace(go.Scatter(x=df["Year"], y=df[c], mode="lines+markers",
                                     name=label, line=dict(width=2.5)))
        fig.update_layout(title="Quarterly Distribution Over Years", template="plotly_white",
                          xaxis_title="Year", yaxis_title="Percentage", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        st.dataframe(df, use_container_width=True)

elif dataset_key == "df4":
    tab_a, tab_b, tab_c = st.tabs(["🌍 World vs India", "🏆 Rank Trend", "📊 Data"])
    with tab_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Year"], y=df["World - Number (in million)"],
                                 mode="lines+markers", name="World",
                                 line=dict(width=3, color="#FF6B35")))
        fig.add_trace(go.Scatter(x=df["Year"], y=df["India - Number (in million)"],
                                 mode="lines+markers", name="India",
                                 line=dict(width=3, color="#1B1B2F")))
        fig.update_layout(title="World vs India Tourism", template="plotly_white",
                          xaxis_title="Year", yaxis_title="Arrivals (million)",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        rank_df = df.copy()
        rank_df["Rank"] = rank_df["Rank of India"].astype(str).str.extract(r'(\d+)')[0].astype(float)
        fig = px.line(rank_df, x="Year", y="Rank", title="India's Global Rank Over Time",
                      markers=True, template="plotly_white")
        fig.update_traces(line=dict(color="#FF6B35", width=3))
        fig.update_layout(yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    with tab_c:
        st.dataframe(df, use_container_width=True)

elif dataset_key == "df5":
    tab_a, tab_b, tab_c = st.tabs(["🌍 Top Countries", "🎯 Purpose & Region", "📊 Data"])
    with tab_a:
        df5_clean = df[~df["Country of Nationality"].str.lower().str.contains("total|grand", na=False)]
        top_c = df5_clean.groupby("Country of Nationality")["Arrivals (in numbers)"].sum().nlargest(15).reset_index()
        fig = px.bar(top_c, x="Country of Nationality", y="Arrivals (in numbers)",
                     title="Top 15 Countries by Arrivals (2019)",
                     color="Arrivals (in numbers)", color_continuous_scale="Oranges",
                     text_auto=".2s")
        fig.update_layout(xaxis_tickangle=-45, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        c1, c2 = st.columns(2)
        with c1:
            pc = ["Business and Professional(%)", "Leisure Holiday and Recreation(%)",
                  "Medical(%)", "Indian Diaspora(%)", "Others(%)"]
            avg_p = df[pc].mean().reset_index()
            avg_p.columns = ["Purpose", "%"]
            fig = px.pie(avg_p, values="%", names="Purpose", hole=0.45,
                         title="Purpose of Visit (Average)",
                         color_discrete_sequence=px.colors.sequential.Oranges_r)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            reg = df5_clean.groupby("Region")["Arrivals (in numbers)"].sum().reset_index()
            fig = px.bar(reg, x="Region", y="Arrivals (in numbers)",
                         title="Arrivals by Region",
                         color="Arrivals (in numbers)", color_continuous_scale="Oranges",
                         text_auto=".2s")
            fig.update_layout(xaxis_tickangle=-30, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with tab_c:
        st.dataframe(df, use_container_width=True)

elif dataset_key == "df6":
    tab_a, tab_b, tab_c = st.tabs(["🏛️ Top Monuments", "📊 Comparison", "📊 Data"])
    with tab_a:
        df6_clean = df[~df["Name of the Monument"].str.lower().str.contains("total|grand", na=False)]
        top_m = df6_clean.nlargest(15, "Domestic-2019-20")
        fig = px.bar(top_m, x="Name of the Monument",
                     y=["Domestic-2019-20", "Foreign-2019-20"],
                     title="Top 15 Monuments by Visitation",
                     barmode="group",
                     color_discrete_sequence=["#FF6B35", "#1B1B2F"],
                     text_auto=".2s")
        fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        fig = px.scatter(df6_clean, x="Domestic-2019-20", y="Foreign-2019-20",
                         hover_data=["Name of the Monument"],
                         title="Domestic vs Foreign Visitation",
                         template="plotly_white", trendline="ols",
                         color_discrete_sequence=["#FF6B35"])
        st.plotly_chart(fig, use_container_width=True)
    with tab_c:
        st.dataframe(df, use_container_width=True)

elif dataset_key == "df7":
    tab_a, tab_b, tab_c = st.tabs(["🌍 By Region", "📅 Yearly Totals", "📊 Data"])
    with tab_a:
        region_cols = [c for c in df.columns if "Region_" in c]
        if region_cols:
            rd = df[region_cols].sum().reset_index()
            rd.columns = ["Region", "Count"]
            rd["Region"] = rd["Region"].str.replace("Region_", "")
            rd = rd.sort_values("Count")
            fig = px.bar(rd, x="Count", y="Region", orientation="h",
                         title="Total by Region",
                         color="Count", color_continuous_scale="Oranges",
                         text_auto=".2s")
            fig.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        if all(c in df.columns for c in ["Number of Arrivals - 2017", "Number of Arrivals - 2018", "Number of Arrivals - 2019"]):
            yd = df[["Number of Arrivals - 2017", "Number of Arrivals - 2018", "Number of Arrivals - 2019"]].sum().reset_index()
            yd.columns = ["Year", "Arrivals"]
            yd["Year"] = yd["Year"].str.extract(r'(\d{4})')
            fig = px.bar(yd, x="Year", y="Arrivals", title="Total Arrivals (2017–2019)",
                         color="Arrivals", color_continuous_scale="Oranges",
                         text_auto=".2s")
            fig.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with tab_c:
        st.dataframe(df, use_container_width=True)

elif dataset_key == "df8":
    tab_a, tab_b, tab_c = st.tabs(["🗺️ State Comparison", "📉 COVID Impact", "📊 Data"])
    with tab_a:
        df8_clean = df[df["S. No."].astype(str).str.match(r"^\d+$")].copy() if "S. No." in df.columns else df
        top_s = df8_clean.nlargest(15, "Domestic -2019") if "Domestic -2019" in df8_clean.columns else df8_clean
        fig = px.bar(top_s, x="States/UTs",
                     y=["Domestic -2019", "Foreign - 2019"],
                     title="Domestic vs Foreign by State (2019)",
                     barmode="group",
                     color_discrete_sequence=["#FF6B35", "#1B1B2F"],
                     text_auto=".2s")
        fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        df8_c = df8_clean.copy()
        df8_c["Drop %"] = ((df8_c["Domestic -2020"] - df8_c["Domestic -2019"]) / df8_c["Domestic -2019"] * 100)
        df8_c = df8_c.sort_values("Drop %")
        fig = px.bar(df8_c.head(15), x="States/UTs", y="Drop %",
                     title="Worst COVID Impact on Domestic Tourism (2020)",
                     color="Drop %", color_continuous_scale="Reds",
                     text_auto=".1f")
        fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with tab_c:
        st.dataframe(df, use_container_width=True)

else:
    st.info("👈 Select a dataset from the sidebar to get started")
