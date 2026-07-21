import streamlit as st

st.set_page_config(page_title="India Tourism Analytics", page_icon="🏛️", layout="wide")

ORANGE = "#FF6B35"
GREEN = "#00B894"
BLUE = "#0984E3"
WHITE = "#FFFFFF"
DARK = "#2D3436"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: #F5F7FA; }}
    .block-container {{ padding-top: 1.5rem !important; }}

    /* HEADER — orange + blue gradient */
    .main-header {{
        background: linear-gradient(135deg, {ORANGE} 0%, {BLUE} 100%);
        padding: 2rem 2.5rem; border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(255,107,53,0.25);
    }}
    .main-header h1 {{ color: white; margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; }}
    .main-header p {{ color: rgba(255,255,255,0.9); margin: 0.3rem 0 0 0; font-size: 1.05rem; }}

    /* METRIC CARDS — orange border left */
    div[data-testid="stMetric"] {{
        background: white; padding: 1.2rem 1.5rem;
        border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border-left: 5px solid {ORANGE};
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,180,148,0.15);
    }}
    div[data-testid="stMetric"] label {{ color: #6c757d; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {DARK}; font-weight: 800; font-size: 1.8rem; }}

    /* SIDEBAR — white bg, dark text */
    section[data-testid="stSidebar"] {{
        background: {WHITE} !important;
        border-right: 1px solid #E9ECEF !important;
        box-shadow: 2px 0 12px rgba(0,0,0,0.04);
    }}
    section[data-testid="stSidebar"] .st-emotion-cache-1v7f65g,
    section[data-testid="stSidebar"] .st-emotion-cache-1v7f65g p {{
        font-size: 1.05rem !important; font-weight: 600 !important;
        color: {DARK} !important;
        padding: 0.8rem 1.2rem !important;
        border-radius: 14px !important; margin: 3px 12px !important;
        transition: all 0.25s ease !important;
        border: none !important;
    }}
    section[data-testid="stSidebar"] .st-emotion-cache-1v7f65g:hover {{
        color: {ORANGE} !important;
        background: rgba(255,107,53,0.08) !important;
        transform: translateX(5px);
    }}
    section[data-testid="stSidebar"] .st-emotion-cache-1v7f65g[data-selected="true"] {{
        color: {WHITE} !important;
        background: {GREEN} !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0,180,148,0.3);
    }}
    section[data-testid="stSidebar"] .st-emotion-cache-1wivap2 {{ background: transparent !important; }}
    section[data-testid="stSidebar"] .st-emotion-cache-1wivap2 .st-emotion-cache-1v7f65g {{ border-left: none !important; }}
    section[data-testid="stSidebar"] .st-emotion-cache-187zlkm {{ color: #888; font-size: 0.8rem; }}
    section[data-testid="stSidebar"] .st-emotion-cache-lf0w8q {{ color: #888; }}

    /* TABS — orange active */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; background: #E9ECEF; border-radius: 16px; padding: 5px; margin-bottom: 1rem; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px; padding: 0.6rem 1.2rem;
        font-weight: 600; font-size: 0.9rem;
        border: none; color: #495057; transition: all 0.25s ease;
    }}
    .stTabs [aria-selected="true"] {{ background: {ORANGE} !important; color: white !important; box-shadow: 0 3px 10px rgba(255,107,53,0.3); }}
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{ background: rgba(255,107,53,0.1); color: {ORANGE}; }}

    /* BUTTONS — green */
    .stButton button {{
        border-radius: 14px !important; font-weight: 700 !important;
        font-size: 1rem !important; border: none !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.25s ease !important;
        background: linear-gradient(135deg, {GREEN}, #00A381) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0,180,148,0.3);
    }}
    .stButton button:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0,180,148,0.45) !important;
    }}

    /* CONTAINERS */
    .st-emotion-cache-1r6slb0, div[data-testid="stExpander"] {{
        border-radius: 16px !important; border: 1px solid #E9ECEF;
        background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .st-emotion-cache-l9fmaj {{ border-radius: 16px !important; }}
    .stAlert {{ border-radius: 14px !important; border: none !important; }}

    /* HEADINGS */
    h1 {{ color: {DARK}; font-weight: 800; letter-spacing: -0.5px; font-size: 2rem; }}
    h2 {{ color: {DARK}; font-weight: 700; letter-spacing: -0.3px; font-size: 1.5rem; }}
    h3 {{ color: {DARK}; font-weight: 600; letter-spacing: -0.2px; }}

    /* INPUTS */
    .stSelectbox label, .stSlider label, .stNumberInput label {{
        font-weight: 600 !important; color: {DARK} !important; font-size: 0.85rem !important;
    }}
    .st-bb, .st-bw, .st-cx {{ border-radius: 12px !important; }}

    /* DIVIDER */
    hr {{ margin: 2rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, #E9ECEF, transparent); }}

    /* DATA FRAME */
    .stDataFrame {{ border-radius: 16px; overflow: hidden; border: 1px solid #E9ECEF; }}
    footer {{ display: none; }}

    /* SIDEBAR LABELS */
    section[data-testid="stSidebar"] .st-bb {{ border-radius: 12px !important; }}
    section[data-testid="stSidebar"] label {{
        color: {DARK} !important; font-weight: 700 !important;
        font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    section[data-testid="stSidebar"] .st-emotion-cache-1v1d5tm {{ color: {DARK} !important; }}

    .stCaption {{ color: #6c757d !important; font-size: 0.85rem !important; }}
    div.row-widget.stColumns > div {{ padding: 0 0.5rem; }}

    /* BLUE & GREEN accent helpers */
    .st-badge-blue {{ background: {BLUE}; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }}
    .st-badge-green {{ background: {GREEN}; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }}
    .st-badge-orange {{ background: {ORANGE}; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
    <h1>🏛️ India Tourism Analytics</h1>
    <p>Data-driven insights &amp; machine learning predictions on India's tourism sector (1981–2021)</p>
</div>
""", unsafe_allow_html=True)

pages = {
    "Overview": st.Page("pages/01_Overview.py", title="Dashboard", icon="📊"),
    "Exploratory Analysis": st.Page("pages/02_EDA.py", title="EDA", icon="📈"),
    "ML Predictions": st.Page("pages/03_Predictions.py", title="Predictions", icon="🤖"),
    "Data Explorer": st.Page("pages/04_Data_Explorer.py", title="Data Explorer", icon="🗃️"),
}

pg = st.navigation(list(pages.values()), position="sidebar")
pg.run()
