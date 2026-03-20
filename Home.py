import streamlit as st

st.set_page_config(
    page_title="Trading System",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Blue theme CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a2744 100%);
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E90FF, #00BFFF, #87CEFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-title {
        text-align: center;
        color: #8899AA;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a2744 0%, #0f1a2e 100%);
        border: 1px solid #1E90FF30;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #1E90FF80;
    }
    .metric-card h3 {
        color: #1E90FF;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .metric-card p {
        color: #E8EDF3;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }
    .section-header {
        color: #1E90FF;
        border-bottom: 2px solid #1E90FF40;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .team-member {
        background: #1a2744;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border-left: 3px solid #1E90FF;
    }
    .step-card {
        background: linear-gradient(135deg, #1a2744 0%, #0f1a2e 100%);
        border: 1px solid #1E90FF20;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .step-number {
        color: #1E90FF;
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Automated Daily Trading System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ML-powered stock predictions using real-time SimFin data</div>', unsafe_allow_html=True)

st.markdown("---")

# About section
st.markdown('<h2 class="section-header">About</h2>', unsafe_allow_html=True)
st.write("""
This application is an automated daily trading system that uses machine learning
to predict stock market movements. It fetches real-time data from SimFin, applies
data transformations, and generates trading signals using a trained classification model.
""")

# How it works
st.markdown('<h2 class="section-header">How It Works</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">01</div>
        <h4>Data Collection</h4>
        <p style="color: #8899AA; font-size: 0.9rem;">
            Historical stock data is downloaded from SimFin and processed through
            an ETL pipeline using Polars.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">02</div>
        <h4>Model Training</h4>
        <p style="color: #8899AA; font-size: 0.9rem;">
            A machine learning model is trained on the processed data to predict
            whether a stock's price will rise or fall the next day.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">03</div>
        <h4>Live Predictions</h4>
        <p style="color: #8899AA; font-size: 0.9rem;">
            The Go Live page fetches fresh data from the SimFin API, applies the
            same transformations, and generates a prediction for the next day.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tech Stack
st.markdown('<h2 class="section-header">Tech Stack</h2>', unsafe_allow_html=True)

cols = st.columns(4)
tech = [
    ("Data Processing", "Polars"),
    ("ML Framework", "scikit-learn"),
    ("Data API", "SimFin"),
    ("Web App", "Streamlit"),
]
for col, (label, value) in zip(cols, tech):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)

# Team
st.markdown('<h2 class="section-header">Development Team</h2>', unsafe_allow_html=True)

team = [
    "Martin Sebastian Schneider Vaquero",
    "Francisco Javier Santiago Concha Bambach",
    "Aylin Yasgul",
    "Qiufeng Cai",
    "Bader Al Eisa",
]
for member in team:
    st.markdown(f'<div class="team-member">{member}</div>', unsafe_allow_html=True)
