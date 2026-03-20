import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

st.title("Automated Daily Trading System")

st.header("About")
st.write("""
This application is an automated daily trading system that uses machine learning
to predict stock market movements. It fetches real-time data from SimFin, applies
data transformations, and generates trading signals using a trained classification model.
""")

st.header("How It Works")
st.write("""
1. **Data Collection**: Historical stock data is downloaded from SimFin and processed
   through an ETL pipeline using Polars.
2. **Model Training**: A machine learning model is trained on the processed data to
   predict whether a stock's price will rise or fall the next trading day.
3. **Live Predictions**: The Go Live page fetches fresh data from the SimFin API,
   applies the same transformations, and feeds it into the trained model to generate
   a prediction for the next day.
""")

st.header("Development Team")
st.write("""
- Martin Sebastian Schneider Vaquero
- Francisco Javier Santiago Concha Bambach
- Aylin Yasgul
- Qiufeng Cai
- Bader Al Eisa
""")

st.header("Tech Stack")
cols = st.columns(4)
cols[0].metric("Data Processing", "Polars")
cols[1].metric("ML Framework", "scikit-learn")
cols[2].metric("API", "SimFin")
cols[3].metric("Web App", "Streamlit")
