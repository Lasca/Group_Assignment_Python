import streamlit as st
import polars as pl
import joblib
from datetime import date, timedelta
import sys
import os

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pysimfin import PySimFin

st.set_page_config(page_title="Go Live", page_icon="chart_with_upwards_trend", layout="wide")

# Reuse blue theme CSS
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
        padding: 1rem 0 0.5rem 0;
    }
    .prediction-up {
        background: linear-gradient(135deg, #0a2e1a 0%, #1a4430 100%);
        border: 1px solid #00FF8840;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-up h2 {
        color: #00FF88;
        margin: 0;
    }
    .prediction-down {
        background: linear-gradient(135deg, #2e0a0a 0%, #441a1a 100%);
        border: 1px solid #FF445540;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-down h2 {
        color: #FF4455;
        margin: 0;
    }
    .info-card {
        background: linear-gradient(135deg, #1a2744 0%, #0f1a2e 100%);
        border: 1px solid #1E90FF30;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .info-card h3 {
        color: #1E90FF;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .info-card p {
        color: #E8EDF3;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
    }
    .section-header {
        color: #1E90FF;
        padding-bottom: 0.3rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
    }

    /* Clean dividers */
    hr {
        border: none;
        border-top: 1px solid #1E90FF15;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Go Live</div>', unsafe_allow_html=True)
st.write("Fetch real-time data and generate next-day predictions.")

st.markdown("---")

TICKERS = ["MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]

# Controls
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.selectbox("Select a company", TICKERS, index=TICKERS.index("TSLA"))
with col2:
    start_date = st.date_input("Start date", date.today() - timedelta(days=90))
with col3:
    end_date = st.date_input("End date", date.today())


@st.cache_data(ttl=300)
def fetch_prices(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Fetch share prices from SimFin API (cached for 5 minutes)."""
    client = PySimFin()
    return client.get_share_prices(ticker, start, end)


def apply_etl(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the same ETL transformations used during training."""
    column_map = {
        "Adjusted Closing Price": "Close",
        "Opening Price": "Open",
        "Highest Price": "High",
        "Lowest Price": "Low",
        "Trading Volume": "Volume",
        "Dividend Paid": "Dividend",
        "Common Shares Outstanding": "Shares Outstanding",
    }
    df = df.rename(column_map).drop("Last Closing Price")

    df = df.with_columns([
        pl.col("Date").str.to_date("%Y-%m-%d"),
        pl.col("Open").cast(pl.Float64),
        pl.col("High").cast(pl.Float64),
        pl.col("Low").cast(pl.Float64),
        pl.col("Close").cast(pl.Float64),
        pl.col("Volume").cast(pl.Int64),
        pl.col("Shares Outstanding").cast(pl.Int64),
    ]).sort("Date")

    df = df.with_columns(pl.col("Dividend").fill_null(0.0))

    df = df.with_columns([
        (pl.col("Close").pct_change()).alias("daily_return"),
        pl.col("Close").rolling_mean(window_size=5).alias("ma_5"),
        pl.col("Close").rolling_mean(window_size=20).alias("ma_20"),
        pl.col("Close").pct_change().rolling_std(window_size=20).alias("volatility_20"),
        (pl.col("Close").pct_change()).shift(1).alias("return_lag_1"),
        (pl.col("Close").pct_change()).shift(2).alias("return_lag_2"),
        (pl.col("Close").pct_change()).shift(3).alias("return_lag_3"),
    ])

    df = df.drop_nulls()

    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64]]
    df = df.with_columns([
        pl.when(pl.col(c).max() == pl.col(c).min())
        .then(0.0)
        .otherwise((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min()))
        .alias(c)
        for c in numeric_cols
    ])

    return df


if st.button("Fetch Data and Predict", type="primary"):
    with st.spinner("Fetching data from SimFin..."):
        try:
            raw_prices = fetch_prices(ticker, str(start_date), str(end_date))

            # Price chart
            st.markdown(f'<h2 class="section-header">{ticker}: Historical Prices</h2>', unsafe_allow_html=True)
            chart_data = raw_prices.select([
                pl.col("Date").str.to_date("%Y-%m-%d"),
                pl.col("Adjusted Closing Price").cast(pl.Float64).alias("Close"),
            ]).sort("Date")
            st.line_chart(chart_data.to_pandas().set_index("Date")["Close"])

            # Apply ETL
            transformed = apply_etl(raw_prices)

            if transformed.is_empty():
                st.warning("Not enough data after transformations. Try a wider date range.")
            else:
                # Load model and predict
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models",
                    f"{ticker.lower()}_best_model.joblib",
                )
                model = joblib.load(model_path)

                feature_cols = [c for c in transformed.columns if c not in ["Date", "target"]]
                latest = transformed.select(feature_cols).tail(1).to_numpy()
                prediction = model.predict(latest)[0]
                probability = model.predict_proba(latest)[0]
                confidence = int(max(probability) * 100)
                model_type = type(model).__name__.replace("Classifier", "")

                latest_date = transformed["Date"].max()

                # Info cards
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Ticker</h3>
                        <p>{ticker}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    last_price = chart_data["Close"].to_list()[-1]
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Latest Close</h3>
                        <p>${last_price:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Data Points</h3>
                        <p>{len(chart_data)}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Prediction display
                st.markdown(f'<h2 class="section-header">Next-Day Prediction</h2>', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-up">
                        <h2>UP after {latest_date} ({confidence}% confidence)</h2>
                        <p style="color: #88DDAA;">The {model_type} model predicts {ticker} will increase in price.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-down">
                        <h2>DOWN after {latest_date} ({confidence}% confidence)</h2>
                        <p style="color: #DD8888;">The {model_type} model predicts {ticker} will decrease in price.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Raw data
                st.markdown(f'<h2 class="section-header">Raw Data (from API)</h2>', unsafe_allow_html=True)
                raw_display = raw_prices.with_columns(pl.col("Date").str.to_date("%Y-%m-%d").cast(pl.Utf8)).to_pandas()
                st.dataframe(raw_display, height=300, use_container_width=True)

                # Transformed data
                st.markdown(f'<h2 class="section-header">Transformed Data (after ETL & Normalization)</h2>', unsafe_allow_html=True)
                display_df = (
                    transformed
                    .filter(~((pl.col("Close") == 0) & (pl.col("High") == 0)))
                    .with_columns(pl.col("Date").cast(pl.Utf8))
                    .select(
                        [c for c in transformed.columns if c not in ["Dividend", "Shares Outstanding"]]
                        + ["Dividend", "Shares Outstanding"]
                    )
                    .to_pandas()
                )
                st.dataframe(
                    display_df,
                    height=400,
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error: {e}")
