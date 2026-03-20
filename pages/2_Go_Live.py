import streamlit as st
import polars as pl
import joblib
from datetime import date, timedelta
import sys
import os

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pysimfin import PySimFin

st.set_page_config(page_title="Go Live", page_icon="🔴", layout="wide")

st.title("Go Live: Real-Time Predictions")

# Available tickers (will expand later to 5+)
TICKERS = ["AAPL"]

# Ticker selector
ticker = st.selectbox("Select a company", TICKERS)

# Date range for historical display
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", date.today() - timedelta(days=90))
with col2:
    end_date = st.date_input("End date", date.today())


@st.cache_data(ttl=300)
def fetch_prices(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Fetch share prices from SimFin API (cached for 5 minutes)."""
    client = PySimFin()
    return client.get_share_prices(ticker, start, end)


def apply_etl(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the same ETL transformations used during training."""
    # Rename API columns to match our training data format
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

    # Cast types
    df = df.with_columns([
        pl.col("Date").str.to_date("%Y-%m-%d"),
        pl.col("Open").cast(pl.Float64),
        pl.col("High").cast(pl.Float64),
        pl.col("Low").cast(pl.Float64),
        pl.col("Close").cast(pl.Float64),
        pl.col("Volume").cast(pl.Int64),
        pl.col("Shares Outstanding").cast(pl.Int64),
    ]).sort("Date")

    # Fill null dividends
    df = df.with_columns(pl.col("Dividend").fill_null(0.0))

    # Feature engineering (same as training)
    df = df.with_columns([
        (pl.col("Close").pct_change()).alias("daily_return"),
        pl.col("Close").rolling_mean(window_size=5).alias("ma_5"),
        pl.col("Close").rolling_mean(window_size=20).alias("ma_20"),
        pl.col("Close").pct_change().rolling_std(window_size=20).alias("volatility_20"),
        (pl.col("Close").pct_change()).shift(1).alias("return_lag_1"),
        (pl.col("Close").pct_change()).shift(2).alias("return_lag_2"),
        (pl.col("Close").pct_change()).shift(3).alias("return_lag_3"),
    ])

    # Drop nulls from rolling windows
    df = df.drop_nulls()

    # Normalize using min-max scaling (fill 0 if column has constant values)
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64]]
    df = df.with_columns([
        pl.when(pl.col(c).max() == pl.col(c).min())
        .then(0.0)
        .otherwise((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min()))
        .alias(c)
        for c in numeric_cols
    ])

    return df


if st.button("Fetch Data and Predict"):
    with st.spinner("Fetching data from SimFin..."):
        try:
            raw_prices = fetch_prices(ticker, str(start_date), str(end_date))

            # Show raw price chart
            st.subheader(f"{ticker}: Historical Prices")
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
                    f"{ticker.lower()}_logistic_regression.joblib",
                )
                model = joblib.load(model_path)

                # Use the latest row for prediction
                feature_cols = [c for c in transformed.columns if c not in ["Date", "target"]]
                latest = transformed.select(feature_cols).tail(1).to_numpy()
                prediction = model.predict(latest)[0]

                # Display prediction
                st.subheader("Next-Day Prediction")
                latest_date = transformed["Date"].max()

                if prediction == 1:
                    st.success(f"Model predicts {ticker} will go **UP** after {latest_date}")
                else:
                    st.error(f"Model predicts {ticker} will go **DOWN** after {latest_date}")

                # Show recent transformed data
                st.subheader("Recent Transformed Data")
                st.dataframe(transformed.tail(10).to_pandas())

        except Exception as e:
            st.error(f"Error: {e}")
