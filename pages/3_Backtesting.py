import streamlit as st
import polars as pl
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pysimfin import PySimFin

st.set_page_config(page_title="Backtesting", page_icon="chart_with_upwards_trend", layout="wide")

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
        padding: 1rem 0 0.5rem 0;
    }
    .section-header {
        color: #1E90FF;
        padding-bottom: 0.3rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
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
    .result-positive {
        background: linear-gradient(135deg, #0a2e1a 0%, #1a4430 100%);
        border: 1px solid #00FF8840;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-positive p { color: #00FF88; font-size: 1.3rem; font-weight: 600; margin: 0; }
    .result-positive h3 { color: #00FF8880; font-size: 0.9rem; margin-bottom: 0.3rem; }
    .result-negative {
        background: linear-gradient(135deg, #2e0a0a 0%, #441a1a 100%);
        border: 1px solid #FF445540;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-negative p { color: #FF4455; font-size: 1.3rem; font-weight: 600; margin: 0; }
    .result-negative h3 { color: #FF445580; font-size: 0.9rem; margin-bottom: 0.3rem; }

    /* Clean dividers */
    hr {
        border: none;
        border-top: 1px solid #1E90FF15;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Backtesting</div>', unsafe_allow_html=True)
st.write("Simulate the trading strategy on historical data and compare against Buy-and-Hold.")

st.markdown("---")

st.markdown("""
**How the strategy works:**
- **BUY** 1 share when the model predicts the price will go up (probability >= 50%)
- **SELL** 1 share when the model predicts the price will go down (probability < 50%), if we own any shares
- **HOLD** when the model predicts up but we don't have enough cash, or predicts down but we have no shares to sell

The strategy is compared against a **Buy-and-Hold** baseline, which buys as many shares as possible on day one and holds until the end.
""")

st.markdown("---")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TICKERS = ["MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]

# Controls
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.selectbox("Select a company", TICKERS)
with col2:
    initial_cash = st.number_input("Initial capital ($)", value=10000, min_value=1000, step=1000)

st.caption("The backtest runs on the last 20% of historical data (the same test period used during model evaluation).")

if st.button("Run Backtest", type="primary"):
    with st.spinner("Running backtest..."):
        try:
            # Load ML-ready data
            df = pl.read_parquet(os.path.join(DATA_DIR, "tickers_ml_ready", f"{ticker.lower()}_ml_ready.parquet"))

            # Fetch raw prices from SimFin API for actual dollar values
            client = PySimFin()
            start_date = str(test_data["Date"].min())
            end_date = str(test_data["Date"].max())
            raw_api = client.get_share_prices(ticker, start_date, end_date)
            prices_raw = (
                raw_api
                .with_columns(pl.col("Date").str.to_date("%Y-%m-%d"))
                .sort("Date")
                .rename({"Adjusted Closing Price": "Price"})
                .select(["Date", "Price"])
            )

            # Load model
            model = joblib.load(os.path.join(MODELS_DIR, f"{ticker.lower()}_best_model.joblib"))

            # Time-based split (same 80/20 as training)
            split_idx = int(len(df) * 0.8)
            test_data = df.slice(split_idx)

            # Generate predictions
            feature_cols = [c for c in df.columns if c not in ["Date", "target"]]
            X_test = test_data.select(feature_cols).to_numpy()
            predictions = model.predict(X_test)

            # Get actual prices for test period
            test_dates = test_data["Date"].to_list()
            test_prices = prices_raw.filter(pl.col("Date").is_in(test_dates))

            prices_list = test_prices["Price"].to_list()
            dates_list = test_prices["Date"].to_list()

            # Simulate Buy-and-Sell strategy
            cash = initial_cash
            shares = 0
            portfolio_values = []
            actions = []
            cash_history = []

            for i in range(len(prices_list)):
                price = prices_list[i]
                pred = int(predictions[i])

                if pred == 1 and cash >= price:
                    shares += 1
                    cash -= price
                    actions.append("BUY")
                elif pred == 0 and shares > 0:
                    shares -= 1
                    cash += price
                    actions.append("SELL")
                else:
                    actions.append("HOLD")

                portfolio_values.append(cash + shares * price)
                cash_history.append(round(cash, 2))

            # Buy-and-Hold baseline
            first_price = prices_list[0]
            bh_shares = int(initial_cash // first_price)
            bh_cash = initial_cash - bh_shares * first_price
            bh_values = [bh_cash + bh_shares * p for p in prices_list]

            strategy_return = ((portfolio_values[-1] / initial_cash) - 1) * 100
            bh_return = ((bh_values[-1] / initial_cash) - 1) * 100

            # Info cards
            st.markdown(f'<h2 class="section-header">{ticker}: Backtest Results</h2>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="info-card">
                    <h3>Test Period</h3>
                    <p>{dates_list[0]} to {dates_list[-1]}</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="info-card">
                    <h3>Trading Days</h3>
                    <p>{len(prices_list)}</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                buy_count = actions.count("BUY")
                sell_count = actions.count("SELL")
                st.markdown(f"""
                <div class="info-card">
                    <h3>Trades</h3>
                    <p>{buy_count} buys, {sell_count} sells</p>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="info-card">
                    <h3>Final Portfolio</h3>
                    <p>${portfolio_values[-1]:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # Return comparison
            st.markdown("")
            r1, r2 = st.columns(2)
            with r1:
                css_class = "result-positive" if strategy_return >= 0 else "result-negative"
                st.markdown(f"""
                <div class="{css_class}">
                    <h3>ML Strategy Return</h3>
                    <p>{strategy_return:+.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                css_class = "result-positive" if bh_return >= 0 else "result-negative"
                st.markdown(f"""
                <div class="{css_class}">
                    <h3>Buy-and-Hold Return</h3>
                    <p>{bh_return:+.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # Chart
            st.markdown(f'<h2 class="section-header">Portfolio Value Over Time</h2>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')

            ax.plot(dates_list, portfolio_values, label=f"ML Strategy ({strategy_return:+.1f}%)", linewidth=2, color="#1E90FF")
            ax.plot(dates_list, bh_values, label=f"Buy-and-Hold ({bh_return:+.1f}%)", linewidth=2, linestyle="--", color="#FF8C00")
            ax.axhline(y=initial_cash, color="#555555", linestyle=":", alpha=0.5, label="Initial Capital")

            ax.set_xlabel("Date", color="#8899AA")
            ax.set_ylabel("Portfolio Value ($)", color="#8899AA")
            ax.tick_params(colors="#8899AA")
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(facecolor='#1a2744', edgecolor='#1E90FF30', labelcolor='#E8EDF3')

            plt.tight_layout()
            st.pyplot(fig)

            # Trade log
            st.markdown(f'<h2 class="section-header">All Trades</h2>', unsafe_allow_html=True)
            trade_log = pl.DataFrame({
                "Date": [str(d) for d in dates_list],
                "Price": [float(p) for p in prices_list],
                "Prediction": [int(p) for p in predictions[:len(prices_list)]],
                "Action": actions,
                "Available Cash": [float(c) for c in cash_history],
                "Portfolio Value": [float(v) for v in portfolio_values],
            })
            trades_only = trade_log.filter(pl.col("Action") != "HOLD")
            st.caption(f"{len(trades_only)} trades total")
            st.dataframe(
                trades_only.reverse().to_pandas(),
                use_container_width=True,
                height=400,
            )

        except Exception as e:
            st.error(f"Error: {e}")
