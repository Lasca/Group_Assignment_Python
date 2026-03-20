"""
ETL pipeline for processing SimFin share price data.

Usage:
    python src/etl.py AAPL
    python src/etl.py AAPL MSFT GOOGL AMZN TSLA META NVDA
    python src/etl.py --all
"""

import argparse
import polars as pl
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA"]


def extract(ticker: str) -> pl.DataFrame:
    """Extract share price data for a given ticker from Parquet."""
    prices = pl.read_parquet(DATA_DIR / "us-shareprices-daily.parquet")
    df = prices.filter(pl.col("Ticker") == ticker)

    if df.is_empty():
        raise ValueError(f"No data found for ticker: {ticker}")

    print(f"[{ticker}] Extracted {df.shape[0]} rows")
    return df


def transform(df: pl.DataFrame) -> pl.DataFrame:
    """Clean data, engineer features, normalize, and create target."""
    # Cast Date, fill Dividend nulls, drop unnecessary columns
    df = (
        df
        .with_columns(
            pl.col("Date").str.to_date("%Y-%m-%d"),
            pl.col("Dividend").fill_null(0.0),
        )
        .drop("SimFinId", "Ticker")
        .sort("Date")
    )

    # Use Adj. Close (accounts for stock splits and dividends)
    df = df.drop("Close").rename({"Adj. Close": "Close"})

    # Feature engineering
    df = df.with_columns([
        (pl.col("Close").pct_change()).alias("daily_return"),
        pl.col("Close").rolling_mean(window_size=5).alias("ma_5"),
        pl.col("Close").rolling_mean(window_size=20).alias("ma_20"),
        pl.col("Close").pct_change().rolling_std(window_size=20).alias("volatility_20"),
        (pl.col("Close").pct_change()).shift(1).alias("return_lag_1"),
        (pl.col("Close").pct_change()).shift(2).alias("return_lag_2"),
        (pl.col("Close").pct_change()).shift(3).alias("return_lag_3"),
    ])

    # Drop nulls from rolling windows and lags
    df = df.drop_nulls()

    # Min-max normalization
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64]]
    df = df.with_columns([
        pl.when(pl.col(c).max() == pl.col(c).min())
        .then(0.0)
        .otherwise((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min()))
        .alias(c)
        for c in numeric_cols
    ])

    # Target: 1 if next day's close is higher, 0 otherwise
    df = df.with_columns(
        (pl.col("Close").shift(-1) > pl.col("Close")).cast(pl.Int8).alias("target")
    )
    df = df.drop_nulls(subset=["target"])

    print(f"  Transformed: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load(df: pl.DataFrame, ticker: str) -> Path:
    """Save the processed DataFrame to Parquet."""
    output_dir = DATA_DIR / "tickers_ml_ready"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{ticker.lower()}_ml_ready.parquet"
    df.write_parquet(output_path)
    print(f"  Saved to {output_path}")
    return output_path


def run_etl(ticker: str) -> Path:
    """Run the full ETL pipeline for a single ticker."""
    print(f"\n{'='*50}")
    print(f"Running ETL for {ticker}")
    print(f"{'='*50}")
    df = extract(ticker)
    df = transform(df)
    return load(df, ticker)


def main():
    parser = argparse.ArgumentParser(description="Run ETL pipeline for SimFin share prices")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols to process")
    parser.add_argument("--all", action="store_true", help="Process all 7 tickers")
    args = parser.parse_args()

    if args.all:
        tickers = TICKERS
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        parser.print_help()
        return

    for ticker in tickers:
        run_etl(ticker)

    print(f"\nDone! Processed {len(tickers)} ticker(s).")


if __name__ == "__main__":
    main()
