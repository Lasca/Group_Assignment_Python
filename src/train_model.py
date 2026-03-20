"""
Train a Logistic Regression model for each ticker.

Usage:
    python src/train_model.py AAPL
    python src/train_model.py AAPL MSFT GOOG AMZN TSLA META NVDA
    python src/train_model.py --all
"""

import argparse
import polars as pl
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA"]


def train_model(ticker: str):
    """Train and export a Logistic Regression model for a single ticker."""
    print(f"\n{'='*50}")
    print(f"Training model for {ticker}")
    print(f"{'='*50}")

    # Load ML-ready data
    data_path = DATA_DIR / "tickers_ml_ready" / f"{ticker.lower()}_ml_ready.parquet"
    df = pl.read_parquet(data_path)

    # Features: everything except Date and target
    feature_cols = [c for c in df.columns if c not in ["Date", "target"]]
    X = df.select(feature_cols).to_numpy()
    y = df["target"].to_numpy()

    # Time-based split: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{ticker.lower()}_logistic_regression.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train ML models for stock prediction")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols to train")
    parser.add_argument("--all", action="store_true", help="Train all 7 tickers")
    args = parser.parse_args()

    if args.all:
        tickers = TICKERS
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        parser.print_help()
        return

    results = {}
    for ticker in tickers:
        results[ticker] = train_model(ticker)

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for ticker, acc in results.items():
        print(f"  {ticker}: {acc:.4f}")


if __name__ == "__main__":
    main()
