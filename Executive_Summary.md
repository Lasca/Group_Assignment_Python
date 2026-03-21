# Executive Summary

## Project: Automated Daily Trading System

### Objective
Build an end-to-end ML-powered trading system that predicts next-day stock price movements using real-time data from the SimFin API.

### Approach

**Offline Phase (Part 1):**
- Downloaded bulk historical share price data from SimFin for 7 US companies
- Built an ETL pipeline using Polars to clean, transform, and engineer features from the raw data
- Features include daily returns, 5-day and 20-day moving averages, 20-day volatility, and 3-day lag returns
- Applied min-max normalization to ensure all features are on the same scale
- Trained both Logistic Regression and Random Forest classifiers, keeping the best model per ticker
- Selected 5 tickers (MSFT, GOOG, AMZN, TSLA, NVDA) that achieved above 50% test accuracy
- Implemented a Buy-and-Sell trading strategy and backtested it against a Buy-and-Hold baseline

**Online Phase (Part 2):**
- Built a Python API wrapper (PySimFin) to fetch live share price data from the SimFin REST API
- Developed a Streamlit web application with three pages:
  - **Home**: Project overview, architecture explanation, tech stack, and team
  - **Go Live**: Real-time data fetching, ETL transformation, and next-day predictions
  - **Backtesting**: Historical trading strategy simulation with portfolio comparison charts

### Key Results
- Models achieve 50-59% accuracy on next-day price direction prediction
- The ML trading strategy shows conservative but positive returns on most tickers
- Buy-and-Hold generally outperforms the ML strategy, which is expected given the simplicity of the model
- The system successfully demonstrates the full pipeline from raw data to live predictions

### Tech Stack
- **Polars**: Chosen over Pandas for superior performance with large datasets and native Parquet support
- **scikit-learn**: Logistic Regression and Random Forest classifiers
- **SimFin**: Bulk data downloads and REST API for live predictions
- **Streamlit**: Rapid prototyping of the interactive web application

### Team
- Martin Sebastian Schneider Vaquero
- Francisco Javier Santiago Concha Bambach
- Aylin Yasgul
- Qiufeng Cai
- Bader Al Eisa
