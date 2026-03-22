# Automated Daily Trading System

An ML-powered stock prediction system that fetches real-time data from SimFin and generates next-day price movement predictions.

## Overview

This project implements a full end-to-end pipeline:

1. **ETL Pipeline**: Downloads bulk stock data from SimFin, cleans it, engineers features (moving averages, volatility, lag returns), normalizes, and outputs ML-ready datasets using Polars
2. **Model Training**: Compares Logistic Regression and Random Forest classifiers, keeps the best model for each ticker
3. **Web Application**: Streamlit app with live predictions via the SimFin API and historical backtesting
4. **Trading Strategy**: Buy-and-Sell strategy based on model predictions, compared against a Buy-and-Hold baseline

## Live Application

**[greatassignment.streamlit.app](https://greatassignment.streamlit.app)**

## Supported Tickers

MSFT, GOOG, AMZN, TSLA, NVDA

## Project Structure

```
Home.py                  # Streamlit entry point (Home page)
pages/
  2_Go_Live.py           # Live prediction page
  3_Backtesting.py       # Trading strategy backtest page
src/
  pysimfin.py            # SimFin API wrapper (PySimFin class)
  etl.py                 # Standalone ETL script
  train_model.py         # Model training script (LR + RF comparison)
notebooks/
  Part_1.1_EDA_ETL.ipynb         # Data exploration and ETL pipeline
  Part_1.2_ML_Model.ipynb        # Model training and evaluation
  Part_1.3_Trading_Strategy.ipynb # Trading strategy backtest
models/                  # Trained model files (.joblib)
data/
  tickers_ml_ready/      # ML-ready Parquet files per ticker
  us-shareprices-daily.csv  # Raw SimFin bulk download (5 years of daily prices)
  us-companies.csv          # Company metadata from SimFin
docs/
  Executive_Summary.pdf  # Final executive summary with screenshots
```

## Setup

1. Clone the repository
2. Create a conda environment: `conda create -n my-assignment python=3.12`
3. Install dependencies: `pip install -r requirements.txt`
4. Add your SimFin API key to a `.env` file: `SIMFIN_API_KEY=your_key_here`
5. Run the app: `streamlit run Home.py`

## Tech Stack

- **Polars** for data processing (superior performance with large datasets and Parquet files)
- **scikit-learn** for ML models (Logistic Regression, Random Forest)
- **SimFin** for financial data (bulk downloads + REST API)
- **Streamlit** for the web application

## Team

- Martin Sebastian Schneider Vaquero
- Francisco Javier Santiago Concha Bambach
- Aylin Yasgul
- Qiufeng Cai
- Bader Al Eisa
