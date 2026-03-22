# AI Usage Log

This document describes how AI tools were used throughout the development of this project.

## Tool Used

**Claude Code** (Anthropic) via the VSCode extension

## How AI Was Used

### Ideation and Planning
- Helped structure the project into logical steps (ETL, model training, API wrapper, web app)
- Suggested feature engineering approaches (moving averages, volatility, lag returns)
- Recommended comparing Logistic Regression vs Random Forest and keeping the best model

### Code Development
- Assisted with writing the ETL pipeline using Polars
- Helped build the SimFin API wrapper (PySimFin class)
- Assisted with Streamlit page development and blue theme styling
- Helped implement the backtesting page with portfolio simulation

### Debugging
- Helped to identify and fix data type issues (e.g. string dates, NaN from min-max normalization on constant columns)
- Resolved feature mismatch between training data and live API data
- Fixed Polars deprecation warnings

### Learning
- Explained financial concepts (Adjusted Close, stock splits, Buy-and-Hold strategy)
- Clarified ML concepts (why time-based splits for time series, why normalization matters for Logistic Regression)
- Helped the team understand the full data pipeline from raw CSV to live prediction

## What Worked Well
- Iterative development: building one step at a time (ETL for one ticker, then scaling to all) was very effective with AI guidance
- Debugging was significantly faster. Claude could read error messages and suggest fixes immediately
- Learning complex concepts (e.g. why Adjusted Close matters, how min-max normalization affects model input) through conversation was more intuitive than reading documentation

## What Did Not Work Well
- Occasionally, generated code had subtle issues (e.g. computing features on normalized data instead of raw data, causing infinity values)
- CSS tricks for Streamlit sidebar customization were unreliable and had to be reverted
- We had to be careful not to accept changes blindly. Testing each change locally before committing was essential

## Reflection

Now, we understand what you meant by a "10x Engineer". This project has allowed us to leverage LLM tools through coding in such a way that we never have before. We now understand the true meaning of this concept, since it has helped us a lot in ideation, structuring, debugging, and more importantly, LEARNING! LLMs can explain such complex concepts in such a simple way sometimes that we can become "10x Learners" as well!
