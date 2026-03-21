import os
import time
import requests
import polars as pl
from dotenv import load_dotenv


class PySimFin:
    """Python wrapper for the SimFin API."""

    BASE_URL = "https://backend.simfin.com/api/v3"
    MIN_REQUEST_INTERVAL = 0.5  # 2 requests per second max

    def __init__(self, api_key: str = None):
        """
        Initialize the PySimFin client.

        Args:
            api_key: SimFin API key. If not provided, loads from .env file.
        """
        if api_key is None:
            # Try Streamlit secrets first (for cloud deployment), then .env
            try:
                import streamlit as st
                api_key = st.secrets.get("SIMFIN_API_KEY")
            except Exception:
                pass

            if not api_key:
                load_dotenv()
                api_key = os.getenv("SIMFIN_API_KEY")

        if not api_key:
            raise ValueError("No API key provided. Set SIMFIN_API_KEY in .env or Streamlit secrets.")

        self.headers = {"Authorization": f"api-key {api_key}"}
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting (max 2 requests per second)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict) -> dict:
        """
        Make a GET request to the SimFin API.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            JSON response as a dictionary.

        Raises:
            requests.exceptions.HTTPError: If the request fails.
        """
        self._rate_limit()
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_share_prices(self, ticker: str, start: str, end: str) -> pl.DataFrame:
        """
        Get daily share prices for a company.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.

        Returns:
            Polars DataFrame with share price data.
        """
        data = self._request("companies/prices/compact", {
            "ticker": ticker,
            "start": start,
            "end": end,
        })

        columns = data[0]["columns"]
        rows = data[0]["data"]
        return pl.DataFrame(rows, schema=columns, orient="row")

    def get_financial_statement(self, ticker: str, start: str, end: str,
                                statements: str = "PL", period: str = "FY") -> pl.DataFrame:
        """
        Get financial statements for a company.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            statements: Statement type: PL (income), BS (balance sheet), CF (cash flow).
            period: Period type: FY, Q1, Q2, Q3, Q4.

        Returns:
            Polars DataFrame with financial statement data.
        """
        data = self._request("companies/statements/compact", {
            "ticker": ticker,
            "start": start,
            "end": end,
            "statements": statements,
            "period": period,
        })

        columns = data[0]["columns"]
        rows = data[0]["data"]
        return pl.DataFrame(rows, schema=columns, orient="row")
