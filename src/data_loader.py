import yfinance as yf
import pandas as pd


def download_stock_data(ticker="AAPL", start="2018-01-01", end=None):
    """
    Download historical stock data from Yahoo Finance.

    Parameters:
        ticker (str): Stock ticker symbol.
        start (str): Start date in YYYY-MM-DD format.
        end (str): End date in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: Historical stock price data.
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    data = data.reset_index()

    return data