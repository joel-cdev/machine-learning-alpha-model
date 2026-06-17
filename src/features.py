import pandas as pd
import numpy as np


def create_features(data):
    """
    Create machine learning features from stock price data.

    Target:
        Predict next day's return.
    """
    df = data.copy()

    df["Return"] = df["Close"].pct_change()

    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    df["Volatility_5"] = df["Return"].rolling(window=5).std()
    df["Volatility_20"] = df["Return"].rolling(window=20).std()

    df["Volume_Change"] = df["Volume"].pct_change()

    df["Lag_Return_1"] = df["Return"].shift(1)
    df["Lag_Return_2"] = df["Return"].shift(2)
    df["Lag_Return_3"] = df["Return"].shift(3)

    df["Target"] = df["Return"].shift(-1)

    df = df.dropna()

    return df