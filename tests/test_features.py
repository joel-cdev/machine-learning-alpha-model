import pandas as pd

from src.features import create_features


def test_create_features_adds_expected_columns():
    data = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=40),
        "Close": [100 + i for i in range(40)],
        "Volume": [1000000 + i * 1000 for i in range(40)]
    })

    df = create_features(data)

    expected_columns = [
        "Return",
        "MA_5",
        "MA_20",
        "Volatility_5",
        "Volatility_20",
        "Volume_Change",
        "Lag_Return_1",
        "Lag_Return_2",
        "Lag_Return_3",
        "Target",
    ]

    for column in expected_columns:
        assert column in df.columns


def test_create_features_removes_missing_values():
    data = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=40),
        "Close": [100 + i for i in range(40)],
        "Volume": [1000000 + i * 1000 for i in range(40)]
    })

    df = create_features(data)

    assert df.isna().sum().sum() == 0