from src.data_loader import download_stock_data
from src.features import create_features


def main():
    ticker = "AAPL"

    data = download_stock_data(ticker=ticker, start="2018-01-01")
    features = create_features(data)

    print(f"Feature dataset for {ticker}")
    print(features.head())
    print()
    print("Columns:")
    print(features.columns.tolist())


if __name__ == "__main__":
    main()