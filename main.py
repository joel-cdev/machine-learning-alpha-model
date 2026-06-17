from src.data_loader import download_stock_data
from src.features import create_features
from src.models import (
    time_series_train_test_split,
    train_random_forest,
    train_xgboost,
    evaluate_model,
)


def main():
    ticker = "AAPL"

    data = download_stock_data(ticker=ticker, start="2018-01-01")
    df = create_features(data)

    X_train, X_test, y_train, y_test = time_series_train_test_split(df)

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    rf_predictions, rf_metrics = evaluate_model(rf_model, X_test, y_test)
    xgb_predictions, xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    print(f"Machine Learning Alpha Model for {ticker}")
    print("--------------------------------------")

    print("Random Forest Metrics:")
    print(rf_metrics)

    print()

    print("XGBoost Metrics:")
    print(xgb_metrics)


if __name__ == "__main__":
    main()