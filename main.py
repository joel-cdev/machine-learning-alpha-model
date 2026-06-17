from src.data_loader import download_stock_data
from src.features import create_features
from src.models import (
    time_series_train_test_split,
    train_random_forest,
    train_xgboost,
    evaluate_model,
)
from src.backtest import create_backtest_results, calculate_sharpe_ratio
from src.visualizations import plot_backtest_results, plot_predictions


def main():
    ticker = "AAPL"

    data = download_stock_data(ticker=ticker, start="2018-01-01")
    df = create_features(data)

    X_train, X_test, y_train, y_test = time_series_train_test_split(df)

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    rf_predictions, rf_metrics = evaluate_model(rf_model, X_test, y_test)
    xgb_predictions, xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    rf_backtest = create_backtest_results(y_test, rf_predictions)
    xgb_backtest = create_backtest_results(y_test, xgb_predictions)

    print(f"Machine Learning Alpha Model for {ticker}")
    print("--------------------------------------")

    print("Random Forest Metrics:")
    print(rf_metrics)
    print("Random Forest Strategy Final Value:", round(rf_backtest["Strategy_Cumulative"].iloc[-1], 4))
    print("Random Forest Buy & Hold Final Value:", round(rf_backtest["Buy_Hold_Cumulative"].iloc[-1], 4))
    print("Random Forest Sharpe:", round(calculate_sharpe_ratio(rf_backtest["Strategy_Return"]), 4))

    print()

    print("XGBoost Metrics:")
    print(xgb_metrics)
    print("XGBoost Strategy Final Value:", round(xgb_backtest["Strategy_Cumulative"].iloc[-1], 4))
    print("XGBoost Buy & Hold Final Value:", round(xgb_backtest["Buy_Hold_Cumulative"].iloc[-1], 4))
    print("XGBoost Sharpe:", round(calculate_sharpe_ratio(xgb_backtest["Strategy_Return"]), 4))

    plot_backtest_results(rf_backtest, title="Random Forest Strategy vs Buy and Hold")
    plot_predictions(y_test, rf_predictions, title="Random Forest Actual vs Predicted Returns")

    plot_backtest_results(xgb_backtest, title="XGBoost Strategy vs Buy and Hold")
    plot_predictions(y_test, xgb_predictions, title="XGBoost Actual vs Predicted Returns")


if __name__ == "__main__":
    main()