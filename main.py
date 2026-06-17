from src.data_loader import download_stock_data
from src.features import create_features
from src.models import (
    FEATURE_COLUMNS,
    time_series_train_test_split,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    create_lstm_sequences,
    train_lstm_model,
    evaluate_lstm_model,
)
from src.backtest import create_backtest_results, calculate_sharpe_ratio


def print_results(model_name, metrics, backtest):
    print(f"{model_name} Metrics:")
    print(metrics)
    print(f"{model_name} Strategy Final Value:", round(backtest["Strategy_Cumulative"].iloc[-1], 4))
    print(f"{model_name} Buy & Hold Final Value:", round(backtest["Buy_Hold_Cumulative"].iloc[-1], 4))
    print(f"{model_name} Sharpe:", round(calculate_sharpe_ratio(backtest["Strategy_Return"]), 4))
    print()


def main():
    ticker = "AAPL"

    data = download_stock_data(ticker=ticker, start="2018-01-01")
    df = create_features(data)

    X_train, X_test, y_train, y_test = time_series_train_test_split(df)

    rf_model = train_random_forest(X_train, y_train)
    rf_predictions, rf_metrics = evaluate_model(rf_model, X_test, y_test)
    rf_backtest = create_backtest_results(y_test, rf_predictions)

    xgb_model = train_xgboost(X_train, y_train)
    xgb_predictions, xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    xgb_backtest = create_backtest_results(y_test, xgb_predictions)

    X_lstm, y_lstm, _ = create_lstm_sequences(df, FEATURE_COLUMNS, sequence_length=20)

    split_index = int(len(X_lstm) * 0.8)
    X_lstm_train = X_lstm[:split_index]
    X_lstm_test = X_lstm[split_index:]
    y_lstm_train = y_lstm[:split_index]
    y_lstm_test = y_lstm[split_index:]

    lstm_model = train_lstm_model(X_lstm_train, y_lstm_train, epochs=5, batch_size=32)
    lstm_predictions, lstm_metrics = evaluate_lstm_model(lstm_model, X_lstm_test, y_lstm_test)
    lstm_backtest = create_backtest_results(
        y_test=df["Target"].iloc[-len(y_lstm_test):],
        predictions=lstm_predictions
    )

    print(f"Machine Learning Alpha Model for {ticker}")
    print("--------------------------------------")
    print_results("Random Forest", rf_metrics, rf_backtest)
    print_results("XGBoost", xgb_metrics, xgb_backtest)
    print_results("LSTM", lstm_metrics, lstm_backtest)


if __name__ == "__main__":
    main()