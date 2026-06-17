import numpy as np
import pandas as pd


def create_backtest_results(y_test, predictions):
    """
    Create a simple long/cash trading strategy.

    If predicted return is positive, hold the stock.
    If predicted return is negative, stay in cash.
    """
    results = pd.DataFrame({
        "Actual_Return": y_test.values,
        "Predicted_Return": predictions
    }, index=y_test.index)

    results["Signal"] = np.where(results["Predicted_Return"] > 0, 1, 0)
    results["Strategy_Return"] = results["Signal"] * results["Actual_Return"]

    results["Buy_Hold_Cumulative"] = (1 + results["Actual_Return"]).cumprod()
    results["Strategy_Cumulative"] = (1 + results["Strategy_Return"]).cumprod()

    return results


def calculate_sharpe_ratio(returns, periods_per_year=252):
    if returns.std() == 0:
        return 0

    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)