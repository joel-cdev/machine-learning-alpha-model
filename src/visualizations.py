import matplotlib.pyplot as plt


def plot_backtest_results(results, title="Strategy vs Buy and Hold"):
    plt.figure(figsize=(10, 6))
    plt.plot(results["Strategy_Cumulative"], label="Strategy")
    plt.plot(results["Buy_Hold_Cumulative"], label="Buy and Hold")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(y_test, predictions, title="Actual vs Predicted Returns"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.title(title)
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.grid(True)
    plt.show()