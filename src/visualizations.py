import matplotlib.pyplot as plt
import pandas as pd


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


def get_feature_importance(model, feature_columns):
    importance = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    })

    return df.sort_values(by="Importance", ascending=False)


def plot_feature_importance(importance_df, title="Feature Importance"):
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()