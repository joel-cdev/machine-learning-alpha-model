import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import download_stock_data
from src.features import create_features
from src.models import (
    time_series_train_test_split,
    train_random_forest,
    train_xgboost,
    evaluate_model,
)
from src.backtest import create_backtest_results, calculate_sharpe_ratio


st.set_page_config(
    page_title="Machine Learning Alpha Model",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617, #111827, #1e293b);
        color: white;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(12px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .subtitle {
        color: #cbd5e1;
        font-size: 19px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Machine Learning Alpha Model")
st.markdown(
    "<p class='subtitle'>Predict stock returns using Random Forest and XGBoost with engineered market features.</p>",
    unsafe_allow_html=True
)

st.sidebar.header("Model Inputs")

ticker = st.sidebar.text_input("Stock ticker", value="AAPL")
start_date = st.sidebar.text_input("Start date", value="2018-01-01")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost"])

if st.sidebar.button("Run Model"):
    data = download_stock_data(ticker=ticker, start=start_date)
    df = create_features(data)

    X_train, X_test, y_train, y_test = time_series_train_test_split(df)

    if model_choice == "Random Forest":
        model = train_random_forest(X_train, y_train)
    else:
        model = train_xgboost(X_train, y_train)

    predictions, metrics = evaluate_model(model, X_test, y_test)
    backtest = create_backtest_results(y_test, predictions)

    strategy_final = backtest["Strategy_Cumulative"].iloc[-1]
    buy_hold_final = backtest["Buy_Hold_Cumulative"].iloc[-1]
    sharpe = calculate_sharpe_ratio(backtest["Strategy_Return"])

    st.subheader(f"Results for {ticker.upper()} using {model_choice}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"<div class='metric-card'><h3>MSE</h3><h2>{metrics['mse']:.6f}</h2></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='metric-card'><h3>MAE</h3><h2>{metrics['mae']:.6f}</h2></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"<div class='metric-card'><h3>Strategy Final</h3><h2>{strategy_final:.2f}x</h2></div>",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"<div class='metric-card'><h3>Sharpe Ratio</h3><h2>{sharpe:.2f}</h2></div>",
            unsafe_allow_html=True
        )

    st.subheader("Backtest: Strategy vs Buy and Hold")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(backtest["Strategy_Cumulative"], label="Strategy")
    ax.plot(backtest["Buy_Hold_Cumulative"], label="Buy and Hold")
    ax.set_title(f"{model_choice} Strategy vs Buy and Hold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Actual vs Predicted Returns")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.scatter(y_test, predictions, alpha=0.5)
    ax2.set_title("Actual vs Predicted Returns")
    ax2.set_xlabel("Actual Returns")
    ax2.set_ylabel("Predicted Returns")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("Recent Feature Data")
    st.dataframe(df.tail(10))
else:
    st.info("Choose inputs in the sidebar and click Run Model.")