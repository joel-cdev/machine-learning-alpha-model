import streamlit as st
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #172554);
        background-size: 400% 400%;
        animation: gradientMove 12s ease infinite;
        color: white;
    }

    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .hero {
        padding: 2rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        backdrop-filter: blur(14px);
        animation: fadeSlide 1s ease-out;
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56,189,248,0.45), transparent 70%);
        top: -80px;
        right: -60px;
        animation: floatOrb 6s ease-in-out infinite;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #e0f2fe;
    }

    .hero p {
        font-size: 1.15rem;
        color: #cbd5e1;
        max-width: 850px;
    }

    @keyframes fadeSlide {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }

    @keyframes floatOrb {
        0%, 100% {transform: translateY(0px);}
        50% {transform: translateY(25px);}
    }

    .metric-card {
        padding: 1.4rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        animation: cardPop 0.8s ease both;
    }

    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 18px 45px rgba(56,189,248,0.25);
        border-color: rgba(56,189,248,0.55);
    }

    .metric-card h3 {
        color: #bae6fd;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .metric-card h2 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }

    @keyframes cardPop {
        from {opacity: 0; transform: translateY(18px) scale(0.98);}
        to {opacity: 1; transform: translateY(0) scale(1);}
    }

    .section-title {
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #e0f2fe;
        font-size: 1.7rem;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        padding: 0.8rem 1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        box-shadow: 0 8px 25px rgba(37,99,235,0.35);
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(6,182,212,0.45);
    }

    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.88);
        border-right: 1px solid rgba(255,255,255,0.12);
    }

    .stDataFrame {
        animation: fadeSlide 0.9s ease;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>Machine Learning Alpha Model</h1>
        <p>
            Predict stock returns using engineered market features and compare
            Random Forest vs XGBoost with a simple backtested trading strategy.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Model Inputs")

ticker = st.sidebar.text_input("Stock ticker", value="AAPL")
start_date = st.sidebar.text_input("Start date", value="2018-01-01")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost"])

run_model = st.sidebar.button("Run Model")

if run_model:
    with st.spinner("Training model and running backtest..."):
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

    st.markdown(
        f"<div class='section-title'>Results for {ticker.upper()} using {model_choice}</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"<div class='metric-card'><h3>Mean Squared Error</h3><h2>{metrics['mse']:.6f}</h2></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='metric-card'><h3>Mean Absolute Error</h3><h2>{metrics['mae']:.6f}</h2></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"<div class='metric-card'><h3>Strategy Final Value</h3><h2>{strategy_final:.2f}x</h2></div>",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"<div class='metric-card'><h3>Sharpe Ratio</h3><h2>{sharpe:.2f}</h2></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='section-title'>Strategy vs Buy and Hold</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(backtest["Strategy_Cumulative"], label="Strategy")
    ax.plot(backtest["Buy_Hold_Cumulative"], label="Buy and Hold")
    ax.set_title(f"{model_choice} Backtest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.markdown("<div class='section-title'>Actual vs Predicted Returns</div>", unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.scatter(y_test, predictions, alpha=0.5)
    ax2.set_title("Actual vs Predicted Returns")
    ax2.set_xlabel("Actual Returns")
    ax2.set_ylabel("Predicted Returns")
    ax2.grid(True)

    st.pyplot(fig2)

    st.markdown("<div class='section-title'>Recent Feature Data</div>", unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True)

else:
    st.info("Choose inputs in the sidebar and click Run Model.")