![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-ff6f00)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![Pytest](https://img.shields.io/badge/Pytest-Tested-brightgreen)

# Machine Learning Alpha Model

A computer science practice project that uses machine learning to predict stock returns from historical market data. The project implements Random Forest, XGBoost, and LSTM neural networks, along with feature engineering, backtesting, explainability, and an interactive Streamlit dashboard.

---

## Project Goal

The goal of this project is to explore how machine learning can be applied to financial time series data while avoiding common pitfalls such as overfitting and look-ahead bias.

This project is intended for educational purposes and is not a real trading system or financial advice.

---

## Features

### Data Collection

- Historical stock prices using `yfinance`

### Feature Engineering

- Daily returns
- 5-day moving average
- 20-day moving average
- 5-day volatility
- 20-day volatility
- Volume changes
- Lagged returns
- Prediction target (next-day return)

### Machine Learning Models

- Random Forest
- XGBoost
- LSTM Neural Network

### Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

### Backtesting

- Strategy cumulative returns
- Buy-and-hold benchmark
- Sharpe ratio

### Explainability

- Feature importance analysis
- Visualization of model importance

### Visualization

- Actual vs Predicted Returns
- Strategy vs Buy-and-Hold
- Feature Importance charts
- Interactive Streamlit dashboard with animations

### Testing

- Unit tests with Pytest

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow
- Matplotlib
- Streamlit
- Pytest
- yfinance

---

## Project Structure

```text
machine-learning-alpha-model/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── pytest.ini
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── backtest.py
│   └── visualizations.py
│
├── tests/
│   └── test_features.py
│
├── data/
└── notebooks/
```

---

## Architecture

```text
Historical Stock Data
           ↓
    Feature Engineering
           ↓
     Random Forest
           ↓
        XGBoost
           ↓
     LSTM Network
           ↓
      Model Metrics
           ↓
       Backtesting
           ↓
 Feature Importance
           ↓
 Streamlit Dashboard
```

---

## How to Run

### Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```

### Run Main Program

```powershell
python main.py
```

### Launch Dashboard

```powershell
streamlit run app.py
```

### Run Tests

```powershell
python -m pytest
```

---

## What I Learned

This project helped me practice:

- Machine learning with real-world financial data
- Feature engineering
- Time-series train/test splitting
- Avoiding look-ahead bias
- Backtesting trading strategies
- Model explainability
- Neural networks with LSTM
- Writing modular Python code
- Building interactive dashboards
- Software testing with Pytest

---

## Disclaimer

This project is for educational purposes only.

It is not financial advice and does not guarantee trading performance.

---

## Repository Description

> CS practice project using Random Forest, XGBoost, and LSTM models to predict stock returns with feature engineering, backtesting, and an animated Streamlit dashboard. 📊🐍