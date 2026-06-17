![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Modeling-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![Pytest](https://img.shields.io/badge/Pytest-Tested-brightgreen)

# Machine Learning Alpha Model

A computer science practice project that uses machine learning to predict stock returns from historical market data. The project includes feature engineering, Random Forest, XGBoost, backtesting, visualizations, and an interactive Streamlit dashboard.

## Project Goal

The goal of this project is to explore how machine learning can be applied to financial time series data while avoiding common mistakes such as overfitting and look-ahead bias.

This is not financial advice or a real trading system. It is a learning project focused on machine learning, feature engineering, and model evaluation.

## Features

- Download historical stock data with `yfinance`
- Create market-based features:
  - Daily returns
  - Moving averages
  - Rolling volatility
  - Volume changes
  - Lagged returns
- Train machine learning models:
  - Random Forest
  - XGBoost
- Evaluate model performance:
  - Mean squared error
  - Mean absolute error
  - R-squared
- Run a simple backtest:
  - Strategy return
  - Buy-and-hold comparison
  - Sharpe ratio
- Interactive Streamlit dashboard
- Unit tests with Pytest

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- yfinance
- Matplotlib
- Streamlit
- Pytest

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

## How to Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the main script:

```powershell
python main.py
```

Run the dashboard:

```powershell
streamlit run app.py
```

Run tests:

```powershell
python -m pytest
```

## What I Learned

This project helped me practice:

- Machine learning with real-world data
- Time series train-test splitting
- Feature engineering
- Model evaluation
- Avoiding future data leakage
- Basic backtesting
- Writing modular Python code
- Building interactive dashboards

## Disclaimer

This project is for educational purposes only. It does not provide financial advice or guarantee trading performance.