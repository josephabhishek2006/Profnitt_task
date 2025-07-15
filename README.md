# Profnitt_task
# 📈 Predicting Stock Volatility using Machine Learning

This project aims to **predict the 5-day future volatility** of Apple's stock ($AAPL) using historical price and volume data, enriched with technical indicators and time-aware machine learning techniques.

---

## 📊 Problem Statement

The goal is to build a model that can forecast the **expected average absolute return (volatility)** over the next 5 trading days, given past trends in price, volume, and indicators. This can be useful in:

- Risk management
- Options pricing
- Portfolio allocation
- Identifying high-volatility windows

---

## 🛠️ Approach

### 🔄 Data Source

- Fetched historical stock data for **AAPL** from **Yahoo Finance** using `yfinance`
- Time range: `2020-01-01` to `2024-12-31`

### 📐 Feature Engineering

Created the following features:

- **Technical indicators**:
  - SMA (20-day)
  - EMA (14-day)
  - RSI (14-day)
  - MACD
  - Bollinger Bands (Upper, Middle, Lower)

- **Custom features**:
  - Volume % change
  - 1-day volatility = `|Daily Return|`
  - Lagged values of 5-day future volatility (lags 1, 2, 3)

- **Target**:
  - 5-day future volatility: Rolling average of 1-day volatility, shifted by 5 days

### 🤖 Model

- **Algorithm**: Random Forest Regressor (`sklearn`)
- **Pipeline**: Scaling + RF model
- **CV Strategy**: 5-fold TimeSeriesSplit
- **Hyperparameter Tuning**: `GridSearchCV` over:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_leaf`: [1, 2, 4]

---

## ✅ Results

- The model explains ~77.8% of the variance in future 5-day volatility on test data.
- Low MSE indicates precise magnitude predictions.

---

## 📊 Exploratory Data Analysis (EDA)

- **Correlation matrix** revealed moderate correlation between:
  - MACD, RSI, and 5-day future volatility
  - Volume changes and near-term volatility

- **Scatter matrix** showed some nonlinear patterns, justifying the use of tree-based models.

---

## 📦 Files

- `volatility_model.pkl`: Trained model saved via `pickle`
- `volatility_forecast.ipynb`: Jupyter notebook with full pipeline
- `requirements.txt`: List of Python libraries used

---

## 💡 Future Work

- Add more stocks (multi-asset modeling)
- Use more advanced models (XGBoost, LightGBM, or hybrid with GARCH)
- Regime-based models (bull/bear)
- Deployment with Flask/Dash

---

## 🧠 Insights

- **Lagged volatility** was one of the strongest predictors—volatility clusters!
- **MACD and RSI** helped capture momentum and overbought/oversold conditions.
- Bollinger Band width helped quantify price spread and mean reversion behavior.

---

## 📚 Requirements

Install dependencies:

```bash
pip install -r requirements.txt


