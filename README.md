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

