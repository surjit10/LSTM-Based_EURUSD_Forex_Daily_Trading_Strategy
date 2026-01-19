
# LSTM-Based EUR/USD Trading Strategy

## Overview

This project implements an **LSTM (Long Short-Term Memory) neural network** to forecast short-term EUR/USD forex price movements and build a rule-based automated trading strategy driven by model predictions.

The system covers a complete quantitative trading pipeline:
- Historical data processing  
- Technical indicator feature engineering  
- Time-series deep learning forecasting  
- Signal generation  
- Strategy backtesting with risk management  

The objective is to demonstrate how machine learning can be applied to **financial time-series forecasting and trading decision optimization**.

---

## Project Objectives

- Predict future EUR/USD price movements using LSTM networks  
- Engineer technical indicators to represent market conditions  
- Generate trading signals from model forecasts  
- Backtest multiple trading strategies with realistic constraints  
- Evaluate both prediction accuracy and trading performance  

---

## Dataset

**Instrument:** EUR/USD Forex  
**Data Fields:** Open, High, Low, Close, Volume  
**Time Period:** Multi-year historical data (hourly resolution)

Raw data file:
```

xlsx_eurusd-forex-data.xlsx

```

---

## Feature Engineering

The project constructs 29+ technical indicators, including:

- Moving Averages (SMA 50 / 100 / 200)  
- MACD and Signal Line  
- Bollinger Bands  
- RSI (14)  
- Stochastic Oscillator (%K, %D)  
- ATR and ADX  
- Returns and Log Returns  
- Rolling Volatility  
- Candlestick body and shadow features  

These features capture **trend, momentum, and volatility** characteristics of the market.

---

## Target Variable

The model predicts the **next-period average price**:

```

Target = (Next Open + Next Close) / 2

```

Trading signals are derived using a small threshold:

- **Buy:** Predicted price > Current Close × (1 + threshold)  
- **Sell:** Predicted price < Current Close × (1 − threshold)  
- **Hold:** Otherwise  

---

## Data Preprocessing

- Train / Validation / Test split: 80 / 10 / 10  
- No random shuffling (time-series order preserved)  
- Min-Max normalization fitted on training data only  
- Sliding window sequences:
  - Lookback window: 30 timesteps  
  - Prediction horizon: 30 timesteps  

---

## Model Architecture

```

LSTM (64 units, return_sequences=True)
Dropout (0.2)

LSTM (32 units, return_sequences=True)
Dropout (0.2)

LSTM (16 units)

Dense (16) + LeakyReLU
Dense (8)  + LeakyReLU
Dense (1)  → Price Forecast

```

**Loss Function:** Mean Squared Error  
**Optimizer:** Adam with learning rate decay  

---

## Model Evaluation

- Training and validation loss monitoring  
- Test-set prediction vs actual visualization  
- Mean Absolute Error (MAE)  

---

## Trading Strategies & Backtesting

Backtesting is implemented using `backtesting.py` with:

- Initial capital: $10,000  
- Commission: 0.01%  
- Stop-loss: 2%  

### Strategies Tested

1. Long-only strategy  
2. Long-short strategy  
3. Full position rebalancing strategy  

### Performance Metrics

- Total Return  
- Sharpe Ratio  
- Maximum Drawdown  
- Win Rate  
- Trade Count  

---

## Results Summary

The LSTM model captures short-term temporal patterns in EUR/USD price data.  
Backtesting demonstrates that model-driven signals produce structured trading behavior under controlled risk constraints.

*This project is designed for research and educational purposes only.*

---

## Technical Stack

- Python  
- pandas, numpy  
- TensorFlow / Keras  
- scikit-learn  
- matplotlib, plotly  
- backtesting.py  

---



