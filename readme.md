--

```markdown
# LSTM-Based EUR/USD Trading Strategy

## Overview

This project implements an **LSTM (Long Short-Term Memory) neural network** to forecast short-term EUR/USD forex price movements and to build a rule-based automated trading strategy driven by model predictions.  

The system covers the full quantitative trading pipeline:
- Historical data processing
- Technical indicator feature engineering
- Time-series deep learning forecasting
- Signal generation
- Strategy backtesting with risk management

The goal is to demonstrate how machine learning can be applied to **financial time-series forecasting and trading decision optimization**.

---

## Project Objectives

- Predict future EUR/USD price movements using LSTM networks  
- Engineer technical indicators to enrich market state representation  
- Generate trading signals from model forecasts  
- Backtest multiple trading strategies with realistic constraints  
- Evaluate both prediction accuracy and trading performance  

---

## Dataset

**Instrument:** EUR/USD Forex  
**Data Fields:** Open, High, Low, Close, Volume  
**Time Period:** Multi-year historical data (hourly resolution)  

Raw data is loaded from:
```

xlsx_eurusd-forex-data.xlsx

```

---

## Feature Engineering

Over 29 technical indicators are constructed, including:

- Moving Averages (SMA 50 / 100 / 200)
- MACD and Signal Line
- Bollinger Bands
- RSI (14)
- Stochastic Oscillator (%K, %D)
- ATR and ADX
- Returns and Log Returns
- Rolling Volatility
- Candlestick body and shadow features

These features provide information on **trend, momentum, and volatility**.

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

- Train / Validation / Test split (80 / 10 / 10)
- No random shuffling (time-series integrity preserved)
- Min-Max normalization fitted only on training data
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

1. **Long-only strategy**
2. **Long-short strategy**
3. **Full position rebalancing strategy**

### Performance Metrics

- Total Return  
- Sharpe Ratio  
- Maximum Drawdown  
- Win Rate  
- Trade Count  

---

## Sample Result

The LSTM model successfully captures short-term temporal patterns in EUR/USD price data.  
Backtesting demonstrates that model-driven signals can produce structured trading behavior under controlled risk constraints.

*(Note: This project is for research and educational purposes. Not intended for live trading.)*

---

## Technical Stack

- Python  
- pandas, numpy  
- TensorFlow / Keras  
- scikit-learn  
- matplotlib, plotly  
- backtesting.py  

---

## File Structure

```

project/
│
├── xlsx_eurusd-forex-data.xlsx
├── trading_strategy_using_lstm.ipynb
├── train_data_normalized.xlsx
├── val_data_normalized.xlsx
├── test_data_normalized.xlsx
└── README.md

```

---

## Future Improvements

- Walk-forward retraining for regime adaptation  
- Hyperparameter optimization  
- Transformer-based sequence models  
- Probabilistic forecasting for risk-aware position sizing  
- Feature importance analysis (SHAP)  
- Multi-asset portfolio extension  

---

## Disclaimer

This project is for **educational and research purposes only**.  
It does **not** constitute financial advice or a production trading system.

---

## Author

**surjit**

---

```

---
