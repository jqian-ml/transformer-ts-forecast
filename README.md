# Transformer Stock Price Forecaster

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

Multi-stock price forecasting using Transformer architecture with self-attention mechanisms.

## Features

- 🎯 5-day ahead price predictions
- 🔄 Multi-stock training (learns shared patterns)
- 💡 Self-attention mechanism (captures long-range dependencies)
- 📊 Automatic data fetching via yfinance
- 📈 Training/validation visualization

## Quick Start

```bash
# Install dependencies
pip install torch numpy yfinance matplotlib scikit-learn

# Run demo
python app/main.py
```

Trains on AAPL, GOOGL, MSFT, TSLA, AMZN and generates 5-day forecasts.

## Architecture

```
60-day sequences → Linear Projection → Positional Encoding 
→ Transformer Encoder (4 layers, 8 heads) → Output → 5-day forecast
```

**Model**: 128d embeddings, 8 attention heads, 4 encoder layers, ~1.2M parameters

## Technical Details

- **Input**: OHLCV data (Open, High, Low, Close, Volume)
- **Normalization**: MinMaxScaler per stock
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler
- **Training**: 80/20 train/val split, gradient clipping

## License

MIT License

---

**Portfolio Project** - Demonstrates Transformer architecture, PyTorch, time series forecasting, and financial data processing.
