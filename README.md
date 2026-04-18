
# Zero-Shot Multivariate Forecasting with Chronos-v2 (Bolt)

## Overview
This project implements a high-fidelity financial forecasting pipeline using **Amazon’s Chronos-v2 (Bolt)** foundation model. Unlike traditional LSTM-based architectures that require extensive training and hyperparameter tuning, Chronos-v2 leverages a transformer-based architecture pretrained on billions of time-series data points to perform zero-shot inference.

The project demonstrates a **multivariate approach**, utilizing both historical price action and trading volume to generate a probabilistic 30-day forecast.

## Key Differences: LSTM vs. Chronos-v2
| Feature | Stacked LSTM (Previous) | Chronos-v2 Bolt (Current) |
| :--- | :--- | :--- |
| **Learning Mode** | Supervised (Trained on SPY data) | Zero-Shot (Pretrained Foundation Model) |
| **Input Data** | Univariate (Price only) | Multivariate (Price + Volume) |
| **Architecture** | Recurrent (RNN) | Transformer (Patch-based) |
| **Uncertainty** | Single-line (Deterministic) | Probabilistic (Quantile-based) |
| **Data Scaling** | Required (MinMaxScaler) | Internal (Quantization-based) |

## Technical Implementation
### Model Architecture
The project utilizes the `chronos-bolt-small` variant, which is optimized for rapid inference. The model treats time-series data as tokens, utilizing a cross-series attention mechanism to identify dependencies between co-evolving variables (Price and Volume).

### Multivariate Context
Instead of a standard 1D array, a **multivariate tensor** of shape `(2, 512)` is fed into the model. This provides the transformer with:
1. **Closing Price:** The primary target variable.
2. **Trading Volume:** A leading indicator used as a covariate to refine price predictions.

## Results and Visualization
The model generates predictions as **statistical quantiles** (10th, 50th, and 90th percentiles), allowing for a nuanced understanding of market risk.

### Multivariate Dashboard
![Multivariate Forecast Dashboard](link-to-your-dashboard-screenshot.png)
*Note: The top graph depicts price targets, while the bottom graph depicts predicted market activity (volume), showing how the model expects these two variables to interact.*

### Analysis of the "Chronos Advantage"
* **Retention of Volatility:** Unlike the LSTM, which produced a smoothed moving average, Chronos preserves the "jagged" nature of market data, reflecting more realistic future scenarios.
* **Confidence Bounds:** The 80% confidence interval visually depicts the decay of mathematical certainty over a 30-day horizon, providing a realistic risk assessment that deterministic models lack.

## Technical Stack
* **Language:** Python 3.12
* **Framework:** PyTorch & Hugging Face Transformers
* **Model:** `amazon/chronos-bolt-small`
* **Infrastructure:** Google Colab (T4 GPU Runtime)

## Usage
1. Ensure the `SPY.csv` dataset is available in the root directory.
2. Install dependencies via `pip install git+https://github.com/amazon-science/chronos-forecasting.git`.
3. Execute the notebook to generate zero-shot multivariate predictions without the need for model retraining.
