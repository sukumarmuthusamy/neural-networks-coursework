# Stock Price Prediction using RNN – Apple Stock Price Forecasting

This assignment implements a Recurrent Neural Network (RNN) to predict future stock prices based on historical data. The implementation uses Apple's historical stock price data and demonstrates how RNNs can learn temporal dependencies in sequential time-series data to forecast future values.

## Files Included

- **apple_stock_prediction_rnn.ipynb** — Complete Jupyter Notebook with code, outputs, visualizations, and detailed analysis for the stock price prediction task.

## Assignment Overview

This assignment covers fundamental concepts in time-series forecasting through recurrent neural networks:
- **Dataset:** Historical stock price data of Apple (AAPL) downloaded using yfinance library
- **Task:** Build an RNN model to predict future stock prices based on historical closing prices
- **Framework:** Python with Pandas, NumPy, TensorFlow/Keras, yfinance, and Matplotlib
- **Focus:** Understanding sequential data modeling, temporal dependencies, and RNN architecture for time-series prediction

## Tasks Implemented

### Task 1: Download Historical Stock Price Data of Apple
- Download historical stock price data of Apple (AAPL) using the yfinance library ([resource on how to download dataset](https://pypi.org/project/yfinance/))
- Dataset contains columns: Date, Open, High, Low, Close, and Volume
- Focus on the 'Close' price as the target variable for prediction
- Parse Date column to datetime format and sort chronologically
- Verify data integrity by checking for missing or duplicate values
- Display dataset shape, date range, and summary statistics

### Task 2: Plot the 'Close' Price with Respect to Time
- Visualize the historical closing price trend over time using line plot
- Display the temporal pattern and overall price movement
- Add proper axis labels, title, and formatting for clear interpretation
- Identify long-term trends, volatility patterns, and price movements

### Task 3: Define RNN Design and Fit the Model
The RNN architecture consists of a Simple RNN layer followed by a dense output layer:

#### **Model Architecture:**
- **Simple RNN Layer:** 64 units processing sequential windows of 60 days
- **Dense Output Layer:** 1 neuron for single-step price prediction

#### **Data Preparation:**
- Normalize closing prices to range [0, 1] using MinMaxScaler
- Create sequential windows of 60 consecutive days as input features
- Target variable: next day's closing price
- Preserve temporal order throughout training (no shuffling)
- Train-test split while maintaining chronological sequence

#### **Training Configuration:**
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** Trained until convergence
- **Batch Size:** 32
- **Sequence Window:** 60 days of historical prices

### Task 4: Analyze the Model Using Metrics (MAE, MSE, R²)
- **Mean Absolute Error (MAE):** ~$3.67 USD — average absolute deviation of predictions from actual prices
- **Mean Squared Error (MSE):** ~24.52 — penalizes larger prediction errors more heavily
- **R² Score:** 0.953 — model explains approximately 95.3% of variance in stock price movements

These metrics indicate strong predictive performance with the model closely tracking actual price trends while maintaining low prediction errors.

### Task 5: Visualize the Predicted Values and Actual Stock Prices
- Plot predicted stock prices alongside actual historical prices
- Visualize model performance on both training and test sets
- Display prediction accuracy through overlapping time-series curves
- Highlight areas where predictions align well and where deviations occur
- Use clear legends, labels, and formatting for comparison

## Model Analysis

The recurrent neural network used in this assignment consists of a single Simple RNN layer with 64 units followed by a dense output layer. This architecture processes the historical sequence of Apple closing prices and learns temporal dependencies across time, allowing the model to generate a prediction for the next day's closing value. The input data was scaled to the range [0, 1], and the model was trained on sequential windows of 60 days, ensuring that the temporal order of the data was preserved throughout the training process.

**Performance Evaluation:**

Evaluating the model on the test set, the RNN achieved a **Mean Absolute Error of approximately $3.67 USD** and a **Mean Squared Error of about 24.52**, with an **R² value of 0.953**. These results indicate that the model captured the general upward movement and fluctuations in Apple's stock price effectively, closely following the true values throughout the evaluation period. The predicted curve aligned well with the actual price trend, suggesting that the model learned meaningful temporal patterns rather than memorizing the data. However, small deviations were visible during rapid price swings, which is expected given the model complexity and the inherent volatility of stock market data.

**Proposed Improvements:**

To further improve this model, I would experiment with more advanced recurrent architectures such as **LSTM (Long Short-Term Memory)** or **GRU (Gated Recurrent Unit)** layers, which are better at retaining long-term dependencies and handling noisy time-series signals. Additionally, tuning the sequence window length, incorporating volume and other market indicators (such as trading volume, moving averages, or technical indicators), and applying dropout or regularization techniques could help reduce error and improve stability. Testing **attention-based time-series models** or **transformer variants** could also provide further performance gains for future extensions of this work.

**Key Insights:**
- The Simple RNN successfully captured the overall upward trend in Apple's stock price over the evaluation period
- The model demonstrated strong generalization with R² = 0.953, indicating excellent fit
- Temporal dependencies across 60-day windows proved effective for next-day price prediction
- Model performance was robust during stable market conditions but showed minor deviations during high volatility periods
- The architecture balances simplicity and effectiveness, making it a solid baseline for time-series forecasting

## How to View Results

Open the Jupyter Notebook file to see:
- All Python code with syntax highlighting and detailed comments
- Execution outputs for each task including data statistics and model metrics
- Data visualizations: historical price trends, prediction vs. actual plots, and performance analysis
- Complete model training history and evaluation metrics (MAE, MSE, R²)
- Detailed model interpretation and analysis

## Requirements

- **Python 3.x**
- **Pandas** — For data manipulation and time-series handling
- **NumPy** — For numerical computations and array operations
- **TensorFlow / Keras** — For building and training the RNN model
- **yfinance** — For downloading historical stock price data
- **Matplotlib** — For data visualization and plotting
- **Scikit-Learn** — For preprocessing (MinMaxScaler) and evaluation metrics
- **Google Colab / Jupyter Notebook** — Execution environment

## Notes

- **Data Source:** Historical Apple stock data downloaded using yfinance library from Yahoo Finance
- **Date Range:** Data spans from 2015-01-02 to 2025-10-31 (approximately 10 years of historical data)
- **Sequence Length:** 60-day windows chosen to capture approximately three months of trading patterns
- **Data Preprocessing:** MinMaxScaler normalization applied to stabilize training and improve convergence
- **Temporal Order:** Critical importance of preserving chronological order in time-series data; no shuffling applied
- **Train-Test Split:** Chronological split to simulate real-world forecasting scenario (training on past, predicting future)
- **Output Format:** Notebook includes all code outputs, visualizations, and analysis as required for grading
- **Model Architecture:** Simple RNN chosen as baseline; serves as foundation for comparison with more advanced architectures (LSTM/GRU)
- **Evaluation Strategy:** Metrics computed on hold-out test set to assess true generalization performance
- **GPU Acceleration:** Training performed on Google Colab with T4 GPU for faster computation
- This notebook was developed in Google Colab and includes all required outputs and analysis as per assignment specifications.
