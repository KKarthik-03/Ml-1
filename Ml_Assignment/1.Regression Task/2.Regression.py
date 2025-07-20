import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
import ta

# Loading the dataset
df = pd.read_csv(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\1.Regression Task\woo-BTC USDT.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df['year'] = df.index.year

# Feature Engineering
df['returns'] = df['close'].pct_change().round(2)
df['target'] = df['close'].shift(-1)
df['sma'] = df['close'].shift(1).rolling(window=7).mean().round(2)
df['ema'] = df['close'].shift(1).ewm(span=7, adjust=False).mean().round(2)
bb_indicator = ta.volatility.BollingerBands(close=df['close'].shift(1), window=7)
df['bbh'] = bb_indicator.bollinger_hband().round(2)
df['bbm'] = bb_indicator.bollinger_mavg().round(2)
df['bbl'] = bb_indicator.bollinger_lband().round(2)
df['volatility'] = df['close'].shift(1).rolling(window=7).std().round(2)
df['rsi'] = ta.momentum.RSIIndicator(close=df['close'].shift(1), window=14).rsi().round(2)
df['close_mom_7'] = df['close'].shift(1) / df['close'].shift(8) - 1

# Drop NA values
df.dropna(inplace=True)

# Define PretrainedModel class for loading pipeline
class PretrainedModel(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.model.predict(X)

# Loading the saved scaler and pipeline
pipeline = joblib.load(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\1.Regression Task\Reg_model_pipeline.pkl')
scaler = joblib.load(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\1.Regression Task\standard_scaler.pkl')
y_scaler = joblib.load(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\1.Regression Task\y_standard_scaler.pkl')

# Feature columns
features = ['open', 'high', 'low', 'volume', 'returns', 'sma', 'ema','bbh', 'bbm', 'bbl', 'volatility', 'rsi', 'close_mom_7']
X = df[features]
y = df['target']

# Make predictions and inverse scale
y_pred_scaled = pipeline.predict(X)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
df['Predicted_Close'] = y_pred

# Split data by year
train_df = df[df['year'] <= 2022]
test_df = df[df['year'] == 2023]
forecast_df = df[df['year'] >= 2024]

# PLOT 1: Forecast Only (2024–2025)

plt.figure(figsize=(14, 5))
plt.plot(forecast_df.index, forecast_df['Predicted_Close'], label='Forecast (2024–2025)', color='blue')
plt.title('Bitcoin Price Forecast (2024–2025)')
plt.xlabel("Date")
plt.ylabel("Predicted Close Price")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# PLOT 2: Final Combined Visualization (2018–2025)

plt.figure(figsize=(16, 6))
# Train: 2018–2022 actual
plt.plot(train_df.index, train_df['target'], label='Train Actual (2018–2022)', color='black')
# Test: 2023 actual
plt.plot(test_df.index, test_df['target'], label='Test Actual (2023)', color='red')
# Test: 2023 predicted
plt.plot(test_df.index, test_df['Predicted_Close'], label='Test Predicted (2023)', color='green')
# Forecast: 2024–2025 predicted
plt.plot(forecast_df.index, forecast_df['Predicted_Close'], label='Forecast Predicted (2024–2025)', color='blue')

plt.title('Bitcoin Price Forecast (2018–2025) – Actual vs Predicted')
plt.xlabel("Date")
plt.ylabel("Close Price (USDT)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper left", fontsize=11)
plt.tight_layout()
plt.show()