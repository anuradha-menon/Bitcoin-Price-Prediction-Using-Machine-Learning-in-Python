import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib  # for saving scaler

# Parameters
stock = "BTC-USD"
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
base_days = 100

print(f"📥 Downloading data for {stock}...")
data = yf.download(stock, start=start, end=end, auto_adjust=True)

# Only use closing prices
close_data = data[["Close"]]

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Train-Test Split
training_len = int(len(scaled_data) * 0.9)
train_data = scaled_data[:training_len]

# Create sequences
x_train, y_train = [], []
for i in range(base_days, len(train_data)):
    x_train.append(train_data[i - base_days:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape for LSTM [samples, timesteps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print("✅ Data prepared for training.")

# Build Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(64, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Train
print("🚀 Training model...")
model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1)

# Save model
model.save("model.keras")
print("💾 Model saved as model.keras")

# Save scaler
joblib.dump(scaler, "scaler.pkl")
print("💾 Scaler saved as scaler.pkl")
