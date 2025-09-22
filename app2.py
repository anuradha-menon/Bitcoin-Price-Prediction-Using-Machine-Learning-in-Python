from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
ad.user_cache_dir = lambda *args, **kwargs: CACHE_DIR
Path(CACHE_DIR).mkdir(exist_ok=True)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pytz
import requests

# Title
st.title("📈 Bitcoin OHLCV and Predictive Analytics Dashboard")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Ticker Symbol", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2014-12-17"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-09-14"))

# Timezone-localize dates (New York)
tz = pytz.timezone("America/New_York")
start_dt = tz.localize(pd.to_datetime(start_date))
end_dt = tz.localize(pd.to_datetime(end_date))

st.write(f"Fetching data for **{ticker}** from {start_date} to {end_date}...")

# Download with fallback
def fetch_yf(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty or len(df) < 10:
            raise ValueError("No data or too little data.")
        return df
    except Exception as e:
        st.warning(f"yfinance error: {e}")
        return pd.DataFrame()

def fetch_coingecko():
    st.info("Attempting fallback: CoinGecko API.")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "max"}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        df_price = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
        df_price['Date'] = pd.to_datetime(df_price['Date'], unit='ms')
        df = df_price.set_index('Date')
        # simple OHLCV stub (just Close for fallback)
        for col in ['Open', 'High', 'Low', 'Adj Close', 'Volume']:
            df[col] = df['Close']
        return df[(df.index >= start_date) & (df.index <= end_date)]
    else:
        st.error("CoinGecko fetch failed!")
        return pd.DataFrame()

ohlcv = fetch_yf(ticker, start_dt, end_dt)
if ohlcv.empty:
    ohlcv = fetch_coingecko()

if ohlcv.empty:
    st.error("No data fetched! Please check the ticker symbol or date range.")
else:
    # Flatten MultiIndex
    if isinstance(ohlcv.columns, pd.MultiIndex):
        ohlcv.columns = ohlcv.columns.get_level_values(0)
    # Ensure columns
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in ohlcv.columns and isinstance(ohlcv[col], pd.Series):
            ohlcv[col] = pd.to_numeric(ohlcv[col], errors='coerce')
    ohlcv = ohlcv.dropna(subset=['Close'])
    
    st.subheader("📊 Data Preview")
    st.dataframe(ohlcv.head())

    # OHLCV Charts
    st.subheader("📉 OHLCV Charts")
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    for idx, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
        if col in ohlcv:
            if col == 'Volume':
                axes[idx].bar(ohlcv.index, ohlcv[col], label=col, color="orange", alpha=0.6)
            else:
                axes[idx].plot(ohlcv.index, ohlcv[col], label=col)
            axes[idx].set_ylabel(col)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Predictive Modeling
    st.subheader("🤖 Model Predictions")
    try:
        model = load_model("model.keras")
        st.success("✅ Model loaded successfully!")
        closing_price = ohlcv[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_price)
        base_days = 60
        test_data = scaled_data[-(base_days+30):]
        x_test, y_test = [], []
        for i in range(base_days, len(test_data)):
            x_test.append(test_data[i-base_days:i, 0])
            y_test.append(test_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = model.predict(x_test)
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        plotting_data = pd.DataFrame(
            {"Original": inv_y_test.flatten(), "Prediction": inv_predictions.flatten()},
            index=closing_price.index[-len(inv_y_test):]
        )
        fig1, ax1 = plt.subplots(figsize=(15, 6))
        ax1.plot(plotting_data.index, plotting_data["Original"], label="Original", color="blue", linewidth=2)
        ax1.plot(plotting_data.index, plotting_data["Prediction"], label="Prediction", color="red", linewidth=2)
        ax1.set_title("Prediction vs Actual Close Price", fontsize=16)
        ax1.set_xlabel("Date", fontsize=14)
        ax1.set_ylabel("Close Price", fontsize=14)
        ax1.legend(); ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        # Future Predictions
        future_input = scaled_data[-base_days:].reshape(1, base_days, 1)
        future_predictions = []
        for _ in range(10):
            pred = model.predict(future_input)[0][0]
            future_predictions.append(pred)
            future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)
        future_predictions_arr = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        ax2.plot(range(1, 11), future_predictions_arr, marker="o", label="Prediction Future Prices", color="purple", linewidth=2)
        for i, val in enumerate(future_predictions_arr.flatten()):
            ax2.text(i+1, val, f"{val:.2f}", fontsize=10, ha="center", va="bottom", color="black")
        ax2.set_title("Future Close Prices for 10 Days", fontsize=16)
        ax2.set_xlabel("Day Ahead", fontsize=14)
        ax2.set_ylabel("Close Price", fontsize=14)
        ax2.legend(); ax2.grid(alpha=0.3)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

    # Candlestick Chart
    st.subheader("🕯️ Candlestick Chart")
    my_style = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=mpf.make_marketcolors(
            up="#22B14C", down="#DC143C",
            edge={"up":"#22B14C", "down":"#DC143C"},
            wick={"up":"#157D2A", "down":"#A60021"},
            volume={"up":"#B4E197", "down":"#F28585"}
        ),
        rc={"font.size": 12}
    )
    try:
        fig_mpf, _ = mpf.plot(
            ohlcv[['Open','High','Low','Close','Volume']],
            type="candle",
            style=my_style,
            title=f"{ticker} Price Action",
            ylabel="Price (USD)",
            ylabel_lower="Volume",
            volume=True,
            mav=(10, 20, 50),
            figscale=1.4,
            figratio=(14,7),
            tight_layout=True,
            returnfig=True
        )
        st.pyplot(fig_mpf)
    except Exception as e:
        st.warning(f"Candlestick chart error: {e}")
