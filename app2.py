from pathlib import Path
import appdirs as ad

# Fix for yfinance cache directory issue on Streamlit Cloud
CACHE_DIR = ".cache"
ad.user_cache_dir = lambda *args, **kwargs: CACHE_DIR
Path(CACHE_DIR).mkdir(exist_ok=True)

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("📈 Bitcoin OHLCV and Predictive Analytics Dashboard")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Ticker Symbol", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2014-12-17"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-09-14"))

# Download data with error handling
st.write(f"Fetching data for **{ticker}** from {start_date} to {end_date}...")
try:
    ohlcv = yf.download(ticker, start=start_date, end=end_date)
except Exception as e:
    st.error(f"Error fetching data for {ticker}: {e}")
    ohlcv = pd.DataFrame()

if ohlcv.empty:
    st.error("No data fetched! Please check the ticker symbol or date range.")
else:
    # Flatten MultiIndex (if present)
    if isinstance(ohlcv.columns, pd.MultiIndex):
        ohlcv.columns = ohlcv.columns.get_level_values(0)
    # Ensure numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in ohlcv.columns and isinstance(ohlcv[col], pd.Series):
            ohlcv[col] = pd.to_numeric(ohlcv[col], errors='coerce')
    # Drop NA
    ohlcv = ohlcv.dropna(subset=['Open', 'High', 'Low', 'Close'])
    # Show data preview
    st.subheader("📊 Data Preview")
    st.dataframe(ohlcv.head())
    # OHLCV Subplots
    st.subheader("📉 OHLCV Charts")
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    axes[0].plot(ohlcv.index, ohlcv['Open'], label="Open", color="blue")
    axes[0].set_ylabel("Open"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ohlcv.index, ohlcv['High'], label="High", color="green")
    axes[1].set_ylabel("High"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(ohlcv.index, ohlcv['Low'], label="Low", color="red")
    axes[2].set_ylabel("Low"); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[3].plot(ohlcv.index, ohlcv['Close'], label="Close", color="purple")
    axes[3].set_ylabel("Close"); axes[3].legend(); axes[3].grid(alpha=0.3)
    axes[4].bar(ohlcv.index, ohlcv['Volume'], label="Volume", color="orange", alpha=0.6)
    axes[4].set_ylabel("Volume"); axes[4].legend(); axes[4].grid(alpha=0.3)
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # ------------------- Predictive Modeling -------------------
    st.subheader("🤖 Model Predictions")
    try:
        # Load model from same folder
        model = load_model("model.keras")
        st.success("✅ Model loaded successfully!")
        # Use Close prices for prediction
        closing_price = ohlcv[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_price)
        # Same sequence length used during training
        base_days = 60  
        # Build test dataset
        test_data = scaled_data[-(base_days+30):]  # last 60+30 days
        x_test, y_test = [], []
        for i in range(base_days, len(test_data)):
            x_test.append(test_data[i-base_days:i, 0])
            y_test.append(test_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # Predictions
        predictions = model.predict(x_test)
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        # Plot Prediction vs Actual
        plotting_data = pd.DataFrame(
            {"Original": inv_y_test.flatten(),
             "Prediction": inv_predictions.flatten()},
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
        # Future Predictions (Next 10 Days)
        future_input = scaled_data[-base_days:].reshape(1, base_days, 1)
        future_predictions = []
        for _ in range(10):
            pred = model.predict(future_input)[0][0]
            future_predictions.append(pred)
            future_input = np.append(future_input[:,1:,:], [[[pred]]], axis=1)
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

    # ------------------- mplfinance Candlestick -------------------
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
    fig_mpf, _ = mpf.plot(
        ohlcv,
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
