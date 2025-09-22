 📈 Bitcoin OHLCV and Predictive Analytics Dashboard
A full-featured Streamlit web app for visualizing Bitcoin OHLCV price data and running deep learning-based price prediction. Fetches data reliably using yfinance and auto-fallbacks to the CoinGecko API (with user-agent workaround) for robust deployment on Streamlit Cloud or your local machine.

🚀 Features
Interactive dashboard for Bitcoin historical price (OHLCV) visualization

Deep learning predictions with user-supplied trained model.keras

Auto-retries using CoinGecko as fallback if Yahoo Finance API throttles or fails

Volume and candlestick subplots using Matplotlib and mplfinance

User-selectable ticker, date range, and flexible error handling

Designed for cloud (Streamlit Cloud) deployment — works out-of-the-box!

📦 Tech Stack
Python 3.9+

Streamlit

yfinance

CoinGecko API (via requests)

pandas, numpy

matplotlib, mplfinance

scikit-learn

tensorflow/keras

appdirs, pytz, requests

🔧 How to Run
Clone this repo and upload to Streamlit Cloud or run locally:

bash
git clone https://github.com/yourusername/bitcoin-ohlcv-predict-dashboard.git
cd bitcoin-ohlcv-predict-dashboard
Ensure these files are present:

app2.py

requirements.txt

(Optional but recommended) model.keras (LSTM/ML model for prediction)

Install dependencies locally (optional):

bash
pip install -r requirements.txt
Run on local machine:

bash
streamlit run app2.py
OR deploy to Streamlit Cloud:

Push repo to GitHub

Create a new app on Streamlit Cloud, linking this repo and setting app2.py as the entry point

✨ Example
![dashboard-screenshot]( Add your screenshot file if available -->

⚠️ Troubleshooting & Cloud Deployment
If you see No data or too little data or CoinGecko fetch failed, check your API/network. This is handled with robust fallback logic.

Ensure your requirements.txt matches package versions used in the code.

Model predictions require a compatible model.keras trained on comparable data (see code for input shapes).

On Streamlit Cloud, the CoinGecko API fetch uses a user-agent header for compatibility.

🌍 Credits & Contributors
Developed by [Your Name/Org].
Inspired by analysis, open APIs, and the Streamlit data app ecosystem.

📜 License
Licensed under the MIT License. See LICENSE for details.

🤝 Contributions
Pull requests, bug reports, and suggestions welcome!
Use GitHub Issues for problems and feature requests.

📣 Contact
Questions or want to collaborate? Open an issue or reach out via email/GitHub profile.


