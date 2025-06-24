import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

plt.style.use('seaborn-v0_8-darkgrid')

st.set_page_config(layout="wide", page_title="Stock Price Prediction with LSTM", page_icon=":chart_with_upwards_trend:")
st.title("📈 Stock Price Prediction using LSTM")

# --- Sidebar for user inputs ---
with st.sidebar:
    stock = st.text_input("Enter Stock Ticker (e.g., GOOG)", "GOOG")
    today = dt.date.today()
    default_start = today - dt.timedelta(days=5*365)
    start_date = st.date_input("Start Date", default_start, min_value=dt.date(1990,1,1), max_value=today)
    end_date = st.date_input("End Date", today, min_value=start_date, max_value=today)
    epochs = st.number_input("Number of Epochs", min_value=1, max_value=200, value=30, step=1)
    batch_size = st.number_input("Batch Size", min_value=8, max_value=512, value=32, step=1)
    future_days = st.number_input("Days to Predict Ahead", min_value=1, max_value=60, value=7, step=1)
    window = st.number_input("Window Size (days)", min_value=30, max_value=300, value=100, step=1)

# --- Load data ---
st.write(f"Fetching data for **{stock}** from {start_date} to {end_date}...")

try:
    stock_data = yf.download(stock, start=start_date, end=end_date, threads=False)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if stock_data.empty:
    st.error("No data found for the given ticker and date range. Please check the ticker symbol or try another stock/date.")
    st.stop()

if 'Adj Close' not in stock_data.columns:
    stock_data['Adj Close'] = stock_data['Close']

# --- Feature Engineering ---
stock_data['MA_100'] = stock_data['Adj Close'].rolling(100).mean()
stock_data['MA_250'] = stock_data['Adj Close'].rolling(250).mean()
stock_data['percent_change'] = stock_data['Adj Close'].pct_change()

# --- Show Graphs ---
st.subheader("Adjusted Closing Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['Adj Close'], label='Adj Close', color='#4F8BF9')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

st.subheader("100-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['Adj Close'], alpha=0.5, label='Actual Data', color='gray')
ax.plot(stock_data['MA_100'], label='100-Day MA', linestyle='dashed')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

st.subheader("250-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['Adj Close'], alpha=0.5, label='Actual Data', color='gray')
ax.plot(stock_data['MA_250'], label='250-Day MA', linestyle='dotted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

st.subheader("Percentage Change in Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['percent_change'], label='Percentage Change', linestyle='solid')
ax.set_xlabel('Date')
ax.set_ylabel('Change')
ax.legend()
st.pyplot(fig)

# --- Preprocessing ---
adj_close = stock_data[['Adj Close']].dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(adj_close)

X_data, y_data = [], []
for i in range(window, len(scaled_data)):
    X_data.append(scaled_data[i-window:i])
    y_data.append(scaled_data[i])

X_data, y_data = np.array(X_data), np.array(y_data)
X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], 1))

split = int(len(X_data) * 0.7)
X_train, y_train = X_data[:split], y_data[:split]
X_test, y_test = X_data[split:], y_data[split:]

# --- Model definition (improved, but simple) ---
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# --- Progress callback ---
class StreamlitProgressCallback:
    def __init__(self, total_epochs, progress_bar, status_placeholder):
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_placeholder = status_placeholder

    def on_epoch_begin(self, epoch, logs=None):
        self.status_placeholder.info(f"Epoch {epoch+1}/{self.total_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch+1)/self.total_epochs)

    def get_callback(self):
        from keras.callbacks import LambdaCallback
        return LambdaCallback(
            on_epoch_begin=self.on_epoch_begin,
            on_epoch_end=self.on_epoch_end
        )

if st.sidebar.button("Train Model"):
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    callback = StreamlitProgressCallback(epochs, progress_bar, status_placeholder).get_callback()

    with st.spinner("Training the model..."):
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[callback],
            validation_data=(X_test, y_test)
        )
    status_placeholder.success("Model Training Completed!")
    progress_bar.empty()

    # --- Plot Loss Curve ---
    st.subheader("Training and Validation Loss")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    # --- Predictions ---
    predictions = model.predict(X_test)
    inv_preds = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test)

    # --- Plot Test Data Predictions ---
    st.subheader("Test Data Predictions vs Actual Data")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(adj_close.index[-len(inv_y_test):], inv_y_test, label='Actual', color='#F9A826')
    ax.plot(adj_close.index[-len(inv_preds):], inv_preds, label='Predicted', color='#21BF73')
    ax.legend()
    st.pyplot(fig)

    # --- Whole Data Predictions ---
    st.subheader("Whole Data Predictions")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(adj_close.index, adj_close["Adj Close"], label="(Adj Close)", color="#4F8BF9")
    ax.plot(adj_close.index[-len(inv_y_test):], inv_y_test, label="original test data", color="#F9A826")
    ax.plot(adj_close.index[-len(inv_preds):], inv_preds, label="preds", color="#21BF73")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Whole Data of the Stock")
    ax.legend()
    st.pyplot(fig)

    # --- Future Predictions ---
    last_window = scaled_data[-window:]
    X_future = last_window.reshape(1, window, 1)

    future_preds = []
    current_input = X_future[0].copy()
    for _ in range(future_days):
        pred = model.predict(current_input.reshape(1, window, 1))
        current_input = np.append(current_input[1:], pred, axis=0)
        future_preds.append(pred[0,0])

    inv_future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
    future_dates = pd.date_range(start=adj_close.index[-1] + pd.DateOffset(1), periods=future_days)

    st.subheader(f"Future Stock Price Predictions (Next {future_days} Days)")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(future_dates, inv_future_preds, marker='o', linestyle='dashed', label='Future Prediction', color='#E84545')
    ax.legend()
    st.pyplot(fig)

st.write("Adjust the parameters and retrain the model to see different results!")
