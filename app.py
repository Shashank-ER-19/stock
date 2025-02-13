import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Stock Price Prediction using LSTM")

# Sidebar for user inputs
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG)", "GOOG")
epochs = st.sidebar.slider("Number of Epochs", 1, 50, 2)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Load data
end = dt.datetime.now()
start = dt.datetime(end.year-20, end.month, end.day)
st.write(f"Fetching data for {stock}...")
stock_data = yf.download(stock, start, end)

# Moving Averages
stock_data['MA_100'] = stock_data['Adj Close'].rolling(100).mean()
stock_data['MA_250'] = stock_data['Adj Close'].rolling(250).mean()
stock_data['percent_change'] = stock_data['Adj Close'].pct_change()

# Plot Closing Price Data
st.subheader("Adjusted Closing Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['Adj Close'], label='Adjusted Close Price')
ax.set_xlabel('Years')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Plot 100-Day Moving Average
st.subheader("100-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['Adj Close'], alpha=0.5, label='Actual Data', color='gray')
ax.plot(stock_data['MA_100'], label='100-Day MA', linestyle='dashed')
ax.set_xlabel('Years')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Plot 250-Day Moving Average
st.subheader("250-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['Adj Close'], alpha=0.5, label='Actual Data', color='gray')
ax.plot(stock_data['MA_250'], label='250-Day MA', linestyle='dotted')
ax.set_xlabel('Years')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Plot Percentage Change
st.subheader("Percentage Change in Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data['percent_change'], label='Percentage Change', linestyle='solid')
ax.set_xlabel('Years')
ax.set_ylabel('Change')
ax.legend()
st.pyplot(fig)

# Preprocessing
adj_close = stock_data[['Adj Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(adj_close)

X_data, y_data = [], []
for i in range(100, len(scaled_data)):
    X_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

X_data, y_data = np.array(X_data), np.array(y_data)
split = int(len(X_data) * 0.7)
X_train, y_train = X_data[:split], y_data[:split]
X_test, y_test = X_data[split:], y_data[split:]

# Model definition
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
if st.sidebar.button("Train Model"):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    st.success("Model Training Completed!")

    # Predictions
    predictions = model.predict(X_test)
    inv_preds = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test)

    # Plot Test Data Predictions
    st.subheader("Test Data Predictions vs Actual Data")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data.index[split+100:], inv_y_test, label='Actual')
    ax.plot(stock_data.index[split+100:], inv_preds, label='Predicted')
    ax.legend()
    st.pyplot(fig)

    # Whole Data Predictions (EXACTLY LIKE YOUR .PY FILE)
    st.subheader("Whole Data Predictions")
    combined_df = pd.concat([stock_data[['Adj Close']][:split+100], pd.DataFrame(inv_preds, index=stock_data.index[split+100:], columns=['Predicted'])])
    fig, ax = plt.subplots(figsize=(18, 5))

    # Keep the same color scheme as your reference image
    ax.plot(stock_data.index, stock_data["Adj Close"], label="(Adj Close, GOOG)", color="blue")  # Blue
    ax.plot(stock_data.index[-len(y_test):], inv_y_test, label="original test data", color="orange")  # Orange
    ax.plot(stock_data.index[-len(y_test):], inv_preds, label="preds", color="green")  # Green

    ax.set_xlabel("Years")
    ax.set_ylabel("Whole Data")
    ax.set_title("Whole Data of the Stock")
    ax.legend()
    st.pyplot(fig)

    # Future Predictions
    future_days = 7
    last_100_days = scaled_data[-100:]
    X_future = np.array([last_100_days])

    future_preds = []
    current_input = X_future[0].copy()
    for _ in range(future_days):
        prediction = model.predict(current_input.reshape(1, 100, 1))
        future_preds.append(prediction[0, 0])
        current_input = np.append(current_input[1:], prediction, axis=0)

    future_preds = np.array(future_preds).reshape(-1,1)
    inv_future_preds = scaler.inverse_transform(future_preds)
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.DateOffset(1), periods=future_days)

    # Future 7 Days Predictions
    st.subheader("Future Stock Price Predictions (Next 7 Days)")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(future_dates, inv_future_preds, marker='o', linestyle='dashed', label='Future Prediction')
    ax.legend()
    st.pyplot(fig)

st.write("Adjust the parameters and retrain the model to see different results!")
