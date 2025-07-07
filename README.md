# 📈 Stock Price Prediction using LSTM

A Streamlit web application for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. This app allows users to fetch historical stock data, visualize trends, train an LSTM model, and predict future stock prices interactively.

---

## 🚀 Features

- **Interactive UI**: Enter any stock ticker, select date ranges, and adjust model parameters from the sidebar.
- **Data Visualization**: View historical prices, moving averages, and percentage changes with clear, interactive plots.
- **LSTM Model Training**: Train a deep learning model on your selected data with customizable epochs, batch size, and window size.
- **Progress Feedback**: Real-time progress bar and status updates during model training.
- **Prediction & Evaluation**: Visualize model predictions vs. actual data, and forecast future stock prices for up to 60 days.

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone <repo-url>
cd assignment/stock
```

2. **Install dependencies**

It is recommended to use a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate #source venv/bin/activate
pip install -r requirements.txt
```

---

## 💡 Usage

1. **Run the Streamlit app**

```bash
streamlit run "streamlit app.py"
```

2. **Open in your browser**

Streamlit will provide a local URL (usually http://localhost:8501). Open it to interact with the app.

3. **How to Use**
   - Enter a valid stock ticker (e.g., `GOOG`, `AAPL`, `MSFT`).
   - Select the date range for historical data.
   - Adjust model parameters (epochs, batch size, window size, days to predict ahead).
   - Click **Train Model** to start training and see results.
   - View various plots: historical prices, moving averages, loss curves, predictions, and future forecasts.

---

## 📦 Project Structure

```
assignment/
  └── stock/
      ├── streamlit app.py        # Main Streamlit application
      ├── requirements.txt        # Python dependencies
      └── README.md               # Project documentation
```

---

## 📚 Dependencies

- streamlit
- yfinance
- numpy
- pandas
- matplotlib
- scikit-learn
- keras
- tensorflow

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Technical Details

### Data Pipeline
- **Data Acquisition**: Historical stock data is fetched using the `yfinance` API, which provides open, high, low, close, adjusted close, and volume data for the selected ticker and date range.
- **Feature Engineering**: The app computes 100-day and 250-day moving averages and the daily percentage change in adjusted close price to provide additional context for the time series.

### Preprocessing
- **Scaling**: The adjusted close price is normalized to the [0, 1] range using `MinMaxScaler` to stabilize and accelerate neural network training.
- **Windowing**: The time series is split into overlapping windows of configurable length (default: 100 days). Each window is used as an input sequence to the LSTM, with the next day's price as the target.
- **Train/Test Split**: The dataset is split into training (70%) and testing (30%) sets to evaluate model generalization.

### Model Architecture
- **LSTM Layers**: The model consists of two stacked LSTM layers (128 and 64 units) to capture temporal dependencies in the stock price sequence.
- **Dropout**: Dropout layers (rate=0.2) are used after each LSTM to reduce overfitting.
- **Dense Layers**: A dense layer with 32 units and ReLU activation is followed by a final output layer with a single neuron for regression.
- **Loss & Optimization**: The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer.

### Training & Prediction Workflow
- **Training**: The model is trained interactively with user-defined epochs and batch size. Training and validation loss curves are displayed for monitoring.
- **Prediction**: After training, the model predicts the test set and future prices. Predictions are inverse-transformed to the original price scale.
- **Visualization**: The app plots actual vs. predicted prices, moving averages, and future forecasts for intuitive evaluation.

### Streamlit Integration
- **User Controls**: All model and data parameters are adjustable via the sidebar.
- **Progress Feedback**: Custom callbacks update the UI with training progress and status.
- **Interactive Plots**: All results are visualized using Matplotlib and rendered in the Streamlit interface.

---

## 📝 Notes

- The app fetches data using [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` library.
- Model training is performed in-browser and may take time depending on your hardware and selected parameters.
- For best results, use stocks with sufficient historical data.

---

## 📄 License

This project is for educational purposes. Feel free to use and modify it for your own learning or research.