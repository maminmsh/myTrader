import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# تنظیم پروکسی
proxies = {
    'http': 'socks5h://127.0.0.1:400',
    'https': 'socks5h://127.0.0.1:400',
}

# 1. دریافت داده‌های بیت‌کوین
def fetch_data():
    exchange = ccxt.binance({'enableRateLimit': True, 'proxies': proxies})
    data = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=1000)
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# 2. پیش‌پردازش داده‌ها
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    look_back = 50  # تعداد کندل‌های قبلی برای پیش‌بینی
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# 3. ساخت و آموزش مدل LSTM
def build_and_train_model(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

# 4. پیش‌بینی و رسم نمودار
def plot_predictions(df, scaler, model, X_test):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(X_test[:, -1].reshape(-1, 1))

    # رسم نمودار
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'][-len(actual):], actual, label='Actual Price', color='blue')
    plt.plot(df['time'][-len(predictions):], predictions, label='Predicted Price', linestyle='dotted', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Price Prediction (LSTM)')
    plt.legend()

    # ذخیره نمودار
    output_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"bitcoin_prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Plot saved at: {file_path}")

# اجرای برنامه
if __name__ == "__main__":
    # دریافت داده‌ها
    df = fetch_data()

    # پیش‌پردازش
    X, y, scaler = preprocess_data(df)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # تقسیم داده‌ها به آموزش و تست
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # ساخت و آموزش مدل
    model = build_and_train_model(X_train, y_train)

    # پیش‌بینی و نمایش
    plot_predictions(df[-len(X_test):], scaler, model, X_test)
