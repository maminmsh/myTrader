import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ----------- متغیرهای تغییرپذیر ----------- #
EPOCHS = 5                # تعداد ایپاک‌ها
LOOK_BACK = 500           # تعداد کندل‌های قبلی برای پیش‌بینی
limit_candle = 1000       # تعداد کندل‌هایی که از صرافی دریافت می‌شود
batch_size = 32           # اندازه دسته برای آموزش مدل
timeframe = '5m'          # تایم‌فریم داده‌ها (5 دقیقه‌ای)
symbol = 'BTC/USDT'       # جفت‌ارز برای پیش‌بینی
proxy_port = 400          # پورت پروکسی
future_steps = 10         # تعداد پیش‌بینی‌های آینده
# ---------------------------------------- #

# تنظیم پروکسی
proxies = {
    'http': 'socks5h://127.0.0.1:400',
    'https': 'socks5h://127.0.0.1:400',
}

# 1. دریافت داده‌های بیت‌کوین از صرافی
def fetch_data():
    exchange = ccxt.binance({'proxies': proxies})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit_candle)
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# 2. پیش‌پردازش داده‌ها برای آموزش مدل
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = ['open', 'high', 'low', 'close', 'volume']
    scaled_data = scaler.fit_transform(df[features].values)
    X, y = [], []
    for i in range(LOOK_BACK, len(scaled_data)):
        X.append(scaled_data[i-LOOK_BACK:i, :])  # ورودی‌ها
        y.append(scaled_data[i, 3])  # مقدار قیمت close به‌عنوان هدف
    return np.array(X), np.array(y), scaler

# 3. ساخت مدل LSTM و آموزش آن
def build_and_train_model(X_train, y_train):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(200, return_sequences=True),
        Dropout(0.2),
        LSTM(200),
        Dropout(0.2),
        Dense(1)  # خروجی پیش‌بینی قیمت
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=batch_size, verbose=1)
    return model

# 4. پیش‌بینی قیمت‌های آینده
def predict_future_prices(model, last_data, scaler, n_steps):
    predictions = []
    current_input = last_data.copy()

    for _ in range(n_steps):
        predicted_price = model.predict(current_input, verbose=0)
        predicted_price_rescaled = scaler.inverse_transform(np.hstack((current_input[:, -1, :-1], predicted_price)))
        predictions.append(predicted_price_rescaled[0, 3])  # مقدار قیمت پیش‌بینی شده close
        next_input = np.hstack((current_input[:, -1, 1:], predicted_price))
        current_input = np.append(current_input[:, 1:, :], next_input.reshape(1, 1, -1), axis=1)

    return predictions

# 5. رسم نمودار نتایج پیش‌بینی
def plot_predictions(df, scaler, model, X_test, future_predictions):
    predictions = model.predict(X_test, verbose=0)
    predictions_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :-1], predictions)))
    actual = scaler.inverse_transform(X_test[:, -1, :])

    plt.figure(figsize=(12, 6))
    plt.plot(df['time'][-len(actual):], actual[:, 3], label='Actual Price', color='blue')
    plt.plot(df['time'][-len(predictions_rescaled):], predictions_rescaled[:, 3], label='Predicted Price', linestyle='dotted', color='orange')

    future_dates = pd.date_range(df['time'].iloc[-1], periods=len(future_predictions) + 1, freq=timeframe)[1:]
    plt.plot(future_dates, future_predictions, label='Future Predicted Price', linestyle='dashed', color='green')

    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Price Prediction (LSTM)')
    plt.legend()

    output_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"bitcoin_prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Plot saved at: {file_path}")

# اجرای برنامه
if __name__ == "__main__":
    df = fetch_data()  # دریافت داده‌ها
    X, y, scaler = preprocess_data(df)  # پیش‌پردازش داده‌ها

    # تقسیم داده‌ها به آموزش و تست
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # ساخت و آموزش مدل
    model = build_and_train_model(X_train, y_train)

    # پیش‌بینی آینده
    future_predictions = predict_future_prices(model, X_test[-1:], scaler, n_steps=future_steps)

    # رسم نمودار
    plot_predictions(df[-len(X_test):], scaler, model, X_test, future_predictions)
