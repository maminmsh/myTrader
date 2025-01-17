import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention
from sklearn.preprocessing import MinMaxScaler

# تنظیمات اولیه
LOOK_BACK = 50  # تعداد کندل‌های قبلی
EPOCHS = 20
BATCH_SIZE = 64

# 1. آماده‌سازی داده‌ها
def preprocess_data(df):
    features = ['open', 'high', 'low', 'close', 'volume']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features].values)
    
    X, y = [], []
    for i in range(LOOK_BACK, len(scaled_data)):
        X.append(scaled_data[i-LOOK_BACK:i, :])
        y.append(scaled_data[i, 3])  # پیش‌بینی قیمت close
    
    return np.array(X), np.array(y), scaler

# 2. طراحی مدل جدید
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # لایه‌های LSTM
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    
    # مکانیزم Attention
    attention_output = Attention()([x, x])
    
    # لایه نهایی
    x = Dense(64, activation='relu')(attention_output)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 3. آموزش و تست مدل
def train_and_evaluate(df):
    X, y, scaler = preprocess_data(df)
    
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)
    return model, scaler, X_test, y_test

# 4. پیش‌بینی قیمت آینده
def predict_future_prices(model, last_data, scaler, n_steps):
    predictions = []
    current_input = last_data.copy()

    for _ in range(n_steps):
        predicted_price = model.predict(current_input, verbose=0)
        predictions.append(predicted_price[0, 0])
        next_input = np.append(current_input[:, 1:, :], [[predicted_price]], axis=1)
        current_input = next_input

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# اجرای برنامه
if __name__ == "__main__":
    # دریافت داده‌ها
    df = fetch_data()
    
    # آموزش مدل
    model, scaler, X_test, y_test = train_and_evaluate(df)
    
    # پیش‌بینی قیمت آینده
    last_data = X_test[-1:]
    future_prices = predict_future_prices(model, last_data, scaler, n_steps=10)
    
    print("Future Prices:", future_prices)

#   import ccxt
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # تنظیم پروکسی
# proxies = {
#     'http': 'socks5h://127.0.0.1:400',
#     'https': 'socks5h://127.0.0.1:400',
# }

# EPOCHS = 10  # تعداد ایپاک‌ها
# LOOK_BACK = 100  # تعداد کندل‌های قبلی برای پیش‌بینی
# limit_candle = 10000  # تعداد کندل‌ها

# # 1. دریافت داده‌های بیت‌کوین
# def fetch_data():
#     exchange = ccxt.binance({'proxies': proxies})
#     data = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=limit_candle)
#     df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
#     df['time'] = pd.to_datetime(df['time'], unit='ms')
#     return df

# # 2. پیش‌پردازش داده‌ها
# def preprocess_data(df):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     features = ['open', 'high', 'low', 'close', 'volume']
#     scaled_data = scaler.fit_transform(df[features].values)
#     X, y = [], []
#     for i in range(LOOK_BACK, len(scaled_data)):
#         X.append(scaled_data[i-LOOK_BACK:i, :])
#         y.append(scaled_data[i, 3])  # پیش‌بینی قیمت close
#     return np.array(X), np.array(y), scaler

# # 3. ساخت و آموزش مدل LSTM
# def build_and_train_model(X_train, y_train):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#         Dropout(0.2),
#         LSTM(100),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, verbose=1)
#     return model

# # 4. پیش‌بینی چند گام آینده
# def predict_future_prices(model, last_data, scaler, n_steps):
#     predictions = []
#     current_input = last_data.copy()

#     for _ in range(n_steps):
#         # پیش‌بینی قیمت
#         predicted_price = model.predict(current_input, verbose=0)

#         # ساخت داده ورودی بعدی
#         next_input = np.hstack((
#             current_input[:, -1, :-1],  # تمام ویژگی‌ها به جز ویژگی close
#             predicted_price.reshape(-1, 1)  # اضافه کردن پیش‌بینی close
#         ))
        
#         next_input = next_input.reshape(1, -1, current_input.shape[2])  # تغییر شکل برای سازگاری با مدل
#         predictions.append(predicted_price[0, 0])
#         current_input = next_input

#     # بازگرداندن پیش‌بینی‌ها به مقیاس اصلی
#     predictions_rescaled = scaler.inverse_transform(
#         np.hstack((np.zeros((len(predictions), current_input.shape[2] - 1)), np.array(predictions).reshape(-1, 1)))
#     )[:, -1]

#     return predictions_rescaled



# # 5. ارزیابی مدل
# def evaluate_model(y_true, y_pred):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     print(f"Mean Absolute Error (MAE): {mae}")
#     print(f"Root Mean Squared Error (RMSE): {rmse}")

# # 6. رسم نمودار نتایج
# def plot_predictions(df, scaler, model, X_test, y_test, future_predictions):
#     predictions = model.predict(X_test, verbose=0)
#     predictions_rescaled = scaler.inverse_transform(
#         np.hstack((X_test[:, -1, :-1], predictions))
#     )[:, -1]
#     actual = scaler.inverse_transform(X_test[:, -1, :])[:, -1]

#     plt.figure(figsize=(12, 6))
#     plt.plot(df['time'][-len(actual):], actual, label='Actual Price', color='blue')
#     plt.plot(df['time'][-len(predictions_rescaled):], predictions_rescaled, label='Predicted Price', linestyle='dotted', color='orange')

#     future_dates = pd.date_range(df['time'].iloc[-1], periods=len(future_predictions) + 1, freq='5T')[1:]
#     plt.plot(future_dates, future_predictions, label='Future Predicted Price', linestyle='dashed', color='green')

#     plt.xlabel('Time')
#     plt.ylabel('Price (USD)')
#     plt.title('Bitcoin Price Prediction (LSTM)')
#     plt.legend()

#     output_dir = os.path.join(os.getcwd(), "plots")
#     os.makedirs(output_dir, exist_ok=True)
#     file_path = os.path.join(output_dir, f"bitcoin_prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
#     plt.savefig(file_path)
#     plt.close()
#     print(f"Plot saved at: {file_path}")

# # اجرای برنامه
# if __name__ == "__main__":
#     df = fetch_data()
#     X, y, scaler = preprocess_data(df)

#     split = int(len(X) * 0.8)
#     X_train, y_train = X[:split], y[:split]
#     X_test, y_test = X[split:], y[split:]

#     model = build_and_train_model(X_train, y_train)

#     y_pred = model.predict(X_test, verbose=0)
#     evaluate_model(y_test, y_pred)

#     future_predictions = predict_future_prices(model, X_test[-1:], scaler, n_steps=10)

#     plot_predictions(df[-len(X_test):], scaler, model, X_test, y_test, future_predictions)
