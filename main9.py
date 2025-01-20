import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import time
import smtplib
import plotly.graph_objects as go

# تنظیم پروکسی
proxies = {
    'http': 'socks5h://127.0.0.1:400',
    'https': 'socks5h://127.0.0.1:400',
}

# گرفتن داده‌ها از بایننس
def fetch_data(timeframe='5m', limit=500):
    exchange = ccxt.binance({'proxies': proxies})
    data = exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# محاسبه RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# محاسبه MACD
def calculate_macd(df, short_span=12, long_span=26, signal_span=9):
    df['MACD'] = df['close'].ewm(span=short_span, adjust=False).mean() - df['close'].ewm(span=long_span, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=signal_span, adjust=False).mean()
    return df

# محاسبه باندهای بولینگر
def calculate_bollinger_bands(df, period=20):
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Bollinger_Upper'] = df['SMA'] + (2 * df['close'].rolling(window=period).std())
    df['Bollinger_Lower'] = df['SMA'] - (2 * df['close'].rolling(window=period).std())
    return df

# ایجاد سیگنال خرید و فروش
def generate_signals(df):
    df['Buy_Signal'] = (df['RSI'] < 30) & (df['MACD'] > df['Signal']) & (df['close'] < df['Bollinger_Lower'])
    df['Sell_Signal'] = (df['RSI'] > 70) & (df['MACD'] < df['Signal']) & (df['close'] > df['Bollinger_Upper'])
    return df

# ذخیره سیگنال‌ها در فایل CSV
def save_signals(df, filename='signals.csv'):
    df.to_csv(filename, index=False)
    print(f"Signals saved to {filename}")

# ارسال ایمیل برای سیگنال‌ها
# def send_email(signal, price, time):
#     sender = "your_email@gmail.com"
#     receiver = "receiver_email@gmail.com"
#     password = "your_password"
#     message = f"Subject: Trading Signal\n\nSignal: {signal}\nPrice: {price}\nTime: {time}"
#     with smtplib.SMTP('smtp.gmail.com', 587) as server:
#         server.starttls()
#         server.login(sender, password)
#         server.sendmail(sender, receiver, message)

# رسم نمودار تعاملی با Plotly
def plot_interactive_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    ))
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['Bollinger_Upper'],
        line=dict(color='red', width=1),
        name='Bollinger Upper'
    ))
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['Bollinger_Lower'],
        line=dict(color='green', width=1),
        name='Bollinger Lower'
    ))
    fig.update_layout(title='BTC/USDT Price with Bollinger Bands', xaxis_title='Time', yaxis_title='Price')
    fig.show()

# تحلیل بلادرنگ
def live_trading():
    while True:
        df = fetch_data()
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_bollinger_bands(df)
        df = generate_signals(df)

        # ارسال هشدارها
        if df['Buy_Signal'].iloc[-1]:
            # send_email('Buy', df['close'].iloc[-1], df['time'].iloc[-1])
            print(f"Buy Signal at {df['close'].iloc[-1]} USDT")
        elif df['Sell_Signal'].iloc[-1]:
            # send_email('Sell', df['close'].iloc[-1], df['time'].iloc[-1])
            print(f"Sell Signal at {df['close'].iloc[-1]} USDT")

        # ذخیره سیگنال‌ها
        save_signals(df)

        # رسم نمودار
        plot_interactive_chart(df)

        time.sleep(300)  # اجرای دوباره هر 5 دقیقه

# اجرای برنامه
if __name__ == '__main__':
    live_trading()
