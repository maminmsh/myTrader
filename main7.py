import ccxt
import pandas as pd
import matplotlib.pyplot as plt

proxies = {
    'http': 'socks5h://127.0.0.1:400',
    'https': 'socks5h://127.0.0.1:400',
}

# تابع برای گرفتن داده‌ها از بایننس
def fetch_data():
    exchange = ccxt.binance({'proxies': proxies})
    data = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=500)  # تایم‌فریم 5 دقیقه
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')  # تبدیل زمان
    return df

# محاسبه RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# محاسبه MACD و سیگنال
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

# محاسبه سیگنال خرید و فروش
def generate_signals(df):
    # شرط‌های سیگنال خرید و فروش
    df['Buy_Signal'] = (df['RSI'] < 30) & (df['MACD'] > df['Signal']) & (df['close'] < df['Bollinger_Lower'])
    df['Sell_Signal'] = (df['RSI'] > 70) & (df['MACD'] < df['Signal']) & (df['close'] > df['Bollinger_Upper'])
    return df

# رسم نمودار قیمت و سیگنال‌ها
def plot_signals(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['close'], label='Close Price', color='blue')
    plt.scatter(df['time'][df['Buy_Signal']], df['close'][df['Buy_Signal']], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df['time'][df['Sell_Signal']], df['close'][df['Sell_Signal']], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.title('Price with Buy and Sell Signals')
    plt.legend()
    plt.show()

# رسم نمودار MACD
def plot_macd(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['MACD'], label='MACD', color='blue')
    plt.plot(df['time'], df['Signal'], label='Signal', color='red')
    plt.title('MACD and Signal')
    plt.legend()
    plt.show()

# رسم نمودار RSI
def plot_rsi(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['RSI'], label='RSI', color='purple')
    plt.axhline(30, linestyle='--', color='green', label='Buy Threshold (30)')
    plt.axhline(70, linestyle='--', color='red', label='Sell Threshold (70)')
    plt.title('RSI Over Time')
    plt.legend()
    plt.show()

# اجرای تحلیل
def main():
    df = fetch_data()  # گرفتن داده‌ها
    df = calculate_rsi(df)  # محاسبه RSI
    df = calculate_macd(df)  # محاسبه MACD
    df = calculate_bollinger_bands(df)  # محاسبه باندهای بولینگر
    df = generate_signals(df)  # تولید سیگنال‌ها

    # نمایش داده‌ها
    print(df[['time', 'close', 'RSI', 'MACD', 'Signal', 'Buy_Signal', 'Sell_Signal']].tail(10))
    
    # رسم نمودارها
    plot_signals(df)  # نمودار قیمت و سیگنال‌ها
    plot_macd(df)  # نمودار MACD
    plot_rsi(df)  # نمودار RSI

# اجرا
if __name__ == '__main__':
    main()
