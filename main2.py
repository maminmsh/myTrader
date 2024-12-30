import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# تنظیم پروکسی
proxies = {
    'http': 'socks5h://127.0.0.1:400',
    'https': 'socks5h://127.0.0.1:400',
}

# تابع برای گرفتن داده‌ها
def fetch_data():
    exchange = ccxt.binance({'proxies': proxies})
    data = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=1000)  # تایم‌فریم 5 دقیقه
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['time_ir'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tehran')
    return df

# تابع برای محاسبه RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# تابع برای محاسبه MACD و Signal
def calculate_macd(df, short_span=12, long_span=26, signal_span=9):
    short_ema = df['close'].ewm(span=short_span, min_periods=1, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_span, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_span, min_periods=1, adjust=False).mean()
    return macd, signal

# تابع برای محاسبه SMA و EMA
def calculate_moving_averages(df, sma_period=20, ema_period=20):
    df['SMA'] = df['close'].rolling(window=sma_period).mean()
    df['EMA'] = df['close'].ewm(span=ema_period, min_periods=1, adjust=False).mean()
    return df

# تابع برای محاسبه Bollinger Bands
def calculate_bollinger_bands(df, period=20):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['Bollinger_Upper'] = sma + (2 * std)
    df['Bollinger_Lower'] = sma - (2 * std)
    return df

# تابع برای محاسبه Stochastic Oscillator
def calculate_stochastic_oscillator(df, period=14):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['%K'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

# تابع برای ذخیره داده‌ها در SQLite
def save_to_sqlite(df, table_name):
    conn = sqlite3.connect('my_database.db')
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# تابع برای انجام تحلیل و تولید سیگنال‌ها
def perform_analysis():
    df = fetch_data()
    
    # محاسبه اندیکاتورها
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_stochastic_oscillator(df)
    
    # سیگنال خرید و فروش
    df['Buy_Signal'] = (df['RSI'] < 40) & (df['MACD'] > df['Signal']) & (df['%K'] < 20)
    df['Sell_Signal'] = (df['RSI'] > 60) & (df['MACD'] < df['Signal']) & (df['%K'] > 80)
    
    return df

# تابع برای رسم نمودار RSI
def plot_rsi(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['RSI'], label='RSI', color='blue')
    plt.axhline(40, linestyle='--', color='green', label='Buy Threshold (40)')
    plt.axhline(60, linestyle='--', color='red', label='Sell Threshold (60)')
    plt.title('RSI Over Time')
    plt.legend()
    plt.show()

# تابع برای رسم نمودار MACD و Signal
def plot_macd(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['MACD'], label='MACD', color='blue')
    plt.plot(df['time'], df['Signal'], label='Signal', color='red')
    plt.title('MACD and Signal Over Time')
    plt.legend()
    plt.show()

# تابع برای رسم نمودار Bollinger Bands
def plot_bollinger_bands(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['close'], label='Close Price', color='blue')
    plt.plot(df['time'], df['Bollinger_Upper'], label='Upper Band', linestyle='--', color='red')
    plt.plot(df['time'], df['Bollinger_Lower'], label='Lower Band', linestyle='--', color='green')
    plt.title('Bollinger Bands')
    plt.legend()
    plt.show()

# تابع برای رسم نمودار Stochastic Oscillator
def plot_stochastic_oscillator(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['%K'], label='%K', color='blue')
    plt.plot(df['time'], df['%D'], label='%D', color='red')
    plt.axhline(20, linestyle='--', color='green', label='Oversold (20)')
    plt.axhline(80, linestyle='--', color='red', label='Overbought (80)')
    plt.title('Stochastic Oscillator')
    plt.legend()
    plt.show()

# اجرای تحلیل
df_result = perform_analysis()

save_to_sqlite(df_result, 'analysis_results')


# نمایش نتایج آخرین سیگنال‌ها
print(df_result[['time_ir', 'close', 'RSI', 'MACD', 'Signal', 'SMA', 'EMA', 'Bollinger_Upper', 'Bollinger_Lower', '%K', '%D', 'Buy_Signal', 'Sell_Signal']].tail(200))

# رسم نمودارها
# plot_rsi(df_result)
# plot_macd(df_result)
# plot_bollinger_bands(df_result)
# plot_stochastic_oscillator(df_result)
