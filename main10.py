import ccxt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time

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

# رسم نمودارهای جداگانه برای هر اندیکاتور
def plot_separate_charts(df):
    # ایجاد ساب‌پلات‌ها
    fig = make_subplots(
        rows=3, cols=1,  # 3 نمودار (قیمت+باندهای بولینگر، RSI، MACD)
        shared_xaxes=True,  # اشتراک‌گذاری محور x
        vertical_spacing=0.05,  # فاصله عمودی بین نمودارها
        row_heights=[0.5, 0.25, 0.25],  # ارتفاع نسبی هر نمودار
        subplot_titles=("BTC/USDT Price + Bollinger Bands", "RSI", "MACD")
    )

    # نمودار قیمت + باندهای بولینگر + SMA
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['Bollinger_Upper'],
            line=dict(color='red', width=1),
            name='Bollinger Upper'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['Bollinger_Lower'],
            line=dict(color='green', width=1),
            name='Bollinger Lower'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['SMA'],
            line=dict(color='blue', width=1.5),
            name='SMA'
        ),
        row=1, col=1
    )

    # نمودار RSI
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['RSI'],
            line=dict(color='brown', width=1.5),
            name='RSI'
        ),
        row=2, col=1
    )

    # نمودار MACD
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['MACD'],
            line=dict(color='purple', width=1.5),
            name='MACD'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['Signal'],
            line=dict(color='orange', width=1.5),
            name='Signal'
        ),
        row=3, col=1
    )

    # تنظیمات نهایی
    fig.update_layout(
        height=900,  # ارتفاع کل نمودار
        title_text="BTC/USDT Analysis with Indicators",
        xaxis_rangeslider_visible=False,  # حذف محدوده اسکرول در کندل‌استیک
        hovermode='x unified'
    )

    # نمایش نمودار
    fig.show()


# تحلیل بلادرنگ
def live_trading():
    while True:
        df = fetch_data()
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_bollinger_bands(df)

        # رسم نمودارها
        plot_separate_charts(df)

        time.sleep(300)  # اجرای دوباره هر 5 دقیقه

# اجرای برنامه
if __name__ == '__main__':
    live_trading()
