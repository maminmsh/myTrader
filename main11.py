import dash
from dash import dcc, html, Input, Output
import ccxt
import pandas as pd
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

# محاسبات شاخص‌ها
def calculate_indicators(df):
    # محاسبه SMA و Bollinger Bands
    df['SMA'] = df['close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['SMA'] + (2 * df['close'].rolling(window=20).std())
    df['Bollinger_Lower'] = df['SMA'] - (2 * df['close'].rolling(window=20).std())

    # محاسبه RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # محاسبه MACD
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ساخت داشبورد
app = dash.Dash(__name__)

# دریافت داده‌ها
df = fetch_data()
df = calculate_indicators(df)

# طراحی رابط کاربری
app.layout = html.Div([
    html.H1("Interactive Trading Chart", style={'textAlign': 'center'}),
    
    # منوی کشویی برای انتخاب شاخص‌ها
    dcc.Dropdown(
        id='indicator-dropdown',
        options=[
            {'label': 'SMA & Bollinger Bands', 'value': 'bollinger'},
            {'label': 'RSI', 'value': 'rsi'},
            {'label': 'MACD', 'value': 'macd'}
        ],
        value='bollinger',  # مقدار پیش‌فرض
        multi=False  # انتخاب تکی
    ),
    
    # نمودار
    dcc.Graph(id='indicator-chart')
])

# به‌روزرسانی نمودار بر اساس شاخص انتخاب‌شده
@app.callback(
    Output('indicator-chart', 'figure'),
    [Input('indicator-dropdown', 'value')]
)
def update_chart(selected_indicator):
    fig = go.Figure()

    # افزودن کندل‌استیک
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    ))

    # افزودن شاخص‌ها بر اساس انتخاب کاربر
    if selected_indicator == 'bollinger':
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
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['SMA'],
            line=dict(color='blue', width=1.5),
            name='SMA'
        ))
    elif selected_indicator == 'rsi':
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['RSI'],
            line=dict(color='purple', width=1.5),
            name='RSI'
        ))
    elif selected_indicator == 'macd':
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['MACD'],
            line=dict(color='orange', width=1.5),
            name='MACD'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Signal'],
            line=dict(color='blue', width=1.5),
            name='Signal'
        ))

    # تنظیمات نمودار
    fig.update_layout(
        title="BTC/USDT Chart with Indicators",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode='x unified'
    )
    return fig

# اجرای برنامه
if __name__ == '__main__':
    app.run_server(debug=True)
