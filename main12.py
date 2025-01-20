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

# محاسبه شاخص‌ها
def calculate_indicators(df, sma_period=20, rsi_period=14):
    # محاسبه SMA و Bollinger Bands
    df['SMA'] = df['close'].rolling(window=sma_period).mean()
    df['Bollinger_Upper'] = df['SMA'] + (2 * df['close'].rolling(window=sma_period).std())
    df['Bollinger_Lower'] = df['SMA'] - (2 * df['close'].rolling(window=sma_period).std())

    # محاسبه RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # محاسبه MACD
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ساخت برنامه Dash
app = dash.Dash(__name__)

# گرفتن داده‌ها
df = fetch_data()
df = calculate_indicators(df)

# طراحی رابط کاربری
app.layout = html.Div([
    html.H1("Interactive Trading Chart", style={'textAlign': 'center'}),

    # ورودی برای تغییر مقادیر SMA و RSI
    html.Div([
        html.Label("SMA Period:"),
        dcc.Input(id='sma-period', type='number', value=20, min=1, max=100, step=1),

        html.Label("RSI Period:", style={'marginLeft': '20px'}),
        dcc.Input(id='rsi-period', type='number', value=14, min=1, max=50, step=1),
    ], style={'marginBottom': '20px'}),

    # نمودارها
    dcc.Graph(id='main-chart'),
    dcc.Graph(id='rsi-chart'),
    dcc.Graph(id='macd-chart'),
])

# Callback برای به‌روزرسانی نمودارها
@app.callback(
    [Output('main-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('macd-chart', 'figure')],
    [Input('sma-period', 'value'),
     Input('rsi-period', 'value')]
)
def update_charts(sma_period, rsi_period):
    # محاسبه شاخص‌ها با مقادیر جدید
    updated_df = calculate_indicators(df, sma_period=sma_period, rsi_period=rsi_period)

    # نمودار اصلی (کندل‌استیک + باندهای بولینگر)
    main_fig = go.Figure()
    main_fig.add_trace(go.Candlestick(
        x=updated_df['time'],
        open=updated_df['open'],
        high=updated_df['high'],
        low=updated_df['low'],
        close=updated_df['close'],
        name='Candlestick'
    ))
    main_fig.add_trace(go.Scatter(
        x=updated_df['time'], y=updated_df['Bollinger_Upper'],
        line=dict(color='red', width=1),
        name='Bollinger Upper'
    ))
    main_fig.add_trace(go.Scatter(
        x=updated_df['time'], y=updated_df['Bollinger_Lower'],
        line=dict(color='green', width=1),
        name='Bollinger Lower'
    ))
    main_fig.add_trace(go.Scatter(
        x=updated_df['time'], y=updated_df['SMA'],
        line=dict(color='blue', width=1.5),
        name='SMA'
    ))
    main_fig.update_layout(title="BTC/USDT with Bollinger Bands", xaxis_title="Time", yaxis_title="Price")

    # نمودار RSI
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=updated_df['time'], y=updated_df['RSI'],
        line=dict(color='purple', width=1.5),
        name='RSI'
    ))
    rsi_fig.update_layout(title="RSI Indicator", xaxis_title="Time", yaxis_title="RSI")

    # نمودار MACD
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(
        x=updated_df['time'], y=updated_df['MACD'],
        line=dict(color='orange', width=1.5),
        name='MACD'
    ))
    macd_fig.add_trace(go.Scatter(
        x=updated_df['time'], y=updated_df['Signal'],
        line=dict(color='blue', width=1.5),
        name='Signal'
    ))
    macd_fig.update_layout(title="MACD Indicator", xaxis_title="Time", yaxis_title="MACD")

    return main_fig, rsi_fig, macd_fig

# اجرای سرور
if __name__ == '__main__':
    app.run_server(debug=True)
