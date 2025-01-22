import ccxt
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output , State

# تنظیم پروکسی
proxies = {
    'http': 'socks5h://127.0.0.1:400',
    'https': 'socks5h://127.0.0.1:400',
}

# گرفتن داده‌ها از بایننس
def fetch_data(symbol='BTC/USDT', timeframe='5m', limit=500):
    exchange = ccxt.binance({'proxies': proxies})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# محاسبه Stochastic
def calculate_stochastic(df, period=14):
    df['Stoch_K'] = 100 * ((df['close'] - df['low'].rolling(window=period).min()) /
                           (df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    return df

# محاسبه ATR
def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

# اضافه کردن به نمودار
def plot_stochastic_and_atr(df, fig):
    # نمودار Stochastic
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['Stoch_K'],
            line=dict(color='blue', width=1.5),
            name='Stochastic %K'
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['Stoch_D'],
            line=dict(color='orange', width=1.5),
            name='Stochastic %D'
        ),
        row=4, col=1
    )
    
    # نمودار ATR
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['ATR'],
            line=dict(color='purple', width=1.5),
            name='ATR'
        ),
        row=5, col=1
    )
    return fig

# اضافه کردن توضیحات برای Stochastic و ATR
def get_stochastic_atr_signals(df):
    signals = []
    # Stochastic Signal
    if df['Stoch_K'].iloc[-1] > 80:
        signals.append("Stochastic is overbought (>80) - Possible reversal.")
    elif df['Stoch_K'].iloc[-1] < 20:
        signals.append("Stochastic is oversold (<20) - Possible buying opportunity.")

    # ATR Signal
    if df['ATR'].iloc[-1] > df['ATR'].mean():
        signals.append("ATR indicates high volatility.")
    else:
        signals.append("ATR indicates low volatility.")

    return "\n".join(signals)

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

# تشخیص سیگنال خرید و فروش
def detect_signals(df):
    df['RSI_Buy'] = (df['RSI'] < 30)
    df['RSI_Sell'] = (df['RSI'] > 70)
    df['MACD_Buy'] = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
    df['MACD_Sell'] = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
    df['Buy_Signal'] = df['RSI_Buy'] & df['MACD_Buy']
    df['Sell_Signal'] = df['RSI_Sell'] & df['MACD_Sell']
    return df

# آماده‌سازی داده‌ها و شاخص‌ها
def calculate_indicators(df, sma_period=20, rsi_period=14):
    df = calculate_bollinger_bands(df, period=sma_period)
    df = calculate_rsi(df, period=rsi_period)
    df = calculate_macd(df)
    df = detect_signals(df)
    df = calculate_stochastic(df)
    df = calculate_atr(df)
    return df

# ایجاد نمودار اصلی
def create_main_chart(df):
    fig = go.Figure()

    # نمودار کندل‌استیک
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Candlestick'
    ))

    # باندهای بولینگر
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

    # سیگنال خرید
    buy_signals = df[df['Buy_Signal']]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['time'], y=buy_signals['close'],
            mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Buy Signal'
        ))

    # سیگنال فروش
    sell_signals = df[df['Sell_Signal']]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['time'], y=sell_signals['close'],
            mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Sell Signal'
        ))

    fig.update_layout(title="BTC/USDT Price + Indicators", xaxis_title="Time", yaxis_title="Price")
    return fig

# ایجاد نمودار RSI
def create_rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'))
    fig.update_layout(title="RSI", xaxis_title="Time", yaxis_title="RSI")
    return fig

# ایجاد نمودار MACD
def create_macd_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], line=dict(color='orange', width=1.5), name='MACD'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['Signal'], line=dict(color='blue', width=1.5), name='Signal'))
    fig.update_layout(title="MACD", xaxis_title="Time", yaxis_title="MACD")
    return fig

# داده‌های اولیه
df = fetch_data()
df = calculate_indicators(df)

# تنظیم Dash
app = Dash(__name__)
# اضافه کردن به layout
app.layout = html.Div([
    html.Div([
        html.H3("Signal Descriptions:"),
        dcc.Markdown(id='signal-description', style={'whiteSpace': 'pre-line', 'margin-top': '10px'})
    ], style={'margin-top': '30px'}),
    html.Div([
        html.Label("SMA Period:"),
        dcc.Input(id='sma-period', type='number', value=20, min=1),
        html.Label("RSI Period:"),
        dcc.Input(id='rsi-period', type='number', value=14, min=1),
    ], style={'margin-bottom': '20px'}),
     dcc.Interval(
        id='refresh-interval',
        interval=0.5*60000,  # هر ۵ دقیقه (بر حسب میلی‌ثانیه)
        n_intervals=0     # تعداد دفعاتی که این تایمر اجرا شده است
    ),
    dcc.Graph(id='main-chart'),
    dcc.Graph(id='rsi-chart'),
    dcc.Graph(id='macd-chart'),
    dcc.Graph(id='stochastic-atr-chart')
])


def generate_signal_description(df):
    descriptions = []

    # بررسی سیگنال‌های خرید و فروش
    buy_signals = df[df['Buy_Signal']]
    sell_signals = df[df['Sell_Signal']]

    if not buy_signals.empty:
        latest_buy = buy_signals.iloc[-1]
        descriptions.append(f"💚 **Buy Signal** detected at {latest_buy['time']} with price {latest_buy['close']:.2f}.")

    if not sell_signals.empty:
        latest_sell = sell_signals.iloc[-1]
        descriptions.append(f"❤️ **Sell Signal** detected at {latest_sell['time']} with price {latest_sell['close']:.2f}.")

    # بررسی وضعیت RSI
    latest_rsi = df['RSI'].iloc[-1]
    if latest_rsi > 70:
        descriptions.append(f"📈 RSI is above 70 ({latest_rsi:.2f}), indicating overbought conditions.")
    elif latest_rsi < 30:
        descriptions.append(f"📉 RSI is below 30 ({latest_rsi:.2f}), indicating oversold conditions.")
    else:
        descriptions.append(f"⚪ RSI is neutral ({latest_rsi:.2f}).")

    # بررسی وضعیت باندهای بولینگر
    latest_price = df['close'].iloc[-1]
    upper_band = df['Bollinger_Upper'].iloc[-1]
    lower_band = df['Bollinger_Lower'].iloc[-1]
    if latest_price > upper_band:
        descriptions.append(f"🚨 Price is above the upper Bollinger Band ({latest_price:.2f}), potential reversal.")
    elif latest_price < lower_band:
        descriptions.append(f"🚨 Price is below the lower Bollinger Band ({latest_price:.2f}), potential reversal.")
    else:
        descriptions.append(f"📊 Price is within Bollinger Bands.")

    # بررسی وضعیت SMA
    sma = df['SMA'].iloc[-1]
    if latest_price > sma:
        descriptions.append(f"✅ Price ({latest_price:.2f}) is above SMA ({sma:.2f}), indicating bullish trend.")
    else:
        descriptions.append(f"❌ Price ({latest_price:.2f}) is below SMA ({sma:.2f}), indicating bearish trend.")

    # بررسی وضعیت MACD
    latest_macd = df['MACD'].iloc[-1]
    latest_signal = df['Signal'].iloc[-1]
    if latest_macd > latest_signal:
        descriptions.append(f"📗 MACD ({latest_macd:.2f}) is above Signal ({latest_signal:.2f}), indicating bullish momentum.")
    elif latest_macd < latest_signal:
        descriptions.append(f"📕 MACD ({latest_macd:.2f}) is below Signal ({latest_signal:.2f}), indicating bearish momentum.")
    else:
        descriptions.append(f"⚪ MACD and Signal are equal ({latest_macd:.2f}).")

    # بررسی وضعیت ATR
    latest_atr = df['ATR'].iloc[-1]
    atr_mean = df['ATR'].mean()
    if latest_atr > atr_mean:
        descriptions.append(f"📈 ATR ({latest_atr:.2f}) is above its average ({atr_mean:.2f}), indicating high volatility.")
    else:
        descriptions.append(f"📉 ATR ({latest_atr:.2f}) is below its average ({atr_mean:.2f}), indicating low volatility.")

    # بررسی وضعیت Stochastic
    latest_stoch_k = df['Stoch_K'].iloc[-1]
    if latest_stoch_k > 80:
        descriptions.append(f"📈 Stochastic %K ({latest_stoch_k:.2f}) is above 80, indicating overbought conditions.")
    elif latest_stoch_k < 20:
        descriptions.append(f"📉 Stochastic %K ({latest_stoch_k:.2f}) is below 20, indicating oversold conditions.")
    else:
        descriptions.append(f"⚪ Stochastic %K ({latest_stoch_k:.2f}) is neutral.")

    # اگر توضیحاتی وجود نداشت
    if not descriptions:
        descriptions.append("⚪ No active signals or indicators detected at this time.")

    return descriptions


def create_stochastic_and_atr_chart(df):
    fig = go.Figure()

    # Stochastic نمودار
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['Stoch_K'],
        line=dict(color='blue', width=1.5),
        name='Stochastic %K'
    ))
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['Stoch_D'],
        line=dict(color='orange', width=1.5),
        name='Stochastic %D'
    ))

    # ATR نمودار
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['ATR'],
        line=dict(color='purple', width=1.5),
        name='ATR'
    ))

    fig.update_layout(title="Stochastic & ATR", xaxis_title="Time", yaxis_title="Value")
    return fig

# اضافه کردن توضیحات مربوط به Stochastic و ATR
def generate_stochastic_atr_signals(df):
    signals = []

    # سیگنال‌های Stochastic
    if df['Stoch_K'].iloc[-1] > 80:
        signals.append("📈 Stochastic %K is above 80, indicating overbought conditions.")
    elif df['Stoch_K'].iloc[-1] < 20:
        signals.append("📉 Stochastic %K is below 20, indicating oversold conditions.")

    # ATR وضعیت
    atr = df['ATR'].iloc[-1]
    atr_mean = df['ATR'].mean()
    if atr > atr_mean:
        signals.append(f"⚡ ATR ({atr:.2f}) is above average ({atr_mean:.2f}), indicating high volatility.")
    else:
        signals.append(f"💤 ATR ({atr:.2f}) is below average ({atr_mean:.2f}), indicating low volatility.")

    return signals


# Callback برای به‌روزرسانی نمودارها
@app.callback(
    [Output('main-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('macd-chart', 'figure'),
     Output('signal-description', 'children'),
     Output('stochastic-atr-chart', 'figure')],
  [Input('refresh-interval', 'n_intervals')], 
          [State('sma-period', 'value'),              # مقادیر SMA و RSI به عنوان State
     State('rsi-period', 'value')]
    # [Input('sma-period', 'value'),
    #  Input('rsi-period', 'value')]
)
def update_charts_and_description(n_intervals,sma_period, rsi_period):
    # محاسبه اندیکاتورها
    updated_df = calculate_indicators(df, sma_period=sma_period, rsi_period=rsi_period)

    # تولید توضیحات سیگنال‌ها
    signal_descriptions = generate_signal_description(updated_df)
    stochastic_atr_signals = generate_stochastic_atr_signals(updated_df)

    # تولید توضیحات سیگنال‌ها
    signal_descriptions = generate_signal_description(updated_df)

    # تبدیل لیست توضیحات به متن Markdown
    description_text = "\n\n".join(signal_descriptions)

    # به‌روزرسانی نمودارها و توضیحات
    return (
        create_main_chart(updated_df),
        create_rsi_chart(updated_df),
        create_macd_chart(updated_df),
        description_text,
        create_stochastic_and_atr_chart(updated_df)
    )

# اجرای برنامه
if __name__ == '__main__':
    app.run_server(debug=True)
