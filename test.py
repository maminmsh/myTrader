import ccxt

# تنظیمات پروکسی
proxies = {
    'http': 'socks5h://127.0.0.1:400',  # پروکسی HTTP
    'https': 'socks5h://127.0.0.1:400', # پروکسی HTTPS
}

# ایجاد نمونه از بایننس با پروکسی
exchange = ccxt.binance({
    'proxies': proxies,  # تنظیم پروکسی
    # 'timeout': 30000,    # تنظیم تایم‌اوت (اختیاری)
})

# دریافت اطلاعات بازار
try:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=1000)  # تایم‌فریم: ۱ دقیقه
    print(ohlcv)
except Exception as e:
    print(f"Error: {e}")
