import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils import create_plots_dir, save_plot

# 1. دریافت داده‌های قیمت بیت‌کوین
def get_bitcoin_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "30",
        "interval": "hourly"
    }
    response = requests.get(url, params=params)
    data = response.json()
    print("Response from API:", data)  # چاپ پاسخ API برای دیباگ
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# 2. پیش‌پردازش داده‌ها
def preprocess_data(df):
    df["price_change"] = df["price"].pct_change()  # تغییرات قیمت
    df["price_change"] = df["price_change"].fillna(0)
    df["price_moving_avg"] = df["price"].rolling(window=24).mean()  # میانگین متحرک 24 ساعت
    df = df.dropna()
    return df

# 3. نمایش و ذخیره نمودارها
def plot_and_save(df, output_dir):
    # نمودار قیمت
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["price"], label="Actual Price", color="blue")
    plt.plot(df["timestamp"], df["price_moving_avg"], label="24-Hour Moving Avg", linestyle="--", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Analysis (Last 30 Days)")
    plt.legend()
    save_plot(plt, output_dir, "bitcoin_price_analysis.png")

# اجرای برنامه
if __name__ == "__main__":
    # 1. دریافت و پیش‌پردازش داده‌ها
    df = get_bitcoin_data()
    df = preprocess_data(df)

    # 2. ایجاد پوشه ذخیره نمودارها
    output_dir = create_plots_dir()

    # 3. نمایش و ذخیره نمودارها
    plot_and_save(df, output_dir)

    print(f"Plots saved to: {output_dir}")
