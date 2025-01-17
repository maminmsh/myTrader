import requests
import pandas as pd
import numpy as np
import tweepy
from textblob import TextBlob
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
import threading

# ======================== CONFIGURATION ========================
COINMARKETCAP_API_KEY = 'your_coinmarketcap_api_key'
TWITTER_API_KEY = 'your_twitter_api_key'
TWITTER_API_SECRET = 'your_twitter_api_secret'
TWITTER_ACCESS_TOKEN = 'your_twitter_access_token'
TWITTER_ACCESS_SECRET = 'your_twitter_access_secret'

app = Flask(__name__)

# ======================== DATA COLLECTION ========================
# Fetch market data from CoinMarketCap

def fetch_market_data(crypto_symbol):
    url = f'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {
        'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY
    }
    params = {
        'symbol': crypto_symbol,
        'convert': 'USD'
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    return data

# Fetch tweets for sentiment analysis

def fetch_tweets(keyword, count=100):
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
    api = tweepy.API(auth)
    tweets = api.search_tweets(q=keyword, lang="en", count=count)
    return [tweet.text for tweet in tweets]

# ======================== ANALYSIS MODULES ========================
# Fundamental Analysis

def fundamental_analysis(data):
    market_cap = data['quote']['USD']['market_cap']
    volume_24h = data['quote']['USD']['volume_24h']
    price_change_24h = data['quote']['USD']['percent_change_24h']

    analysis = {
        'Market Cap': market_cap,
        '24h Volume': volume_24h,
        '24h Price Change (%)': price_change_24h
    }
    return analysis

# Sentiment Analysis

def sentiment_analysis(tweets):
    sentiments = []
    for tweet in tweets:
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments.append('Positive')
        elif polarity < 0:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    sentiment_counts = pd.Series(sentiments).value_counts()
    sentiment_summary = {
        'Positive': sentiment_counts.get('Positive', 0),
        'Negative': sentiment_counts.get('Negative', 0),
        'Neutral': sentiment_counts.get('Neutral', 0)
    }
    return sentiment_summary

# Money Management

def money_management_strategy(balance, allocation, risk_level):
    # Example strategy: invest a portion based on risk level
    allocation_amount = balance * allocation
    risk_adjusted_allocation = allocation_amount * risk_level
    return {
        'Total Balance': balance,
        'Allocation Amount': allocation_amount,
        'Risk Adjusted Allocation': risk_adjusted_allocation
    }

# ======================== WEB SERVER ========================
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    crypto_symbol = data.get('crypto_symbol', 'BTC')
    allocation = data.get('allocation', 0.1)
    risk_level = data.get('risk_level', 0.5)
    balance = data.get('balance', 1000)

    market_data = fetch_market_data(crypto_symbol)
    market_analysis = fundamental_analysis(market_data['data'][crypto_symbol])

    tweets = fetch_tweets(crypto_symbol, count=100)
    sentiment_summary = sentiment_analysis(tweets)

    money_management = money_management_strategy(balance, allocation, risk_level)

    result = {
        'Fundamental Analysis': market_analysis,
        'Sentiment Analysis': sentiment_summary,
        'Money Management': money_management
    }
    return jsonify(result)

# ======================== MAIN ========================
if __name__ == '__main__':
    app.run(debug=True, port=5000)