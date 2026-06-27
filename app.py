import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Smart Stock Assistant", layout="wide")
st.title("📊 Smart Stock Assistant (Stable + Auto Understanding)")

# =========================
# SMART TICKER RESOLVER
# =========================
def extract_ticker(user_input):
    text = user_input.lower().strip()

    # Case 1: already ticker-like
    if re.match(r"^[a-zA-Z.]{1,10}$", text) and " " not in text:
        return text.upper()

    # Case 2: try Yahoo Finance search fallback
    try:
        data = yf.Ticker(text)
        info = data.info

        symbol = info.get("symbol", None)
        if symbol:
            return symbol
    except:
        pass

    # Case 3: direct download test
    try:
        df = yf.download(text, period="5d", progress=False)
        if df is not None and not df.empty:
            return text.upper()
    except:
        pass

    return None


# =========================
# DATA FETCH
# =========================
def get_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    except:
        return None


# =========================
# INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()

    if len(df) < 50:
        return None

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    std = df["Close"].rolling(20).std()
    df["BBU"] = df["MA20"] + (2 * std)
    df["BBL"] = df["MA20"] - (2 * std)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()


# =========================
# SCORE ENGINE
# =========================
def score_stock(df):
    last = df.iloc[-1]
    score = 0

    # RSI
    if last["RSI"] < 30:
        score += 2
    elif last["RSI"] > 70:
        score -= 2

    # Trend
    if last["Close"] > last["MA20"] > last["MA50"]:
        score += 3
    elif last["Close"] < last["MA20"] < last["MA50"]:
        score -= 3

    # Bollinger
    if last["Close"] < last["BBL"]:
        score += 1
    elif last["Close"] > last["BBU"]:
        score -= 1

    if score >= 3:
        return "BUY", score
    elif score <= -3:
        return "SELL", score
    else:
        return "HOLD", score


# =========================
# PLOT
# =========================
def plot_stock(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Close"], label="Close")
    ax.plot(df["MA20"], label="MA20")
    ax.plot(df["MA50"], label="MA50")
    ax.set_title(f"{ticker} Price Chart")
    ax.legend()
    st.pyplot(fig)


# =========================
# SINGLE ANALYSIS
# =========================
def analyze_stock(user_input):
    ticker = extract_ticker(user_input)

    if not ticker:
        st.error("❌ Could not understand stock name. Try: Tata, Infosys, AAPL, TSLA")
        return

    st.info(f"Detected Ticker: {ticker}")

    df = get_data(ticker)

    if df is None:
        st.error("❌ No stock data found")
        return

    df = add_indicators(df)

    if df is None or len(df) < 10:
        st.warning("⚠️ Not enough data for analysis")
        return

    rec, score = score_stock(df)

    col1, col2 = st.columns(2)
    col1.metric("Recommendation", rec)
    col2.metric("Score", score)

    plot_stock(df, ticker)


# =========================
# COMPARE STOCKS
# =========================
def compare_stocks(s1, s2):
    t1 = extract_ticker(s1)
    t2 = extract_ticker(s2)

    df1 = add_indicators(get_data(t1)) if t1 else None
    df2 = add_indicators(get_data(t2)) if t2 else None

    if df1 is None or df2 is None:
        st.error("❌ One or both stocks invalid or insufficient data")
        return

    r1, sc1 = score_stock(df1)
    r2, sc2 = score_stock(df2)

    st.subheader("📊 Comparison Result")

    col1, col2 = st.columns(2)

    with col1:
        st.write(t1)
        st.metric("Recommendation", r1)
        st.metric("Score", sc1)

    with col2:
        st.write(t2)
        st.metric("Recommendation", r2)
        st.metric("Score", sc2)

    winner = t1 if sc1 > sc2 else t2
    st.success(f"🏆 Better Stock: {winner}")


# =========================
# UI
# =========================
mode = st.radio("Choose Mode", ["Single Stock", "Compare Stocks"])

if mode == "Single Stock":
    user_input = st.text_input("Ask anything (e.g., 'Should I invest in Tata?', 'Infosys', 'AAPL')")

    if st.button("Analyze"):
        if user_input:
            analyze_stock(user_input)
        else:
            st.warning("Enter something")

else:
    a = st.text_input("Stock 1")
    b = st.text_input("Stock 2")

    if st.button("Compare"):
        if a and b:
            compare_stocks(a, b)
        else:
            st.warning("Enter both stocks")
