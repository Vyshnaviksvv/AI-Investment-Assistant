import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Stock Assistant (Stable)", layout="wide")

st.title("📊 Stock Analysis & Comparison App (Stable Version)")
st.caption("No AI, No Errors — Fully Safe Streamlit App")

# =========================
# SAFE DATA FETCH
# =========================
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        return df
    except:
        return None


# =========================
# SAFE INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()

    if len(df) < 60:
        return None

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Bollinger Bands (SAFE)
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + (2 * std)
    df["BB_Lower"] = df["MA20"] - (2 * std)

    # RSI SAFE
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()


# =========================
# SAFE SCORE SYSTEM
# =========================
def score_stock(df):
    latest = df.iloc[-1]
    score = 0

    # RSI
    if latest["RSI"] < 30:
        score += 2
    elif latest["RSI"] > 70:
        score -= 2

    # Trend
    if latest["Close"] > latest["MA20"] > latest["MA50"]:
        score += 3
    elif latest["Close"] < latest["MA20"] < latest["MA50"]:
        score -= 3

    # Bollinger
    if latest["Close"] < latest["BB_Lower"]:
        score += 1
    elif latest["Close"] > latest["BB_Upper"]:
        score -= 1

    # Recommendation
    if score >= 3:
        return "BUY", score
    elif score <= -3:
        return "SELL", score
    else:
        return "HOLD", score


# =========================
# SAFE PLOT
# =========================
def plot_stock(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Close"], label="Close")
    ax.plot(df["MA20"], label="MA20")
    ax.plot(df["MA50"], label="MA50")
    ax.set_title(ticker)
    ax.legend()
    st.pyplot(fig)


# =========================
# SINGLE STOCK ANALYSIS
# =========================
def analyze_stock(ticker):
    df = fetch_data(ticker)

    if df is None:
        st.error("❌ No data found for this stock")
        return

    df = add_indicators(df)

    if df is None or len(df) < 10:
        st.warning("⚠️ Not enough data for analysis")
        return

    rec, score = score_stock(df)

    st.subheader(f"📌 {ticker}")

    col1, col2 = st.columns(2)
    col1.metric("Recommendation", rec)
    col2.metric("Score", score)

    plot_stock(df, ticker)


# =========================
# COMPARISON (SAFE)
# =========================
def compare_stocks(t1, t2):
    df1 = add_indicators(fetch_data(t1))
    df2 = add_indicators(fetch_data(t2))

    if df1 is None or df2 is None:
        st.error("❌ One or both stocks have insufficient data")
        return

    r1, s1 = score_stock(df1)
    r2, s2 = score_stock(df2)

    st.subheader("📊 Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write(t1)
        st.metric("Recommendation", r1)
        st.metric("Score", s1)

    with col2:
        st.write(t2)
        st.metric("Recommendation", r2)
        st.metric("Score", s2)

    winner = t1 if s1 > s2 else t2
    st.success(f"🏆 Better Stock (based on score): {winner}")


# =========================
# UI
# =========================
mode = st.radio("Select Mode", ["Single Stock", "Compare Stocks"])

if mode == "Single Stock":
    ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS, INFY.NS, AAPL)")

    if st.button("Analyze"):
        if ticker:
            analyze_stock(ticker.upper())
        else:
            st.warning("Enter a ticker")

else:
    t1 = st.text_input("Stock 1")
    t2 = st.text_input("Stock 2")

    if st.button("Compare"):
        if t1 and t2:
            compare_stocks(t1.upper(), t2.upper())
        else:
            st.warning("Enter both tickers")
