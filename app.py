import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Smart Stock AI", layout="wide")
st.title("📊 Smart Stock Assistant (ULTIMATE STABLE VERSION)")

# =========================
# STOCK DATABASE
# =========================
STOCK_MAP = {
    "tata": "TATAMOTORS.NS",
    "tata motors": "TATAMOTORS.NS",
    "infosys": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "apple": "AAPL",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "nvidia": "NVDA"
}

# =========================
# SMART STOCK EXTRACTION (FIXED)
# =========================
def extract_stock(text):
    text = text.lower()

    # direct keyword match
    for key in STOCK_MAP:
        if key in text:
            return STOCK_MAP[key], key

    # ticker detection (AAPL, TSLA etc)
    words = re.findall(r"[A-Za-z]{1,6}", text.upper())
    for w in words:
        if len(w) <= 6:
            return w, w

    return None, None


# =========================
# FETCH DATA
# =========================
def get_data(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# =========================
# INDICATORS
# =========================
def indicators(df):
    df = df.copy()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()


# =========================
# AI-FREE DECISION ENGINE
# =========================
def decision(df):
    last = df.iloc[-1]

    score = 0

    # RSI logic
    if last["RSI"] < 30:
        score += 30
    elif last["RSI"] > 70:
        score -= 30

    # trend logic
    if last["Close"] > last["MA20"] > last["MA50"]:
        score += 40
    elif last["Close"] < last["MA20"] < last["MA50"]:
        score -= 40

    # final decision
    if score > 25:
        return "BUY", min(95, 50 + abs(score))
    elif score < -25:
        return "SELL", min(95, 50 + abs(score))
    else:
        return "HOLD", min(80, 40 + abs(score))


# =========================
# PLOT
# =========================
def plot(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["MA20"], label="MA20")
    ax.plot(df.index, df["MA50"], label="MA50")
    ax.set_title(ticker)
    ax.legend()
    st.pyplot(fig)


# =========================
# MAIN ANALYSIS
# =========================
def analyze(text):
    ticker, name = extract_stock(text)

    if ticker is None:
        st.error("❌ Stock not understood. Try: Tata, Infosys, Apple, Tesla")
        return

    st.success(f"Detected: {name} → {ticker}")

    df = get_data(ticker)

    if df is None:
        st.error("❌ No market data found for this stock.")
        return

    df = indicators(df)

    if df is None or len(df) < 50:
        st.error("❌ Not enough data for analysis.")
        return

    rec, conf = decision(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Stock", ticker)
    col2.metric("Signal", rec)
    col3.metric("Confidence", f"{conf}%")

    st.subheader("📊 Price Chart")
    plot(df, ticker)

    st.subheader("📌 Simple Insight")

    if rec == "BUY":
        st.write("📈 Trend is bullish. Stock showing upward momentum.")
    elif rec == "SELL":
        st.write("📉 Weak trend. Risk of downside movement.")
    else:
        st.write("⚖️ Mixed signals. Wait for clear trend.")


# =========================
# UI
# =========================
query = st.text_input("Ask anything (e.g., 'Should I invest in Tata?', 'Infosys', 'AAPL')")

if st.button("Analyze"):
    if query.strip():
        analyze(query)
    else:
        st.warning("Enter a query")
