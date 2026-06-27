import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Stock Assistant", layout="wide")

# =========================
# SAFE STOCK MAPPING
# =========================
STOCK_MAP = {
    "tata motors": "TATAMOTORS.NS",
    "tata": "TATAMOTORS.NS",
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
    "microsoft": "MSFT"
}

# =========================
# RESOLVE STOCK NAME SAFELY
# =========================
def resolve_stock(name: str):
    if not name:
        return None

    name = name.lower().strip()

    # direct ticker input
    if ".ns" in name or len(name) <= 6:
        return name.upper()

    # mapped name
    if name in STOCK_MAP:
        return STOCK_MAP[name]

    # fuzzy match fallback
    for k in STOCK_MAP:
        if k in name:
            return STOCK_MAP[k]

    return None

# =========================
# FETCH DATA (SAFE)
# =========================
def get_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        return df
    except:
        return None

# =========================
# INDICATORS (FIXED SAFE VERSION)
# =========================
def add_indicators(df):
    df = df.copy()

    if len(df) < 60:
        return None

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Bollinger Bands SAFE FIX
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + (2 * std)
    df["BB_Lower"] = df["MA20"] - (2 * std)

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()

# =========================
# SIGNAL ENGINE
# =========================
def analyze(df):
    last = df.iloc[-1]

    score = 0

    # RSI logic
    if last["RSI"] < 30:
        score += 25
    elif last["RSI"] > 70:
        score -= 25

    # Trend
    if last["Close"] > last["MA50"]:
        score += 15
    else:
        score -= 10

    # Bollinger
    if last["Close"] < last["BB_Lower"]:
        score += 20
    elif last["Close"] > last["BB_Upper"]:
        score -= 20

    if score >= 25:
        return "BUY", score
    elif score <= -20:
        return "SELL", score
    return "HOLD", score

# =========================
# PLOT
# =========================
def plot(df, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["MA20"], label="MA20")
    ax.plot(df.index, df["MA50"], label="MA50")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# =========================
# SINGLE STOCK
# =========================
def single_stock(user_input):
    ticker = resolve_stock(user_input)

    if not ticker:
        st.error("❌ Could not understand stock name. Try Tata, Infosys, AAPL, TSLA")
        return

    df = get_data(ticker)
    if df is None:
        st.error("❌ No data found for this stock")
        return

    df = add_indicators(df)
    if df is None:
        st.error("❌ Not enough data for analysis")
        return

    rec, score = analyze(df)

    st.subheader(f"{ticker}")
    st.metric("Recommendation", rec)
    st.metric("Score", score)

    plot(df, ticker)

# =========================
# COMPARE STOCKS
# =========================
def compare(stock1, stock2):
    t1 = resolve_stock(stock1)
    t2 = resolve_stock(stock2)

    if not t1 or not t2:
        st.error("❌ Could not understand stock names")
        return

    df1 = add_indicators(get_data(t1))
    df2 = add_indicators(get_data(t2))

    if df1 is None or df2 is None:
        st.error("❌ Not enough data for comparison")
        return

    r1, s1 = analyze(df1)
    r2, s2 = analyze(df2)

    st.subheader("📊 Comparison")
    st.write(f"{t1}: {r1} ({s1})")
    st.write(f"{t2}: {r2} ({s2})")

# =========================
# UI
# =========================
st.title("📊 Smart Stock Assistant (Stable No-Error Version)")

mode = st.radio("Select Mode", ["Single Stock", "Compare Stocks"])

if mode == "Single Stock":
    user_input = st.text_input("Enter stock name (e.g., Tata, Infosys, AAPL)")
    if st.button("Analyze"):
        single_stock(user_input)

else:
    c1 = st.text_input("Stock 1")
    c2 = st.text_input("Stock 2")

    if st.button("Compare"):
        compare(c1, c2)
