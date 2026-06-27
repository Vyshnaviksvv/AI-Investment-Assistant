import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Smart Stock Assistant", layout="wide")

st.title("📊 Smart Stock Assistant (Stable No-Error Version)")

# =========================
# STOCK MAP (IMPORTANT)
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
# RESOLVE STOCK NAME
# =========================
def resolve_stock(name: str):
    name = name.lower().strip()
    if name in STOCK_MAP:
        return STOCK_MAP[name]

    # direct ticker fallback
    return name.upper()


# =========================
# FETCH DATA
# =========================
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# =========================
# INDICATORS (SAFE)
# =========================
def add_indicators(df):
    df = df.copy()

    if len(df) < 50:
        return None

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

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

    # RSI
    if last["RSI"] < 30:
        score += 25
    elif last["RSI"] > 70:
        score -= 25

    # Trend
    if last["Close"] > last["MA20"] > last["MA50"]:
        score += 30
    elif last["Close"] < last["MA20"] < last["MA50"]:
        score -= 30

    # Decision
    if score > 20:
        return "BUY", min(95, abs(score) + 40)
    elif score < -20:
        return "SELL", min(95, abs(score) + 40)
    else:
        return "HOLD", min(70, abs(score) + 30)


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
# SINGLE STOCK
# =========================
def single_stock(name):
    ticker = resolve_stock(name)

    st.info(f"Resolved: {name} → {ticker}")

    df = fetch_data(ticker)

    if df is None:
        st.error("❌ No data found. Try different stock name or ticker.")
        return

    df = add_indicators(df)

    if df is None:
        st.error("❌ Not enough data for analysis.")
        return

    rec, conf = analyze(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", ticker)
    col2.metric("Recommendation", rec)
    col3.metric("Confidence", f"{conf}%")

    plot(df, ticker)


# =========================
# COMPARE STOCKS
# =========================
def compare(s1, s2):
    t1 = resolve_stock(s1)
    t2 = resolve_stock(s2)

    st.info(f"{s1} → {t1} | {s2} → {t2}")

    df1 = add_indicators(fetch_data(t1))
    df2 = add_indicators(fetch_data(t2))

    if df1 is None or df2 is None:
        st.error("❌ Not enough data for comparison")
        return

    r1, _ = analyze(df1)
    r2, _ = analyze(df2)

    st.subheader("📊 Comparison Result")
    st.write(f"{t1}: {r1}")
    st.write(f"{t2}: {r2}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Stock 1")
        plot(df1, t1)

    with col2:
        st.write("Stock 2")
        plot(df2, t2)


# =========================
# UI
# =========================
mode = st.radio("Select Mode", ["Single Stock", "Compare Stocks"])

if mode == "Single Stock":
    query = st.text_input("Enter stock name (e.g., Tata, Infosys, AAPL)")

    if st.button("Analyze"):
        if query:
            single_stock(query)
        else:
            st.warning("Enter stock name")

else:
    s1 = st.text_input("Stock 1")
    s2 = st.text_input("Stock 2")

    if st.button("Compare"):
        if s1 and s2:
            compare(s1, s2)
        else:
            st.warning("Enter both stocks")
