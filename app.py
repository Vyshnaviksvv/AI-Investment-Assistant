
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Smart Stock Assistant", layout="wide")
st.title("📊 Smart Stock Assistant (FINAL STABLE VERSION)")

# =========================
# STOCK MAP
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
# EXTRACT STOCK (SAFE NLP)
# =========================
def extract_stock(text):
    text = text.lower()

    for k, v in STOCK_MAP.items():
        if k in text:
            return v, k

    # fallback ticker detection
    words = re.findall(r"[A-Za-z]{1,6}", text.upper())
    for w in words:
        if len(w) <= 6:
            return w, w

    return None, None


# =========================
# FETCH DATA (SAFE)
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
# INDICATORS (SAFE)
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

    df = df.dropna()

    return df


# =========================
# SAFE DECISION ENGINE (FIXED CRASH)
# =========================
def decision(df):
    last = df.iloc[-1]

    rsi = last.get("RSI", np.nan)
    close = last.get("Close", np.nan)
    ma20 = last.get("MA20", np.nan)
    ma50 = last.get("MA50", np.nan)

    # SAFE CHECK
    if np.isnan(rsi) or np.isnan(close) or np.isnan(ma20) or np.isnan(ma50):
        return "HOLD", 50

    score = 0

    # RSI logic
    if rsi < 30:
        score += 30
    elif rsi > 70:
        score -= 30

    # trend logic
    if close > ma20 > ma50:
        score += 40
    elif close < ma20 < ma50:
        score -= 40

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
# SINGLE ANALYSIS
# =========================
def analyze(text):
    ticker, name = extract_stock(text)

    if ticker is None:
        st.error("❌ Stock not understood. Try Tata, Infosys, Apple, Tesla, AAPL")
        return

    st.success(f"Detected: {name} → {ticker}")

    df = get_data(ticker)

    if df is None:
        st.error("❌ No data found for this stock.")
        return

    df = indicators(df)

    if df is None or len(df) < 20:
        st.error("❌ Not enough data for analysis.")
        return

    rec, conf = decision(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Stock", ticker)
    col2.metric("Signal", rec)
    col3.metric("Confidence", f"{conf}%")

    st.subheader("📊 Price Chart")
    plot(df, ticker)

    st.subheader("📌 Insight")

    if rec == "BUY":
        st.success("📈 Bullish trend detected. Momentum positive.")
    elif rec == "SELL":
        st.error("📉 Bearish trend detected. Risk high.")
    else:
        st.warning("⚖️ Mixed signals. Wait for confirmation.")


# =========================
# COMPARE STOCKS
# =========================
def compare(a, b):
    t1, n1 = extract_stock(a)
    t2, n2 = extract_stock(b)

    if t1 is None or t2 is None:
        st.error("❌ Could not understand stocks")
        return

    df1 = indicators(get_data(t1))
    df2 = indicators(get_data(t2))

    if df1 is None or df2 is None:
        st.error("❌ Not enough data for comparison")
        return

    r1, _ = decision(df1)
    r2, _ = decision(df2)

    st.subheader("📊 Comparison Result")
    st.write(f"{t1}: {r1}")
    st.write(f"{t2}: {r2}")

    col1, col2 = st.columns(2)

    with col1:
        st.write(n1)
        plot(df1, t1)

    with col2:
        st.write(n2)
        plot(df2, t2)


# =========================
# UI
# =========================
mode = st.radio("Select Mode", ["Single Stock", "Compare Stocks"])

if mode == "Single Stock":
    q = st.text_input("Ask anything (e.g., 'Should I invest in Apple')")

    if st.button("Analyze"):
        if q.strip():
            analyze(q)
        else:
            st.warning("Enter a query")

else:
    s1 = st.text_input("Stock 1")
    s2 = st.text_input("Stock 2")

    if st.button("Compare"):
        if s1 and s2:
            compare(s1, s2)
        else:
            st.warning("Enter both stocks")
