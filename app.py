import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Investment Assistant", layout="wide")

st.title("🤖 AI Investment Assistant (Stable Version)")

# =========================
# STOCK DATA
# =========================
def fetch_stock_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True)
        if df.empty:
            return None
        return df
    except:
        return None


# =========================
# INDICATORS
# =========================
def calculate_indicators(df):
    df = df.copy()

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    df["MA20"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + 2 * std
    df["BB_Lower"] = df["MA20"] - 2 * std

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=14).mean()
    avg_loss = loss.ewm(span=14).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

    return df


# =========================
# RULE-BASED AI ENGINE
# =========================
def analyze_stock(df):
    latest = df.iloc[-1]

    score = 0
    reasons = []

    # RSI
    if latest["RSI"] < 30:
        score += 30
        reasons.append("RSI oversold → bullish")
    elif latest["RSI"] > 70:
        score -= 30
        reasons.append("RSI overbought → bearish")

    # Trend
    if latest["Close"] > latest["MA50"] > latest["MA200"]:
        score += 35
        reasons.append("Strong uptrend (Golden structure)")
    elif latest["Close"] < latest["MA50"] < latest["MA200"]:
        score -= 35
        reasons.append("Strong downtrend")

    # MACD
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 15
        reasons.append("MACD bullish crossover")
    else:
        score -= 15
        reasons.append("MACD bearish signal")

    # Decision
    if score >= 25:
        decision = "BUY"
    elif score <= -25:
        decision = "SELL"
    else:
        decision = "HOLD"

    confidence = min(95, max(50, abs(score) + 50))

    return decision, confidence, reasons


# =========================
# CHART
# =========================
def plot_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df.index[-120:], df["Close"][-120:], label="Close")
    ax.plot(df.index[-120:], df["MA50"][-120:], label="MA50")
    ax.plot(df.index[-120:], df["MA200"][-120:], label="MA200")

    ax.set_title(f"{ticker} Technical Chart")
    ax.legend()

    st.pyplot(fig)


# =========================
# SIMPLE STOCK INSIGHT TEXT
# =========================
def generate_report(ticker, df, decision, confidence, reasons):
    price = df["Close"].iloc[-1]

    risk = "High" if confidence < 60 else "Medium" if confidence < 80 else "Low"

    text = f"""
### 📊 Investment Report: {ticker}

**Current Price:** {price:.2f}  
**Recommendation:** {decision}  
**Confidence:** {confidence}%  
**Risk Level:** {risk}

---

### 📌 Key Signals
"""

    for r in reasons:
        text += f"- {r}\n"

    text += """
---

### ⚠️ Disclaimer
This is a rule-based analysis, not financial advice.
"""

    return text


# =========================
# MAIN APP LOGIC
# =========================
query = st.text_input("Ask something (e.g., Should I invest in Infosys?)")

if st.button("Analyze"):

    if not query:
        st.warning("Enter a query")
    else:
        st.info("Processing...")

        # simple extraction (NO AI)
        query_lower = query.lower()

        ticker_map = {
            "infosys": "INFY.NS",
            "tata": "TATAMOTORS.NS",
            "tcs": "TCS.NS",
            "reliance": "RELIANCE.NS",
            "hdfc": "HDFCBANK.NS",
            "apple": "AAPL",
            "tesla": "TSLA",
            "microsoft": "MSFT"
        }

        ticker = None
        for name, tk in ticker_map.items():
            if name in query_lower:
                ticker = tk
                break

        if not ticker:
            st.error("Stock not recognized. Try Infosys, Tata, Reliance etc.")
            st.stop()

        df = fetch_stock_data(ticker)

        if df is None:
            st.error("Data not available")
            st.stop()

        df = calculate_indicators(df)

        decision, confidence, reasons = analyze_stock(df)

        st.subheader(f"{ticker}")

        st.metric("Recommendation", decision)
        st.metric("Confidence", f"{confidence}%")

        plot_chart(df, ticker)

        st.markdown(generate_report(ticker, df, decision, confidence, reasons))
