import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Investment Assistant", layout="wide")
st.title("🤖 AI Investment Assistant (Stable No-Error Version)")

# =========================
# SAFE DATA FETCH
# =========================
def fetch_stock_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True)

        # Fix MultiIndex issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        if df.empty:
            return None

        return df

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None


# =========================
# INDICATORS (SAFE)
# =========================
def calculate_indicators(df):
    df = df.copy()

    df = df.select_dtypes(include=[np.number])

    if len(df) < 50:
        return df

    # Moving averages
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # Bollinger Bands
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + (2 * std)
    df["BB_Lower"] = df["MA20"] - (2 * std)

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df.dropna()


# =========================
# SIMPLE AI SCORE ENGINE
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
        reasons.append("Strong uptrend (Golden crossover structure)")
    elif latest["Close"] < latest["MA50"] < latest["MA200"]:
        score -= 35
        reasons.append("Strong downtrend")

    # MACD
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 15
        reasons.append("MACD bullish momentum")
    else:
        score -= 15
        reasons.append("MACD bearish momentum")

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
# PLOT
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
# REPORT
# =========================
def generate_report(ticker, df, decision, confidence, reasons):
    price = df["Close"].iloc[-1]

    risk = "Low" if confidence > 75 else "Medium" if confidence > 60 else "High"

    text = f"""
### 📊 Stock Report: {ticker}

**Price:** {price:.2f}  
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

⚠️ This is rule-based analysis, not financial advice.
"""

    return text


# =========================
# SIMPLE TICKER MAP
# =========================
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


# =========================
# UI
# =========================
query = st.text_input("Ask (e.g., Should I invest in Infosys?)")

if st.button("Analyze"):

    if not query:
        st.warning("Please enter a query")
    else:
        st.info("Processing...")

        query_lower = query.lower()

        ticker = None
        for k, v in ticker_map.items():
            if k in query_lower:
                ticker = v
                break

        if not ticker:
            st.error("Stock not recognized. Try Infosys, Tata, Reliance, Apple etc.")
            st.stop()

        df = fetch_stock_data(ticker)

        if df is None or len(df) < 50:
            st.error("Not enough data available")
            st.stop()

        df = calculate_indicators(df)

        decision, confidence, reasons = analyze_stock(df)

        st.subheader(ticker)

        st.metric("Recommendation", decision)
        st.metric("Confidence", f"{confidence}%")

        plot_chart(df, ticker)

        st.markdown(generate_report(ticker, df, decision, confidence, reasons))
