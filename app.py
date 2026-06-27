import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Investment Assistant", layout="wide")

# Gemini API Key
api_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning("Please set GEMINI_API_KEY in environment or Streamlit secrets.")
else:
    genai.configure(api_key=api_key)


# =========================
# GEMINI INTENT PARSER
# =========================
def resolve_tickers_and_intent(query):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Analyze query: "{query}"

    Return JSON with:
    intent: single_analysis | compare | sector_recommendation | general_qa
    companies: list
    tickers: max 2 tickers (NSE .NS or US symbols)

    Return ONLY JSON.
    """

    try:
        res = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(res.text)
    except:
        return {"intent": "general_qa", "companies": [], "tickers": []}


# =========================
# DATA FETCH
# =========================
def fetch_stock_data(ticker, period="1y"):
    df = yf.Ticker(ticker).history(period="2y")
    return df if not df.empty else None


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

    avg_gain = gain.ewm(com=13).mean()
    avg_loss = loss.ewm(com=13).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

    return df


# =========================
# TECHNICAL SCORE
# =========================
def generate_technical_recommendation(df):
    latest = df.iloc[-1]

    score = 0

    # RSI
    if latest["RSI"] < 30:
        score += 25
    elif latest["RSI"] > 70:
        score -= 25

    # Trend
    if latest["Close"] > latest["MA50"] > latest["MA200"]:
        score += 30
    elif latest["Close"] < latest["MA50"] < latest["MA200"]:
        score -= 30

    # MACD
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 15
    else:
        score -= 15

    # Decision
    if score > 20:
        rec = "BUY"
    elif score < -20:
        rec = "SELL"
    else:
        rec = "HOLD"

    return rec, int(min(95, max(20, abs(score))))


# =========================
# PLOT
# =========================
def plot_data(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index[-120:], df["Close"][-120:], label="Close")
    ax.plot(df.index[-120:], df["MA50"][-120:], label="MA50")
    ax.plot(df.index[-120:], df["MA200"][-120:], label="MA200")
    ax.set_title(f"{ticker} Price Chart")
    ax.legend()
    st.pyplot(fig)


# =========================
# AI REPORT
# =========================
def generate_ai_report(ticker, df, rec):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Stock: {ticker}
    Recommendation: {rec}
    Price: {df['Close'].iloc[-1]}

    Write:
    1. Summary
    2. Technical view
    3. Risk
    """

    res = model.generate_content(prompt)
    return res.text


# =========================
# MAIN ENGINE
# =========================
def process_query(query):
    data = resolve_tickers_and_intent(query)
    intent = data["intent"]
    tickers = data["tickers"]

    st.info(f"Intent: {intent}")

    if intent == "single_analysis" and tickers:
        ticker = tickers[0]

        df = fetch_stock_data(ticker)
        if df is None:
            st.error("No data found")
            return

        df = calculate_indicators(df)
        rec, conf = generate_technical_recommendation(df)

        st.subheader(f"{ticker}")
        st.metric("Recommendation", rec)
        st.metric("Confidence", f"{conf}%")

        plot_data(df, ticker)

        st.markdown(generate_ai_report(ticker, df, rec))

    elif intent == "compare" and len(tickers) >= 2:
        t1, t2 = tickers[:2]

        df1 = calculate_indicators(fetch_stock_data(t1))
        df2 = calculate_indicators(fetch_stock_data(t2))

        r1, _ = generate_technical_recommendation(df1)
        r2, _ = generate_technical_recommendation(df2)

        st.subheader("Comparison")
        st.write(t1, "→", r1)
        st.write(t2, "→", r2)

    else:
        model = genai.GenerativeModel("gemini-2.5-flash")
        res = model.generate_content(query)
        st.write(res.text)


# =========================
# UI
# =========================
st.title("🤖 AI Investment Assistant")

query = st.text_input("Ask your question")

if st.button("Analyze"):
    if query:
        process_query(query)
    else:
        st.warning("Enter a query")
