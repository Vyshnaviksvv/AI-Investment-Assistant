import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import google.generativeai as genai
from datetime import datetime

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="AI Investment Assistant", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash"

api_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠️ Add GEMINI_API_KEY to secrets or env")

# =========================
# SAFE GEMINI (NO CRASH)
# =========================
def safe_gemini(prompt):
    if not api_key:
        return "⚠️ AI disabled (no API key)"

    model = genai.GenerativeModel(GEMINI_MODEL)

    try:
        return model.generate_content(prompt).text

    except Exception as e:
        err = str(e)

        if "429" in err:
            return "⚠️ AI limit reached. Showing technical analysis instead."

        if "404" in err:
            return "⚠️ Model not available. Use gemini-2.5-flash."

        return "⚠️ AI temporarily unavailable."


# =========================
# SIMPLE INTENT (NO GEMINI CALL = FIX QUOTA ISSUE)
# =========================
def detect_intent(query):
    q = query.lower()

    if "compare" in q:
        return "compare"
    if "buy" in q or "invest" in q or "should i" in q:
        return "single"
    return "general"


def extract_tickers(query):
    mapping = {
        "infosys": "INFY.NS",
        "tata": "TATAMOTORS.NS",
        "reliance": "RELIANCE.NS",
        "tcs": "TCS.NS",
        "apple": "AAPL",
        "tesla": "TSLA",
        "microsoft": "MSFT"
    }

    tickers = []
    q = query.lower()

    for k, v in mapping.items():
        if k in q:
            tickers.append(v)

    return tickers[:2]


# =========================
# DATA
# =========================
def fetch_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    return df if not df.empty else None


# =========================
# INDICATORS
# =========================
def indicators(df):
    df = df.copy()

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.ewm(span=14).mean() / (loss.ewm(span=14).mean() + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# =========================
# BUY SCORE (NO AI DEPENDENCY)
# =========================
def buy_score(df):
    last = df.iloc[-1]
    score = 50

    if last["RSI"] < 30:
        score += 25
    elif last["RSI"] > 70:
        score -= 25

    if last["Close"] > last["MA50"] > last["MA200"]:
        score += 20
    elif last["Close"] < last["MA50"] < last["MA200"]:
        score -= 20

    return max(5, min(95, score))


# =========================
# NEWS SENTIMENT (SAFE)
# =========================
def get_news_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news[:5]
        titles = [n.get("title", "") for n in news]

        pos = sum(1 for t in titles if any(w in t.lower() for w in ["rise","profit","gain"]))
        neg = sum(1 for t in titles if any(w in t.lower() for w in ["fall","loss","drop"]))

        if pos > neg:
            return "Positive 😊"
        elif neg > pos:
            return "Negative 😟"
        return "Neutral 😐"
    except:
        return "No news"


# =========================
# ANALYSIS ENGINE
# =========================
def analyze(ticker):
    df = fetch_data(ticker)

    if df is None:
        st.error(f"No data for {ticker}")
        return

    df = indicators(df)

    prob = buy_score(df)
    rec = "BUY" if prob > 65 else "SELL" if prob < 40 else "HOLD"

    st.subheader(ticker)

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommendation", rec)
    c2.metric("Buy Probability", f"{prob}%")
    c3.metric("Price", round(df["Close"].iloc[-1], 2))

    st.line_chart(df[["Close", "MA50", "MA200"]].tail(100))

    st.info("Sentiment: " + get_news_sentiment(ticker))

    prompt = f"""
    Stock: {ticker}
    Price: {df['Close'].iloc[-1]}
    Recommendation: {rec}
    Explain in simple terms:
    1. Why this rating
    2. Risk
    3. Outlook
    """

    st.markdown("### 🤖 AI Insight")
    st.write(safe_gemini(prompt))


# =========================
# PORTFOLIO
# =========================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

st.sidebar.title("📊 Portfolio")

stock = st.sidebar.text_input("Stock")
qty = st.sidebar.number_input("Qty", 1)

if st.sidebar.button("Add"):
    st.session_state.portfolio[stock] = qty

st.sidebar.write(st.session_state.portfolio)


# =========================
# UI
# =========================
st.title("🤖 AI Investment Assistant")

query = st.text_input("Ask something")

if st.button("Analyze"):
    if not query:
        st.warning("Enter query")
    else:
        intent = detect_intent(query)
        st.info(f"Intent: {intent}")

        tickers = extract_tickers(query)

        if not tickers:
            st.write(safe_gemini(query))
        else:
            for t in tickers:
                analyze(t)


# =========================
# AUTO PICKS
# =========================
st.markdown("### 🔥 Auto Picks")

watchlist = ["AAPL", "MSFT", "TSLA", "INFY.NS", "TCS.NS"]

for t in watchlist:
    df = fetch_data(t)
    if df is not None:
        df = indicators(df)
        st.write(t, "→", buy_score(df), "% Buy Score")
