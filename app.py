import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import google.generativeai as genai
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Investment Assistant", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash"

api_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠️ Add GEMINI_API_KEY in Streamlit secrets or environment")

# =========================
# SAFE GEMINI WRAPPER
# =========================
def safe_gemini(prompt):
    if not api_key:
        return "⚠️ Gemini not configured"

    model = genai.GenerativeModel(GEMINI_MODEL)

    for i in range(2):  # retry twice
        try:
            res = model.generate_content(prompt)
            return res.text
        except Exception as e:
            err = str(e)

            # quota / rate limit fallback
            if "429" in err:
                time.sleep(2)
                continue

            # model not found fallback
            if "404" in err:
                return "⚠️ Model error. Check GEMINI_MODEL name."

            return "⚠️ AI temporarily unavailable. Using fallback analysis."

    return "⚠️ Gemini quota exceeded. Try again later."


# =========================
# INTENT PARSER
# =========================
def resolve_tickers(query):
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
    Extract stock tickers from:
    "{query}"

    Return JSON:
    {{
      "intent": "single|compare|general",
      "tickers": ["AAPL", "TCS.NS"]
    }}

    ONLY JSON.
    """

    try:
        res = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(res.text)
    except:
        return {"intent": "general", "tickers": []}


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
# BUY PROBABILITY
# =========================
def buy_probability(df):
    last = df.iloc[-1]
    score = 0

    if last["RSI"] < 30:
        score += 30
    elif last["RSI"] > 70:
        score -= 30

    if last["Close"] > last["MA50"] > last["MA200"]:
        score += 40
    elif last["Close"] < last["MA50"] < last["MA200"]:
        score -= 40

    prob = min(95, max(5, 50 + score))
    return prob


# =========================
# SENTIMENT (simple fallback)
# =========================
def sentiment_score(news_list):
    positive_words = ["rise", "profit", "growth", "gain", "up", "strong"]
    negative_words = ["fall", "loss", "drop", "weak", "down"]

    score = 0
    for n in news_list:
        text = n.lower()
        for w in positive_words:
            if w in text:
                score += 1
        for w in negative_words:
            if w in text:
                score -= 1

    if score > 2:
        return "Positive 😊"
    elif score < -2:
        return "Negative 😟"
    return "Neutral 😐"


# =========================
# NEWS
# =========================
def get_news(ticker):
    try:
        return yf.Ticker(ticker).news[:5]
    except:
        return []


# =========================
# AI REPORT
# =========================
def ai_report(ticker, df, rec):
    prompt = f"""
    Stock: {ticker}
    Price: {df['Close'].iloc[-1]}
    Recommendation: {rec}

    Give:
    1. Summary
    2. Technical view
    3. Risk factors
    """

    return safe_gemini(prompt)


# =========================
# ANALYSIS
# =========================
def analyze(ticker):
    df = fetch_data(ticker)
    if df is None:
        st.error("No data found")
        return

    df = indicators(df)
    prob = buy_probability(df)

    rec = "BUY" if prob > 65 else "SELL" if prob < 40 else "HOLD"

    st.subheader(ticker)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommendation", rec)
    col2.metric("Buy Probability", f"{prob}%")
    col3.metric("Price", round(df["Close"].iloc[-1], 2))

    st.line_chart(df[["Close", "MA50", "MA200"]].tail(100))

    # NEWS
    news = get_news(ticker)
    if news:
        st.markdown("### 📰 News")
        titles = []
        for n in news:
            title = n.get("title", "")
            titles.append(title)
            st.write("•", title)

        st.info("Sentiment: " + sentiment_score(titles))

    # AI REPORT
    st.markdown("### 🤖 AI Report")
    st.write(ai_report(ticker, df, rec))


# =========================
# PORTFOLIO
# =========================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

def portfolio_ui():
    st.sidebar.title("📊 Portfolio")

    ticker = st.sidebar.text_input("Add Stock")
    qty = st.sidebar.number_input("Quantity", 1)

    if st.sidebar.button("Add"):
        st.session_state.portfolio[ticker] = qty

    st.sidebar.write(st.session_state.portfolio)


# =========================
# CHAT UI
# =========================
st.title("🤖 AI Investment Assistant")

portfolio_ui()

query = st.text_input("Ask: Should I invest in Infosys?")

if st.button("Analyze"):
    if not query:
        st.warning("Enter query")
    else:
        parsed = resolve_tickers(query)

        st.info(f"Intent: {parsed['intent']}")

        if parsed["tickers"]:
            for t in parsed["tickers"]:
                analyze(t)
        else:
            st.markdown(safe_gemini(query))


# =========================
# AUTO RECOMMENDATIONS
# =========================
st.markdown("### 🔥 Auto Recommendations")

watchlist = ["AAPL", "MSFT", "TSLA", "TCS.NS", "INFY.NS"]

for t in watchlist:
    df = fetch_data(t)
    if df is not None:
        df = indicators(df)
        prob = buy_probability(df)
        st.write(t, "→ Buy Probability:", prob)
