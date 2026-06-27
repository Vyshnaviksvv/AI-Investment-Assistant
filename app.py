import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Investment Assistant", layout="wide")

# API KEY
api_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets or environment.")
    st.stop()

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash"   # ✅ FIXED MODEL


# =========================
# SAFE GEMINI CALL (IMPORTANT)
# =========================
def safe_gemini(prompt, json_mode=False):
    try:
        model = genai.GenerativeModel(MODEL_NAME)

        if json_mode:
            res = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
        else:
            res = model.generate_content(prompt)

        return res.text

    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"


# =========================
# INTENT PARSER
# =========================
def resolve_tickers_and_intent(query):
    prompt = f"""
    Analyze: "{query}"

    Return ONLY JSON:
    {{
      "intent": "single_analysis|compare|general_qa",
      "companies": [],
      "tickers": []
    }}

    Rules:
    - Infosys → INFY.NS
    - TCS → TCS.NS
    - Reliance → RELIANCE.NS
    - Apple → AAPL
    - Tesla → TSLA
    """

    try:
        res = safe_gemini(prompt, json_mode=True)
        return json.loads(res)
    except:
        return {"intent": "general_qa", "companies": [], "tickers": []}


# =========================
# DATA
# =========================
def fetch_stock_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y")
        return df if not df.empty else None
    except:
        return None


# =========================
# INDICATORS
# =========================
def calculate_indicators(df):
    df = df.copy()

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.ewm(com=13).mean() / (loss.ewm(com=13).mean() + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

    return df


# =========================
# SIGNAL ENGINE
# =========================
def generate_signal(df):
    latest = df.iloc[-1]
    score = 0

    if latest["RSI"] < 30:
        score += 25
    elif latest["RSI"] > 70:
        score -= 25

    if latest["Close"] > latest["MA50"] > latest["MA200"]:
        score += 30
    elif latest["Close"] < latest["MA50"] < latest["MA200"]:
        score -= 30

    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 15
    else:
        score -= 15

    if score > 20:
        return "BUY", min(95, abs(score))
    elif score < -20:
        return "SELL", min(95, abs(score))
    else:
        return "HOLD", min(95, abs(score))


# =========================
# PLOT
# =========================
def plot_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index[-120:], df["Close"][-120:], label="Close")
    ax.plot(df.index[-120:], df["MA50"][-120:], label="MA50")
    ax.plot(df.index[-120:], df["MA200"][-120:], label="MA200")
    ax.set_title(f"{ticker} Analysis")
    ax.legend()
    st.pyplot(fig)


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
# MAIN LOGIC
# =========================
def process_query(query):
    st.info("Processing request...")

    data = resolve_tickers_and_intent(query)
    intent = data.get("intent")
    tickers = data.get("tickers", [])

    st.write("Intent:", intent)

    # SINGLE STOCK
    if intent == "single_analysis" and tickers:
        t = tickers[0]

        df = fetch_stock_data(t)
        if df is None:
            st.error("No data found")
            return

        df = calculate_indicators(df)
        rec, conf = generate_signal(df)

        st.metric("Recommendation", rec)
        st.metric("Confidence", f"{conf}%")

        plot_chart(df, t)

        st.markdown(ai_report(t, df, rec))

    # COMPARE
    elif intent == "compare" and len(tickers) >= 2:
        t1, t2 = tickers[:2]

        df1 = calculate_indicators(fetch_stock_data(t1))
        df2 = calculate_indicators(fetch_stock_data(t2))

        r1, _ = generate_signal(df1)
        r2, _ = generate_signal(df2)

        st.subheader("Comparison")
        st.write(f"{t1}: {r1}")
        st.write(f"{t2}: {r2}")

    # GENERAL QA (FIXED ERROR HERE)
    else:
        prompt = f"""
        You are a financial advisor.
        Answer simply and clearly:

        Question: {query}

        IMPORTANT:
        - No hallucination
        - Add risk disclaimer
        """

        st.markdown(safe_gemini(prompt))


# =========================
# UI
# =========================
st.title("🤖 AI Investment Assistant")

query = st.text_input("Ask your question (e.g., Should I buy Infosys?)")

if st.button("Analyze"):
    if query.strip():
        process_query(query)
    else:
        st.warning("Enter a query")
