import os
import json
import time
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

# Rate limit protection
if "last_call" not in st.session_state:
    st.session_state.last_call = 0

# API KEY
api_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("❌ GEMINI_API_KEY not found")


# =========================
# SAFE GEMINI CALL
# =========================
def safe_gemini_call(prompt):
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 400
            }
        )
        return model.generate_content(prompt).text
    except Exception:
        return "⚠️ AI service temporarily unavailable. Please try again later."


# =========================
# INTENT PARSER
# =========================
def resolve_tickers_and_intent(query):
    prompt = f"""
    Classify query: "{query}"

    Return ONLY JSON:
    {{
      "intent": "single_analysis | compare | sector_recommendation | general_qa",
      "companies": [],
      "tickers": []
    }}

    Rules:
    - Indian stocks use .NS
    - US stocks normal ticker
    - max 2 tickers
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        res = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(res.text)
    except:
        return {"intent": "general_qa", "companies": [], "tickers": []}


# =========================
# STOCK DATA
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
        rec = "BUY"
    elif score < -20:
        rec = "SELL"
    else:
        rec = "HOLD"

    confidence = int(max(20, min(95, abs(score))))

    return rec, confidence


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
# MAIN ENGINE
# =========================
def process_query(query):

    # RATE LIMIT
    if time.time() - st.session_state.last_call < 3:
        st.warning("⏳ Please wait 3 seconds before next request")
        st.stop()

    st.session_state.last_call = time.time()

    data = resolve_tickers_and_intent(query)

    intent = data.get("intent", "general_qa")
    tickers = data.get("tickers", [])

    st.info(f"Intent: {intent}")

    # =========================
    # SINGLE STOCK
    # =========================
    if intent == "single_analysis" and tickers:
        ticker = tickers[0]

        df = fetch_stock_data(ticker)
        if df is None:
            st.error("No data found")
            return

        df = calculate_indicators(df)
        rec, conf = generate_technical_recommendation(df)

        st.subheader(ticker)
        st.metric("Recommendation", rec)
        st.metric("Confidence", f"{conf}%")

        plot_data(df, ticker)

        prompt = f"""
        Stock: {ticker}
        Price: {df['Close'].iloc[-1]}
        Recommendation: {rec}

        Give:
        1. Summary
        2. Technical view
        3. Risks
        """

        st.markdown("### AI Report")
        st.write(safe_gemini_call(prompt))

    # =========================
    # COMPARE
    # =========================
    elif intent == "compare" and len(tickers) >= 2:
        t1, t2 = tickers[:2]

        df1 = fetch_stock_data(t1)
        df2 = fetch_stock_data(t2)

        if df1 is None or df2 is None:
            st.error("Invalid tickers")
            return

        df1 = calculate_indicators(df1)
        df2 = calculate_indicators(df2)

        r1, _ = generate_technical_recommendation(df1)
        r2, _ = generate_technical_recommendation(df2)

        st.subheader("Comparison")
        st.write(f"{t1} → {r1}")
        st.write(f"{t2} → {r2}")

    # =========================
    # GENERAL Q&A (FIXED CRASH)
    # =========================
    else:
        st.markdown("### AI Answer")
        st.write(safe_gemini_call(query))


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
