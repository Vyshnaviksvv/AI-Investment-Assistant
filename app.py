import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import re
from datetime import datetime, timedelta
import feedparser
from textblob import TextBlob
# Attempt to load vaderSentiment with safe fallback
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    vader_analyzer = None
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
def format_val(val, fmt_type="float", prefix="", suffix="", fallback="N/A"):
    if val is None or pd.isna(val) or val == "":
        return fallback
    try:
        if fmt_type == "int":
            return f"{prefix}{int(val):,d}{suffix}"
        elif fmt_type == "float":
            return f"{prefix}{float(val):,.2f}{suffix}"
        elif fmt_type == "currency":
            return f"{prefix}${float(val):,.2f}{suffix}"
        else:
            return f"{prefix}{str(val)}{suffix}"
    except Exception:
        return fallback
# --- PAGE CONFIGURATION & THEME ---
st.set_page_config(
    page_title="Aegis AI Investment Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for custom dark theme, glassmorphism cards, and premium typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container background */
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
    }
    
    /* Card Glassmorphism Effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #8a99ad;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 5px;
        font-weight: 600;
    }
    
    .delta-positive {
        color: #00e676;
    }
    
    .delta-negative {
        color: #ff1744;
    }
    
    .delta-neutral {
        color: #b0bec5;
    }
    
    /* Section headers */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .section-header {
        border-left: 4px solid #00e676;
        padding-left: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)
# --- DICTIONARIES & CONSTANTS ---
COMPANY_TICKER_MAP = {
    'apple': 'AAPL', 'microsoft': 'MSFT', 'tesla': 'TSLA', 'google': 'GOOGL', 'alphabet': 'GOOGL',
    'amazon': 'AMZN', 'nvidia': 'NVDA', 'meta': 'META', 'facebook': 'META', 'netflix': 'NFLX',
    'amd': 'AMD', 'intel': 'INTC', 'jpmorgan': 'JPM', 'chase': 'JPM', 'berkshire': 'BRK-B',
    'exxon': 'XOM', 'chevron': 'CVX', 'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST',
    'disney': 'DIS', 'nike': 'NKE', 'visa': 'V', 'mastercard': 'MA', 'salesforce': 'CRM',
    'oracle': 'ORCL', 'adobe': 'ADBE', 'cisco': 'CSCO', 'comcast': 'CMCSA', 'pepsi': 'PEP',
    'coca-cola': 'KO', 'coca cola': 'KO', 'johnson': 'JNJ', 'pfizer': 'PFE', 'moderna': 'MRNA',
    'eli lilly': 'LLY', 'broadcom': 'AVGO', 'qualcomm': 'QCOM', 'asml': 'ASML'
}
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    'Financials': 'XLF',
    'Utilities': 'XLU',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE'
}
# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_historical_data(ticker, period="2y"):
    """Fetch historical stock price data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            # Try to fetch without period restriction using start date
            df = stock.history(start=(datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"))
        return df
    except Exception as e:
        return pd.DataFrame()
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_info(ticker):
    """Fetch general company profile and info metrics."""
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        return {}
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ticker_news(ticker):
    """Fetch recent news articles for a specific ticker."""
    try:
        stock = yf.Ticker(ticker)
        return stock.news
    except Exception as e:
        return []
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_general_financial_news():
    """Fetch general market financial news using Yahoo Finance RSS."""
    try:
        feed = feedparser.parse("https://finance.yahoo.com/news/rss")
        articles = []
        for entry in feed.entries[:10]:
            articles.append({
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', '')
            })
        return articles
    except Exception as e:
        return []
# --- INTELLIGENT QUERY UNDERSTANDING (NLP) ---
def parse_query(query: str):
    """Parse search query using regex & keywords to extract tickers and analysis intents."""
    if not query:
        return {"is_compare": False, "screener_type": None, "tickers": []}
        
    query_lower = query.lower().strip()
    
    # Intent 1: Comparison
    compare_triggers = ["compare", "vs", "versus", "difference between"]
    is_compare = any(t in query_lower for t in compare_triggers)
    
    # Intent 2: Screeners
    screener_type = None
    if "dividend" in query_lower:
        screener_type = "dividend"
    elif "under $100" in query_lower or "under 100" in query_lower:
        screener_type = "under_100"
    elif "growth" in query_lower:
        screener_type = "growth"
    elif "ai stock" in query_lower or "artificial intelligence" in query_lower:
        screener_type = "ai"
    elif "semiconductor" in query_lower or "chip" in query_lower:
        screener_type = "semiconductor"
    elif "bank" in query_lower or "banking" in query_lower or "financial" in query_lower:
        screener_type = "banking"
    elif "long term" in query_lower or "long-term" in query_lower:
        screener_type = "long_term"
    elif "swing" in query_lower or "short term" in query_lower or "short-term" in query_lower:
        screener_type = "swing"
        
    # Extract tickers
    found_tickers = []
    
    # Check manual company maps
    for comp_name, tick in COMPANY_TICKER_MAP.items():
        if comp_name in query_lower:
            found_tickers.append(tick)
            
    # Regex search for 1-5 letter uppercase strings or alphabetic sequences
    words = re.findall(r'\b[a-zA-Z]{1,5}\b', query)
    stop_words = {"i", "a", "and", "or", "the", "to", "for", "in", "is", "it", "on", "of", "by", "at", "an", "be", "as", "do", "go", "no", "so", "we", "us", "me", "my", "he", "up", "if", "analyze", "should", "stocks", "stock", "best", "good", "ideas", "buy", "sell"}
    
    for w in words:
        w_lower = w.lower()
        w_upper = w.upper()
        if w_lower not in stop_words:
            # If word is uppercase in original query OR matches known map, treat as ticker
            if w.isupper() or w_upper in COMPANY_TICKER_MAP.values():
                if w_upper not in found_tickers:
                    found_tickers.append(w_upper)
                    
    # Remove duplicates preserving order
    seen = set()
    found_tickers = [x for x in found_tickers if not (x in seen or seen.add(x))]
    
    return {
        "is_compare": is_compare or len(found_tickers) > 1,
        "screener_type": screener_type,
        "tickers": found_tickers
    }
# --- TECHNICAL INDICATOR COMPILING ---
def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
def compute_technical_indicators(df):
    """Compute technical analysis indicators for stock historical data."""
    if df.empty or len(df) < 30:
        return df
        
    df = df.copy()
    
    # SMA and EMA
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    df['RSI'] = compute_rsi(df['Close'], period=14)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (2 * std_dev)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # ADX (Average Directional Index)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    atr_smooth = true_range.ewm(alpha=1/14, min_periods=14).mean()
    plus_di = 100 * (df['+DM'].ewm(alpha=1/14, min_periods=14).mean() / atr_smooth)
    minus_di = 100 * (df['-DM'].ewm(alpha=1/14, min_periods=14).mean() / atr_smooth)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['ADX'] = dx.ewm(alpha=1/14, min_periods=14).mean()
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # VWAP (Volume Weighted Average Price) - Rolling 14 period
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=14).sum() / df['Volume'].rolling(window=14).sum()
    
    return df
def find_levels_and_signals(df):
    """Determine support, resistance, trend strength, and buy/sell signals."""
    if df.empty or len(df) < 50:
        return 0.0, 0.0, "Neutral", "HOLD", 50.0, {}
        
    last_row = df.iloc[-1]
    close = last_row['Close']
    
    # Fibonacci Retracement Levels based on past 120 trading days
    lookback = df.iloc[-120:]
    high_val = lookback['High'].max()
    low_val = lookback['Low'].min()
    diff = high_val - low_val
    fib_levels = {
        "0.0%": high_val,
        "23.6%": high_val - 0.236 * diff,
        "38.2%": high_val - 0.382 * diff,
        "50.0%": high_val - 0.5 * diff,
        "61.8%": high_val - 0.618 * diff,
        "100.0%": low_val
    }
    
    # Support / Resistance (using rolling local minimums/maximums)
    pivot_window = 15
    supports = []
    resistances = []
    recent = df.iloc[-80:]
    for i in range(pivot_window, len(recent) - pivot_window):
        if recent['Low'].iloc[i] == recent['Low'].iloc[i-pivot_window:i+pivot_window].min():
            supports.append(recent['Low'].iloc[i])
        if recent['High'].iloc[i] == recent['High'].iloc[i-pivot_window:i+pivot_window].max():
            resistances.append(recent['High'].iloc[i])
            
    # Consolidate and clean levels
    supports = sorted(list(set([round(x, 2) for x in supports])))
    resistances = sorted(list(set([round(x, 2) for x in resistances])))
    
    # Fallbacks if list is empty
    supp = supports[-1] if supports else round(close * 0.95, 2)
    res = resistances[0] if resistances and resistances[0] > close else round(close * 1.05, 2)
    
    # Trend Strength Category
    adx = last_row.get('ADX', 20.0)
    if adx < 20:
        trend = "Weak / Sideways"
    elif adx < 30:
        trend = "Moderate Trend"
    elif adx < 50:
        trend = "Strong Trend"
    else:
        trend = "Very Strong Trend"
        
    # Compile raw BUY / HOLD / SELL signals
    buy_signals = 0
    sell_signals = 0
    total_rules = 0
    
    # RSI Rule
    rsi = last_row.get('RSI', 50.0)
    if rsi < 30:
        buy_signals += 1
    elif rsi > 70:
        sell_signals += 1
    total_rules += 1
    
    # MACD Rule
    macd = last_row.get('MACD', 0.0)
    macd_sig = last_row.get('MACD_Signal', 0.0)
    if macd > macd_sig:
        buy_signals += 1
    else:
        sell_signals += 1
    total_rules += 1
    
    # Moving Average Cross
    if close > last_row.get('EMA_20', close):
        buy_signals += 1
    else:
        sell_signals += 1
    total_rules += 1
    
    # Bollinger Bands Rule
    if close < last_row.get('BB_Lower', close):
        buy_signals += 1
    elif close > last_row.get('BB_Upper', close):
        sell_signals += 1
    total_rules += 1
    
    # Trend Strength Check
    sma_50 = last_row.get('SMA_50', close)
    if close > sma_50:
        buy_signals += 0.5
    else:
        sell_signals += 0.5
    total_rules += 0.5
    
    buy_pct = (buy_signals / total_rules) * 100.0
    sell_pct = (sell_signals / total_rules) * 100.0
    
    if buy_pct > 65:
        signal = "BUY"
        conf = buy_pct
    elif sell_pct > 65:
        signal = "SELL"
        conf = sell_pct
    else:
        signal = "HOLD"
        conf = max(buy_pct, sell_pct)
        
    return supp, res, trend, signal, conf, fib_levels
# --- NEWS SENTIMENT ENGINE ---
def analyze_article_sentiment(text):
    """Analyze article headline/summary text to get positive/negative/neutral breakdown."""
    if not text:
        return 0.0, {"pos": 0.0, "neg": 0.0, "neu": 1.0}
        
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if vader_analyzer:
        vs = vader_analyzer.polarity_scores(text)
        return vs['compound'], {"pos": vs['pos'], "neg": vs['neg'], "neu": vs['neu']}
    else:
        # Fallback dictionary formulation using TextBlob Polarity
        pos = max(0.0, polarity)
        neg = max(0.0, -polarity)
        neu = 1.0 - pos - neg
        # Normalize
        tot = pos + neg + neu
        return polarity, {"pos": pos/tot, "neg": neg/tot, "neu": neu/tot}
def get_sentiment_summary(news_list):
    """Compile overall sentiment scores across multiple news articles."""
    if not news_list:
        return 0.0, {"pos": 0.0, "neg": 0.0, "neu": 1.0}, "Neutral"
        
    compound_scores = []
    pos_scores = []
    neg_scores = []
    neu_scores = []
    
    for item in news_list:
        content = item.get('title', '') + " " + item.get('summary', '')
        comp, breakdown = analyze_article_sentiment(content)
        compound_scores.append(comp)
        pos_scores.append(breakdown['pos'])
        neg_scores.append(breakdown['neg'])
        neu_scores.append(breakdown['neu'])
        
    avg_compound = np.mean(compound_scores) if compound_scores else 0.0
    avg_breakdown = {
        "pos": np.mean(pos_scores) if pos_scores else 0.0,
        "neg": np.mean(neg_scores) if neg_scores else 0.0,
        "neu": np.mean(neu_scores) if neu_scores else 1.0
    }
    
    # Categorize label
    if avg_compound > 0.15:
        verdict = "Positive"
    elif avg_compound < -0.15:
        verdict = "Negative"
    else:
        verdict = "Neutral"
        
    return avg_compound, avg_breakdown, verdict
# --- GLOBAL MARKET TRENDS & FEAR & GREED PROXY ---
@st.cache_data(ttl=1800, show_spinner=False)
def get_global_market_indices():
    """Download daily metrics for major international stock indices."""
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'Dow Jones': '^DJI',
        'Nifty 50': '^NSEI',
        'Sensex': '^BSESN',
        'VIX': '^VIX'
    }
    
    results = {}
    for name, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                pct_change = ((last_close - prev_close) / prev_close) * 100
                results[name] = {
                    "price": last_close,
                    "change": pct_change,
                    "ticker": ticker
                }
        except Exception as e:
            results[name] = {"price": 0.0, "change": 0.0, "ticker": ticker}
    return results
def calculate_fear_greed_proxy():
    """Calculate an interactive Fear & Greed Proxy score (0-100) using public market indices."""
    try:
        sp500 = yf.Ticker("^GSPC").history(period="1y")
        vix = yf.Ticker("^VIX").history(period="1y")
        
        if sp500.empty or vix.empty:
            return 50, "Neutral"
            
        sp_close = sp500['Close'].iloc[-1]
        vix_close = vix['Close'].iloc[-1]
        
        # 1. Price Momentum (S&P 500 vs 125-Day Moving Average)
        ma_125 = sp500['Close'].rolling(window=125).mean().iloc[-1]
        if pd.isna(ma_125):
            ma_125 = sp500['Close'].mean()
        dist_ma = (sp_close - ma_125) / ma_125
        dist_clipped = max(-0.10, min(0.10, dist_ma))
        score_momentum = ((dist_clipped + 0.10) / 0.20) * 30 # 0 to 30 points
        
        # 2. Market Volatility (VIX level vs historical limits)
        vix_clipped = max(12.0, min(30.0, vix_close))
        score_volatility = ((30.0 - vix_clipped) / 18.0) * 25 # 0 to 25 points
        
        # 3. Market RSI (Relative Strength Index of S&P 500)
        sp_rsi_series = compute_rsi(sp500['Close'])
        sp_rsi = sp_rsi_series.iloc[-1] if not sp_rsi_series.empty else 50.0
        if pd.isna(sp_rsi):
            sp_rsi = 50.0
        rsi_clipped = max(25.0, min(75.0, sp_rsi))
        score_rsi = ((rsi_clipped - 25.0) / 50.0) * 25 # 0 to 25 points
        
        # 4. Safe Haven Demand Proxy (20-day rate of change of S&P 500)
        ret_20 = (sp_close - sp500['Close'].iloc[-20]) / sp500['Close'].iloc[-20]
        ret_clipped = max(-0.05, min(0.05, ret_20))
        score_returns = ((ret_clipped + 0.05) / 0.10) * 20 # 0 to 20 points
        
        total_score = int(score_momentum + score_volatility + score_rsi + score_returns)
        
        # Classify state
        if total_score < 25:
            state = "Extreme Fear"
        elif total_score < 45:
            state = "Fear"
        elif total_score < 55:
            state = "Neutral"
        elif total_score < 75:
            state = "Greed"
        else:
            state = "Extreme Greed"
            
        return total_score, state
    except Exception as e:
        return 50, "Neutral"
# --- AI RECOMMENDATION ENGINE (NO LLM REQUIRED) ---
def compute_ai_recommendation(ticker_info, df_tech, sentiment_compound, fear_greed):
    """Evaluate stock metrics dynamically using mathematical weight distribution."""
    # Weights configuration
    W_TECH = 0.35
    W_FUND = 0.35
    W_SENT = 0.15
    W_MKT = 0.15
    
    # 1. Technical Analysis Score Calculation
    tech_score = 50.0
    if not df_tech.empty and len(df_tech) > 0:
        last = df_tech.iloc[-1]
        tech_rules = []
        
        # RSI Indicator
        rsi = last.get('RSI', 50.0)
        if rsi < 30:
            tech_rules.append(100.0) # Oversold (Bullish)
        elif rsi > 70:
            tech_rules.append(0.0) # Overbought (Bearish)
        else:
            tech_rules.append(50.0)
            
        # MACD Signal
        macd = last.get('MACD', 0.0)
        macd_sig = last.get('MACD_Signal', 0.0)
        tech_rules.append(100.0 if macd > macd_sig else 0.0)
        
        # SMA Crossovers
        close = last.get('Close', 0.0)
        tech_rules.append(100.0 if close > last.get('SMA_50', close) else 0.0)
        tech_rules.append(100.0 if close > last.get('SMA_200', close) else 0.0)
        
        # Bollinger Bands Position
        bb_upper = last.get('BB_Upper', close)
        bb_lower = last.get('BB_Lower', close)
        if close < bb_lower:
            tech_rules.append(100.0)
        elif close > bb_upper:
            tech_rules.append(0.0)
        else:
            tech_rules.append(50.0)
            
        tech_score = np.mean(tech_rules) if tech_rules else 50.0
        
    # 2. Fundamental Score Calculation
    fund_score = 50.0
    fund_rules = []
    
    # PE ratio metric check
    pe = ticker_info.get('trailingPE')
    fwd_pe = ticker_info.get('forwardPE')
    
    if pe:
        if pe < 15:
            fund_rules.append(90.0)
        elif pe < 30:
            fund_rules.append(65.0)
        else:
            fund_rules.append(30.0)
    else:
        fund_rules.append(50.0)
        
    # Earnings growth signifier
    if pe and fwd_pe:
        fund_rules.append(100.0 if fwd_pe < pe else 40.0)
        
    # Profit Margin
    margin = ticker_info.get('profitMargins')
    if margin:
        if margin > 0.20:
            fund_rules.append(100.0)
        elif margin > 0.05:
            fund_rules.append(70.0)
        elif margin > 0.0:
            fund_rules.append(50.0)
        else:
            fund_rules.append(10.0)
            
    # Beta / Volatility risk adjustments
    beta = ticker_info.get('beta')
    if beta:
        if beta < 0.9:
            fund_rules.append(85.0) # Defensive
        elif beta < 1.3:
            fund_rules.append(65.0) # Growth
        else:
            fund_rules.append(40.0) # High risk
            
    fund_score = np.mean(fund_rules) if fund_rules else 50.0
    
    # 3. Sentiment Score mapping from [-1, 1] to [0, 100]
    sent_score = ((sentiment_compound + 1.0) / 2.0) * 100.0
    
    # 4. Market Trend Score based on Fear & Greed Proxy
    mkt_score = fear_greed
    
    # Total Score calculation
    total_score = (W_TECH * tech_score) + (W_FUND * fund_score) + (W_SENT * sent_score) + (W_MKT * mkt_score)
    
    # Category Assignment
    if total_score >= 80:
        recommendation = "STRONG BUY"
        confidence = total_score
    elif total_score >= 60:
        recommendation = "BUY"
        confidence = total_score
    elif total_score >= 40:
        recommendation = "HOLD"
        confidence = 100 - abs(total_score - 50.0) * 2.0
    elif total_score >= 20:
        recommendation = "SELL"
        confidence = 100.0 - total_score
    else:
        recommendation = "STRONG SELL"
        confidence = 100.0 - total_score
        
    return {
        "verdict": recommendation,
        "confidence": confidence,
        "technical_score": tech_score,
        "fundamental_score": fund_score,
        "sentiment_score": sent_score,
        "market_score": mkt_score,
        "total_score": total_score
    }
# --- MACHINE LEARNING PREDICTION ENGINE ---
def run_ml_forecasting(df):
    """Build Random Forest models to predict future price outcomes."""
    if df.empty or len(df) < 120:
        return None
        
    df = df.copy()
    
    # 1. Feature Engineering
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    
    # MACD Calculation
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Lag Features
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_5'] = df['Close'].shift(5)
    
    # Volatility indicator
    df['Vol_10'] = df['Close'].pct_change().rolling(window=10).std()
    
    feature_cols = ['SMA_10', 'SMA_30', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Lag_1', 'Lag_2', 'Lag_5', 'Vol_10']
    df_clean = df.dropna(subset=feature_cols).copy()
    
    if len(df_clean) < 80:
        return None
        
    X = df_clean[feature_cols]
    results = {}
    
    # Generate predictions for different horizons
    horizons = [('Next Day', -1), ('Next Week', -5), ('Next Month', -20)]
    for label, shift_val in horizons:
        try:
            y = df_clean['Close'].shift(shift_val)
            valid_idx = y.dropna().index
            
            X_train_val = X.loc[valid_idx]
            y_train_val = y.loc[valid_idx]
            
            if len(X_train_val) < 40:
                continue
                
            # Perform train/test split to calculate accuracy/confidence
            X_train, X_test, y_train, y_test = train_test_split(
                X_train_val, y_train_val, test_size=0.15, shuffle=False
            )
            
            model = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Predict validation segment
            preds_test = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, preds_test)
            confidence = max(0.0, min(100.0, 100.0 * (1.0 - mape)))
            
            # Predict target future value from latest active values
            last_feat = X.iloc[[-1]]
            forecast = model.predict(last_feat)[0]
            
            results[label] = {
                "val": forecast,
                "confidence": confidence,
                "current": df_clean['Close'].iloc[-1]
            }
        except Exception as e:
            results[label] = {"val": None, "confidence": 0.0, "current": 0.0}
            
    return results
# --- SINGLE STOCK RISK ANALYSIS ---
def analyze_stock_risk(stock_close, sp500_close, risk_free_rate=0.04):
    """Compute risk profiles: Beta, volatility, Sharpe Ratio, Max Drawdown."""
    try:
        combined = pd.DataFrame({'Stock': stock_close, 'SP500': sp500_close}).dropna()
        if len(combined) < 20:
            return 1.0, 0.0, 0.0, 0.0
            
        returns = combined.pct_change().dropna()
        
        # Beta
        covariance = returns['Stock'].cov(returns['SP500'])
        market_variance = returns['SP500'].var()
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        # Volatility
        volatility = returns['Stock'].std() * np.sqrt(252)
        
        # Return metric
        ann_return = returns['Stock'].mean() * 252
        
        # Sharpe ratio
        sharpe = (ann_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Max Drawdown calculation
        cum_returns = (1 + returns['Stock']).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_dd = drawdowns.min()
        
        return beta, volatility, sharpe, max_dd
    except Exception as e:
        return 1.0, 0.0, 0.0, 0.0
# --- MULTI-TICKER PORTFOLIO CALCULATIONS ---
def get_portfolio_summary(portfolio, risk_free_rate=0.04):
    """Fetch live data and calculate overall portfolio parameters."""
    if not portfolio:
        return None
        
    tickers = list(portfolio.keys())
    
    # Download 1-year daily close prices for assets
    try:
        data = yf.download(tickers, period="1y", group_by="ticker", progress=False)
    except Exception as e:
        return None
        
    returns_df = pd.DataFrame()
    live_prices = {}
    betas = {}
    sectors = {}
    
    # Download S&P 500 benchmark for risk calculations
    try:
        spy = yf.Ticker("^GSPC").history(period="1y")
    except Exception as e:
        spy = pd.DataFrame()
        
    for tick in tickers:
        try:
            # Handle df output variations for single vs multi yf download
            if len(tickers) == 1:
                hist = data
            else:
                hist = data[tick]
                
            close_col = hist['Close']
            returns_df[tick] = close_col.pct_change()
            live_prices[tick] = close_col.iloc[-1]
            
            info = fetch_company_info(tick)
            betas[tick] = info.get('beta', 1.0)
            sectors[tick] = info.get('sector', 'Unknown')
        except Exception as e:
            live_prices[tick] = portfolio[tick]['buy_price']
            betas[tick] = 1.0
            sectors[tick] = 'Unknown'
            
    # Calculate weighted allocations
    invested_cost = 0.0
    current_value = 0.0
    values = []
    
    for tick in tickers:
        qty = portfolio[tick]['shares']
        buy = portfolio[tick]['buy_price']
        cost = qty * buy
        val = qty * live_prices[tick]
        invested_cost += cost
        current_value += val
        values.append(val)
        
    if current_value == 0:
        return None
        
    weights = [v / current_value for v in values]
    
    # Calculate portfolio beta
    portfolio_beta = sum(w * betas[tickers[i]] for i, w in enumerate(weights))
    
    # Volatility and Sharpe Ratio
    if not returns_df.empty and len(returns_df) > 10:
        aligned_weights = [weights[tickers.index(col)] for col in returns_df.columns]
        portfolio_returns = returns_df.multiply(aligned_weights).sum(axis=1)
        
        # Volatility
        vol = portfolio_returns.std() * np.sqrt(252)
        # Expected return proxy
        ann_ret = portfolio_returns.mean() * 252
        # Sharpe
        sharpe = (ann_ret - risk_free_rate) / vol if vol > 0 else 0.0
        
        # Portfolio Max Drawdown
        cum_ret = (1 + portfolio_returns).cumprod()
        running_max = cum_ret.cummax()
        drawdown_series = (cum_ret - running_max) / running_max
        max_dd = drawdown_series.min()
    else:
        vol = 0.0
        sharpe = 0.0
        max_dd = 0.0
        
    # Herfindahl-Hirschman Index (HHI) for diversification
    hhi = sum(w**2 for w in weights)
    diversification_score = max(0.0, min(100.0, (1.0 - hhi) * 125.0))
    
    # Classify Risk State
    if portfolio_beta < 0.8:
        risk_class = "Conservative / Low Risk"
    elif portfolio_beta < 1.25:
        risk_class = "Moderate / Balanced"
    else:
        risk_class = "Aggressive / High Risk"
        
    return {
        "cost": invested_cost,
        "value": current_value,
        "pnl": current_value - invested_cost,
        "pnl_pct": ((current_value - invested_cost) / invested_cost * 100.0) if invested_cost > 0 else 0.0,
        "beta": portfolio_beta,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "diversification": diversification_score,
        "risk_class": risk_class,
        "weights": weights,
        "current_prices": live_prices,
        "sectors": sectors
    }
# --- SECTOR RANKING ENGINE ---
@st.cache_data(ttl=1800, show_spinner=False)
def calculate_sector_rankings():
    """Rank market sectors based on 3-month and YTD returns using ETFs."""
    sector_performances = []
    
    for sector, etf in SECTOR_ETF_MAP.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(period="1y")
            if not hist.empty:
                close = hist['Close']
                # 3 Months return
                returns_3m = 0.0
                if len(close) > 63:
                    returns_3m = ((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]) * 100
                # YTD calculation
                ytd_start_date = datetime(datetime.now().year, 1, 1)
                hist_ytd = hist[hist.index >= pd.Timestamp(ytd_start_date)]
                returns_ytd = 0.0
                if not hist_ytd.empty:
                    returns_ytd = ((close.iloc[-1] - hist_ytd['Close'].iloc[0]) / hist_ytd['Close'].iloc[0]) * 100
                
                sector_performances.append({
                    "Sector": sector,
                    "ETF": etf,
                    "3M Return (%)": returns_3m,
                    "YTD Return (%)": returns_ytd,
                    "Last Close": close.iloc[-1]
                })
        except Exception as e:
            continue
            
    if not sector_performances:
        return pd.DataFrame()
        
    df_perf = pd.DataFrame(sector_performances)
    return df_perf.sort_values(by="3M Return (%)", ascending=False)
# --- INITIAL SESSION STATES ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {
        "AAPL": {"shares": 10, "buy_price": 175.50},
        "MSFT": {"shares": 5, "buy_price": 380.20},
        "NVDA": {"shares": 15, "buy_price": 450.00}
    }
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = "AAPL"
# --- SIDEBAR & GLOBAL SEARCH CONTROLS ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00e676;'>Aegis AI Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8a99ad;'>Advanced No-LLM Financial Dashboard</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Global search box with intelligent query understanding
    search_input = st.text_input("💡 Ask Aegis anything...", placeholder="e.g. Compare AAPL and MSFT", key="search_bar")
    
    if search_input:
        st.session_state.search_query = search_input
        parsed = parse_query(search_input)
        
        # Trigger based on parsed results
        if parsed['tickers']:
            st.session_state.active_ticker = parsed['tickers'][0]
            st.success(f"Detected Tickers: {', '.join(parsed['tickers'])}")
            
        if parsed['screener_type']:
            st.info(f"Target Screening: {parsed['screener_type'].replace('_', ' ').title()}")
            
    st.markdown("---")
    
    # Market Indices Widget
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 10px;'>Market Indices</h4>", unsafe_allow_html=True)
    index_data = get_global_market_indices()
    
    for index_name, val in index_data.items():
        if val['price'] == 0.0:
            price_disp = "N/A"
            change_disp = "N/A"
            color_class = "delta-neutral"
        else:
            price_disp = f"{val['price']:,.2f}"
            color_class = "delta-positive" if val['change'] >= 0 else "delta-negative"
            sign = "+" if val['change'] >= 0 else ""
            change_disp = f"{sign}{val['change']:.2f}%"
            
        st.markdown(f"""
        <div class="metric-card" style="padding: 10px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; font-size: 0.95rem; color:#ffffff;">{index_name}</span>
                <span class="{color_class}" style="font-size: 0.9rem; font-weight: 700;">{change_disp}</span>
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-top: 4px;">
                {price_disp}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    # Quick Ticker Selector
    ticker_choice = st.text_input("Active Ticker Direct Input", value=st.session_state.active_ticker).upper()
    if ticker_choice != st.session_state.active_ticker:
        st.session_state.active_ticker = ticker_choice
# --- DYNAMIC APP LAYOUT ---
# Fetch base data for the active ticker
ticker = st.session_state.active_ticker
df_history = fetch_historical_data(ticker)
info = fetch_company_info(ticker)
news = fetch_ticker_news(ticker)
# Handle offline / invalid ticker outcomes
is_data_valid = not df_history.empty and 'Close' in df_history.columns
if not is_data_valid:
    st.title("⚠️ Ticker Offline or Data Unavailable")
    st.warning(f"Could not load data for **{ticker}**. Please check the ticker name and verify your internet connection.")
    st.markdown("### Suggested Actions:")
    st.markdown("- Verify the ticker symbol is correct (e.g. `AAPL`, `MSFT`, `TSLA`, `RELIANCE.NS`).")
    st.markdown("- Check if Yahoo Finance matches the index formatting.")
    st.markdown("- Try again later if the API is experiencing high latency.")
    st.stop()
# --- HEADER DASHBOARD ---
company_name = info.get('longName', ticker)
current_price = df_history['Close'].iloc[-1]
prev_close = info.get('previousClose') or df_history['Close'].iloc[-2]
pct_change = ((current_price - prev_close) / prev_close) * 100
sign = "+" if pct_change >= 0 else ""
delta_class = "delta-positive" if pct_change >= 0 else "delta-negative"
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown(f"<h1 style='margin-bottom: 5px; color:#ffffff;'>{company_name} <span style='font-size:1.5rem; color:#8a99ad;'>({ticker})</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.1rem; color:#8a99ad;'>{info.get('sector', 'Unknown Sector')} | {info.get('industry', 'Unknown Industry')}</p>", unsafe_allow_html=True)
with col_head2:
    st.markdown(f"""
    <div style='text-align: right;'>
        <div style='font-size: 2.2rem; font-weight: 700; color:#ffffff;'>${current_price:.2f}</div>
        <div class='{delta_class}' style='font-size: 1.2rem; font-weight: 600;'>{sign}{pct_change:.2f}% (Daily)</div>
    </div>
    """, unsafe_allow_html=True)
# Parse any global query intents and prompt user
parsed_query_results = parse_query(st.session_state.search_query) if st.session_state.search_query else None
if parsed_query_results and parsed_query_results['screener_type']:
    st.info(f"💡 Query matched screener: **{parsed_query_results['screener_type'].replace('_', ' ').title()}**. You can check the screeners or comparisons in their respective tabs below.")
# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Dashboard",
    "📊 Technical Analysis",
    "💼 Fundamentals & Risk",
    "🤖 AI Recommendation",
    "🔮 ML Forecast",
    "📰 News & Sentiment",
    "💼 Portfolio Tracker",
    "⚖️ Stock Comparison"
])
# --- TAB 1: DASHBOARD ---
with tab1:
    st.markdown("<h3 class='section-header'>Real-Time Overview</h3>", unsafe_allow_html=True)
    
    # 4 metric cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        high_val = format_val(info.get('fiftyTwoWeekHigh'), 'currency')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">52 Week High</div>
            <div class="metric-value">{high_val}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        low_val = format_val(info.get('fiftyTwoWeekLow'), 'currency')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">52 Week Low</div>
            <div class="metric-value">{low_val}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        mcap_val = format_val(info.get('marketCap'), 'int', prefix='$')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Market Cap</div>
            <div class="metric-value">{mcap_val}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        pe_val = format_val(info.get('trailingPE'), 'float')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">P/E Ratio</div>
            <div class="metric-value">{pe_val}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Main Plotly interactive Candlestick & Volume Chart
    st.markdown("### Interactive Price Chart")
    
    # Compute moving averages for the chart overlay
    df_chart = df_history.copy()
    df_chart['SMA_50'] = df_chart['Close'].rolling(window=50).mean()
    df_chart['SMA_200'] = df_chart['Close'].rolling(window=200).mean()
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        name='Candlestick'
    ))
    
    # SMAs
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], name='50 SMA', line=dict(color='#00e676', width=1.5)))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_200'], name='200 SMA', line=dict(color='#2979ff', width=1.5)))
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Profile information
    col_prof1, col_prof2 = st.columns([2, 1])
    with col_prof1:
        st.markdown("### Business Summary")
        st.write(info.get('longBusinessSummary', 'Profile data is currently unavailable.'))
    with col_prof2:
        st.markdown("### Corporate Leadership")
        st.markdown(f"**CEO:** {info.get('ceo', 'N/A')}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        employees_val = format_val(info.get('fullTimeEmployees'), 'int')
        st.markdown(f"**Employees:** {employees_val}")
# --- TAB 2: TECHNICAL ANALYSIS ---
with tab2:
    st.markdown("<h3 class='section-header'>Technical Diagnostics</h3>", unsafe_allow_html=True)
    
    # Perform computing
    df_indicators = compute_technical_indicators(df_history)
    support, resistance, trend_str, signal, conf, fibs = find_levels_and_signals(df_indicators)
    
    # Main signal panel
    color_sig = "#00e676" if signal == "BUY" else ("#ff1744" if signal == "SELL" else "#b0bec5")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 25px; text-align: center;">
            <div style="font-size: 1.1rem; color: #8a99ad; text-transform: uppercase;">Technical Indicator Verdict</div>
            <div style="font-size: 3rem; font-weight: 800; color: {color_sig}; margin: 15px 0;">{signal}</div>
            <div style="font-size: 1.1rem; color: #ffffff;">Confidence Score: <b>{conf:.1f}%</b></div>
            <div style="font-size: 1.0rem; color: #8a99ad; margin-top: 10px;">Trend: {trend_str}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Support & Resistance levels
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="padding: 15px;">
            <div style="font-weight: 600; color: #ffffff; margin-bottom: 8px;">Calculated Levels</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #ff1744; font-weight: 600;">Immediate Resistance:</span>
                <span style="color: #ffffff; font-weight: 700;">${resistance:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #00e676; font-weight: 600;">Immediate Support:</span>
                <span style="color: #ffffff; font-weight: 700;">${support:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_t2:
        # Technical metrics grid
        last_ind = df_indicators.iloc[-1]
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.metric("RSI (14)", f"{last_ind.get('RSI', 50.0):.2f}", "Oversold (<30)" if last_ind.get('RSI', 50.0) < 30 else ("Overbought (>70)" if last_ind.get('RSI', 50.0) > 70 else "Neutral"))
            st.metric("MACD Line", f"{last_ind.get('MACD', 0.0):.4f}")
            st.metric("ATR (Volatility)", f"{last_ind.get('ATR', 0.0):.2f}")
        with col_g2:
            st.metric("ADX (Trend Strength)", f"{last_ind.get('ADX', 0.0):.2f}", trend_str)
            st.metric("MACD Signal Line", f"{last_ind.get('MACD_Signal', 0.0):.4f}")
            st.metric("VWAP (Rolling)", f"${last_ind.get('VWAP', 0.0):.2f}")
            
    # Charting Sub-panels for Technical indicators
    st.markdown("### Technical Indicators Overlay")
    
    # RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['RSI'], name='RSI', line=dict(color='#ff9100', width=1.5)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff1744")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00e676")
    fig_rsi.update_layout(
        title="RSI (14)",
        template='plotly_dark',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD Chart
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD'], name='MACD', line=dict(color='#2979ff', width=1.5)))
    fig_macd.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD_Signal'], name='Signal', line=dict(color='#ff1744', width=1.5)))
    fig_macd.add_bar(x=df_indicators.index, y=df_indicators['MACD_Hist'], name='Histogram')
    fig_macd.update_layout(
        title="MACD",
        template='plotly_dark',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_macd, use_container_width=True)
    
    # Fibonacci Levels Table
    st.markdown("### Fibonacci Retracement Levels")
    fib_data = [{"Level": k, "Price": f"${v:.2f}"} for k, v in fibs.items()]
    st.table(pd.DataFrame(fib_data))
# --- TAB 3: FUNDAMENTALS & RISK ---
with tab3:
    st.markdown("<h3 class='section-header'>Fundamental Profile & Asset Risk</h3>", unsafe_allow_html=True)
    
    # Get comparison benchmark
    sp500_data = fetch_historical_data("^GSPC")
    
    beta, annual_vol, sharpe, max_dd = analyze_stock_risk(df_history['Close'], sp500_data['Close'])
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("### Valuation & Finance Metrics")
        fund_metrics = {
            "Market Cap": format_val(info.get('marketCap'), 'int', prefix='$'),
            "PE Ratio": format_val(info.get('trailingPE'), 'float'),
            "Forward PE": format_val(info.get('forwardPE'), 'float'),
            "EPS (Trailing)": format_val(info.get('trailingEps'), 'float', prefix='$'),
            "Price to Book": format_val(info.get('priceToBook'), 'float'),
            "Forward Dividend Yield": format_val(info.get('dividendYield') * 100.0 if info.get('dividendYield') is not None else None, 'float', suffix='%'),
            "Beta": format_val(info.get('beta'), 'float')
        }
        st.table(pd.DataFrame(list(fund_metrics.items()), columns=["Metric", "Value"]))
        
    with col_f2:
        st.markdown("### Risk Diagnostic Panel")
        
        # Draw category assessment
        risk_cat = "Low Risk / Defensive" if annual_vol < 0.18 else ("Moderate Risk" if annual_vol < 0.35 else "High Risk / Aggressive")
        
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; margin-bottom: 15px;">
            <div style="font-size: 0.9rem; color: #8a99ad; text-transform: uppercase;">Aegis Risk Rating</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #ff1744; margin-top: 5px;">{risk_cat}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Annualized Volatility", f"{annual_vol * 100:.2f}%")
        st.metric("Sharpe Ratio (Rf=4%)", f"{sharpe:.2f}")
        st.metric("Historical Max Drawdown", f"{max_dd * 100:.2f}%")
# --- TAB 4: AI RECOMMENDATION ENGINE ---
with tab4:
    st.markdown("<h3 class='section-header'>Aegis Weight-Scored Evaluation</h3>", unsafe_allow_html=True)
    
    # Run recommendation engine calculations
    f_g_score, _ = calculate_fear_greed_proxy()
    
    sentiment_compound = 0.0
    if news:
        sentiment_compound, _, _ = get_sentiment_summary(news)
        
    rec = compute_ai_recommendation(info, df_history, sentiment_compound, f_g_score)
    
    rec_color = "#00e676" if "BUY" in rec['verdict'] else ("#ff1744" if "SELL" in rec['verdict'] else "#b0bec5")
    
    col_rec1, col_rec2 = st.columns([1, 1])
    with col_rec1:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 30px; text-align: center;">
            <div style="font-size: 1.2rem; color: #8a99ad; text-transform: uppercase;">Aegis Multi-Factor Rating</div>
            <div style="font-size: 3.5rem; font-weight: 800; color: {rec_color}; margin: 20px 0;">{rec['verdict']}</div>
            <div style="font-size: 1.3rem; color: #ffffff;">Confidence Rating: <b>{rec['confidence']:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_rec2:
        st.markdown("### Scoring Weights Distribution")
        
        st.markdown(f"**Technical Analysis Component:** {rec['technical_score']:.1f}/100")
        st.progress(int(rec['technical_score']) / 100.0)
        
        st.markdown(f"**Fundamental Assessment Component:** {rec['fundamental_score']:.1f}/100")
        st.progress(int(rec['fundamental_score']) / 100.0)
        
        st.markdown(f"**News & Sentiment Score:** {rec['sentiment_score']:.1f}/100")
        st.progress(int(rec['sentiment_score']) / 100.0)
        
        st.markdown(f"**Global Market Score:** {rec['market_score']:.1f}/100")
        st.progress(int(rec['market_score']) / 100.0)
# --- TAB 5: MACHINE LEARNING FORECASTING ---
with tab5:
    st.markdown("<h3 class='section-header'>Machine Learning Forecast (Random Forest Regressor)</h3>", unsafe_allow_html=True)
    
    with st.spinner("Training predictive models on historical sequence data..."):
        predictions = run_ml_forecasting(df_history)
        
    if predictions:
        col_p1, col_p2, col_p3 = st.columns(3)
        
        for idx, (horizon, data) in enumerate(predictions.items()):
            col_target = [col_p1, col_p2, col_p3][idx]
            
            if data['val'] is not None:
                chg = ((data['val'] - data['current']) / data['current']) * 100.0
                sign = "+" if chg >= 0 else ""
                class_chg = "delta-positive" if chg >= 0 else "delta-negative"
                
                with col_target:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div class="metric-title">{horizon} Forecast</div>
                        <div class="metric-value">${data['val']:.2f}</div>
                        <div class="metric-delta {class_chg}">{sign}{chg:.2f}% vs Current</div>
                        <div style="font-size:0.85rem; color:#8a99ad; margin-top:10px;">Confidence Score: {data['confidence']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with col_target:
                    st.info("Insufficient training samples for this forecasting timeframe.")
                    
        # Model description alert
        st.markdown("""
        > [!NOTE]
        > Predictions are computed using a locally optimized **Random Forest Regressor** trained on historical closing prices, lag returns, volatility coefficients, moving averages, and RSI parameters. 
        > This model predicts direct price adjustments and calculates statistical confidence based on test-validation error profiles.
        """)
    else:
        st.warning("Forecasting requires a minimum of 120 trading days of historical data.")
# --- TAB 6: NEWS & SENTIMENT ---
with tab6:
    st.markdown("<h3 class='section-header'>News & Sentiment Metrics</h3>", unsafe_allow_html=True)
    
    if news:
        avg_comp, breakdown, verdict = get_sentiment_summary(news)
        
        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            color_sent = "#00e676" if verdict == "Positive" else ("#ff1744" if verdict == "Negative" else "#b0bec5")
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px;">
                <div style="font-size: 1.0rem; color: #8a99ad; text-transform: uppercase;">Aggregated News Sentiment</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {color_sent}; margin: 10px 0;">{verdict}</div>
                <div style="font-size: 0.95rem; color: #ffffff;">Compound Score: <b>{avg_comp:.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Pie Chart of positive, negative, neutral breakdown
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[breakdown['pos'], breakdown['neg'], breakdown['neu']],
                hole=.3,
                marker=dict(colors=['#00e676', '#ff1744', '#b0bec5'])
            )])
            fig_pie.update_layout(
                template='plotly_dark',
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_n2:
            st.markdown("### Recent Articles")
            for article in news[:5]:
                st.markdown(f"""
                <div style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <a href="{article.get('link', '#')}" target="_blank" style="font-weight: 600; color: #2979ff; font-size: 1.05rem; text-decoration: none;">
                        {article.get('title', 'Headline Unavailable')}
                    </a>
                    <div style="font-size: 0.85rem; color: #8a99ad; margin-top: 4px;">
                        Source: {article.get('publisher', 'N/A')} | Published: {article.get('providerPublishTime', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No current news matches for this ticker.")
        
    # General Market news section
    st.markdown("---")
    st.markdown("### General Financial Market RSS News")
    rss_news = fetch_general_financial_news()
    if rss_news:
        for entry in rss_news[:5]:
            st.markdown(f"""
            <div style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <a href="{entry.get('link', '#')}" target="_blank" style="font-weight: 600; color: #00e676; font-size: 1.05rem; text-decoration: none;">
                    {entry.get('title')}
                </a>
                <div style="font-size: 0.85rem; color: #8a99ad; margin-top: 4px;">
                    Published: {entry.get('published')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("General financial RSS news feed currently offline.")
# --- TAB 7: PORTFOLIO TRACKER ---
with tab7:
    st.markdown("<h3 class='section-header'>Asset Portfolio Manager</h3>", unsafe_allow_html=True)
    
    col_port1, col_port2 = st.columns([1, 2])
    
    with col_port1:
        st.markdown("### Add Ticker Position")
        with st.form("add_ticker_position_form"):
            add_ticker = st.text_input("Stock Ticker", value="GOOGL").upper().strip()
            add_shares = st.number_input("Number of Shares", min_value=0.1, step=1.0, value=10.0)
            add_buy_price = st.number_input("Purchase Price ($)", min_value=0.01, step=1.0, value=150.0)
            
            submit_btn = st.form_submit_button("Add Asset")
            if submit_btn and add_ticker:
                st.session_state.portfolio[add_ticker] = {
                    "shares": add_shares,
                    "buy_price": add_buy_price
                }
                st.success(f"Successfully added position for {add_ticker}!")
                
        # Remove positions
        if st.session_state.portfolio:
            st.markdown("### Remove Ticker Position")
            remove_ticker = st.selectbox("Select Asset to Remove", list(st.session_state.portfolio.keys()))
            if st.button("Remove Position"):
                del st.session_state.portfolio[remove_ticker]
                st.success(f"Removed position for {remove_ticker}!")
                st.rerun()
                
    with col_port2:
        st.markdown("### Portfolio Returns & Health Metrics")
        
        portfolio_summary = get_portfolio_summary(st.session_state.portfolio)
        
        if portfolio_summary:
            col_pm1, col_pm2, col_pm3 = st.columns(3)
            with col_pm1:
                st.metric("Total Investment", f"${portfolio_summary['cost']:,.2f}")
                st.metric("Portfolio Beta", f"{portfolio_summary['beta']:.2f}")
            with col_pm2:
                st.metric("Current Portfolio Value", f"${portfolio_summary['value']:,.2f}")
                st.metric("Annualized Volatility", f"{portfolio_summary['volatility'] * 100:.2f}%")
            with col_pm3:
                pnl_sig = "+" if portfolio_summary['pnl'] >= 0 else ""
                st.metric("Total Gain/Loss (P&L)", f"{pnl_sig}${portfolio_summary['pnl']:,.2f}", f"{pnl_sig}{portfolio_summary['pnl_pct']:.2f}%")
                st.metric("Portfolio Sharpe Ratio", f"{portfolio_summary['sharpe']:.2f}")
                
            col_pd1, col_pd2 = st.columns(2)
            with col_pd1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 15px; margin-top: 15px;">
                    <div class="metric-title">Diversification Index</div>
                    <div class="metric-value">{portfolio_summary['diversification']:.1f}%</div>
                    <div style="font-size: 0.9rem; color: #8a99ad; margin-top: 5px;">Risk Assessment: <b>{portfolio_summary['risk_class']}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_pd2:
                # Allocation pie chart
                fig_port_pie = go.Figure(data=[go.Pie(
                    labels=list(st.session_state.portfolio.keys()),
                    values=portfolio_summary['weights'],
                    hole=0.4
                )])
                fig_port_pie.update_layout(
                    title="Portfolio Asset Allocation",
                    template='plotly_dark',
                    height=200,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_port_pie, use_container_width=True)
                
            # Detailed Positions Table
            st.markdown("### Current Active Positions")
            positions_data = []
            for tick, pos in st.session_state.portfolio.items():
                cur_price = portfolio_summary['current_prices'].get(tick, pos['buy_price'])
                current_val = pos['shares'] * cur_price
                invested_val = pos['shares'] * pos['buy_price']
                val_pnl = current_val - invested_val
                val_pnl_pct = (val_pnl / invested_val * 100.0) if invested_val > 0 else 0.0
                
                positions_data.append({
                    "Ticker": tick,
                    "Shares": pos['shares'],
                    "Avg Buy Price": f"${pos['buy_price']:.2f}",
                    "Current Price": f"${cur_price:.2f}",
                    "Total Cost": f"${invested_val:,.2f}",
                    "Current Value": f"${current_val:,.2f}",
                    "P&L ($)": f"${val_pnl:,.2f}",
                    "P&L (%)": f"{val_pnl_pct:+.2f}%"
                })
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("Add stock tickers on the left sidebar to start tracking your portfolio positions.")
# --- TAB 8: STOCK COMPARISON PANEL ---
with tab8:
    st.markdown("<h3 class='section-header'>Multi-Company Head-to-Head Comparison</h3>", unsafe_allow_html=True)
    
    tickers_to_compare = st.multiselect("Select Tickers to Compare", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V"], default=["AAPL", "MSFT", "GOOGL"])
    
    if len(tickers_to_compare) > 0:
        comp_data = []
        fig_norm = go.Figure()
        
        for tick in tickers_to_compare:
            try:
                hist = fetch_historical_data(tick, period="1y")
                comp_info = fetch_company_info(tick)
                
                if not hist.empty:
                    # Line chart of normalized returns
                    norm_series = (hist['Close'] / hist['Close'].iloc[0]) * 100.0
                    fig_norm.add_trace(go.Scatter(x=hist.index, y=norm_series, name=tick, mode='lines'))
                    
                    comp_data.append({
                        "Ticker": tick,
                        "Name": comp_info.get('longName', tick),
                        "Current Price": format_val(hist['Close'].iloc[-1], 'float', prefix='$'),
                        "Market Cap": format_val(comp_info.get('marketCap'), 'int', prefix='$'),
                        "PE Ratio": format_val(comp_info.get('trailingPE'), 'float'),
                        "Forward PE": format_val(comp_info.get('forwardPE'), 'float'),
                        "Beta": format_val(comp_info.get('beta'), 'float'),
                        "Dividend Yield": format_val(comp_info.get('dividendYield') * 100.0 if comp_info.get('dividendYield') is not None else None, 'float', suffix='%')
                    })
            except Exception as e:
                continue
                
        if comp_data:
            # Table comparison
            st.markdown("### Financial Metric Comparison Table")
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
            
            # Normalized comparative returns chart
            fig_norm.update_layout(
                title="Relative Stock Return (Normalized to 100)",
                template='plotly_dark',
                height=450,
                xaxis_title="Date",
                yaxis_title="Normalized Price (%)",
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_norm, use_container_width=True)
        else:
            st.warning("Could not download comparison metrics for the selected assets.")
    else:
        st.info("Select at least one ticker from the dropdown input above.")
# --- ADDITIONAL COMPONENT: SECTOR ROTATION ANALYSIS ---
st.markdown("---")
st.markdown("<h2 class='section-header'>Market Sector Performance Analysis</h2>", unsafe_allow_html=True)
with st.spinner("Downloading sector index trends..."):
    sector_df = calculate_sector_rankings()
    
if not sector_df.empty:
    col_sec1, col_sec2 = st.columns([2, 1])
    
    with col_sec1:
        # Plotly horizontal performance chart
        fig_sec = px.bar(
            sector_df,
            x='3M Return (%)',
            y='Sector',
            orientation='h',
            color='3M Return (%)',
            color_continuous_scale='RdYlGn',
            labels={'3M Return (%)': '3-Month Performance (%)'},
            title="Sectors Ranked by 3-Month Performance"
        )
        fig_sec.update_layout(
            template='plotly_dark',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_sec, use_container_width=True)
        
    with col_sec2:
        st.markdown("### Sector Metrics Breakdown")
        st.dataframe(sector_df[['Sector', 'ETF', '3M Return (%)', 'YTD Return (%)']], use_container_width=True)
else:
    st.info("Sector metrics download is currently unavailable.")
