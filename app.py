import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SmartTrade AI", page_icon="📈", layout="wide")

# --- HELPER FUNCTIONS FOR TECHNICAL ANALYSIS ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short=12, long=26, signal=9):
    exp1 = data['Close'].ewm(span=short, adjust=False).mean()
    exp2 = data['Close'].ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🤖 SmartTrade AI")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Navigation", 
    ["📊 Portfolio Dashboard", "📈 Advanced Charting", "🛡️ Risk Management", "🧠 AI & Sentiment Insights"]
)

# --- 1. PORTFOLIO DASHBOARD ---
if menu == "📊 Portfolio Dashboard":
    st.title("📊 Portfolio Dashboard")
    st.markdown("Real-time tracking of your assets and overall P&L.")
    
    # Mock Portfolio Data
    portfolio_data = {
        'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS'],
        'Qty': [50, 20, 100, 75],
        'Avg_Price': [2400.50, 3200.00, 1500.25, 1400.00],
        'LTP': [2950.00, 3900.00, 1450.00, 1650.00] # Mock Latest Traded Price
    }
    df_port = pd.DataFrame(portfolio_data)
    df_port['Invested'] = df_port['Qty'] * df_port['Avg_Price']
    df_port['Current_Value'] = df_port['Qty'] * df_port['LTP']
    df_port['P&L'] = df_port['Current_Value'] - df_port['Invested']
    df_port['P&L %'] = (df_port['P&L'] / df_port['Invested']) * 100

    # Top Level Metrics
    total_invested = df_port['Invested'].sum()
    total_current = df_port['Current_Value'].sum()
    total_pnl = df_port['P&L'].sum()
    total_pnl_pct = (total_pnl / total_invested) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Invested", f"₹{total_invested:,.2f}")
    col2.metric("Current Value", f"₹{total_current:,.2f}", f"{total_pnl_pct:.2f}%")
    col3.metric("Total P&L", f"₹{total_pnl:,.2f}", "Profit" if total_pnl > 0 else "Loss")

    st.markdown("### Your Holdings")
    st.dataframe(df_port.style.format({
        'Avg_Price': '₹{:.2f}', 'LTP': '₹{:.2f}', 'Invested': '₹{:.2f}', 
        'Current_Value': '₹{:.2f}', 'P&L': '₹{:.2f}', 'P&L %': '{:.2f}%'
    }), use_container_width=True)

    # Asset Allocation Pie Chart
    fig = go.Figure(data=[go.Pie(labels=df_port['Symbol'], values=df_port['Current_Value'], hole=.4)])
    fig.update_layout(title_text="Asset Allocation", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --- 2. ADVANCED CHARTING ---
elif menu == "📈 Advanced Charting":
    st.title("📈 Advanced Charting & Technicals")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Enter Ticker Symbol (e.g., RELIANCE.NS, AAPL, TSLA)", value="RELIANCE.NS")
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col3:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            data = yf.download(ticker, period=period, interval=interval)
            
        if not data.empty:
            # Flatten MultiIndex columns if necessary (handling yfinance updates)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Calculate Indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = calculate_rsi(data)
            data['MACD'], data['Signal'] = calculate_macd(data)

            # Create Subplots: Candlestick + Volume + RSI
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_heights=[0.6, 0.2, 0.2])

            # Candlestick
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                         low=data['Low'], close=data['Close'], name='Price'), 
                          row=1, col=1)
            
            # SMAs
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color='orange', width=1.5), name='SMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='blue', width=1.5), name='SMA 50'), row=1, col=1)

            # Volume
            colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in data.iterrows()]
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", title=f"{ticker} Technical Chart", showlegend=False)
            fig.update_xaxes(rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Quick Stats
            st.markdown("### Current Technical Summary")
            c1, c2, c3, c4 = st.columns(4)
            latest = data.iloc[-1]
            c1.metric("LTP", f"{latest['Close']:.2f}")
            c2.metric("RSI (14)", f"{latest['RSI']:.2f}", "Overbought" if latest['RSI']>70 else "Oversold" if latest['RSI']<30 else "Neutral")
            c3.metric("SMA 20", f"{latest['SMA_20']:.2f}")
            trend = "Bullish 🟢" if latest['SMA_20'] > latest['SMA_50'] else "Bearish 🔴"
            c4.metric("Trend (20/50 SMA)", trend)

        else:
            st.error("Invalid Ticker or No Data Found. Please try again.")

# --- 3. RISK MANAGEMENT ---
elif menu == "🛡️ Risk Management":
    st.title("🛡️ Risk Management & Position Sizing")
    st.markdown("Calculate exactly how many shares to buy to protect your capital.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Account Details")
        capital = st.number_input("Total Trading Capital (₹)", min_value=1000, value=100000, step=1000)
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        
    with col2:
        st.subheader("Trade Setup")
        entry_price = st.number_input("Entry Price (₹)", min_value=1.0, value=500.0, step=1.0)
        stop_loss = st.number_input("Stop Loss (₹)", min_value=0.1, value=480.0, step=1.0)

    if entry_price <= stop_loss:
        st.error("Stop Loss must be strictly lower than Entry Price for a Long trade.")
    else:
        # Calculations
        risk_amount = capital * (risk_pct / 100)
        risk_per_share = entry_price - stop_loss
        position_size = int(risk_amount // risk_per_share)
        total_investment = position_size * entry_price
        
        target_1_2 = entry_price + (risk_per_share * 2)
        target_1_3 = entry_price + (risk_per_share * 3)

        st.markdown("---")
        st.subheader("📊 Trade Execution Plan")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Max Allowed Risk", f"₹{risk_amount:.2f}")
        r2.metric("Qty to Buy", f"{position_size} Shares")
        r3.metric("Total Investment", f"₹{total_investment:.2f}")
        r4.metric("Risk/Share", f"₹{risk_per_share:.2f}")

        st.markdown("### Targets (Risk:Reward)")
        t1, t2 = st.columns(2)
        t1.success(f"**Target 1 (1:2 RR):** ₹{target_1_2:.2f} (Profit: ₹{risk_amount * 2:.2f})")
        t2.success(f"**Target 2 (1:3 RR):** ₹{target_1_3:.2f} (Profit: ₹{risk_amount * 3:.2f})")

# --- 4. AI & SENTIMENT INSIGHTS ---
elif menu == "🧠 AI & Sentiment Insights":
    st.title("🧠 AI & Market Sentiment (Simulated)")
    st.markdown("AI-driven analysis of news headlines, SEC filings, and social media trends.")

    ticker_ai = st.text_input("Enter Ticker for AI Analysis", value="RELIANCE.NS")
    
    if st.button("Generate AI Report"):
        with st.spinner("Analyzing millions of data points, tweets, and news articles..."):
            # Simulating AI processing delay
            import time
            time.sleep(2)
            
            # Generate Mock Sentiment based on random distribution
            sentiment_score = np.random.uniform(20, 90)
            
            if sentiment_score > 65:
                status = "Bullish 🚀"
                color = "green"
            elif sentiment_score < 40:
                status = "Bearish 📉"
                color = "red"
            else:
                status = "Neutral ⚖️"
                color = "yellow"

            st.markdown(f"### Overall AI Sentiment for {ticker_ai}: :{color}[**{status}**] (Score: {sentiment_score:.1f}/100)")
            st.progress(int(sentiment_score))
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📰 Recent News Sentiment")
                news = [
                    {"headline": f"{ticker_ai} reports strong quarterly earnings, beats estimates.", "sentiment": "Positive"},
                    {"headline": f"Global supply chain issues might impact {ticker_ai}'s margins.", "sentiment": "Negative"},
                    {"headline": f"New institutional buying detected in {ticker_ai} block deals.", "sentiment": "Positive"}
                ]
                for n in news:
                    st.write(f"- {n['headline']} (**{n['sentiment']}**)")
                    
            with col2:
                st.subheader("🤖 Algorithmic Prediction")
                st.info(f"**Short Term (1 Week):** Our ML model predicts a {np.random.randint(55, 85)}% probability of an upward breakout based on historical volume patterns.")
                st.warning(f"**Volatility Alert:** Expected price fluctuation of ±{np.random.uniform(2.5, 5.0):.1f}% in the next 3 trading sessions.")

st.sidebar.markdown("---")
st.sidebar.caption("Made with Streamlit & Python")
