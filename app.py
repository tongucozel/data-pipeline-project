"""
Financial Data Pipeline - Dashboard
====================================
Interactive dashboard for stock data analysis.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# Page config
st.set_page_config(
    page_title="Financial Data Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean look
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    .stMetric {background-color: #f8f9fa; padding: 1rem; border-radius: 4px;}
    h1, h2, h3 {font-weight: 500;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df = df.reset_index()
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate key metrics from stock data."""
    if df.empty:
        return {}

    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[0]
    price_change = ((current_price - prev_price) / prev_price) * 100

    return {
        "current_price": current_price,
        "price_change": price_change,
        "high": df["High"].max(),
        "low": df["Low"].min(),
        "avg_volume": df["Volume"].mean(),
        "volatility": df["Close"].pct_change().std() * (252 ** 0.5) * 100,
    }


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    df = df.copy()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Volatility"] = df["Close"].pct_change().rolling(window=20).std() * (252 ** 0.5)
    return df


def create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create price chart with moving averages."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price", "Volume", "Volatility")
    )

    # Price and MAs
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#1f77b4", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MA_20"], name="MA 20", line=dict(color="#ff7f0e", width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MA_50"], name="MA 50", line=dict(color="#2ca02c", width=1)),
        row=1, col=1
    )

    # Volume
    colors = ["#ef553b" if df["Close"].iloc[i] < df["Open"].iloc[i] else "#00cc96"
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color=colors, showlegend=False),
        row=2, col=1
    )

    # Volatility
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Volatility"], name="Volatility",
                   fill="tozeroy", line=dict(color="#d62728", width=1)),
        row=3, col=1
    )

    fig.update_layout(
        title=f"{symbol} Stock Analysis",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Volatility", row=3, col=1)

    return fig


def main():
    # Header
    st.title("Financial Data Pipeline")
    st.markdown("ETL pipeline for stock market data analysis")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        symbol = st.selectbox(
            "Stock Symbol",
            options=["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA"],
            index=0,
        )

        period = st.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=3,
            format_func=lambda x: {
                "1mo": "1 Month",
                "3mo": "3 Months",
                "6mo": "6 Months",
                "1y": "1 Year",
                "2y": "2 Years"
            }[x]
        )

        st.markdown("---")
        st.markdown("**Pipeline Steps**")
        st.markdown("1. Extract: Yahoo Finance API")
        st.markdown("2. Transform: Calculate indicators")
        st.markdown("3. Load: Display results")
        st.markdown("4. Validate: Data quality checks")

        st.markdown("---")
        st.markdown("**Tech Stack**")
        st.code("Python, Pandas, yfinance\nSQLite, Pydantic, Pytest\nDocker, GitHub Actions", language=None)

    # Fetch data
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(symbol, period)
        df = add_technical_indicators(df)
        metrics = calculate_metrics(df)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Price",
            value=f"${metrics['current_price']:.2f}",
            delta=f"{metrics['price_change']:.2f}%"
        )

    with col2:
        st.metric(
            label="52W High",
            value=f"${metrics['high']:.2f}"
        )

    with col3:
        st.metric(
            label="52W Low",
            value=f"${metrics['low']:.2f}"
        )

    with col4:
        st.metric(
            label="Volatility (Ann.)",
            value=f"{metrics['volatility']:.1f}%"
        )

    st.markdown("---")

    # Chart
    fig = create_price_chart(df, symbol)
    st.plotly_chart(fig, use_container_width=True)

    # Data tables
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent Data")
        display_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(10).copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        for col in ["Open", "High", "Low", "Close"]:
            display_df[col] = display_df[col].round(2)
        display_df["Volume"] = display_df["Volume"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Statistics")
        stats = pd.DataFrame({
            "Metric": [
                "Mean Price",
                "Std Dev",
                "Min Price",
                "Max Price",
                "Total Volume",
                "Avg Daily Return"
            ],
            "Value": [
                f"${df['Close'].mean():.2f}",
                f"${df['Close'].std():.2f}",
                f"${df['Close'].min():.2f}",
                f"${df['Close'].max():.2f}",
                f"{df['Volume'].sum():,.0f}",
                f"{df['Daily_Return'].mean():.3f}%"
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.caption(f"Data fetched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Source: Yahoo Finance")


if __name__ == "__main__":
    main()
