import streamlit as st
from datetime import date
import pandas as pd

# Import correct : load SINGLE asset
from data_loader_multi import load_single_asset

# Import correct : backtests
from portfolio_calculations import run_all_strategies

# ---------------------------
# CONFIG STREAMLIT
# ---------------------------
st.set_page_config(layout="wide")
st.title("Portfolio Dashboard – Multi Assets with Bitcoin")


# ---------------------------
# SIDEBAR PARAMETERS
# ---------------------------
with st.sidebar:
    st.header("Paramètres")

    ticker = st.selectbox(
        "Actif :",
        ["AAPL", "MSFT", "^GSPC", "BTC-USD", "ETH-USD", "EURUSD=X"],
        index=3
    )

    start = st.date_input("Début", date(2020, 1, 1))
    end = st.date_input("Fin", date.today())

    mom = st.slider("Momentum SMA", 10, 100, 50)
    sw = st.slider("Short Window", 5, 50, 20)
    lw = st.slider("Long Window", 50, 300, 100)
    bb = st.slider("Bollinger Window", 10, 50, 20)
    std = st.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)

    run = st.button("Lancer le Backtest")


# ---------------------------
# MAIN AREA
# ---------------------------
if run:

    st.info(f"Téléchargement des données pour **{ticker}** ...")

    # Load price series (Close)
    prices_df = load_single_asset(ticker, start, end)

    # Convert the DataFrame 1-column → Series
    prices = prices_df[ticker]

    # Run all strategies
    df, metrics = run_all_strategies(
        prices,
        ticker,
        mom,
        sw,
        lw,
        bb,
        std
    )

    st.success("Backtest terminé ✔")

    # ---- CHART ----
    st.header("📈 Performance cumulée des stratégies")
    st.line_chart(df)

    # ---- METRICS ----
    st.header("📊 Metrics de Performance")
    mdf = pd.DataFrame(metrics).T
    mdf.columns = ["Sharpe", "Max Drawdown", "Performance Totale"]

    # Formatting for readability
    mdf["Sharpe"] = mdf["Sharpe"].map(lambda x: f"{x:.2f}")
    mdf["Max Drawdown"] = mdf["Max Drawdown"].map(lambda x: f"{x:.2%}")
    mdf["Performance Totale"] = mdf["Performance Totale"].map(lambda x: f"{x:.2%}")

    st.dataframe(mdf)
