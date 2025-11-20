import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date


# =============================================================
# 1 — DATA LOADER (plus besoin du fichier data_loader_multi.py)
# =============================================================
def load_multi_assets(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data[["Close"]]
        data.columns = tickers

    data = data.dropna()
    return data


# =============================================================
# 2 — MARKOWITZ FONCTIONS (tout est ici, aucun import)
# =============================================================

def compute_returns(prices):
    """Daily returns"""
    return prices.pct_change().dropna()


def optimize_markowitz(returns):
    """Minimum variance portfolio (Markowitz)"""
    cov = returns.cov()
    n = len(returns.columns)

    inv_cov = np.linalg.inv(cov.values)
    ones = np.ones(n)

    weights = inv_cov.dot(ones)
    weights = weights / weights.sum()

    annual_return = np.dot(weights, returns.mean()) * 252
    annual_vol = np.sqrt(weights.T @ (cov * 252).values @ weights)

    return weights, annual_return, annual_vol


def compute_portfolio_cumulative(returns, weights):
    """Cumulative curve over time"""
    port_daily = returns.dot(weights)
    cumulative = (1 + port_daily).cumprod() * 100
    return cumulative


# =============================================================
# 3 — STREAMLIT DASHBOARD
# =============================================================

st.set_page_config(layout="wide")
st.title("📊 Portfolio Dashboard – Optimisation Markowitz Multi-Assets")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Paramètres")

    tickers = st.multiselect(
        "Sélectionne plusieurs actifs :",
        ["AAPL", "MSFT", "AMZN", "GOOGL", "^GSPC", "BTC-USD", "ETH-USD", "GLD", "EURUSD=X"],
        default=["AAPL", "MSFT", "BTC-USD"]
    )

    start = st.date_input("Date de début", date(2020, 1, 1))
    end = st.date_input("Date de fin", date.today())

    run = st.button("🚀 Lancer l’analyse")


# ---------------- Main ----------------
if run:

    if len(tickers) < 2:
        st.error("❗ Choisis au moins **2 actifs**.")
        st.stop()

    st.info(f"Téléchargement des données pour : {', '.join(tickers)} …")

    prices = load_multi_assets(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if prices.empty:
        st.error("❌ Impossible de télécharger les prix.")
        st.stop()

    st.success("✔ Données chargées")

    # RETURNS
    returns_df = compute_returns(prices)

    # MARKOWITZ OPTIMIZATION
    weights, expected_ret, expected_vol = optimize_markowitz(returns_df)

    # RESULTATS
    st.subheader("🎯 Résultats Markowitz – Minimum Volatility Portfolio")

    c1, c2 = st.columns(2)
    c1.metric("📈 Rendement annuel", f"{expected_ret*100:.2f} %")
    c2.metric("📉 Volatilité annuelle", f"{expected_vol*100:.2f} %")

    # Tableau des poids
    st.subheader("📊 Poids Optimaux")
    st.table(pd.DataFrame({"Actif": tickers, "Poids (%)": (weights*100).round(2)}))

    # COURBE CUMULÉE
    st.subheader("📈 Valeur cumulée du portefeuille (base 100)")
    cumulative = compute_portfolio_cumulative(returns_df, weights)
    st.line_chart(cumulative)

    # Raw Data
    with st.expander("📦 Données brutes"):
        st.dataframe(prices)
