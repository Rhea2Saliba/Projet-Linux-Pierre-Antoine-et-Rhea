import streamlit as st
import pandas as pd
import numpy as np

def display_portfolio_metrics(returns_df, weights):
    st.header("Portfolio Metrics")

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = returns_df.corr()
    st.dataframe(correlation_matrix.style.background_gradient(cmap="coolwarm"))

    # Individual Asset Volatility
    st.subheader("Asset Volatility (Annualized)")
    asset_volatility = returns_df.std() * np.sqrt(252)
    st.write(asset_volatility)

    # Portfolio Volatility
    st.subheader("Portfolio Volatility (Annualized)")
    covariance_matrix = returns_df.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    st.write(f"Portfolio volatility: {portfolio_volatility:.4f}")

    # Diversification Effect
    st.subheader("Diversification Effect")
    average_volatility = asset_volatility.mean()
    st.write(f"Average asset volatility: {average_volatility:.4f}")
    st.write(f"Portfolio volatility: {portfolio_volatility:.4f}")

    if portfolio_volatility < average_volatility:
        st.write("Diversification reduces risk (portfolio volatility is lower than the average asset volatility).")
    else:
        st.write("No diversification effect detected.")

    # Portfolio Returns
    st.subheader("Portfolio Returns")
    portfolio_returns = returns_df.dot(weights)
    st.line_chart(portfolio_returns.cumsum())
    st.write(portfolio_returns.describe())
