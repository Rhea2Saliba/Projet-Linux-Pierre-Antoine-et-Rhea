import streamlit as st
import numpy as np
from data_loader_multi import load_multi_assets
from portfolio_calculations import compute_daily_returns
from portfolioDashboard import display_portfolio_metrics

tickers = ["AAPL", "MSFT", "^GSPC"]

prices = load_multi_assets(tickers, "2020-01-01", "2020-01-10")
returns = compute_daily_returns(prices)

weights = np.array([1/3, 1/3, 1/3])

display_portfolio_metrics(returns, weights)
