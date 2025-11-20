import pandas as pd
import numpy as np
import yfinance as yf


# ---------------------------------------------------------
# MÉTRIQUES DE PERFORMANCE
# ---------------------------------------------------------

def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return abs(drawdown.min())


def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.03, annualization_factor=252):
    daily_returns = daily_returns[daily_returns != 0].dropna()

    if daily_returns.empty:
        return 0

    mean_return = daily_returns.mean()
    std_dev_return = daily_returns.std()

 
    try:
        if float(std_dev_return) == 0:
            return 0
    except:
        return 0

    daily_rf = (1 + risk_free_rate) ** (1/annualization_factor) - 1
    return (mean_return - daily_rf) / std_dev_return * np.sqrt(annualization_factor)


# ---------------------------------------------------------
# STRATÉGIES
# ---------------------------------------------------------

def run_all_strategies(prices, ticker, mom_window, short_w, long_w, bb_window, n_std_dev):

    initial_investment = 100
    df = pd.DataFrame()
    metrics = {}

    # Daily returns
    daily_returns = prices.pct_change().fillna(0)

    # Normalisé
    df["Raw_Price"] = (prices / prices.iloc[0]) * initial_investment

    # BUY & HOLD
    df["Buy_and_Hold"] = (1 + daily_returns).cumprod() * initial_investment
    metrics["Buy_and_Hold"] = {
        "Sharpe": calculate_sharpe_ratio(daily_returns),
        "Max Drawdown": calculate_max_drawdown(df["Buy_and_Hold"]),
        "Performance Totale": df["Buy_and_Hold"].iloc[-1] / initial_investment - 1
    }

    # MOMENTUM (SMA)
    sma = prices.rolling(window=mom_window).mean()
    signal_mom = (prices > sma).astype(int)
    strat_mom = daily_returns.shift(-1) * signal_mom.shift(1)
    df[f"Momentum_{mom_window}"] = (1 + strat_mom).cumprod() * initial_investment
    metrics[f"Momentum_{mom_window}"] = {
        "Sharpe": calculate_sharpe_ratio(strat_mom),
        "Max Drawdown": calculate_max_drawdown(df[f"Momentum_{mom_window}"]),
        "Performance Totale": df[f"Momentum_{mom_window}"].iloc[-1] / initial_investment - 1
    }

    # CROSSOVER
    sma_short = prices.rolling(short_w).mean()
    sma_long = prices.rolling(long_w).mean()
    signal_cross = (sma_short > sma_long).astype(int)
    strat_cross = daily_returns.shift(-1) * signal_cross.shift(1)
    df[f"Cross_{short_w}_{long_w}"] = (1 + strat_cross).cumprod() * initial_investment
    metrics[f"Cross_{short_w}_{long_w}"] = {
        "Sharpe": calculate_sharpe_ratio(strat_cross),
        "Max Drawdown": calculate_max_drawdown(df[f"Cross_{short_w}_{long_w}"]),
        "Performance Totale": df[f"Cross_{short_w}_{long_w}"].iloc[-1] / initial_investment - 1
    }

    # BOLLINGER BANDS
    mid = prices.rolling(bb_window).mean()
    std = prices.rolling(bb_window).std()
    lower = mid - n_std_dev * std

    buy = (prices < lower).astype(int)
    position = buy.ffill().fillna(0)
    position[prices > mid] = 0

    strat_bb = daily_returns.shift(-1) * position.shift(1)
    df[f"BB_{bb_window}_{n_std_dev}"] = (1 + strat_bb).cumprod() * initial_investment
    metrics[f"BB_{bb_window}_{n_std_dev}"] = {
        "Sharpe": calculate_sharpe_ratio(strat_bb),
        "Max Drawdown": calculate_max_drawdown(df[f"BB_{bb_window}_{n_std_dev}"]),
        "Performance Totale": df[f"BB_{bb_window}_{n_std_dev}"].iloc[-1] / initial_investment - 1
    }

    return df, metrics
