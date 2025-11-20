import yfinance as yf
import pandas as pd

# ============================================================
#  LOAD A SINGLE ASSET (for univariate backtests)
# ============================================================
def load_single_asset(ticker, start_date, end_date):
    """
    Downloads a single asset and returns a clean price DataFrame
    with only the 'Close' column renamed as the ticker.

    Returns:
        DataFrame with one column = asset close prices
    """
    data = yf.download(ticker, start=start_date, end=end_date)

    if "Close" not in data.columns:
        raise KeyError(f" yfinance n’a pas renvoyé de 'Close' pour {ticker} ")

    df = data[["Close"]].copy()
    df.columns = [ticker]  # rename column

    return df


# ============================================================
#  LOAD MULTIPLE ASSETS AT ONCE (SP500 + Bitcoin + Gold etc.)
# ============================================================
def load_multi_assets(tickers, start_date, end_date):
    """
    Downloads multiple assets at once and returns a clean merged DataFrame.

    Returns:
        DataFrame with columns = tickers, values = Close prices
    """
    data = yf.download(tickers, start=start_date, end=end_date)

    # Case: yfinance returns multi-index columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data["Close"].copy()
    else:
        # Case: only 1 ticker, fallback to single column
        close_prices = data[["Close"]].copy()
        close_prices.columns = tickers

    # Remove rows with all NaN
    close_prices = close_prices.dropna(how="all")

    return close_prices
