from data_loader_multi import load_multi_assets
from portfolio_calculations import compute_daily_returns

# --- Test Step 2 : compute_daily_returns ---
tickers = ['AAPL', 'MSFT', '^GSPC']
start_date = '2020-01-01'
end_date = '2020-03-01'

print("\n--- TEST : Chargement multi-actifs ---")
prices = load_multi_assets(tickers, start_date, end_date)
print(prices.head())

print("\n--- TEST : Rendements journaliers ---")
returns = compute_daily_returns(prices)
print(returns.head())
