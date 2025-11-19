import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. FONCTIONS DE PERFORMANCE (ASSUREZ-VOUS QUE CES FONCTIONS SONT BIEN DÉFINIES AU DÉBUT !) ---

def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return abs(drawdown.min()) 

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.03, annualization_factor=252):
    daily_returns = daily_returns[daily_returns != 0].dropna()
    mean_return = daily_returns.mean()
    std_dev_return = daily_returns.std()
    daily_risk_free_rate = (1 + risk_free_rate)**(1/annualization_factor) - 1
    sharpe_ratio = (mean_return - daily_risk_free_rate) / std_dev_return * np.sqrt(annualization_factor)
    return sharpe_ratio

# --- 2. CONFIGURATION ET RÉCUPÉRATION DES DONNÉES ---
ticker_symbol = '^GSPC'
start_date = '2020-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
initial_investment = 100 

print(f"--- Démarrage de l'analyse Buy-and-Hold pour le {ticker_symbol} ({start_date} à {end_date}) ---")

try:
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    asset_prices = data[['Close']].copy() # Utilisez .copy() pour éviter SettingWithCopyWarning
    asset_prices.columns = [ticker_symbol]

except Exception as e:
    print(f"Erreur fatale lors de la récupération des données : {e}")
    exit() 

# --- 3. STRATÉGIE BUY-AND-HOLD ---

# Calcul des rendements journaliers (CORRECTION DE LA NAMEERROR)
daily_returns = asset_prices[ticker_symbol].pct_change().fillna(0)

# Calcul de la valeur cumulative de la stratégie Buy-and-Hold
cumulative_value_buy_hold = (1 + daily_returns).cumprod() * initial_investment
cumulative_value_buy_hold.name = 'Buy_and_Hold_Value'

# --- 4. PRÉPARATION DES DONNÉES POUR LE DASHBOARD ---

# Normalisation du prix brut pour la comparaison (base 100)
normalized_raw_price = (asset_prices[ticker_symbol] / asset_prices[ticker_symbol].iloc[0]) * initial_investment
normalized_raw_price.name = 'Raw_Price_Value'

# DataFrame final pour le graphique
final_df = pd.concat([normalized_raw_price, cumulative_value_buy_hold], axis=1)

# --- 5. CALCUL ET AFFICHAGE DES MÉTRIQUES (Buy-and-Hold) ---
max_dd_bh = calculate_max_drawdown(cumulative_value_buy_hold)
sharpe_bh = calculate_sharpe_ratio(daily_returns, risk_free_rate=0.03) 
print(f"\n--- Performance Buy-and-Hold ---")
print(f"Sharpe Ratio : {sharpe_bh:.2f}, Max Drawdown : {max_dd_bh:.2%}")

# --- 6. IMPLÉMENTATION DE LA STRATÉGIE DE MOMENTUM (MMS) ---
WINDOW_SIZE = 50 
STRATEGY_NAME = 'Momentum_50D_MMS'

# 1. Calculer la Moyenne Mobile Simple (MMS)
# Correction : Utilisation de .loc pour éviter le SettingWithCopyWarning
asset_prices.loc[:, 'MMS_50'] = asset_prices[ticker_symbol].rolling(window=WINDOW_SIZE).mean()

# 2. Générer le signal de trading (Signal)
# Correction : Utilisation de .loc pour éviter le SettingWithCopyWarning
asset_prices.loc[:, 'Signal'] = np.where(asset_prices[ticker_symbol] > asset_prices['MMS_50'], 1.0, 0.0)

# 3. Calculer les rendements de la stratégie
# On utilise les rendements journaliers (daily_returns) qui sont maintenant définis !
strategy_returns = daily_returns.shift(-1) * asset_prices['Signal'].shift(1)
strategy_returns = strategy_returns.fillna(0)

# 4. Calculer la valeur cumulative de la stratégie de Momentum
cumulative_value_momentum = (1 + strategy_returns).cumprod() * initial_investment
cumulative_value_momentum.name = STRATEGY_NAME


# --- 7. MISE À JOUR DU DASHBOARD ET DES MÉTRIQUES ---

# A. Mettre à jour le DataFrame final pour le graphique
final_df[STRATEGY_NAME] = cumulative_value_momentum

# B. Calculer les métriques pour la stratégie Momentum
max_dd_mom = calculate_max_drawdown(cumulative_value_momentum)
sharpe_mom = calculate_sharpe_ratio(strategy_returns, risk_free_rate=0.03)

print("\n--- Aperçu des performances de la stratégie Momentum ---")
print(f"Sharpe Ratio annualisé : {sharpe_mom:.2f}, Max Drawdown : {max_dd_mom:.2%}") 
print(f"Performance Totale: {(cumulative_value_momentum.iloc[-1] - initial_investment) / initial_investment:.2%}")

print("\n--- Tableau de comparaison (dernières lignes) ---")
print(final_df[['Raw_Price_Value', 'Buy_and_Hold_Value', STRATEGY_NAME]].tail())