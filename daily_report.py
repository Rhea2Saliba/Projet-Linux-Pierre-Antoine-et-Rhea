import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os

# --- CONFIGURATION ---
TICKERS = ["AAPL", "MSFT", "BTC-USD", "AI.PA", "TTE.PA"]
REPORT_DIR = "/home/ubuntu/Projet-Linux-Pierre-Antoine-et-Rhea/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def get_data_and_stats():
    # Téléchargement
    df = yf.download(TICKERS, period="1y", progress=False)
    
    # --- FIX N/A : Aplatir proprement les colonnes ---
    # Si on a des colonnes à étages (Price, Ticker), on garde que le Ticker
    if isinstance(df.columns, pd.MultiIndex):
        # On extrait 'Close' et on SUPPRIME le niveau du haut (drop_level=True)
        try:
            df = df.xs('Close', axis=1, level=0, drop_level=True)
        except:
            pass # Si ça rate, on laisse tel quel

    # Nettoyage
    df = df.select_dtypes(include=[np.number]).dropna()
    returns = df.pct_change().dropna()
    return df, returns

def run_markowitz_simulation(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 3000
    rf = 0.03
    
    best_sharpe = -100
    best_weights = []
    best_vol = 0
    best_ret = 0
    
    for _ in range(num_portfolios):
        weights = np.random.random(len(TICKERS))
        weights /= np.sum(weights)
        
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe = (port_return - rf) / port_vol
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
            best_vol = port_vol
            best_ret = port_return
            
    return best_weights, best_sharpe, best_vol, best_ret

def generate_report():
    try:
        df_prices, returns = get_data_and_stats()
        weights, sharpe, vol, ret = run_markowitz_simulation(returns)
        
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""
=========================================================
   RAPPORT DE GESTION - QUANT B (AUTOMATISÉ)
   Date : {today}
=========================================================

1. ANALYSE DU MARCHÉ (Derniers Prix)
------------------------------------
"""
        # Récupération des prix corrigée
        last_prices = df_prices.iloc[-1]
        for ticker in TICKERS:
            try:
                val = last_prices[ticker]
                content += f"  - {ticker: <10} : {val:.2f} $\n"
            except:
                content += f"  - {ticker: <10} : N/A\n"

        content += f"""
------------------------------------

2. STRATÉGIE OPTIMALE (Markowitz)
------------------------------------
Basé sur 3000 simulations Monte-Carlo.

> Performance Attendue (Annuelle) : {ret:.2%}
> Risque (Volatilité)             : {vol:.2%}
> Score de Sharpe                 : {sharpe:.2f}

ALLOCATION RECOMMANDÉE POUR DEMAIN :
"""
        for ticker, w in zip(TICKERS, weights):
            content += f"  - {ticker: <10} : {w*100:.2f} %\n"

        content += f"""
=========================================================
Statut : Succès
Serveur : AWS EC2 - Cron Job
=========================================================
"""
        filename = f"{REPORT_DIR}/report_{datetime.date.today()}.txt"
        with open(filename, "w") as f:
            f.write(content)
        print(f"Rapport généré : {filename}")

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    generate_report()