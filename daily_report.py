import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os

# --- 1. CONFIGURATION DU PORTEFEUILLE ---
# On définit ici les actifs que le "Fonds" surveille par défaut.
# Ce sont les mêmes que dans ton Dashboard Quant B.
TICKERS = ["AAPL", "MSFT", "BTC-USD", "AI.PA", "TTE.PA"]
REPORT_DIR = "/home/ubuntu/Projet-Linux-Pierre-Antoine-et-Rhea/reports"

# Création du dossier si nécessaire
os.makedirs(REPORT_DIR, exist_ok=True)

def get_data_and_stats():
    """Récupère les données et calcule les stats du jour"""
    print("1. Téléchargement des données...")
    # On télécharge 1 an d'historique pour les calculs de covariance
    df = yf.download(TICKERS, period="1y", progress=False)
    
    # Nettoyage des données (Gestion du format complexe de yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('Close', axis=1, level=0, drop_level=False)
        except:
            pass # On garde df tel quel si ça échoue
            
    # On garde uniquement les colonnes numériques et on enlève les lignes vides
    df = df.select_dtypes(include=[np.number]).dropna()
    
    # Calcul des rendements quotidiens
    returns = df.pct_change().dropna()
    
    return df, returns

def run_markowitz_simulation(returns):
    """Fait tourner le moteur d'optimisation (Quant B logic)"""
    print("2. Optimisation Markowitz (Monte-Carlo)...")
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 3000 # Nombre de simulations
    rf = 0.03 # Taux sans risque
    
    best_sharpe = -100
    best_weights = []
    best_vol = 0
    best_ret = 0
    
    # Simulation rapide sans boucle Python lente (Vectorisée)
    # Note: Pour un script cron, on reste sur une boucle simple pour la robustesse
    for _ in range(num_portfolios):
        weights = np.random.random(len(TICKERS))
        weights /= np.sum(weights)
        
        # Rendement et Volatilité annuels
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
        # 1. Récupération des données fraîches de 20h00
        df_prices, returns = get_data_and_stats()
        
        # 2. Lancement de l'IA (Markowitz)
        weights, sharpe, vol, ret = run_markowitz_simulation(returns)
        
        # 3. Rédaction du rapport
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""
=========================================================
   RAPPORT DE GESTION - QUANT B (AUTOMATISÉ)
   Date de génération : {today}
=========================================================

1. ANALYSE DU MARCHÉ (Derniers Prix)
------------------------------------
"""
        # On récupère les derniers prix proprement
        last_prices = df_prices.iloc[-1]
        for ticker in TICKERS:
            try:
                # Gestion sécurisée des formats (float/series)
                val = last_prices[ticker]
                if hasattr(val, 'item'): val = val.item()
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
        # 4. Sauvegarde sur le disque dur du serveur
        filename = f"{REPORT_DIR}/report_{datetime.date.today()}.txt"
        with open(filename, "w") as f:
            f.write(content)
            
        print(f"Rapport généré : {filename}")

    except Exception as e:
        # En cas de crash, on note l'erreur dans un fichier de log
        print(f"Erreur fatale : {e}")
        with open(f"{REPORT_DIR}/error.log", "a") as f:
            f.write(f"{datetime.datetime.now()} - ERREUR: {e}\n")

if __name__ == "__main__":
    generate_report()