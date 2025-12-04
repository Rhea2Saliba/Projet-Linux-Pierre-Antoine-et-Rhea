import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os

# --- CONFIGURATION ---
TICKERS = ["AAPL", "MSFT", "BTC-USD", "AI.PA", "TTE.PA"]
REPORT_DIR = "/home/ubuntu/Projet-Linux-Pierre-Antoine-et-Rhea/reports"
INITIAL_CAPITAL = 10000  # capital de départ du portefeuille

os.makedirs(REPORT_DIR, exist_ok=True)


def get_data_and_stats():
    """
    Télécharge 1 an de données pour les tickers,
    corrige les colonnes MultiIndex, renvoie prix & rendements.
    """
    df = yf.download(TICKERS, period="1y", progress=False)

    # --- FIX N/A : aplatir correctement les colonnes si MultiIndex ---
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # On garde uniquement les prix de clôture, avec les tickers en colonnes
            df = df.xs("Close", axis=1, level=0, drop_level=True)
        except Exception:
            pass  # si ça rate, on laisse df comme il est

    # Nettoyage : on garde que les colonnes numériques et on enlève les lignes avec NaN
    df = df.select_dtypes(include=[np.number]).dropna()

    # Rendements journaliers
    returns = df.pct_change().dropna()

    return df, returns


def compute_equal_weight_portfolio(returns, initial_capital=INITIAL_CAPITAL):
    """
    Portefeuille égal-pondéré (20% par actif si 5 actifs).
    Rebalancement à chaque période (ici chaque jour).
    Retourne :
      - courbe du portefeuille
      - rendement annuel
      - volatilité annuelle
      - Sharpe (avec rf = 3%)
    """
    rf = 0.03  # taux sans risque annuel

    # On garde seulement les tickers disponibles dans les colonnes
    available_tickers = [t for t in TICKERS if t in returns.columns]
    if len(available_tickers) == 0:
        raise ValueError("Aucun ticker disponible dans les données de rendement.")

    # Poids égal-pondéré
    w = np.ones(len(available_tickers)) / len(available_tickers)

    # Rendements du portefeuille (rebalancement à chaque période)
    port_returns = returns[available_tickers].dot(w)

    # Courbe de valeur du portefeuille
    port_curve = (1 + port_returns).cumprod() * initial_capital

    # Statistiques
    mean_daily = port_returns.mean()
    std_daily = port_returns.std()

    annual_return = mean_daily * 252
    annual_vol = std_daily * np.sqrt(252)
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0.0

    stats = {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
    }

    return port_curve, stats, available_tickers, w


def generate_report():
    try:
        df_prices, returns = get_data_and_stats()

        # Calcul du portefeuille égal-pondéré (rebalancé)
        port_curve, stats, used_tickers, w = compute_equal_weight_portfolio(returns)

        # Dernier prix dispo
        last_prices = df_prices.iloc[-1]

        # Dernière valeur du portefeuille
        last_port_value = port_curve.iloc[-1]

        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""
=========================================================
   RAPPORT DE GESTION - QUANT B (AUTOMATISÉ)
   Date : {today}
=========================================================

1. ANALYSE DU MARCHÉ (Derniers Prix)
------------------------------------
"""
        for ticker in TICKERS:
            try:
                val = last_prices[ticker]
                content += f"  - {ticker: <10} : {val:.2f} $\n"
            except Exception:
                content += f"  - {ticker: <10} : N/A\n"

        content += f"""
------------------------------------

2. PORTEFEUILLE ÉGAL-PONDÉRÉ (REBALANCÉ)
----------------------------------------
Capital initial supposé : {INITIAL_CAPITAL:,.2f} $

> Valeur actuelle théorique du portefeuille : {last_port_value:,.2f} $
> Performance cumulée sur la période      : {(last_port_value/INITIAL_CAPITAL - 1):.2%}
> Rendement annuel (approx.)              : {stats['annual_return']:.2%}
> Volatilité annuelle (approx.)           : {stats['annual_vol']:.2%}
> Sharpe (rf = 3%)                        : {stats['sharpe']:.2f}

Allocation (rebalancée à chaque période de marché) :
"""
        # Comme on est égal-pondéré : même poids pour tous les tickers utilisés
        for ticker, weight in zip(used_tickers, w):
            content += f"  - {ticker: <10} : {weight*100:.2f} %\n"

        content += f"""
=========================================================
Statut : Succès
Serveur : AWS EC2 - Cron Job (exécution régulière, ex. toutes les 5 minutes)
Commentaire : Le portefeuille est théoriquement rebalancé à poids égaux
              à chaque nouvelle période de marché (ici chaque point de données).
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
