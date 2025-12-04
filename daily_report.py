import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os

# --- CONFIGURATION GÉNÉRALE ---

TICKERS = ["AAPL", "MSFT", "BTC-USD", "AI.PA", "TTE.PA"]

REPORT_DIR = "/home/ubuntu/Projet-Linux-Pierre-Antoine-et-Rhea/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(REPORT_DIR, "portfolio_history.csv")

INITIAL_CAPITAL = 10000.0  # capital de départ
TARGET_WEIGHTS = np.array([1 / len(TICKERS)] * len(TICKERS))  # 20 % chacun


def get_last_prices():
    """
    Récupère les derniers prix de clôture pour tous les tickers.
    """
    df = yf.download(TICKERS, period="1d", interval="1m", progress=False)

    # Si MultiIndex -> on garde juste 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("Close", axis=1, level=0, drop_level=True)
    else:
        df = df["Close"]

    # On prend la dernière ligne
    last_row = df.dropna().iloc[-1]
    # On réordonne pour être sûr d'avoir les colonnes dans l'ordre de TICKERS
    prices = last_row.reindex(TICKERS)
    return prices  # Series index = tickers


def init_portfolio(prices: pd.Series):
    """
    Initialise le portefeuille :
    - capital initial
    - calcul du nombre de titres par actif pour avoir 20 % chacun
    """
    capital = INITIAL_CAPITAL
    shares = (capital * TARGET_WEIGHTS) / prices.values  # nb d'actions pour chaque ticker
    return capital, shares


def load_last_state():
    """
    Charge le dernier état du portefeuille si l'historique existe.
    Retourne (capital, shares_array, df_history) ou (None, None, None)
    """
    if not os.path.exists(HISTORY_FILE):
        return None, None, None

    hist = pd.read_csv(HISTORY_FILE)
    if hist.empty:
        return None, None, hist

    last = hist.iloc[-1]
    capital = float(last["capital"])
    shares = np.array([last[f"{t}_shares"] for t in TICKERS], dtype=float)

    return capital, shares, hist


def step_rebalance():
    """
    Une étape de rebalancement :
    - lit le dernier état (ou initialise)
    - met à jour la valeur du portefeuille avec les nouveaux prix
    - rebalance à 20 % / actif
    - sauvegarde une nouvelle ligne dans HISTORY_FILE
    - génère un rapport texte simple
    """
    prices = get_last_prices()

    # On essaie de charger l'état précédent
    capital_prev, shares_prev, hist = load_last_state()

    if hist is None:
        # pas d'historique → on va créer un DataFrame vide
        hist = pd.DataFrame()

    # 1) Si c'est la première fois → on initialise
    if capital_prev is None or shares_prev is None:
        capital = INITIAL_CAPITAL
        shares = (capital * TARGET_WEIGHTS) / prices.values
    else:
        # 2) Sinon : on valorise l'ancien portefeuille avec les nouveaux prix
        capital = float(np.sum(shares_prev * prices.values))
        # Rebalancement à 20 % chacun
        shares = (capital * TARGET_WEIGHTS) / prices.values

    # Construction de la nouvelle ligne
    now = datetime.datetime.now()
    row = {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "capital": capital,
    }
    for t, s in zip(TICKERS, shares):
        row[f"{t}_shares"] = s
        row[f"{t}_weight_target"] = 100.0 / len(TICKERS)  # 20 % chacun, fixe

    # Ajout à l'historique
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

    # Sauvegarde CSV
    hist.to_csv(HISTORY_FILE, index=False)

    # Génération d'un petit rapport texte (optionnel)
    report_txt = f"""
=========================================================
   RAPPORT DE GESTION - QUANT B (AUTOMATISÉ)
   Date : {now.strftime("%Y-%m-%d %H:%M:%S")}
=========================================================

1. PORTEFEUILLE ÉGAL-PONDÉRÉ (20 % chaque actif)
------------------------------------------------
Capital actuel : {capital:,.2f} $

Détail des positions (après rebalancement) :
"""
    for t, s, p in zip(TICKERS, shares, prices.values):
        value = s * p
        report_txt += f"  - {t:<8} : {s:.4f} titres  (~{value:,.2f} $)\n"

    report_txt += """
Rebalancement : 20 % par actif, exécuté automatiquement.

=========================================================
Statut : Succès
Serveur : AWS EC2 - Cron Job (toutes les 5 minutes)
=========================================================
"""

    # On écrase le rapport du jour (1 fichier par jour par ex.)
    filename = os.path.join(REPORT_DIR, f"report_{now.date()}.txt")
    with open(filename, "w") as f:
        f.write(report_txt)

    print(f"[OK] Rebalancement effectué - Capital: {capital:,.2f} $")
    print(f"[OK] Historique mis à jour : {HISTORY_FILE}")
    print(f"[OK] Rapport texte : {filename}")


if __name__ == "__main__":
    step_rebalance()
    # ================================
    # 🔎 HISTORIQUE RÉEL DU PORTEFEUILLE (CRON 5 min)
    # ================================
    import os

    HISTORY_FILE = "/home/ubuntu/Projet-Linux-Pierre-Antoine-et-Rhea/reports/portfolio_history.csv"

    st.markdown("---")
    st.subheader("📊 Évolution réelle du portefeuille (bot 20 % / actif)")

    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE, parse_dates=["datetime"])

        # Courbe du capital
        fig_hist = px.line(
            hist,
            x="datetime",
            y="capital",
            labels={"datetime": "Date / Heure", "capital": "Valeur du portefeuille ($)"},
            title="Capital du portefeuille au fil des rebalancements (cron 5 min)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Dernier état (pour voir clairement le rebalancement)
        st.subheader("🧾 Dernier rebalancement exécuté")
        last_row = hist.tail(1).T
        st.dataframe(last_row, use_container_width=True)

    else:
        st.info(
            "Aucun historique trouvé pour le portefeuille réel. "
            "Lance d'abord le script `report.py` (manuellement ou via cron) pour commencer l'enregistrement."
        )
