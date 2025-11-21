
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import itertools


# --- 1. CLASSE D'ANALYSE (Backend Logic) ---
class SingleAssetAnalyzer:
    def __init__(self, ticker, start_date, end_date, initial_investment=1000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.data = pd.DataFrame()
        self.daily_returns = pd.Series(dtype=float)
        self.best_params = {} # Pour stocker les recommandations

    def load_data(self):
        """Télécharge les données."""
        try:
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('Close', axis=1, level=0, drop_level=False)
                df.columns = ['Close']
            else:
                df = df[['Close']]

            if df.empty: return False

            self.data = df
            self.daily_returns = self.data['Close'].pct_change().fillna(0)
            return True
        except Exception as e:
            st.error(f"Erreur chargement : {e}")
            return False

    def compute_metrics(self, strategy_returns):
        """Calcule Sharpe, Max Drawdown et Performance Totale."""
        strategy_returns = strategy_returns.dropna()
        if strategy_returns.empty:
            return {'Sharpe': 0.0, 'Max Drawdown': 0.0, 'Total Perf': 0.0}

        rf = 0.03
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        
        if std_ret == 0: sharpe = 0
        else: sharpe = (mean_ret - (rf/252)) / std_ret * np.sqrt(252)

        cum_ret = (1 + strategy_returns).cumprod()
        peak = cum_ret.expanding(min_periods=1).max()
        dd = (cum_ret - peak) / peak
        max_dd = abs(dd.min())
        total_perf = cum_ret.iloc[-1] - 1

        return {
            'Sharpe': round(sharpe, 2),
            'Max Drawdown': f"{max_dd:.2%}",
            'Total Perf': f"{total_perf:.2%}",
            'Raw_Sharpe': sharpe # Pour le tri interne
        }

    def run_strategy(self, strat_name, **params):
        """Exécute une stratégie spécifique avec des paramètres donnés."""
        bh_curve = (1 + self.daily_returns).cumprod() * self.initial_investment
        signals = pd.Series(0, index=self.data.index)

        # --- LOGIQUE DES STRATÉGIES ---
        if strat_name == "Momentum":
            window = int(params.get('window', 50))
            mms = self.data['Close'].rolling(window=window).mean()
            signals = np.where(self.data['Close'] > mms, 1.0, 0.0)
            
        elif strat_name == "Cross MMS":
            short_w = int(params.get('short_w', 20))
            long_w = int(params.get('long_w', 50))
            mms_short = self.data['Close'].rolling(window=short_w).mean()
            mms_long = self.data['Close'].rolling(window=long_w).mean()
            signals = np.where(mms_short > mms_long, 1.0, 0.0)

        elif strat_name == "Mean Reversion (BB)":
            window = int(params.get('window', 20))
            std_dev = float(params.get('std_dev', 2.0))
            sma = self.data['Close'].rolling(window=window).mean()
            std = self.data['Close'].rolling(window=window).std()
            lower_band = sma - (std * std_dev)
            # Achat si < Lower Band, Vente si > SMA (simplifié)
            signals = np.where(self.data['Close'] < lower_band, 1.0, 0.0)

        # Backtest
        signals = pd.Series(signals, index=self.data.index)
        strat_returns = self.daily_returns.shift(-1) * signals.shift(1).fillna(0)
        strat_curve = (1 + strat_returns).cumprod() * self.initial_investment
        strat_curve = strat_curve.ffill() # Correction du bug NaN à la fin

        return strat_curve, strat_returns

    # --- PARTIE OPTIMISATION (LE CERVEAU) ---
    def find_best_params(self):
        """Teste plein de combinaisons et stocke les gagnantes."""
        
        # 1. Optimisation Momentum
        best_sharpe = -999
        best_p = {'window': 50}
        for w in range(10, 100, 10):
            _, rets = self.run_strategy("Momentum", window=w)
            m = self.compute_metrics(rets)
            if m['Raw_Sharpe'] > best_sharpe:
                best_sharpe = m['Raw_Sharpe']
                best_p = {'window': w}
        self.best_params['Momentum'] = best_p

        # 2. Optimisation Cross MMS
        best_sharpe = -999
        best_p = {'short_w': 20, 'long_w': 50}
        for s, l in itertools.product(range(10, 50, 10), range(50, 150, 20)):
            if s >= l: continue
            _, rets = self.run_strategy("Cross MMS", short_w=s, long_w=l)
            m = self.compute_metrics(rets)
            if m['Raw_Sharpe'] > best_sharpe:
                best_sharpe = m['Raw_Sharpe']
                best_p = {'short_w': s, 'long_w': l}
        self.best_params['Cross MMS'] = best_p

        # 3. Optimisation BB
        best_sharpe = -999
        best_p = {'window': 20, 'std_dev': 2.0}
        for w, std in itertools.product(range(10, 50, 10), [1.5, 2.0, 2.5]):
            _, rets = self.run_strategy("Mean Reversion (BB)", window=w, std_dev=std)
            m = self.compute_metrics(rets)
            if m['Raw_Sharpe'] > best_sharpe:
                best_sharpe = m['Raw_Sharpe']
                best_p = {'window': w, 'std_dev': std}
        self.best_params['Mean Reversion (BB)'] = best_p

#new fontion pour les différents modèles    
    def predict_future(self, days_ahead=30, model_type="Linear Regression"):
        """
        Génère des prédictions selon le modèle choisi :
        1. Linear Regression (Tendance simple)
        2. ARIMA (Séries temporelles, capture les cycles)
        3. Random Forest (Machine Learning, capture les motifs complexes)
        """
        df = self.data.copy()
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
        
        # --- MODÈLE 1 : RÉGRESSION LINÉAIRE (CORRIGÉ AVEC ANCRAGE) ---
        if model_type == "Linear Regression":
            df = df.reset_index()
            df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
            X = df[['Date_Ordinal']].values
            y = df['Close'].values
            
            # Entrainement sur tout l'historique pour avoir la PENTE (la direction)
            model = LinearRegression().fit(X, y)
            
            future_ordinals = [[d.toordinal()] for d in future_dates]
            preds = model.predict(future_ordinals)
            
            # --- LE FIX MAGIQUE EST ICI ---
            # 1. On demande au modèle : "Selon toi, à combien on devrait être aujourd'hui ?"
            last_day_ordinal = [[X[-1][0]]]
            theoretical_price_today = model.predict(last_day_ordinal)[0]
            
            # 2. On regarde le vrai prix : "En réalité, on est à combien ?"
            actual_price_today = y[-1]
            
            # 3. On calcule l'écart (le gap)
            offset = actual_price_today - theoretical_price_today
            
            # 4. On décale toute la prédiction future pour recoller les morceaux
            preds = preds + offset
            # -----------------------------
            # on va faire des volatilité plus restreinte, en ne prenant que 90 jours pour faire les calculs
            recent_returns = df['Close'].pct_change().tail(90)
            
            # 2. On calcule l'écart-type de ces variations
            sigma_pct = recent_returns.std()
            
            # 3. On convertit ça en dollars par rapport au dernier prix
            # (Ex: si le BTC est à 90k et la vol à 2%, l'écart-type est 1800$)
            std_dev = sigma_pct * df['Close'].iloc[-1]
            
            return future_dates, preds, std_dev
        # --- MODÈLE 2 : ARIMA (AutoRegressive Integrated Moving Average) ---
        elif model_type == "ARIMA":
            # On utilise (5,1,0) : regarde les 5 derniers jours, différencie 1 fois
            history = df['Close'].values
            # Suppress warnings si nécessaire dans Kaggle
            model = ARIMA(history, order=(5,1,0)) 
            model_fit = model.fit()
            
            preds = model_fit.forecast(steps=days_ahead)
            
            # Erreur estimée via les résidus du modèle
            residuals = model_fit.resid
            # On exclut le tout premier résidu souvent aberrant (NaN ou 0)
            std_dev = np.std(residuals[1:]) 
            return future_dates, preds, std_dev

        # --- MODÈLE 3 : RANDOM FOREST (Machine Learning) ---
        elif model_type == "Machine Learning (RF)":
            # Création des "Features" (ce que le modèle regarde pour apprendre)
            # Lag1 = prix d'hier, Lag2 = avant-hier, MA5 = moyenne 5 jours
            df['Lag1'] = df['Close'].shift(1)
            df['Lag2'] = df['Close'].shift(2)
            df['MA5'] = df['Close'].rolling(5).mean()
            df = df.dropna() # On enlève les lignes vides créées par le lag

            X = df[['Lag1', 'Lag2', 'MA5']].values
            y = df['Close'].values
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Prédiction Récursive (On prédit jour après jour)
            preds = []
            current_lag1 = df['Close'].iloc[-1]
            current_lag2 = df['Close'].iloc[-2]
            current_ma = df['MA5'].iloc[-1] # Simplification pour la démo

            for _ in range(days_ahead):
                # Le modèle prédit demain
                pred = model.predict([[current_lag1, current_lag2, current_ma]])[0]
                preds.append(pred)
                
                # On met à jour les variables pour le jour d'après
                current_lag2 = current_lag1
                current_lag1 = pred
                # Note: Idéalement on recalculerait la MA ici, on garde fixe pour simplifier
            
            # Calcul de l'erreur moyenne sur l'entrainement
            train_preds = model.predict(X)
            std_dev = np.std(y - train_preds)
            
            return future_dates, np.array(preds), std_dev
        
        return [], [], 0 # Fallback si erreur
# --- 2. INTERFACE STREAMLIT ---

st.set_page_config(layout="wide", page_title="Smart Quant Lab")
st.title("🧠 Smart Quant Lab: Analyse & Optimisation")

# Initialisation session state pour ne pas perdre les calculs
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

with st.sidebar:
    st.header("1. Paramètres Généraux")
    ticker = st.text_input("Ticker", "BTC-USD")
    s_date = st.date_input("Début", date(2020, 1, 1))
    e_date = st.date_input("Fin", date.today())
    
    if st.button("📥 Charger Données & Scanner"):
        an = SingleAssetAnalyzer(ticker, s_date, e_date)
        if an.load_data():
            with st.spinner("Le robot cherche les meilleurs paramètres..."):
                an.find_best_params() # On lance l'optimisation ici
            st.session_state.analyzer = an
            st.success("Scan terminé !")
    
    st.markdown("---")
    
    # On affiche les contrôles seulement si l'analyseur est chargé
    #ici, on fait que le contrôle manuel
    if st.session_state.analyzer:
        an = st.session_state.analyzer
        
        st.header("2. Contrôle Manuel")
        # 1. On ajoute l'option "TOUT COMPARER" dans la liste
        strat_choice = st.selectbox("Stratégie Active", ["Momentum", "Cross MMS", "Mean Reversion (BB)", "TOUT COMPARER"])
        
        # On va stocker tous les paramètres ici
        manual_params = {}
        
        # BLOC MOMENTUM (S'affiche si Momentum OU Tout Comparer est choisi)
        if strat_choice == "Momentum" or strat_choice == "TOUT COMPARER":
            st.markdown("### Paramètres Momentum")
            rec = an.best_params['Momentum']['window']
            st.caption(f"💡 Suggestion IA : {rec}")
            # On stocke dans manual_params avec des clés précises
            manual_params['mom_window'] = st.slider("Fenêtre Momentum", 10, 200, 50, key="mom_slider")
            
        # BLOC CROSS MMS
        if strat_choice == "Cross MMS" or strat_choice == "TOUT COMPARER":
            st.markdown("### Paramètres Cross MMS")
            rec_s = an.best_params['Cross MMS']['short_w']
            rec_l = an.best_params['Cross MMS']['long_w']
            st.caption(f"💡 Suggestion IA : Court={rec_s}, Long={rec_l}")
            manual_params['cross_short'] = st.slider("MMS Court", 5, 50, 20, key="cross_s_slider")
            manual_params['cross_long'] = st.slider("MMS Long", 50, 200, 100, key="cross_l_slider")

        # BLOC BOLLINGER
        if strat_choice == "Mean Reversion (BB)" or strat_choice == "TOUT COMPARER":
            st.markdown("### Paramètres Bollinger")
            rec_w = an.best_params['Mean Reversion (BB)']['window']
            rec_std = an.best_params['Mean Reversion (BB)']['std_dev']
            st.caption(f"💡 Suggestion IA : Fenêtre={rec_w}, Std={rec_std}")
            manual_params['bb_window'] = st.slider("Fenêtre BB", 10, 100, 20, key="bb_w_slider")
            manual_params['bb_std'] = st.slider("Écart-Type", 1.0, 3.0, 2.0, key="bb_std_slider")
        st.markdown("---")


        st.markdown("---")
        st.header("3. Prédiction (Bonus)")
        
        # Plus de checkbox, on affiche directement les contrôles
        model_choice = st.selectbox(
            "Choisir le Modèle", 
            ["Linear Regression", "ARIMA", "Machine Learning (RF)"]
        )
        forecast_days = st.slider("Jours à prédire", 7, 90, 30)

# --- AFFICHAGE PRINCIPAL ---

if st.session_state.analyzer:
    an = st.session_state.analyzer
    bh_curve = (1 + an.daily_returns).cumprod() * an.initial_investment

    # --- CAS 1 : COMPARAISON GLOBALE ---
    if strat_choice == "TOUT COMPARER":
        st.subheader("⚡ Comparaison Multi-Stratégies (Paramètres Manuels)")
        
        # On lance les 3 stratégies avec les paramètres récupérés des sliders
        c_mom, _ = an.run_strategy("Momentum", window=manual_params['mom_window'])
        c_cross, _ = an.run_strategy("Cross MMS", short_w=manual_params['cross_short'], long_w=manual_params['cross_long'])
        c_bb, _ = an.run_strategy("Mean Reversion (BB)", window=manual_params['bb_window'], std_dev=manual_params['bb_std'])
        
        # On crée un gros DataFrame avec tout
        df_all = pd.DataFrame({
            "Buy & Hold": bh_curve,
            "Momentum": c_mom,
            "Cross MMS": c_cross,
            "Bollinger": c_bb
        })
        
        st.line_chart(df_all)
        
        # Petit tableau récapitulatif des gains finaux
        st.write("### Valeurs Finales du Portefeuille")
        res_finaux = df_all.iloc[-1].sort_values(ascending=False)
        st.dataframe(res_finaux.map('{:.2f} $'.format))

    # --- CAS 2 : MODE SOLO (Comme avant) ---
    else:
        # On prépare les arguments selon la stratégie choisie
        args = {}
        if strat_choice == "Momentum": args = {'window': manual_params['mom_window']}
        elif strat_choice == "Cross MMS": args = {'short_w': manual_params['cross_short'], 'long_w': manual_params['cross_long']}
        elif strat_choice == "Mean Reversion (BB)": args = {'window': manual_params['bb_window'], 'std_dev': manual_params['bb_std']}
        
        # Calcul
        strat_curve, strat_rets = an.run_strategy(strat_choice, **args)
        met_strat = an.compute_metrics(strat_rets)
        met_bh = an.compute_metrics(an.daily_returns)

        # KPI (Haut de page)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stratégie", strat_choice)
        c2.metric("Sharpe Ratio", met_strat['Sharpe'], delta=f"{met_strat['Sharpe'] - met_bh['Sharpe']:.2f} vs B&H")
        c3.metric("Max Drawdown", met_strat['Max Drawdown'])
        c4.metric("Gain Total", met_strat['Total Perf'])

        # GRAPHIQUE SOLO
        st.subheader(f"📈 Analyse : {strat_choice} vs Marché")
        df_chart = pd.DataFrame({
            "Buy & Hold (Marché)": bh_curve,
            f"Ma Stratégie ({strat_choice})": strat_curve
        })
        st.line_chart(df_chart, color=["#FF4B4B", "#0068C9"])

# 4. SECTION COMPARATIVE (Le "Battle" des stratégies optimisées)
    st.markdown("---")
    st.subheader("⚔️ Battle Royale : Comparaison des Modèles Optimisés")
    st.caption("Voici ce que ça donnerait si on prenait les MEILLEURS paramètres pour chaque stratégie sur cette période.")

    # 1. On relance les calculs (cette fois on récupère 'ret' pour les métriques)
    curve_mom, ret_mom = an.run_strategy("Momentum", **an.best_params['Momentum'])
    curve_cross, ret_cross = an.run_strategy("Cross MMS", **an.best_params['Cross MMS'])
    curve_bb, ret_bb = an.run_strategy("Mean Reversion (BB)", **an.best_params['Mean Reversion (BB)'])

    # 2. Affichage du Graphique
    df_battle = pd.DataFrame({
        "Buy & Hold": bh_curve,
        f"Momentum (Opti)": curve_mom,
        f"Cross MMS (Opti)": curve_cross,
        f"Bollinger (Opti)": curve_bb
    })
    st.line_chart(df_battle)

    # 3. TABLEAU DES RÉSULTATS (Le Podium)
    st.subheader("🏆 Le Bulletin de Notes")

    # On calcule les métriques pour tout le monde
    met_bh = an.compute_metrics(an.daily_returns)
    met_mom = an.compute_metrics(ret_mom)
    met_cross = an.compute_metrics(ret_cross)
    met_bb = an.compute_metrics(ret_bb)

    # On construit un tableau propre
    leaderboard_data = [
        {
            "Stratégie": "Buy & Hold (Marché)",
            "Sharpe Ratio": met_bh['Sharpe'],
            "Max Drawdown": met_bh['Max Drawdown'],
            "Perf Totale": met_bh['Total Perf'],
            "Capital Final ($)": f"{bh_curve.iloc[-1]:.2f} $"
        },
        {
            "Stratégie": f"Momentum (Win: {an.best_params['Momentum']['window']})",
            "Sharpe Ratio": met_mom['Sharpe'],
            "Max Drawdown": met_mom['Max Drawdown'],
            "Perf Totale": met_mom['Total Perf'],
            "Capital Final ($)": f"{curve_mom.iloc[-1]:.2f} $"
        },
        {
            "Stratégie": f"Cross MMS (S:{an.best_params['Cross MMS']['short_w']} L:{an.best_params['Cross MMS']['long_w']})",
            "Sharpe Ratio": met_cross['Sharpe'],
            "Max Drawdown": met_cross['Max Drawdown'],
            "Perf Totale": met_cross['Total Perf'],
            "Capital Final ($)": f"{curve_cross.iloc[-1]:.2f} $"
        },
        {
            "Stratégie": f"Bollinger (W:{an.best_params['Mean Reversion (BB)']['window']} Std:{an.best_params['Mean Reversion (BB)']['std_dev']})",
            "Sharpe Ratio": met_bb['Sharpe'],
            "Max Drawdown": met_bb['Max Drawdown'],
            "Perf Totale": met_bb['Total Perf'],
            "Capital Final ($)": f"{curve_bb.iloc[-1]:.2f} $"
        }
    ]

    # Création du DataFrame pour l'affichage
    df_leaderboard = pd.DataFrame(leaderboard_data)
    
    # On met la stratégie en index pour que ce soit plus joli
    df_leaderboard.set_index("Stratégie", inplace=True)

    # On trie par Sharpe Ratio décroissant (le meilleur en haut)
    # Note : Sharpe est un float, les autres sont des strings formatés (%), donc on trie sur Sharpe
    df_leaderboard.sort_values(by="Sharpe Ratio", ascending=False, inplace=True)

    # Affichage du tableau
    st.dataframe(df_leaderboard, use_container_width=True)
    # B. SECTION PRÉDICTION
    
# B. SECTION PRÉDICTION (Plus de "if show_pred")
    st.markdown("---")
    st.subheader(f"🔮 Prédiction Future : {model_choice}")
    
    # Appel de la fonction
    with st.spinner(f"Calcul du modèle {model_choice} en cours..."):
        fut_d, fut_p, std = an.predict_future(forecast_days, model_type=model_choice)
    
    # Préparation des données
    recent = an.data['Close'].tail(180)
    df_fut = pd.DataFrame({"Pred": fut_p}, index=fut_d)
    
    # Cône d'incertitude
    import numpy as np
    time_scaling = np.sqrt(np.arange(1, len(df_fut) + 1))
    df_fut["High"] = df_fut["Pred"] + (1.96 * std * time_scaling)
    df_fut["Low"] = df_fut["Pred"] - (1.96 * std * time_scaling)
        
        # -------------------------------------

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
        
    ax.plot(recent.index, recent.values, label="Historique Récent", color="black", alpha=0.6)
    ax.plot(df_fut.index, df_fut["Pred"], label=f"Prédiction ({model_choice})", color="#0068C9", linestyle="--", linewidth=2)
    ax.fill_between(df_fut.index, df_fut["Low"], df_fut["High"], color="#0068C9", alpha=0.15, label="Zone de Confiance 95%")
        
    ax.set_title(f"Projection {ticker} sur {forecast_days} jours")
    ax.legend()
    ax.grid(True, alpha=0.2)
        
    st.pyplot(fig)
        
        # Petit texte explicatif selon le modèle choisi
    if model_choice == "ARIMA":
        st.info("ℹ️ **ARIMA** analyse les cycles passés. Idéal pour les marchés volatils à court terme, essayez de l'appliquer au bitcoin par exemple.")
    elif model_choice == "Machine Learning (RF)":
        st.info("ℹ️ **Random Forest** utilise l'IA pour repérer des motifs complexes (prix d'hier, avant-hier, moyennes).")
    else:
        st.warning("⚠️ **Régression Linéaire** : Trace juste une tendance droite. Attention, ne prédit pas les chutes ! Ce modèle est plus adapté pour les  cours stables, essayez plutôt une action de père de famille, comme air liquide ;)")

    ticker_clean = ticker.upper()
        #ajout du retour sur experience
        # CAS 1 : BITCOIN + ARIMA
    if "BTC" in ticker_clean and model_choice == "ARIMA":
        st.success("✅ Excellent choix ! Le Bitcoin est très volatil et cyclique, ARIMA est théoriquement le meilleur modèle pour capturer ces mouvements.")

        # CAS 2 : AIR LIQUIDE + REGRESSION LINEAIRE
        # (Le ticker Air Liquide sur Yahoo est souvent AI.PA)
    elif ("AI.PA" in ticker_clean or "AIR LIQUIDE" in ticker_clean) and model_choice == "Linear Regression":
        st.success("✅ Bien vu ! Air Liquide est une action très stable avec une tendance long terme claire. La Régression Linéaire suffit largement et sera très propre.")

else:
    st.info("👈 Veuillez cliquer sur 'Charger Données & Scanner' dans la barre latérale pour commencer.")
    # ============================================================
# =====================   QUANT B   ==========================
# ======== MULTI-ASSET PORTFOLIO — MARKOWITZ / MONTE-CARLO ===
# ============================================================

st.markdown("---")
st.header("📊 QuantB — Portfolio Multi-Assets (Markowitz & Monte-Carlo)")

with st.sidebar:
    st.subheader("⚙ Paramètres du portefeuille – QuantB")

    # Choix des actifs
    tickers = st.multiselect(
        "Sélectionne plusieurs actifs :",
        ["AAPL", "MSFT", "BTC-USD", "GOOG", "AMZN", "TSLA", "META"],
        default=["AAPL", "MSFT", "BTC-USD"]
    )

    # Période
    start_b = st.date_input("Date de début", date(2020,1,1))
    end_b = st.date_input("Date de fin", date.today())

    # Sharpe
    rf_qb = st.number_input("Taux sans risque annuel", value=0.02, step=0.005)

    # Nombre de portefeuilles simulés
    N_sim = st.slider("Nombre de portefeuilles simulés (Monte Carlo)", 100, 10000, 3000)

def load_multi_assets(tickers, start, end):
    """Télécharge les prix des actifs actifs."""
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data.dropna()

def compute_portfolio_simulation(returns, N=3000, rf=0.02):
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252

    results = []
    weights_list = []

    for _ in range(N):
        w = np.random.random(len(mean_ret))
        w = w / w.sum()

        port_ret = np.dot(w, mean_ret)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (port_ret - rf) / port_vol

        results.append([port_ret, port_vol, sharpe])
        weights_list.append(w)

    df_results = pd.DataFrame(results, columns=["Return", "Vol", "Sharpe"])
    return df_results, weights_list, mean_ret, cov

if len(tickers) >= 2:
    st.subheader("📥 Téléchargement des données (QuantB)")
    df_prices = load_multi_assets(tickers, start_b, end_b)
    st.success("Données téléchargées !")

    returns = df_prices.pct_change().dropna()

    st.subheader("📈 Valeur cumulée du portefeuille (base 100)")
    norm = df_prices / df_prices.iloc[0] * 100
    st.line_chart(norm)

    st.subheader("🧮 Matrice de Corrélation")
    st.dataframe(returns.corr())

    st.subheader("🎯 Simulation Markowitz — Monte Carlo")
    df_sim, w_list, mean_ret, cov = compute_portfolio_simulation(returns, N_sim, rf_qb)

    # Meilleur Sharpe
    idx_best = df_sim["Sharpe"].idxmax()
    best_weights = w_list[idx_best]

    st.success("Portefeuille à Sharpe Maximum trouvé !")

    df_w = pd.DataFrame({
        "Actif": tickers,
        "Poids (%)": [round(w*100,4) for w in best_weights]
    })
    st.dataframe(df_w)

    st.subheader("📊 Frontière de Markowitz (Monte Carlo)")
    st.scatter_chart(df_sim, x="Vol", y="Return", color="Sharpe")

