import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import itertools

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="Smart Quant Lab")

# --- 2. CLASSE D'ANALYSE (LE CERVEAU) ---
class SingleAssetAnalyzer:
    def __init__(self, ticker, start_date, end_date, initial_investment=1000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.data = pd.DataFrame()
        self.daily_returns = pd.Series(dtype=float)
        self.best_params = {} # Pour stocker les recommandations de l'IA

    def load_data(self):
        """Télécharge les données depuis Yahoo Finance."""
        try:
            # Téléchargement
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
            
            # Gestion du MultiIndex (problème fréquent avec yfinance récent)
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

        rf = 0.03 # Taux sans risque (3%)
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        
        # Sharpe Ratio
        if std_ret == 0: sharpe = 0
        else: sharpe = (mean_ret - (rf/252)) / std_ret * np.sqrt(252)

        # Max Drawdown
        cum_ret = (1 + strategy_returns).cumprod()
        peak = cum_ret.expanding(min_periods=1).max()
        dd = (cum_ret - peak) / peak
        max_dd = abs(dd.min())
        
        # Perf Totale
        total_perf = cum_ret.iloc[-1] - 1

        return {
            'Sharpe': round(sharpe, 2),
            'Max Drawdown': f"{max_dd:.2%}",
            'Total Perf': f"{total_perf:.2%}",
            'Raw_Sharpe': sharpe # Utilisé pour le tri
        }

    def run_strategy(self, strat_name, **params):
        """Exécute une stratégie spécifique et renvoie la courbe + les retours."""
        signals = pd.Series(0, index=self.data.index)

        # --- LOGIQUE DES STRATÉGIES ---
        if strat_name == "Momentum":
            window = int(params.get('window', 50))
            mms = self.data['Close'].rolling(window=window).mean()
            # Achat si Prix > Moyenne Mobile
            signals = np.where(self.data['Close'] > mms, 1.0, 0.0)
            
        elif strat_name == "Cross MMS":
            short_w = int(params.get('short_w', 20))
            long_w = int(params.get('long_w', 50))
            mms_short = self.data['Close'].rolling(window=short_w).mean()
            mms_long = self.data['Close'].rolling(window=long_w).mean()
            # Achat si Moyenne Courte > Moyenne Longue
            signals = np.where(mms_short > mms_long, 1.0, 0.0)

        elif strat_name == "Mean Reversion (BB)":
            window = int(params.get('window', 20))
            std_dev = float(params.get('std_dev', 2.0))
            sma = self.data['Close'].rolling(window=window).mean()
            std = self.data['Close'].rolling(window=window).std()
            lower_band = sma - (std * std_dev)
            # Achat si Prix < Bande Inférieure (Rebond espéré)
            signals = np.where(self.data['Close'] < lower_band, 1.0, 0.0)

        # Calcul des retours (Lag de 1 jour pour éviter le biais du futur)
        signals = pd.Series(signals, index=self.data.index)
        strat_returns = self.daily_returns.shift(-1) * signals.shift(1).fillna(0)
        
        # Courbe cumulée (Base 1000$)
        strat_curve = (1 + strat_returns).cumprod() * self.initial_investment
        strat_curve = strat_curve.ffill() # Bouche les trous à la fin

        return strat_curve, strat_returns

    def find_best_params(self):
        """Optimisation IA : Teste des centaines de combinaisons."""
        
        # 1. Optimisation Momentum
        best_sharpe = -999
        best_p = {'window': 50}
        for w in range(10, 150, 10):
            _, rets = self.run_strategy("Momentum", window=w)
            m = self.compute_metrics(rets)
            if m['Raw_Sharpe'] > best_sharpe:
                best_sharpe = m['Raw_Sharpe']
                best_p = {'window': w}
        self.best_params['Momentum'] = best_p

        # 2. Optimisation Cross MMS
        best_sharpe = -999
        best_p = {'short_w': 20, 'long_w': 50}
        for s, l in itertools.product(range(10, 60, 10), range(50, 200, 20)):
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
        for w, std in itertools.product(range(10, 60, 10), [1.5, 2.0, 2.5]):
            _, rets = self.run_strategy("Mean Reversion (BB)", window=w, std_dev=std)
            m = self.compute_metrics(rets)
            if m['Raw_Sharpe'] > best_sharpe:
                best_sharpe = m['Raw_Sharpe']
                best_p = {'window': w, 'std_dev': std}
        self.best_params['Mean Reversion (BB)'] = best_p

    def predict_future(self, days_ahead=30, model_type="Linear Regression"):
        """Génère des prédictions (Linear, ARIMA, RF)."""
        df = self.data.copy()
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
        
        # --- MODÈLE 1 : RÉGRESSION LINÉAIRE (AVEC CORRECTIFS) ---
        if model_type == "Linear Regression":
            df = df.reset_index()
            df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
            X = df[['Date_Ordinal']].values
            y = df['Close'].values
            
            # Entrainement global pour la tendance
            model = LinearRegression().fit(X, y)
            future_ordinals = [[d.toordinal()] for d in future_dates]
            preds = model.predict(future_ordinals)
            
            # FIX 1 : ANCRAGE (Recoller au prix réel d'aujourd'hui)
            last_day_ordinal = [[X[-1][0]]]
            theoretical_price = model.predict(last_day_ordinal)[0]
            actual_price = y[-1]
            offset = actual_price - theoretical_price
            preds = preds + offset
            
            # FIX 2 : VOLATILITÉ LOCALE (Calcul du risque sur les 90 derniers jours seulement)
            recent_returns = df['Close'].pct_change().tail(90)
            sigma_pct = recent_returns.std()
            std_dev = sigma_pct * df['Close'].iloc[-1]
            
            return future_dates, preds, std_dev

        # --- MODÈLE 2 : ARIMA ---
        elif model_type == "ARIMA":
            history = df['Close'].values
            # Modèle (5,1,0) standard
            model = ARIMA(history, order=(5,1,0)) 
            model_fit = model.fit()
            preds = model_fit.forecast(steps=days_ahead)
            
            residuals = model_fit.resid
            std_dev = np.std(residuals[1:]) 
            return future_dates, preds, std_dev

        # --- MODÈLE 3 : RANDOM FOREST ---
        elif model_type == "Machine Learning (RF)":
            # Feature Engineering
            df['Lag1'] = df['Close'].shift(1)
            df['Lag2'] = df['Close'].shift(2)
            df['MA5'] = df['Close'].rolling(5).mean()
            df = df.dropna()

            X = df[['Lag1', 'Lag2', 'MA5']].values
            y = df['Close'].values
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Prédiction récursive
            preds = []
            current_lag1 = df['Close'].iloc[-1]
            current_lag2 = df['Close'].iloc[-2]
            current_ma = df['MA5'].iloc[-1] 

            for _ in range(days_ahead):
                pred = model.predict([[current_lag1, current_lag2, current_ma]])[0]
                preds.append(pred)
                current_lag2 = current_lag1
                current_lag1 = pred
            
            train_preds = model.predict(X)
            std_dev = np.std(y - train_preds)
            
            return future_dates, np.array(preds), std_dev
        
        return [], [], 0

# --- 3. INTERFACE STREAMLIT (FRONTEND) ---

st.title("🧠 Smart Quant Lab: Analyse & Optimisation")

# Initialisation du cache de session
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("1. Paramètres Généraux")
    ticker = st.text_input("Ticker", "BTC-USD")
    s_date = st.date_input("Début", date(2020, 1, 1))
    e_date = st.date_input("Fin", date.today())
    
    if st.button("📥 Charger Données & Scanner"):
        an = SingleAssetAnalyzer(ticker, s_date, e_date)
        if an.load_data():
            with st.spinner("Le robot cherche les meilleurs paramètres..."):
                an.find_best_params()
            st.session_state.analyzer = an
            st.success("Scan terminé !")
    
    st.markdown("---")
    
    # Contrôles dynamiques (visibles seulement si données chargées)
    if st.session_state.analyzer:
        an = st.session_state.analyzer
        
        st.header("2. Contrôle Manuel")
        strat_choice = st.selectbox("Stratégie Active", ["Momentum", "Cross MMS", "Mean Reversion (BB)", "TOUT COMPARER"])
        
        manual_params = {}
        
        # Affichage conditionnel des sliders + Conseils IA
        if strat_choice == "Momentum" or strat_choice == "TOUT COMPARER":
            st.markdown("### 🚀 Momentum")
            rec = an.best_params['Momentum']['window']
            st.caption(f"💡 Suggestion IA : {rec} jours")
            manual_params['mom_window'] = st.slider("Fenêtre", 10, 200, 50, key="mom_s")
            
        if strat_choice == "Cross MMS" or strat_choice == "TOUT COMPARER":
            st.markdown("### ❌ Cross MMS")
            rec_s = an.best_params['Cross MMS']['short_w']
            rec_l = an.best_params['Cross MMS']['long_w']
            st.caption(f"💡 Suggestion IA : {rec_s} / {rec_l}")
            manual_params['cross_short'] = st.slider("Court", 5, 50, 20, key="crs_s")
            manual_params['cross_long'] = st.slider("Long", 50, 200, 100, key="crs_l")

        if strat_choice == "Mean Reversion (BB)" or strat_choice == "TOUT COMPARER":
            st.markdown("### 📉 Bollinger")
            rec_w = an.best_params['Mean Reversion (BB)']['window']
            rec_std = an.best_params['Mean Reversion (BB)']['std_dev']
            st.caption(f"💡 Suggestion IA : {rec_w}j / {rec_std} std")
            manual_params['bb_window'] = st.slider("Fenêtre BB", 10, 100, 20, key="bb_w")
            manual_params['bb_std'] = st.slider("Écart-Type", 1.0, 3.0, 2.0, key="bb_std")
        
        st.markdown("---")
        st.header("3. Prédiction (Bonus)")
        model_choice = st.selectbox("Choisir le Modèle", ["Linear Regression", "ARIMA", "Machine Learning (RF)"])
        forecast_days = st.slider("Jours à prédire", 7, 90, 30)

# --- CORPS PRINCIPAL ---

if st.session_state.analyzer:
    an = st.session_state.analyzer
    bh_curve = (1 + an.daily_returns).cumprod() * an.initial_investment

    # --- SECTION A : ANALYSE DES STRATÉGIES ---
    
    # CAS 1 : MODE COMPARAISON (Tout en même temps)
    if strat_choice == "TOUT COMPARER":
        st.subheader("⚡ Comparaison Multi-Stratégies (Paramètres Manuels)")
        
        c_mom, _ = an.run_strategy("Momentum", window=manual_params['mom_window'])
        c_cross, _ = an.run_strategy("Cross MMS", short_w=manual_params['cross_short'], long_w=manual_params['cross_long'])
        c_bb, _ = an.run_strategy("Mean Reversion (BB)", window=manual_params['bb_window'], std_dev=manual_params['bb_std'])
        
        df_all = pd.DataFrame({
            "Buy & Hold": bh_curve,
            "Momentum": c_mom,
            "Cross MMS": c_cross,
            "Bollinger": c_bb
        })
        st.line_chart(df_all)
        
        st.write("### 💰 Valeurs Finales")
        res_finaux = df_all.iloc[-1].sort_values(ascending=False)
        st.dataframe(res_finaux.map('{:.2f} $'.format))

    # CAS 2 : MODE SOLO AVEC SIGNAUX D'ACHAT/VENTE
    else:
        # Préparation des paramètres
        args = {}
        if strat_choice == "Momentum": args = {'window': manual_params['mom_window']}
        elif strat_choice == "Cross MMS": args = {'short_w': manual_params['cross_short'], 'long_w': manual_params['cross_long']}
        elif strat_choice == "Mean Reversion (BB)": args = {'window': manual_params['bb_window'], 'std_dev': manual_params['bb_std']}
        
        # Exécution
        strat_curve, strat_rets = an.run_strategy(strat_choice, **args)
        met_strat = an.compute_metrics(strat_rets)
        met_bh = an.compute_metrics(an.daily_returns)

        # Affichage des KPI
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stratégie", strat_choice)
        c2.metric("Sharpe Ratio", met_strat['Sharpe'], delta=f"{met_strat['Sharpe'] - met_bh['Sharpe']:.2f} vs B&H")
        c3.metric("Max Drawdown", met_strat['Max Drawdown'])
        c4.metric("Gain Total", met_strat['Total Perf'])

        # Graphique Avancé (Matplotlib avec Triangles)
        st.subheader(f"📈 Analyse Détaillée : {strat_choice}")
        
        # Recalcul local des signaux pour le placement des marqueurs
        signals = pd.Series(0, index=an.data.index)
        indicator = None
        label_ind = ""
        
        if strat_choice == "Momentum":
            win = manual_params['mom_window']
            mms = an.data['Close'].rolling(win).mean()
            signals = np.where(an.data['Close'] > mms, 1.0, 0.0)
            indicator = mms
            label_ind = f"Moyenne Mobile ({win}j)"
            
        elif strat_choice == "Cross MMS":
            s, l = manual_params['cross_short'], manual_params['cross_long']
            short = an.data['Close'].rolling(s).mean()
            long = an.data['Close'].rolling(l).mean()
            signals = np.where(short > long, 1.0, 0.0)
            indicator = long
            label_ind = f"MMS Longue ({l}j)"

        elif strat_choice == "Mean Reversion (BB)":
            w, std = manual_params['bb_window'], manual_params['bb_std']
            mid = an.data['Close'].rolling(w).mean()
            sigma = an.data['Close'].rolling(w).std()
            lower = mid - (sigma * std)
            signals = np.where(an.data['Close'] < lower, 1.0, 0.0)
            indicator = lower
            label_ind = "Bande Inférieure"

        # Détection des entrées/sorties
        signals = pd.Series(signals, index=an.data.index)
        positions = signals.diff() # +1 = Achat, -1 = Vente
        
        # Dessin
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(an.data.index, an.data['Close'], label="Prix Actif", color="black", alpha=0.6, linewidth=1)
        
        if strat_choice != "Cross MMS" and indicator is not None:
             ax.plot(an.data.index, indicator, label=label_ind, color="orange", linestyle="--", alpha=0.8)
        
        # Marqueurs Achat (Vert)
        buys = an.data.loc[positions == 1.0]
        ax.plot(buys.index, buys['Close'], '^', markersize=10, color='green', label='Achat', zorder=5)
        
        # Marqueurs Vente (Rouge)
        sells = an.data.loc[positions == -1.0]
        ax.plot(sells.index, sells['Close'], 'v', markersize=10, color='red', label='Vente', zorder=5)

        ax.set_title(f"Entrées et Sorties : {strat_choice}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


    # --- SECTION B : BATTLE ROYALE (OPTIMISATION) ---
    st.markdown("---")
    st.subheader("⚔️ Battle Royale : Comparaison des Modèles Optimisés")
    st.caption("Voici les résultats avec les MEILLEURS paramètres trouvés par l'IA.")

    # Calculs optimisés
    curve_mom, ret_mom = an.run_strategy("Momentum", **an.best_params['Momentum'])
    curve_cross, ret_cross = an.run_strategy("Cross MMS", **an.best_params['Cross MMS'])
    curve_bb, ret_bb = an.run_strategy("Mean Reversion (BB)", **an.best_params['Mean Reversion (BB)'])

    # Graphique comparatif
    df_battle = pd.DataFrame({
        "Buy & Hold": bh_curve,
        f"Momentum (Opti)": curve_mom,
        f"Cross MMS (Opti)": curve_cross,
        f"Bollinger (Opti)": curve_bb
    })
    st.line_chart(df_battle)

    # Leaderboard (Tableau des scores)
    st.subheader("🏆 Le Bulletin de Notes")
    met_bh = an.compute_metrics(an.daily_returns)
    met_mom = an.compute_metrics(ret_mom)
    met_cross = an.compute_metrics(ret_cross)
    met_bb = an.compute_metrics(ret_bb)

    leaderboard_data = [
        {"Stratégie": "Buy & Hold", "Sharpe Ratio": met_bh['Sharpe'], "Max Drawdown": met_bh['Max Drawdown'], "Perf Totale": met_bh['Total Perf'], "Capital Final": f"{bh_curve.iloc[-1]:.2f} $"},
        {"Stratégie": "Momentum (IA)", "Sharpe Ratio": met_mom['Sharpe'], "Max Drawdown": met_mom['Max Drawdown'], "Perf Totale": met_mom['Total Perf'], "Capital Final": f"{curve_mom.iloc[-1]:.2f} $"},
        {"Stratégie": "Cross MMS (IA)", "Sharpe Ratio": met_cross['Sharpe'], "Max Drawdown": met_cross['Max Drawdown'], "Perf Totale": met_cross['Total Perf'], "Capital Final": f"{curve_cross.iloc[-1]:.2f} $"},
        {"Stratégie": "Bollinger (IA)", "Sharpe Ratio": met_bb['Sharpe'], "Max Drawdown": met_bb['Max Drawdown'], "Perf Totale": met_bb['Total Perf'], "Capital Final": f"{curve_bb.iloc[-1]:.2f} $"}
    ]
    df_leaderboard = pd.DataFrame(leaderboard_data).set_index("Stratégie").sort_values(by="Sharpe Ratio", ascending=False)
    st.dataframe(df_leaderboard, use_container_width=True)


    # --- SECTION C : PRÉDICTION (BONUS) ---
    st.markdown("---")
    st.subheader(f"🔮 Prédiction Future : {model_choice}")
    
    with st.spinner(f"Calcul du modèle {model_choice} en cours..."):
        fut_d, fut_p, std = an.predict_future(forecast_days, model_type=model_choice)
    
    recent = an.data['Close'].tail(180)
    df_fut = pd.DataFrame({"Pred": fut_p}, index=fut_d)
    
    # Cône d'incertitude temporel (Racine carrée du temps)
    time_scaling = np.sqrt(np.arange(1, len(df_fut) + 1))
    df_fut["High"] = df_fut["Pred"] + (1.96 * std * time_scaling)
    df_fut["Low"] = df_fut["Pred"] - (1.96 * std * time_scaling)
    
    # Graphique de prédiction
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent.values, label="Historique Récent", color="black", alpha=0.6)
    ax.plot(df_fut.index, df_fut["Pred"], label=f"Prédiction ({model_choice})", color="#0068C9", linestyle="--", linewidth=2)
    ax.fill_between(df_fut.index, df_fut["Low"], df_fut["High"], color="#0068C9", alpha=0.15, label="Zone de Confiance 95%")
    
    ax.set_title(f"Projection {ticker} sur {forecast_days} jours")
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    
    # Messages contextuels
    if model_choice == "ARIMA":
        st.info("ℹ️ **ARIMA** analyse les cycles passés. Idéal pour les marchés volatils (Crypto).")
    elif model_choice == "Machine Learning (RF)":
        st.info("ℹ️ **Random Forest** utilise l'IA pour repérer des motifs complexes récents.")
    else:
        st.warning("⚠️ **Régression Linéaire** : Trace une tendance. Mieux pour les actions stables (Air Liquide).")

    ticker_clean = ticker.upper()
    if "BTC" in ticker_clean and model_choice == "ARIMA":
        st.success("✅ Excellent choix pour le Bitcoin !")
    elif ("AI.PA" in ticker_clean or "AIR LIQUIDE" in ticker_clean) and model_choice == "Linear Regression":
        st.success("✅ Excellent choix pour Air Liquide !")

else:
    st.info("👈 Veuillez cliquer sur 'Charger Données & Scanner' dans la barre latérale pour commencer.")