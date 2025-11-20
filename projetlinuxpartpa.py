import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

    def predict_future(self, days_ahead=30):
        """Régression linéaire pour prédiction."""
        df_pred = self.data.copy().reset_index()
        df_pred['Date_Ordinal'] = df_pred['Date'].map(pd.Timestamp.toordinal)
        X = df_pred[['Date_Ordinal']].values
        y = df_pred['Close'].values
        model = LinearRegression().fit(X, y)
        
        last_date = df_pred['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
        future_ordinals = [[d.toordinal()] for d in future_dates]
        predictions = model.predict(future_ordinals)
        
        residuals = y - model.predict(X)
        std_resid = np.std(residuals)
        return future_dates, predictions, std_resid

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
        strat_choice = st.selectbox("Stratégie Active", ["Momentum", "Cross MMS", "Mean Reversion (BB)"])
        
        current_params = {}
        
        # AFFICHAGE INTELLIGENT : On montre les sliders MAIS aussi les valeurs recommandées
        if strat_choice == "Momentum":
            rec = an.best_params['Momentum']['window']
            st.info(f"💡 Suggestion IA : Fenêtre = {rec}")
            current_params['window'] = st.slider("Fenêtre", 10, 200, 50)
            
        elif strat_choice == "Cross MMS":
            rec_s = an.best_params['Cross MMS']['short_w']
            rec_l = an.best_params['Cross MMS']['long_w']
            st.info(f"💡 Suggestion IA : Court={rec_s}, Long={rec_l}")
            current_params['short_w'] = st.slider("Moyenne Courte", 5, 50, 20)
            current_params['long_w'] = st.slider("Moyenne Longue", 50, 200, 100)
            
        elif strat_choice == "Mean Reversion (BB)":
            rec_w = an.best_params['Mean Reversion (BB)']['window']
            rec_std = an.best_params['Mean Reversion (BB)']['std_dev']
            st.info(f"💡 Suggestion IA : Fenêtre={rec_w}, Std={rec_std}")
            current_params['window'] = st.slider("Fenêtre BB", 10, 100, 20)
            current_params['std_dev'] = st.slider("Écart-Type", 1.0, 3.0, 2.0)

        st.markdown("---")
        st.header("3. Options")
        show_pred = st.checkbox("Voir Prédiction (ML)")

# --- AFFICHAGE PRINCIPAL ---

if st.session_state.analyzer:
    an = st.session_state.analyzer
    
    # 1. CALCULS (Stratégie Manuelle Choisie)
    strat_curve, strat_rets = an.run_strategy(strat_choice, **current_params)
    bh_curve = (1 + an.daily_returns).cumprod() * an.initial_investment
    
    met_strat = an.compute_metrics(strat_rets)
    met_bh = an.compute_metrics(an.daily_returns)

    # 2. KPI (Haut de page)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stratégie", strat_choice)
    c2.metric("Sharpe Ratio", met_strat['Sharpe'], delta=f"{met_strat['Sharpe'] - met_bh['Sharpe']:.2f} vs B&H")
    c3.metric("Max Drawdown", met_strat['Max Drawdown'])
    c4.metric("Gain Total", met_strat['Total Perf'])

    # 3. GRAPHIQUE : MANUEL vs BUY & HOLD
    st.subheader("📈 Analyse Détaillée : Manuel vs Marché")
    df_chart = pd.DataFrame({
        "Buy & Hold (Marché)": bh_curve,
        f"Ma Stratégie ({strat_choice})": strat_curve
    })
    st.line_chart(df_chart, color=["#FF4B4B", "#0068C9"])

    # 4. SECTION COMPARATIVE (Le "Battle" des stratégies optimisées)
    st.markdown("---")
    st.subheader("⚔️ Battle Royale : Comparaison des Modèles Optimisés")
    st.caption("Voici ce que ça donnerait si on prenait les MEILLEURS paramètres pour chaque stratégie sur cette période.")

    # On recalcule les courbes optimales pour les afficher
    curve_mom, _ = an.run_strategy("Momentum", **an.best_params['Momentum'])
    curve_cross, _ = an.run_strategy("Cross MMS", **an.best_params['Cross MMS'])
    curve_bb, _ = an.run_strategy("Mean Reversion (BB)", **an.best_params['Mean Reversion (BB)'])

    df_battle = pd.DataFrame({
        "Buy & Hold": bh_curve,
        f"Momentum (Opti: {an.best_params['Momentum']['window']})": curve_mom,
        f"Cross MMS (Opti)": curve_cross,
        f"Bollinger (Opti)": curve_bb
    })
    st.line_chart(df_battle)

    # 5. BONUS PRÉDICTION
    if show_pred:
        st.markdown("---")
        st.subheader("🔮 Boule de Cristal (Prédiction)")
        fut_d, fut_p, std = an.predict_future(30)
        
        recent = an.data['Close'].tail(100)
        df_fut = pd.DataFrame({"Pred": fut_p, "High": fut_p+1.96*std, "Low": fut_p-1.96*std}, index=fut_d)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(recent.index, recent.values, label="Historique", color="black")
        ax.plot(df_fut.index, df_fut["Pred"], label="Prédiction", color="blue", linestyle="--")
        ax.fill_between(df_fut.index, df_fut["Low"], df_fut["High"], color="blue", alpha=0.1)
        ax.legend()
        st.pyplot(fig)

else:
    st.info("👈 Veuillez cliquer sur 'Charger Données & Scanner' dans la barre latérale pour commencer.")