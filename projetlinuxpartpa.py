import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import itertools

# --- 1. CLASSE D'ANALYSE D'ACTIF (Le coeur du système) ---

class AssetAnalyzer:
    def __init__(self, ticker, start_date, end_date, initial_investment=100):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.data = None
        self.daily_returns = None
        self.results = {} # Stockera les courbes de prix
        self.metrics = {} # Stockera les performances (Sharpe, etc.)
        self.best_params = {} # Pour savoir quels paramètres ont gagné

    def load_data(self):
        """Télécharge et prépare les données."""
        try:
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
            if df.empty:
                return False
            
            # Gestion des formats multi-index de yfinance récents
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('Close', axis=1, level=0, drop_level=False)
                df.columns = ['Close']
            else:
                df = df[['Close']]

            self.data = df
            self.daily_returns = self.data['Close'].pct_change().fillna(0)
            
            # Stratégie de base : Buy & Hold
            bh_curve = (1 + self.daily_returns).cumprod() * self.initial_investment
            self.results['Buy_and_Hold'] = bh_curve
            self.metrics['Buy_and_Hold'] = self._calculate_metrics(self.daily_returns)
            return True
        except Exception as e:
            st.error(f"Erreur data {self.ticker}: {e}")
            return False

    def _calculate_metrics(self, returns_series):
        """Calcule Sharpe, Drawdown et Perf Totale."""
        # NETTOYAGE : On enlève les lignes vides (le dernier jour souvent NaN à cause du shift)
        returns_series = returns_series.dropna()
        
        if returns_series.empty:
            return {'Sharpe': 0, 'Max Drawdown': 0, 'Total Perf': 0}

        # Sharpe
        risk_free = 0.03
        ann_factor = 252
        mean_ret = returns_series.mean()
        std_ret = returns_series.std()
        
        if std_ret == 0: 
            sharpe = 0
        else: 
            daily_rf = (1 + risk_free)**(1/ann_factor) - 1
            sharpe = (mean_ret - daily_rf) / std_ret * np.sqrt(ann_factor)
        
        # Drawdown & Perf
        cum_ret = (1 + returns_series).cumprod()
        peak = cum_ret.expanding(min_periods=1).max()
        dd = (cum_ret - peak) / peak
        max_dd = abs(dd.min())

        # Correction ici : on prend la dernière valeur valide
        total_perf = cum_ret.iloc[-1] - 1

        return {'Sharpe': sharpe, 'Max Drawdown': max_dd, 'Total Perf': total_perf}
    # --- MOTEUR D'OPTIMISATION ---
    
    def optimize_momentum(self):
        """Teste plusieurs fenêtres et garde la meilleure."""
        best_sharpe = -np.inf
        best_window = 20
        best_curve = None
        best_rets = None

        # Plage de recherche : de 10 à 100 par pas de 5
        windows = range(10, 105, 5)
        
        for w in windows:
            mms = self.data['Close'].rolling(window=w).mean()
            signal = np.where(self.data['Close'] > mms, 1.0, 0.0)
            # Shift(1) pour éviter le look-ahead bias
            strat_ret = self.daily_returns.shift(-1) * pd.Series(signal, index=self.data.index).shift(1).fillna(0)
            
            metrics = self._calculate_metrics(strat_ret)
            if metrics['Sharpe'] > best_sharpe:
                best_sharpe = metrics['Sharpe']
                best_window = w
                best_rets = strat_ret
                best_curve = (1 + strat_ret).cumprod() * self.initial_investment

        name = f"Mom_Best({best_window}d)"
        self.results[name] = best_curve
        self.metrics[name] = self._calculate_metrics(best_rets)
        self.best_params['Momentum'] = f"Window: {best_window}"

    def optimize_cross_mms(self):
        """Teste croisements MMS Court/Long."""
        best_sharpe = -np.inf
        best_params = (10, 50)
        best_curve = None
        best_rets = None

        short_windows = range(5, 30, 5)
        long_windows = range(40, 100, 10)

        for s, l in itertools.product(short_windows, long_windows):
            if s >= l: continue
            
            mms_s = self.data['Close'].rolling(window=s).mean()
            mms_l = self.data['Close'].rolling(window=l).mean()
            signal = np.where(mms_s > mms_l, 1.0, 0.0)
            strat_ret = self.daily_returns.shift(-1) * pd.Series(signal, index=self.data.index).shift(1).fillna(0)

            metrics = self._calculate_metrics(strat_ret)
            if metrics['Sharpe'] > best_sharpe:
                best_sharpe = metrics['Sharpe']
                best_params = (s, l)
                best_rets = strat_ret
                best_curve = (1 + strat_ret).cumprod() * self.initial_investment

        name = f"Cross_Best({best_params[0]}/{best_params[1]})"
        self.results[name] = best_curve
        self.metrics[name] = self._calculate_metrics(best_rets)
        self.best_params['Cross MMS'] = f"Short: {best_params[0]}, Long: {best_params[1]}"

    def optimize_bollinger(self):
        """Teste Bollinger (Mean Reversion)."""
        best_sharpe = -np.inf
        best_params = (20, 2.0)
        best_curve = None
        best_rets = None

        windows = range(10, 50, 5)
        stds = [1.5, 2.0, 2.5]

        for w, n in itertools.product(windows, stds):
            mid = self.data['Close'].rolling(window=w).mean()
            std = self.data['Close'].rolling(window=w).std()
            lower = mid - (std * n)
            
            # Signal : Achat si prix < lower, Vente si prix > mid
            signal = np.where(self.data['Close'] < lower, 1.0, 0.0)
            position = pd.Series(signal, index=self.data.index).replace(0, np.nan) # Astuce pour ffill
            
            # On sort quand on touche la moyenne (simplification vectorielle)
            # Pour être précis vectoriellement sans boucle, c'est complexe. 
            # Ici on garde une logique simple : 1 si < lower, 0 sinon (approche pure signal, moins "position holding")
            # Pour l'optimisation rapide, on va considérer : Long si Close < Lower, Exit si Close > Mid
            # Approximation vectorielle :
            sig_entry = (self.data['Close'] < lower)
            sig_exit = (self.data['Close'] > mid)
            
            # Logique de position stateful (un peu lent mais nécessaire pour Bollinger)
            pos = 0
            pos_arr = []
            for i in range(len(self.data)):
                if sig_entry.iloc[i]: pos = 1
                elif sig_exit.iloc[i]: pos = 0
                pos_arr.append(pos)
            
            position_final = pd.Series(pos_arr, index=self.data.index)
            
            strat_ret = self.daily_returns.shift(-1) * position_final.shift(1).fillna(0)
            
            metrics = self._calculate_metrics(strat_ret)
            if metrics['Sharpe'] > best_sharpe:
                best_sharpe = metrics['Sharpe']
                best_params = (w, n)
                best_rets = strat_ret
                best_curve = (1 + strat_ret).cumprod() * self.initial_investment

        name = f"BB_Best({best_params[0]}d/{best_params[1]}std)"
        self.results[name] = best_curve
        self.metrics[name] = self._calculate_metrics(best_rets)
        self.best_params['Bollinger'] = f"Win: {best_params[0]}, Std: {best_params[1]}"

    def run_all_optimizations(self):
        self.optimize_momentum()
        self.optimize_cross_mms()
        self.optimize_bollinger()

    def get_results_df(self):
        return pd.DataFrame(self.results)

# --- 2. INTERFACE STREAMLIT ---

st.set_page_config(layout="wide", page_title="Auto-Quant Optimizer")
st.title("⚡ Auto-Quant: Optimisation Automatique")
st.markdown("Ce système teste automatiquement des centaines de combinaisons pour trouver les paramètres optimaux (Max Sharpe) sur la période donnée.")

with st.sidebar:
    st.header("Configuration")
    ticker_input = st.text_input("Ticker (Yahoo Finance)", value="BTC-USD")
    start_date = st.date_input("Début", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Fin", date.today())
    
    st.markdown("---")
    st.info("Plus besoin de sélectionner les plages. L'algo va chercher les meilleurs paramètres tout seul.")
    
    run_btn = st.button("Lancer l'Optimisation & Backtest")

if run_btn:
    with st.spinner(f'Optimisation des stratégies pour {ticker_input}...'):
        # Création de l'objet (C'est ici que la POO aide pour le futur portefeuille)
        asset = AssetAnalyzer(ticker_input, start_date, end_date)
        
        if asset.load_data():
            # Lancement des recherches
            asset.run_all_optimizations()
            
            # Récupération des résultats
            df_results = asset.get_results_df()
            metrics_dict = asset.metrics
            best_params = asset.best_params

            # --- AFFICHAGE ---
            
            # 1. Paramètres Gagnants
            st.subheader("🏆 Paramètres Optimaux Découverts")
            c1, c2, c3 = st.columns(3)
            c1.success(f"**Momentum**\n\n{best_params.get('Momentum')}")
            c2.success(f"**Cross MMS**\n\n{best_params.get('Cross MMS')}")
            c3.success(f"**Bollinger**\n\n{best_params.get('Bollinger')}")

            # 2. Graphique
            st.subheader("Performance Comparée (Base 100)")
            st.line_chart(df_results)

            # 3. Tableau Métriques
            st.subheader("Détails de Performance")
            metrics_df = pd.DataFrame(metrics_dict).T
            metrics_df = metrics_df.sort_values(by="Sharpe", ascending=False)
            
            # Formatage
            st.dataframe(metrics_df.style.format({
                'Sharpe': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                'Total Perf': '{:.2%}'
            }))
            
        else:
            st.error("Impossible de récupérer les données.")