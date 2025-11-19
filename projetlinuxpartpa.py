import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# --- 1. FONCTIONS DE PERFORMANCE ---

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

# --- 2. FONCTION PRINCIPALE DE L'APPLICATION (Mise en cache) ---

# st.cache_data est utilisé pour que Streamlit n'exécute pas tout le calcul à chaque interaction, 
# seulement si les paramètres d'entrée changent.
@st.cache_data
def load_and_backtest(ticker, start_date, end_date, mom_window, short_w, long_w, bb_window, n_std_dev):
    """ Récupère les données et exécute le backtesting pour les quatre stratégies. """
    
    initial_investment = 100
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        asset_prices = data[['Close']].copy()
        asset_prices.columns = [ticker]
    except Exception as e:
        st.error(f"Erreur de chargement des données pour {ticker} : {e}")
        return None, None
    
    # ------------------ PRÉPARATION DES DONNÉES DE BASE ------------------
    daily_returns = asset_prices[ticker].pct_change().fillna(0)
    
    # Normalisation du prix brut pour la comparaison (Raw Price Value)
    normalized_price = (asset_prices[ticker] / asset_prices[ticker].iloc[0]) * initial_investment
    normalized_price.name = 'Raw_Price_Value'
    
    final_df = normalized_price.to_frame() # Initialiser le DataFrame pour le graphique
    metrics = {}

    # ------------------ STRATÉGIE 1: BUY-AND-HOLD ------------------
    cumulative_value_bh = (1 + daily_returns).cumprod() * initial_investment
    cumulative_value_bh.name = 'Buy_and_Hold'
    final_df = pd.concat([final_df, cumulative_value_bh], axis=1)
    
    metrics['Buy-and-Hold'] = {
        'Sharpe': calculate_sharpe_ratio(daily_returns),
        'Max Drawdown': calculate_max_drawdown(cumulative_value_bh),
        'Performance Totale': (cumulative_value_bh.iloc[-1] / initial_investment) - 1
    }

    # ------------------ STRATÉGIE 2: MOMENTUM (MMS Simple) ------------------
    asset_prices.loc[:, 'MMS'] = asset_prices[ticker].rolling(window=mom_window).mean()
    asset_prices.loc[:, 'Signal_Mom'] = np.where(asset_prices[ticker] > asset_prices['MMS'], 1.0, 0.0)
    strategy_returns_mom = daily_returns.shift(-1) * asset_prices['Signal_Mom'].shift(1).fillna(0)
    cumulative_value_mom = (1 + strategy_returns_mom).cumprod() * initial_investment
    
    strategy_name_mom = f'Momentum_{mom_window}D'
    cumulative_value_mom.name = strategy_name_mom
    final_df[strategy_name_mom] = cumulative_value_mom
    
    metrics[strategy_name_mom] = {
        'Sharpe': calculate_sharpe_ratio(strategy_returns_mom),
        'Max Drawdown': calculate_max_drawdown(cumulative_value_mom),
        'Performance Totale': (cumulative_value_mom.iloc[-1] / initial_investment) - 1
    }

    # ------------------ STRATÉGIE 3: CROISEMENT MMS ------------------
    asset_prices.loc[:, 'MMS_Short'] = asset_prices[ticker].rolling(window=short_w).mean()
    asset_prices.loc[:, 'MMS_Long'] = asset_prices[ticker].rolling(window=long_w).mean()
    asset_prices.loc[:, 'Signal_Cross'] = np.where(asset_prices['MMS_Short'] > asset_prices['MMS_Long'], 1.0, 0.0)
    strategy_returns_cross = daily_returns.shift(-1) * asset_prices['Signal_Cross'].shift(1).fillna(0)
    cumulative_value_cross = (1 + strategy_returns_cross).cumprod() * initial_investment
    
    strategy_name_cross = f'Cross_{short_w}_{long_w}D'
    cumulative_value_cross.name = strategy_name_cross
    final_df[strategy_name_cross] = cumulative_value_cross
    
    metrics[strategy_name_cross] = {
        'Sharpe': calculate_sharpe_ratio(strategy_returns_cross),
        'Max Drawdown': calculate_max_drawdown(cumulative_value_cross),
        'Performance Totale': (cumulative_value_cross.iloc[-1] / initial_investment) - 1
    }

    # ------------------ STRATÉGIE 4: BANDES DE BOLLINGER (Mean Reversion) ------------------
    asset_prices.loc[:, 'BB_Mid'] = asset_prices[ticker].rolling(window=bb_window).mean()
    asset_prices.loc[:, 'BB_Std'] = asset_prices[ticker].rolling(window=bb_window).std()
    asset_prices.loc[:, 'BB_Lower'] = asset_prices['BB_Mid'] - (asset_prices['BB_Std'] * n_std_dev)
    
    # Logique d'Entrée (Achat si sous la bande inférieure)
    asset_prices.loc[:, 'Signal_BB'] = np.where(asset_prices[ticker] < asset_prices['BB_Lower'], 1.0, 0.0)
    
    # Logique de Sortie (Sortie si repasse au-dessus de la bande moyenne)
    # Remplir les jours de '1.0' (position ouverte) jusqu'à ce que la sortie soit déclenchée.
    asset_prices.loc[:, 'Position_BB'] = asset_prices['Signal_BB'].ffill().fillna(0)
    asset_prices.loc[asset_prices[ticker] > asset_prices['BB_Mid'], 'Position_BB'] = 0.0
    
    strategy_returns_bb = daily_returns.shift(-1) * asset_prices['Position_BB'].shift(1).fillna(0)
    cumulative_value_bb = (1 + strategy_returns_bb).cumprod() * initial_investment
    
    strategy_name_bb = f'BB_{bb_window}D_{n_std_dev}x'
    cumulative_value_bb.name = strategy_name_bb
    final_df[strategy_name_bb] = cumulative_value_bb
    
    metrics[strategy_name_bb] = {
        'Sharpe': calculate_sharpe_ratio(strategy_returns_bb),
        'Max Drawdown': calculate_max_drawdown(cumulative_value_bb),
        'Performance Totale': (cumulative_value_bb.iloc[-1] / initial_investment) - 1
    }
    
    # Renvoyer le DataFrame final et les métriques
    return final_df, metrics


# --- 3. MISE EN PAGE STREAMLIT ---

st.set_page_config(layout="wide")
st.title("🔬 Quant A: Analyse Univariée d'Actif Unique")
st.markdown("Plateforme de backtesting pour évaluer les stratégies quantitatives sur un seul actif.")

# --- BARRE LATÉRALE (Contrôles Interactifs) ---
with st.sidebar:
    st.header("⚙️ Paramètres de l'Analyse")
    
    # Sélecteur d'Actif
    selected_ticker = st.selectbox(
        "Sélectionner l'Actif", 
        ['^GSPC', 'BTC-USD', 'EURUSD=X', 'GC=F'], 
        index=1 # BTC-USD par défaut pour les résultats intéressants
    )

    # Période de backtesting
    st.subheader("Période")
    today = date.today()
    start_date_input = st.date_input("Date de Début", pd.to_datetime('2020-01-01'))
    end_date_input = st.date_input("Date de Fin", today)
    
    # Paramètres de la Stratégie (Momentum)
    st.subheader("Momentum (MMS Simple)")
    momentum_window = st.slider("Période MMS (jours)", 20, 100, 50, 10)
    
    # Paramètres de la Stratégie (Croisement MMS)
    st.subheader("Croisement MMS")
    short_window = st.slider("MMS Rapide (jours)", 5, 50, 20)
    long_window = st.slider("MMS Lente (jours)", 50, 200, 100)
    
    # Paramètres de la Stratégie (Bollinger)
    st.subheader("Bandes de Bollinger")
    bb_window = st.slider("Période BB (jours)", 10, 50, 20)
    n_std_dev = st.slider("Multiplicateur Écart-Type", 1.0, 3.0, 2.0, 0.1)

    # Exécuter le backtesting
    st.markdown("---")
    if st.button("Lancer l'Analyse & Backtesting"):
        st.session_state.run_analysis = True
    
# ----------------------------------------------------------------------

# --- LOGIQUE D'AFFICHAGE DES RÉSULTATS ---
if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    
    st.info(f"Analyse en cours pour l'actif : **{selected_ticker}**")
    
    data_df, metrics_results = load_and_backtest(
        selected_ticker, 
        start_date_input.strftime('%Y-%m-%d'), 
        end_date_input.strftime('%Y-%m-%d'), 
        momentum_window, short_window, long_window, 
        bb_window, n_std_dev
    )

    if data_df is not None:
        
        # 1. GRAPHIQUE PRINCIPAL (avec sélection des stratégies à afficher)
        st.header("Graphique des Performances Cumulées (Base 100)")
        
        strategies_to_plot = st.multiselect(
            "Sélectionner les stratégies à afficher :", 
            options=data_df.columns.tolist(),
            default=data_df.columns.tolist() # Toutes affichées par défaut
        )
        
        # Affichage du graphique de la performance
        if strategies_to_plot:
            st.line_chart(data_df[strategies_to_plot], use_container_width=True) 
        else:
            st.warning("Veuillez sélectionner au moins une courbe à afficher.")

        st.markdown("---")
        
        # 2. AFFICHAGE DES MÉTRIQUES (Tableau de comparaison)
        st.header("Synthèse des Métriques de Performance")
        
        # Créer le DataFrame des métriques pour un affichage clair
        metrics_df = pd.DataFrame(metrics_results).T
        
        # Mise en forme des valeurs
        metrics_df.columns = ['Sharpe Ratio', 'Max Drawdown', 'Performance Totale']
        metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].map('{:.2f}'.format)
        metrics_df['Max Drawdown'] = metrics_df['Max Drawdown'].map('{:.2%}'.format)
        metrics_df['Performance Totale'] = metrics_df['Performance Totale'].map('{:.2%}'.format)
        
        st.table(metrics_df.sort_values(by='Sharpe Ratio', ascending=False))
        
        st.markdown("---")
        
        # 3. AFFICHAGE DES DONNÉES BRUTES (Optionnel, pour le débogage)
        if st.checkbox("Afficher les données brutes du Backtesting"):
            st.subheader("Données des Prix et Signaux")
            st.dataframe(data_df)
    
# --- Fin de l'application Streamlit ---