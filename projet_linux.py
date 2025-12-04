
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import itertools

# NEW: Plotly
import plotly.express as px
import plotly.graph_objects as go


# --- 1. CLASSE D'ANALYSE (Backend Logic) ---
class SingleAssetAnalyzer:
    def __init__(self, ticker, start_date, end_date, initial_investment=1000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.data = pd.DataFrame()
        self.daily_returns = pd.Series(dtype=float)
        self.best_params = {}  # Pour stocker les recommandations

    def load_data(self):
        """Télécharge les données."""
        try:
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs("Close", axis=1, level=0, drop_level=False)
                df.columns = ["Close"]
            else:
                df = df[["Close"]]

            if df.empty:
                return False

            self.data = df
            self.daily_returns = self.data["Close"].pct_change().fillna(0)
            return True
        except Exception as e:
            st.error(f"Erreur chargement : {e}")
            return False

    def compute_metrics(self, strategy_returns):
        """Calcule Sharpe, Max Drawdown et Performance Totale."""
        strategy_returns = strategy_returns.dropna()
        if strategy_returns.empty:
            return {"Sharpe": 0.0, "Max Drawdown": 0.0, "Total Perf": 0.0}

        rf = 0.03
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()

        if std_ret == 0:
            sharpe = 0
        else:
            sharpe = (mean_ret - (rf / 252)) / std_ret * np.sqrt(252)

        cum_ret = (1 + strategy_returns).cumprod()
        peak = cum_ret.expanding(min_periods=1).max()
        dd = (cum_ret - peak) / peak
        max_dd = abs(dd.min())
        total_perf = cum_ret.iloc[-1] - 1

        return {
            "Sharpe": round(sharpe, 2),
            "Max Drawdown": f"{max_dd:.2%}",
            "Total Perf": f"{total_perf:.2%}",
            "Raw_Sharpe": sharpe,  # Pour le tri interne
        }

    def run_strategy(self, strat_name, **params):
        """Exécute une stratégie spécifique avec des paramètres donnés."""
        signals = pd.Series(0, index=self.data.index)

        # --- LOGIQUE DES STRATÉGIES ---
        if strat_name == "Momentum":
            window = int(params.get("window", 50))
            mms = self.data["Close"].rolling(window=window).mean()
            signals = np.where(self.data["Close"] > mms, 1.0, 0.0)

        elif strat_name == "Cross MMS":
            short_w = int(params.get("short_w", 20))
            long_w = int(params.get("long_w", 50))
            mms_short = self.data["Close"].rolling(window=short_w).mean()
            mms_long = self.data["Close"].rolling(window=long_w).mean()
            signals = np.where(mms_short > mms_long, 1.0, 0.0)

        elif strat_name == "Mean Reversion (BB)":
            window = int(params.get("window", 20))
            std_dev = float(params.get("std_dev", 2.0))
            sma = self.data["Close"].rolling(window=window).mean()
            std = self.data["Close"].rolling(window=window).std()
            lower_band = sma - (std * std_dev)
            # Achat si < Lower Band, Vente si > SMA (simplifié)
            signals = np.where(self.data["Close"] < lower_band, 1.0, 0.0)

        # Backtest
        signals = pd.Series(signals, index=self.data.index)
        strat_returns = self.daily_returns.shift(-1) * signals.shift(1).fillna(0)
        strat_curve = (1 + strat_returns).cumprod() * self.initial_investment
        strat_curve = strat_curve.ffill()  # Correction du bug NaN à la fin

        return strat_curve, strat_returns

    # --- PARTIE OPTIMISATION (LE CERVEAU) ---
    def find_best_params(self):
        """Teste plein de combinaisons et stocke les gagnantes."""

        # 1. Optimisation Momentum
        best_sharpe = -999
        best_p = {"window": 50}
        for w in range(10, 100, 10):
            _, rets = self.run_strategy("Momentum", window=w)
            m = self.compute_metrics(rets)
            if m["Raw_Sharpe"] > best_sharpe:
                best_sharpe = m["Raw_Sharpe"]
                best_p = {"window": w}
        self.best_params["Momentum"] = best_p

        # 2. Optimisation Cross MMS
        best_sharpe = -999
        best_p = {"short_w": 20, "long_w": 50}
        for s, l in itertools.product(range(10, 50, 10), range(50, 150, 20)):
            if s >= l:
                continue
            _, rets = self.run_strategy("Cross MMS", short_w=s, long_w=l)
            m = self.compute_metrics(rets)
            if m["Raw_Sharpe"] > best_sharpe:
                best_sharpe = m["Raw_Sharpe"]
                best_p = {"short_w": s, "long_w": l}
        self.best_params["Cross MMS"] = best_p

        # 3. Optimisation BB
        best_sharpe = -999
        best_p = {"window": 20, "std_dev": 2.0}
        for w, std in itertools.product(range(10, 50, 10), [1.5, 2.0, 2.5]):
            _, rets = self.run_strategy("Mean Reversion (BB)", window=w, std_dev=std)
            m = self.compute_metrics(rets)
            if m["Raw_Sharpe"] > best_sharpe:
                best_sharpe = m["Raw_Sharpe"]
                best_p = {"window": w, "std_dev": std}
        self.best_params["Mean Reversion (BB)"] = best_p

    # new fontion pour les différents modèles
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
            df["Date_Ordinal"] = df["Date"].map(pd.Timestamp.toordinal)
            X = df[["Date_Ordinal"]].values
            y = df["Close"].values

            # Entrainement sur tout l'historique pour avoir la PENTE (la direction)
            model = LinearRegression().fit(X, y)

            future_ordinals = [[d.toordinal()] for d in future_dates]
            preds = model.predict(future_ordinals)

            # Ancrage
            last_day_ordinal = [[X[-1][0]]]
            theoretical_price_today = model.predict(last_day_ordinal)[0]
            actual_price_today = y[-1]
            offset = actual_price_today - theoretical_price_today
            preds = preds + offset

            # Volatilité locale
            recent_returns = df["Close"].pct_change().tail(90)
            sigma_pct = recent_returns.std()
            std_dev = sigma_pct * df["Close"].iloc[-1]

            return future_dates, preds, std_dev

        # --- MODÈLE 2 : ARIMA (AutoRegressive Integrated Moving Average) ---
        elif model_type == "ARIMA":
            history = df["Close"].values
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit()

            preds = model_fit.forecast(steps=days_ahead)

            residuals = model_fit.resid
            std_dev = np.std(residuals[1:])
            return future_dates, preds, std_dev

        # --- MODÈLE 3 : RANDOM FOREST (Machine Learning) ---
        elif model_type == "Machine Learning (RF)":
            df["Lag1"] = df["Close"].shift(1)
            df["Lag2"] = df["Close"].shift(2)
            df["MA5"] = df["Close"].rolling(5).mean()
            df = df.dropna()

            X = df[["Lag1", "Lag2", "MA5"]].values
            y = df["Close"].values

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            preds = []
            current_lag1 = df["Close"].iloc[-1]
            current_lag2 = df["Close"].iloc[-2]
            current_ma = df["MA5"].iloc[-1]

            for _ in range(days_ahead):
                pred = model.predict([[current_lag1, current_lag2, current_ma]])[0]
                preds.append(pred)
                current_lag2 = current_lag1
                current_lag1 = pred

            train_preds = model.predict(X)
            std_dev = np.std(y - train_preds)

            return future_dates, np.array(preds), std_dev

        return [], [], 0  # Fallback si erreur


# --- 2. INTERFACE STREAMLIT ---

st.set_page_config(layout="wide", page_title="Smart Quant Lab")
st.title("🧠 Smart Quant Lab: Analyse & Optimisation")

# Initialisation session state pour ne pas perdre les calculs
if "analyzer" not in st.session_state:
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
                an.find_best_params()  # On lance l'optimisation ici
            st.session_state.analyzer = an
            st.success("Scan terminé !")

    st.markdown("---")

    # On affiche les contrôles seulement si l'analyseur est chargé
    if st.session_state.analyzer:
        an = st.session_state.analyzer

        st.header("2. Contrôle Manuel")
        strat_choice = st.selectbox(
            "Stratégie Active", ["Momentum", "Cross MMS", "Mean Reversion (BB)", "TOUT COMPARER"]
        )

        manual_params = {}

        if strat_choice == "Momentum" or strat_choice == "TOUT COMPARER":
            st.markdown("### Paramètres Momentum")
            rec = an.best_params["Momentum"]["window"]
            st.caption(f"💡 Suggestion IA : {rec}")
            manual_params["mom_window"] = st.slider(
                "Fenêtre Momentum", 10, 200, 50, key="mom_slider"
            )

        if strat_choice == "Cross MMS" or strat_choice == "TOUT COMPARER":
            st.markdown("### Paramètres Cross MMS")
            rec_s = an.best_params["Cross MMS"]["short_w"]
            rec_l = an.best_params["Cross MMS"]["long_w"]
            st.caption(f"💡 Suggestion IA : Court={rec_s}, Long={rec_l}")
            manual_params["cross_short"] = st.slider(
                "MMS Court", 5, 50, 20, key="cross_s_slider"
            )
            manual_params["cross_long"] = st.slider(
                "MMS Long", 50, 200, 100, key="cross_l_slider"
            )

        if strat_choice == "Mean Reversion (BB)" or strat_choice == "TOUT COMPARER":
            st.markdown("### Paramètres Bollinger")
            rec_w = an.best_params["Mean Reversion (BB)"]["window"]
            rec_std = an.best_params["Mean Reversion (BB)"]["std_dev"]
            st.caption(f"💡 Suggestion IA : Fenêtre={rec_w}, Std={rec_std}")
            manual_params["bb_window"] = st.slider(
                "Fenêtre BB", 10, 100, 20, key="bb_w_slider"
            )
            manual_params["bb_std"] = st.slider(
                "Écart-Type", 1.0, 3.0, 2.0, key="bb_std_slider"
            )

        st.markdown("---")
        st.header("3. Prédiction (Bonus)")

        model_choice = st.selectbox(
            "Choisir le Modèle", ["Linear Regression", "ARIMA", "Machine Learning (RF)"]
        )
        forecast_days = st.slider("Jours à prédire", 7, 90, 30)

# --- AFFICHAGE PRINCIPAL ---

if st.session_state.analyzer:
    an = st.session_state.analyzer
    bh_curve = (1 + an.daily_returns).cumprod() * an.initial_investment

    # --- CAS 1 : COMPARAISON GLOBALE ---
    if strat_choice == "TOUT COMPARER":
        st.subheader("⚡ Comparaison Multi-Stratégies (Paramètres Manuels)")

        c_mom, _ = an.run_strategy("Momentum", window=manual_params["mom_window"])
        c_cross, _ = an.run_strategy(
            "Cross MMS",
            short_w=manual_params["cross_short"],
            long_w=manual_params["cross_long"],
        )
        c_bb, _ = an.run_strategy(
            "Mean Reversion (BB)",
            window=manual_params["bb_window"],
            std_dev=manual_params["bb_std"],
        )

        df_all = pd.DataFrame(
            {
                "Buy & Hold": bh_curve,
                "Momentum": c_mom,
                "Cross MMS": c_cross,
                "Bollinger": c_bb,
            }
        )

        fig_all = px.line(
            df_all,
            labels={"index": "Date", "value": "Valeur du portefeuille", "variable": "Stratégie"},
        )
        st.plotly_chart(fig_all, use_container_width=True)

        st.write("### Valeurs Finales du Portefeuille")
        res_finaux = df_all.iloc[-1].sort_values(ascending=False)
        st.dataframe(res_finaux.map("{:.2f} $".format))

    # --- CAS 2 : MODE SOLO ---
    else:
        args = {}
        if strat_choice == "Momentum":
            args = {"window": manual_params["mom_window"]}
        elif strat_choice == "Cross MMS":
            args = {
                "short_w": manual_params["cross_short"],
                "long_w": manual_params["cross_long"],
            }
        elif strat_choice == "Mean Reversion (BB)":
            args = {
                "window": manual_params["bb_window"],
                "std_dev": manual_params["bb_std"],
            }

        strat_curve, strat_rets = an.run_strategy(strat_choice, **args)
        met_strat = an.compute_metrics(strat_rets)
        met_bh = an.compute_metrics(an.daily_returns)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stratégie", strat_choice)
        c2.metric(
            "Sharpe Ratio",
            met_strat["Sharpe"],
            delta=f"{met_strat['Sharpe'] - met_bh['Sharpe']:.2f} vs B&H",
        )
        c3.metric("Max Drawdown", met_strat["Max Drawdown"])
        c4.metric("Gain Total", met_strat["Total Perf"])

        st.subheader(f"📈 Analyse : {strat_choice} vs Marché")
        df_chart = pd.DataFrame(
            {"Buy & Hold (Marché)": bh_curve, f"Ma Stratégie ({strat_choice})": strat_curve}
        )

        # Couleurs proches de ta version
        color_map = {}
        cols = df_chart.columns.tolist()
        if len(cols) > 0:
            color_map[cols[0]] = "#FF4B4B"
        if len(cols) > 1:
            color_map[cols[1]] = "#0068C9"

        fig_chart = px.line(
            df_chart,
            labels={"index": "Date", "value": "Valeur du portefeuille", "variable": "Courbe"},
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig_chart, use_container_width=True)

    # 4. SECTION COMPARATIVE (Le "Battle" des stratégies optimisées)
    st.markdown("---")
    st.subheader("⚔️ Battle Royale : Comparaison des Modèles Optimisés")
    st.caption(
        "Voici ce que ça donnerait si on prenait les MEILLEURS paramètres pour chaque stratégie sur cette période."
    )

    curve_mom, ret_mom = an.run_strategy("Momentum", **an.best_params["Momentum"])
    curve_cross, ret_cross = an.run_strategy("Cross MMS", **an.best_params["Cross MMS"])
    curve_bb, ret_bb = an.run_strategy(
        "Mean Reversion (BB)", **an.best_params["Mean Reversion (BB)"]
    )

    df_battle = pd.DataFrame(
        {
            "Buy & Hold": bh_curve,
            "Momentum (Opti)": curve_mom,
            "Cross MMS (Opti)": curve_cross,
            "Bollinger (Opti)": curve_bb,
        }
    )

    fig_battle = px.line(
        df_battle,
        labels={"index": "Date", "value": "Valeur du portefeuille", "variable": "Stratégie"},
    )
    st.plotly_chart(fig_battle, use_container_width=True)

    st.subheader("🏆 Le Bulletin de Notes")

    met_bh = an.compute_metrics(an.daily_returns)
    met_mom = an.compute_metrics(ret_mom)
    met_cross = an.compute_metrics(ret_cross)
    met_bb = an.compute_metrics(ret_bb)

    leaderboard_data = [
        {
            "Stratégie": "Buy & Hold (Marché)",
            "Sharpe Ratio": met_bh["Sharpe"],
            "Max Drawdown": met_bh["Max Drawdown"],
            "Perf Totale": met_bh["Total Perf"],
            "Capital Final ($)": f"{bh_curve.iloc[-1]:.2f} $",
        },
        {
            "Stratégie": f"Momentum (Win: {an.best_params['Momentum']['window']})",
            "Sharpe Ratio": met_mom["Sharpe"],
            "Max Drawdown": met_mom["Max Drawdown"],
            "Perf Totale": met_mom["Total Perf"],
            "Capital Final ($)": f"{curve_mom.iloc[-1]:.2f} $",
        },
        {
            "Stratégie": f"Cross MMS (S:{an.best_params['Cross MMS']['short_w']} L:{an.best_params['Cross MMS']['long_w']})",
            "Sharpe Ratio": met_cross["Sharpe"],
            "Max Drawdown": met_cross["Max Drawdown"],
            "Perf Totale": met_cross["Total Perf"],
            "Capital Final ($)": f"{curve_cross.iloc[-1]:.2f} $",
        },
        {
            "Stratégie": f"Bollinger (W:{an.best_params['Mean Reversion (BB)']['window']} Std:{an.best_params['Mean Reversion (BB)']['std_dev']})",
            "Sharpe Ratio": met_bb["Sharpe"],
            "Max Drawdown": met_bb["Max Drawdown"],
            "Perf Totale": met_bb["Total Perf"],
            "Capital Final ($)": f"{curve_bb.iloc[-1]:.2f} $",
        },
    ]

    df_leaderboard = pd.DataFrame(leaderboard_data)
    df_leaderboard.set_index("Stratégie", inplace=True)
    df_leaderboard.sort_values(by="Sharpe Ratio", ascending=False, inplace=True)

    st.dataframe(df_leaderboard, use_container_width=True)

    # B. SECTION PRÉDICTION
    st.markdown("---")
    st.subheader(f"🔮 Prédiction Future : {model_choice}")

    with st.spinner(f"Calcul du modèle {model_choice} en cours..."):
        fut_d, fut_p, std = an.predict_future(forecast_days, model_type=model_choice)

    recent = an.data["Close"].tail(180)
    df_fut = pd.DataFrame({"Pred": fut_p}, index=fut_d)

    time_scaling = np.sqrt(np.arange(1, len(df_fut) + 1))
    df_fut["High"] = df_fut["Pred"] + (1.96 * std * time_scaling)
    df_fut["Low"] = df_fut["Pred"] - (1.96 * std * time_scaling)

    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.values,
            mode="lines",
            name="Historique Récent",
            line=dict(color="black"),
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=df_fut.index,
            y=df_fut["Pred"],
            mode="lines",
            name=f"Prédiction ({model_choice})",
            line=dict(color="#0068C9", dash="dash"),
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=df_fut.index,
            y=df_fut["High"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=df_fut.index,
            y=df_fut["Low"],
            mode="lines",
            fill="tonexty",
            name="Zone de Confiance 95%",
            line=dict(width=0),
            fillcolor="rgba(0,104,201,0.15)",
        )
    )

    fig_pred.update_layout(
        title=f"Projection {ticker} sur {forecast_days} jours",
        xaxis_title="Date",
        yaxis_title="Prix",
        legend_title="Série",
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    if model_choice == "ARIMA":
        st.info(
            "ℹ️ **ARIMA** analyse les cycles passés. Idéal pour les marchés volatils à court terme, essayez de l'appliquer au bitcoin par exemple."
        )
    elif model_choice == "Machine Learning (RF)":
        st.info(
            "ℹ️ **Random Forest** utilise l'IA pour repérer des motifs complexes (prix d'hier, avant-hier, moyennes)."
        )
    else:
        st.warning(
            "⚠️ **Régression Linéaire** : Trace juste une tendance droite. Attention, ne prédit pas les chutes ! Ce modèle est plus adapté pour les cours stables, essayez plutôt une action de père de famille, comme Air Liquide ;)"
        )

    ticker_clean = ticker.upper()
    if "BTC" in ticker_clean and model_choice == "ARIMA":
        st.success(
            "✅ Excellent choix ! Le Bitcoin est très volatil et cyclique, ARIMA est théoriquement le meilleur modèle pour capturer ces mouvements."
        )
    elif ("AI.PA" in ticker_clean or "AIR LIQUIDE" in ticker_clean) and model_choice == "Linear Regression":
        st.success(
            "✅ Bien vu ! Air Liquide est une action très stable avec une tendance long terme claire. La Régression Linéaire suffit largement et sera très propre."
        )

else:
    st.info("👈 Veuillez cliquer sur 'Charger Données & Scanner' dans la barre latérale pour commencer.")


# ============================================================
# =====================   QUANT B   ==========================
# ===== MULTI-ASSET PORTFOLIO — MARKOWITZ / MONTE-CARLO ======
# ============================================================

st.markdown("---")
st.header("📊 QuantB — Portfolio Multi-Assets (Markowitz & Monte-Carlo)")

with st.sidebar:
    st.subheader("⚙ Paramètres du portefeuille – QuantB")

    tickers = st.multiselect(
        "Sélectionne plusieurs actifs :",
        [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "META",
            "TSLA",  # USA
            "BTC-USD",  # Crypto
            "AI.PA",
            "TTE.PA",  # France
            "GC=F",  # Or
            "^GSPC",  # S&P500
        ],
        default=["AAPL", "MSFT", "BTC-USD", "AI.PA", "TTE.PA"],
    )

    start_b = st.date_input("Date de début", date(2020, 1, 1))
    end_b = st.date_input("Date de fin", date.today())

    rf_qb = st.number_input("Taux sans risque annuel", value=0.02, step=0.005)

    N_sim = st.slider(
        "Nombre de portefeuilles simulés (Monte Carlo)",
        min_value=500,
        max_value=10000,
        value=3000,
        step=500,
    )


# ---------- Fonctions ----------
def load_multi_assets(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)["Close"]
    return df.dropna()


def compute_portfolio_stats(weights, mean_returns, cov_matrix):
    """Retourne rendement, volatilité, sharpe."""
    weights = np.array(weights)
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe = (port_return - rf_qb) / port_vol
    return port_return, port_vol, sharpe


def efficient_frontier(mean_returns, cov_matrix, n_points=100):
    """Calcule la frontière de Markowitz."""
    results = {"return": [], "vol": [], "weights": []}

    for _ in range(n_points):
        w = np.random.random(len(mean_returns))
        w /= np.sum(w)

        ret, vol, _ = compute_portfolio_stats(w, mean_returns, cov_matrix)

        results["return"].append(ret)
        results["vol"].append(vol)
        results["weights"].append(w)

    return pd.DataFrame(results)


def plot_efficient_frontier(df_random, df_frontier, max_sharpe_point):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_random["vol"],
            y=df_random["ret"],
            mode="markers",
            name="Simulations Monte-Carlo",
            marker=dict(
                size=6,
                color=df_random["sharpe"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                opacity=0.7,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_frontier["vol"],
            y=df_frontier["return"],
            mode="lines",
            name="Frontière Efficiente",
            line=dict(color="red", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_point["vol"]],
            y=[max_sharpe_point["ret"]],
            mode="markers",
            name="Portefeuille Max Sharpe",
            marker=dict(color="gold", size=12, line=dict(color="black", width=1.5)),
        )
    )

    fig.update_layout(
        xaxis_title="Volatilité (σ)",
        yaxis_title="Rendement Annuel (%)",
        title="Frontière de Markowitz & Portefeuilles simulés",
        legend_title="Légende",
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------- Exécution ----------
if len(tickers) >= 2:
    st.subheader("📥 Téléchargement des données")
    df_prices = load_multi_assets(tickers, start_b, end_b)
    returns = df_prices.pct_change().dropna()
    st.success("Données chargées !")

    st.subheader("📈 Valeur cumulée des actifs (Base 100)")
    norm = df_prices / df_prices.iloc[0] * 100
    fig_norm = px.line(
        norm,
        labels={"index": "Date", "value": "Valeur (base 100)", "variable": "Actif"},
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    st.subheader("🔗 Matrice de corrélation")
    st.dataframe(returns.corr())

    st.subheader("🎯 Simulation Markowitz — Monte-Carlo")
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    sim_results = {"ret": [], "vol": [], "sharpe": [], "weights": []}

    for _ in range(N_sim):
        w = np.random.random(len(tickers))
        w /= w.sum()

        ret, vol, sharpe = compute_portfolio_stats(w, mean_returns, cov_matrix)

        sim_results["ret"].append(ret)
        sim_results["vol"].append(vol)
        sim_results["sharpe"].append(sharpe)
        sim_results["weights"].append(w)

    df_random = pd.DataFrame(sim_results)

    idx_max = df_random["sharpe"].idxmax()
    best_weights = df_random.loc[idx_max, "weights"]

    st.success("Portefeuille Max Sharpe trouvé ✔")

    df_w = pd.DataFrame(
        {"Actifs": tickers, "Poids (%)": [round(w * 100, 2) for w in best_weights]}
    )
    st.dataframe(df_w)

    st.subheader("📈 Frontière de Markowitz ")

    df_frontier = efficient_frontier(mean_returns, cov_matrix)

    max_point = {
        "ret": df_random.loc[idx_max, "ret"],
        "vol": df_random.loc[idx_max, "vol"],
    }

    plot_efficient_frontier(df_random, df_frontier, max_point)
