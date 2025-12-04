import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

import plotly.express as px
import plotly.graph_objects as go
import itertools

# ======================================================================
#                               CONFIG
# ======================================================================

st.set_page_config(layout="wide", page_title="Projet Finance: Quant A & B")

# ======================================================================
#                         CLASSE QUANT A
# ======================================================================

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
            signals = np.where(self.data["Close"] < lower_band, 1.0, 0.0)

        # Backtest
        signals = pd.Series(signals, index=self.data.index)
        strat_returns = self.daily_returns.shift(-1) * signals.shift(1).fillna(0)
        strat_curve = (1 + strat_returns).cumprod() * self.initial_investment
        strat_curve = strat_curve.ffill()  # Correction du bug NaN à la fin

        return strat_curve, strat_returns

    def find_best_params(self):
        """Teste plein de combinaisons et stocke les gagnantes."""

        # 1. Optimisation Momentum
        best_sharpe = -999
        best_p = {"window": 50}
        for w in range(10, 150, 10):
            _, rets = self.run_strategy("Momentum", window=w)
            m = self.compute_metrics(rets)
            if m["Raw_Sharpe"] > best_sharpe:
                best_sharpe = m["Raw_Sharpe"]
                best_p = {"window": w}
        self.best_params["Momentum"] = best_p

        # 2. Optimisation Cross MMS
        best_sharpe = -999
        best_p = {"short_w": 20, "long_w": 50}
        for s, l in itertools.product(range(10, 60, 10), range(50, 200, 20)):
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
        for w, std in itertools.product(range(10, 60, 10), [1.5, 2.0, 2.5]):
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
        1. Linear Regression
        2. ARIMA
        3. Random Forest
        """
        df = self.data.copy()
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

        # --- MODÈLE 1 : RÉGRESSION LINÉAIRE (ANCRAGE) ---
        if model_type == "Linear Regression":
            df = df.reset_index()
            df["Date_Ordinal"] = df["Date"].map(pd.Timestamp.toordinal)
            X = df[["Date_Ordinal"]].values
            y = df["Close"].values

            model = LinearRegression().fit(X, y)
            future_ordinals = [[d.toordinal()] for d in future_dates]
            preds = model.predict(future_ordinals)

            last_day_ordinal = [[X[-1][0]]]
            theoretical_price_today = model.predict(last_day_ordinal)[0]
            actual_price_today = y[-1]
            offset = actual_price_today - theoretical_price_today
            preds = preds + offset

            recent_returns = df["Close"].pct_change().tail(90)
            sigma_pct = recent_returns.std()
            std_dev = sigma_pct * df["Close"].iloc[-1]

            return future_dates, preds, std_dev

        # --- MODÈLE 2 : ARIMA ---
        elif model_type == "ARIMA":
            history = df["Close"].values
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit()
            preds = model_fit.forecast(steps=days_ahead)

            residuals = model_fit.resid
            std_dev = np.std(residuals[1:])
            return future_dates, preds, std_dev

        # --- MODÈLE 3 : RANDOM FOREST ---
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


# ======================================================================
#                        FONCTIONS QUANT B
# ======================================================================

def load_multi_assets(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    return df.dropna()

def compute_portfolio_stats(weights, mean_returns, cov_matrix, rf_qb):
    """Retourne rendement, volatilité, sharpe."""
    weights = np.array(weights)
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe = (port_return - rf_qb) / port_vol
    return port_return, port_vol, sharpe


# ======================================================================
#                            NAVIGATION
# ======================================================================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    ["Quant A - Analyse Actif Unique", "Quant B - Gestion de Portefeuille"]
)
st.sidebar.markdown("---")

# Init état Quant A
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None

# ======================================================================
#                          MODULE QUANT A
# ======================================================================
if page == "Quant A - Analyse Actif Unique":

    st.title("🧠 Smart Quant Lab: Analyse & Optimisation")

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

        if st.session_state.analyzer:
            an = st.session_state.analyzer

            st.header("2. Contrôle Manuel")
            strat_choice = st.selectbox(
                "Stratégie Active",
                ["Momentum", "Cross MMS", "Mean Reversion (BB)", "TOUT COMPARER"],
            )

            manual_params = {}

            if strat_choice == "Momentum" or strat_choice == "TOUT COMPARER":
                st.markdown("### 🚀 Momentum")
                rec = an.best_params["Momentum"]["window"]
                st.caption(f"💡 Suggestion IA : {rec}")
                manual_params["mom_window"] = st.slider(
                    "Fenêtre Momentum", 10, 200, 50, key="mom_slider"
                )

            if strat_choice == "Cross MMS" or strat_choice == "TOUT COMPARER":
                st.markdown("### ❌ Cross MMS")
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
                st.markdown("### 📉 Bollinger")
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
                "Choisir le Modèle",
                ["Linear Regression", "ARIMA", "Machine Learning (RF)"],
            )
            forecast_days = st.slider("Jours à prédire", 7, 90, 30)

    if st.session_state.analyzer:
        an = st.session_state.analyzer
        bh_curve = (1 + an.daily_returns).cumprod() * an.initial_investment

        # ===== CAS 1 : TOUT COMPARER =====
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
                labels={
                    "index": "Date",
                    "value": "Valeur du portefeuille",
                    "variable": "Stratégie",
                },
            )
            st.plotly_chart(fig_all, use_container_width=True)

            st.write("### Valeurs Finales du Portefeuille")
            res_finaux = df_all.iloc[-1].sort_values(ascending=False)
            st.dataframe(res_finaux.map("{:.2f} $".format))

        # ===== CAS 2 : MODE SOLO =====
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

            # Recalcul des signaux pour affichage graphique complet
            signals = pd.Series(0, index=an.data.index)
            indicator1 = None
            indicator2 = None
            label_ind1 = ""
            label_ind2 = ""

            if strat_choice == "Momentum":
                win = manual_params["mom_window"]
                mms = an.data["Close"].rolling(win).mean()
                signals = np.where(an.data["Close"] > mms, 1.0, 0.0)
                indicator1 = mms
                label_ind1 = f"Moyenne Mobile ({win}j)"

            elif strat_choice == "Cross MMS":
                s, l = manual_params["cross_short"], manual_params["cross_long"]
                short = an.data["Close"].rolling(s).mean()
                long = an.data["Close"].rolling(l).mean()
                signals = np.where(short > long, 1.0, 0.0)
                indicator1 = short
                label_ind1 = f"MMS Rapide ({s}j)"
                indicator2 = long
                label_ind2 = f"MMS Lente ({l}j)"

            elif strat_choice == "Mean Reversion (BB)":
                w, std_bb = manual_params["bb_window"], manual_params["bb_std"]
                mid = an.data["Close"].rolling(w).mean()
                sigma = an.data["Close"].rolling(w).std()
                lower = mid - (sigma * std_bb)
                signals = np.where(an.data["Close"] < lower, 1.0, 0.0)
                indicator1 = lower
                label_ind1 = "Bande Inférieure"

            signals = pd.Series(signals, index=an.data.index)
            positions = signals.diff().fillna(0)
            nb_trades = int(positions.abs().sum())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Stratégie", strat_choice)
            c2.metric(
                "Sharpe Ratio",
                met_strat["Sharpe"],
                delta=f"{met_strat['Sharpe'] - met_bh['Sharpe']:.2f} vs B&H",
            )
            c3.metric("Max Drawdown", met_strat["Max Drawdown"])
            c4.metric("Gain Total", met_strat["Total Perf"])
            c5.metric("Nbr. Ordres", nb_trades, help="Nombre total d'achats et de ventes")

            st.subheader(f"📈 Analyse détaillée : {strat_choice}")

            fig_price = go.Figure()
            fig_price.add_trace(
                go.Scatter(
                    x=an.data.index,
                    y=an.data["Close"],
                    mode="lines",
                    name="Prix Actif",
                    line=dict(color="black"),
                )
            )

            if indicator1 is not None:
                fig_price.add_trace(
                    go.Scatter(
                        x=an.data.index,
                        y=indicator1,
                        mode="lines",
                        name=label_ind1,
                        line=dict(dash="dash"),
                    )
                )
            if indicator2 is not None:
                fig_price.add_trace(
                    go.Scatter(
                        x=an.data.index,
                        y=indicator2,
                        mode="lines",
                        name=label_ind2,
                        line=dict(dash="dot"),
                    )
                )

            buys = an.data.loc[positions == 1.0]
            sells = an.data.loc[positions == -1.0]

            fig_price.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["Close"],
                    mode="markers",
                    name="Achat",
                    marker=dict(color="green", symbol="triangle-up", size=10),
                )
            )
            fig_price.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells["Close"],
                    mode="markers",
                    name="Vente",
                    marker=dict(color="red", symbol="triangle-down", size=10),
                )
            )

            fig_price.update_layout(
                title=f"Signaux de Trading : {strat_choice} ({nb_trades} ordres)",
                xaxis_title="Date",
                yaxis_title="Prix",
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # Courbe B&H vs Stratégie
            st.subheader("📊 Portefeuille : Stratégie vs Buy & Hold")
            df_chart = pd.DataFrame(
                {"Buy & Hold": bh_curve, f"Stratégie ({strat_choice})": strat_curve}
            )
            fig_curve = px.line(
                df_chart,
                labels={
                    "index": "Date",
                    "value": "Valeur du portefeuille",
                    "variable": "Courbe",
                },
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        # ===== BATTLE ROYALE =====
        st.markdown("---")
        st.subheader("⚔️ Battle Royale : Comparaison des modèles optimisés")
        st.caption(
            "Voici ce que ça donnerait avec les MEILLEURS paramètres trouvés par l'IA."
        )

        curve_mom, ret_mom = an.run_strategy("Momentum", **an.best_params["Momentum"])
        curve_cross, ret_cross = an.run_strategy(
            "Cross MMS", **an.best_params["Cross MMS"]
        )
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
            labels={
                "index": "Date",
                "value": "Valeur du portefeuille",
                "variable": "Stratégie",
            },
        )
        st.plotly_chart(fig_battle, use_container_width=True)

        met_bh = an.compute_metrics(an.daily_returns)
        met_mom = an.compute_metrics(ret_mom)
        met_cross = an.compute_metrics(ret_cross)
        met_bb = an.compute_metrics(ret_bb)

        leaderboard_data = [
            {
                "Stratégie": "Buy & Hold",
                "Sharpe Ratio": met_bh["Sharpe"],
                "Max Drawdown": met_bh["Max Drawdown"],
                "Perf Totale": met_bh["Total Perf"],
                "Capital Final": f"{bh_curve.iloc[-1]:.2f} $",
            },
            {
                "Stratégie": "Momentum (IA)",
                "Sharpe Ratio": met_mom["Sharpe"],
                "Max Drawdown": met_mom["Max Drawdown"],
                "Perf Totale": met_mom["Total Perf"],
                "Capital Final": f"{curve_mom.iloc[-1]:.2f} $",
            },
            {
                "Stratégie": "Cross MMS (IA)",
                "Sharpe Ratio": met_cross["Sharpe"],
                "Max Drawdown": met_cross["Max Drawdown"],
                "Perf Totale": met_cross["Total Perf"],
                "Capital Final": f"{curve_cross.iloc[-1]:.2f} $",
            },
            {
                "Stratégie": "Bollinger (IA)",
                "Sharpe Ratio": met_bb["Sharpe"],
                "Max Drawdown": met_bb["Max Drawdown"],
                "Perf Totale": met_bb["Total Perf"],
                "Capital Final": f"{curve_bb.iloc[-1]:.2f} $",
            },
        ]
        df_leaderboard = (
            pd.DataFrame(leaderboard_data)
            .set_index("Stratégie")
            .sort_values(by="Sharpe Ratio", ascending=False)
        )
        st.dataframe(df_leaderboard, use_container_width=True)

        # ===== PREDICTION =====
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


# ======================================================================
#                          MODULE QUANT B
# ======================================================================
elif page == "Quant B - Gestion de Portefeuille":

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

        st.markdown("---")
        st.subheader("⚖️ Poids du portefeuille utilisateur")

        # Poids en % puis rebalancés automatiquement
        raw_weights_pct = []
        for t in tickers:
            w = st.slider(
                f"Poids cible pour {t} (%)",
                min_value=0.0,
                max_value=100.0,
                value=round(100.0 / len(tickers), 1),
                step=1.0,
            )
            raw_weights_pct.append(w)

        raw_weights = np.array(raw_weights_pct)
        if raw_weights.sum() == 0:
            raw_weights = np.ones_like(raw_weights)

        user_weights = raw_weights / raw_weights.sum()

    # ---------- Exécution Quant B ----------
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

            ret, vol, sharpe = compute_portfolio_stats(w, mean_returns, cov_matrix, rf_qb)

            sim_results["ret"].append(ret)
            sim_results["vol"].append(vol)
            sim_results["sharpe"].append(sharpe)
            sim_results["weights"].append(w)

        df_random = pd.DataFrame(sim_results)

        idx_max = df_random["sharpe"].idxmax()
        best_weights = df_random.loc[idx_max, "weights"]
        best_ret = df_random.loc[idx_max, "ret"]
        best_vol = df_random.loc[idx_max, "vol"]
        best_sharpe = df_random.loc[idx_max, "sharpe"]

        user_ret, user_vol, user_sharpe = compute_portfolio_stats(
            user_weights, mean_returns, cov_matrix, rf_qb
        )

        st.subheader("📋 Poids recommandés vs Poids utilisateur (normalisés)")
        df_w = pd.DataFrame(
            {
                "Actif": tickers,
                "Poids Max Sharpe (%)": [round(w * 100, 2) for w in best_weights],
                "Poids Utilisateur (%)": [round(w * 100, 2) for w in user_weights],
            }
        )
        st.dataframe(df_w, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Max Sharpe", f"{best_sharpe:.2f}")
        col2.metric("Sharpe Portefeuille Utilisateur", f"{user_sharpe:.2f}")
        col3.metric("Différence Sharpe", f"{user_sharpe - best_sharpe:.2f}")

        st.subheader("📈 Frontière de Markowitz ")

        # Frontière efficiente propre à partir des points simulés
        df_frontier = df_random.copy()
        df_frontier = df_frontier.sort_values(by="vol")
        df_frontier["ret_frontier"] = df_frontier["ret"].cummax()
        df_frontier = df_frontier.drop_duplicates(subset=["ret_frontier"])

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
                    opacity=0.6,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_frontier["vol"],
                y=df_frontier["ret_frontier"],
                mode="lines",
                name="Frontière Efficiente",
                line=dict(color="red", width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[best_vol],
                y=[best_ret],
                mode="markers",
                name="Portefeuille Max Sharpe",
                marker=dict(
                    color="gold",
                    size=14,
                    line=dict(color="black", width=1.5),
                    symbol="circle-open",
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[user_vol],
                y=[user_ret],
                mode="markers",
                name="Portefeuille Utilisateur",
                marker=dict(
                    color="magenta",
                    size=12,
                    line=dict(color="black", width=1.5),
                    symbol="diamond",
                ),
            )
        )

        fig.update_layout(
            title="Frontière de Markowitz & Portefeuilles simulés",
            xaxis_title="Volatilité (σ)",
            yaxis_title="Rendement Annuel",
            legend_title="Légende",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Veuillez sélectionner au moins 2 actifs dans la barre latérale.")
HISTORY_FILE = "reports/portfolio_history.csv"
        
st.markdown("---")
st.subheader("📊 Évolution réelle du portefeuille (Bot Rebalance 20%)")
        
if os.path.exists(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
            
            # Plot Total Value over time
    fig_hist = px.line(
        hist, 
        x="Date", 
        y="Total_Value", 
        title="Valeur Totale du Portefeuille (Rebalancement Quotidien)",
        markers=True
    )
    st.plotly_chart(fig_hist, use_container_width=True)
            
            # Show the last rebalancing details
    st.write("### 🧾 Dernier Rebalancement Exécuté")
    last_row = hist.iloc[-1]
    st.write(f"**Date :** {last_row['Date']}")
    st.write(f"**Valeur Totale :** {last_row['Total_Value']:.2f} $")
            
            # Create a nice table for current holdings
    holdings = []
    tickers = ["AAPL", "MSFT", "BTC-USD", "AI.PA", "TTE.PA"]
    for t in tickers:
        if f"{t}_shares" in last_row:
            holdings.append({
                "Actif": t,
                "Parts Détenues": f"{last_row[f'{t}_shares']:.4f}",
                "Prix": f"{last_row[f'{t}_price']:.2f} $",
                "Valeur Position": f"{last_row[f'{t}_value']:.2f} $"
            })
            
    st.dataframe(pd.DataFrame(holdings))
            
else:
    st.info("⚠️ Aucun historique trouvé. Le script automatique (Cron) n'a pas encore tourné aujourd'hui.")
    st.caption("Le premier rapport sera généré à 20h00, ou lancez `python3 daily_report.py` sur le serveur pour initialiser.")

# ======================================================================
#                      ESPACE ADMIN / RAPPORTS
# ======================================================================
with st.sidebar:
    st.markdown("---")
    st.header("📂 Espace Admin")

    report_folder = "reports"

    if os.path.exists(report_folder) and os.listdir(report_folder):
        files = [f for f in os.listdir(report_folder) if f.endswith(".txt")]
        files.sort(reverse=True)

        if files:
            selected_file = st.selectbox("Choisir un rapport :", files)

            if st.button("Lire le rapport"):
                with open(os.path.join(report_folder, selected_file), "r") as f:
                    st.session_state["log_content"] = f.read()
    else:
        st.caption("Aucun rapport disponible pour le moment.")

if "log_content" in st.session_state:
    st.markdown("---")
    st.subheader("📄 Visualiseur de Rapport")
    st.text_area("Contenu du fichier :", st.session_state["log_content"], height=300)

    if st.button("Fermer le rapport"):
        del st.session_state["log_content"]
        st.rerun()
