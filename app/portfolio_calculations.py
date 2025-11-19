import pandas as pd
import numpy as np

def compute_daily_returns(price_df):
    """
    Calcule les rendements journaliers pour plusieurs actifs.

    Paramètres
    ----------
    price_df : DataFrame
        Colonnes = tickers, lignes = dates

    Retourne
    -------
    DataFrame
        Rendements journaliers (pct_change)
    """
    returns = price_df.pct_change().dropna()
    return returns


def compute_covariance_matrix(returns_df):
    """
    Calcule la matrice de covariance des rendements.

    Paramètres
    ----------
    returns_df : DataFrame

    Retourne
    -------
    DataFrame
        Matrice de covariance
    """
    return returns_df.cov()


def compute_portfolio_performance(returns_df, weights):
    """
    Calcule la performance d’un portefeuille multi-actifs.

    Paramètres
    ----------
    returns_df : DataFrame
        Rendements journaliers des actifs
    weights : array-like
        Poids du portefeuille, ex : [0.3, 0.3, 0.4]

    Retourne
    -------
    Series
        Valeur cumulée du portefeuille (base 100)
    """
    weights = np.array(weights)
    portfolio_returns = (returns_df * weights).sum(axis=1)

    cumulative_value = (1 + portfolio_returns).cumprod() * 100
    cumulative_value.name = "Portfolio_Value"

    return cumulative_value


def compute_asset_returns(prices_df):
    """
    Calcule les rendements journaliers de chaque actif.
    
    prices_df : DataFrame contenant les prix (colonnes = tickers)
    Retourne : DataFrame des rendements journaliers
    """
    returns = prices_df.pct_change().fillna(0)
    return returns


def compute_portfolio_returns(returns_df, weights):
    """
    Calcule les rendements journaliers du portefeuille à partir :
    - returns_df : DataFrame des returns des actifs
    - weights : liste ou array des poids (ex : [0.3, 0.5, 0.2])
    """
    weights = np.array(weights)
    daily_portfolio_returns = (returns_df * weights).sum(axis=1)
    return daily_portfolio_returns


def compute_portfolio_performance(returns_df, weights):
    """
    Calcule la valeur cumulée du portefeuille (base 100).
    """
    portfolio_returns = compute_portfolio_returns(returns_df, weights)
    cumulative_value = (1 + portfolio_returns).cumprod() * 100
    cumulative_value.name = "Portfolio_Value"

    return cumulative_value
