import yfinance as yf
import pandas as pd

def load_multi_assets(tickers, start_date, end_date):
    """
    Télécharge les prix de clôture pour plusieurs actifs simultanément.

    Paramètres
    ----------
    tickers : list
        Liste des tickers, ex : ['AAPL', 'MSFT', '^GSPC']
    start_date : str
        Date de début au format 'YYYY-MM-DD'
    end_date : str
        Date de fin au format 'YYYY-MM-DD'

    Retourne
    -------
    DataFrame
        Un tableau où :
            - les colonnes = tickers des actifs
            - les lignes   = dates
            - les valeurs  = prix de clôture
    """

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date
        )['Close']   # On garde uniquement les prix de clôture

        # Nettoyage (supprime les colonnes vides si un ticker manque)
        data = data.dropna(axis=1, how='all')

        return data

    except Exception as e:
        print(f"Erreur lors du téléchargement multi-actifs : {e}")
        return None
