import yfinance as yf
import numpy as np
import datetime
import os
import pandas as pd

# --- CONFIGURATION ---
ASSET = "BTC-USD"
REPORT_DIR = "/home/ubuntu/Projet-Linux-Pierre-Antoine-et-Rhea/reports"

# Créer le dossier s'il n'existe pas
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_report():
    print(f"Téléchargement des données pour {ASSET}...")
    # On télécharge les données
    df = yf.download(ASSET, period="1mo", progress=False)
    
    # --- NETTOYAGE DES DONNEES (Le Fix) ---
    # Si yfinance renvoie un MultiIndex (colonnes complexes), on aplatit tout
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs('Close', axis=1, level=0, drop_level=False)
        df.columns = ['Close']
    else:
        df = df[['Close']]

    # 2. Calculs simples (Extraction sécurisée des valeurs)
    # On utilise .item() pour être sûr de récupérer un chiffre unique (float)
    try:
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        
        # Conversion forcée en float si c'est encore une Series
        if hasattr(last_price, 'item'): last_price = last_price.item()
        if hasattr(prev_price, 'item'): prev_price = prev_price.item()
            
    except Exception as e:
        print(f"Erreur extraction prix: {e}")
        last_price = 0.0
        prev_price = 0.0

    daily_return = (last_price / prev_price) - 1
    
    # Volatilité
    returns = df['Close'].pct_change().dropna()
    std_dev = returns.std()
    
    # Si std_dev est une Series, on prend la valeur
    if hasattr(std_dev, 'item'):
        std_dev = std_dev.item()
    elif hasattr(std_dev, 'iloc'):
        std_dev = std_dev.iloc[0]
        
    volatility = float(std_dev) * np.sqrt(252)

    # 3. Contenu du rapport
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""
    ========================================
    RAPPORT AUTOMATIQUE - {today}
    ========================================
    
    Actif Analysé : {ASSET}
    
    PRIX ACTUEL   : {last_price:.2f} $
    VARIATION 24H : {daily_return:.2%}
    VOLATILITÉ    : {volatility:.2%} (Annuelle)
    
    ----------------------------------------
    Statut système : OK
    Généré par le serveur AWS.
    ========================================
    """
    
    # 4. Sauvegarde
    filename = f"{REPORT_DIR}/report_{datetime.date.today()}.txt"
    with open(filename, "w") as f:
        f.write(content)
    
    print(f"Succès ! Rapport sauvegardé : {filename}")

if __name__ == "__main__":
    generate_report()