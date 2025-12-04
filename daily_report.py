import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os

# --- CONFIGURATION ---
TICKERS = ["AAPL", "MSFT", "BTC-USD", "AI.PA", "TTE.PA"]
REPORT_DIR = "/home/ubuntu/Projet-Linux-Pierre-Antoine-et-Rhea/reports"
HISTORY_FILE = f"{REPORT_DIR}/portfolio_history.csv"

# Initial capital (only used the very first time script runs)
INITIAL_CAPITAL = 10000.0 
TARGET_WEIGHT = 1.0 / len(TICKERS) # 20% each

os.makedirs(REPORT_DIR, exist_ok=True)

def get_current_prices():
    """Download latest prices for all tickers"""
    print("Downloading latest prices...")
    # We ask for 1 day, 1 minute interval to get the very latest price
    df = yf.download(TICKERS, period="1d", interval="1m", progress=False)
    
    # Flatten MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('Close', axis=1, level=0, drop_level=True)
        except:
            df = df['Close'] # Fallback
            
    # Get the last available row (latest price)
    last_prices = df.iloc[-1]
    return last_prices

def load_portfolio_state():
    """Load previous state from CSV or initialize if new"""
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        last_row = history.iloc[-1]
        # We need to know how many shares we had yesterday to calculate today's value before rebalancing
        shares = {t: last_row[f"{t}_shares"] for t in TICKERS}
        return shares, history
    else:
        return None, pd.DataFrame()

def rebalance_portfolio():
    try:
        # 1. Get real-time prices
        prices = get_current_prices()
        
        # 2. Load previous state
        prev_shares, history = load_portfolio_state()
        
        current_value = 0.0
        
        # --- CASE A: FIRST RUN (Initialization) ---
        if prev_shares is None:
            print(f"First run! Starting with ${INITIAL_CAPITAL}")
            current_value = INITIAL_CAPITAL
            # Calculate shares to buy to have 20% of capital in each
            new_shares = {}
            for t in TICKERS:
                allocation = current_value * TARGET_WEIGHT
                new_shares[t] = allocation / prices[t]
                
        # --- CASE B: DAILY REBALANCE ---
        else:
            # Calculate current portfolio value based on yesterday's shares * today's price
            for t in TICKERS:
                val = prev_shares[t] * prices[t]
                current_value += val
            
            print(f"Current Portfolio Value: ${current_value:.2f}")
            
            # Rebalance: We want 20% of this NEW total value in each asset
            new_shares = {}
            for t in TICKERS:
                target_allocation = current_value * TARGET_WEIGHT
                new_shares[t] = target_allocation / prices[t]

        # 3. Save new state to history CSV
        today_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = {"Date": today_str, "Total_Value": current_value}
        for t in TICKERS:
            new_row[f"{t}_shares"] = new_shares[t]
            new_row[f"{t}_price"] = prices[t]
            new_row[f"{t}_value"] = new_shares[t] * prices[t]
            
        # Append to dataframe and save
        new_df = pd.DataFrame([new_row])
        if not history.empty:
            history = pd.concat([history, new_df], ignore_index=True)
        else:
            history = new_df
            
        history.to_csv(HISTORY_FILE, index=False)
        
        # 4. Generate the readable Text Report
        generate_text_report(today_str, current_value, new_shares, prices)
        
    except Exception as e:
        print(f"Error rebalancing: {e}")
        with open(f"{REPORT_DIR}/error_log.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} - Error: {e}\n")

def generate_text_report(date_str, value, shares, prices):
    content = f"""
=========================================================
   DAILY REBALANCING REPORT - EQUAL WEIGHT (20%)
   Date : {date_str}
=========================================================

PORTFOLIO STATUS
----------------
Total Value : {value:.2f} $

REBALANCING ACTIONS EXECUTED:
Target per asset: {value * TARGET_WEIGHT:.2f} $ (20%)

"""
    for t in TICKERS:
        content += f"  - {t:<8} : {shares[t]:.4f} shares  (@ {prices[t]:.2f} $)\n"

    content += f"""
=========================================================
Statut : Success
Mode   : Auto-Rebalance to 20%
=========================================================
"""
    # Save the daily text file
    filename = f"{REPORT_DIR}/report_{datetime.date.today()}.txt"
    with open(filename, "w") as f:
        f.write(content)
    print(f"Report generated: {filename}")

if __name__ == "__main__":
    rebalance_portfolio()