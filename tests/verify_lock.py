import pandas as pd
import re
import sys
import numpy as np
import json
import os

# Windows Console Encoding Fix (Emoji ප්‍රශ්න මඟහරවා ගැනීමට)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

from app.data_loader import get_data

# --- Helper Functions ---
def extract_price(price_str):
    try:
        if isinstance(price_str, (int, float)): return float(price_str)
        match = re.search(r"[\d,]+(\.\d+)?", str(price_str))
        if match:
            return float(match.group(0).replace(',', ''))
    except:
        return None
    return None

def check_trade_status_debug(scenario, historical_df):
    print(f"\n[SEARCH] STARTING CHECK FOR: {scenario.get('trade_type')} | Entry: {scenario.get('entry_zone') or scenario.get('entry')} | TP: {scenario.get('targets', ['N/A'])[0]}")
    print("-" * 70)
    
    trade_type = scenario.get("trade_type", "").upper()
    targets = scenario.get("targets", [])
    if not targets and "target" in scenario: targets = [scenario.get("target")]
    
    tp_levels = sorted([extract_price(t) for t in targets if extract_price(t)])
    entry = extract_price(scenario.get("entry_zone") or scenario.get("entry"))
    stop_loss = extract_price(scenario.get("invalidation"))
    
    highest_tp_hit_idx = -1
    
    # --- CANDLE BY CANDLE CHECK ---
    for row in historical_df.itertuples():
        date_str = row.Index.strftime('%Y-%m-%d %H:%M')
        curr_low = row.Low
        curr_high = row.High
        
        # Log for every candle
        status_log = f"   [TIME] {date_str} | Low:{curr_low:.2f} High:{curr_high:.2f}"

        if "LONG" in trade_type:
            if curr_low <= stop_loss:
                print(f"{status_log} -> [STOP] 🛑 SL HIT! ({curr_low} <= {stop_loss})")
                return "LOST"
            
            tp_hit = False
            for i, tp in enumerate(tp_levels):
                if curr_high >= tp:
                    if i > highest_tp_hit_idx:
                        highest_tp_hit_idx = i
                        print(f"{status_log} -> [TARGET] 🎯 TP{i+1} HIT! ({curr_high} >= {tp})")
                        tp_hit = True
            
            if not tp_hit:
                 print(f"{status_log} -> [SAFE] Checking...")

        elif "SHORT" in trade_type:
            if curr_high >= stop_loss:
                print(f"{status_log} -> [STOP] 🛑 SL HIT! ({curr_high} >= {stop_loss})")
                return "LOST"
            
            tp_hit = False
            for i, tp in enumerate(tp_levels):
                if curr_low <= tp:
                    if i > highest_tp_hit_idx:
                        highest_tp_hit_idx = i
                        print(f"{status_log} -> [TARGET] 🎯 TP{i+1} HIT! ({curr_low} <= {tp})")
                        tp_hit = True
                        
            if not tp_hit:
                 print(f"{status_log} -> [SAFE] Checking...")

    print("-" * 70)
    if highest_tp_hit_idx == len(tp_levels) - 1:
        print("[WON] ✅ RESULT: WON (All Targets Hit)")
        return "WON"
    elif highest_tp_hit_idx >= 0:
        print(f"[PROFIT] ⚠️ RESULT: RUNNING_PROFIT (Hit TP {highest_tp_hit_idx+1})")
        return "RUNNING_PROFIT"
    else:
        print("[WAIT] ⏳ RESULT: STILL RUNNING")
        return "RUNNING"

# --- RUN SIMULATION ---
def run_test():
    symbol = "BTCUSDT"
    print(f"[>] Fetching Data for {symbol}...")
    
    df = get_data(symbol, interval="15m", period="5d")
    
    if df.empty:
        print("[ERROR] No data fetched.")
        return

    # --- ATTEMPT TO LOAD REAL SCENARIO FROM SOFTWARE ---
    real_scenario = None
    if os.path.exists("website_data.json"):
        try:
            with open("website_data.json", "r") as f:
                data = json.load(f)
                if "ai_prediction" in data and "trade_scenarios" in data["ai_prediction"]:
                    scenarios = data["ai_prediction"]["trade_scenarios"]
                    if len(scenarios) > 0:
                        real_scenario = scenarios[0]
                        print("\n✅ FOUND REAL SCENARIO in website_data.json")
                        print(f"   Name: {real_scenario.get('name')}")
                        print(f"   Type: {real_scenario.get('trade_type')}")
        except Exception as e:
            print(f"[WARN] Error reading website_data.json: {e}")

    # --- SELECT SCENARIO TO TEST ---
    if real_scenario:
        print("   [INFO] Testing the ACTUAL setup from the software...")
        scenario_to_test = real_scenario
        # සැබෑ Scenario එකක් සඳහා අපි පසුගිය දින 2ක (Candles 200ක්) පරීක්ෂා කරමු.
        lock_time = df.index[-200] 
    else:
        print("\n⚠️ No website_data.json found. Creating a SIMULATED TEST scenario...")
        # 1. Lock Time (50 candles ago)
        lock_time = df.index[-50]
        df_slice = df[df.index >= lock_time]
        
        # 2. Guaranteed WIN Scenario Logic (Simulation)
        min_price = df_slice['Low'].min()
        max_price = df_slice['High'].max()
        
        print(f"   [SIMULATION] Creating a Fake Scenario to verify 'WON' logic...")
        scenario_to_test = {
            "trade_type": "LONG",
            "entry": str(min_price * 1.001), 
            "targets": [str(max_price * 0.999)],
            "invalidation": str(min_price * 0.99)
        }

    print(f"[START TIME] Checking data since: {lock_time}")
    df_since_lock = df[df.index >= lock_time]
    
    # 3. Check
    check_trade_status_debug(scenario_to_test, df_since_lock)

if __name__ == "__main__":
    run_test()