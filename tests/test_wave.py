# test_wave.py - Major Structure (Impulse + ABC Only) - No Triangles
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from app.models.WaveAnalyzer import WaveAnalyzer
from app.models.WaveOptions import WaveOptionsGenerator5
from app.models.WaveRules import Impulse, Correction
from app.models.WavePattern import WavePattern
from app.data_loader import get_data

# --- Helper Classes ---

class WaveOptionsGenerator3:
    """
    ABC (Correction) සඳහා අවශ්‍ය Combinations හදන Class එක.
    """
    def __init__(self, up_to=1):
        self.options_sorted = []
        for i in range(up_to + 1):
            for j in range(up_to + 1):
                for k in range(up_to + 1):
                     self.options_sorted.append(type('obj', (object,), {'values': [i, j, k]})())
        self.options_sorted.sort(key=lambda x: sum(x.values))

# --- Visualization ---

def visualize_phases(df, phase1_waves, phase2_waves, filename="btc_major_structure_no_tri.html"):
    print(f"🎨 Saving Chart to {filename}...")
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
    )])

    # Colors
    c_impulse = '#00CC96'   # Green (Impulse)
    c_correct = '#FFA15A'   # Orange (Correction)
    c_phase2 = '#FF6692'    # Pink/Red (Current Drop/Correction)

    # --- Plot Phase 1 (Bull Run) ---
    for p_type, waves in phase1_waves:
        # Phase 1 එකේදී Impulse නම් කොළ පාට, Correction නම් තැඹිලි පාට
        color = c_impulse if p_type == 'Impulse' else c_correct
        width = 4 if p_type == 'Impulse' else 2
        labels = ['1','2','3','4','5'] if p_type == 'Impulse' else ['A','B','C']
        
        wx, wy, lx, ly = [], [], [], []
        for i, w in enumerate(waves):
            is_up = 'Up' in w.__class__.__name__
            if i==0:
                wx.append(w.date_start)
                wy.append(w.low if is_up else w.high)
            
            end_p = w.high if is_up else w.low
            wx.append(w.date_end)
            wy.append(end_p)
            if i < len(labels):
                lx.append(w.date_end)
                ly.append(end_p)

        fig.add_trace(go.Scatter(x=wx, y=wy, mode='lines', line=dict(color=color, width=width), name=f"Bull: {p_type}"))
        fig.add_trace(go.Scatter(x=lx, y=ly, mode='text', text=labels, textposition="top center", textfont=dict(size=12, color=color), showlegend=False))

    # --- Plot Phase 2 (High to Now) ---
    for p_type, waves in phase2_waves:
        # Phase 2 එක වෙනම පාටකින් (Pink)
        color = c_phase2 
        width = 3
        labels = ['1','2','3','4','5'] if p_type == 'Impulse' else ['A','B','C']
        
        wx, wy, lx, ly = [], [], [], []
        for i, w in enumerate(waves):
            is_up = 'Up' in w.__class__.__name__
            if i==0:
                wx.append(w.date_start)
                wy.append(w.low if is_up else w.high)
            
            end_p = w.high if is_up else w.low
            wx.append(w.date_end)
            wy.append(end_p)
            if i < len(labels):
                lx.append(w.date_end)
                ly.append(end_p)

        fig.add_trace(go.Scatter(x=wx, y=wy, mode='lines+markers', line=dict(color=color, width=width, dash='dot'), name=f"Current: {p_type}"))
        fig.add_trace(go.Scatter(x=lx, y=ly, mode='text', text=labels, textposition="bottom center", textfont=dict(size=14, color=color), showlegend=False))

    fig.update_layout(title="BTC Major Structure (Impulse + ABC Only)", template="plotly_dark", height=800)
    fig.write_html(filename)
    try: os.system(f'start {filename}')
    except: pass

def find_best_fit(wa, start_idx, end_idx_limit, generators, rules):
    """
    Finds the longest valid Impulse or Correction pattern.
    NO TRIANGLES.
    """
    best_waves = None
    best_type = ""
    max_len = 0
    
    # 1. Check Impulse (1-2-3-4-5)
    for opt in generators['impulse'].options_sorted:
        waves = wa.find_impulsive_wave(idx_start=start_idx, wave_config=opt.values)
        if waves:
            # High limit check (Phase 1 එක High එක පහු කරන්න හොඳ නෑ)
            if waves[-1].idx_end > end_idx_limit + 10: continue 
            
            wp = WavePattern(waves, verbose=False)
            if wp.check_rule(rules['impulse']):
                dur = waves[-1].idx_end - waves[0].idx_start
                if dur > max_len:
                    max_len = dur
                    best_waves = waves
                    best_type = "Impulse"

    # 2. Check Correction (A-B-C)
    # ABC scan කරන්නේ Impulse එකක් හරියටම සෙට් වුනේ නැත්නම් හෝ ABC එක Impulse එකට වඩා පැහැදිලි නම් විතරයි.
    try:
        for opt in generators['correction'].options_sorted:
            waves = wa.find_corrective_wave(idx_start=start_idx, wave_config=opt.values)
            if waves:
                if waves[-1].idx_end > end_idx_limit + 10: continue
                
                wp = WavePattern(waves, verbose=False)
                if wp.check_rule(rules['correction']):
                    dur = waves[-1].idx_end - waves[0].idx_start
                    
                    # Impulse vs Correction තෝරාගැනීම
                    # සාමාන්‍යයෙන් Impulse එකක් Correction එකකට වඩා දිගයි.
                    if dur > max_len: 
                        max_len = dur
                        best_waves = waves
                        best_type = "Correction"
    except: pass

    return best_type, best_waves

# --- Main Logic ---
def run_analysis():
    symbol = "BTCUSDT"
    print(f"🔄 Fetching DAILY data for {symbol}...")
    df = get_data(symbol, interval="1d", period="2y")
    if df.empty: return
    df = df.reset_index()
    if 'Date' not in df.columns and 'index' in df.columns: df['Date'] = df['index']
    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    df = df.rename(columns=rename_map)

    # 1. IDENTIFY MAJOR LOW AND HIGH
    lows = df['Low'].values
    highs = df['High'].values
    
    idx_abs_low = np.argmin(lows)
    idx_abs_high = np.argmax(highs)
    
    if idx_abs_low > idx_abs_high:
         print("⚠️ Warning: Bear Market Scenario (Low is after High). Scanning from Low anyway.")
         # Low එක High එකට පස්සේ නම්, අපිට High එක වෙනකම් scan කරන්න බෑ.
         # ඒ නිසා අන්තිම වෙනකම්ම scan කරනවා.
         idx_abs_high = len(df) - 1

    print(f"📍 Major Low Index: {idx_abs_low} | Major High Index: {idx_abs_high}")

    wa = WaveAnalyzer(df=df, verbose=False)
    
    # Generators (No Triangle options needed)
    gens = {
        'impulse': WaveOptionsGenerator5(up_to=3),
        'correction': WaveOptionsGenerator3(up_to=3)
    }
    
    # Rules (No Triangle rule)
    rules = {
        'impulse': Impulse('impulse'),
        'correction': Correction('correction')
    }

    # --- PHASE 1: LOW TO HIGH (The Climb) ---
    print("\n🚀 PHASE 1: Mapping the Bull Run (Low -> High)...")
    phase1_patterns = []
    curr_idx = idx_abs_low
    
    # High එකට ලං වෙනකම් Scan කරනවා
    while curr_idx < idx_abs_high - 5:
        p_type, waves = find_best_fit(wa, curr_idx, idx_abs_high, gens, rules)
        
        if waves:
            print(f"   ✅ Found {p_type} (Len: {len(waves)}). Ending at {waves[-1].idx_end}")
            phase1_patterns.append((p_type, waves))
            curr_idx = waves[-1].idx_end
            
            if curr_idx >= idx_abs_high:
                break
        else:
            curr_idx += 2

    # --- PHASE 2: HIGH TO NOW (The Current Pattern) ---
    print("\n🚀 PHASE 2: Analyzing Current Structure (High -> Now)...")
    phase2_patterns = []
    
    curr_idx = idx_abs_high
    end_of_chart = len(df) - 1
    
    while curr_idx < end_of_chart - 5:
        # Phase 2 එකේදී High Limit එකක් නෑ. අන්තිම වෙනකම්ම බලනවා.
        p_type, waves = find_best_fit(wa, curr_idx, end_of_chart + 50, gens, rules)
        
        if waves:
            print(f"   🔻 Found Current {p_type}. Ending at {waves[-1].idx_end}")
            phase2_patterns.append((p_type, waves))
            curr_idx = waves[-1].idx_end
        else:
             curr_idx += 1

    # Plotting
    visualize_phases(df, phase1_patterns, phase2_patterns, "btc_major_structure_no_tri.html")

if __name__ == "__main__":
    run_analysis()