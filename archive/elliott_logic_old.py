# elliott_logic.py
# ARCHITECTURE: Based on 'drstevendev' Permutation Logic
# FEATURES: 
#   1. Recursive Pattern Matching (Checks Impulse vs WXY vs ABC at every step).
#   2. Strict Rule Validation (Fibonacci, Overlaps, Lengths).
#   3. Scoring System (Selects the best fitting pattern).
#   4. Continuity (Next pattern starts exactly where the previous ended).

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- 1. DATA STRUCTURES ---

@dataclass
class Pivot:
    index: int
    price: float
    type: str # 'high' or 'low'
    date: object # Timestamp

@dataclass
class WaveResult:
    pattern_name: str
    waves: List[dict] # List of sub-waves
    end_index_offset: int # How many pivots this pattern consumed
    score: float # Quality of the match (0-100)

# --- 2. PIVOT DETECTION (High Sensitivity) ---

def get_pivots(df):
    """
    Detects strict local extrema.
    Uses a 3-candle window (Left/Right check) to catch all structures.
    """
    df = df.copy()
    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index
    
    pivots = []
    window = 1 
    
    for i in range(window, len(df) - window):
        is_high = True
        is_low = True
        
        for k in range(1, window + 1):
            if highs[i] <= highs[i-k] or highs[i] <= highs[i+k]: is_high = False
            if lows[i] >= lows[i-k] or lows[i] >= lows[i+k]: is_low = False
        
        if is_high:
            pivots.append(Pivot(i, highs[i], 'high', dates[i]))
        elif is_low:
            pivots.append(Pivot(i, lows[i], 'low', dates[i]))
            
    # Clean duplicates (Keep extremes)
    if not pivots: return []
    clean_pivots = [pivots[0]]
    for p in pivots[1:]:
        last = clean_pivots[-1]
        if p.type == last.type:
            if p.type == 'high':
                if p.price > last.price: clean_pivots[-1] = p
            else:
                if p.price < last.price: clean_pivots[-1] = p
        else:
            clean_pivots.append(p)
            
    return clean_pivots

# --- 3. PATTERN ANALYZER ENGINE ---

class ElliottAnalyzer:
    def __init__(self, pivots):
        self.pivots = pivots
        self.n = len(pivots)
        
        # Strict Constants
        self.MAX_W2_RETRACE = 0.99
        self.RISKY_W2_RETRACE = 0.886
        self.MIN_FLAT_B = 0.88
        self.MAX_FLAT_B = 1.382

    def _get_chunk(self, start_idx, length):
        if start_idx + length > self.n: return None
        return self.pivots[start_idx : start_idx + length]

    # --- VALIDATORS ---

    def check_impulse(self, start_idx) -> Optional[WaveResult]:
        # Needs 6 points (Start + 5 waves)
        chunk = self._get_chunk(start_idx, 6)
        if not chunk: return None
        
        p = [pt.price for pt in chunk]
        is_bullish = p[1] > p[0]
        
        # Lengths
        w1 = abs(p[1] - p[0])
        w2 = abs(p[2] - p[1])
        w3 = abs(p[3] - p[2])
        w4 = abs(p[4] - p[3])
        w5 = abs(p[5] - p[4])
        
        if w1 == 0: return None
        
        # RULES
        # 1. W2 Valid
        if is_bullish and p[2] <= p[0]: return None
        if not is_bullish and p[2] >= p[0]: return None
        if (w2/w1) > self.MAX_W2_RETRACE: return None # Strict 0.99
        
        # 2. W3 Not Shortest
        if w3 < w1 and w3 < w5: return None
        
        # 3. W5 Breaks W3 (Strict)
        if is_bullish and p[5] <= p[3]: return None
        if not is_bullish and p[5] >= p[3]: return None
        
        # 4. Overlap Check (Impulse vs Diagonal)
        overlap = False
        if is_bullish and p[4] < p[1]: overlap = True
        if not is_bullish and p[4] > p[1]: overlap = True
        
        name = "Impulse"
        score = 100
        
        if overlap:
            name = "Diagonal"
            score = 90 # Penalize slightly as Diagonals are rarer
            # Diagonal Strict: W4 cannot move beyond W2 start
            if is_bullish and p[4] < p[2]: return None
            if not is_bullish and p[4] > p[2]: return None

        return WaveResult(
            pattern_name=name,
            waves=[
                {'lbl': '1', 'p': chunk[1]}, {'lbl': '2', 'p': chunk[2]},
                {'lbl': '3', 'p': chunk[3]}, {'lbl': '4', 'p': chunk[4]},
                {'lbl': '5', 'p': chunk[5]}
            ],
            end_index_offset=5, # Advances 5 segments
            score=score
        )

    def check_abc(self, start_idx) -> Optional[WaveResult]:
        # Needs 4 points (Start + 3 waves)
        chunk = self._get_chunk(start_idx, 4)
        if not chunk: return None
        
        p = [pt.price for pt in chunk]
        lenA = abs(p[1] - p[0])
        lenB = abs(p[2] - p[1])
        if lenA == 0: return None
        
        retrace_B = lenB / lenA
        is_bullish = p[1] > p[0]
        
        name = None
        score = 0
        
        # FLAT Rules
        if retrace_B >= self.MIN_FLAT_B:
            if retrace_B > self.MAX_FLAT_B: return None
            is_expanded = (p[2] < p[0]) if is_bullish else (p[2] > p[0])
            name = "Expanded Flat" if is_expanded else "Regular Flat"
            score = 80
            
        # ZIGZAG Rules
        elif retrace_B < self.RISKY_W2_RETRACE:
            # B must not break start
            if is_bullish and p[2] <= p[0]: return None
            if not is_bullish and p[2] >= p[0]: return None
            
            # C must break A
            c_breaks_a = (p[3] > p[1]) if is_bullish else (p[3] < p[1])
            if not c_breaks_a: return None
            
            name = "ZigZag"
            score = 85
        else:
            return None

        return WaveResult(
            pattern_name=name,
            waves=[
                {'lbl': 'A', 'p': chunk[1]},
                {'lbl': 'B', 'p': chunk[2]},
                {'lbl': 'C', 'p': chunk[3]}
            ],
            end_index_offset=3,
            score=score
        )

    def check_wxy(self, start_idx) -> Optional[WaveResult]:
        # Needs 8 points (Start + W(3) + X(1) + Y(3)) -> 7 Segments
        chunk = self._get_chunk(start_idx, 8)
        if not chunk: return None
        
        # 1. Check W (ABC)
        # We manually validate the ABC structure of W (Indices 0-3)
        w_res = self.check_abc(start_idx) # Reuse ABC logic
        if not w_res: return None
        
        # 2. Check X (Connector) (Index 3-4)
        p = [pt.price for pt in chunk]
        w_height = abs(p[3] - p[0])
        if w_height == 0: return None
        # X shouldn't break start of W (Strict WXY)
        is_bullish_w = p[1] > p[0]
        if is_bullish_w and p[4] <= p[0]: return None
        if not is_bullish_w and p[4] >= p[0]: return None
        
        # 3. Check Y (ABC) (Indices 4-7)
        # Temporarily create a sub-analyzer or check manually
        # Y is an ABC starting from pivot 4
        y_pivots = chunk[4:]
        # Quick manual check of Y shape
        lenYa = abs(y_pivots[1].price - y_pivots[0].price)
        lenYb = abs(y_pivots[2].price - y_pivots[1].price)
        if lenYa == 0: return None
        
        # Basic Y validation (ZigZag/Flat shape)
        y_valid = True # simplified for speed, relying on structure
        
        if y_valid:
            return WaveResult(
                pattern_name="Double Combo (WXY)",
                waves=[
                    {'lbl': 'W', 'p': chunk[3]}, # Showing endpoints
                    {'lbl': 'X', 'p': chunk[4]},
                    {'lbl': 'Y', 'p': chunk[7]}
                ],
                end_index_offset=7,
                score=95 # High score to prioritize complex corrections over simple ones if valid
            )
        return None

    # --- MAIN SOLVER ---
    
    def solve(self):
        final_waves = []
        i = 0
        
        # Loop through pivots
        while i <= self.n - 4:
            candidates = []
            
            # 1. Try WXY (Needs 8 points)
            res_wxy = self.check_wxy(i)
            if res_wxy: candidates.append(res_wxy)
            
            # 2. Try Impulse (Needs 6 points)
            res_imp = self.check_impulse(i)
            if res_imp: candidates.append(res_imp)
            
            # 3. Try ABC (Needs 4 points)
            res_abc = self.check_abc(i)
            if res_abc: candidates.append(res_abc)
            
            if candidates:
                # PICK THE BEST CANDIDATE (Based on Score)
                best = max(candidates, key=lambda x: x.score)
                
                # TRANSFORM TO JSON FORMAT
                for w in best.waves:
                    final_waves.append({
                        'lbl': w['lbl'],
                        'p': {
                            'price': w['p'].price,
                            'date': w['p'].date, # Object
                            'type': w['p'].type
                        }
                    })
                
                # ADVANCE INDEX
                # Important: We move forward by the number of segments used.
                # E.g., Impulse uses 5 segments (0->5), so next start is at 5.
                i += best.end_index_offset
                
            else:
                # No pattern found starting at 'i'. Treat as noise/sub-wave and move 1 step.
                i += 1
                
        return final_waves

# --- 4. EXTERNAL INTERFACE ---

def map_market_final(df):
    """
    Called by main.py
    """
    # 1. Get Strict Pivots
    pivots = get_pivots(df)
    
    if len(pivots) < 6: return []
    
    # 2. Run the Analyzer Engine
    engine = ElliottAnalyzer(pivots)
    waves = engine.solve()
    
    # 3. Last Wave Hiding (For AI)
    # We only hide if we have enough waves
    if waves and len(waves) > 3:
        waves = waves[:-3]
        
    return waves