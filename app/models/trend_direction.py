import math
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Global Constants
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"
MIXED = "MIXED"
UNKNOWN = "UNKNOWN"

# ==========================================================
# LEAD AI v2: TREND DIRECTION GATEKEEPER & SCORING SYSTEM
# ==========================================================

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None: return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            val = float(value)
            return None if math.isnan(val) or math.isinf(val) else val
        text = str(value).replace(",", "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None
    except Exception: return None

def _format_price(value: Any) -> str:
    val = _safe_float(value)
    if val is None: return "N/A"
    if abs(val) >= 1000: return f"{val:.2f}"
    if abs(val) >= 10: return f"{val:.3f}"
    return f"{val:.6f}"

def _prepare_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    data = df.copy()
    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    data = data.rename(columns=rename_map)
    data.columns = [c.capitalize() for c in data.columns]
    return data

def _detect_swing_points(df: pd.DataFrame, lookback: int = 180, window: int = 3) -> List[Dict[str, Any]]:
    points = []
    try:
        data = df.tail(lookback)
        highs, lows = data["High"].values, data["Low"].values
        idxs = list(data.index)
        for i in range(window, len(data) - window):
            if highs[i] > np.nanmax(highs[i-window:i]) and highs[i] > np.nanmax(highs[i+1:i+1+window]):
                points.append({"type": "high", "price": float(highs[i]), "position": i})
            if lows[i] < np.nanmin(lows[i-window:i]) and lows[i] < np.nanmin(lows[i+1:i+1+window]):
                points.append({"type": "low", "price": float(lows[i]), "position": i})
        return points
    except Exception: return points

def _market_structure_bias(df: Optional[pd.DataFrame], timeframe_name: str) -> Dict[str, Any]:
    res = {"bias": UNKNOWN, "last_high": None, "last_low": None}
    data = _prepare_df(df)
    if data.empty: return res
    swings = _detect_swing_points(data)
    h = [p["price"] for p in swings if p["type"] == "high"]
    l = [p["price"] for p in swings if p["type"] == "low"]
    if len(h) < 2 or len(l) < 2: return res
    
    res["last_high"], res["last_low"] = h[-1], l[-1]
    hh, hl = h[-1] > h[-2], l[-1] > l[-2]
    lh, ll = h[-1] < h[-2], l[-1] < l[-2]

    if hh and hl: res["bias"] = BULLISH
    elif lh and ll: res["bias"] = BEARISH
    else: res["bias"] = MIXED
    return res

def _evaluate_elliott_bias(waves: List[Dict[str, Any]]) -> str:
    if not waves:
        return NEUTRAL

    try:
        last_wave = waves[-1]
        text = str(last_wave.get("pattern_type", "")).lower()
        direction = str(last_wave.get("direction", "")).lower()
        bias = str(last_wave.get("bias", "")).lower()

        combined = f"{text} {direction} {bias}"

        if any(x in combined for x in ["bearish", "down", "short"]):
            return BEARISH

        if any(x in combined for x in ["bullish", "up", "long"]):
            return BULLISH

        if "impulse" in combined:
            return BULLISH

        return NEUTRAL

    except Exception:
        return NEUTRAL

# --- MAIN ENGINE (HYBRID VERSION) ---

def _latest_close(*dfs) -> Optional[float]:
    for df in dfs:
        data = _prepare_df(df)
        if not data.empty and "Close" in data.columns:
            val = _safe_float(data["Close"].iloc[-1])
            if val is not None:
                return val
    return None


def _safe_wait_context(reason: str) -> str:
    return "\n".join([
        "--- DOMINANT TREND CONTEXT (LEAD AI v2 - UNIVERSAL) ---",
        "Dominant Direction: UNKNOWN",
        "Preferred Trade Side: WAIT",
        "EMA 200 Filter (Confirm TF): UNKNOWN",
        "Structure Broken (CHoCH): UNKNOWN",
        "",
        "CRITICAL DIRECTION RULES:",
        f"- {reason}",
        "- AI MUST return WAIT until enough reliable trend data is available."
    ])

def build_dominant_trend_context(
    df_macro: Optional[pd.DataFrame] = None,
    df_swing: Optional[pd.DataFrame] = None,
    df_confirm_1h: Optional[pd.DataFrame] = None,
    df_micro: Optional[pd.DataFrame] = None,
    user_tf: str = "15m",
    **kwargs # Other wave data
) -> str:
    try:
        df_sw = _prepare_df(df_swing)
        df_conf = _prepare_df(df_confirm_1h)
        df_mic = _prepare_df(df_micro)
        
        curr_price = _latest_close(df_micro, df_confirm_1h, df_swing, df_macro)

        if curr_price is None:
            return _safe_wait_context("Current price could not be detected.")
        
        # --- 1. GATEKEEPER LOGIC (Remains Relaxed to keep signal count High) ---

        
        if df_conf.empty or "Close" not in df_conf.columns:
            return _safe_wait_context("Confirm TF candle data is missing.")

        confirm_closes = df_conf["Close"].dropna()

        if len(confirm_closes) < 50:
            return _safe_wait_context("Not enough Confirm TF candle data for EMA/trend validation.")

        ema200_confirm = confirm_closes.ewm(span=200, adjust=False).mean().iloc[-1]
        ema_bias = BULLISH if curr_price > ema200_confirm else BEARISH

        struct_swing = _market_structure_bias(df_sw, "Swing TF")
        struct_confirm = _market_structure_bias(df_conf, "Confirm TF")

        preferred_side = "WAIT"
        is_broken = False
        
        if ema_bias == BULLISH and (struct_confirm["bias"] == BULLISH or struct_swing["bias"] == BULLISH):
            preferred_side = "LONG"
            if struct_confirm["last_low"] and curr_price < struct_confirm["last_low"]:
                is_broken, preferred_side = True, "WAIT"

        elif ema_bias == BEARISH and (struct_confirm["bias"] == BEARISH or struct_swing["bias"] == BEARISH):
            preferred_side = "SHORT"
            if struct_confirm["last_high"] and curr_price > struct_confirm["last_high"]:
                is_broken, preferred_side = True, "WAIT"

        # --- 2. ADVANCED SCORING SYSTEM (The "Brain" / Identity of the bot) ---
        swing_waves = kwargs.get("swing_waves", [])
        confirm_waves = kwargs.get("confirm_1h_waves", [])
        
        ew_swing_bias = _evaluate_elliott_bias(swing_waves)
        ew_confirm_bias = _evaluate_elliott_bias(confirm_waves)
        
        # Calculate composite score (0 to 100)
        score = 50 # Start neutral
        
        if ema_bias == BULLISH: score += 10
        elif ema_bias == BEARISH: score -= 10
        
        if struct_confirm["bias"] == BULLISH: score += 15
        elif struct_confirm["bias"] == BEARISH: score -= 15
        
        if struct_swing["bias"] == BULLISH: score += 15
        elif struct_swing["bias"] == BEARISH: score -= 15
        
        if ew_confirm_bias == BULLISH: score += 5
        elif ew_confirm_bias == BEARISH: score -= 5
        
        if ew_swing_bias == BULLISH: score += 5
        elif ew_swing_bias == BEARISH: score -= 5
        
        # Clamp score between 0 and 100
        score = max(0, min(100, score))
        
        if score >= 75: conf_level = "STRONG BULLISH"
        elif score >= 55: conf_level = "BULLISH"
        elif score >= 45: conf_level = "NEUTRAL"
        elif score >= 25: conf_level = "BEARISH"
        else: conf_level = "STRONG BEARISH"

        if preferred_side == "LONG":
            dominant_direction = BULLISH
        elif preferred_side == "SHORT":
            dominant_direction = BEARISH
        else:
            dominant_direction = struct_confirm["bias"]

        # --- 3. TEXT BUILDER ---
        lines = [
            "--- DOMINANT TREND CONTEXT (LEAD AI v2 - UNIVERSAL) ---",
            f"Dominant Direction: {dominant_direction}",
            f"Confirm TF Structure: {struct_confirm['bias']}",
            f"Swing TF Structure: {struct_swing['bias']}",
            f"Preferred Trade Side: {preferred_side}",
            f"EMA 200 Filter (Confirm TF): {ema_bias}",
            f"Confirm TF EMA200 Value: {_format_price(ema200_confirm)}",
            f"Current Price: {_format_price(curr_price)}",
            f"Structure Broken (CHoCH): {'YES' if is_broken else 'NO'}",
            "",
            "CRITICAL DIRECTION RULES:",
            "- NO LONGs below Confirm TF EMA200 unless strong reclaim/retest confirmation exists.",
            "- NO SHORTs above Confirm TF EMA200 unless strong rejection/breakdown confirmation exists.",
            "- NO LONGs if price breaks below Confirm TF last swing low.",
            "- NO SHORTs if price breaks above Confirm TF last swing high.",
            "- If CHoCH is YES, AI MUST return WAIT.",
            f"- Swing TF Structure is currently {struct_swing['bias']}.",
            "",
            "--- ADVANCED SCORING & CONFLUENCE ---",
            f"Dominant Confidence Score: {score}/100 ({conf_level})",
            f"Confirm TF Elliott Bias: {ew_confirm_bias}",
            f"Swing TF Elliott Bias: {ew_swing_bias}",
            "(Use scoring to determine probability: >75/100 or <25/100 implies High Probability trend)"
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"Trend Logic Error: {str(e)}"