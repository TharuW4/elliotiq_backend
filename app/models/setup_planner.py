# app/models/setup_planner.py

"""
Backend Setup Planner - Template/Priority Version

Final architecture target:
1) trend_regime.py detects allowed/blocked side
2) context_bundle.py converts existing Fib/S/R/EMA/ATR/rejection/Elliott text into structured data
3) this planner generates backend-owned candidate levels
4) candidates pass hard validation: entry side, SL realism, TP realism, no-chase, HTF alignment
5) selection is by setup-template priority, NOT highest score
6) AI may only explain/review selected backend candidates; it cannot invent prices

This module does NOT call AI.
This module does NOT depend on trade_engine.py.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from .wave_template_detector import (
        detect_wave_templates,
        format_wave_template_confirmation,
        compact_wave_templates_for_debug,
    )
except Exception:
    detect_wave_templates = None
    format_wave_template_confirmation = None
    compact_wave_templates_for_debug = None
try:
    from .mtf_elliott_confirmation import confirm_mtf_elliott_alignment
except Exception:
    confirm_mtf_elliott_alignment = None
try:
    from .abc_c_ending_setup import make_abc_c_ending_idea
except Exception:
    make_abc_c_ending_idea = None


# ==========================================================
# Tunable constants
# ==========================================================

MAX_RISK_PCT = 5.0
MIN_RR = 2.5
NO_CHASE_TP1_PROGRESS = 0.70
PREFERRED_RR = 3.0

TP1_R = 1.5
TP2_R_MIN = 3.0
TP2_R_MAX = 5.0

MIN_REGIME_CONFIDENCE = 0.55

DEFAULT_SWING_WINDOW = 3
DEFAULT_LOOKBACK = 180

# SL protection
MIN_STOP_ATR_MULTIPLIER = 1.20
STRUCTURE_ATR_BUFFER_MULTIPLIER = 0.80

# Extra SL gap added after backend generates the original SL.
# LONG: final SL = generated SL - ATR * 0.20
# SHORT: final SL = generated SL + ATR * 0.20
EXTRA_SL_ATR_BUFFER = 0.20

# HTF alignment protection
HTF_ALIGN_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m"}
HTF_EMA_BUFFER_ATR_MULTIPLIER = 0.25

# Context confluence tolerance
CONTEXT_ZONE_TOLERANCE_ATR = 0.75
CONTEXT_ZONE_TOLERANCE_PCT = 0.0035
# TP1 wall protection
# If wall is closer than this ATR distance from entry, treat it as near-entry micro level.
# If wall is farther/equal, treat it as meaningful TP1 blocker.
TP1_WALL_MIN_DISTANCE_ATR = 0.75


# ==========================================================
# Basic helpers
# ==========================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float, np.number)):
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        text = str(value).replace(",", "").replace("$", "").strip()
        if not text:
            return default
        v = float(text)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def _normalize_ohlc(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        out = df.copy()
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "date": "Date",
            "time": "Date",
        }
        out = out.rename(columns={c: rename_map.get(str(c).lower(), c) for c in out.columns})
        required = ["Open", "High", "Low", "Close"]
        for col in required:
            if col not in out.columns:
                return pd.DataFrame()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=required)
        return out
    except Exception:
        return pd.DataFrame()


def format_setup_price(value: Any) -> str:
    v = _safe_float(value)
    if v is None:
        return ""
    av = abs(v)
    if av >= 1000:
        return f"{v:.2f}"
    if av >= 100:
        return f"{v:.2f}"
    if av >= 10:
        return f"{v:.3f}"
    if av >= 1:
        return f"{v:.4f}"
    if av >= 0.01:
        return f"{v:.5f}"
    if av >= 0.0001:
        return f"{v:.7f}"
    return f"{v:.10f}"


def _ema(series: pd.Series, span: int) -> Optional[float]:
    try:
        if series is None or len(series) < 5:
            return None
        return float(series.ewm(span=span, adjust=False).mean().iloc[-1])
    except Exception:
        return None


def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)
        if len(d) < period + 2:
            return None
        high = d["High"]
        low = d["Low"]
        close = d["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        val = tr.rolling(period).mean().iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _tf_key(user_tf: str) -> str:
    return str(user_tf or "").lower().strip()


def _timeframe_max_entry_distance_pct(user_tf: str) -> float:
    tf = _tf_key(user_tf)
    if tf in ["1m", "3m", "5m"]:
        return 1.2
    if tf in ["15m", "30m"]:
        return 2.5
    if tf in ["1h", "60m"]:
        return 4.0
    if tf in ["4h", "240m"]:
        return 7.0
    if tf in ["1d", "d", "daily"]:
        return 10.0
    return 4.0


def _timeframe_max_stop_atr(user_tf: str) -> float:
    tf = _tf_key(user_tf)
    if tf in ["1m", "3m", "5m"]:
        return 2.0
    if tf in ["15m"]:
        return 2.8
    if tf in ["30m"]:
        return 3.0
    if tf in ["1h", "60m"]:
        return 3.5
    if tf in ["4h", "240m"]:
        return 5.0
    if tf in ["1d", "d", "daily"]:
        return 6.0
    return 3.5


def _timeframe_max_tp2_atr(user_tf: str) -> float:
    tf = _tf_key(user_tf)
    if tf in ["1m", "3m", "5m"]:
        return 4.0
    if tf in ["15m", "30m"]:
        return 5.0
    if tf in ["1h", "60m"]:
        return 8.0
    if tf in ["4h", "240m"]:
        return 12.0
    if tf in ["1d", "d", "daily"]:
        return 16.0
    return 8.0


def _min_rr_for_timeframe(user_tf: str) -> float:
    tf = _tf_key(user_tf)

    if tf in ["1m", "3m", "5m", "15m", "30m"]:
        return 2.0

    if tf in ["1h", "60m"]:
        return 2.2

    if tf in ["4h", "240m"]:
        return 2.4

    return MIN_RR


def _has_strong_context_override(context_bundle: Optional[Dict[str, Any]]) -> bool:
    """
    Allow cautious candidate generation even when dominant trend says WAIT,
    but only if there is real Fib/S/R/rejection context.

    This does NOT override CHoCH or both-side rejection/chop hard blocks.
    It only prevents dominant WAIT from killing candidate generation too early.
    """
    try:
        cb = _context_dict(context_bundle)
        if not cb:
            return False

        auto_conf = cb.get("automated_confluence", {}) or {}
        rejection = cb.get("rejection", {}) or {}
        derived = cb.get("derived", {}) or {}
        fib = cb.get("fibonacci", {}) or {}

        high_confluence = bool(auto_conf.get("high_confluence"))

        support_rejection = bool(rejection.get("support_rejection_confirmed"))
        resistance_rejection = bool(rejection.get("resistance_rejection_confirmed"))

        elliott_usable = bool(derived.get("elliott_fib_usable"))
        visible_available = bool(derived.get("visible_fib_available"))

        visible = fib.get("visible", {}) or {}
        visible_direction = bool(visible.get("direction"))

        return bool(
            high_confluence
            or support_rejection
            or resistance_rejection
            or elliott_usable
            or visible_available
            or visible_direction
        )

    except Exception:
        return False


def _is_lower_execution_tf(user_tf: str) -> bool:
    return _tf_key(user_tf) in HTF_ALIGN_TIMEFRAMES


# ==========================================================
# Context-bundle helpers
# ==========================================================

def _context_dict(context_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return context_bundle if isinstance(context_bundle, dict) else {}


def _context_prefers_wait(context_bundle: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    cb = _context_dict(context_bundle)
    derived = cb.get("derived", {}) or {}
    dom = cb.get("dominant_trend", {}) or {}
    if derived.get("choch") or dom.get("choch") is True:
        return True, "WAIT. Dominant trend context reports CHoCH / structure break."
    if derived.get("dominant_wait") or str(dom.get("preferred_trade_side") or "").upper() == "WAIT":
        return True, "WAIT. Dominant trend context preferred side is WAIT."
    if derived.get("both_support_and_resistance_rejection"):
        return True, "WAIT. Both support and resistance rejection detected; market is choppy."
    return False, ""


def _context_execution_price(context_bundle: Optional[Dict[str, Any]], fallback: Optional[float]) -> Optional[float]:
    cb = _context_dict(context_bundle)
    for path in [
        ("derived", "execution_price"),
        ("execution", "live_price"),
        ("market_structure", "current_price"),
    ]:
        d = cb.get(path[0], {}) or {}
        v = _safe_float(d.get(path[1]))
        if v is not None:
            return v
    return fallback


def _context_atr(context_bundle: Optional[Dict[str, Any]], fallback: Optional[float]) -> Optional[float]:
    cb = _context_dict(context_bundle)
    for path in [("derived", "atr"), ("market_structure", "atr")]:
        d = cb.get(path[0], {}) or {}
        v = _safe_float(d.get(path[1]))
        if v is not None and v > 0:
            return v
    return fallback


def _zone_valid(zone: Any) -> bool:
    return isinstance(zone, dict) and bool(zone.get("valid")) and zone.get("low") is not None and zone.get("high") is not None


def _zone_mid(zone: Dict[str, Any]) -> Optional[float]:
    if not _zone_valid(zone):
        return None
    return _safe_float(zone.get("mid"), (_safe_float(zone.get("low")) + _safe_float(zone.get("high"))) / 2.0)


def _distance_to_zone(price: float, zone: Dict[str, Any]) -> Optional[float]:
    try:
        if not _zone_valid(zone):
            return None
        low = float(zone["low"])
        high = float(zone["high"])
        if low <= price <= high:
            return 0.0
        if price < low:
            return low - price
        return price - high
    except Exception:
        return None


def _price_near_zone(price: float, zone: Dict[str, Any], tolerance: float) -> bool:
    d = _distance_to_zone(price, zone)
    return d is not None and d <= tolerance


def _context_zone_confluences(side: str, entry: float, atr_val: Optional[float], context_bundle: Optional[Dict[str, Any]]) -> List[str]:
    confluences: List[str] = []
    try:
        cb = _context_dict(context_bundle)
        ms = cb.get("market_structure", {}) or {}
        fib = cb.get("fibonacci", {}) or {}
        derived = cb.get("derived", {}) or {}

        tolerance = max(abs(entry) * CONTEXT_ZONE_TOLERANCE_PCT, (atr_val or 0) * CONTEXT_ZONE_TOLERANCE_ATR)

        zones = []
        if side == "LONG":
            for label in ["major_support_zone", "long_lookback_major_support_zone"]:
                z = ms.get(label, {}) or {}
                zones.append((label, z))
        elif side == "SHORT":
            for label in ["major_resistance_zone", "long_lookback_major_resistance_zone"]:
                z = ms.get(label, {}) or {}
                zones.append((label, z))

        for label, zone in zones:
            if _price_near_zone(entry, zone, tolerance):
                confluences.append(f"Context {label} confluence")

        # Fib zones: prefer best source but allow visible as reliable fallback.
        source = derived.get("fib_entry_source") or "visible"
        fib_sources = []
        if source == "elliott":
            fib_sources.append(("Elliott", fib.get("elliott", {}) or {}))
        fib_sources.append(("Visible", fib.get("visible", {}) or {}))

        for src_name, f in fib_sources:
            for key in ["normal_zone", "shallow_zone"]:
                zone = f.get(key, {}) or {}
                if _price_near_zone(entry, zone, tolerance):
                    confluences.append(f"{src_name} Fib {key} confluence")
            gp = _safe_float(f.get("golden_pocket"))
            if gp is not None and abs(entry - gp) <= tolerance:
                confluences.append(f"{src_name} Fib golden pocket confluence")
            deep = _safe_float(f.get("deep_pullback"))
            if deep is not None and abs(entry - deep) <= tolerance:
                confluences.append(f"{src_name} Fib deep pullback confluence")

        return list(dict.fromkeys(confluences))
    except Exception:
        return confluences


def _context_targets(side: str, entry: float, tp1: float, atr_val: Optional[float], context_bundle: Optional[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """
    Structure-aware TP candidates from context. Only targets beyond TP1 are useful for TP2.
    """
    targets: List[Tuple[str, float]] = []
    try:
        cb = _context_dict(context_bundle)
        ms = cb.get("market_structure", {}) or {}
        fib = cb.get("fibonacci", {}) or {}

        def add(label: str, v: Any) -> None:
            fv = _safe_float(v)
            if fv is None or fv <= 0:
                return
            if side == "LONG" and fv > tp1:
                targets.append((label, fv))
            elif side == "SHORT" and fv < tp1:
                targets.append((label, fv))

        if side == "LONG":
            add("nearest_resistance", ms.get("nearest_resistance"))
            add("major_swing_high", ms.get("major_swing_high"))
            add("long_lookback_high", ms.get("long_lookback_high"))
            for label in ["major_resistance_zone", "long_lookback_major_resistance_zone"]:
                z = ms.get(label, {}) or {}
                if _zone_valid(z):
                    add(label + "_low", z.get("low"))
                    add(label + "_mid", z.get("mid"))
                    add(label + "_high", z.get("high"))
        elif side == "SHORT":
            add("nearest_support", ms.get("nearest_support"))
            add("major_swing_low", ms.get("major_swing_low"))
            add("long_lookback_low", ms.get("long_lookback_low"))
            for label in ["major_support_zone", "long_lookback_major_support_zone"]:
                z = ms.get(label, {}) or {}
                if _zone_valid(z):
                    add(label + "_high", z.get("high"))
                    add(label + "_mid", z.get("mid"))
                    add(label + "_low", z.get("low"))

        # Fib extensions from visible / Elliott. These are directional continuation targets.
        for src in ["visible", "elliott"]:
            f = fib.get(src, {}) or {}
            for k, v in (f.get("extensions", {}) or {}).items():
                add(f"{src}_extension_{k}", v)

        # Sort nearest useful target first.
        if side == "LONG":
            targets.sort(key=lambda x: x[1])
        else:
            targets.sort(key=lambda x: -x[1])

        # Remove near-duplicates
        out: List[Tuple[str, float]] = []
        for label, value in targets:
            if not out or abs(value - out[-1][1]) > max(abs(entry) * 0.0005, (atr_val or 0) * 0.25):
                out.append((label, value))
        return out
    except Exception:
        return targets


# ==========================================================
# Swing / HTF helpers
# ==========================================================

def _find_pivots(df: pd.DataFrame, window: int = DEFAULT_SWING_WINDOW, lookback: int = DEFAULT_LOOKBACK) -> Dict[str, Any]:
    d = _normalize_ohlc(df).tail(lookback).copy()
    result = {
        "highs": [], "lows": [],
        "recent_high": None, "previous_high": None,
        "recent_low": None, "previous_low": None,
        "highest": None, "lowest": None,
        "highest_idx": None, "lowest_idx": None,
    }
    try:
        if len(d) < max(20, window * 4):
            return result
        h = d["High"].values
        l = d["Low"].values
        idx = list(d.index)
        highs: List[Tuple[Any, float]] = []
        lows: List[Tuple[Any, float]] = []
        for i in range(window, len(d) - window):
            if h[i] >= np.nanmax(h[i - window:i + window + 1]):
                highs.append((idx[i], float(h[i])))
            if l[i] <= np.nanmin(l[i - window:i + window + 1]):
                lows.append((idx[i], float(l[i])))
        if not highs:
            highs.append((d["High"].idxmax(), float(d["High"].max())))
        if not lows:
            lows.append((d["Low"].idxmin(), float(d["Low"].min())))
        result["highs"] = highs
        result["lows"] = lows
        result["recent_high"] = highs[-1][1] if highs else None
        result["previous_high"] = highs[-2][1] if len(highs) >= 2 else None
        result["recent_low"] = lows[-1][1] if lows else None
        result["previous_low"] = lows[-2][1] if len(lows) >= 2 else None
        highest_idx = d["High"].idxmax()
        lowest_idx = d["Low"].idxmin()
        result["highest"] = float(d.loc[highest_idx, "High"])
        result["lowest"] = float(d.loc[lowest_idx, "Low"])
        result["highest_idx"] = highest_idx
        result["lowest_idx"] = lowest_idx
        return result
    except Exception:
        return result


def _nearest_level(levels: List[Tuple[Any, float]], price: float, side: str) -> Optional[float]:
    try:
        vals = [float(x[1]) for x in levels if _safe_float(x[1]) is not None]
        if not vals:
            return None
        if side == "LONG":
            below = [v for v in vals if v <= price]
            return max(below) if below else min(vals, key=lambda v: abs(v - price))
        if side == "SHORT":
            above = [v for v in vals if v >= price]
            return min(above) if above else min(vals, key=lambda v: abs(v - price))
    except Exception:
        return None
    return None


def _impulse_range(df: pd.DataFrame, side: str, lookback: int = DEFAULT_LOOKBACK) -> Dict[str, Any]:
    d = _normalize_ohlc(df).tail(lookback).copy()
    result = {"valid": False, "start": None, "end": None, "high": None, "low": None, "range": None, "reason": ""}
    try:
        if len(d) < 30:
            result["reason"] = "Not enough candles for impulse range."
            return result
        side = str(side).upper()
        if side == "LONG":
            high_idx = d["High"].idxmax()
            high_pos = list(d.index).index(high_idx)
            before_high = d.iloc[:high_pos + 1]
            if before_high.empty:
                before_high = d
            low_idx = before_high["Low"].idxmin()
            impulse_low = float(before_high.loc[low_idx, "Low"])
            impulse_high = float(d.loc[high_idx, "High"])
            if impulse_high <= impulse_low:
                result["reason"] = "Bullish impulse range invalid."
                return result
            result.update({"valid": True, "start": impulse_low, "end": impulse_high, "high": impulse_high, "low": impulse_low, "range": impulse_high - impulse_low, "reason": "Bullish impulse low-to-high range detected."})
            return result
        if side == "SHORT":
            low_idx = d["Low"].idxmin()
            low_pos = list(d.index).index(low_idx)
            before_low = d.iloc[:low_pos + 1]
            if before_low.empty:
                before_low = d
            high_idx = before_low["High"].idxmax()
            impulse_high = float(before_low.loc[high_idx, "High"])
            impulse_low = float(d.loc[low_idx, "Low"])
            if impulse_high <= impulse_low:
                result["reason"] = "Bearish impulse range invalid."
                return result
            result.update({"valid": True, "start": impulse_high, "end": impulse_low, "high": impulse_high, "low": impulse_low, "range": impulse_high - impulse_low, "reason": "Bearish impulse high-to-low range detected."})
            return result
    except Exception as e:
        result["reason"] = f"Impulse range error: {str(e)}"
    return result


def _derive_htf_context(df_confirm: Optional[pd.DataFrame], side: str, user_tf: str) -> Dict[str, Any]:
    ctx = {
        "enabled": False,
        "side": str(side or "").upper(),
        "htf_close": None,
        "htf_ema50": None,
        "htf_ema200": None,
        "htf_recent_high": None,
        "htf_recent_low": None,
        "htf_anchor_level": None,
        "reason": "HTF alignment not active.",
    }
    try:
        if not _is_lower_execution_tf(user_tf):
            return ctx
        d = _normalize_ohlc(df_confirm)
        if d.empty or len(d) < 50:
            ctx["reason"] = "Confirm TF data unavailable or too short."
            return ctx
        side = str(side or "").upper()
        if side not in ["LONG", "SHORT"]:
            return ctx
        close = float(d["Close"].iloc[-1])
        ema50 = _ema(d["Close"], 50)
        ema200 = _ema(d["Close"], 200)
        pivots = _find_pivots(d, window=DEFAULT_SWING_WINDOW, lookback=min(DEFAULT_LOOKBACK, len(d)))
        ctx.update({
            "enabled": True,
            "htf_close": close,
            "htf_ema50": ema50,
            "htf_ema200": ema200,
            "htf_recent_high": pivots.get("recent_high"),
            "htf_recent_low": pivots.get("recent_low"),
        })
        levels = []
        if side == "SHORT":
            for v in [ema200, ema50, pivots.get("recent_high")]:
                fv = _safe_float(v)
                if fv is not None and fv >= close:
                    levels.append(fv)
            if levels:
                ctx["htf_anchor_level"] = min(levels)
                ctx["reason"] = "Lower TF SHORT must align with HTF resistance/EMA retest zone."
            else:
                ctx["reason"] = "No HTF resistance anchor above current close."
        elif side == "LONG":
            for v in [ema200, ema50, pivots.get("recent_low")]:
                fv = _safe_float(v)
                if fv is not None and fv <= close:
                    levels.append(fv)
            if levels:
                ctx["htf_anchor_level"] = max(levels)
                ctx["reason"] = "Lower TF LONG must align with HTF support/EMA retest zone."
            else:
                ctx["reason"] = "No HTF support anchor below current close."
        return ctx
    except Exception as e:
        ctx["enabled"] = False
        ctx["reason"] = f"HTF context error: {str(e)}"
        return ctx


# ==========================================================
# Candidate construction and validations
# ==========================================================

def _dynamic_tp2_r(user_tf: str, regime: Dict[str, Any], confluences: List[str], risk_pct: Optional[float] = None) -> float:
    tf = _tf_key(user_tf)
    confidence = float(regime.get("confidence") or 0)
    regime_name = str(regime.get("regime") or "").upper()
    tp2_r = TP2_R_MIN
    if confidence >= 0.75:
        tp2_r += 0.75
    elif confidence >= 0.65:
        tp2_r += 0.45
    if "PULLBACK" in regime_name:
        tp2_r += 0.50
    if "BREAKOUT" in regime_name or "BREAKDOWN" in regime_name:
        tp2_r += 0.25
    if len(confluences or []) >= 2:
        tp2_r += 0.35
    elif len(confluences or []) == 1:
        tp2_r += 0.15
    if tf in ["1m", "3m", "5m"]:
        tp2_r -= 0.50
    elif tf in ["15m", "30m"]:
        tp2_r -= 0.25
    elif tf in ["4h", "240m", "1d", "daily"]:
        tp2_r += 0.25
    if risk_pct is not None:
        if risk_pct > 3.5:
            tp2_r = min(tp2_r, 3.25)
        elif risk_pct > 3.0:
            tp2_r = min(tp2_r, 3.5)
    return round(max(TP2_R_MIN, min(TP2_R_MAX, tp2_r)), 2)


def _status_from_current_price(side: str, entry: float, stop: float, tp1: float, current_price: float) -> Tuple[str, str]:
    try:
        side = str(side).upper()
        if side == "LONG":
            if current_price <= stop:
                return "INVALIDATED", "Current price is already beyond long invalidation."
            if current_price >= entry + ((tp1 - entry) * NO_CHASE_TP1_PROGRESS):
                return "LIMIT_ORDER", "Valid setup, but price is already extended toward TP1. Keep as limit only; do not mark running."
            return "LIMIT_ORDER", "Fresh backend setup. Wait for price to reach entry after signal creation."
        if side == "SHORT":
            if current_price >= stop:
                return "INVALIDATED", "Current price is already beyond short invalidation."
            if current_price <= entry - ((entry - tp1) * NO_CHASE_TP1_PROGRESS):
                return "LIMIT_ORDER", "Valid setup, but price is already extended toward TP1. Keep as limit only; do not mark running."
            return "LIMIT_ORDER", "Fresh backend setup. Wait for price to reach entry after signal creation."
    except Exception:
        pass
    return "LIMIT_ORDER", "Fresh backend setup. Wait for entry after signal creation."


def _candidate_stop_for_side(side: str, entry: float, pivots: Dict[str, Any], atr_val: Optional[float], impulse: Dict[str, Any], ema50: Optional[float] = None, ema200: Optional[float] = None, context_bundle: Optional[Dict[str, Any]] = None) -> Optional[float]:
    try:
        side = str(side).upper()
        atr_buffer = (atr_val or 0) * STRUCTURE_ATR_BUFFER_MULTIPLIER
        cb = _context_dict(context_bundle)
        ms = cb.get("market_structure", {}) or {}

        if side == "LONG":
            lows = []
            for key in ["recent_low", "previous_low", "lowest"]:
                v = _safe_float(pivots.get(key))
                if v is not None and v < entry:
                    lows.append(v)
            for v in [_safe_float(impulse.get("low")), _safe_float(ema200), _safe_float(ema50)]:
                if v is not None and v < entry:
                    lows.append(v)
            for zkey in ["major_support_zone", "long_lookback_major_support_zone"]:
                z = ms.get(zkey, {}) or {}
                if _zone_valid(z) and _safe_float(z.get("low")) is not None and float(z["low"]) < entry:
                    lows.append(float(z["low"]))
            if lows:
                lows = sorted(set(lows), reverse=True)
                for low in lows:
                    stop = low - atr_buffer
                    if stop < entry:
                        return float(stop)
            return float(entry * 0.97)

        if side == "SHORT":
            highs = []
            for key in ["recent_high", "previous_high", "highest"]:
                v = _safe_float(pivots.get(key))
                if v is not None and v > entry:
                    highs.append(v)
            for v in [_safe_float(impulse.get("high")), _safe_float(ema200), _safe_float(ema50)]:
                if v is not None and v > entry:
                    highs.append(v)
            for zkey in ["major_resistance_zone", "long_lookback_major_resistance_zone"]:
                z = ms.get(zkey, {}) or {}
                if _zone_valid(z) and _safe_float(z.get("high")) is not None and float(z["high"]) > entry:
                    highs.append(float(z["high"]))
            if highs:
                highs = sorted(set(highs))
                for high in highs:
                    stop = high + atr_buffer
                    if stop > entry:
                        return float(stop)
            return float(entry * 1.03)
    except Exception:
        return None
    return None


def _build_confluences(side: str, entry: float, pivots: Dict[str, Any], ema50: Optional[float], ema200: Optional[float], atr_val: Optional[float], context_bundle: Optional[Dict[str, Any]] = None, fib_label: str = "") -> List[str]:
    confluences: List[str] = []
    try:
        side = str(side).upper()
        tolerance = max(abs(entry) * CONTEXT_ZONE_TOLERANCE_PCT, (atr_val or 0) * CONTEXT_ZONE_TOLERANCE_ATR)
        if fib_label:
            confluences.append(f"Internal Fib {fib_label}")
        if side == "LONG":
            support = _nearest_level(pivots.get("lows", []), entry, "LONG")
            if support is not None and abs(entry - support) <= tolerance:
                confluences.append("Pivot support confluence")
        elif side == "SHORT":
            resistance = _nearest_level(pivots.get("highs", []), entry, "SHORT")
            if resistance is not None and abs(entry - resistance) <= tolerance:
                confluences.append("Pivot resistance confluence")
        if ema50 is not None and abs(entry - ema50) <= tolerance:
            confluences.append("EMA50 retest confluence")
        if ema200 is not None and abs(entry - ema200) <= tolerance:
            confluences.append("EMA200 retest confluence")
        confluences.extend(_context_zone_confluences(side, entry, atr_val, context_bundle))
        return list(dict.fromkeys(confluences))
    except Exception:
        return confluences

def _confluence_bucket(label: Any) -> str:
    """
    Group similar confluence labels so one price-area does not
    over-score the candidate as if it had many independent confirmations.

    Important:
    - This does NOT remove labels.
    - This does NOT reject candidates.
    - This only makes score more honest.
    """
    try:
        text = str(label or "").lower()

        if not text:
            return "other"

        # Elliott / template confirmation
        if "elliott wave template" in text or "wave template" in text:
            return "elliott_template"

        # Momentum / breakout evidence
        if (
            "momentum" in text
            or "breakout tf" in text
            or "breakout" in text
            or "breakdown" in text
            or "volume expansion" in text
        ):
            return "momentum_breakout"

        # EMA evidence: EMA50 + EMA200 should not count as two full independent zones.
        if "ema50" in text or "ema200" in text or "ema" in text:
            return "ema_retest"

        # Fib evidence: visible/elliott/internal/golden/deep all belong to fib evidence.
        if (
            "fib" in text
            or "golden" in text
            or "deep pullback" in text
            or "0.236" in text
            or "0.382" in text
            or "0.500" in text
            or "0.618" in text
            or "0.705" in text
            or "0.786" in text
        ):
            return "fib_zone"

        # Structure evidence: pivot/context/support/resistance are one structural area.
        if (
            "pivot" in text
            or "support" in text
            or "resistance" in text
            or "context" in text
            or "major_" in text
            or "major " in text
            or "zone" in text
        ):
            return "structure_zone"

        return "other"

    except Exception:
        return "other"


def _score_confluence_evidence(confluences: List[str]) -> Tuple[int, Dict[str, Any]]:
    """
    Soft confluence scoring.

    Old behavior:
      score += min(len(confluences) * 4, 16)

    Problem:
      EMA50 + EMA200 + Pivot + Fib can all sit in the same price zone,
      causing duplicate score inflation.

    Safe behavior:
      - Keep all labels for explanation.
      - Score by evidence buckets.
      - Apply only a soft duplicate penalty, capped to avoid making the
        system too strict or reducing good signals too much.
    """
    try:
        labels = list(dict.fromkeys([str(x) for x in (confluences or []) if str(x or "").strip()]))

        if not labels:
            return 0, {
                "raw_label_count": 0,
                "bucket_count": 0,
                "raw_score": 0,
                "bucket_score": 0,
                "final_score": 0,
                "duplicate_penalty": 0,
                "buckets": [],
                "mode": "NO_CONFLUENCE",
            }

        buckets = sorted(set(_confluence_bucket(x) for x in labels))

        raw_score = min(len(labels) * 4, 16)
        bucket_score = min(len(buckets) * 4, 16)

        # Conservative first version:
        # If labels are duplicated inside the same bucket, reduce score only softly.
        # This avoids making the system too strict.
        if raw_score <= 8:
            final_score = raw_score
            mode = "RAW_SMALL_COUNT"
        else:
            final_score = max(bucket_score, raw_score - 4)
            mode = "SOFT_BUCKET_CAP"

        final_score = int(max(0, min(16, final_score)))
        duplicate_penalty = int(max(0, raw_score - final_score))

        return final_score, {
            "raw_label_count": len(labels),
            "bucket_count": len(buckets),
            "raw_score": int(raw_score),
            "bucket_score": int(bucket_score),
            "final_score": int(final_score),
            "duplicate_penalty": duplicate_penalty,
            "buckets": buckets,
            "labels": labels,
            "mode": mode,
        }

    except Exception:
        # Fallback to old safe behavior if anything goes wrong.
        fallback = min(len(confluences or []) * 4, 16)
        return int(fallback), {
            "raw_label_count": len(confluences or []),
            "bucket_count": 0,
            "raw_score": int(fallback),
            "bucket_score": int(fallback),
            "final_score": int(fallback),
            "duplicate_penalty": 0,
            "buckets": [],
            "mode": "FALLBACK_OLD_SCORING",
        }

def _apply_realistic_tp2(
    side: str,
    entry: float,
    stop: float,
    tp1: float,
    raw_tp2: float,
    atr_val: Optional[float],
    user_tf: str,
    context_bundle: Optional[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    TP2 realism safety layer.

    Goal:
    - Do NOT replace good TP2 logic.
    - Keep raw TP2 when it is realistic.
    - Cap only obvious outliers:
        1) TP2 beyond timeframe ATR max distance
        2) strong structure/SR target appears before raw TP2
        3) TP2 beyond max 5R hard cap

    Important:
    - No minimum ATR rule is used here.
    - TP2 is never forced closer just because it is "small".
    - If cap makes RR too low, later validation will reject the candidate.
    """
    try:
        side = str(side or "").upper().strip()

        entry = float(entry)
        stop = float(stop)
        tp1 = float(tp1)
        raw_tp2 = float(raw_tp2)

        risk = abs(entry - stop)
        if risk <= 0:
            return raw_tp2, "raw_tp2_invalid_risk"

        if side not in ["LONG", "SHORT"]:
            return raw_tp2, "raw_tp2_invalid_side"

        # TP2 must be beyond TP1. If not, keep raw and let validation reject.
        if side == "LONG" and raw_tp2 <= tp1:
            return raw_tp2, "raw_tp2_not_beyond_tp1"

        if side == "SHORT" and raw_tp2 >= tp1:
            return raw_tp2, "raw_tp2_not_beyond_tp1"

        # Absolute 5R safety cap.
        max_5r_tp2 = (
            entry + (risk * TP2_R_MAX)
            if side == "LONG"
            else entry - (risk * TP2_R_MAX)
        )

        # Timeframe ATR cap.
        # This catches unrealistic targets without adding any "minimum ATR" restriction.
        atr_cap = raw_tp2
        if atr_val and atr_val > 0:
            max_atr = _timeframe_max_tp2_atr(user_tf)
            atr_cap = (
                entry + (float(atr_val) * max_atr)
                if side == "LONG"
                else entry - (float(atr_val) * max_atr)
            )

        # Structure targets from context:
        # LONG  -> nearest resistance / swing high / fib extension beyond TP1
        # SHORT -> nearest support / swing low / fib extension beyond TP1
        structure_targets = _context_targets(
            side=side,
            entry=entry,
            tp1=tp1,
            atr_val=atr_val,
            context_bundle=context_bundle,
        )

        chosen = raw_tp2
        reason = "raw_r_multiple"

        if side == "LONG":
            # Hard max cap.
            hard_cap = min(raw_tp2, max_5r_tp2)

            # ATR cap only matters if it is below raw TP2 but still beyond TP1.
            if atr_cap > tp1 and atr_cap < hard_cap:
                hard_cap = atr_cap
                reason = "atr_cap"

            # Structure cap only matters if a valid obstacle/target is before current cap.
            if structure_targets:
                for label, level in structure_targets:
                    level = _safe_float(level)
                    if level is None:
                        continue

                    # Must be a useful TP2 candidate, not inside TP1 area.
                    if tp1 < level < hard_cap:
                        chosen = float(level)
                        reason = f"structure_{label}"
                        break
                else:
                    chosen = hard_cap
                    if chosen != raw_tp2 and reason == "raw_r_multiple":
                        reason = "max5r_cap" if chosen == max_5r_tp2 else "cap"
            else:
                chosen = hard_cap
                if chosen != raw_tp2 and reason == "raw_r_multiple":
                    reason = "max5r_cap" if chosen == max_5r_tp2 else "cap"

            return float(chosen), reason

        if side == "SHORT":
            # For shorts, lower price = farther target.
            hard_cap = max(raw_tp2, max_5r_tp2)

            # ATR cap only matters if it is above raw TP2 but still beyond TP1 for short.
            if atr_cap < tp1 and atr_cap > hard_cap:
                hard_cap = atr_cap
                reason = "atr_cap"

            # Structure cap only matters if valid support/target is before current cap.
            if structure_targets:
                for label, level in structure_targets:
                    level = _safe_float(level)
                    if level is None:
                        continue

                    # Must be a useful TP2 candidate, not inside TP1 area.
                    if hard_cap < level < tp1:
                        chosen = float(level)
                        reason = f"structure_{label}"
                        break
                else:
                    chosen = hard_cap
                    if chosen != raw_tp2 and reason == "raw_r_multiple":
                        reason = "max5r_cap" if chosen == max_5r_tp2 else "cap"
            else:
                chosen = hard_cap
                if chosen != raw_tp2 and reason == "raw_r_multiple":
                    reason = "max5r_cap" if chosen == max_5r_tp2 else "cap"

            return float(chosen), reason

    except Exception:
        pass

    return raw_tp2, "raw_tp2_error"


def _template_priority(entry_type: str, confluences: List[str], htf_context: Dict[str, Any], user_tf: str) -> int:
    et = str(entry_type or "").lower()
    conf_text = " | ".join(confluences or []).lower()
    if "ema200_retest" in et:
        return 10
    if "context major" in conf_text and "fib" in conf_text:
        return 20
    if "0.618" in et or "golden" in conf_text:
        return 30
    if "0.500" in et or "0.705" in et:
        return 35
    if "ema50_retest" in et:
        return 40
    if "breakout_retest" in et or "breakdown_retest" in et:
        return 90
    return 60

def _apply_extra_sl_buffer(
    side: str,
    entry: float,
    stop: float,
    atr_val: Optional[float],
) -> float:
    """
    Add a small extra ATR gap to the generated backend SL.

    Important:
    - This is applied to the generated SL value.
    - TP1/TP2/RR must be calculated AFTER this final SL is created.
    - Max ATR validation still applies later.
    """
    try:
        side = str(side or "").upper()

        if atr_val is None or atr_val <= 0:
            return float(stop)

        extra = float(atr_val) * EXTRA_SL_ATR_BUFFER

        if side == "LONG":
            new_stop = float(stop) - extra
            if new_stop < float(entry):
                return new_stop

        if side == "SHORT":
            new_stop = float(stop) + extra
            if new_stop > float(entry):
                return new_stop

        return float(stop)

    except Exception:
        return float(stop)

def _apply_mtf_elliott_confirmation(
    candidate: Dict[str, Any],
    elliott_mtf_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach backend-owned Multi-Timeframe Elliott confirmation fields
    to Elliott template candidates.

    This does not create prices.
    It only grades whether local Elliott setup aligns with higher TF waves.
    """
    try:
        if not isinstance(candidate, dict):
            return candidate

        # Only apply to Elliott template detector candidates.
        if str(candidate.get("source") or "") != "elliott_wave_template_detector":
            return candidate

        if confirm_mtf_elliott_alignment is None:
            candidate.update({
                "mtf_elliott_alignment": "UNKNOWN",
                "mtf_wave_side": "UNKNOWN",
                "mtf_wave_phase": "UNKNOWN",
                "mtf_confirmation_score": 0,
                "mtf_confirmation_reason": "MTF Elliott module is not available.",
            })
            return candidate

        mtf_info = confirm_mtf_elliott_alignment(
            candidate=candidate,
            mtf_context=elliott_mtf_context or {},
        )

        candidate.update(mtf_info)
        return candidate

    except Exception as exc:
        candidate.update({
            "mtf_elliott_alignment": "UNKNOWN",
            "mtf_wave_side": "UNKNOWN",
            "mtf_wave_phase": "UNKNOWN",
            "mtf_confirmation_score": 0,
            "mtf_confirmation_reason": f"MTF Elliott confirmation failed: {exc}",
        })
        return candidate

def _apply_elliott_phase_context(
    candidate: Dict[str, Any],
    elliott_phase_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach full Elliott phase context to Elliott candidates.

    This does not create prices.
    It only tells planner whether the current wave phase supports
    Wave 2->3 / Wave 4->5 continuation, or whether it should be cautious.
    """
    try:
        if not isinstance(candidate, dict):
            return candidate

        if str(candidate.get("source") or "") != "elliott_wave_template_detector":
            return candidate

        ctx = elliott_phase_context if isinstance(elliott_phase_context, dict) else {}

        candidate["elliott_phase_alignment"] = ctx.get("elliott_phase_alignment", "UNKNOWN")
        candidate["elliott_phase_score"] = ctx.get("elliott_phase_score", 0)
        candidate["elliott_phase_permission"] = ctx.get("elliott_phase_permission", "WAIT")
        candidate["elliott_phase_reason"] = ctx.get("elliott_phase_reason", "")

        execution = ctx.get("execution", {}) or {}
        confirm = ctx.get("confirm", {}) or {}
        swing = ctx.get("swing", {}) or {}
        macro = ctx.get("macro", {}) or {}

        candidate["execution_elliott_phase"] = execution.get("elliott_phase", "UNKNOWN")
        candidate["confirm_elliott_phase"] = confirm.get("elliott_phase", "UNKNOWN")
        candidate["swing_elliott_phase"] = swing.get("elliott_phase", "UNKNOWN")
        candidate["macro_elliott_phase"] = macro.get("elliott_phase", "UNKNOWN")

        candidate["execution_elliott_side"] = execution.get("elliott_side", "UNKNOWN")
        candidate["confirm_elliott_side"] = confirm.get("elliott_side", "UNKNOWN")
        candidate["swing_elliott_side"] = swing.get("elliott_side", "UNKNOWN")
        candidate["macro_elliott_side"] = macro.get("elliott_side", "UNKNOWN")

        candidate["abc_status"] = execution.get("abc_status", "NONE")
        candidate["execution_phase_trade_permission"] = execution.get("trade_permission", "WAIT")

        return candidate

    except Exception as exc:
        candidate["elliott_phase_alignment"] = "UNKNOWN"
        candidate["elliott_phase_score"] = 0
        candidate["elliott_phase_permission"] = "WAIT"
        candidate["elliott_phase_reason"] = f"Elliott phase context failed: {exc}"
        return candidate

def _apply_cross_tf_risk_mode(
    candidate: Dict[str, Any],
    cross_tf_setups: Optional[List[Dict[str, Any]]] = None,
    user_tf: str = "",
) -> Dict[str, Any]:
    """
    Cross-timeframe risk classification.

    Important:
    - Does NOT reject candidate.
    - Does NOT reduce score.
    - Marks lower-TF signals as SCALP_ONLY / CAUTION when higher TF pullback can invalidate them.
    """
    try:
        if not isinstance(candidate, dict):
            return candidate

        setups = cross_tf_setups if isinstance(cross_tf_setups, list) else []

        candidate.setdefault("cross_tf_conflict", False)
        candidate.setdefault("cross_tf_risk_mode", "NORMAL")
        candidate.setdefault("cross_tf_conflict_type", "")
        candidate.setdefault("cross_tf_conflict_reason", "")

        side = str(candidate.get("side") or "").upper()
        entry = _safe_float(candidate.get("entry"))
        stop = _safe_float(candidate.get("stop_loss"))
        tf = _tf_key(user_tf)

        if side not in ["LONG", "SHORT"] or entry is None or stop is None:
            return candidate

        # Only lower TF should be marked scalp/caution.
        # 15m conflicting with 1H/4H = scalp/caution, not reject.
        lower_tf = tf in ["1m", "3m", "5m", "15m", "30m"]

        if not lower_tf:
            return candidate

        for htf in setups:
            htf_side = str(htf.get("side") or "").upper()
            htf_tf = str(htf.get("timeframe") or "").lower()
            htf_entry = _safe_float(htf.get("entry"))

            if htf_side != side or htf_entry is None:
                continue

            # LONG:
            # If higher-TF desired entry is below lower-TF stop,
            # the lower TF signal can die before HTF entry is reached.
            if side == "LONG" and htf_entry < stop:
                candidate["cross_tf_conflict"] = True
                candidate["cross_tf_risk_mode"] = "SCALP_ONLY"
                candidate["cross_tf_conflict_type"] = "HTF_DEEPER_PULLBACK_BELOW_LTF_SL"
                candidate["cross_tf_conflict_reason"] = (
                    f"{user_tf} LONG is valid only as a quick/scalp setup because "
                    f"{htf_tf} LONG entry {format_setup_price(htf_entry)} is below this setup SL "
                    f"{format_setup_price(stop)}. If price pulls back to the higher-TF entry, "
                    f"this lower-TF trade may be invalidated first."
                )
                return candidate

            # SHORT:
            # If higher-TF desired entry is above lower-TF stop,
            # the lower TF short can die before HTF entry is reached.
            if side == "SHORT" and htf_entry > stop:
                candidate["cross_tf_conflict"] = True
                candidate["cross_tf_risk_mode"] = "SCALP_ONLY"
                candidate["cross_tf_conflict_type"] = "HTF_DEEPER_PULLBACK_ABOVE_LTF_SL"
                candidate["cross_tf_conflict_reason"] = (
                    f"{user_tf} SHORT is valid only as a quick/scalp setup because "
                    f"{htf_tf} SHORT entry {format_setup_price(htf_entry)} is above this setup SL "
                    f"{format_setup_price(stop)}. If price pulls back to the higher-TF entry, "
                    f"this lower-TF trade may be invalidated first."
                )
                return candidate

        return candidate

    except Exception as exc:
        candidate["cross_tf_conflict"] = False
        candidate["cross_tf_risk_mode"] = "NORMAL"
        candidate["cross_tf_conflict_reason"] = f"Cross-TF risk check failed: {exc}"
        return candidate

def _attach_watch_trigger(
    candidate: Dict[str, Any],
    watch_type: str,
    side: str,
    level: float,
    level_label: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Attach structured watch trigger to WAIT candidate.

    This does NOT approve a trade.
    It only tells main.py / Firebase what level should be watched.
    """
    try:
        side = str(side or "").upper()
        watch_type = str(watch_type or "").upper()
        level = float(level)

        if watch_type == "WAIT_FOR_RECLAIM":
            condition = (
                f"Wait for candle body close above {format_setup_price(level)} "
                f"and then retest/hold above that level."
            )
            next_candidate_type = "LONG_BREAKOUT_RETEST"

        elif watch_type == "WAIT_FOR_RETEST_HOLD":
            condition = (
                f"Level {format_setup_price(level)} was reclaimed. "
                f"Wait for bullish retest hold from above."
            )
            next_candidate_type = "LONG_RETEST_HOLD"

        elif watch_type == "WAIT_FOR_BREAKDOWN":
            condition = (
                f"Wait for candle body close below {format_setup_price(level)} "
                f"and then retest/rejection below that level."
            )
            next_candidate_type = "SHORT_BREAKDOWN_RETEST"

        elif watch_type == "WAIT_FOR_RETEST_REJECTION":
            condition = (
                f"Level {format_setup_price(level)} was broken. "
                f"Wait for bearish retest rejection from below."
            )
            next_candidate_type = "SHORT_RETEST_REJECTION"

        else:
            condition = "Wait for clean structure confirmation before entry."
            next_candidate_type = "UNKNOWN"

        candidate.update({
            "watch_setup": True,
            "watch_type": watch_type,
            "watch_side": side,
            "watch_level": float(level),
            "watch_level_label": str(level_label or ""),
            "watch_condition": condition,
            "watch_next_candidate_type": next_candidate_type,
            "watch_recheck_needed": True,
            "watch_reason": reason,
        })

        return candidate

    except Exception:
        candidate["watch_setup"] = False
        return candidate

def _wave3_blocked_by_tp1_wall(
    candidate: Dict[str, Any],
    context_bundle: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Elliott Wave 2 -> Wave 3 TP1 wall protection.

    Important design:
    - This applies ONLY to Elliott Wave 2->3 candidates.
    - Normal/non-Elliott signals are not affected.
    - Medium/nearest S/R before TP1 should NOT kill Elliott Wave 3 setups.
    - Wave 3 is expected to break intermediate S/R levels.
    - Hard WAIT is used only when the wall is major/HTF and Elliott confirmation is weak.
    - MTF Elliott / phase conflict protections still run later in planner validation.
    """
    try:
        if not isinstance(candidate, dict):
            return False, ""

        source = str(candidate.get("source") or "")
        entry_type = str(candidate.get("entry_type") or "").upper()
        side = str(candidate.get("side") or "").upper()

        if source != "elliott_wave_template_detector":
            return False, ""

        if "WAVE_2_TO_WAVE_3" not in entry_type:
            return False, ""

        entry = _safe_float(candidate.get("entry"))
        tp1 = _safe_float(candidate.get("tp1"))

        if entry is None or tp1 is None:
            return False, ""

        wave3_status = str(candidate.get("wave3_status") or "").upper()
        wave3_confirmed = bool(candidate.get("wave3_confirmed"))

        # If Wave 3 is already confirmed by breakout/breakdown momentum,
        # do not hard-block it because of an intermediate TP1 wall.
        if wave3_status == "CONFIRMED" or wave3_confirmed:
            return False, ""

        atr_val = _safe_float(candidate.get("atr"), 0.0) or 0.0
        min_wall_distance = atr_val * TP1_WALL_MIN_DISTANCE_ATR if atr_val > 0 else 0.0

        cb = _context_dict(context_bundle)
        ms = cb.get("market_structure", {}) or {}
        derived = cb.get("derived", {}) or {}

        analysis_price = (
            _safe_float(ms.get("current_price"))
            or _safe_float(derived.get("analysis_price"))
            or _safe_float(candidate.get("current_price"))
        )

        live_price = (
            _safe_float(derived.get("execution_price"))
            or _safe_float(derived.get("live_price"))
            or _safe_float(candidate.get("current_price"))
        )

        wall_tolerance = max((atr_val * 0.15) if atr_val > 0 else 0.0, abs(entry) * 0.001)

        # --------------------------------------------------
        # Elliott confirmation strength.
        # This is what prevents fake Wave 3 from being allowed too easily.
        # --------------------------------------------------
        mtf_alignment = str(candidate.get("mtf_elliott_alignment") or "UNKNOWN").upper()
        mtf_score = _safe_float(candidate.get("mtf_confirmation_score"), 0.0) or 0.0

        phase_alignment = str(candidate.get("elliott_phase_alignment") or "UNKNOWN").upper()
        phase_score = _safe_float(candidate.get("elliott_phase_score"), 0.0) or 0.0
        phase_permission = str(candidate.get("elliott_phase_permission") or "WAIT").upper()
        execution_phase = str(candidate.get("execution_elliott_phase") or "UNKNOWN").upper()

        mtf_strong = mtf_alignment == "STRONG" and mtf_score >= 75
        mtf_medium_or_better = (
            mtf_alignment in ["STRONG", "MEDIUM", "ALIGNED"]
            and mtf_score >= 60
        )

        phase_good = (
            phase_alignment in ["ALIGNED", "SUPPORTIVE", "STRONG", "MEDIUM"]
            or phase_permission in ["ALLOW", "FULL_TRADE", "TRADE", "CAUTION"]
            or phase_score >= 60
        )

        weak_elliott_confirmation = (
            mtf_alignment in ["WEAK", "CONFLICT"]
            or mtf_score < 50
            or phase_alignment == "CONFLICT"
            or execution_phase in ["WAVE_5_ENDING", "ABC_C_ENDING"]
        )

        # If MTF is strong/medium and phase is not conflicting, do not let
        # normal/nearest SR kill Wave 3. Only very major walls can still wait.
        elliott_can_challenge_medium_wall = (
            mtf_strong
            or (mtf_medium_or_better and phase_good)
            or phase_score >= 70
        )

        def _wall_strength(label: str, level: float) -> str:
            """
            Classify S/R wall strength for Elliott Wave 3 only.

            nearest support/resistance = medium watch level
            major zone = strong
            long-lookback major zone = very strong HTF wall
            """
            text = str(label or "").lower()

            if "long lookback" in text:
                return "VERY_STRONG"

            if "major" in text:
                return "STRONG"

            return "MEDIUM"

        def _store_wall_watch(
            *,
            label: str,
            level: float,
            strength: str,
            watch_type: str,
            reason: str,
        ) -> None:
            """
            Attach non-blocking Elliott wall context.
            This keeps the signal alive but lets UI / AI explain the watch level.
            """
            try:
                candidate["elliott_tp1_wall_watch"] = True
                candidate["elliott_tp1_wall_level"] = float(level)
                candidate["elliott_tp1_wall_label"] = str(label or "")
                candidate["elliott_tp1_wall_strength"] = str(strength or "MEDIUM")
                candidate["elliott_tp1_wall_watch_type"] = str(watch_type or "")
                candidate["elliott_tp1_wall_reason"] = str(reason or "")
            except Exception:
                pass

        def _should_hard_wait_for_wall(strength: str) -> bool:
            """
            Only hard-wait on strong/very-strong walls when Elliott confirmation is not strong enough.
            """
            strength = str(strength or "").upper()

            if strength == "MEDIUM":
                return False

            if strength == "STRONG":
                # Strong wall is allowed if MTF/phase supports Wave 3.
                return not elliott_can_challenge_medium_wall

            if strength == "VERY_STRONG":
                # Long-lookback HTF wall still matters.
                # But if MTF is very strong, make it watch-only.
                return not mtf_strong

            return False

        if side == "LONG":
            resistance_levels = []

            nearest_resistance = _safe_float(ms.get("nearest_resistance"))
            if nearest_resistance is not None:
                resistance_levels.append(("nearest resistance", nearest_resistance))

            for zone_key in ["major_resistance_zone", "long_lookback_major_resistance_zone"]:
                z = ms.get(zone_key, {}) or {}
                if _zone_valid(z):
                    z_low = _safe_float(z.get("low"))
                    z_high = _safe_float(z.get("high"))

                    if z_low is not None:
                        resistance_levels.append((zone_key.replace("_", " "), z_low))
                    elif z_high is not None:
                        resistance_levels.append((zone_key.replace("_", " "), z_high))

            for label, level in resistance_levels:
                if not (entry < level < tp1):
                    continue

                if abs(level - entry) < min_wall_distance:
                    # Very close wall = micro level; do not block Elliott Wave 3.
                    continue

                strength = _wall_strength(label, level)

                closed_reclaimed = analysis_price is not None and analysis_price > level + wall_tolerance
                live_retesting = live_price is not None and live_price <= level + wall_tolerance

                if closed_reclaimed:
                    if live_retesting and _should_hard_wait_for_wall(strength):
                        reason = (
                            f"WAIT_FOR_RETEST_HOLD. Elliott Wave 2→3 LONG idea exists and "
                            f"{label} at {format_setup_price(level)} was already reclaimed by a closed candle, "
                            f"but live price is retesting that {strength.lower()} wall. "
                            f"Wait for price to hold above {format_setup_price(level)} with bullish rejection before entering."
                        )

                        _attach_watch_trigger(
                            candidate=candidate,
                            watch_type="WAIT_FOR_RETEST_HOLD",
                            side=side,
                            level=level,
                            level_label=label,
                            reason=reason,
                        )

                        return True, reason

                    watch_reason = (
                        f"Elliott Wave 2→3 LONG has {label} near {format_setup_price(level)} before TP1, "
                        f"but this wall is already reclaimed or MTF/phase context supports Wave 3. "
                        f"Treat it as a watch level, not a hard rejection."
                    )

                    _store_wall_watch(
                        label=label,
                        level=level,
                        strength=strength,
                        watch_type="RETEST_HOLD_WATCH",
                        reason=watch_reason,
                    )
                    continue

                # Wall not reclaimed yet.
                if _should_hard_wait_for_wall(strength):
                    reason = (
                        f"WAIT_FOR_RECLAIM. Elliott Wave 2→3 LONG idea exists, "
                        f"but {strength.lower()} {label} at {format_setup_price(level)} is between entry "
                        f"{format_setup_price(entry)} and TP1 {format_setup_price(tp1)}. "
                        f"MTF/phase confirmation is not strong enough to ignore this wall. "
                        f"Wait for candle body close above {format_setup_price(level)} and retest/hold before entering."
                    )

                    _attach_watch_trigger(
                        candidate=candidate,
                        watch_type="WAIT_FOR_RECLAIM",
                        side=side,
                        level=level,
                        level_label=label,
                        reason=reason,
                    )

                    return True, reason

                watch_reason = (
                    f"Elliott Wave 2→3 LONG has {label} near {format_setup_price(level)} before TP1. "
                    f"Because Elliott confirmation is acceptable, this is a watch level only. "
                    f"Wave 3 can still break intermediate resistance if momentum confirms."
                )

                _store_wall_watch(
                    label=label,
                    level=level,
                    strength=strength,
                    watch_type="RECLAIM_WATCH",
                    reason=watch_reason,
                )

                # Do not block. Keep checking only in case a later very strong wall exists.
                continue

        if side == "SHORT":
            support_levels = []

            nearest_support = _safe_float(ms.get("nearest_support"))
            if nearest_support is not None:
                support_levels.append(("nearest support", nearest_support))

            for zone_key in ["major_support_zone", "long_lookback_major_support_zone"]:
                z = ms.get(zone_key, {}) or {}
                if _zone_valid(z):
                    z_low = _safe_float(z.get("low"))
                    z_high = _safe_float(z.get("high"))

                    if z_high is not None:
                        support_levels.append((zone_key.replace("_", " "), z_high))
                    elif z_low is not None:
                        support_levels.append((zone_key.replace("_", " "), z_low))

            for label, level in support_levels:
                if not (tp1 < level < entry):
                    continue

                if abs(entry - level) < min_wall_distance:
                    # Very close wall = micro level; do not block Elliott Wave 3.
                    continue

                strength = _wall_strength(label, level)

                closed_broken = analysis_price is not None and analysis_price < level - wall_tolerance
                live_retesting = live_price is not None and live_price >= level - wall_tolerance

                if closed_broken:
                    if live_retesting and _should_hard_wait_for_wall(strength):
                        reason = (
                            f"WAIT_FOR_RETEST_REJECTION. Elliott Wave 2→3 SHORT idea exists and "
                            f"{label} at {format_setup_price(level)} was already broken by a closed candle, "
                            f"but live price is retesting that {strength.lower()} wall. "
                            f"Wait for price to reject below {format_setup_price(level)} before entering."
                        )

                        _attach_watch_trigger(
                            candidate=candidate,
                            watch_type="WAIT_FOR_RETEST_REJECTION",
                            side=side,
                            level=level,
                            level_label=label,
                            reason=reason,
                        )

                        return True, reason

                    watch_reason = (
                        f"Elliott Wave 2→3 SHORT has {label} near {format_setup_price(level)} before TP1, "
                        f"but this wall is already broken or MTF/phase context supports Wave 3. "
                        f"Treat it as a watch level, not a hard rejection."
                    )

                    _store_wall_watch(
                        label=label,
                        level=level,
                        strength=strength,
                        watch_type="RETEST_REJECTION_WATCH",
                        reason=watch_reason,
                    )
                    continue

                # Wall not broken yet.
                if _should_hard_wait_for_wall(strength):
                    reason = (
                        f"WAIT_FOR_BREAKDOWN. Elliott Wave 2→3 SHORT idea exists, "
                        f"but {strength.lower()} {label} at {format_setup_price(level)} is between entry "
                        f"{format_setup_price(entry)} and TP1 {format_setup_price(tp1)}. "
                        f"MTF/phase confirmation is not strong enough to ignore this wall. "
                        f"Wait for candle body close below {format_setup_price(level)} and retest/rejection before entering."
                    )

                    _attach_watch_trigger(
                        candidate=candidate,
                        watch_type="WAIT_FOR_BREAKDOWN",
                        side=side,
                        level=level,
                        level_label=label,
                        reason=reason,
                    )

                    return True, reason

                watch_reason = (
                    f"Elliott Wave 2→3 SHORT has {label} near {format_setup_price(level)} before TP1. "
                    f"Because Elliott confirmation is acceptable, this is a watch level only. "
                    f"Wave 3 can still break intermediate support if momentum confirms."
                )

                _store_wall_watch(
                    label=label,
                    level=level,
                    strength=strength,
                    watch_type="BREAKDOWN_WATCH",
                    reason=watch_reason,
                )

                # Do not block. Keep checking only in case a later very strong wall exists.
                continue

        return False, ""

    except Exception:
        return False, ""

def _make_wave_template_candidate(
    wave_template: Dict[str, Any],
    current_price: float,
    regime: Dict[str, Any],
    user_tf: str,
    atr_val: Optional[float],
    context_bundle: Optional[Dict[str, Any]],
    elliott_mtf_context: Optional[Dict[str, Any]] = None,
    elliott_phase_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert wave_template_detector output into setup_planner candidate format.
    Elliott candidates get highest priority, but still pass hard validation.
    """
    try:
        if not wave_template or not wave_template.get("valid"):
            return None

        side = str(wave_template.get("side") or "").upper()
        if side not in ["LONG", "SHORT"]:
            return None

        entry = _safe_float(wave_template.get("entry"))
        stop = _safe_float(wave_template.get("stop_loss"))
        tp1 = _safe_float(wave_template.get("tp1"))
        tp2 = _safe_float(wave_template.get("tp2"))

        if entry is None or stop is None or tp1 is None or tp2 is None:
            return None

        # Add extra SL gap to Elliott template SL as well.
        # This protects against small wick stop-outs near wave invalidation.
        stop = _apply_extra_sl_buffer(
            side=side,
            entry=entry,
            stop=stop,
            atr_val=atr_val,
        )

        if side == "LONG":
            risk = entry - stop
        else:
            risk = stop - entry

        if risk <= 0:
            return None

        # Keep TP1 strict at exact 1.5R after SL buffer adjustment.
        if side == "LONG":
            tp1 = entry + (risk * TP1_R)
        else:
            tp1 = entry - (risk * TP1_R)

        risk_pct = (risk / entry) * 100.0

        # Apply same TP2 realism safety layer to Elliott candidates too.
        # Elliott extension TP2 is valuable, but should not ignore ATR/HTF S/R obstacles.
        raw_tp2 = float(tp2)
        tp2, tp2_reason = _apply_realistic_tp2(
            side=side,
            entry=entry,
            stop=stop,
            tp1=tp1,
            raw_tp2=raw_tp2,
            atr_val=atr_val,
            user_tf=user_tf,
            context_bundle=context_bundle,
        )

        rr = abs(tp2 - entry) / risk if risk else 0.0
        distance_pct = abs(current_price - entry) / current_price * 100.0

        status, status_reason = _status_from_current_price(
            side=side,
            entry=entry,
            stop=stop,
            tp1=tp1,
            current_price=current_price,
        )

        template_name = str(wave_template.get("template") or "ELLIOTT_WAVE_TEMPLATE")
        confluences = wave_template.get("confluences") or []
        confluences = list(dict.fromkeys(
            ["Elliott Wave Template Confirmed"] + confluences
        ))

        # Wave 2 -> Wave 3 is highest priority.
        # Wave 4 -> Wave 5 is next.
        if "WAVE_2_TO_WAVE_3" in template_name.upper():
            priority = 1
        elif "WAVE_4_TO_WAVE_5" in template_name.upper():
            priority = 5
        else:
            priority = 8

        candidate = {
            "side": side,
            "entry": float(entry),
            "stop_loss": float(stop),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "raw_tp2": float(raw_tp2),
            "tp2_reason": tp2_reason,
            "tp2_r": None,
            "risk": float(risk),
            "risk_pct": float(risk_pct),
            "rr": float(rr),
            "entry_type": template_name,
            "source": "elliott_wave_template_detector",
            "template_priority": priority,
            "confluences": confluences,
            "distance_pct": float(distance_pct),
            "current_price": float(current_price),
            "atr": atr_val,
            "htf_context": {},
            "status": status,
            "status_reason": status_reason,
            "regime": regime.get("regime"),
            "market_bias": regime.get("market_bias"),
            "market_phase": regime.get("market_phase"),
            "setup_bias": regime.get("setup_bias"),
            "regime_confidence": regime.get("confidence"),
            "score": int(wave_template.get("confidence") or 80),
            "rejected": False,
            "reject_reason": "",
            "user_tf": user_tf,
            "context_bundle": context_bundle,

            # Extra Elliott debug fields
            "wave_template": wave_template,
            "wave_template_name": template_name,
            "wave_target": wave_template.get("wave_target"),
            "wave_confidence": wave_template.get("confidence"),
            "wave_status": wave_template.get("status"),
            "fib_used": wave_template.get("fib"),
            "wave_points": wave_template.get("wave_points"),
            "wave_logic": wave_template.get("logic"),
            "wave3_status": wave_template.get("wave3_status"),
            "wave3_confirmed": wave_template.get("wave3_confirmed"),
            "wave3_confirmation_reason": wave_template.get("wave3_confirmation_reason"),
            "wave3_break_level": wave_template.get("wave3_break_level"),

            # MTF Elliott fields will be attached immediately after candidate dict is built.
            "mtf_elliott_alignment": "UNKNOWN",
            "mtf_wave_side": "UNKNOWN",
            "mtf_wave_phase": "UNKNOWN",
            "mtf_confirmation_score": 0,
            "mtf_confirmation_reason": "",
            # Full Elliott phase context fields
            "elliott_phase_alignment": "UNKNOWN",
            "elliott_phase_score": 0,
            "elliott_phase_permission": "WAIT",
            "elliott_phase_reason": "",
            "execution_elliott_phase": "UNKNOWN",
            "confirm_elliott_phase": "UNKNOWN",
            "swing_elliott_phase": "UNKNOWN",
            "macro_elliott_phase": "UNKNOWN",
            "abc_status": "NONE",
        }

        candidate = _apply_mtf_elliott_confirmation(
            candidate=candidate,
            elliott_mtf_context=elliott_mtf_context,
        )

        candidate = _apply_elliott_phase_context(
            candidate=candidate,
            elliott_phase_context=elliott_phase_context,
        )

        return candidate

    except Exception:
        return None


def _detect_wave_template_candidates_for_debug(
    df_micro: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]],
    user_tf: str,
    symbol: str,
    current_price: float,
    regime: Dict[str, Any],
    atr_val: Optional[float],
    allowed_side: str = "ANY",
    include_opposite_debug: bool = True,
    elliott_mtf_context: Optional[Dict[str, Any]] = None,
    elliott_phase_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run Elliott wave detector even when trend_regime preferred_side is NONE.
    This helps debug whether Elliott setup was absent or blocked by planner gates.
    """
    out: List[Dict[str, Any]] = []

    try:
        if detect_wave_templates is None:
            return out

        wave_result = detect_wave_templates(
            df_micro=df_micro,
            context_bundle=context_bundle,
            user_tf=user_tf,
            symbol=symbol,
            include_wave5=True,
        )

        wave_templates = wave_result.get("templates") or []
        allowed_side = str(allowed_side or "ANY").upper()

        for wt in wave_templates:
            wt_side = str(wt.get("side") or "").upper()

            wc = _make_wave_template_candidate(
                wave_template=wt,
                current_price=current_price,
                regime=regime,
                user_tf=user_tf,
                atr_val=atr_val,
                context_bundle=context_bundle,
                elliott_mtf_context=elliott_mtf_context,
                elliott_phase_context=elliott_phase_context,
            )

            if not wc:
                continue

            if allowed_side in ["LONG", "SHORT"] and wt_side != allowed_side:
                wc["rejected"] = True
                wc["reject_reason"] = (
                    f"Elliott candidate side {wt_side} does not match planner allowed side {allowed_side}."
                )

                if include_opposite_debug:
                    out.append(wc)

                continue

            out.append(wc)

        return out

    except Exception:
        return out

def _make_candidate(side: str, entry: float, stop: float, current_price: float, entry_type: str, source: str, confluences: List[str], regime: Dict[str, Any], user_tf: str = "15m", atr_val: Optional[float] = None, htf_context: Optional[Dict[str, Any]] = None, context_bundle: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        side = str(side).upper()
        entry = float(entry)
        stop = float(stop)
        current_price = float(current_price)
        if entry <= 0 or stop <= 0 or current_price <= 0:
            return None
        # --------------------------------------------------
        # Auto-widen SL if it is tighter than minimum ATR rule.
        # Example:
        #   MIN_STOP_ATR_MULTIPLIER = 1.20
        #   If SL distance is only 0.90 ATR, widen it to 1.20 ATR.
        # TP1 and TP2 are calculated AFTER this, so targets match the adjusted SL.
        # --------------------------------------------------
        if atr_val and atr_val > 0:
            min_stop_distance = atr_val * MIN_STOP_ATR_MULTIPLIER

            if side == "LONG":
                current_stop_distance = entry - stop
                if current_stop_distance > 0 and current_stop_distance < min_stop_distance:
                    stop = entry - min_stop_distance

            elif side == "SHORT":
                current_stop_distance = stop - entry
                if current_stop_distance > 0 and current_stop_distance < min_stop_distance:
                    stop = entry + min_stop_distance

        # Add final extra SL buffer after minimum-SL auto-widen.
        # TP1/TP2/RR are calculated after this final SL.
        stop = _apply_extra_sl_buffer(
            side=side,
            entry=entry,
            stop=stop,
            atr_val=atr_val,
        )

        if side == "LONG":
            if stop >= entry:
                return None
            risk = entry - stop
            tp1 = entry + (risk * TP1_R)

        elif side == "SHORT":
            if stop <= entry:
                return None
            risk = stop - entry
            tp1 = entry - (risk * TP1_R)

        else:
            return None
        if risk <= 0:
            return None

        risk_pct = (risk / entry) * 100.0
        tp2_r = _dynamic_tp2_r(user_tf=user_tf, regime=regime, confluences=confluences, risk_pct=risk_pct)
        raw_tp2 = entry + (risk * tp2_r) if side == "LONG" else entry - (risk * tp2_r)
        tp2, tp2_reason = _apply_realistic_tp2(side, entry, stop, tp1, raw_tp2, atr_val, user_tf, context_bundle)
        rr = abs(tp2 - entry) / risk if risk else 0.0
        distance_pct = abs(current_price - entry) / current_price * 100.0

        status, status_reason = _status_from_current_price(side, entry, stop, tp1, current_price)
        priority = _template_priority(entry_type, confluences, htf_context or {}, user_tf)

        return {
            "side": side,
            "entry": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "raw_tp2": raw_tp2,
            "tp2_reason": tp2_reason,
            "tp2_r": tp2_r,
            "risk": risk,
            "risk_pct": risk_pct,
            "rr": rr,
            "entry_type": entry_type,
            "source": source,
            "template_priority": priority,
            "confluences": confluences,
            "distance_pct": distance_pct,
            "current_price": current_price,
            "atr": atr_val,
            "htf_context": htf_context or {},
            "status": status,
            "status_reason": status_reason,
            "regime": regime.get("regime"),
            "market_bias": regime.get("market_bias"),
            "market_phase": regime.get("market_phase"),
            "setup_bias": regime.get("setup_bias"),
            "regime_confidence": regime.get("confidence"),
            "score": 0,
            "rejected": False,
            "reject_reason": "",
            "user_tf": user_tf,
            "context_bundle": context_bundle,
        }
    except Exception:
        return None

def _reject_impractical_15m_candidate(c: Dict[str, Any], user_tf: str) -> Dict[str, Any]:
    """
    15m practical-quality filter.

    Purpose:
    - Prevent 15m signals from becoming tiny 1m-style scalps.
    - Applies ONLY to user selected 15m.
    - Does NOT affect 1h / 4h / 1d signals.

    Why:
    RR can look good even when absolute SL/TP distance is too small.
    A 15m signal needs enough room to survive normal wick/noise/spread.
    """
    try:
        tf = _tf_key(user_tf)

        # IMPORTANT: only affect 15m.
        if tf != "15m":
            return c

        if not isinstance(c, dict) or c.get("rejected"):
            return c

        side = str(c.get("side") or "").upper()
        entry = _safe_float(c.get("entry"))
        stop = _safe_float(c.get("stop_loss"))
        tp1 = _safe_float(c.get("tp1"))
        tp2 = _safe_float(c.get("tp2"))
        atr_val = _safe_float(c.get("atr"), 0.0) or 0.0
        risk_pct = _safe_float(c.get("risk_pct"), 0.0) or 0.0

        if side not in ["LONG", "SHORT"]:
            return c

        if entry is None or stop is None or tp1 is None or tp2 is None or entry <= 0:
            return c

        risk = abs(entry - stop)
        tp1_distance = abs(tp1 - entry)
        tp2_distance = abs(tp2 - entry)

        risk_atr = (risk / atr_val) if atr_val > 0 else None
        tp1_distance_pct = (tp1_distance / entry) * 100.0
        tp2_distance_pct = (tp2_distance / entry) * 100.0

        mtf_alignment = str(c.get("mtf_elliott_alignment") or "UNKNOWN").upper()
        mtf_score = _safe_float(c.get("mtf_confirmation_score"), 0.0) or 0.0

        # --------------------------------------------------
        # Rule 1: 15m SL too tight.
        # Use AND so we don't reject low-price/high-ATR situations too aggressively.
        # --------------------------------------------------
        if risk_pct < 0.60 and (risk_atr is None or risk_atr < 1.35):
            c["rejected"] = True
            c["reject_reason"] = (
                "15m practical filter: setup is too tight/noisy. "
                f"Risk {risk_pct:.2f}%"
                + (f" / {risk_atr:.2f} ATR" if risk_atr is not None else "")
                + " is below practical 15m threshold."
            )
            c["status"] = "WAIT_IMPRACTICAL_15M"
            return c

        # --------------------------------------------------
        # Rule 2: TP1 too close.
        # This blocks trades that have good RR but tiny practical reward.
        # --------------------------------------------------
        if tp1_distance_pct < 0.75:
            c["rejected"] = True
            c["reject_reason"] = (
                "15m practical filter: TP1 is too close for a meaningful 15m setup. "
                f"TP1 distance {tp1_distance_pct:.2f}% is below 0.75% threshold."
            )
            c["status"] = "WAIT_IMPRACTICAL_15M"
            return c

        # --------------------------------------------------
        # Rule 3: TP2 too close.
        # 15m can be fast, but TP2 should still be meaningful.
        # --------------------------------------------------
        if tp2_distance_pct < 1.05:
            c["rejected"] = True
            c["reject_reason"] = (
                "15m practical filter: TP2 is too close for a meaningful 15m setup. "
                f"TP2 distance {tp2_distance_pct:.2f}% is below 1.05% threshold."
            )
            c["status"] = "WAIT_IMPRACTICAL_15M"
            return c

        # --------------------------------------------------
        # Rule 4: MTF weak + tiny risk.
        # If higher timeframe confirmation is weak, avoid tiny local trades.
        # UNKNOWN is not blocked here, only explicit WEAK with low score / small risk.
        # --------------------------------------------------
        if mtf_alignment == "WEAK" and mtf_score < 45 and risk_pct < 0.80:
            c["rejected"] = True
            c["reject_reason"] = (
                "15m practical filter: MTF Elliott support is weak and setup is too small. "
                f"MTF={mtf_alignment} score={mtf_score:.0f}, risk={risk_pct:.2f}%."
            )
            c["status"] = "WAIT_IMPRACTICAL_15M"
            return c

        return c

    except Exception as exc:
        # Never break planner because of this filter.
        try:
            c["practical_15m_filter_error"] = str(exc)
        except Exception:
            pass
        return c

def _reject_invalid_candidate(c: Dict[str, Any], user_tf: str) -> Dict[str, Any]:
    try:
        if c.get("status") == "INVALIDATED":
            c["rejected"] = True
            c["reject_reason"] = c.get("status_reason") or "Already invalidated."
            return c

        side = str(c.get("side") or "").upper()
        entry = float(c.get("entry") or 0)
        stop = float(c.get("stop_loss") or 0)
        current_price = float(c.get("current_price") or 0)
        tp1 = float(c.get("tp1") or 0)
        tp2 = float(c.get("tp2") or 0)
        risk_pct = float(c.get("risk_pct") or 999)
        rr = float(c.get("rr") or 0)
        distance_pct = float(c.get("distance_pct") or 999)
        atr_val = _safe_float(c.get("atr"), 0.0) or 0.0
        max_entry_distance = _timeframe_max_entry_distance_pct(user_tf)
        htf_context = c.get("htf_context") or {}

        if entry <= 0 or stop <= 0 or current_price <= 0 or tp1 <= 0 or tp2 <= 0:
            c["rejected"] = True
            c["reject_reason"] = "Candidate has invalid price fields."
            return c

        # --------------------------------------------------
        # 15m practical filter.
        # Only affects user-selected 15m.
        # Keeps 1h / 4h behavior unchanged.
        # --------------------------------------------------
        c = _reject_impractical_15m_candidate(c, user_tf=user_tf)
        if c.get("rejected"):
            return c

        # --------------------------------------------------
        # Elliott Wave 3 wall protection.
        # If TP1 is blocked by nearby resistance/support,
        # do not show a fresh trade yet. Ask user to wait for reclaim/breakdown.
        # --------------------------------------------------
        blocked_by_wall, wall_reason = _wave3_blocked_by_tp1_wall(
            candidate=c,
            context_bundle=c.get("context_bundle"),
        )

        if blocked_by_wall:
            c["rejected"] = True
            c["reject_reason"] = wall_reason

            reason_upper = str(wall_reason or "").upper()

            if "WAIT_FOR_RETEST_HOLD" in reason_upper:
                c["status"] = "WAIT_FOR_RETEST_HOLD"
            elif "WAIT_FOR_RETEST_REJECTION" in reason_upper:
                c["status"] = "WAIT_FOR_RETEST_REJECTION"
            elif "WAIT_FOR_RECLAIM" in reason_upper:
                c["status"] = "WAIT_FOR_RECLAIM"
            elif "WAIT_FOR_BREAKDOWN" in reason_upper:
                c["status"] = "WAIT_FOR_BREAKDOWN"
            else:
                c["status"] = "WAIT_FOR_RECLAIM" if side == "LONG" else "WAIT_FOR_BREAKDOWN"

            c["status_reason"] = wall_reason
            return c
        # --------------------------------------------------
        # MTF Elliott confirmation protection.
        # If local Elliott setup conflicts with higher timeframe waves,
        # reject it before normal price validation.
        # --------------------------------------------------
        if str(c.get("source") or "") == "elliott_wave_template_detector":
            mtf_alignment = str(c.get("mtf_elliott_alignment") or "UNKNOWN").upper()
            mtf_score = _safe_float(c.get("mtf_confirmation_score"), 0.0) or 0.0

            if mtf_alignment == "CONFLICT":
                c["rejected"] = True
                c["reject_reason"] = (
                    "MTF Elliott conflict: local Elliott setup is against higher-timeframe wave structure. "
                    + str(c.get("mtf_confirmation_reason") or "")
                ).strip()
                return c

            # Keep this soft for now. Do not hard reject normal WEAK setups yet.
            if mtf_alignment == "WEAK" and mtf_score < 35:
                c["rejected"] = True
                c["reject_reason"] = (
                    "Very weak MTF Elliott confirmation: local Elliott setup lacks higher-timeframe support. "
                    + str(c.get("mtf_confirmation_reason") or "")
                ).strip()
                return c
        # --------------------------------------------------
        # Elliott phase permission protection.
        # This uses full Elliott phase context:
        # 1-2-3-4-5 / A-B-C.
        #
        # First version is intentionally soft:
        # - CONFLICT = reject
        # - WAVE_5_ENDING = caution unless other validations fail
        # - ABC_C_ENDING = not traded here yet; future ABC setup will use it
        # --------------------------------------------------
        if str(c.get("source") or "") == "elliott_wave_template_detector":
            phase_alignment = str(c.get("elliott_phase_alignment") or "UNKNOWN").upper()
            phase_permission = str(c.get("elliott_phase_permission") or "WAIT").upper()
            execution_phase = str(c.get("execution_elliott_phase") or "UNKNOWN").upper()
            entry_type = str(c.get("entry_type") or "").upper()

            if phase_alignment == "CONFLICT":
                c["rejected"] = True
                c["reject_reason"] = (
                    "Elliott phase conflict: local Elliott setup is not aligned with higher timeframe phase context. "
                    + str(c.get("elliott_phase_reason") or "")
                ).strip()
                return c

            # Avoid late Wave 2->3 entries if execution phase already looks like Wave 5 ending.
            if "WAVE_2_TO_WAVE_3" in entry_type and execution_phase == "WAVE_5_ENDING":
                c["rejected"] = True
                c["reject_reason"] = (
                    "Elliott phase caution: candidate is Wave 2->3, but execution phase appears to be Wave 5 ending. "
                    "Avoid late continuation entries."
                )
                return c

            # ABC C-ending is useful, but we will add a separate ABC setup later.
            # Do not allow normal Wave 2->3 template to pretend it is an ABC ending trade.
            if execution_phase == "ABC_C_ENDING" and "WAVE_2_TO_WAVE_3" in entry_type:
                c["phase_caution"] = True
                c["phase_caution_reason"] = (
                    "Execution phase looks like ABC C-ending. This is a caution for Wave 2->3 continuation; "
                    "future ABC ending setup should handle this separately."
                )

        # Pending-limit side validation
        if side == "LONG" and entry > current_price:
            c["rejected"] = True
            c["reject_reason"] = "Invalid limit setup: LONG entry is above current price."
            return c
        if side == "SHORT" and entry < current_price:
            c["rejected"] = True
            c["reject_reason"] = "Invalid limit setup: SHORT entry is below current price."
            return c

        # No-chase protections
        if side == "LONG" and current_price >= tp1:
            c["rejected"] = True
            c["reject_reason"] = "No chase: current price is already at/above TP1 for this long setup."
            return c
        if side == "SHORT" and current_price <= tp1:
            c["rejected"] = True
            c["reject_reason"] = "No chase: current price is already at/below TP1 for this short setup."
            return c
        if side == "LONG" and current_price > entry and tp1 > entry:
            if (current_price - entry) / (tp1 - entry) >= NO_CHASE_TP1_PROGRESS:
                c["rejected"] = True
                c["reject_reason"] = "No chase: price already moved more than 70% from entry toward TP1."
                return c
        if side == "SHORT" and current_price < entry and tp1 < entry:
            if (entry - current_price) / (entry - tp1) >= NO_CHASE_TP1_PROGRESS:
                c["rejected"] = True
                c["reject_reason"] = "No chase: price already moved more than 70% from entry toward TP1."
                return c

        # HTF alignment for lower timeframes
        if htf_context.get("enabled"):
            anchor = _safe_float(htf_context.get("htf_anchor_level"))
            if anchor is not None and atr_val > 0:
                htf_buffer = atr_val * HTF_EMA_BUFFER_ATR_MULTIPLIER
                if side == "SHORT" and entry < anchor - htf_buffer:
                    c["rejected"] = True
                    c["reject_reason"] = f"HTF alignment rejected: SHORT entry is below HTF resistance/EMA zone ({anchor:.8f})."
                    return c
                if side == "LONG" and entry > anchor + htf_buffer:
                    c["rejected"] = True
                    c["reject_reason"] = f"HTF alignment rejected: LONG entry is above HTF support/EMA zone ({anchor:.8f})."
                    return c

        # SL realism: not too tight, not too wide.
        if atr_val > 0:
            stop_distance = abs(entry - stop)
            if stop_distance < atr_val * MIN_STOP_ATR_MULTIPLIER:
                c["rejected"] = True
                c["reject_reason"] = f"Stop loss is too tight: {stop_distance:.8f} is less than {MIN_STOP_ATR_MULTIPLIER:.2f} ATR."
                return c
            max_stop = atr_val * _timeframe_max_stop_atr(user_tf)
            if stop_distance > max_stop:
                c["rejected"] = True
                c["reject_reason"] = f"Stop loss is too wide: {stop_distance:.8f} exceeds {_timeframe_max_stop_atr(user_tf):.2f} ATR for {user_tf}."
                return c

        # Risk / RR / distance filters
        if risk_pct > MAX_RISK_PCT:
            c["rejected"] = True
            c["reject_reason"] = f"Risk {risk_pct:.2f}% exceeds max {MAX_RISK_PCT:.2f}%."
            return c
        min_rr = _min_rr_for_timeframe(user_tf)
        if rr < min_rr:
            c["rejected"] = True
            c["reject_reason"] = f"RR {rr:.2f} is below minimum {min_rr:.2f} after realistic TP2 cap."
            return c
        if distance_pct > max_entry_distance:
            c["rejected"] = True
            c["reject_reason"] = f"Entry is {distance_pct:.2f}% away from current price; max allowed for {user_tf} is {max_entry_distance:.2f}%."
            return c

        return c
    except Exception as e:
        c["rejected"] = True
        c["reject_reason"] = f"Candidate validation error: {str(e)}"
        return c


def _score_candidate(c: Dict[str, Any], regime: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug score only. Selection is NOT score-first anymore.
    """
    try:
        if c.get("rejected"):
            c["score"] = 0
            return c
        score = 50.0
        risk_pct = float(c.get("risk_pct") or 999)
        rr = float(c.get("rr") or 0)
        confluences = c.get("confluences") or []
        priority = int(c.get("template_priority") or 99)
        confidence = float(regime.get("confidence") or 0)
        if priority <= 10:
            score += 18
        elif priority <= 20:
            score += 15
        elif priority <= 40:
            score += 10
        elif priority >= 90:
            score -= 8
        if risk_pct <= 2.0:
            score += 10
        elif risk_pct <= 3.5:
            score += 6
        elif risk_pct > 4.0:
            score -= 8
        if rr >= 4.0:
            score += 10
        elif rr >= 3.0:
            score += 6
        confluence_score, confluence_score_details = _score_confluence_evidence(confluences)
        score += confluence_score

        # Debug/visibility only. Does not affect entry/SL/TP.
        c["confluence_score"] = confluence_score
        c["confluence_score_details"] = confluence_score_details
        if confidence >= 0.80:
            score += 8
        elif confidence >= 0.65:
            score += 4
        if str(c.get("tp2_reason") or "").startswith("raw"):
            score -= 3
        # MTF Elliott quality boost/penalty.
        if str(c.get("source") or "") == "elliott_wave_template_detector":
            mtf_alignment = str(c.get("mtf_elliott_alignment") or "UNKNOWN").upper()
            mtf_score = _safe_float(c.get("mtf_confirmation_score"), 0.0) or 0.0

            if mtf_alignment == "STRONG":
                score += 10
            elif mtf_alignment == "MEDIUM":
                score += 5
            elif mtf_alignment == "WEAK":
                score -= 2
            elif mtf_alignment == "CONFLICT":
                score -= 25

            # Small score nudge from actual MTF score.
            score += min(max(mtf_score, 0), 100) * 0.03

        # Full Elliott phase context boost/penalty.
        if str(c.get("source") or "") == "elliott_wave_template_detector":
            phase_alignment = str(c.get("elliott_phase_alignment") or "UNKNOWN").upper()
            phase_score = _safe_float(c.get("elliott_phase_score"), 0.0) or 0.0
            execution_phase = str(c.get("execution_elliott_phase") or "UNKNOWN").upper()

            if phase_alignment == "STRONG":
                score += 8
            elif phase_alignment == "MEDIUM":
                score += 4
            elif phase_alignment == "WEAK":
                score -= 1
            elif phase_alignment == "CONFLICT":
                score -= 20

            if execution_phase in ["WAVE_2_PULLBACK", "WAVE_4_PULLBACK"]:
                score += 5
            elif execution_phase == "WAVE_5_ENDING":
                score -= 8
            elif execution_phase in ["ABC_CORRECTION", "ABC_C_ENDING"]:
                score -= 4

            score += min(max(phase_score, 0), 100) * 0.02

        c["score"] = int(round(_clamp(score, 0, 96)))
        return c
    except Exception:
        c["score"] = 0
        return c


# ==========================================================
# Candidate generation and selection
# ==========================================================

def generate_backend_candidates(
    df_micro: pd.DataFrame,
    backend_trend_regime: Dict[str, Any],
    df_live: Optional[pd.DataFrame] = None,
    user_tf: str = "15m",
    df_confirm: Optional[pd.DataFrame] = None,
    df_swing: Optional[pd.DataFrame] = None,
    context_bundle: Optional[Dict[str, Any]] = None,
    symbol: str = "",
    elliott_mtf_context: Optional[Dict[str, Any]] = None,
    elliott_phase_context: Optional[Dict[str, Any]] = None,
    momentum_breakout_context: Optional[Dict[str, Any]] = None,
    cross_tf_setups: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    d = _normalize_ohlc(df_micro)
    live = _normalize_ohlc(df_live) if df_live is not None else pd.DataFrame()
    try:
        if d.empty or len(d) < 40:
            return candidates
        side = str(backend_trend_regime.get("preferred_side") or "NONE").upper()
        blocked_side = str(backend_trend_regime.get("blocked_side") or "NONE").upper()
        confidence = float(backend_trend_regime.get("confidence") or 0)

        current_price = float(live["Close"].iloc[-1]) if not live.empty else float(d["Close"].iloc[-1])
        current_price = _context_execution_price(context_bundle, current_price) or current_price

        atr_val = _context_atr(context_bundle, _atr(d, 14))
        ema50 = _ema(d["Close"], 50)
        ema200 = _ema(d["Close"], 200)

        # --------------------------------------------------
        # Always run Elliott wave detector first.
        # Even when preferred_side is NONE, this tells us:
        # - no Elliott template exists, OR
        # - template exists but was blocked/rejected.
        # --------------------------------------------------
        wave_debug_candidates = _detect_wave_template_candidates_for_debug(
            df_micro=d,
            context_bundle=context_bundle,
            user_tf=user_tf,
            symbol=symbol,
            current_price=current_price,
            regime=backend_trend_regime,
            atr_val=atr_val,
            allowed_side=side if side in ["LONG", "SHORT"] else "ANY",
            include_opposite_debug=True,
            elliott_mtf_context=elliott_mtf_context,
            elliott_phase_context=elliott_phase_context,
        )

        # If planner has no clear side, return wave candidates as debug only.
        # Do not select trade yet unless we later add controlled Elliott override.
        if side not in ["LONG", "SHORT"]:
            final_debug = []
            for c in wave_debug_candidates:
                c = _reject_invalid_candidate(c, user_tf=user_tf)
                c = _apply_cross_tf_risk_mode(
                    c,
                    cross_tf_setups=cross_tf_setups,
                    user_tf=user_tf
                )
                c = _score_candidate(c, backend_trend_regime)
                final_debug.append(c)
            return final_debug

        if confidence < MIN_REGIME_CONFIDENCE:
            final_debug = []
            for c in wave_debug_candidates:
                c["rejected"] = True
                c["reject_reason"] = (
                    c.get("reject_reason")
                    or f"Regime confidence {confidence:.2f} is below minimum {MIN_REGIME_CONFIDENCE:.2f}."
                )
                c = _score_candidate(c, backend_trend_regime)
                final_debug.append(c)
            return final_debug

        if side == blocked_side:
            final_debug = []
            for c in wave_debug_candidates:
                c["rejected"] = True
                c["reject_reason"] = (
                    c.get("reject_reason")
                    or f"Planner side {side} is blocked by backend trend regime."
                )
                c = _score_candidate(c, backend_trend_regime)
                final_debug.append(c)
            return final_debug

        pivots = _find_pivots(d, window=DEFAULT_SWING_WINDOW, lookback=DEFAULT_LOOKBACK)
        impulse = _impulse_range(d, side=side, lookback=DEFAULT_LOOKBACK)
        htf_context = _derive_htf_context(df_confirm=df_confirm, side=side, user_tf=user_tf)

        # Add Elliott wave candidates first.
        # Valid same-side Elliott candidates can be selected.
        # Rejected/opposite Elliott candidates stay in debug only.
        for wc in wave_debug_candidates:
            candidates.append(wc)

        if not impulse.get("valid"):
            final_candidates = []
            seen_keys = set()

            for c in candidates:
                try:
                    key = (
                        c.get("side"),
                        round(float(c.get("entry")), 8),
                        round(float(c.get("stop_loss")), 8),
                        c.get("entry_type"),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    c = _reject_invalid_candidate(c, user_tf=user_tf)
                    c = _apply_cross_tf_risk_mode(
                        c,
                        cross_tf_setups=cross_tf_setups,
                        user_tf=user_tf
                    )
                    c = _score_candidate(c, backend_trend_regime)
                    final_candidates.append(c)

                except Exception:
                    continue

            final_candidates.sort(
                key=lambda x: (
                    0 if not x.get("rejected") else 1,
                    int(x.get("template_priority") or 99),
                    0 if str(x.get("mtf_elliott_alignment") or "").upper() == "STRONG" else
                    1 if str(x.get("mtf_elliott_alignment") or "").upper() == "MEDIUM" else
                    2 if str(x.get("mtf_elliott_alignment") or "").upper() == "WEAK" else
                    3,
                    -float(x.get("rr") or 0),
                    float(x.get("risk_pct") or 999),
                )
            )
            return final_candidates
        rng = _safe_float(impulse.get("range"))
        high = _safe_float(impulse.get("high"))
        low = _safe_float(impulse.get("low"))
        if rng is None or high is None or low is None or rng <= 0:
            return candidates

        def add_candidate(entry: float, entry_type: str, source: str, base_confluences: Optional[List[str]] = None) -> None:
            if entry <= 0:
                return
            stop = _candidate_stop_for_side(side, entry, pivots, atr_val, impulse, ema50=ema50, ema200=ema200, context_bundle=context_bundle)
            if stop is None:
                return
            confluences = base_confluences[:] if base_confluences else []
            confluences.extend(_build_confluences(side, entry, pivots, ema50, ema200, atr_val, context_bundle=context_bundle))
            confluences = list(dict.fromkeys(confluences))
            c = _make_candidate(side=side, entry=entry, stop=stop, current_price=current_price, entry_type=entry_type, source=source, confluences=confluences, regime=backend_trend_regime, user_tf=user_tf, atr_val=atr_val, htf_context=htf_context, context_bundle=context_bundle)
            if c:
                candidates.append(c)

        # Fib pullbacks from internal impulse range.
        for fib in [0.500, 0.618, 0.705, 0.786]:
            entry = high - (rng * fib) if side == "LONG" else low + (rng * fib)
            add_candidate(entry, f"fib_{fib:.3f}_pullback", "impulse_fibonacci_pullback", [f"Internal Fib {fib:.3f}"])

        # EMA retests.
        for ema_name, ema_val in [("ema200_retest", ema200), ("ema50_retest", ema50)]:
            if ema_val is None or ema_val <= 0:
                continue
            entry = float(ema_val)
            if side == "LONG" and entry >= current_price * 1.01:
                continue
            if side == "SHORT" and entry <= current_price * 0.99:
                continue
            add_candidate(entry, ema_name, "ema_retest", [ema_name.upper()])

        # Breakout/breakdown retest last priority.
        if side == "LONG":
            recent_high = _safe_float(pivots.get("recent_high"))
            if recent_high is not None:
                add_candidate(recent_high, "bullish_breakout_retest", "recent_swing_high_retest", ["Bullish breakout/retest level"])
        elif side == "SHORT":
            recent_low = _safe_float(pivots.get("recent_low"))
            if recent_low is not None:
                add_candidate(recent_low, "bearish_breakdown_retest", "recent_swing_low_retest", ["Bearish breakdown/retest level"])

        # --------------------------------------------------
        # ABC C-ending candidate.
        # Conservative correction-ending setup:
        # - Uses Elliott phase context
        # - Requires support/resistance/rejection/confluence
        # - Still passes normal backend validation
        # --------------------------------------------------
        try:
            if make_abc_c_ending_idea is not None:
                abc_idea = make_abc_c_ending_idea(
                    side=side,
                    current_price=current_price,
                    pivots=pivots,
                    ema50=ema50,
                    ema200=ema200,
                    atr_val=atr_val,
                    backend_trend_regime=backend_trend_regime,
                    user_tf=user_tf,
                    context_bundle=context_bundle,
                    elliott_phase_context=elliott_phase_context,
                )

                if abc_idea and abc_idea.get("valid"):
                    abc_candidate = _make_candidate(
                        side=abc_idea.get("side"),
                        entry=abc_idea.get("entry"),
                        stop=abc_idea.get("stop_loss"),
                        current_price=current_price,
                        entry_type=abc_idea.get("entry_type"),
                        source=abc_idea.get("source", "elliott_phase_detector"),
                        confluences=abc_idea.get("confluences") or [],
                        regime=backend_trend_regime,
                        user_tf=user_tf,
                        atr_val=atr_val,
                        htf_context=htf_context,
                        context_bundle=context_bundle,
                    )

                    if abc_candidate:
                        abc_candidate["template_priority"] = int(abc_idea.get("template_priority") or 18)
                        abc_candidate["abc_c_ending"] = True
                        abc_candidate["abc_status"] = abc_idea.get("abc_status", "C_ENDING")
                        abc_candidate["elliott_phase_alignment"] = abc_idea.get("elliott_phase_alignment", "UNKNOWN")
                        abc_candidate["elliott_phase_score"] = abc_idea.get("elliott_phase_score", 0)
                        abc_candidate["elliott_phase_reason"] = abc_idea.get("elliott_phase_reason", "")
                        abc_candidate["execution_elliott_phase"] = abc_idea.get("execution_elliott_phase", "ABC_C_ENDING")
                        abc_candidate["execution_elliott_side"] = abc_idea.get("execution_elliott_side", "UNKNOWN")
                        abc_candidate["phase_setup_note"] = abc_idea.get("phase_setup_note", "")
                        abc_candidate["abc_entry_label"] = abc_idea.get("entry_label", "")
                        candidates.append(abc_candidate)

        except Exception:
            pass

        # --------------------------------------------------
        # Early momentum breakout continuation candidate.
        # This is for cases where selected TF is slow:
        # - 1h setup uses 15m early breakout
        # - 4h setup uses 1h early breakout
        #
        # Strict safety:
        # - only same side as backend preferred side
        # - only if detector says not extended
        # - still passes normal _reject_invalid_candidate()
        # --------------------------------------------------
        try:
            mb = momentum_breakout_context if isinstance(momentum_breakout_context, dict) else {}

            mb_ok = bool(mb.get("momentum_breakout"))
            mb_side = str(mb.get("side") or "").upper()
            mb_extended = bool(mb.get("is_extended"))
            mb_entry = _safe_float(mb.get("suggested_entry"))
            mb_stop = _safe_float(mb.get("suggested_stop"))
            mb_score = int(mb.get("score") or 0)
            mb_strength = str(mb.get("strength") or "NONE").upper()
            phase_alignment = str(
                (elliott_phase_context or {}).get("elliott_phase_alignment") or "UNKNOWN"
            ).upper()

            if (
                mb_ok
                and not mb_extended
                and mb_side == side
                and mb_entry is not None
                and mb_stop is not None
                and mb_score >= 62
                and phase_alignment != "CONFLICT"
            ):
                confluences = [
                    f"Lower-TF momentum breakout {mb_strength}",
                    f"Breakout TF {mb.get('timeframe')}",
                ]

                if mb.get("volume_ratio") is not None:
                    confluences.append(f"Volume expansion {mb.get('volume_ratio')}x")

                confluences.extend(
                    _build_confluences(
                        side,
                        mb_entry,
                        pivots,
                        ema50,
                        ema200,
                        atr_val,
                        context_bundle=context_bundle,
                    )
                )
                confluences = list(dict.fromkeys(confluences))

                c = _make_candidate(
                    side=side,
                    entry=mb_entry,
                    stop=mb_stop,
                    current_price=current_price,
                    entry_type="momentum_breakout_continuation",
                    source="momentum_breakout_detector",
                    confluences=confluences,
                    regime=backend_trend_regime,
                    user_tf=user_tf,
                    atr_val=atr_val,
                    htf_context=htf_context,
                    context_bundle=context_bundle,
                )

                if c:
                    c["template_priority"] = 25
                    c["momentum_breakout"] = True
                    c["momentum_breakout_score"] = mb_score
                    c["momentum_breakout_strength"] = mb_strength
                    c["momentum_breakout_reason"] = mb.get("reason", "")
                    c["breakout_level"] = mb.get("breakout_level")
                    candidates.append(c)

        except Exception:
            pass

        final_candidates: List[Dict[str, Any]] = []
        seen_keys = set()
        for c in candidates:
            try:
                key = (c.get("side"), round(float(c.get("entry")), 8), round(float(c.get("stop_loss")), 8), c.get("entry_type"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                c = _reject_invalid_candidate(c, user_tf=user_tf)
                c = _apply_cross_tf_risk_mode(
                    c,
                    cross_tf_setups=cross_tf_setups,
                    user_tf=user_tf
                )
                c = _score_candidate(c, backend_trend_regime)
                final_candidates.append(c)
            except Exception:
                continue

        final_candidates.sort(
            key=lambda x: (
                0 if not x.get("rejected") else 1,
                int(x.get("template_priority") or 99),
                0 if str(x.get("mtf_elliott_alignment") or "").upper() == "STRONG" else
                1 if str(x.get("mtf_elliott_alignment") or "").upper() == "MEDIUM" else
                2 if str(x.get("mtf_elliott_alignment") or "").upper() == "WEAK" else
                3,
                -float(x.get("rr") or 0),
                float(x.get("risk_pct") or 999),
            )
        )
        return final_candidates
    except Exception:
        return candidates


def select_best_backend_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        valid = [c for c in candidates if not c.get("rejected")]
        if not valid:
            return None
        valid.sort(
            key=lambda x: (
                int(x.get("template_priority") or 99),
                0 if str(x.get("mtf_elliott_alignment") or "").upper() == "STRONG" else
                1 if str(x.get("mtf_elliott_alignment") or "").upper() == "MEDIUM" else
                2 if str(x.get("mtf_elliott_alignment") or "").upper() == "WEAK" else
                3,
                -float(x.get("rr") or 0),
                float(x.get("risk_pct") or 999),
                -int(x.get("score") or 0),
            )
        )
        return valid[0]
    except Exception:
        return None


# ==========================================================
# Public planner formatting
# ==========================================================

def _probability_from_score(score: int) -> str:
    try:
        score = int(score)
        if score >= 82:
            return "High"
        if score >= 70:
            return "Medium-High"
        if score >= 62:
            return "Medium"
    except Exception:
        pass
    return "Low"


def _setup_type_from_regime(regime: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    try:
        entry_type = str(candidate.get("entry_type") or "").upper()

        if "ABC_C_ENDING_LONG" in entry_type:
            return "ABC_C_ENDING_LONG"

        if "ABC_C_ENDING_SHORT" in entry_type:
            return "ABC_C_ENDING_SHORT"

        if "BULLISH_WAVE_2_TO_WAVE_3" in entry_type:
            return "BULLISH_WAVE_2_TO_WAVE_3"
        if "BEARISH_WAVE_2_TO_WAVE_3" in entry_type:
            return "BEARISH_WAVE_2_TO_WAVE_3"
        if "BULLISH_WAVE_4_TO_WAVE_5" in entry_type:
            return "BULLISH_WAVE_4_TO_WAVE_5"
        if "BEARISH_WAVE_4_TO_WAVE_5" in entry_type:
            return "BEARISH_WAVE_4_TO_WAVE_5"

        setup_bias = str(regime.get("setup_bias") or "").upper()
        side = str(candidate.get("side") or "").upper()

        if "WAVE_2" in setup_bias or "PULLBACK" in setup_bias:
            return "WAVE_2_PULLBACK" if side == "LONG" else "BEARISH_WAVE_2_PULLBACK"
        if "BREAKOUT" in entry_type:
            return "BREAKOUT_RETEST"
        if "BREAKDOWN" in entry_type:
            return "BREAKDOWN_RETEST"
        if "PULLBACK" in entry_type:
            return "TREND_PULLBACK"
        if "EMA200" in entry_type:
            return "HTF_EMA200_RETEST"

    except Exception:
        pass

    return "BACKEND_SELECTED_SETUP"


def _candidate_to_selected_setup(candidate: Dict[str, Any], backend_trend_regime: Dict[str, Any], symbol: str = "", user_tf: str = "15m") -> Dict[str, Any]:
    side = str(candidate.get("side") or "WAIT").upper()
    score = int(candidate.get("score") or 0)
    entry = float(candidate["entry"])
    stop = float(candidate["stop_loss"])
    tp1 = float(candidate["tp1"])
    tp2 = float(candidate["tp2"])
    setup_type = _setup_type_from_regime(backend_trend_regime, candidate)
    confluences = candidate.get("confluences") or []
    confluence_text = ", ".join(confluences) if confluences else "Backend structural confluence"
    status = str(candidate.get("status") or "LIMIT_ORDER").upper()
    if status not in ["LIMIT_ORDER", "RUNNING", "INVALIDATED"]:
        status = "LIMIT_ORDER"
    trade_type = "WAIT" if status == "INVALIDATED" else side
    summary = (
        f"Backend selected {side} {setup_type} from {candidate.get('entry_type')} "
        f"by template priority {candidate.get('template_priority')}. "
        f"Confluence: {confluence_text}. TP2 reason: {candidate.get('tp2_reason')}. "
        f"{candidate.get('status_reason', '')}"
    ).strip()
    if candidate.get("source") == "elliott_wave_template_detector":
        wave_logic = candidate.get("wave_logic") or {}
        summary = (
            f"Elliott wave template confirmed: {candidate.get('wave_template_name')}. "
            f"{wave_logic.get('entry_logic', '')} "
            f"{wave_logic.get('sl_logic', '')} "
            f"{wave_logic.get('tp2_logic', '')} "
            f"{candidate.get('status_reason', '')}"
        ).strip()
        mtf_alignment = str(candidate.get("mtf_elliott_alignment") or "UNKNOWN").upper()
        mtf_score = candidate.get("mtf_confirmation_score", 0)
        mtf_reason = candidate.get("mtf_confirmation_reason") or ""

        summary = (
            summary
            + f" MTF Elliott alignment: {mtf_alignment} "
            + f"(score {mtf_score}). {mtf_reason}"
        ).strip()
    if candidate.get("cross_tf_conflict"):
        summary += (
            " Cross-TF caution: "
            + str(candidate.get("cross_tf_conflict_reason") or "")
            + " Treat this as TP1-focused / scalp-only; do not blindly hold for full TP2."
        )

    if candidate.get("abc_c_ending"):
        summary += (
            " ABC C-ending setup: correction may be ending near a confirmed structure zone. "
            "This setup was generated by Elliott phase context and still passed backend validation. "
            "Do not chase if price leaves the entry zone."
        )
    return {
        "name": "Sniper Setup",
        "trade_type": trade_type,
        "side": side,
        "status": status,
        "order_type": "LIMIT" if status == "LIMIT_ORDER" else "ACTIVE",
        "probability": _probability_from_score(score),
        "quality_score": score,
        "summary": summary,
        "entry_zone": format_setup_price(entry),
        "entry": format_setup_price(entry),
        "invalidation": format_setup_price(stop),
        "stop_loss": format_setup_price(stop),
        "targets": [format_setup_price(tp1), format_setup_price(tp2)],
        "target": format_setup_price(tp2),
        "risk_pct": round(float(candidate.get("risk_pct") or 0), 2),
        "rr": round(float(candidate.get("rr") or 0), 2),
        "tp1_r": TP1_R,
        "tp2_r": round(float(candidate.get("tp2_r") or TP2_R_MIN), 2),
        "setup_type": setup_type,
        "entry_type": candidate.get("entry_type"),
        "entry_source": candidate.get("source"),
        "setup_source": "ELLIOTT_WAVE_TEMPLATE"
        if candidate.get("source") == "elliott_wave_template_detector"
        else "STANDARD_BACKEND_PLANNER",

        "elliott_confirmed": candidate.get("source") == "elliott_wave_template_detector",
        "wave_template": candidate.get("wave_template_name"),
        "wave_target": candidate.get("wave_target"),
        "wave_confidence": candidate.get("wave_confidence"),
        "wave_status": candidate.get("wave_status"),
        "wave_points": candidate.get("wave_points"),
        "fib_used": candidate.get("fib_used"),
        "wave_logic": candidate.get("wave_logic"),
        "wave3_status": candidate.get("wave3_status"),
        "wave3_confirmed": candidate.get("wave3_confirmed"),
        "wave3_confirmation_reason": candidate.get("wave3_confirmation_reason"),
        "wave3_break_level": candidate.get("wave3_break_level"),

        "mtf_elliott_alignment": candidate.get("mtf_elliott_alignment", "UNKNOWN"),
        "mtf_wave_side": candidate.get("mtf_wave_side", "UNKNOWN"),
        "mtf_wave_phase": candidate.get("mtf_wave_phase", "UNKNOWN"),
        "mtf_confirmation_score": candidate.get("mtf_confirmation_score", 0),
        "mtf_confirmation_reason": candidate.get("mtf_confirmation_reason", ""),
        "abc_status": candidate.get("abc_status", "NONE"),
        "abc_c_ending": candidate.get("abc_c_ending", False),
        "phase_setup_note": candidate.get("phase_setup_note", ""),
        "execution_elliott_phase": candidate.get("execution_elliott_phase", "UNKNOWN"),
        "execution_elliott_side": candidate.get("execution_elliott_side", "UNKNOWN"),
        "elliott_phase_alignment": candidate.get("elliott_phase_alignment", "UNKNOWN"),
        "elliott_phase_score": candidate.get("elliott_phase_score", 0),
        "elliott_phase_reason": candidate.get("elliott_phase_reason", ""),
        "elliott_phase_alignment": candidate.get("elliott_phase_alignment", "UNKNOWN"),
        "elliott_phase_score": candidate.get("elliott_phase_score", 0),
        "elliott_phase_permission": candidate.get("elliott_phase_permission", "WAIT"),
        "elliott_phase_reason": candidate.get("elliott_phase_reason", ""),
        "execution_elliott_phase": candidate.get("execution_elliott_phase", "UNKNOWN"),
        "confirm_elliott_phase": candidate.get("confirm_elliott_phase", "UNKNOWN"),
        "swing_elliott_phase": candidate.get("swing_elliott_phase", "UNKNOWN"),
        "macro_elliott_phase": candidate.get("macro_elliott_phase", "UNKNOWN"),
        "abc_status": candidate.get("abc_status", "NONE"),
        "momentum_breakout": candidate.get("momentum_breakout", False),
        "momentum_breakout_score": candidate.get("momentum_breakout_score"),
        "momentum_breakout_strength": candidate.get("momentum_breakout_strength"),
        "momentum_breakout_reason": candidate.get("momentum_breakout_reason", ""),
        "cross_tf_conflict": candidate.get("cross_tf_conflict", False),
        "cross_tf_risk_mode": candidate.get("cross_tf_risk_mode", "NORMAL"),
        "cross_tf_conflict_type": candidate.get("cross_tf_conflict_type", ""),
        "cross_tf_conflict_reason": candidate.get("cross_tf_conflict_reason", ""),
        "breakout_level": candidate.get("breakout_level"),
        "template_priority": candidate.get("template_priority"),
        "confluences": confluences,
        "tp2_reason": candidate.get("tp2_reason"),
        "backend_selected": True,
        "ai_can_modify_prices": False,
        "backend_trend_regime": backend_trend_regime,
        "backend_regime": backend_trend_regime.get("regime"),
        "backend_market_bias": backend_trend_regime.get("market_bias"),
        "backend_market_phase": backend_trend_regime.get("market_phase"),
        "backend_preferred_side": backend_trend_regime.get("preferred_side"),
        "backend_blocked_side": backend_trend_regime.get("blocked_side"),
        "backend_regime_confidence": backend_trend_regime.get("confidence"),
        "symbol": symbol,
        "timeframe": user_tf,
        "raw_values": {
            "entry": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "raw_tp2": float(candidate.get("raw_tp2") or tp2),
            "tp2_r": float(candidate.get("tp2_r") or TP2_R_MIN),
            "risk": float(candidate.get("risk") or 0),
            "risk_pct": float(candidate.get("risk_pct") or 0),
            "rr": float(candidate.get("rr") or 0),
            "current_price": float(candidate.get("current_price") or 0),
            "current_price_distance_pct": float(candidate.get("distance_pct") or 0),
            "atr": float(candidate.get("atr") or 0),
            "htf_anchor_level": float((candidate.get("htf_context") or {}).get("htf_anchor_level") or 0),
        },
    }


def _compact_candidates_for_debug(candidates: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    compact = []
    try:
        for c in candidates[:limit]:
            compact.append({
                "side": c.get("side"),
                "entry": format_setup_price(c.get("entry")),
                "stop_loss": format_setup_price(c.get("stop_loss")),
                "tp1": format_setup_price(c.get("tp1")),
                "tp2": format_setup_price(c.get("tp2")),
                "raw_tp2": format_setup_price(c.get("raw_tp2")),
                "tp2_reason": c.get("tp2_reason"),
                "risk_pct": round(float(c.get("risk_pct") or 0), 2),
                "rr": round(float(c.get("rr") or 0), 2),
                "score": c.get("score"),
                "priority": c.get("template_priority"),
                "entry_type": c.get("entry_type"),
                "status": c.get("status"),
                "current_price": format_setup_price(c.get("current_price")),
                "atr": format_setup_price(c.get("atr")),
                "htf_anchor": format_setup_price((c.get("htf_context") or {}).get("htf_anchor_level")),
                "rejected": c.get("rejected"),
                "reject_reason": c.get("reject_reason", ""),
                "source": c.get("source"),
                "wave_template": c.get("wave_template_name"),
                "wave_target": c.get("wave_target"),
                "wave_confidence": c.get("wave_confidence"),
            })
    except Exception:
        pass
    return compact


def build_wait_setup(reason: str, backend_trend_regime: Optional[Dict[str, Any]] = None, candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    backend_trend_regime = backend_trend_regime or {}
    candidates = candidates or []
    watch_candidate = None
    for c in candidates:
        if c.get("watch_setup"):
            watch_candidate = c
            break
    return {
        "name": "Sniper Setup",
        "trade_type": "WAIT",
        "side": "NONE",
        "status": "WAIT_NO_BACKEND_SETUP",
        "order_type": "NONE",
        "probability": "Low",
        "quality_score": 0,
        "summary": reason,
        "entry_zone": "",
        "entry": "",
        "invalidation": "",
        "stop_loss": "",
        "targets": [],
        "target": "",
        "setup_type": "WAIT_NO_BACKEND_SETUP",
        "backend_selected": False,
        "ai_can_modify_prices": False,
        "backend_trend_regime": backend_trend_regime,
        "backend_candidate_count": len(candidates),
        "backend_candidates_debug": _compact_candidates_for_debug(candidates),
        "watch_setup": bool(watch_candidate),
        "watch_type": watch_candidate.get("watch_type", "") if watch_candidate else "",
        "watch_side": watch_candidate.get("watch_side", "") if watch_candidate else "",
        "watch_level": watch_candidate.get("watch_level") if watch_candidate else None,
        "watch_level_label": watch_candidate.get("watch_level_label", "") if watch_candidate else "",
        "watch_condition": watch_candidate.get("watch_condition", "") if watch_candidate else "",
        "watch_next_candidate_type": watch_candidate.get("watch_next_candidate_type", "") if watch_candidate else "",
        "watch_recheck_needed": watch_candidate.get("watch_recheck_needed", False) if watch_candidate else False,
        "watch_reason": watch_candidate.get("watch_reason", "") if watch_candidate else "",
    }


def plan_backend_setup(
    df_micro: pd.DataFrame,
    backend_trend_regime: Dict[str, Any],
    df_live: Optional[pd.DataFrame] = None,
    symbol: str = "",
    user_tf: str = "15m",
    df_confirm: Optional[pd.DataFrame] = None,
    df_swing: Optional[pd.DataFrame] = None,
    context_bundle: Optional[Dict[str, Any]] = None,
    elliott_mtf_context: Optional[Dict[str, Any]] = None,
    elliott_phase_context: Optional[Dict[str, Any]] = None,
    momentum_breakout_context: Optional[Dict[str, Any]] = None,
    cross_tf_setups: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        regime = backend_trend_regime or {}
        wait_by_context, wait_reason = _context_prefers_wait(context_bundle)

        # Hard WAIT only for CHoCH / both-side rejection / real chop.
        # Dominant WAIT alone should not always block candidate generation.
        hard_context_wait = False
        try:
            cb = _context_dict(context_bundle)
            derived = cb.get("derived", {}) or {}
            dom = cb.get("dominant_trend", {}) or {}

            hard_context_wait = bool(
                derived.get("choch")
                or dom.get("choch") is True
                or derived.get("both_support_and_resistance_rejection")
            )
        except Exception:
            hard_context_wait = False

        if wait_by_context and hard_context_wait:
            wait_setup = build_wait_setup(wait_reason, regime, [])
            return {
                "has_setup": False,
                "selected_setup": wait_setup,
                "candidates": [],
                "planner_status": "WAIT_CONTEXT_CONFLICT",
                "planner_reason": wait_reason,
            }

        if wait_by_context and not _has_strong_context_override(context_bundle):
            reason = wait_reason + " No strong Fib/S/R/rejection override exists."
            wait_setup = build_wait_setup(reason, regime, [])
            return {
                "has_setup": False,
                "selected_setup": wait_setup,
                "candidates": [],
                "planner_status": "WAIT_CONTEXT_CONFLICT",
                "planner_reason": reason,
            }

        preferred_side = str(regime.get("preferred_side") or "NONE").upper()
        if preferred_side not in ["LONG", "SHORT"]:
            try:
                cb = _context_dict(context_bundle)
                dom_side = (
                    cb.get("derived", {}).get("preferred_side_from_dominant")
                    or cb.get("dominant_trend", {}).get("preferred_trade_side")
                )
                dom_side = str(dom_side or "").upper().strip()

                if dom_side in ["LONG", "SHORT"] and _has_strong_context_override(context_bundle):
                    preferred_side = dom_side
                    regime = dict(regime)
                    regime["preferred_side"] = preferred_side
                    regime["setup_bias"] = f"CAUTIOUS_CONTEXT_OVERRIDE_{preferred_side}"
                    regime["confidence"] = max(float(regime.get("confidence") or 0), MIN_REGIME_CONFIDENCE)
            except Exception:
                pass

        confidence = float(regime.get("confidence") or 0)
        if preferred_side not in ["LONG", "SHORT"]:
            debug_regime = dict(regime)
            debug_regime["preferred_side"] = "NONE"
            debug_regime["confidence"] = max(float(debug_regime.get("confidence") or 0), MIN_REGIME_CONFIDENCE)

            debug_candidates = generate_backend_candidates(
                df_micro=df_micro,
                backend_trend_regime=debug_regime,
                df_live=df_live,
                user_tf=user_tf,
                df_confirm=df_confirm,
                df_swing=df_swing,
                context_bundle=context_bundle,
                symbol=symbol,
                elliott_mtf_context=elliott_mtf_context,
                elliott_phase_context=elliott_phase_context,
                momentum_breakout_context=momentum_breakout_context,
            )

            reason = (
                "WAIT. Backend setup planner did not create a trade because trend regime has no clear preferred side. "
                "Elliott detector was still checked for debug candidates."
            )

            return {
                "has_setup": False,
                "selected_setup": build_wait_setup(reason, regime, debug_candidates),
                "candidates": debug_candidates,
                "planner_status": "WAIT_NO_CLEAR_SIDE",
                "planner_reason": reason,
            }
        if confidence < MIN_REGIME_CONFIDENCE:
            reason = f"WAIT. Backend regime confidence {confidence:.2f} is below minimum {MIN_REGIME_CONFIDENCE:.2f}."
            return {"has_setup": False, "selected_setup": build_wait_setup(reason, regime, []), "candidates": [], "planner_status": "WAIT_LOW_REGIME_CONFIDENCE", "planner_reason": reason}

        candidates = generate_backend_candidates(
            df_micro=df_micro,
            backend_trend_regime=regime,
            df_live=df_live,
            user_tf=user_tf,
            df_confirm=df_confirm,
            df_swing=df_swing,
            context_bundle=context_bundle,
            symbol=symbol,
            elliott_mtf_context=elliott_mtf_context,
            elliott_phase_context=elliott_phase_context,
            momentum_breakout_context=momentum_breakout_context,
        )
        selected_candidate = select_best_backend_candidate(candidates)
        if not selected_candidate:
            wall_candidates = [
                c for c in candidates
                if str(c.get("status") or "").upper() in [
                    "WAIT_FOR_RECLAIM",
                    "WAIT_FOR_BREAKDOWN",
                    "WAIT_FOR_RETEST_HOLD",
                    "WAIT_FOR_RETEST_REJECTION",
                ]
            ]

            if wall_candidates:
                reason = wall_candidates[0].get("reject_reason") or wall_candidates[0].get("status_reason") or (
                    "WAIT. Elliott Wave 3 idea exists, but TP1 is blocked by nearby structure."
                )
                planner_status = str(wall_candidates[0].get("status") or "WAIT_FOR_RECLAIM").upper()
            else:
                reason = (
                    "WAIT. Backend generated candidates but none passed template priority + "
                    "SL/TP realism + no-chase + HTF alignment filters."
                )
                planner_status = "WAIT_NO_VALID_CANDIDATE"

            return {
                "has_setup": False,
                "selected_setup": build_wait_setup(reason, regime, candidates),
                "candidates": candidates,
                "planner_status": planner_status,
                "planner_reason": reason,
            }

        selected_setup = _candidate_to_selected_setup(selected_candidate, backend_trend_regime=regime, symbol=symbol, user_tf=user_tf)
        if wait_by_context:
            selected_setup["dominant_wait_caution"] = True
            selected_setup["summary"] = (
                selected_setup.get("summary", "")
                + " Caution: dominant trend context preferred WAIT, but strong Fib/S/R/rejection context allowed candidate generation."
            ).strip()
        return {"has_setup": selected_setup.get("trade_type") in ["LONG", "SHORT"], "selected_setup": selected_setup, "selected_candidate": selected_candidate, "candidates": candidates, "planner_status": "BACKEND_SETUP_SELECTED", "planner_reason": selected_setup.get("summary", "")}
    except Exception as e:
        reason = f"WAIT. Backend setup planner error: {str(e)}"
        return {"has_setup": False, "selected_setup": build_wait_setup(reason, backend_trend_regime or {}, []), "candidates": [], "planner_status": "PLANNER_ERROR", "planner_reason": reason}


def format_backend_setup_plan_context(plan: Dict[str, Any]) -> str:
    try:
        setup = plan.get("selected_setup") or {}
        candidates = plan.get("candidates") or []
        compact = _compact_candidates_for_debug(candidates, limit=10)
        candidate_lines = []
        for i, c in enumerate(compact, start=1):
            rejected = "YES" if c.get("rejected") else "NO"
            candidate_lines.append(
                f"{i}. {c.get('side')} {c.get('entry_type')} [{c.get('source', '')}] | "
                f"Priority:{c.get('priority')} E:{c.get('entry')} SL:{c.get('stop_loss')} "
                f"TP1:{c.get('tp1')} TP2:{c.get('tp2')} RawTP2:{c.get('raw_tp2')} "
                f"TP2Reason:{c.get('tp2_reason')} Risk:{c.get('risk_pct')}% RR:{c.get('rr')} "
                f"Score:{c.get('score')} Status:{c.get('status')} Current:{c.get('current_price')} "
                f"ATR:{c.get('atr')} "
                f"MTF:{c.get('mtf_elliott_alignment', 'UNKNOWN')} "
                f"MTFScore:{c.get('mtf_confirmation_score', '')} "
                f"MB:{c.get('momentum_breakout', False)} "
                f"ABC:{c.get('abc_c_ending', False)} "
                f"MBScore:{c.get('momentum_breakout_score', '')} "
                f"Rejected:{rejected} {c.get('reject_reason', '')}"
            )
        if not candidate_lines:
            candidate_lines.append("No backend candidates generated.")
        
        elliott_lines = []
        if setup.get("elliott_confirmed"):
            elliott_lines = [
                "",
                "--- ELLIOTT WAVE SETUP CONFIRMATION ---",
                f"Wave Template: {setup.get('wave_template')}",
                f"Status: {setup.get('wave_status')}",
                f"Confidence: {setup.get('wave_confidence')}",
                f"Wave Target: {setup.get('wave_target')}",
                f"Setup Source: {setup.get('setup_source')}",
                f"Wave Points: {setup.get('wave_points')}",
                f"Fib Used: {setup.get('fib_used')}",
                f"Wave Logic: {setup.get('wave_logic')}",
                f"MTF Alignment: {setup.get('mtf_elliott_alignment', 'UNKNOWN')}",
                f"MTF Wave Side: {setup.get('mtf_wave_side', 'UNKNOWN')}",
                f"MTF Wave Phase: {setup.get('mtf_wave_phase', 'UNKNOWN')}",
                f"MTF Score: {setup.get('mtf_confirmation_score', 0)}",
                f"MTF Reason: {setup.get('mtf_confirmation_reason', '')}",
            ]

        return "\n".join([
            "--- BACKEND SETUP PLANNER ---",
            f"Planner Status: {plan.get('planner_status', 'UNKNOWN')}",
            f"Has Setup: {'YES' if plan.get('has_setup') else 'NO'}",
            f"Selected Trade Type: {setup.get('trade_type', 'WAIT')}",
            f"Selected Status: {setup.get('status', 'N/A')}",
            f"Selected Setup Type: {setup.get('setup_type', 'N/A')}",
            f"Selected Priority: {setup.get('template_priority', '')}",
            f"Entry: {setup.get('entry_zone', '')}",
            f"Stop Loss: {setup.get('invalidation', '')}",
            f"Targets: {setup.get('targets', [])}",
            f"TP2 Reason: {setup.get('tp2_reason', '')}",
            f"Risk %: {setup.get('risk_pct', '')}",
            f"RR: {setup.get('rr', '')}",
            f"Quality Score: {setup.get('quality_score', '')} (debug only)",
            f"AI Can Modify Prices: {setup.get('ai_can_modify_prices', False)}",
            f"Planner Reason: {plan.get('planner_reason', '')}",
            "Candidate Debug:",
            "\n".join(candidate_lines),
            *elliott_lines,
            "Planner Rule: Selection uses setup-template priority + hard validation, not score-only. AI must not change entry, SL, TP, side, or targets.",
        ])
    except Exception as e:
        return "--- BACKEND SETUP PLANNER ---\n" + f"Setup Planner Formatting Error: {str(e)}"


# ==========================================================
# Final validation after AI
# ==========================================================

def validate_ai_kept_backend_setup(backend_setup: Dict[str, Any], ai_setup: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    try:
        if not backend_setup:
            return False, "Missing backend setup."
        if not ai_setup:
            return True, "AI setup missing; use backend setup."
        backend_side = str(backend_setup.get("trade_type") or "").upper()
        ai_side = str(ai_setup.get("trade_type") or "").upper()
        if backend_side in ["LONG", "SHORT"] and ai_side and backend_side != ai_side:
            return False, f"AI changed side from {backend_side} to {ai_side}."
        for backend_key, alt_key in [("entry_zone", "entry"), ("invalidation", "stop_loss")]:
            b_val = _safe_float(backend_setup.get(backend_key) or backend_setup.get(alt_key))
            a_val = _safe_float(ai_setup.get(backend_key) or ai_setup.get(alt_key))
            if b_val is None or a_val is None:
                continue
            tolerance = max(abs(b_val) * 0.000001, 0.00000001)
            if abs(b_val - a_val) > tolerance:
                return False, f"AI changed {backend_key} from {b_val} to {a_val}."
        b_targets = backend_setup.get("targets") or []
        a_targets = ai_setup.get("targets") or []
        if b_targets and a_targets:
            for i, b_target in enumerate(b_targets):
                if i >= len(a_targets):
                    return False, "AI removed one or more backend targets."
                b_val = _safe_float(b_target)
                a_val = _safe_float(a_targets[i])
                if b_val is None or a_val is None:
                    continue
                tolerance = max(abs(b_val) * 0.000001, 0.00000001)
                if abs(b_val - a_val) > tolerance:
                    return False, f"AI changed TP{i + 1} from {b_val} to {a_val}."
        return True, "OK"
    except Exception as e:
        return False, f"Backend setup validation error: {str(e)}"
