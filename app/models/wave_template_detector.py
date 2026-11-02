# app/models/wave_template_detector.py

"""
Wave Template Detector

Purpose:
- Detect practical Elliott Wave trading templates before normal EMA/Fib/SR setup planning.
- Focus on:
    1) Bullish Wave 2 -> Wave 3 continuation
    2) Bearish Wave 2 -> Wave 3 continuation
    3) Bullish Wave 4 -> Wave 5 continuation
    4) Bearish Wave 4 -> Wave 5 continuation

Important:
- This file does NOT call AI.
- This file does NOT save trades.
- This file returns backend-owned candidate data only.
- setup_planner.py should validate final risk/RR/no-chase before using the candidate.

Main idea:
- Wave 2 / Wave 4 ends are estimated using Fibonacci retracement zones.
- Wave 3 / Wave 5 targets are estimated using Fibonacci extension/projection.
- SL is based on wave invalidation level, not random ATR-only logic.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==========================================================
# Tunable constants
# ==========================================================

DEFAULT_LOOKBACK = 240
DEFAULT_PIVOT_WINDOW = 3

# Elliott multi-window scan settings.
# Wave 3 needs wider structure context than Wave 5.
# 300 is used as background confirmation, not as the main owner.
WAVE3_SCAN_WINDOWS = [180, 240, 300]
WAVE5_SCAN_WINDOWS = [140, 180, 220]

WAVE3_PRIMARY_WINDOW = 240
WAVE5_PRIMARY_WINDOW = 180

# Wave 2 usually can be deeper.
WAVE2_MIN_RETRACE = 0.382
WAVE2_MAX_RETRACE = 0.886

# Preferred Wave 2 entry area.
WAVE2_PREFERRED_RETRACE_LEVELS = [0.500, 0.618, 0.705, 0.786]

# Wave 4 usually should not be too deep.
WAVE4_MIN_RETRACE = 0.236
WAVE4_MAX_RETRACE = 0.500

# Preferred Wave 4 entry area.
WAVE4_PREFERRED_RETRACE_LEVELS = [0.382, 0.500]

# Extensions/projections.
WAVE3_EXTENSION_PRIMARY = 1.618
WAVE3_EXTENSION_CONSERVATIVE = 1.272
WAVE5_EXTENSION_PRIMARY = 1.000
WAVE5_EXTENSION_EXTENDED = 1.272

# Safety
MIN_IMPULSE_ATR_MULTIPLIER = 1.20
MAX_ENTRY_DISTANCE_ATR = 4.0
INVALIDATION_ATR_BUFFER = 0.15
# Wave 3 confirmation logic
WAVE3_BREAK_CONFIRM_ATR_BUFFER = 0.10
WAVE3_STRONG_BODY_ATR_MULTIPLIER = 0.60
WAVE3_CONFIRM_LOOKBACK_CANDLES = 3

# Confidence scoring
HIGH_CONFIDENCE = 80
MEDIUM_CONFIDENCE = 65


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


def _tf_key(user_tf: str) -> str:
    return str(user_tf or "").lower().strip()


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


def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)
        if len(d) < period + 2:
            return None

        high = d["High"]
        low = d["Low"]
        close = d["Close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        val = tr.rolling(period).mean().iloc[-1]
        return float(val) if pd.notna(val) else None

    except Exception:
        return None


def _ema(series: pd.Series, span: int) -> Optional[float]:
    try:
        if series is None or len(series) < 5:
            return None
        return float(series.ewm(span=span, adjust=False).mean().iloc[-1])
    except Exception:
        return None


def _price_distance_pct(a: float, b: float) -> float:
    try:
        if b == 0:
            return 999.0
        return abs(a - b) / abs(b) * 100.0
    except Exception:
        return 999.0


def _format_float(value: Any, digits: int = 8) -> Optional[float]:
    v = _safe_float(value)
    if v is None:
        return None
    return round(float(v), digits)


# ==========================================================
# Context helpers
# ==========================================================

def _context_dict(context_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return context_bundle if isinstance(context_bundle, dict) else {}


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


def _context_hard_block(context_bundle: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Hard block only true for serious no-trade context:
    - CHoCH / structure break
    - both support and resistance rejection / chop
    """
    try:
        cb = _context_dict(context_bundle)
        derived = cb.get("derived", {}) or {}
        dom = cb.get("dominant_trend", {}) or {}

        if derived.get("choch") or dom.get("choch") is True:
            return True, "CHoCH / structure break detected."

        if derived.get("both_support_and_resistance_rejection"):
            return True, "Both support and resistance rejection detected; choppy market."

        if derived.get("is_choppy"):
            return True, "Context bundle marks market as choppy."

        return False, ""

    except Exception:
        return False, ""


def _zone_valid(zone: Any) -> bool:
    return (
        isinstance(zone, dict)
        and bool(zone.get("valid"))
        and zone.get("low") is not None
        and zone.get("high") is not None
    )


def _context_major_levels(context_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cb = _context_dict(context_bundle)
    ms = cb.get("market_structure", {}) or {}
    return {
        "nearest_support": _safe_float(ms.get("nearest_support")),
        "nearest_resistance": _safe_float(ms.get("nearest_resistance")),
        "recent_swing_high": _safe_float(ms.get("recent_swing_high")),
        "recent_swing_low": _safe_float(ms.get("recent_swing_low")),
        "major_support_zone": ms.get("major_support_zone", {}) or {},
        "major_resistance_zone": ms.get("major_resistance_zone", {}) or {},
        "long_lookback_high": _safe_float(ms.get("long_lookback_high")),
        "long_lookback_low": _safe_float(ms.get("long_lookback_low")),
        "major_swing_high": _safe_float(ms.get("major_swing_high")),
        "major_swing_low": _safe_float(ms.get("major_swing_low")),
    }


def _dominant_side(context_bundle: Optional[Dict[str, Any]]) -> str:
    try:
        cb = _context_dict(context_bundle)
        side = (
            cb.get("derived", {}).get("preferred_side_from_dominant")
            or cb.get("dominant_trend", {}).get("preferred_trade_side")
            or ""
        )
        side = str(side).upper().strip()
        if side in ["LONG", "SHORT", "WAIT"]:
            return side
        return ""
    except Exception:
        return ""


def _ema_bias(df: pd.DataFrame) -> str:
    try:
        d = _normalize_ohlc(df)
        if d.empty:
            return "UNKNOWN"

        close = float(d["Close"].iloc[-1])
        ema50 = _ema(d["Close"], 50)
        ema200 = _ema(d["Close"], 200)

        if ema50 is None or ema200 is None:
            return "UNKNOWN"

        if close > ema50 and close > ema200 and ema50 >= ema200:
            return "BULLISH"

        if close < ema50 and close < ema200 and ema50 <= ema200:
            return "BEARISH"

        return "MIXED"

    except Exception:
        return "UNKNOWN"


# ==========================================================
# Pivot detection
# ==========================================================

def _find_pivots(
    df: pd.DataFrame,
    window: int = DEFAULT_PIVOT_WINDOW,
    lookback: int = DEFAULT_LOOKBACK,
) -> Dict[str, Any]:
    """
    Simple pivot detector.
    Returns chronological pivot highs/lows.
    """
    result = {
        "highs": [],
        "lows": [],
        "recent_high": None,
        "previous_high": None,
        "recent_low": None,
        "previous_low": None,
        "highest": None,
        "lowest": None,
        "highest_idx": None,
        "lowest_idx": None,
    }

    try:
        d = _normalize_ohlc(df).tail(lookback).copy()
        if len(d) < max(20, window * 4):
            return result

        highs: List[Tuple[Any, float]] = []
        lows: List[Tuple[Any, float]] = []

        h = d["High"].values
        l = d["Low"].values
        idx = list(d.index)

        for i in range(window, len(d) - window):
            if h[i] >= np.nanmax(h[i - window:i + window + 1]):
                highs.append((idx[i], float(h[i])))

            if l[i] <= np.nanmin(l[i - window:i + window + 1]):
                lows.append((idx[i], float(l[i])))

        if not highs:
            high_idx = d["High"].idxmax()
            highs.append((high_idx, float(d.loc[high_idx, "High"])))

        if not lows:
            low_idx = d["Low"].idxmin()
            lows.append((low_idx, float(d.loc[low_idx, "Low"])))

        highest_idx = d["High"].idxmax()
        lowest_idx = d["Low"].idxmin()

        result["highs"] = highs
        result["lows"] = lows
        result["recent_high"] = highs[-1][1] if highs else None
        result["previous_high"] = highs[-2][1] if len(highs) >= 2 else None
        result["recent_low"] = lows[-1][1] if lows else None
        result["previous_low"] = lows[-2][1] if len(lows) >= 2 else None
        result["highest"] = float(d.loc[highest_idx, "High"])
        result["lowest"] = float(d.loc[lowest_idx, "Low"])
        result["highest_idx"] = highest_idx
        result["lowest_idx"] = lowest_idx

        return result

    except Exception:
        return result


def _index_position(df: pd.DataFrame, idx_value: Any) -> int:
    try:
        indices = list(df.index)
        return indices.index(idx_value)
    except Exception:
        return -1


def _latest_price(df: pd.DataFrame) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)
        if d.empty:
            return None
        return float(d["Close"].iloc[-1])
    except Exception:
        return None


# ==========================================================
# Fibonacci helpers
# ==========================================================

def _retracement_price(side: str, wave_start: float, wave_end: float, retracement: float) -> float:
    """
    side LONG:
        Wave 1 = low -> high.
        Wave 2 retraces down from high.
        entry = high - length * retracement

    side SHORT:
        Wave 1 = high -> low.
        Wave 2 retraces up from low.
        entry = low + length * retracement
    """
    length = abs(wave_end - wave_start)

    if side == "LONG":
        return wave_end - (length * retracement)

    if side == "SHORT":
        return wave_end + (length * retracement)

    return wave_end


def _extension_target(side: str, wave_start: float, wave_end: float, correction_end: float, extension: float) -> float:
    """
    Project Wave 3 from Wave 2 end using Wave 1 length.
    """
    length = abs(wave_end - wave_start)

    if side == "LONG":
        return correction_end + (length * extension)

    if side == "SHORT":
        return correction_end - (length * extension)

    return correction_end


def _projection_from_wave3(side: str, wave3_start: float, wave3_end: float, wave4_end: float, projection: float) -> float:
    """
    Project Wave 5 using Wave 3 length.
    """
    length = abs(wave3_end - wave3_start)

    if side == "LONG":
        return wave4_end + (length * projection)

    if side == "SHORT":
        return wave4_end - (length * projection)

    return wave4_end


def _retrace_ratio(side: str, wave_start: float, wave_end: float, correction_price: float) -> Optional[float]:
    try:
        length = abs(wave_end - wave_start)
        if length <= 0:
            return None

        if side == "LONG":
            # correction from high downward
            return (wave_end - correction_price) / length

        if side == "SHORT":
            # correction from low upward
            return (correction_price - wave_end) / length

        return None

    except Exception:
        return None

def _wave3_confirmation_status(
    side: str,
    df: pd.DataFrame,
    wave1_start: float,
    wave1_end: float,
    entry: float,
    current_price: float,
    atr_val: float,
) -> Dict[str, Any]:
    """
    Separate early Wave 3 anticipation from confirmed Wave 3 continuation.

    ANTICIPATION:
      Wave 1-2 template exists, but price has not yet proven continuation.

    CONFIRMED:
      Price has broken the Wave 1 end in Wave 3 direction with candle body
      and/or recent momentum is strong in the expected direction.
    """
    result = {
        "wave3_status": "ANTICIPATION",
        "wave3_confirmation_reason": "Wave 1-2 template detected, but Wave 3 breakout confirmation is not yet proven.",
        "wave3_break_level": _format_float(wave1_end),
        "wave3_confirmed": False,
    }

    try:
        d = _normalize_ohlc(df)
        if d.empty or atr_val is None or atr_val <= 0:
            return result

        side = str(side or "").upper()
        recent = d.tail(max(2, WAVE3_CONFIRM_LOOKBACK_CANDLES)).copy()
        last = recent.iloc[-1]

        last_open = float(last["Open"])
        last_close = float(last["Close"])
        last_high = float(last["High"])
        last_low = float(last["Low"])

        body = abs(last_close - last_open)
        strong_body = body >= atr_val * WAVE3_STRONG_BODY_ATR_MULTIPLIER
        break_buffer = atr_val * WAVE3_BREAK_CONFIRM_ATR_BUFFER

        if side == "LONG":
            # Confirmed only after candle body closes above Wave 1 end / breakout level.
            body_break = last_close > wave1_end + break_buffer

            # Momentum confirmation: recent closes pushing upward and last candle strong.
            recent_closes = [float(x) for x in recent["Close"].tolist()]
            rising_closes = len(recent_closes) >= 2 and recent_closes[-1] > recent_closes[0]

            if body_break and (strong_body or rising_closes):
                result.update({
                    "wave3_status": "CONFIRMED",
                    "wave3_confirmation_reason": (
                        "Wave 3 confirmed: candle body closed above Wave 1 end / breakout level "
                        "with bullish momentum."
                    ),
                    "wave3_confirmed": True,
                })
                return result

        if side == "SHORT":
            # Confirmed only after candle body closes below Wave 1 end / breakdown level.
            body_break = last_close < wave1_end - break_buffer

            recent_closes = [float(x) for x in recent["Close"].tolist()]
            falling_closes = len(recent_closes) >= 2 and recent_closes[-1] < recent_closes[0]

            if body_break and (strong_body or falling_closes):
                result.update({
                    "wave3_status": "CONFIRMED",
                    "wave3_confirmation_reason": (
                        "Wave 3 confirmed: candle body closed below Wave 1 end / breakdown level "
                        "with bearish momentum."
                    ),
                    "wave3_confirmed": True,
                })
                return result

        return result

    except Exception:
        return result


def _nearest_retracement_level(ratio: float, levels: List[float]) -> float:
    try:
        return min(levels, key=lambda x: abs(x - ratio))
    except Exception:
        return levels[0] if levels else 0.618


def _context_wave5_leg(
    side: str,
    df: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """
    Try to build a larger Wave 3 leg from context_bundle.
    This helps detect Wave 4 -> Wave 5 setups like:
    major swing low -> major swing high -> current Wave 4 pullback.
    """
    try:
        d = _normalize_ohlc(df)
        if d.empty:
            return None

        levels = _context_major_levels(context_bundle)
        side = str(side or "").upper()

        major_high = _safe_float(levels.get("major_swing_high"))
        major_low = _safe_float(levels.get("major_swing_low"))
        long_high = _safe_float(levels.get("long_lookback_high"))
        long_low = _safe_float(levels.get("long_lookback_low"))

        if side == "LONG":
            start = major_low or long_low
            end = major_high or long_high

            if start is None or end is None or end <= start:
                return None

            return {
                "wave3_start": float(start),
                "wave3_end": float(end),
                "wave3_length": float(end - start),
            }

        if side == "SHORT":
            start = major_high or long_high
            end = major_low or long_low

            if start is None or end is None or start <= end:
                return None

            return {
                "wave3_start": float(start),
                "wave3_end": float(end),
                "wave3_length": float(start - end),
            }

        return None

    except Exception:
        return None


# ==========================================================
# Wave 1-2 -> Wave 3 detection
# ==========================================================

def _build_bullish_wave3_candidate(
    df: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]],
    user_tf: str,
    symbol: str,
    lookback: int = DEFAULT_LOOKBACK,
) -> Optional[Dict[str, Any]]:
    """
    Bullish Wave 1-2 -> Wave 3:
    - Find impulse low -> high.
    - Current price / pullback should be in Wave 2 retracement area.
    - Entry near 0.5 / 0.618 / 0.705 / 0.786 retracement.
    - SL below Wave 1 origin / Wave 2 invalidation.
    - TP1 near Wave 1 high.
    - TP2 Wave 3 extension.
    """
    try:
        lookback_used = int(lookback or DEFAULT_LOOKBACK)
        d = _normalize_ohlc(df).tail(lookback_used).copy()
        if len(d) < 40:
            return None

        current_price = _context_execution_price(context_bundle, float(d["Close"].iloc[-1]))
        atr_val = _context_atr(context_bundle, _atr(d, 14))
        if current_price is None or atr_val is None or atr_val <= 0:
            return None

        pivots = _find_pivots(d)
        lows = pivots.get("lows", [])
        highs = pivots.get("highs", [])
        if not lows or not highs:
            return None

        # Wave 1 origin = a recent meaningful low before a later high.
        # Use the lowest low before the recent/major high.
        high_idx = pivots.get("highest_idx")
        if high_idx is None:
            return None

        high_pos = _index_position(d, high_idx)
        if high_pos <= 5:
            return None

        before_high = d.iloc[: high_pos + 1]
        low_idx = before_high["Low"].idxmin()

        wave1_start = float(before_high.loc[low_idx, "Low"])
        wave1_end = float(d.loc[high_idx, "High"])

        if wave1_end <= wave1_start:
            return None

        wave1_length = wave1_end - wave1_start
        if wave1_length < atr_val * MIN_IMPULSE_ATR_MULTIPLIER:
            return None

        # Need current price below/near wave1 high but above invalidation.
        if current_price <= wave1_start:
            return None

        # Estimate current correction ratio.
        correction_price = min(current_price, float(d["Low"].iloc[-1]))
        ratio = _retrace_ratio("LONG", wave1_start, wave1_end, correction_price)
        if ratio is None:
            return None

        # If current price has not pulled back enough, still create a limit entry at preferred zone.
        best_level = _nearest_retracement_level(ratio, WAVE2_PREFERRED_RETRACE_LEVELS)

        # Prefer 0.618 if current ratio is too shallow.
        if ratio < 0.382:
            best_level = 0.618

        entry = _retracement_price("LONG", wave1_start, wave1_end, best_level)

        if ratio < WAVE2_MIN_RETRACE:
            status_note = "Wave 2 pullback is still shallow; limit order waits deeper."
        elif ratio > WAVE2_MAX_RETRACE:
            return None
        else:
            status_note = "Wave 2 pullback is within valid retracement zone."

        # Entry should not be too far from current.
        if abs(current_price - entry) > atr_val * MAX_ENTRY_DISTANCE_ATR:
            return None

        # SL below wave1 origin with small buffer.
        stop = wave1_start - (atr_val * INVALIDATION_ATR_BUFFER)

        # TP1 = Wave 1 high / prior impulse end. If entry is too close to high, TP1 can be 1.5R fallback.
        risk = entry - stop
        if risk <= 0:
            return None

        tp1 = entry + (risk * 1.5)

        tp2_1272 = _extension_target("LONG", wave1_start, wave1_end, entry, WAVE3_EXTENSION_CONSERVATIVE)
        tp2_1618 = _extension_target("LONG", wave1_start, wave1_end, entry, WAVE3_EXTENSION_PRIMARY)

        # Use 1.618 as primary but keep 1.272 in debug.
        tp2 = tp2_1618

        rr = abs(tp2 - entry) / risk
        if rr < 2.0:
            return None

        confidence = 65

        ema_bias = _ema_bias(d)
        if ema_bias == "BULLISH":
            confidence += 10
        elif ema_bias == "BEARISH":
            confidence -= 10

        dom_side = _dominant_side(context_bundle)
        if dom_side == "LONG":
            confidence += 8
        elif dom_side == "SHORT":
            confidence -= 15

        levels = _context_major_levels(context_bundle)
        support_zone = levels.get("major_support_zone", {})
        if _zone_valid(support_zone):
            z_low = float(support_zone["low"])
            z_high = float(support_zone["high"])
            if z_low <= entry <= z_high or abs(entry - z_high) <= atr_val:
                confidence += 8

        wave3_confirm = _wave3_confirmation_status(
            side="LONG",
            df=d,
            wave1_start=wave1_start,
            wave1_end=wave1_end,
            entry=entry,
            current_price=current_price,
            atr_val=atr_val,
        )

        if wave3_confirm.get("wave3_confirmed"):
            confidence += 8

        confidence = int(round(_clamp(confidence, 0, 96)))

        return {
            "valid": True,
            "side": "LONG",
            "template": "BULLISH_WAVE_2_TO_WAVE_3",
            "wave_target": "WAVE_3",
            "wave_personality": "Impulsive",
            "status": wave3_confirm.get("wave3_status", "ANTICIPATION"),
            "wave3_status": wave3_confirm.get("wave3_status", "ANTICIPATION"),
            "wave3_confirmed": bool(wave3_confirm.get("wave3_confirmed")),
            "wave3_confirmation_reason": wave3_confirm.get("wave3_confirmation_reason", ""),
            "wave3_break_level": wave3_confirm.get("wave3_break_level"),
            "confidence": confidence,
            "symbol": symbol,
            "timeframe": user_tf,
            "lookback_used": lookback_used,
            "scan_window_family": "WAVE3",
            "scan_window_role": "SINGLE",

            "entry": _format_float(entry),
            "stop_loss": _format_float(stop),
            "tp1": _format_float(tp1),
            "tp2": _format_float(tp2),

            "risk": _format_float(risk),
            "rr": _format_float(rr, 4),
            "current_price": _format_float(current_price),
            "atr": _format_float(atr_val),

            "wave_points": {
                "wave1_start": _format_float(wave1_start),
                "wave1_end": _format_float(wave1_end),
                "wave2_estimated_end": _format_float(entry),
                "wave1_length": _format_float(wave1_length),
            },

            "fib": {
                "retracement_used": best_level,
                "current_retracement_ratio": _format_float(ratio, 4),
                "tp2_extension_used": WAVE3_EXTENSION_PRIMARY,
                "tp2_1_272": _format_float(tp2_1272),
                "tp2_1_618": _format_float(tp2_1618),
            },

            "logic": {
                "entry_logic": f"Buy Wave 2 pullback near {best_level:.3f} retracement.",
                "sl_logic": "SL below Wave 1 origin / Wave 2 invalidation.",
                "tp1_logic": "TP1 fixed at exact 1.5R from entry.",
                "tp2_logic": "TP2 projected using Wave 3 1.618 extension.",
                "note": status_note,
            },

            "confluences": [
                "Elliott bullish Wave 1-2 structure",
                f"Wave 2 retracement {best_level:.3f}",
                "Wave 3 continuation target",
            ],
        }

    except Exception:
        return None


def _build_bearish_wave3_candidate(
    df: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]],
    user_tf: str,
    symbol: str,
    lookback: int = DEFAULT_LOOKBACK,
) -> Optional[Dict[str, Any]]:
    """
    Bearish Wave 1-2 -> Wave 3:
    - Find impulse high -> low.
    - Current price / pullback should be in Wave 2 retracement area.
    - Entry near 0.5 / 0.618 / 0.705 / 0.786 retracement.
    - SL above Wave 1 origin / Wave 2 invalidation.
    - TP1 near Wave 1 low.
    - TP2 Wave 3 extension.
    """
    try:
        lookback_used = int(lookback or DEFAULT_LOOKBACK)
        d = _normalize_ohlc(df).tail(lookback_used).copy()
        if len(d) < 40:
            return None

        current_price = _context_execution_price(context_bundle, float(d["Close"].iloc[-1]))
        atr_val = _context_atr(context_bundle, _atr(d, 14))
        if current_price is None or atr_val is None or atr_val <= 0:
            return None

        pivots = _find_pivots(d)
        low_idx = pivots.get("lowest_idx")
        if low_idx is None:
            return None

        low_pos = _index_position(d, low_idx)
        if low_pos <= 5:
            return None

        before_low = d.iloc[: low_pos + 1]
        high_idx = before_low["High"].idxmax()

        wave1_start = float(before_low.loc[high_idx, "High"])
        wave1_end = float(d.loc[low_idx, "Low"])

        if wave1_start <= wave1_end:
            return None

        wave1_length = wave1_start - wave1_end
        if wave1_length < atr_val * MIN_IMPULSE_ATR_MULTIPLIER:
            return None

        # Current price should not already be above invalidation.
        if current_price >= wave1_start:
            return None

        correction_price = max(current_price, float(d["High"].iloc[-1]))
        ratio = _retrace_ratio("SHORT", wave1_start, wave1_end, correction_price)
        if ratio is None:
            return None

        best_level = _nearest_retracement_level(ratio, WAVE2_PREFERRED_RETRACE_LEVELS)

        if ratio < 0.382:
            best_level = 0.618

        entry = _retracement_price("SHORT", wave1_start, wave1_end, best_level)

        if ratio < WAVE2_MIN_RETRACE:
            status_note = "Wave 2 pullback is still shallow; limit order waits higher."
        elif ratio > WAVE2_MAX_RETRACE:
            return None
        else:
            status_note = "Wave 2 pullback is within valid retracement zone."

        if abs(current_price - entry) > atr_val * MAX_ENTRY_DISTANCE_ATR:
            return None

        # SL above wave1 origin.
        stop = wave1_start + (atr_val * INVALIDATION_ATR_BUFFER)

        risk = stop - entry
        if risk <= 0:
            return None

        tp1 = entry - (risk * 1.5)

        tp2_1272 = _extension_target("SHORT", wave1_start, wave1_end, entry, WAVE3_EXTENSION_CONSERVATIVE)
        tp2_1618 = _extension_target("SHORT", wave1_start, wave1_end, entry, WAVE3_EXTENSION_PRIMARY)

        tp2 = tp2_1618

        rr = abs(entry - tp2) / risk
        if rr < 2.0:
            return None

        confidence = 65

        ema_bias = _ema_bias(d)
        if ema_bias == "BEARISH":
            confidence += 10
        elif ema_bias == "BULLISH":
            confidence -= 10

        dom_side = _dominant_side(context_bundle)
        if dom_side == "SHORT":
            confidence += 8
        elif dom_side == "LONG":
            confidence -= 15

        levels = _context_major_levels(context_bundle)
        resistance_zone = levels.get("major_resistance_zone", {})
        if _zone_valid(resistance_zone):
            z_low = float(resistance_zone["low"])
            z_high = float(resistance_zone["high"])
            if z_low <= entry <= z_high or abs(entry - z_low) <= atr_val:
                confidence += 8

        wave3_confirm = _wave3_confirmation_status(
            side="SHORT",
            df=d,
            wave1_start=wave1_start,
            wave1_end=wave1_end,
            entry=entry,
            current_price=current_price,
            atr_val=atr_val,
        )

        if wave3_confirm.get("wave3_confirmed"):
            confidence += 8

        confidence = int(round(_clamp(confidence, 0, 96)))

        return {
            "valid": True,
            "side": "SHORT",
            "template": "BEARISH_WAVE_2_TO_WAVE_3",
            "wave_target": "WAVE_3",
            "wave_personality": "Impulsive",
            "status": wave3_confirm.get("wave3_status", "ANTICIPATION"),
            "wave3_status": wave3_confirm.get("wave3_status", "ANTICIPATION"),
            "wave3_confirmed": bool(wave3_confirm.get("wave3_confirmed")),
            "wave3_confirmation_reason": wave3_confirm.get("wave3_confirmation_reason", ""),
            "wave3_break_level": wave3_confirm.get("wave3_break_level"),
            "confidence": confidence,
            "symbol": symbol,
            "timeframe": user_tf,
            "lookback_used": lookback_used,
            "scan_window_family": "WAVE3",
            "scan_window_role": "SINGLE",

            "entry": _format_float(entry),
            "stop_loss": _format_float(stop),
            "tp1": _format_float(tp1),
            "tp2": _format_float(tp2),

            "risk": _format_float(risk),
            "rr": _format_float(rr, 4),
            "current_price": _format_float(current_price),
            "atr": _format_float(atr_val),

            "wave_points": {
                "wave1_start": _format_float(wave1_start),
                "wave1_end": _format_float(wave1_end),
                "wave2_estimated_end": _format_float(entry),
                "wave1_length": _format_float(wave1_length),
            },

            "fib": {
                "retracement_used": best_level,
                "current_retracement_ratio": _format_float(ratio, 4),
                "tp2_extension_used": WAVE3_EXTENSION_PRIMARY,
                "tp2_1_272": _format_float(tp2_1272),
                "tp2_1_618": _format_float(tp2_1618),
            },

            "logic": {
                "entry_logic": f"Short Wave 2 pullback near {best_level:.3f} retracement.",
                "sl_logic": "SL above Wave 1 origin / Wave 2 invalidation.",
                "tp1_logic": "TP1 fixed at exact 1.5R from entry.",
                "tp2_logic": "TP2 projected using Wave 3 1.618 extension.",
                "note": status_note,
            },

            "confluences": [
                "Elliott bearish Wave 1-2 structure",
                f"Wave 2 retracement {best_level:.3f}",
                "Wave 3 continuation target",
            ],
        }

    except Exception:
        return None


# ==========================================================
# Wave 4 -> Wave 5 detection
# ==========================================================

def _build_bullish_wave5_candidate(
    df: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]],
    user_tf: str,
    symbol: str,
    lookback: int = DEFAULT_LOOKBACK,
) -> Optional[Dict[str, Any]]:
    """
    Practical Wave 4 -> Wave 5 detector.

    This is intentionally conservative:
    - Requires bullish EMA or dominant LONG.
    - Uses recent impulse leg as Wave 3.
    - Entry at shallow/medium Wave 4 retracement.
    """
    try:
        lookback_used = int(lookback or DEFAULT_LOOKBACK)
        d = _normalize_ohlc(df).tail(lookback_used).copy()
        if len(d) < 60:
            return None

        current_price = _context_execution_price(context_bundle, float(d["Close"].iloc[-1]))
        atr_val = _context_atr(context_bundle, _atr(d, 14))
        if current_price is None or atr_val is None or atr_val <= 0:
            return None

        ema_bias = _ema_bias(d)
        dom_side = _dominant_side(context_bundle)

        if ema_bias != "BULLISH" and dom_side != "LONG":
            return None

        pivots = _find_pivots(d)
        recent_high = _safe_float(pivots.get("recent_high"))
        recent_low = _safe_float(pivots.get("recent_low"))
        previous_low = _safe_float(pivots.get("previous_low"))

        if recent_high is None or recent_low is None:
            return None

        # Prefer larger context Wave 3 leg if available.
        context_leg = _context_wave5_leg("LONG", d, context_bundle)

        if context_leg:
            wave3_start = context_leg["wave3_start"]
            wave3_end = context_leg["wave3_end"]
        else:
            # Fallback: approximate Wave 3 as previous low -> recent high.
            wave3_start = previous_low if previous_low is not None and previous_low < recent_high else recent_low
            wave3_end = recent_high

        if wave3_end <= wave3_start:
            return None

        wave3_length = wave3_end - wave3_start
        if wave3_length < atr_val * MIN_IMPULSE_ATR_MULTIPLIER:
            return None

        correction_price = min(current_price, float(d["Low"].iloc[-1]))
        ratio = _retrace_ratio("LONG", wave3_start, wave3_end, correction_price)
        if ratio is None:
            return None

        if ratio > WAVE4_MAX_RETRACE:
            return None

        best_level = _nearest_retracement_level(max(ratio, WAVE4_MIN_RETRACE), WAVE4_PREFERRED_RETRACE_LEVELS)

        # If current pullback is already inside valid Wave 4 retracement zone,
        # use the current/support area instead of forcing a deeper fib limit.
        if WAVE4_MIN_RETRACE <= ratio <= WAVE4_MAX_RETRACE:
            levels = _context_major_levels(context_bundle)
            support_zone = levels.get("major_support_zone", {}) or {}

            if _zone_valid(support_zone):
                z_low = _safe_float(support_zone.get("low"))
                z_high = _safe_float(support_zone.get("high"))
                if z_low is not None and z_high is not None and z_low <= current_price:
                    entry = min(current_price, z_high)
                else:
                    entry = current_price
            else:
                entry = current_price
        else:
            entry = _retracement_price("LONG", wave3_start, wave3_end, best_level)

        if abs(current_price - entry) > atr_val * MAX_ENTRY_DISTANCE_ATR:
            return None

        # Wave 4 invalidation: conservative stop below Wave 4 entry zone / recent low.
        stop_base = min(recent_low, entry)
        stop = stop_base - (atr_val * 0.60)

        risk = entry - stop
        if risk <= 0:
            return None

        # TP1 retest of Wave 3 high.
        tp1 = entry + (risk * 1.5)

        tp2_1000 = _projection_from_wave3("LONG", wave3_start, wave3_end, entry, WAVE5_EXTENSION_PRIMARY)
        tp2_1272 = _projection_from_wave3("LONG", wave3_start, wave3_end, entry, WAVE5_EXTENSION_EXTENDED)

        levels = _context_major_levels(context_bundle)
        major_res = _safe_float(levels.get("major_swing_high")) or _safe_float(levels.get("long_lookback_high"))

        if major_res is not None and major_res > tp1:
            tp2 = min(tp2_1000, major_res)
        else:
            tp2 = tp2_1000

        rr = abs(tp2 - entry) / risk
        if rr < 2.0:
            return None

        confidence = 60
        if ema_bias == "BULLISH":
            confidence += 10
        if dom_side == "LONG":
            confidence += 8
        if WAVE4_MIN_RETRACE <= ratio <= WAVE4_MAX_RETRACE:
            confidence += 8

        confidence = int(round(_clamp(confidence, 0, 92)))

        return {
            "valid": True,
            "side": "LONG",
            "template": "BULLISH_WAVE_4_TO_WAVE_5",
            "wave_target": "WAVE_5",
            "wave_personality": "Impulsive",
            "status": "CONFIRMED" if confidence >= HIGH_CONFIDENCE else "POSSIBLE",
            "confidence": confidence,
            "symbol": symbol,
            "timeframe": user_tf,
            "lookback_used": lookback_used,
            "scan_window_family": "WAVE5",
            "scan_window_role": "SINGLE",

            "entry": _format_float(entry),
            "stop_loss": _format_float(stop),
            "tp1": _format_float(tp1),
            "tp2": _format_float(tp2),

            "risk": _format_float(risk),
            "rr": _format_float(rr, 4),
            "current_price": _format_float(current_price),
            "atr": _format_float(atr_val),

            "wave_points": {
                "wave3_start": _format_float(wave3_start),
                "wave3_end": _format_float(wave3_end),
                "wave4_estimated_end": _format_float(entry),
                "wave3_length": _format_float(wave3_length),
            },

            "fib": {
                "retracement_used": best_level,
                "current_retracement_ratio": _format_float(ratio, 4),
                "tp2_projection_used": WAVE5_EXTENSION_PRIMARY,
                "tp2_1_000": _format_float(tp2_1000),
                "tp2_1_272": _format_float(tp2_1272),
            },

            "logic": {
                "entry_logic": f"Buy Wave 4 pullback near {best_level:.3f} retracement.",
                "sl_logic": "SL below Wave 4 correction invalidation zone.",
                "tp1_logic": "TP1 fixed at exact 1.5R from entry.",
                "tp2_logic": "TP2 projected using Wave 5 projection.",
                "note": "Wave 4 pullback candidate for Wave 5 continuation.",
            },

            "confluences": [
                "Elliott bullish Wave 4 pullback",
                f"Wave 4 retracement {best_level:.3f}",
                "Wave 5 continuation target",
            ],
        }

    except Exception:
        return None


def _build_bearish_wave5_candidate(
    df: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]],
    user_tf: str,
    symbol: str,
    lookback: int = DEFAULT_LOOKBACK,
) -> Optional[Dict[str, Any]]:
    """
    Practical Bearish Wave 4 -> Wave 5 detector.
    """
    try:
        lookback_used = int(lookback or DEFAULT_LOOKBACK)
        d = _normalize_ohlc(df).tail(lookback_used).copy()
        if len(d) < 60:
            return None

        current_price = _context_execution_price(context_bundle, float(d["Close"].iloc[-1]))
        atr_val = _context_atr(context_bundle, _atr(d, 14))
        if current_price is None or atr_val is None or atr_val <= 0:
            return None

        ema_bias = _ema_bias(d)
        dom_side = _dominant_side(context_bundle)

        if ema_bias != "BEARISH" and dom_side != "SHORT":
            return None

        pivots = _find_pivots(d)
        recent_high = _safe_float(pivots.get("recent_high"))
        recent_low = _safe_float(pivots.get("recent_low"))
        previous_high = _safe_float(pivots.get("previous_high"))

        if recent_high is None or recent_low is None:
            return None

        # Prefer larger context Wave 3 leg if available.
        context_leg = _context_wave5_leg("SHORT", d, context_bundle)

        if context_leg:
            wave3_start = context_leg["wave3_start"]
            wave3_end = context_leg["wave3_end"]
        else:
            # Fallback: approximate bearish Wave 3 as previous high -> recent low.
            wave3_start = previous_high if previous_high is not None and previous_high > recent_low else recent_high
            wave3_end = recent_low

        if wave3_start <= wave3_end:
            return None

        wave3_length = wave3_start - wave3_end
        if wave3_length < atr_val * MIN_IMPULSE_ATR_MULTIPLIER:
            return None

        correction_price = max(current_price, float(d["High"].iloc[-1]))
        ratio = _retrace_ratio("SHORT", wave3_start, wave3_end, correction_price)
        if ratio is None:
            return None

        if ratio > WAVE4_MAX_RETRACE:
            return None

        best_level = _nearest_retracement_level(max(ratio, WAVE4_MIN_RETRACE), WAVE4_PREFERRED_RETRACE_LEVELS)

        # If current pullback is already inside valid Wave 4 retracement zone,
        # use the current/resistance area instead of forcing a higher fib limit.
        if WAVE4_MIN_RETRACE <= ratio <= WAVE4_MAX_RETRACE:
            levels = _context_major_levels(context_bundle)
            resistance_zone = levels.get("major_resistance_zone", {}) or {}

            if _zone_valid(resistance_zone):
                z_low = _safe_float(resistance_zone.get("low"))
                z_high = _safe_float(resistance_zone.get("high"))

                if z_low is not None and z_high is not None and z_high >= current_price:
                    entry = max(current_price, z_low)
                else:
                    entry = current_price
            else:
                entry = current_price
        else:
            entry = _retracement_price("SHORT", wave3_start, wave3_end, best_level)

        if abs(current_price - entry) > atr_val * MAX_ENTRY_DISTANCE_ATR:
            return None

        stop_base = max(recent_high, entry)
        stop = stop_base + (atr_val * 0.60)

        risk = stop - entry
        if risk <= 0:
            return None

        tp1 = entry - (risk * 1.5)

        tp2_1000 = _projection_from_wave3("SHORT", wave3_start, wave3_end, entry, WAVE5_EXTENSION_PRIMARY)
        tp2_1272 = _projection_from_wave3("SHORT", wave3_start, wave3_end, entry, WAVE5_EXTENSION_EXTENDED)

        levels = _context_major_levels(context_bundle)
        major_sup = _safe_float(levels.get("major_swing_low")) or _safe_float(levels.get("long_lookback_low"))

        if major_sup is not None and major_sup < tp1:
            tp2 = max(tp2_1000, major_sup)
        else:
            tp2 = tp2_1000

        rr = abs(entry - tp2) / risk
        if rr < 2.0:
            return None

        confidence = 60
        if ema_bias == "BEARISH":
            confidence += 10
        if dom_side == "SHORT":
            confidence += 8
        if WAVE4_MIN_RETRACE <= ratio <= WAVE4_MAX_RETRACE:
            confidence += 8

        confidence = int(round(_clamp(confidence, 0, 92)))

        return {
            "valid": True,
            "side": "SHORT",
            "template": "BEARISH_WAVE_4_TO_WAVE_5",
            "wave_target": "WAVE_5",
            "wave_personality": "Impulsive",
            "status": "CONFIRMED" if confidence >= HIGH_CONFIDENCE else "POSSIBLE",
            "confidence": confidence,
            "symbol": symbol,
            "timeframe": user_tf,
            "lookback_used": lookback_used,
            "scan_window_family": "WAVE5",
            "scan_window_role": "SINGLE",

            "entry": _format_float(entry),
            "stop_loss": _format_float(stop),
            "tp1": _format_float(tp1),
            "tp2": _format_float(tp2),

            "risk": _format_float(risk),
            "rr": _format_float(rr, 4),
            "current_price": _format_float(current_price),
            "atr": _format_float(atr_val),

            "wave_points": {
                "wave3_start": _format_float(wave3_start),
                "wave3_end": _format_float(wave3_end),
                "wave4_estimated_end": _format_float(entry),
                "wave3_length": _format_float(wave3_length),
            },

            "fib": {
                "retracement_used": best_level,
                "current_retracement_ratio": _format_float(ratio, 4),
                "tp2_projection_used": WAVE5_EXTENSION_PRIMARY,
                "tp2_1_000": _format_float(tp2_1000),
                "tp2_1_272": _format_float(tp2_1272),
            },

            "logic": {
                "entry_logic": f"Short Wave 4 pullback near {best_level:.3f} retracement.",
                "sl_logic": "SL above Wave 4 correction invalidation zone.",
                "tp1_logic": "TP1 fixed at exact 1.5R from entry.",
                "tp2_logic": "TP2 projected using Wave 5 projection.",
                "note": "Bearish Wave 4 pullback candidate for Wave 5 continuation.",
            },

            "confluences": [
                "Elliott bearish Wave 4 pullback",
                f"Wave 4 retracement {best_level:.3f}",
                "Wave 5 continuation target",
            ],
        }

    except Exception:
        return None

def _scan_wave_builders_multi_window(
    *,
    df: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]],
    user_tf: str,
    symbol: str,
    builders: List[Any],
    windows: List[int],
    primary_window: int,
    family: str,
) -> List[Dict[str, Any]]:
    """
    Run Elliott builders across multiple lookback windows.

    This is Elliott-only.
    It does not touch normal setup logic.
    It does not let the longest window own the signal alone.
    """
    templates: List[Dict[str, Any]] = []

    try:
        for window in windows:
            for builder in builders:
                c = builder(
                    df=df,
                    context_bundle=context_bundle,
                    user_tf=user_tf,
                    symbol=symbol,
                    lookback=window,
                )

                if not c or not c.get("valid"):
                    continue

                c["lookback_used"] = int(window)
                c["scan_window_family"] = str(family or "").upper()
                c["scan_window_role"] = (
                    "PRIMARY" if int(window) == int(primary_window) else "SECONDARY"
                )

                templates.append(c)

        return templates

    except Exception:
        return templates


def _apply_multi_window_confirmation(templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adjust Elliott confidence based on agreement across scan windows.

    Rules:
    - Same side across more windows = confidence boost.
    - Opposite sides across windows = confidence penalty.
    - Long/background window alone = not strong enough.
    """
    try:
        if not templates:
            return templates

        by_family_side: Dict[Tuple[str, str], List[int]] = {}

        for t in templates:
            family = str(t.get("scan_window_family") or "").upper()
            side = str(t.get("side") or "").upper()
            window = int(t.get("lookback_used") or 0)

            if family and side in ["LONG", "SHORT"] and window > 0:
                by_family_side.setdefault((family, side), []).append(window)

        family_sides: Dict[str, set] = {}
        for family, side in by_family_side.keys():
            family_sides.setdefault(family, set()).add(side)

        for t in templates:
            family = str(t.get("scan_window_family") or "").upper()
            side = str(t.get("side") or "").upper()
            window = int(t.get("lookback_used") or 0)
            confidence = int(t.get("confidence") or 0)

            same_side_windows = sorted(set(by_family_side.get((family, side), [])))
            opposite_conflict = len(family_sides.get(family, set())) > 1

            t["multi_window_same_side_count"] = len(same_side_windows)
            t["multi_window_same_side_windows"] = same_side_windows
            t["multi_window_conflict"] = bool(opposite_conflict)

            if opposite_conflict:
                t["confidence"] = int(_clamp(confidence - 12, 0, 96))
                t["multi_window_note"] = (
                    f"{family} has conflicting LONG/SHORT signals across scan windows. "
                    "Treat this Elliott setup as lower confidence."
                )
                continue

            if len(same_side_windows) >= 3:
                t["confidence"] = int(_clamp(confidence + 8, 0, 96))
                t["multi_window_note"] = (
                    f"{family} same-side Elliott setup appears across windows {same_side_windows}."
                )

            elif len(same_side_windows) == 2:
                t["confidence"] = int(_clamp(confidence + 4, 0, 96))
                t["multi_window_note"] = (
                    f"{family} same-side Elliott setup appears across 2 windows {same_side_windows}."
                )

            else:
                # Single-window only.
                # For Wave 3, 300-only is background-only. For Wave 5, 220-only is also weaker.
                if (
                    (family == "WAVE3" and window >= max(WAVE3_SCAN_WINDOWS))
                    or (family == "WAVE5" and window >= max(WAVE5_SCAN_WINDOWS))
                ):
                    t["confidence"] = int(_clamp(confidence - 10, 0, 96))
                    t["multi_window_note"] = (
                        f"{family} setup appears only on the longest/background window. "
                        "Do not treat it as a strong direct entry signal."
                    )
                else:
                    t["confidence"] = int(_clamp(confidence - 4, 0, 96))
                    t["multi_window_note"] = (
                        f"{family} setup appears only on one scan window. "
                        "Require stronger confirmation."
                    )

        return templates

    except Exception:
        return templates

# ==========================================================
# Candidate selection
# ==========================================================

def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[int, int, float]:
    """
    Lower is better for first two fields.
    Priority:
    1) Wave 2 -> Wave 3
    2) Wave 4 -> Wave 5
    3) Higher confidence
    """
    template = str(candidate.get("template") or "").upper()
    side = str(candidate.get("side") or "").upper()
    confidence = int(candidate.get("confidence") or 0)

    if "WAVE_2_TO_WAVE_3" in template:
        priority = 1
    elif "WAVE_4_TO_WAVE_5" in template:
        priority = 2
    else:
        priority = 9

    # Prefer confirmed over possible.
    status = str(candidate.get("status") or "").upper()
    status_rank = 0 if status == "CONFIRMED" else 1

    # Negative confidence because sort ascending.
    return (priority, status_rank, -confidence)


def detect_wave_templates(
    df_micro: pd.DataFrame,
    context_bundle: Optional[Dict[str, Any]] = None,
    user_tf: str = "15m",
    symbol: str = "",
    include_wave5: bool = True,
) -> Dict[str, Any]:
    """
    Public function.

    Returns:
    {
        "has_template": bool,
        "selected_template": dict,
        "templates": [dict...],
        "status": "...",
        "reason": "...",
        "confirmation_text": "..."
    }
    """
    try:
        d = _normalize_ohlc(df_micro)
        if d.empty or len(d) < 40:
            return {
                "has_template": False,
                "selected_template": {},
                "templates": [],
                "status": "NO_DATA",
                "reason": "Not enough OHLCV data for wave template detection.",
                "confirmation_text": "",
            }

        hard_block, hard_reason = _context_hard_block(context_bundle)
        if hard_block:
            return {
                "has_template": False,
                "selected_template": {},
                "templates": [],
                "status": "HARD_CONTEXT_BLOCK",
                "reason": hard_reason,
                "confirmation_text": "",
            }

        templates: List[Dict[str, Any]] = []

        # Always check both directions.
        # Multi-window scan helps separate:
        # - local-only Elliott ideas
        # - multi-window confirmed Elliott ideas
        # - old/background-only Elliott ideas
        templates.extend(
            _scan_wave_builders_multi_window(
                df=d,
                context_bundle=context_bundle,
                user_tf=user_tf,
                symbol=symbol,
                builders=[
                    _build_bullish_wave3_candidate,
                    _build_bearish_wave3_candidate,
                ],
                windows=WAVE3_SCAN_WINDOWS,
                primary_window=WAVE3_PRIMARY_WINDOW,
                family="WAVE3",
            )
        )

        if include_wave5:
            templates.extend(
                _scan_wave_builders_multi_window(
                    df=d,
                    context_bundle=context_bundle,
                    user_tf=user_tf,
                    symbol=symbol,
                    builders=[
                        _build_bullish_wave5_candidate,
                        _build_bearish_wave5_candidate,
                    ],
                    windows=WAVE5_SCAN_WINDOWS,
                    primary_window=WAVE5_PRIMARY_WINDOW,
                    family="WAVE5",
                )
            )

        templates = _apply_multi_window_confirmation(templates)

        if not templates:
            return {
                "has_template": False,
                "selected_template": {},
                "templates": [],
                "status": "NO_VALID_WAVE_TEMPLATE",
                "reason": "No valid Wave 2→3 or Wave 4→5 template detected.",
                "confirmation_text": "",
            }

        templates.sort(key=_candidate_sort_key)
        selected = templates[0]

        return {
            "has_template": True,
            "selected_template": selected,
            "templates": templates,
            "status": "WAVE_TEMPLATE_DETECTED",
            "reason": f"Selected {selected.get('template')} with confidence {selected.get('confidence')}.",
            "confirmation_text": format_wave_template_confirmation(selected),
        }

    except Exception as e:
        return {
            "has_template": False,
            "selected_template": {},
            "templates": [],
            "status": "WAVE_TEMPLATE_ERROR",
            "reason": f"Wave template detection error: {str(e)}",
            "confirmation_text": "",
        }


def format_wave_template_confirmation(template: Optional[Dict[str, Any]]) -> str:
    """
    Text block to print in market context / signal debug.
    """
    try:
        if not template:
            return "--- ELLIOTT WAVE SETUP CONFIRMATION ---\nStatus: NOT DETECTED"

        wave_points = template.get("wave_points", {}) or {}
        fib = template.get("fib", {}) or {}
        logic = template.get("logic", {}) or {}

        lines = [
            "--- ELLIOTT WAVE SETUP CONFIRMATION ---",
            f"Wave Template: {template.get('template', 'N/A')}",
            f"Status: {template.get('status', 'N/A')}",
            f"Confidence: {template.get('confidence', 'N/A')}",
            f"Wave 3 Status: {template.get('wave3_status', 'N/A')}",
            f"Wave 3 Confirmed: {template.get('wave3_confirmed', 'N/A')}",
            f"Wave 3 Confirmation Reason: {template.get('wave3_confirmation_reason', '')}",
            f"Wave 3 Break Level: {template.get('wave3_break_level', '')}",
            f"Side: {template.get('side', 'N/A')}",
            f"Wave Target: {template.get('wave_target', 'N/A')}",
            f"Wave Personality: {template.get('wave_personality', 'N/A')}",
            f"Lookback Used: {template.get('lookback_used', 'N/A')}",
            f"Scan Window Family: {template.get('scan_window_family', 'N/A')}",
            f"Scan Window Role: {template.get('scan_window_role', 'N/A')}",
            f"Multi-Window Same Side Windows: {template.get('multi_window_same_side_windows', [])}",
            f"Multi-Window Conflict: {template.get('multi_window_conflict', False)}",
            f"Multi-Window Note: {template.get('multi_window_note', '')}",
            f"Current Price: {template.get('current_price', '')}",
            f"Selected Entry: {template.get('entry', '')}",
            f"Stop Loss: {template.get('stop_loss', '')}",
            f"TP1: {template.get('tp1', '')}",
            f"TP2: {template.get('tp2', '')}",
            f"Risk: {template.get('risk', '')}",
            f"RR: {template.get('rr', '')}",
        ]

        # Wave 2 -> Wave 3 fields
        if "wave1_start" in wave_points:
            lines.extend([
                f"Wave 1 Start / Invalidation: {wave_points.get('wave1_start')}",
                f"Wave 1 End / TP1 Reference: {wave_points.get('wave1_end')}",
                f"Wave 2 Estimated End / Entry Zone: {wave_points.get('wave2_estimated_end')}",
                f"Wave 1 Length: {wave_points.get('wave1_length')}",
                f"Fib Retracement Used: {fib.get('retracement_used')}",
                f"Current Retracement Ratio: {fib.get('current_retracement_ratio')}",
                f"Fib Extension Used For TP2: {fib.get('tp2_extension_used')}",
                f"TP2 1.272 Projection: {fib.get('tp2_1_272')}",
                f"TP2 1.618 Projection: {fib.get('tp2_1_618')}",
            ])

        # Wave 4 -> Wave 5 fields
        if "wave3_start" in wave_points:
            lines.extend([
                f"Wave 3 Start: {wave_points.get('wave3_start')}",
                f"Wave 3 End / TP1 Reference: {wave_points.get('wave3_end')}",
                f"Wave 4 Estimated End / Entry Zone: {wave_points.get('wave4_estimated_end')}",
                f"Wave 3 Length: {wave_points.get('wave3_length')}",
                f"Fib Retracement Used: {fib.get('retracement_used')}",
                f"Current Retracement Ratio: {fib.get('current_retracement_ratio')}",
                f"TP2 Projection Used: {fib.get('tp2_projection_used')}",
                f"TP2 1.000 Projection: {fib.get('tp2_1_000')}",
                f"TP2 1.272 Projection: {fib.get('tp2_1_272')}",
            ])

        lines.extend([
            f"Entry Logic: {logic.get('entry_logic', '')}",
            f"SL Logic: {logic.get('sl_logic', '')}",
            f"TP1 Logic: {logic.get('tp1_logic', '')}",
            f"TP2 Logic: {logic.get('tp2_logic', '')}",
            f"Setup Note: {logic.get('note', '')}",
        ])

        return "\n".join(lines)

    except Exception as e:
        return (
            "--- ELLIOTT WAVE SETUP CONFIRMATION ---\n"
            f"Formatting Error: {str(e)}"
        )


def compact_wave_templates_for_debug(result: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    """
    Compact form for API/debug output.
    """
    out: List[Dict[str, Any]] = []
    try:
        templates = result.get("templates") or []
        for t in templates[:limit]:
            out.append({
                "template": t.get("template"),
                "side": t.get("side"),
                "status": t.get("status"),
                "confidence": t.get("confidence"),
                "entry": t.get("entry"),
                "stop_loss": t.get("stop_loss"),
                "tp1": t.get("tp1"),
                "tp2": t.get("tp2"),
                "rr": t.get("rr"),
                "wave_target": t.get("wave_target"),
                "retracement_used": (t.get("fib") or {}).get("retracement_used"),
                "wave3_status": t.get("wave3_status"),
                "wave3_confirmed": t.get("wave3_confirmed"),
                "wave3_break_level": t.get("wave3_break_level"),
                "lookback_used": t.get("lookback_used"),
                "scan_window_family": t.get("scan_window_family"),
                "scan_window_role": t.get("scan_window_role"),
                "multi_window_same_side_count": t.get("multi_window_same_side_count"),
                "multi_window_same_side_windows": t.get("multi_window_same_side_windows"),
                "multi_window_conflict": t.get("multi_window_conflict"),
                "multi_window_note": t.get("multi_window_note"),
                "extension_or_projection": (
                    (t.get("fib") or {}).get("tp2_extension_used")
                    or (t.get("fib") or {}).get("tp2_projection_used")
                ),
            })
    except Exception:
        pass
    return out