# app/models/trend_regime.py

"""
Backend Trend Regime Observer

Purpose:
- Detect the preferred trading side and market regime.
- Use OHLC structure, EMA bias, swing structure, candle momentum, and structured context_bundle.
- This module does NOT create trades.
- This module does NOT create entries, SL, or TP.
- setup_planner.py is responsible for candidates.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==========================================================
# Utility helpers
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
        text = str(value).replace(",", "").strip()
        if not text:
            return default
        v = float(text)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def _fmt_price(value: Any) -> str:
    value = _safe_float(value)
    if value is None:
        return "N/A"

    av = abs(value)

    if av >= 1000:
        return f"{value:.2f}"
    if av >= 100:
        return f"{value:.2f}"
    if av >= 10:
        return f"{value:.3f}"
    if av >= 1:
        return f"{value:.4f}"
    if av >= 0.01:
        return f"{value:.5f}"
    if av >= 0.0001:
        return f"{value:.7f}"
    return f"{value:.10f}"


def _normalize_ohlc(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Make sure dataframe has Open/High/Low/Close columns as numeric values.
    This function is intentionally defensive because different loaders may
    return lowercase columns.
    """
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
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr_val = tr.rolling(period).mean().iloc[-1]
        return float(atr_val) if pd.notna(atr_val) else None

    except Exception:
        return None


# ==========================================================
# Context bundle helpers
# ==========================================================

def _extract_context_flags(context_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Read structured flags from context_bundle.py.

    This keeps trend_regime.py backward compatible:
    if context_bundle is missing, all fields default safely.
    """
    flags = {
        "available": False,
        "choch": False,
        "dominant_wait": False,
        "dominant_preferred_side": None,
        "is_choppy": False,
        "both_rejections": False,
        "support_rejection": False,
        "resistance_rejection": False,
        "elliott_fib_usable": True,
        "fib_entry_source": "unknown",
        "near_major_support": False,
        "near_major_resistance": False,
        "notes": [],
    }

    try:
        if not context_bundle or not isinstance(context_bundle, dict):
            return flags

        flags["available"] = True

        derived = context_bundle.get("derived", {}) or {}
        rejection = context_bundle.get("rejection", {}) or {}
        dominant = context_bundle.get("dominant_trend", {}) or {}
        fib = context_bundle.get("fibonacci", {}) or {}

        flags["choch"] = bool(derived.get("choch") or dominant.get("choch"))
        flags["dominant_wait"] = bool(derived.get("dominant_wait"))
        flags["dominant_preferred_side"] = (
            derived.get("preferred_side_from_dominant")
            or dominant.get("preferred_trade_side")
        )

        flags["is_choppy"] = bool(derived.get("is_choppy"))
        flags["both_rejections"] = bool(
            derived.get("both_support_and_resistance_rejection")
            or rejection.get("both_rejections")
        )

        flags["support_rejection"] = bool(
            derived.get("has_support_rejection")
            or rejection.get("support_rejection_confirmed")
        )
        flags["resistance_rejection"] = bool(
            derived.get("has_resistance_rejection")
            or rejection.get("resistance_rejection_confirmed")
        )

        elliott = fib.get("elliott", {}) or {}
        flags["elliott_fib_usable"] = bool(
            derived.get("elliott_fib_usable")
            if "elliott_fib_usable" in derived
            else elliott.get("usable_for_entry", True)
        )
        flags["fib_entry_source"] = derived.get("fib_entry_source") or "unknown"

        flags["near_major_support"] = bool(derived.get("near_major_support"))
        flags["near_major_resistance"] = bool(derived.get("near_major_resistance"))

        notes = derived.get("notes") or []
        if isinstance(notes, list):
            flags["notes"] = [str(n) for n in notes[:6]]

        return flags

    except Exception as e:
        flags["notes"].append(f"Context flag parse error: {str(e)}")
        return flags


def _dominant_side_to_normal(side: Any) -> str:
    s = str(side or "").upper().strip()
    if "LONG" in s or "BULL" in s:
        return "LONG"
    if "SHORT" in s or "BEAR" in s:
        return "SHORT"
    if "WAIT" in s or "NONE" in s:
        return "WAIT"
    return "UNKNOWN"


# ==========================================================
# Swing / structure helpers
# ==========================================================

def _find_recent_swings(
    df: pd.DataFrame,
    window: int = 3,
    lookback: int = 100
) -> Dict[str, Any]:
    """
    Detect recent pivot highs/lows.
    This is not Elliott counting. It is only a structure observer.
    """
    d = _normalize_ohlc(df).tail(lookback).copy()

    result = {
        "recent_swing_high": None,
        "previous_swing_high": None,
        "recent_swing_low": None,
        "previous_swing_low": None,
        "structure": "UNKNOWN",
        "highs": [],
        "lows": [],
    }

    try:
        if len(d) < max(15, window * 4):
            return result

        highs: List[Tuple[Any, float]] = []
        lows: List[Tuple[Any, float]] = []

        h_values = d["High"].values
        l_values = d["Low"].values
        idx_values = list(d.index)

        for i in range(window, len(d) - window):
            h_slice = h_values[i - window:i + window + 1]
            l_slice = l_values[i - window:i + window + 1]

            if h_values[i] >= np.nanmax(h_slice):
                highs.append((idx_values[i], float(h_values[i])))

            if l_values[i] <= np.nanmin(l_slice):
                lows.append((idx_values[i], float(l_values[i])))

        # Fallback if pivots are too few
        if not highs:
            prev = d.iloc[:-1] if len(d) > 10 else d
            if not prev.empty:
                highs.append((prev["High"].idxmax(), float(prev["High"].max())))

        if not lows:
            prev = d.iloc[:-1] if len(d) > 10 else d
            if not prev.empty:
                lows.append((prev["Low"].idxmin(), float(prev["Low"].min())))

        result["highs"] = highs
        result["lows"] = lows

        if highs:
            result["recent_swing_high"] = highs[-1][1]
        if len(highs) >= 2:
            result["previous_swing_high"] = highs[-2][1]

        if lows:
            result["recent_swing_low"] = lows[-1][1]
        if len(lows) >= 2:
            result["previous_swing_low"] = lows[-2][1]

        rsh = result["recent_swing_high"]
        psh = result["previous_swing_high"]
        rsl = result["recent_swing_low"]
        psl = result["previous_swing_low"]

        if all(v is not None for v in [rsh, psh, rsl, psl]):
            if rsh > psh and rsl > psl:
                result["structure"] = "HH_HL"
            elif rsh < psh and rsl < psl:
                result["structure"] = "LH_LL"
            elif rsh > psh and rsl < psl:
                result["structure"] = "EXPANDING"
            elif rsh < psh and rsl > psl:
                result["structure"] = "COMPRESSING"
            else:
                result["structure"] = "RANGE"

        return result

    except Exception:
        return result


# ==========================================================
# Timeframe bias / candle momentum
# ==========================================================

def _timeframe_bias(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    d = _normalize_ohlc(df)

    out = {
        "name": name,
        "bias": "UNKNOWN",
        "close": None,
        "ema50": None,
        "ema200": None,
        "bull_score": 0.0,
        "bear_score": 0.0,
        "reasons": [],
    }

    try:
        if len(d) < 30:
            out["reasons"].append(f"{name}: not enough candles")
            return out

        close = float(d["Close"].iloc[-1])
        ema50 = _ema(d["Close"], 50)
        ema200 = _ema(d["Close"], 200)

        out["close"] = close
        out["ema50"] = ema50
        out["ema200"] = ema200

        bull = 0.0
        bear = 0.0

        if ema200 is not None:
            if close > ema200:
                bull += 0.45
                out["reasons"].append(f"{name}: close above EMA200")
            else:
                bear += 0.45
                out["reasons"].append(f"{name}: close below EMA200")

        if ema50 is not None:
            if close > ema50:
                bull += 0.20
                out["reasons"].append(f"{name}: close above EMA50")
            else:
                bear += 0.20
                out["reasons"].append(f"{name}: close below EMA50")

        if ema50 is not None and ema200 is not None:
            if ema50 > ema200:
                bull += 0.25
                out["reasons"].append(f"{name}: EMA50 above EMA200")
            elif ema50 < ema200:
                bear += 0.25
                out["reasons"].append(f"{name}: EMA50 below EMA200")

        # Small momentum component from latest closed candle
        if len(d) >= 2:
            prev_close = float(d["Close"].iloc[-2])
            if close > prev_close:
                bull += 0.10
            elif close < prev_close:
                bear += 0.10

        out["bull_score"] = _clamp(bull)
        out["bear_score"] = _clamp(bear)

        if bull - bear >= 0.20:
            out["bias"] = "BULLISH"
        elif bear - bull >= 0.20:
            out["bias"] = "BEARISH"
        else:
            out["bias"] = "MIXED"

        return out

    except Exception as e:
        out["reasons"].append(f"{name}: bias error: {str(e)}")
        return out


def _latest_candle_momentum(df: pd.DataFrame) -> Dict[str, Any]:
    d = _normalize_ohlc(df)

    out = {
        "direction": "NEUTRAL",
        "strong_body": False,
        "atr_expansion": False,
        "volume_expansion": False,
        "body_pct_of_range": 0.0,
        "range_to_atr": 0.0,
        "reasons": [],
    }

    try:
        if len(d) < 20:
            return out

        last = d.iloc[-1]

        open_ = float(last["Open"])
        high = float(last["High"])
        low = float(last["Low"])
        close = float(last["Close"])

        candle_range = max(high - low, 0.0)
        body = close - open_
        abs_body = abs(body)

        if candle_range > 0:
            body_pct = abs_body / candle_range
        else:
            body_pct = 0.0

        out["body_pct_of_range"] = float(body_pct)
        out["strong_body"] = body_pct >= 0.55

        if body > 0:
            out["direction"] = "BULLISH"
        elif body < 0:
            out["direction"] = "BEARISH"

        atr_val = _atr(d, 14)
        if atr_val and atr_val > 0:
            range_to_atr = candle_range / atr_val
            out["range_to_atr"] = float(range_to_atr)
            out["atr_expansion"] = range_to_atr >= 1.15

        if "Volume" in d.columns and len(d) >= 25:
            last_vol = _safe_float(d["Volume"].iloc[-1])
            avg_vol = _safe_float(d["Volume"].tail(21).iloc[:-1].mean())
            if last_vol is not None and avg_vol is not None and avg_vol > 0:
                out["volume_expansion"] = last_vol >= avg_vol * 1.20

        if out["strong_body"]:
            out["reasons"].append(f"Strong {out['direction'].lower()} candle body detected")

        if out["atr_expansion"]:
            out["reasons"].append("Latest candle range expanded vs ATR")

        if out["volume_expansion"]:
            out["reasons"].append("Volume expansion detected")

        return out

    except Exception:
        return out


# ==========================================================
# Pullback / breakout observer
# ==========================================================

def _pullback_state(df: pd.DataFrame) -> Dict[str, Any]:
    d = _normalize_ohlc(df).tail(100)

    out = {
        "bullish_pullback": False,
        "bearish_pullback": False,
        "bull_retrace": None,
        "bear_retrace": None,
        "lookback_high": None,
        "lookback_low": None,
        "reasons": [],
    }

    try:
        if len(d) < 25:
            return out

        close = float(d["Close"].iloc[-1])
        lookback_high = float(d["High"].max())
        lookback_low = float(d["Low"].min())
        rng = lookback_high - lookback_low

        out["lookback_high"] = lookback_high
        out["lookback_low"] = lookback_low

        if rng <= 0:
            return out

        # In bullish context, current price pulling down from recent high.
        bull_retrace = (lookback_high - close) / rng

        # In bearish context, current price pulling up from recent low.
        bear_retrace = (close - lookback_low) / rng

        out["bull_retrace"] = float(bull_retrace)
        out["bear_retrace"] = float(bear_retrace)

        if 0.236 <= bull_retrace <= 0.786:
            out["bullish_pullback"] = True
            out["reasons"].append(
                f"Bullish-style correction depth {bull_retrace:.2f} within 0.236-0.786 zone"
            )

        if 0.236 <= bear_retrace <= 0.786:
            out["bearish_pullback"] = True
            out["reasons"].append(
                f"Bearish-style correction depth {bear_retrace:.2f} within 0.236-0.786 zone"
            )

        return out

    except Exception:
        return out

# ==========================================================
# Multi-timeframe direction / CHoCH helpers
# ==========================================================

def _bias_to_direction(bias: Any) -> str:
    """
    Convert BULLISH/BEARISH/MIXED/UNKNOWN style bias into LONG/SHORT/NEUTRAL.
    """
    b = str(bias or "").upper().strip()

    if b in ["BULLISH", "HH_HL", "LONG"]:
        return "LONG"

    if b in ["BEARISH", "LH_LL", "SHORT"]:
        return "SHORT"

    return "NEUTRAL"


def _structure_to_direction(structure: Any) -> str:
    s = str(structure or "").upper().strip()

    if s == "HH_HL":
        return "LONG"

    if s == "LH_LL":
        return "SHORT"

    return "NEUTRAL"


def _opposite_side(side: str) -> str:
    s = str(side or "").upper().strip()
    if s == "LONG":
        return "SHORT"
    if s == "SHORT":
        return "LONG"
    return "NEUTRAL"


def _direction_agreement_score(*directions: str) -> Dict[str, Any]:
    clean = [str(d or "").upper() for d in directions if str(d or "").upper() in ["LONG", "SHORT"]]

    long_count = clean.count("LONG")
    short_count = clean.count("SHORT")

    if long_count > short_count:
        primary = "LONG"
        agreement = long_count / max(1, len(clean))
    elif short_count > long_count:
        primary = "SHORT"
        agreement = short_count / max(1, len(clean))
    else:
        primary = "NEUTRAL"
        agreement = 0.0

    return {
        "primary": primary,
        "agreement": float(agreement),
        "long_count": int(long_count),
        "short_count": int(short_count),
        "total_directional": int(len(clean)),
    }


def _detect_swing_break(
    *,
    side_to_break: str,
    current_price: float,
    swings: Dict[str, Any],
    atr_val: Optional[float],
) -> Dict[str, Any]:
    """
    Detect whether price broke a recent structure level.

    side_to_break:
      - "BEARISH" means check break below recent swing low
      - "BULLISH" means check break above recent swing high
    """
    out = {
        "broke": False,
        "broken_level": None,
        "break_direction": "NONE",
    }

    try:
        buffer = float(atr_val or 0.0) * 0.10

        if str(side_to_break).upper() == "BEARISH":
            recent_low = _safe_float(swings.get("recent_swing_low"))
            if recent_low is not None and current_price < recent_low - buffer:
                out.update({
                    "broke": True,
                    "broken_level": recent_low,
                    "break_direction": "BEARISH",
                })

        elif str(side_to_break).upper() == "BULLISH":
            recent_high = _safe_float(swings.get("recent_swing_high"))
            if recent_high is not None and current_price > recent_high + buffer:
                out.update({
                    "broke": True,
                    "broken_level": recent_high,
                    "break_direction": "BULLISH",
                })

        return out

    except Exception:
        return out

def _detect_choch_retest(
    *,
    df: Optional[pd.DataFrame],
    broken_level: Any,
    break_direction: str,
    atr_val: Optional[float],
    lookback: int = 24,
) -> Dict[str, Any]:
    """Detect a later closed-candle retest of a CHoCH level."""
    out = {
        "status": "NOT_AVAILABLE",
        "break_age_bars": None,
        "reason": "Retest evidence is unavailable.",
    }

    try:
        level = _safe_float(broken_level)
        atr = _safe_float(atr_val)
        direction = str(
            break_direction or "NONE"
        ).upper()

        # main.py passes the closed-candle-only analysis dataframe.
        # Do not discard its latest confirmed candle again here.
        d = _normalize_ohlc(df).tail(
            max(8, int(lookback))
        ).copy()

        if (
            level is None
            or atr is None
            or atr <= 0
            or direction not in {
                "BULLISH",
                "BEARISH",
            }
            or len(d) < 3
        ):
            return out

        break_buffer = atr * 0.10
        retest_tolerance = atr * 0.25
        hold_buffer = atr * 0.03
        break_index: Optional[int] = None

        closes = d["Close"].astype(float).tolist()

        # Use the latest genuine crossing event.
        for index in range(1, len(d)):
            previous_close = closes[index - 1]
            close = closes[index]

            if direction == "BULLISH":
                crossed = bool(
                    previous_close
                    <= level + break_buffer
                    and close
                    > level + break_buffer
                )
            else:
                crossed = bool(
                    previous_close
                    >= level - break_buffer
                    and close
                    < level - break_buffer
                )

            if crossed:
                break_index = index

        if break_index is None:
            out.update({
                "status": "BREAK_NOT_FOUND",
                "reason": (
                    "No recent closed-candle crossing of the "
                    "CHoCH level was found."
                ),
            })
            return out

        out.update({
            "status": "NOT_SEEN",
            "break_age_bars": int(
                (len(d) - 1) - break_index
            ),
            "reason": (
                "A later closed-candle retest has not occurred."
            ),
        })

        # Breakout candle cannot confirm its own retest.
        for index in range(
            break_index + 1,
            len(d),
        ):
            row = d.iloc[index]
            high = _safe_float(row.get("High"))
            low = _safe_float(row.get("Low"))
            close = _safe_float(row.get("Close"))

            if None in {high, low, close}:
                continue

            touched = bool(
                low <= level + retest_tolerance
                and high >= level - retest_tolerance
            )

            if direction == "BULLISH":
                failed = close < level - break_buffer
                held = bool(
                    touched
                    and close > level + hold_buffer
                )
            else:
                failed = close > level + break_buffer
                held = bool(
                    touched
                    and close < level - hold_buffer
                )

            if failed:
                out.update({
                    "status": "FAILED",
                    "reason": (
                        "The CHoCH level was reclaimed from "
                        "the wrong side."
                    ),
                })
            elif held:
                out.update({
                    "status": "HELD",
                    "reason": (
                        "A later closed candle retested and "
                        "held the CHoCH level."
                    ),
                })
            elif touched:
                out.update({
                    "status": "UNCONFIRMED",
                    "reason": (
                        "The level was touched, but no clear "
                        "closed-candle hold is confirmed."
                    ),
                })

        return out

    except Exception as exc:
        out["reason"] = (
            "CHoCH retest detection error: "
            f"{str(exc)}"
        )
        return out

def _detect_choch_severity(
    *,
    current_price: float,
    atr_val: Optional[float],
    confirm_atr_val: Optional[float],
    micro_df: Optional[pd.DataFrame],
    micro_swings: Dict[str, Any],
    confirm_swings: Dict[str, Any],
    swing_swings: Dict[str, Any],
    micro_direction: str,
    confirm_direction: str,
    swing_direction: str,
    confirm_bias: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect CHoCH / direction-change risk using multiple timeframes.

    This is intentionally staged:
    - Micro break only = minor / early warning
    - Micro break + confirm weakness = medium
    - Confirm structure break = strong
    - Confirm break + swing confirms = confirmed reversal risk
    """
    out = {
        "choch_status": "CHOCH_NONE",
        "choch_severity": "NONE",
        "choch_timeframe": "NONE",
        "choch_direction": "NONE",
        "choch_broken_level": None,
        "choch_retest_status": "NOT_AVAILABLE",
        "choch_break_age_bars": None,
        "choch_retest_reason": "",
        "direction_change_risk": "LOW",
        "recommended_trade_mode": "FULL_TRADE",
        "reason": "No meaningful CHoCH detected.",
    }

    try:
        micro_direction = str(micro_direction or "NEUTRAL").upper()
        confirm_direction = str(confirm_direction or "NEUTRAL").upper()
        swing_direction = str(swing_direction or "NEUTRAL").upper()

        # A Confirm-TF structure break must be confirmed by the
        # latest CLOSED Confirm-TF candle and Confirm-TF ATR.
        # Do not promote a Main/Micro-TF close into a strong
        # Confirm-TF CHoCH before that higher-timeframe candle closes.
        confirm_break_price = _safe_float(
            (confirm_bias or {}).get("close")
        )

        if confirm_break_price is None:
            confirm_break_price = current_price

        confirm_break_atr = _safe_float(
            confirm_atr_val
        )

        if (
            confirm_break_atr is None
            or confirm_break_atr <= 0
        ):
            confirm_break_atr = atr_val

        # If confirm direction is known, break opposite structure on micro/confirm.
        bearish_micro_break = _detect_swing_break(
            side_to_break="BEARISH",
            current_price=current_price,
            swings=micro_swings,
            atr_val=atr_val,
        )
        bullish_micro_break = _detect_swing_break(
            side_to_break="BULLISH",
            current_price=current_price,
            swings=micro_swings,
            atr_val=atr_val,
        )

        bearish_confirm_break = _detect_swing_break(
            side_to_break="BEARISH",
            current_price=confirm_break_price,
            swings=confirm_swings,
            atr_val=confirm_break_atr,
        )

        bullish_confirm_break = _detect_swing_break(
            side_to_break="BULLISH",
            current_price=confirm_break_price,
            swings=confirm_swings,
            atr_val=confirm_break_atr,
        )

        # Direction of possible change.
        possible_bearish = bearish_micro_break["broke"] or bearish_confirm_break["broke"]
        possible_bullish = bullish_micro_break["broke"] or bullish_confirm_break["broke"]

        # Avoid double-count ambiguity.
        if possible_bearish and possible_bullish:
            out.update({
                "choch_status": "CHOCH_MIXED",
                "choch_severity": "MEDIUM",
                "choch_timeframe": "MICRO_CONFIRM",
                "choch_direction": "MIXED",
                "direction_change_risk": "MEDIUM",
                "recommended_trade_mode": "CONFIRMATION_REQUIRED",
                "reason": "Both bullish and bearish structure breaks appeared; market may be expanding/choppy.",
            })
            return out

        if possible_bearish:
            level = bearish_confirm_break.get("broken_level") or bearish_micro_break.get("broken_level")

            retest = _detect_choch_retest(
                df=micro_df,
                broken_level=level,
                break_direction="BEARISH",
                atr_val=atr_val,
            )

            if bearish_confirm_break["broke"]:
                severity = "STRONG"
                status = "STRONG_BEARISH_CHOCH"
                timeframe = "CONFIRM"
                risk = "HIGH"
                mode = "WAIT_FOR_RETEST"
                reason = "Price broke below Confirm TF recent swing low; bearish CHoCH risk is strong."

                if swing_direction == "SHORT":
                    severity = "CONFIRMED_REVERSAL"
                    status = "CONFIRMED_BEARISH_REVERSAL"
                    timeframe = "CONFIRM_SWING"
                    mode = "CONFIRMATION_REQUIRED"
                    reason = "Confirm TF broke bearish and Swing TF is already bearish; reversal/continuation risk is confirmed."

            else:
                severity = "MINOR"
                status = "EARLY_BEARISH_CHOCH"
                timeframe = "MICRO"
                risk = "MEDIUM"
                mode = "CONFIRMATION_REQUIRED"
                reason = "Micro TF broke below recent swing low; early bearish CHoCH warning."

                if confirm_direction == "SHORT":
                    severity = "MEDIUM"
                    status = "MEDIUM_BEARISH_CHOCH"
                    risk = "HIGH"
                    reason = "Micro TF broke bearish while Confirm TF direction is bearish."

            out.update({
                "choch_status": status,
                "choch_severity": severity,
                "choch_timeframe": timeframe,
                "choch_direction": "BEARISH",
                "choch_broken_level": _fmt_price(level),
                "choch_retest_status": str(
                    retest.get("status")
                    or "NOT_AVAILABLE"
                ),
                "choch_break_age_bars": retest.get(
                    "break_age_bars"
                ),
                "choch_retest_reason": str(
                    retest.get("reason") or ""
                ),
                "direction_change_risk": risk,
                "recommended_trade_mode": mode,
                "reason": reason,
            })
            return out

        if possible_bullish:
            level = bullish_confirm_break.get("broken_level") or bullish_micro_break.get("broken_level")

            retest = _detect_choch_retest(
                df=micro_df,
                broken_level=level,
                break_direction="BULLISH",
                atr_val=atr_val,
            )

            if bullish_confirm_break["broke"]:
                severity = "STRONG"
                status = "STRONG_BULLISH_CHOCH"
                timeframe = "CONFIRM"
                risk = "HIGH"
                mode = "WAIT_FOR_RETEST"
                reason = "Price broke above Confirm TF recent swing high; bullish CHoCH risk is strong."

                if swing_direction == "LONG":
                    severity = "CONFIRMED_REVERSAL"
                    status = "CONFIRMED_BULLISH_REVERSAL"
                    timeframe = "CONFIRM_SWING"
                    mode = "CONFIRMATION_REQUIRED"
                    reason = "Confirm TF broke bullish and Swing TF is already bullish; reversal/continuation risk is confirmed."

            else:
                severity = "MINOR"
                status = "EARLY_BULLISH_CHOCH"
                timeframe = "MICRO"
                risk = "MEDIUM"
                mode = "CONFIRMATION_REQUIRED"
                reason = "Micro TF broke above recent swing high; early bullish CHoCH warning."

                if confirm_direction == "LONG":
                    severity = "MEDIUM"
                    status = "MEDIUM_BULLISH_CHOCH"
                    risk = "HIGH"
                    reason = "Micro TF broke bullish while Confirm TF direction is bullish."

            out.update({
                "choch_status": status,
                "choch_severity": severity,
                "choch_timeframe": timeframe,
                "choch_direction": "BULLISH",
                "choch_broken_level": _fmt_price(level),
                "choch_retest_status": str(
                    retest.get("status")
                    or "NOT_AVAILABLE"
                ),
                "choch_break_age_bars": retest.get(
                    "break_age_bars"
                ),
                "choch_retest_reason": str(
                    retest.get("reason") or ""
                ),
                "direction_change_risk": risk,
                "recommended_trade_mode": mode,
                "reason": reason,
            })
            return out

        return out

    except Exception as e:
        out["reason"] = f"CHoCH severity detection error: {str(e)}"
        return out


def _build_mtf_direction_state(
    *,
    macro_bias: Dict[str, Any],
    swing_bias: Dict[str, Any],
    confirm_bias: Dict[str, Any],
    micro_bias: Dict[str, Any],
    macro_swings: Dict[str, Any],
    swing_swings: Dict[str, Any],
    confirm_swings: Dict[str, Any],
    micro_swings: Dict[str, Any],
    current_price: float,
    atr_val: Optional[float],
    confirm_atr_val: Optional[float],
    micro_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Build a structured MTF direction state.

    This does not create trades or levels.
    It gives setup_planner/execution layers a cleaner direction object.
    """
    out = {
        "local_direction": "NEUTRAL",
        "confirm_direction": "NEUTRAL",
        "swing_direction": "NEUTRAL",
        "macro_direction": "NEUTRAL",
        "primary_direction": "WAIT",
        "htf_alignment": "NEUTRAL",
        "trend_transition_state": "UNKNOWN",
        "direction_agreement": 0.0,
        "direction_score": 50,
        "allowed_trade_mode": "FULL_TRADE",
        "direction_change_risk": "LOW",
        "choch_status": "CHOCH_NONE",
        "choch_severity": "NONE",
        "choch_timeframe": "NONE",
        "choch_direction": "NONE",
        "choch_broken_level": None,
        "choch_retest_status": "NOT_AVAILABLE",
        "choch_break_age_bars": None,
        "choch_retest_reason": "",
        "reason": "",
    }

    try:
        macro_dir = _bias_to_direction((macro_bias or {}).get("bias"))
        swing_dir = _bias_to_direction((swing_bias or {}).get("bias"))
        confirm_dir = _bias_to_direction((confirm_bias or {}).get("bias"))
        micro_dir = _bias_to_direction((micro_bias or {}).get("bias"))

        # Structure can refine EMA bias.
        macro_struct_dir = _structure_to_direction((macro_swings or {}).get("structure"))
        swing_struct_dir = _structure_to_direction((swing_swings or {}).get("structure"))
        confirm_struct_dir = _structure_to_direction((confirm_swings or {}).get("structure"))
        micro_struct_dir = _structure_to_direction((micro_swings or {}).get("structure"))

        # Prefer structure direction if EMA bias is neutral.
        if macro_dir == "NEUTRAL" and macro_struct_dir != "NEUTRAL":
            macro_dir = macro_struct_dir
        if swing_dir == "NEUTRAL" and swing_struct_dir != "NEUTRAL":
            swing_dir = swing_struct_dir
        if confirm_dir == "NEUTRAL" and confirm_struct_dir != "NEUTRAL":
            confirm_dir = confirm_struct_dir
        if micro_dir == "NEUTRAL" and micro_struct_dir != "NEUTRAL":
            micro_dir = micro_struct_dir

        agreement = _direction_agreement_score(micro_dir, confirm_dir, swing_dir, macro_dir)

        primary = agreement["primary"]
        if primary == "NEUTRAL":
            primary = "WAIT"

        # HTF = confirm + swing + macro. Micro is local/setup timing.
        htf = _direction_agreement_score(confirm_dir, swing_dir, macro_dir)
        local_vs_htf_conflict = (
            micro_dir in ["LONG", "SHORT"]
            and htf["primary"] in ["LONG", "SHORT"]
            and micro_dir != htf["primary"]
        )

        confirm_swing_conflict = (
            confirm_dir in ["LONG", "SHORT"]
            and swing_dir in ["LONG", "SHORT"]
            and confirm_dir != swing_dir
        )

        if confirm_swing_conflict:
            htf_alignment = "CONFLICT"
            transition = "CONFIRM_SWING_CONFLICT"
            mode = "CONFIRMATION_REQUIRED"
        elif local_vs_htf_conflict:
            htf_alignment = "AGAINST"
            transition = "LOCAL_PULLBACK_AGAINST_HTF"
            mode = "CONFIRMATION_REQUIRED"
        elif htf["agreement"] >= 0.67 and htf["primary"] in ["LONG", "SHORT"]:
            htf_alignment = "ALIGNED"
            transition = "TREND_ALIGNED"
            mode = "FULL_TRADE"
        else:
            htf_alignment = "NEUTRAL"
            transition = "MIXED_OR_RANGE"
            mode = "CONFIRMATION_REQUIRED"

        # Direction score 0-100 from agreement + HTF alignment.
        score = 50
        if primary == "LONG":
            score += int(agreement["agreement"] * 35)
        elif primary == "SHORT":
            score -= int(agreement["agreement"] * 35)

        # Convert to absolute confidence style while keeping side separately.
        confidence = int(round(50 + abs(score - 50) * 1.3))
        confidence = int(max(0, min(100, confidence)))

        choch = _detect_choch_severity(
            current_price=current_price,
            atr_val=atr_val,
            confirm_atr_val=confirm_atr_val,
            micro_df=micro_df,
            micro_swings=micro_swings,
            confirm_swings=confirm_swings,
            swing_swings=swing_swings,
            micro_direction=micro_dir,
            confirm_direction=confirm_dir,
            swing_direction=swing_dir,
            confirm_bias=confirm_bias,
        )

        if choch.get("recommended_trade_mode") in ["WAIT_FOR_RETEST", "CONFIRMATION_REQUIRED"]:
            mode = choch.get("recommended_trade_mode")

        if choch.get("choch_severity") in ["STRONG", "CONFIRMED_REVERSAL"]:
            transition = "REVERSAL_FORMING"

        out.update({
            "local_direction": micro_dir,
            "confirm_direction": confirm_dir,
            "swing_direction": swing_dir,
            "macro_direction": macro_dir,
            "primary_direction": primary,
            "htf_alignment": htf_alignment,
            "trend_transition_state": transition,
            "direction_agreement": round(float(agreement["agreement"]), 3),
            "direction_score": confidence,
            "allowed_trade_mode": mode,
            "direction_change_risk": choch.get("direction_change_risk", "LOW"),
            "choch_status": choch.get("choch_status", "CHOCH_NONE"),
            "choch_severity": choch.get("choch_severity", "NONE"),
            "choch_timeframe": choch.get("choch_timeframe", "NONE"),
            "choch_direction": choch.get("choch_direction", "NONE"),
            "choch_broken_level": choch.get("choch_broken_level"),
            "choch_retest_status": str(
                choch.get("choch_retest_status")
                or "NOT_AVAILABLE"
            ),
            "choch_break_age_bars": choch.get(
                "choch_break_age_bars"
            ),
            "choch_retest_reason": str(
                choch.get("choch_retest_reason") or ""
            ),
            "reason": choch.get("reason", ""),
        })

        return out

    except Exception as e:
        out["reason"] = f"MTF direction state error: {str(e)}"
        return out

# ==========================================================
# Context conflict adjustment
# ==========================================================

def _apply_context_adjustments(
    result: Dict[str, Any],
    bull_score: float,
    bear_score: float,
    reasons: List[str],
    context_flags: Dict[str, Any],
) -> Tuple[float, float, List[str], Dict[str, Any]]:
    """
    Apply context_bundle awareness before final regime classification.
    """
    adjustments = {
        "hard_wait": False,
        "confidence_cap": None,
        "conflict": False,
        "conflict_reason": "",
    }

    try:
        if not context_flags.get("available"):
            return bull_score, bear_score, reasons, adjustments

        # Stage CHoCH/chop instead of hard-blocking every warning.
        choch_active = bool(
            context_flags.get("choch")
        )

        choch_severity = str(
            result.get("choch_severity")
            or "NONE"
        ).upper()

        choch_direction = str(
            result.get("choch_direction")
            or "NONE"
        ).upper()

        htf_state = _direction_agreement_score(
            result.get("confirm_direction"),
            result.get("swing_direction"),
            result.get("macro_direction"),
        )

        htf_side = str(
            htf_state.get("primary")
            or "NEUTRAL"
        ).upper()

        choch_side = {
            "BULLISH": "LONG",
            "BEARISH": "SHORT",
        }.get(
            choch_direction,
            "NEUTRAL",
        )

        confirmed_adverse_choch = bool(
            choch_severity in {
                "STRONG",
                "CONFIRMED_REVERSAL",
            }
            and htf_side in {
                "LONG",
                "SHORT",
            }
            and choch_side in {
                "LONG",
                "SHORT",
            }
            and choch_side != htf_side
        )

        choch_retest_status = str(
            result.get("choch_retest_status")
            or "NOT_AVAILABLE"
        ).upper()

        adverse_retest_held = bool(
            confirmed_adverse_choch
            and choch_retest_status == "HELD"
        )

        if (
            confirmed_adverse_choch
            and not adverse_retest_held
        ):
            wait_detail = {
                "FAILED": "the CHoCH retest failed",
                "UNCONFIRMED": (
                    "the CHoCH retest did not confirm a hold"
                ),
                "NOT_SEEN": (
                    "a later closed-candle retest has not occurred"
                ),
                "BREAK_NOT_FOUND": (
                    "the recent break/retest sequence is unavailable"
                ),
                "NOT_AVAILABLE": (
                    "retest evidence is unavailable"
                ),
            }.get(
                choch_retest_status,
                "a valid retest hold is not confirmed",
            )

            adjustments.update({
                "hard_wait": True,
                "confidence_cap": 0.42,
                "conflict": True,
                "conflict_reason": (
                    "Confirmed adverse HTF CHoCH: "
                    f"HTF={htf_side}, "
                    f"CHoCH={choch_direction}; "
                    f"{wait_detail}."
                ),
            })

            reasons.append(
                "Context: confirmed CHoCH opposes "
                "the established HTF direction; "
                f"{wait_detail}; wait for a valid "
                "retest/alignment."
            )

            return (
                bull_score,
                bear_score,
                reasons,
                adjustments,
            )

        if adverse_retest_held:
            adjustments.update({
                "hard_wait": False,
                "confidence_cap": 0.62,
                "conflict": True,
                "conflict_reason": (
                    "Adverse HTF CHoCH retest held; Gemini "
                    "may evaluate the new direction with "
                    "confirmation."
                ),
            })

            result["allowed_trade_mode"] = (
                "CONFIRMATION_REQUIRED"
            )

            reasons.append(
                "Context: a later closed candle retested and "
                "held the adverse CHoCH level; release the "
                "global hard block but keep normal validation."
            )

        if (
            choch_active
            or choch_severity != "NONE"
        ):
            choch_cap = (
                0.62
                if choch_severity in {
                    "MEDIUM",
                    "STRONG",
                    "CONFIRMED_REVERSAL",
                }
                else 0.68
            )

            adjustments["confidence_cap"] = min(
                adjustments["confidence_cap"]
                or 1.0,
                choch_cap,
            )

            adjustments["conflict"] = True

            if adverse_retest_held:
                adjustments["conflict_reason"] = (
                    "CHoCH retest held; the new direction "
                    "still requires confirmation."
                )
            else:
                adjustments["conflict_reason"] = (
                    "CHoCH requires confirmation; "
                    "not an automatic hard block."
                )

            if adverse_retest_held:
                result["allowed_trade_mode"] = (
                    "CONFIRMATION_REQUIRED"
                )
            elif (
                result.get("allowed_trade_mode")
                == "FULL_TRADE"
            ):
                result["allowed_trade_mode"] = (
                    "CONFIRMATION_REQUIRED"
                )

            if not adverse_retest_held:
                reasons.append(
                    "Context: early/mixed or "
                    "HTF-aligned CHoCH is advisory; "
                    "confirmation is required without "
                    "skipping Gemini."
                )

        choppy_active = bool(
            context_flags.get(
                "both_rejections"
            )
            or context_flags.get(
                "is_choppy"
            )
        )

        htf_alignment = str(
            result.get("htf_alignment")
            or "NEUTRAL"
        ).upper()

        strong_directionless_chop = bool(
            choppy_active
            and context_flags.get(
                "both_rejections"
            )
            and (
                htf_side == "NEUTRAL"
                or htf_alignment in {
                    "NEUTRAL",
                    "CONFLICT",
                }
            )
        )

        if strong_directionless_chop:
            adjustments.update({
                "hard_wait": True,

                "confidence_cap": min(
                    adjustments[
                        "confidence_cap"
                    ]
                    or 1.0,
                    0.45,
                ),

                "conflict": True,

                "conflict_reason": (
                    "Strong directionless chop "
                    "with both-side rejection."
                ),
            })

            reasons.append(
                "Context: both-side rejection plus "
                "neutral/conflicted HTF structure; "
                "wait for a clean directional break."
            )

            return (
                bull_score,
                bear_score,
                reasons,
                adjustments,
            )

        if choppy_active:
            adjustments["confidence_cap"] = min(
                adjustments["confidence_cap"]
                or 1.0,
                0.65,
            )

            adjustments["conflict"] = True

            adjustments["conflict_reason"] = (
                "Choppy/rejection context "
                "requires confirmation."
            )

            if (
                result.get("allowed_trade_mode")
                == "FULL_TRADE"
            ):
                result["allowed_trade_mode"] = (
                    "CONFIRMATION_REQUIRED"
                )

            reasons.append(
                "Context: choppy/rejection evidence "
                "is advisory; Gemini may analyse it "
                "but confirmation is required."
            )

        dominant_side = _dominant_side_to_normal(context_flags.get("dominant_preferred_side"))

        # If dominant trend already says WAIT, cap confidence and reduce directional scores.
        if context_flags.get("dominant_wait") or dominant_side == "WAIT":
            adjustments["confidence_cap"] = 0.55
            adjustments["conflict"] = True
            adjustments["conflict_reason"] = "Dominant trend preferred side is WAIT."
            bull_score *= 0.82
            bear_score *= 0.82
            reasons.append("Context: dominant trend preferred WAIT; confidence capped.")

        # Rejection context nudges, but does not override by itself.
        support_rej = bool(context_flags.get("support_rejection"))
        resistance_rej = bool(context_flags.get("resistance_rejection"))

        if support_rej and not resistance_rej:
            bull_score += 0.06
            reasons.append("Context: support rejection confirmed; bullish side gets small boost.")

        if resistance_rej and not support_rej:
            bear_score += 0.06
            reasons.append("Context: resistance rejection confirmed; bearish side gets small boost.")

        # If dominant side directly conflicts with computed score, cap confidence.
        if dominant_side == "LONG" and bear_score > bull_score + 0.10:
            adjustments["confidence_cap"] = min(adjustments["confidence_cap"] or 1.0, 0.62)
            adjustments["conflict"] = True
            adjustments["conflict_reason"] = "Dominant side LONG conflicts with backend bearish score."
            reasons.append("Context: dominant side LONG conflicts with bearish regime score.")

        if dominant_side == "SHORT" and bull_score > bear_score + 0.10:
            adjustments["confidence_cap"] = min(adjustments["confidence_cap"] or 1.0, 0.62)
            adjustments["conflict"] = True
            adjustments["conflict_reason"] = "Dominant side SHORT conflicts with backend bullish score."
            reasons.append("Context: dominant side SHORT conflicts with bullish regime score.")

        # Weak/stale Elliott fib should reduce certainty in pullback regimes.
        if context_flags.get("elliott_fib_usable") is False:
            adjustments["confidence_cap"] = min(adjustments["confidence_cap"] or 1.0, 0.72)
            reasons.append("Context: Elliott Fib is weak/stale; trend regime confidence capped moderately.")

        return _clamp(bull_score), _clamp(bear_score), reasons, adjustments

    except Exception as e:
        reasons.append(f"Context adjustment error: {str(e)}")
        return bull_score, bear_score, reasons, adjustments


# ==========================================================
# Public detector
# ==========================================================

def detect_backend_trend_regime(
    df_macro: Optional[pd.DataFrame] = None,
    df_swing: Optional[pd.DataFrame] = None,
    df_confirm: Optional[pd.DataFrame] = None,
    df_micro: Optional[pd.DataFrame] = None,
    user_tf: str = "15m",
    context_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backend Trend Regime Observer.

    Important:
    - This does NOT create trades.
    - This does NOT create entry/SL/TP.
    - This detects preferred side/regime.
    - context_bundle can force/cap WAIT/conflict states when dominant trend or candle context says unsafe.
    """

    macro = _normalize_ohlc(df_macro)
    swing = _normalize_ohlc(df_swing)
    confirm = _normalize_ohlc(df_confirm)
    micro = _normalize_ohlc(df_micro)

    reasons: List[str] = []
    context_flags = _extract_context_flags(context_bundle)

    result = {
        "regime": "UNKNOWN",
        "market_bias": "UNKNOWN",
        "market_phase": "UNKNOWN",
        "preferred_side": "NONE",
        "blocked_side": "NONE",
        "confidence": 0.0,
        "setup_bias": "NO_CLEAR_SETUP",
        "hard_block_active": False,
        "current_price": None,
        "recent_swing_high": None,
        "recent_swing_low": None,
        "confirm_bias": "UNKNOWN",
        "micro_bias": "UNKNOWN",
        "micro_structure": "UNKNOWN",
        "confirm_structure": "UNKNOWN",
        "reasons": reasons,

        # Context-aware debug fields
        "context_available": bool(context_flags.get("available")),
        "context_conflict": False,
        "context_conflict_reason": "",
        "dominant_preferred_side": context_flags.get("dominant_preferred_side"),
        "choch": bool(context_flags.get("choch")),
        "is_choppy": bool(context_flags.get("is_choppy") or context_flags.get("both_rejections")),
        "support_rejection": bool(context_flags.get("support_rejection")),
        "resistance_rejection": bool(context_flags.get("resistance_rejection")),
        "elliott_fib_usable": bool(context_flags.get("elliott_fib_usable")),
        "fib_entry_source": context_flags.get("fib_entry_source"),
        "confidence_before_context": None,
        "confidence_after_context": None,
        "bull_score": 0.0,
        "bear_score": 0.0,
        # Multi-timeframe direction shadow fields
        "local_direction": "NEUTRAL",
        "confirm_direction": "NEUTRAL",
        "swing_direction": "NEUTRAL",
        "macro_direction": "NEUTRAL",
        "primary_direction": "WAIT",
        "htf_alignment": "NEUTRAL",
        "trend_transition_state": "UNKNOWN",
        "direction_agreement": 0.0,
        "direction_score": 50,
        "allowed_trade_mode": "FULL_TRADE",
        "direction_change_risk": "LOW",
        "choch_status": "CHOCH_NONE",
        "choch_severity": "NONE",
        "choch_timeframe": "NONE",
        "choch_direction": "NONE",
        "choch_broken_level": None,
        "choch_retest_status": "NOT_AVAILABLE",
        "choch_break_age_bars": None,
        "choch_retest_reason": "",
        "choch_reason": "",
    }

    try:
        if micro.empty:
            reasons.append("Micro dataframe unavailable")
            return result

        current_price = float(micro["Close"].iloc[-1])
        result["current_price"] = current_price

        macro_bias = _timeframe_bias(macro, "Macro TF")
        swing_bias = _timeframe_bias(swing, "Swing TF")
        confirm_bias = _timeframe_bias(confirm, "Confirm TF")
        micro_bias = _timeframe_bias(micro, "Micro TF")

        result["confirm_bias"] = confirm_bias["bias"]
        result["micro_bias"] = micro_bias["bias"]

        # Weighted regime score.
        # Confirm + swing are more important than micro noise.
        weights = {
            "macro": 0.15,
            "swing": 0.25,
            "confirm": 0.38,
            "micro": 0.22,
        }

        bull_score = (
            macro_bias["bull_score"] * weights["macro"]
            + swing_bias["bull_score"] * weights["swing"]
            + confirm_bias["bull_score"] * weights["confirm"]
            + micro_bias["bull_score"] * weights["micro"]
        )

        bear_score = (
            macro_bias["bear_score"] * weights["macro"]
            + swing_bias["bear_score"] * weights["swing"]
            + confirm_bias["bear_score"] * weights["confirm"]
            + micro_bias["bear_score"] * weights["micro"]
        )

        micro_swings = _find_recent_swings(micro, window=3, lookback=100)
        confirm_swings = _find_recent_swings(confirm, window=3, lookback=120)

        swing_swings = _find_recent_swings(swing, window=3, lookback=120)
        macro_swings = _find_recent_swings(macro, window=3, lookback=120)

        atr_val_for_direction = _atr(
            micro,
            14,
        )

        confirm_atr_val_for_direction = _atr(
            confirm,
            14,
        )

        mtf_direction = _build_mtf_direction_state(
            macro_bias=macro_bias,
            swing_bias=swing_bias,
            confirm_bias=confirm_bias,
            micro_bias=micro_bias,
            macro_swings=macro_swings,
            swing_swings=swing_swings,
            confirm_swings=confirm_swings,
            micro_swings=micro_swings,
            current_price=current_price,
            atr_val=atr_val_for_direction,
            confirm_atr_val=(
                confirm_atr_val_for_direction
            ),
            micro_df=micro,
        )

        result.update({
            "local_direction": mtf_direction.get("local_direction"),
            "confirm_direction": mtf_direction.get("confirm_direction"),
            "swing_direction": mtf_direction.get("swing_direction"),
            "macro_direction": mtf_direction.get("macro_direction"),
            "primary_direction": mtf_direction.get("primary_direction"),
            "htf_alignment": mtf_direction.get("htf_alignment"),
            "trend_transition_state": mtf_direction.get("trend_transition_state"),
            "direction_agreement": mtf_direction.get("direction_agreement"),
            "direction_score": mtf_direction.get("direction_score"),
            "allowed_trade_mode": mtf_direction.get("allowed_trade_mode"),
            "direction_change_risk": mtf_direction.get("direction_change_risk"),
            "choch_status": mtf_direction.get("choch_status"),
            "choch_severity": mtf_direction.get("choch_severity"),
            "choch_timeframe": mtf_direction.get("choch_timeframe"),
            "choch_direction": mtf_direction.get("choch_direction"),
            "choch_broken_level": mtf_direction.get("choch_broken_level"),
            "choch_retest_status": mtf_direction.get(
                "choch_retest_status"
            ),
            "choch_break_age_bars": mtf_direction.get(
                "choch_break_age_bars"
            ),
            "choch_retest_reason": mtf_direction.get(
                "choch_retest_reason"
            ),
            "choch_reason": mtf_direction.get("reason"),
        })

        if mtf_direction.get("trend_transition_state") in ["LOCAL_PULLBACK_AGAINST_HTF", "CONFIRM_SWING_CONFLICT"]:
            reasons.append(
                f"MTF direction state: {mtf_direction.get('trend_transition_state')} "
                f"mode={mtf_direction.get('allowed_trade_mode')}"
            )

        if mtf_direction.get("choch_status") != "CHOCH_NONE":
            reasons.append(
                f"CHoCH state: {mtf_direction.get('choch_status')} "
                f"severity={mtf_direction.get('choch_severity')} "
                f"risk={mtf_direction.get('direction_change_risk')}"
            )

        result["recent_swing_high"] = micro_swings.get("recent_swing_high")
        result["recent_swing_low"] = micro_swings.get("recent_swing_low")
        result["micro_structure"] = micro_swings.get("structure", "UNKNOWN")
        result["confirm_structure"] = confirm_swings.get("structure", "UNKNOWN")

        # Structure boost
        if result["confirm_structure"] == "HH_HL":
            bull_score += 0.12
            reasons.append("Confirm TF structure is HH/HL")
        elif result["confirm_structure"] == "LH_LL":
            bear_score += 0.12
            reasons.append("Confirm TF structure is LH/LL")
        elif result["confirm_structure"] in ["EXPANDING", "COMPRESSING", "RANGE"]:
            reasons.append(f"Confirm TF structure is {result['confirm_structure']}")

        if result["micro_structure"] == "HH_HL":
            bull_score += 0.06
            reasons.append("Micro TF structure is HH/HL")
        elif result["micro_structure"] == "LH_LL":
            bear_score += 0.06
            reasons.append("Micro TF structure is LH/LL")
        elif result["micro_structure"] in ["EXPANDING", "COMPRESSING", "RANGE"]:
            reasons.append(f"Micro TF structure is {result['micro_structure']}")

        momentum = _latest_candle_momentum(micro)
        pullback = _pullback_state(micro)

        recent_high = micro_swings.get("recent_swing_high")
        recent_low = micro_swings.get("recent_swing_low")

        atr_val = _atr(micro, 14)
        buffer = 0.0
        if atr_val and atr_val > 0:
            # Small buffer so tiny wick breaks do not count as real breaks.
            buffer = atr_val * 0.10

        close_above_recent_high = (
            recent_high is not None and current_price > float(recent_high) + buffer
        )
        close_below_recent_low = (
            recent_low is not None and current_price < float(recent_low) - buffer
        )

        bullish_impulse = (
            momentum["direction"] == "BULLISH"
            and momentum["strong_body"]
            and (momentum["atr_expansion"] or momentum["volume_expansion"])
        )

        bearish_impulse = (
            momentum["direction"] == "BEARISH"
            and momentum["strong_body"]
            and (momentum["atr_expansion"] or momentum["volume_expansion"])
        )

        if close_above_recent_high:
            bull_score += 0.10
            reasons.append("Price closed above recent micro swing high")

        if close_below_recent_low:
            bear_score += 0.10
            reasons.append("Price closed below recent micro swing low")

        if bullish_impulse:
            bull_score += 0.08
            reasons.append("Bullish impulse candle detected")

        if bearish_impulse:
            bear_score += 0.08
            reasons.append("Bearish impulse candle detected")

        # Confirm/micro conflict penalty.
        if confirm_bias["bias"] == "BULLISH" and micro_bias["bias"] == "BEARISH":
            bear_score *= 0.88
            reasons.append("Conflict: micro bearish while confirm TF is bullish; bearish confidence reduced.")
        elif confirm_bias["bias"] == "BEARISH" and micro_bias["bias"] == "BULLISH":
            bull_score *= 0.88
            reasons.append("Conflict: micro bullish while confirm TF is bearish; bullish confidence reduced.")

        bull_score = _clamp(bull_score, 0.0, 1.0)
        bear_score = _clamp(bear_score, 0.0, 1.0)

        raw_diff = bull_score - bear_score
        raw_abs_diff = abs(raw_diff)
        raw_confidence = _clamp(0.45 + raw_abs_diff, 0.35, 0.92)
        result["confidence_before_context"] = round(float(raw_confidence), 2)

        # Apply context_bundle conflict/rejection/dominant WAIT awareness.
        bull_score, bear_score, reasons, context_adj = _apply_context_adjustments(
            result=result,
            bull_score=bull_score,
            bear_score=bear_score,
            reasons=reasons,
            context_flags=context_flags,
        )

        result["context_conflict"] = bool(context_adj.get("conflict"))
        result["context_conflict_reason"] = context_adj.get("conflict_reason", "")

        # Soft MTF direction safety:
        # Do not hard block here. Just cap confidence and mark trade mode.
        if result.get("allowed_trade_mode") in ["CONFIRMATION_REQUIRED", "WAIT_FOR_RETEST"]:
            reasons.append(
                f"MTF direction recommends {result.get('allowed_trade_mode')} "
                f"because {result.get('trend_transition_state')} / {result.get('choch_status')}."
            )

            if context_adj.get("confidence_cap") is None:
                context_adj["confidence_cap"] = 0.72
            else:
                context_adj["confidence_cap"] = min(float(context_adj["confidence_cap"]), 0.72)

        # If context says hard WAIT, return a clean no-side regime.
        if context_adj.get("hard_wait"):
            confidence = float(context_adj.get("confidence_cap") or 0.42)
            for bias_obj in [macro_bias, swing_bias, confirm_bias, micro_bias]:
                for r in bias_obj.get("reasons", [])[:2]:
                    reasons.append(r)
            for n in context_flags.get("notes", []):
                reasons.append(f"Context note: {n}")

            clean_reasons = _dedupe_reasons(reasons)

            result.update({
                "regime": "WAIT_CONTEXT_CONFLICT",
                "market_bias": "MIXED",
                "market_phase": "CONFLICT_OR_CHOP",
                "preferred_side": "NONE",
                "blocked_side": "NONE",
                "confidence": round(float(confidence), 2),
                "confidence_after_context": round(float(confidence), 2),
                "setup_bias": "WAIT_FOR_CLEAR_STRUCTURE",
                "hard_block_active": True,
                "bull_score": round(float(bull_score), 3),
                "bear_score": round(float(bear_score), 3),
                "reasons": clean_reasons[:14],
            })
            return result

        diff = bull_score - bear_score
        abs_diff = abs(diff)

        if diff >= 0.14:
            market_bias = "BULLISH"
        elif diff <= -0.14:
            market_bias = "BEARISH"
        else:
            market_bias = "MIXED"

        result["market_bias"] = market_bias

        # --------------------------------------------------
        # Regime classification
        # --------------------------------------------------
        regime = "RANGE_CHOP"
        phase = "NO_CLEAR_IMPULSE"
        preferred_side = "NONE"
        blocked_side = "NONE"
        setup_bias = "WAIT_FOR_CLEAR_STRUCTURE"

        if market_bias == "BULLISH":
            preferred_side = "LONG"
            blocked_side = "SHORT"

            if close_above_recent_high and bullish_impulse:
                regime = "BULLISH_BREAKOUT"
                phase = "IMPULSE_BREAKOUT"
                setup_bias = "WAIT_FOR_PULLBACK_OR_BREAKOUT_RETEST"
            elif pullback["bullish_pullback"]:
                regime = "BULLISH_PULLBACK"
                phase = "CORRECTIVE_PULLBACK"
                setup_bias = "WAVE_2_OR_B_PULLBACK_LONG"
            else:
                regime = "BULLISH_TREND"
                phase = "TREND_CONTINUATION_CONTEXT"
                setup_bias = "LONG_ONLY_IF_ENTRY_IS_SAFE"

        elif market_bias == "BEARISH":
            preferred_side = "SHORT"
            blocked_side = "LONG"

            if close_below_recent_low and bearish_impulse:
                regime = "BEARISH_BREAKDOWN"
                phase = "IMPULSE_BREAKDOWN"
                setup_bias = "WAIT_FOR_PULLBACK_OR_BREAKDOWN_RETEST"
            elif pullback["bearish_pullback"]:
                regime = "BEARISH_PULLBACK"
                phase = "CORRECTIVE_PULLBACK"
                setup_bias = "WAVE_2_OR_B_PULLBACK_SHORT"
            else:
                regime = "BEARISH_TREND"
                phase = "TREND_CONTINUATION_CONTEXT"
                setup_bias = "SHORT_ONLY_IF_ENTRY_IS_SAFE"

        else:
            regime = "RANGE_CHOP"
            phase = "MIXED_OR_SIDEWAYS"
            preferred_side = "NONE"
            blocked_side = "NONE"
            setup_bias = "WAIT_FOR_BREAK_OR_REJECTION"

        confidence = 0.45 + abs_diff
        if regime in ["BULLISH_BREAKOUT", "BEARISH_BREAKDOWN"]:
            confidence += 0.05
        if regime in ["BULLISH_PULLBACK", "BEARISH_PULLBACK"]:
            confidence += 0.04

        confidence = _clamp(confidence, 0.35, 0.92)

        confidence_cap = context_adj.get("confidence_cap")
        if confidence_cap is not None:
            confidence = min(confidence, float(confidence_cap))

        result["confidence_after_context"] = round(float(confidence), 2)

        # If context/dominant side conflicts after classification, avoid hard side if confidence is weak.
        dominant_side = _dominant_side_to_normal(context_flags.get("dominant_preferred_side"))
        if dominant_side == "LONG" and preferred_side == "SHORT" and confidence < 0.64:
            reasons.append("Final conflict: dominant LONG vs backend SHORT with weak confidence; switching to WAIT.")
            regime = "RANGE_CHOP"
            phase = "DIRECTION_CONFLICT"
            preferred_side = "NONE"
            blocked_side = "NONE"
            setup_bias = "WAIT_FOR_ALIGNMENT"
            confidence = min(confidence, 0.55)
            result["context_conflict"] = True
            result["context_conflict_reason"] = "Dominant LONG conflicts with backend SHORT."

        if dominant_side == "SHORT" and preferred_side == "LONG" and confidence < 0.64:
            reasons.append("Final conflict: dominant SHORT vs backend LONG with weak confidence; switching to WAIT.")
            regime = "RANGE_CHOP"
            phase = "DIRECTION_CONFLICT"
            preferred_side = "NONE"
            blocked_side = "NONE"
            setup_bias = "WAIT_FOR_ALIGNMENT"
            confidence = min(confidence, 0.55)
            result["context_conflict"] = True
            result["context_conflict_reason"] = "Dominant SHORT conflicts with backend LONG."

        # Add useful bias reasons
        for bias_obj in [macro_bias, swing_bias, confirm_bias, micro_bias]:
            for r in bias_obj.get("reasons", [])[:2]:
                reasons.append(r)

        for r in pullback.get("reasons", [])[:2]:
            reasons.append(r)

        for r in momentum.get("reasons", [])[:2]:
            reasons.append(r)

        for n in context_flags.get("notes", []):
            reasons.append(f"Context note: {n}")

        clean_reasons = _dedupe_reasons(reasons)

        result.update({
            "regime": regime,
            "market_phase": phase,
            "preferred_side": preferred_side,
            "blocked_side": blocked_side,
            "confidence": round(float(confidence), 2),
            "confidence_after_context": round(float(confidence), 2),
            "setup_bias": setup_bias,
            "hard_block_active": bool(regime == "WAIT_CONTEXT_CONFLICT"),
            "bull_score": round(float(bull_score), 3),
            "bear_score": round(float(bear_score), 3),
            "reasons": clean_reasons[:14],
        })

        return result

    except Exception as e:
        result["reasons"] = [f"Trend regime detector error: {str(e)}"]
        return result


def _dedupe_reasons(reasons: List[str]) -> List[str]:
    clean_reasons: List[str] = []
    seen = set()
    for r in reasons:
        if not r:
            continue
        if r not in seen:
            clean_reasons.append(r)
            seen.add(r)
    return clean_reasons


def format_backend_trend_regime_context(regime: Dict[str, Any]) -> str:
    """
    Convert regime dict to a clean text block for debug/AI context.
    """
    try:
        reasons = regime.get("reasons", [])
        reason_text = "\n".join([f"- {r}" for r in reasons]) if reasons else "- No strong reason detected"

        return "\n".join([
            "--- BACKEND TREND REGIME OBSERVER ---",
            f"Regime: {regime.get('regime', 'UNKNOWN')}",
            f"Market Bias: {regime.get('market_bias', 'UNKNOWN')}",
            f"Market Phase: {regime.get('market_phase', 'UNKNOWN')}",
            f"Preferred Side: {regime.get('preferred_side', 'NONE')}",
            f"Blocked Side: {regime.get('blocked_side', 'NONE')}",
            f"Confidence: {regime.get('confidence', 0.0)}",
            f"Setup Bias: {regime.get('setup_bias', 'NO_CLEAR_SETUP')}",
            f"Hard Block Active: {'YES' if regime.get('hard_block_active') else 'NO'}",
            f"Current Price: {_fmt_price(regime.get('current_price'))}",
            f"Recent Micro Swing High: {_fmt_price(regime.get('recent_swing_high'))}",
            f"Recent Micro Swing Low: {_fmt_price(regime.get('recent_swing_low'))}",
            f"Confirm TF Bias: {regime.get('confirm_bias', 'UNKNOWN')}",
            f"Micro TF Bias: {regime.get('micro_bias', 'UNKNOWN')}",
            f"Confirm Structure: {regime.get('confirm_structure', 'UNKNOWN')}",
            f"Micro Structure: {regime.get('micro_structure', 'UNKNOWN')}",
            f"Local Direction: {regime.get('local_direction', 'NEUTRAL')}",
            f"Confirm Direction: {regime.get('confirm_direction', 'NEUTRAL')}",
            f"Swing Direction: {regime.get('swing_direction', 'NEUTRAL')}",
            f"Macro Direction: {regime.get('macro_direction', 'NEUTRAL')}",
            f"Primary Direction: {regime.get('primary_direction', 'WAIT')}",
            f"HTF Alignment: {regime.get('htf_alignment', 'NEUTRAL')}",
            f"Trend Transition State: {regime.get('trend_transition_state', 'UNKNOWN')}",
            f"Direction Agreement: {regime.get('direction_agreement', 0.0)}",
            f"Direction Score: {regime.get('direction_score', 50)}",
            f"Allowed Trade Mode: {regime.get('allowed_trade_mode', 'FULL_TRADE')}",
            f"Direction Change Risk: {regime.get('direction_change_risk', 'LOW')}",
            f"CHoCH Status: {regime.get('choch_status', 'CHOCH_NONE')}",
            f"CHoCH Severity: {regime.get('choch_severity', 'NONE')}",
            f"CHoCH Timeframe: {regime.get('choch_timeframe', 'NONE')}",
            f"CHoCH Direction: {regime.get('choch_direction', 'NONE')}",
            f"CHoCH Broken Level: {regime.get('choch_broken_level')}",
            f"CHoCH Retest Status: {regime.get('choch_retest_status', 'NOT_AVAILABLE')}",
            f"CHoCH Break Age Bars: {regime.get('choch_break_age_bars')}",
            f"CHoCH Retest Reason: {regime.get('choch_retest_reason', '')}",
            f"CHoCH Reason: {regime.get('choch_reason', '')}",
            f"Bull Score: {regime.get('bull_score', 0.0)}",
            f"Bear Score: {regime.get('bear_score', 0.0)}",
            f"Context Available: {'YES' if regime.get('context_available') else 'NO'}",
            f"Context Conflict: {'YES' if regime.get('context_conflict') else 'NO'}",
            f"Context Conflict Reason: {regime.get('context_conflict_reason', '')}",
            f"Dominant Preferred Side: {regime.get('dominant_preferred_side', '')}",
            f"CHoCH: {'YES' if regime.get('choch') else 'NO'}",
            f"Choppy/Both Rejection: {'YES' if regime.get('is_choppy') else 'NO'}",
            f"Support Rejection: {'YES' if regime.get('support_rejection') else 'NO'}",
            f"Resistance Rejection: {'YES' if regime.get('resistance_rejection') else 'NO'}",
            f"Elliott Fib Usable: {'YES' if regime.get('elliott_fib_usable') else 'NO'}",
            f"Fib Entry Source: {regime.get('fib_entry_source', '')}",
            f"Confidence Before Context: {regime.get('confidence_before_context')}",
            f"Confidence After Context: {regime.get('confidence_after_context')}",
            "Observer Note: Context-aware regime observer. It can force WAIT on CHoCH/chop conflicts.",
            "Reasons:",
            reason_text,
        ])

    except Exception as e:
        return (
            "--- BACKEND TREND REGIME OBSERVER ---\n"
            f"Trend Regime Formatting Error: {str(e)}"
        )


def build_backend_trend_regime_context(
    df_macro: Optional[pd.DataFrame] = None,
    df_swing: Optional[pd.DataFrame] = None,
    df_confirm: Optional[pd.DataFrame] = None,
    df_micro: Optional[pd.DataFrame] = None,
    user_tf: str = "15m",
    context_bundle: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Main function to call from main.py if text-only context is needed.
    """
    regime = detect_backend_trend_regime(
        df_macro=df_macro,
        df_swing=df_swing,
        df_confirm=df_confirm,
        df_micro=df_micro,
        user_tf=user_tf,
        context_bundle=context_bundle,
    )
    return format_backend_trend_regime_context(regime)
