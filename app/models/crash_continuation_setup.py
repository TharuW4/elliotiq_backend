# app/models/crash_continuation_setup.py

"""
Crash Continuation Setup Generator

Purpose:
- Build conditional continuation setups during strong bearish crash / bullish squeeze moves.
- Intended as a fallback when the normal backend planner returns WAIT / MISSED / NO_CHASE.
- Generates entry, SL, TP1, TP2 levels for a pullback/retest continuation setup.

Important:
- Does NOT fetch candles.
- Does NOT call AI.
- Does NOT replace normal valid LONG/SHORT setups.
- Does NOT create market-chase entries.
- Does NOT set can_enter_now=True in v1.
- Designed for safe conditional setup output:
    SHORT crash -> wait for pullback/retest + bearish rejection.
    LONG squeeze -> wait for pullback/retest + bullish reclaim.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ==========================================================
# Tunable constants
# ==========================================================

TP1_R = 1.5
TP2_R = 3.0

MIN_FAST_OR_SELECTED_PRESSURE_SCORE = 34
MIN_STRONG_PRESSURE_SCORE = 45

MIN_MOVE_ATR = 1.05
MIN_BODY_RATIO = 0.45

# Early continuation acceptance:
# Allows conditional retest setup when fast TF already shows real displacement,
# even if full crash regime score is still neutral.
# This still does NOT allow market entry.
EARLY_FAST_PRESSURE_SCORE = 25
EARLY_FAST_MOVE_ATR = 1.15
EARLY_SELECTED_PRESSURE_SCORE = 10
EARLY_SELECTED_MOVE_ATR = 0.80

MIN_RR = 2.0
MAX_RISK_PCT_1H = 2.8
MAX_RISK_PCT_4H = 4.0
MAX_RISK_PCT_15M = 1.8

MIN_TP1_DISTANCE_ATR = 0.65
MAX_STOP_ATR = 2.20

MIN_STOP_ATR_MULTIPLIER = 1.20
EXTRA_SL_ATR_BUFFER = 0.15

PULLBACK_ATR_FRACTION = 0.45
FALLBACK_STOP_ATR = 1.20
STRUCTURE_STOP_BUFFER_ATR = 0.25

LOOKBACK_STRUCTURE = 40


# ==========================================================
# Basic helpers
# ==========================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = (
                value.replace(",", "")
                .replace("$", "")
                .replace("%", "")
                .strip()
            )
            if not value:
                return default

        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except Exception:
        return default


def _upper(value: Any) -> str:
    return str(value or "").upper().strip()


def _tf_key(value: Any) -> str:
    return str(value or "").lower().strip()


def _fmt_price(value: Any) -> str:
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


def _normalize_side(value: Any) -> str:
    s = _upper(value)

    if s in ["LONG", "BUY", "BULL", "BULLISH"] or "LONG" in s:
        return "LONG"

    if s in ["SHORT", "SELL", "BEAR", "BEARISH"] or "SHORT" in s:
        return "SHORT"

    return "NONE"


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

        if pd.isna(val) or val <= 0:
            return None

        return float(val)

    except Exception:
        return None


def _ema(series: pd.Series, span: int) -> Optional[float]:
    try:
        if series is None or len(series) < max(10, span // 2):
            return None

        val = series.ewm(span=span, adjust=False).mean().iloc[-1]
        if pd.isna(val):
            return None

        return float(val)

    except Exception:
        return None


def _body_ratio(row: pd.Series) -> float:
    try:
        high = float(row["High"])
        low = float(row["Low"])
        open_ = float(row["Open"])
        close = float(row["Close"])

        rng = max(high - low, 0.0)
        if rng <= 0:
            return 0.0

        return abs(close - open_) / rng

    except Exception:
        return 0.0


def _close_location(row: pd.Series) -> float:
    """
    0.0 = close near low
    1.0 = close near high
    """
    try:
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        rng = high - low
        if rng <= 0:
            return 0.5

        return max(0.0, min(1.0, (close - low) / rng))

    except Exception:
        return 0.5


def _max_risk_pct_for_tf(interval: str) -> float:
    tf = _tf_key(interval)

    if tf in ["15m", "30m"]:
        return MAX_RISK_PCT_15M

    if tf in ["4h", "240m"]:
        return MAX_RISK_PCT_4H

    return MAX_RISK_PCT_1H


# ==========================================================
# Side inference helpers
# ==========================================================

def _side_from_wait_text(scenario: Optional[Dict[str, Any]]) -> str:
    try:
        s = scenario or {}

        text = " ".join([
            str(s.get("summary") or ""),
            str(s.get("generated_advice") or ""),
            str(s.get("advice") or ""),
            str(s.get("reason") or ""),
            str(s.get("status") or ""),
        ]).upper()

        if (
            "SHORT IDEA" in text
            or "SHORT SETUP" in text
            or "BEARISH IDEA" in text
            or "BEARISH SETUP" in text
            or "SELL IDEA" in text
        ):
            return "SHORT"

        if (
            "LONG IDEA" in text
            or "LONG SETUP" in text
            or "BULLISH IDEA" in text
            or "BULLISH SETUP" in text
            or "BUY IDEA" in text
        ):
            return "LONG"

    except Exception:
        pass

    return "NONE"


def _side_from_backend_plan(backend_setup_plan: Optional[Dict[str, Any]]) -> str:
    try:
        plan = backend_setup_plan or {}
        candidates = plan.get("candidates") or []

        counts = {"LONG": 0, "SHORT": 0}

        for c in candidates:
            if not isinstance(c, dict):
                continue

            side = _normalize_side(c.get("side") or c.get("trade_type"))
            if side in ["LONG", "SHORT"]:
                counts[side] += 1

        if counts["LONG"] > counts["SHORT"]:
            return "LONG"

        if counts["SHORT"] > counts["LONG"]:
            return "SHORT"

    except Exception:
        pass

    return "NONE"


def _side_from_crash_regime(crash_regime: Optional[Dict[str, Any]]) -> str:
    try:
        crash = crash_regime or {}
        direction = _normalize_side(crash.get("direction"))
        score = int(_safe_float(crash.get("score"), 0) or 0)

        if direction in ["LONG", "SHORT"] and score >= 25:
            return direction

    except Exception:
        pass

    return "NONE"


def _infer_attempted_side(
    *,
    scenario: Optional[Dict[str, Any]],
    backend_setup_plan: Optional[Dict[str, Any]],
    crash_regime: Optional[Dict[str, Any]],
) -> str:
    side = _normalize_side(
        (scenario or {}).get("trade_type")
        or (scenario or {}).get("side")
    )

    if side in ["LONG", "SHORT"]:
        return side

    side = _side_from_backend_plan(backend_setup_plan)
    if side in ["LONG", "SHORT"]:
        return side

    side = _side_from_wait_text(scenario)
    if side in ["LONG", "SHORT"]:
        return side

    side = _side_from_crash_regime(crash_regime)
    if side in ["LONG", "SHORT"]:
        return side

    return "NONE"


# ==========================================================
# Crash pressure helpers
# ==========================================================

def _pressure_scores(crash_regime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "direction": "NONE",
        "score": 0,
        "selected_bearish_score": 0,
        "selected_bullish_score": 0,
        "fast_bearish_score": 0,
        "fast_bullish_score": 0,
        "selected_move_atr_5": 0.0,
        "fast_move_atr_5": 0.0,
        "selected_tf_signal": "",
        "fast_tf_signal": "",
        "agreement": "",
    }

    try:
        crash = crash_regime or {}

        out["direction"] = _normalize_side(crash.get("direction"))
        out["score"] = int(_safe_float(crash.get("score"), 0) or 0)
        out["selected_tf_signal"] = str(crash.get("selected_tf_signal") or "")
        out["fast_tf_signal"] = str(crash.get("fast_tf_signal") or "")
        out["agreement"] = str(crash.get("agreement") or "")

        selected = crash.get("selected_tf_details") or {}
        fast = crash.get("fast_tf_details") or {}

        out["selected_bearish_score"] = int(_safe_float(selected.get("bearish_score"), 0) or 0)
        out["selected_bullish_score"] = int(_safe_float(selected.get("bullish_score"), 0) or 0)
        out["fast_bearish_score"] = int(_safe_float(fast.get("bearish_score"), 0) or 0)
        out["fast_bullish_score"] = int(_safe_float(fast.get("bullish_score"), 0) or 0)

        out["selected_move_atr_5"] = float(_safe_float(selected.get("move_atr_5"), 0.0) or 0.0)
        out["fast_move_atr_5"] = float(_safe_float(fast.get("move_atr_5"), 0.0) or 0.0)

        return out

    except Exception:
        return out


def _has_enough_pressure(
    side: str,
    crash_regime: Optional[Dict[str, Any]],
    d: pd.DataFrame,
) -> Tuple[bool, str, Dict[str, Any]]:
    info = _pressure_scores(crash_regime)

    try:
        side = _normalize_side(side)

        if side not in ["LONG", "SHORT"]:
            return False, "No attempted side for crash continuation.", info

        if d.empty or len(d) < 30:
            return False, "Not enough candles for pressure validation.", info

        last = d.iloc[-1]
        atr_val = _atr(d)

        if atr_val is None or atr_val <= 0:
            return False, "ATR unavailable for pressure validation.", info

        close = float(last["Close"])
        open_ = float(last["Open"])
        body = abs(close - open_)
        body_atr = body / atr_val
        body_ratio = _body_ratio(last)
        close_loc = _close_location(last)

        if len(d) >= 6:
            prev_close = float(d["Close"].iloc[-6])
            move = close - prev_close
            move_atr = abs(move) / atr_val
        else:
            move = 0.0
            move_atr = 0.0

        selected_bear = info["selected_bearish_score"]
        fast_bear = info["fast_bearish_score"]
        selected_bull = info["selected_bullish_score"]
        fast_bull = info["fast_bullish_score"]

        if side == "SHORT":
            directional_scores_ok = (
                selected_bear >= MIN_FAST_OR_SELECTED_PRESSURE_SCORE
                or fast_bear >= MIN_FAST_OR_SELECTED_PRESSURE_SCORE
                or info["score"] >= MIN_STRONG_PRESSURE_SCORE
            )

            candle_ok = (
                close < open_
                and body_ratio >= MIN_BODY_RATIO
                and close_loc <= 0.45
                and (body_atr >= 0.55 or move_atr >= MIN_MOVE_ATR)
            )

            move_ok = move < 0 and move_atr >= MIN_MOVE_ATR

            early_fast_continuation_ok = (
                fast_bear >= EARLY_FAST_PRESSURE_SCORE
                and info["fast_move_atr_5"] >= EARLY_FAST_MOVE_ATR
                and selected_bear >= EARLY_SELECTED_PRESSURE_SCORE
                and info["selected_move_atr_5"] >= EARLY_SELECTED_MOVE_ATR
                and move < 0
            )

            if directional_scores_ok or candle_ok or move_ok or early_fast_continuation_ok:
                reason_mode = (
                    "early fast-TF continuation"
                    if early_fast_continuation_ok and not (directional_scores_ok or candle_ok or move_ok)
                    else "standard pressure"
                )

                return True, (
                    f"Bearish crash-continuation pressure accepted by {reason_mode}. "
                    f"selectedBear={selected_bear}, fastBear={fast_bear}, "
                    f"selectedMoveATR5={round(info['selected_move_atr_5'], 2)}, "
                    f"fastMoveATR5={round(info['fast_move_atr_5'], 2)}, "
                    f"moveATR={round(move_atr, 2)}, bodyATR={round(body_atr, 2)}."
                ), info

            return False, (
                f"Bearish pressure not strong enough. "
                f"selectedBear={selected_bear}, fastBear={fast_bear}, "
                f"selectedMoveATR5={round(info['selected_move_atr_5'], 2)}, "
                f"fastMoveATR5={round(info['fast_move_atr_5'], 2)}, "
                f"moveATR={round(move_atr, 2)}, bodyATR={round(body_atr, 2)}."
            ), info

        if side == "LONG":
            directional_scores_ok = (
                selected_bull >= MIN_FAST_OR_SELECTED_PRESSURE_SCORE
                or fast_bull >= MIN_FAST_OR_SELECTED_PRESSURE_SCORE
                or info["score"] >= MIN_STRONG_PRESSURE_SCORE
            )

            candle_ok = (
                close > open_
                and body_ratio >= MIN_BODY_RATIO
                and close_loc >= 0.55
                and (body_atr >= 0.55 or move_atr >= MIN_MOVE_ATR)
            )

            move_ok = move > 0 and move_atr >= MIN_MOVE_ATR

            early_fast_continuation_ok = (
                fast_bull >= EARLY_FAST_PRESSURE_SCORE
                and info["fast_move_atr_5"] >= EARLY_FAST_MOVE_ATR
                and selected_bull >= EARLY_SELECTED_PRESSURE_SCORE
                and info["selected_move_atr_5"] >= EARLY_SELECTED_MOVE_ATR
                and move > 0
            )

            if directional_scores_ok or candle_ok or move_ok or early_fast_continuation_ok:
                reason_mode = (
                    "early fast-TF continuation"
                    if early_fast_continuation_ok and not (directional_scores_ok or candle_ok or move_ok)
                    else "standard pressure"
                )

                return True, (
                    f"Bullish squeeze-continuation pressure accepted by {reason_mode}. "
                    f"selectedBull={selected_bull}, fastBull={fast_bull}, "
                    f"selectedMoveATR5={round(info['selected_move_atr_5'], 2)}, "
                    f"fastMoveATR5={round(info['fast_move_atr_5'], 2)}, "
                    f"moveATR={round(move_atr, 2)}, bodyATR={round(body_atr, 2)}."
                ), info

            return False, (
                f"Bullish pressure not strong enough. "
                f"selectedBull={selected_bull}, fastBull={fast_bull}, "
                f"selectedMoveATR5={round(info['selected_move_atr_5'], 2)}, "
                f"fastMoveATR5={round(info['fast_move_atr_5'], 2)}, "
                f"moveATR={round(move_atr, 2)}, bodyATR={round(body_atr, 2)}."
            ), info

    except Exception as exc:
        return False, f"Pressure validation failed: {exc}", info

    return False, "Unknown pressure validation result.", info


# ==========================================================
# Structure / level building
# ==========================================================

def _recent_structure(d: pd.DataFrame, lookback: int = LOOKBACK_STRUCTURE) -> Dict[str, Any]:
    out = {
        "recent_high": None,
        "recent_low": None,
        "previous_high": None,
        "previous_low": None,
        "highest": None,
        "lowest": None,
    }

    try:
        x = _normalize_ohlc(d).tail(lookback)

        if len(x) < 10:
            return out

        out["highest"] = float(x["High"].max())
        out["lowest"] = float(x["Low"].min())

        recent = x.tail(min(12, len(x)))
        prior = x.iloc[:-min(12, len(x))] if len(x) > 12 else x

        out["recent_high"] = float(recent["High"].max())
        out["recent_low"] = float(recent["Low"].min())
        out["previous_high"] = float(prior["High"].max()) if not prior.empty else out["recent_high"]
        out["previous_low"] = float(prior["Low"].min()) if not prior.empty else out["recent_low"]

        return out

    except Exception:
        return out


def _build_pullback_entry(
    *,
    side: str,
    d: pd.DataFrame,
    atr_val: float,
) -> Dict[str, Any]:
    out = {
        "entry": None,
        "entry_zone_low": None,
        "entry_zone_high": None,
        "entry_source": "",
        "reason": "",
    }

    try:
        side = _normalize_side(side)

        if side not in ["LONG", "SHORT"]:
            out["reason"] = "Invalid side for pullback entry."
            return out

        x = _normalize_ohlc(d)
        close = float(x.iloc[-1]["Close"])

        ema20 = _ema(x["Close"], 20)
        ema50 = _ema(x["Close"], 50)

        structure = _recent_structure(x)
        recent_high = _safe_float(structure.get("recent_high"))
        recent_low = _safe_float(structure.get("recent_low"))
        previous_high = _safe_float(structure.get("previous_high"))
        previous_low = _safe_float(structure.get("previous_low"))

        candidates: List[Tuple[str, float]] = []

        if side == "SHORT":
            for label, level in [
                ("EMA20 pullback retest", ema20),
                ("EMA50 pullback retest", ema50),
                ("Recent low breakdown retest", recent_low),
                ("Previous low breakdown retest", previous_low),
            ]:
                lv = _safe_float(level)
                if lv is not None and lv > close:
                    candidates.append((label, lv))

            if candidates:
                candidates.sort(key=lambda x_: abs(x_[1] - close))
                label, entry = candidates[0]
            else:
                label = "ATR pullback retest"
                entry = close + (atr_val * PULLBACK_ATR_FRACTION)

            zone_low = entry - (atr_val * 0.12)
            zone_high = entry + (atr_val * 0.12)

            out.update({
                "entry": float(entry),
                "entry_zone_low": float(zone_low),
                "entry_zone_high": float(zone_high),
                "entry_source": label,
                "reason": f"SHORT continuation entry built from {label}.",
            })
            return out

        if side == "LONG":
            for label, level in [
                ("EMA20 pullback retest", ema20),
                ("EMA50 pullback retest", ema50),
                ("Recent high breakout retest", recent_high),
                ("Previous high breakout retest", previous_high),
            ]:
                lv = _safe_float(level)
                if lv is not None and lv < close:
                    candidates.append((label, lv))

            if candidates:
                candidates.sort(key=lambda x_: abs(x_[1] - close))
                label, entry = candidates[0]
            else:
                label = "ATR pullback retest"
                entry = close - (atr_val * PULLBACK_ATR_FRACTION)

            zone_low = entry - (atr_val * 0.12)
            zone_high = entry + (atr_val * 0.12)

            out.update({
                "entry": float(entry),
                "entry_zone_low": float(zone_low),
                "entry_zone_high": float(zone_high),
                "entry_source": label,
                "reason": f"LONG continuation entry built from {label}.",
            })
            return out

    except Exception as exc:
        out["reason"] = f"Pullback entry build failed: {exc}"
        return out

    return out


def _build_stop_loss(
    *,
    side: str,
    d: pd.DataFrame,
    entry: float,
    atr_val: float,
) -> Dict[str, Any]:
    out = {
        "stop_loss": None,
        "source": "",
        "reason": "",
    }

    try:
        side = _normalize_side(side)
        structure = _recent_structure(d)

        recent_high = _safe_float(structure.get("recent_high"))
        recent_low = _safe_float(structure.get("recent_low"))

        if side == "SHORT":
            structure_stop = None
            if recent_high is not None and recent_high > entry:
                structure_stop = recent_high + (atr_val * STRUCTURE_STOP_BUFFER_ATR)

            fallback_stop = entry + (atr_val * FALLBACK_STOP_ATR)

            if structure_stop is not None:
                stop = min(structure_stop, fallback_stop + (atr_val * 0.75))
                source = "recent high structure"
            else:
                stop = fallback_stop
                source = "ATR fallback"

            if stop <= entry:
                stop = entry + (atr_val * FALLBACK_STOP_ATR)

            out.update({
                "stop_loss": float(stop),
                "source": source,
                "reason": f"SHORT stop placed above {source}.",
            })
            return out

        if side == "LONG":
            structure_stop = None
            if recent_low is not None and recent_low < entry:
                structure_stop = recent_low - (atr_val * STRUCTURE_STOP_BUFFER_ATR)

            fallback_stop = entry - (atr_val * FALLBACK_STOP_ATR)

            if structure_stop is not None:
                stop = max(structure_stop, fallback_stop - (atr_val * 0.75))
                source = "recent low structure"
            else:
                stop = fallback_stop
                source = "ATR fallback"

            if stop >= entry:
                stop = entry - (atr_val * FALLBACK_STOP_ATR)

            out.update({
                "stop_loss": float(stop),
                "source": source,
                "reason": f"LONG stop placed below {source}.",
            })
            return out

    except Exception as exc:
        out["reason"] = f"Stop build failed: {exc}"
        return out

    return out

def _apply_minimum_stop_distance(
    *,
    side: str,
    entry: float,
    stop: float,
    atr_val: float,
) -> float:
    """
    Make sure crash continuation SL is not too tight.

    This protects against:
    - wick stop-outs
    - shallow retest noise
    - crash pullback volatility

    TP1/TP2 must be recalculated after this adjustment.
    """
    try:
        side = _normalize_side(side)

        if atr_val <= 0:
            return float(stop)

        min_distance = atr_val * MIN_STOP_ATR_MULTIPLIER
        extra_buffer = atr_val * EXTRA_SL_ATR_BUFFER

        if side == "SHORT":
            current_distance = float(stop) - float(entry)
            if current_distance < min_distance:
                return float(entry) + min_distance + extra_buffer

        if side == "LONG":
            current_distance = float(entry) - float(stop)
            if current_distance < min_distance:
                return float(entry) - min_distance - extra_buffer

        return float(stop)

    except Exception:
        return float(stop)

def _build_targets(
    *,
    side: str,
    entry: float,
    stop: float,
    atr_val: float,
) -> Dict[str, Any]:
    out = {
        "tp1": None,
        "tp2": None,
        "risk": None,
        "rr": None,
        "reason": "",
    }

    try:
        side = _normalize_side(side)

        if side == "SHORT":
            risk = stop - entry
            if risk <= 0:
                out["reason"] = "Invalid SHORT risk."
                return out

            tp1 = entry - (risk * TP1_R)
            tp2 = entry - (risk * TP2_R)

        elif side == "LONG":
            risk = entry - stop
            if risk <= 0:
                out["reason"] = "Invalid LONG risk."
                return out

            tp1 = entry + (risk * TP1_R)
            tp2 = entry + (risk * TP2_R)

        else:
            out["reason"] = "Invalid side for target build."
            return out

        rr = abs(tp2 - entry) / risk if risk > 0 else 0.0

        out.update({
            "tp1": float(tp1),
            "tp2": float(tp2),
            "risk": float(risk),
            "rr": float(rr),
            "reason": "Targets built from fixed R model.",
        })

        return out

    except Exception as exc:
        out["reason"] = f"Target build failed: {exc}"
        return out


# ==========================================================
# Continuation safety guard
# ==========================================================

def _continuation_safety_guard(
    *,
    side: str,
    trend_regime: Optional[Dict[str, Any]],
) -> Tuple[bool, str, List[str], int, Dict[str, Any]]:
    """
    Guard crash/squeeze continuation setups using only:
    - CHoCH
    - HTF alignment
    - direction_change_risk

    This affects only crash_continuation_setup fallback.
    Normal setup planner is not touched.
    """
    warnings: List[str] = []
    penalty = 0

    try:
        side = _normalize_side(side)
        tr = trend_regime or {}

        if not isinstance(tr, dict) or not tr:
            return True, "Continuation guard unavailable; allowing conditional setup.", warnings, penalty, {
                "available": False,
            }

        htf_alignment = _upper(tr.get("htf_alignment"))
        direction_change_risk = _upper(tr.get("direction_change_risk"))

        # Support both old/new CHoCH style fields.
        choch_raw = tr.get("choch")
        choch_status = _upper(tr.get("choch_status"))
        choch_severity = _upper(tr.get("choch_severity"))
        choch_direction = _upper(tr.get("choch_direction"))

        # Old binary CHoCH fallback.
        choch_active = bool(choch_raw) or (
            choch_status not in ["", "NONE", "CHOCH_NONE", "UNKNOWN"]
        )

        details = {
            "available": True,
            "side": side,
            "choch": bool(choch_active),
            "choch_status": choch_status,
            "choch_severity": choch_severity,
            "choch_direction": choch_direction,
            "htf_alignment": htf_alignment,
            "direction_change_risk": direction_change_risk,
        }

        # 1) HTF conflict = no continuation fallback.
        if htf_alignment == "CONFLICT":
            return False, (
                "Continuation blocked: HTF alignment is CONFLICT."
            ), warnings, penalty, details

        # 2) Direction-change risk HIGH = block continuation fallback.
        if direction_change_risk == "HIGH":
            return False, (
                "Continuation blocked: direction change risk is HIGH."
            ), warnings, penalty, details

        # 3) CHoCH against the attempted side = block if meaningful.
        adverse_choch = (
            (side == "LONG" and choch_direction == "BEARISH")
            or (side == "SHORT" and choch_direction == "BULLISH")
        )

        if adverse_choch and choch_severity in ["MEDIUM", "STRONG", "CONFIRMED_REVERSAL"]:
            return False, (
                f"Continuation blocked: adverse {choch_severity} {choch_direction} CHoCH against {side}."
            ), warnings, penalty, details

        # 4) Old CHoCH active but no direction/severity detail.
        # Do not block blindly; reduce quality only.
        if choch_active and not choch_direction:
            penalty += 8
            warnings.append(
                "CHoCH is active but direction/severity is unclear; continuation quality reduced."
            )

        # 5) HTF against = allow only as conditional setup, lower quality.
        if htf_alignment == "AGAINST":
            penalty += 10
            warnings.append(
                "HTF alignment is AGAINST; continuation setup must wait for retest confirmation."
            )

        # 6) Medium risk = allow but reduce quality.
        if direction_change_risk == "MEDIUM":
            penalty += 8
            warnings.append(
                "Direction change risk is MEDIUM; continuation setup quality reduced."
            )

        # 7) Minor adverse CHoCH = warning only.
        if adverse_choch and choch_severity in ["MINOR", "LOW"]:
            penalty += 6
            warnings.append(
                f"Minor adverse {choch_direction} CHoCH detected; require stronger confirmation."
            )

        return True, "Continuation safety guard passed.", warnings, penalty, details

    except Exception as exc:
        warnings.append(f"Continuation safety guard error: {exc}")
        return True, f"Continuation safety guard error: {exc}", warnings, penalty, {
            "available": False,
            "error": str(exc),
        }

# ==========================================================
# Validation
# ==========================================================

def _validate_setup(
    *,
    side: str,
    interval: str,
    current_price: float,
    entry: float,
    stop: float,
    tp1: float,
    tp2: float,
    risk: float,
    atr_val: float,
) -> Tuple[bool, List[str], List[str], int]:
    reject_reasons: List[str] = []
    warnings: List[str] = []
    score = 55

    try:
        side = _normalize_side(side)

        if side not in ["LONG", "SHORT"]:
            reject_reasons.append("Invalid side.")
            return False, reject_reasons, warnings, 0

        if entry <= 0 or stop <= 0 or tp1 <= 0 or tp2 <= 0:
            reject_reasons.append("Invalid price level.")
            return False, reject_reasons, warnings, 0

        if risk <= 0:
            reject_reasons.append("Invalid risk distance.")
            return False, reject_reasons, warnings, 0

        risk_pct = (risk / entry) * 100.0
        max_risk_pct = _max_risk_pct_for_tf(interval)

        if risk_pct > max_risk_pct:
            reject_reasons.append(
                f"Risk percent too high for timeframe: {risk_pct:.2f}% > {max_risk_pct:.2f}%."
            )

        stop_atr = risk / atr_val if atr_val > 0 else 99.0
        if stop_atr > MAX_STOP_ATR:
            reject_reasons.append(
                f"Stop too wide for crash continuation: {stop_atr:.2f} ATR."
            )

        tp1_distance_atr = abs(tp1 - entry) / atr_val if atr_val > 0 else 0.0
        if tp1_distance_atr < MIN_TP1_DISTANCE_ATR:
            reject_reasons.append(
                f"TP1 distance too small: {tp1_distance_atr:.2f} ATR."
            )

        rr = abs(tp2 - entry) / risk if risk > 0 else 0.0
        if rr < MIN_RR:
            reject_reasons.append(f"RR too low: {rr:.2f}R.")

        if side == "SHORT":
            if entry <= current_price:
                reject_reasons.append("SHORT continuation entry is not a pullback above current price.")
            if stop <= entry:
                reject_reasons.append("SHORT stop is not above entry.")
            if not (tp1 < entry and tp2 < tp1):
                reject_reasons.append("SHORT targets are not below entry in correct order.")

        if side == "LONG":
            if entry >= current_price:
                reject_reasons.append("LONG continuation entry is not a pullback below current price.")
            if stop >= entry:
                reject_reasons.append("LONG stop is not below entry.")
            if not (tp1 > entry and tp2 > tp1):
                reject_reasons.append("LONG targets are not above entry in correct order.")

        if risk_pct <= max_risk_pct * 0.65:
            score += 8
        elif risk_pct <= max_risk_pct:
            score += 3

        if stop_atr <= 1.6:
            score += 7
        elif stop_atr <= MAX_STOP_ATR:
            score += 3

        if rr >= 3.0:
            score += 8
        elif rr >= 2.3:
            score += 4

        if tp1_distance_atr >= 1.0:
            score += 6
        elif tp1_distance_atr >= MIN_TP1_DISTANCE_ATR:
            score += 2

        if reject_reasons:
            return False, reject_reasons, warnings, max(0, min(100, score - 25))

        return True, reject_reasons, warnings, max(0, min(100, score))

    except Exception as exc:
        return False, [f"Validation failed: {exc}"], warnings, 0


# ==========================================================
# Main evaluator
# ==========================================================

def evaluate_crash_continuation_setup(
    *,
    df_selected: Optional[pd.DataFrame],
    df_fast: Optional[pd.DataFrame] = None,
    interval: str = "",
    scenario: Optional[Dict[str, Any]] = None,
    backend_setup_plan: Optional[Dict[str, Any]] = None,
    crash_regime: Optional[Dict[str, Any]] = None,
    trend_regime: Optional[Dict[str, Any]] = None,
    symbol: str = "",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "enabled": True,
        "setup_found": False,
        "side": "NONE",
        "setup_type": "CRASH_NO_SETUP",
        "status": "CRASH_NO_SETUP",
        "order_type": "NONE",
        "can_enter_now": False,
        "is_executable_signal": False,
        "entry": None,
        "entry_zone_low": None,
        "entry_zone_high": None,
        "stop_loss": None,
        "tp1": None,
        "tp2": None,
        "risk": None,
        "rr": None,
        "quality_score": 0,
        "risk_level": "NONE",
        "advice": "No crash continuation setup.",
        "reason": "",
        "reject_reasons": [],
        "warnings": [],
        "details": {},
    }

    try:
        d = _normalize_ohlc(df_selected)

        if len(d) < 50:
            result.update({
                "reason": "Not enough selected timeframe candles.",
                "reject_reasons": ["Not enough selected timeframe candles."],
            })
            return result

        side = _infer_attempted_side(
            scenario=scenario,
            backend_setup_plan=backend_setup_plan,
            crash_regime=crash_regime,
        )

        result["side"] = side

        if side not in ["LONG", "SHORT"]:
            result.update({
                "reason": "No attempted LONG/SHORT side available for crash continuation.",
                "reject_reasons": ["No attempted LONG/SHORT side."],
            })
            return result

        atr_val = _atr(d)
        if atr_val is None or atr_val <= 0:
            result.update({
                "reason": "ATR unavailable.",
                "reject_reasons": ["ATR unavailable."],
            })
            return result

        current_price = float(d.iloc[-1]["Close"])

        guard_ok, guard_reason, guard_warnings, guard_penalty, guard_details = _continuation_safety_guard(
            side=side,
            trend_regime=(
                trend_regime
                or (scenario or {}).get("trend_regime")
                or (scenario or {}).get("backend_trend_regime")
                or (scenario or {}).get("trend_regime_observer")
                or {}
            ),
        )

        if not guard_ok:
            result.update({
                "setup_type": "CRASH_CONTINUATION_GUARD_BLOCKED",
                "status": "CRASH_NO_SETUP",
                "reason": guard_reason,
                "reject_reasons": [guard_reason],
                "warnings": guard_warnings,
                "details": {
                    "current_price": current_price,
                    "atr": atr_val,
                    "continuation_guard": guard_details,
                },
            })
            return result

        pressure_ok, pressure_reason, pressure_info = _has_enough_pressure(
            side=side,
            crash_regime=crash_regime,
            d=d,
        )

        if not pressure_ok:
            result.update({
                "setup_type": "CRASH_PRESSURE_TOO_WEAK",
                "status": "CRASH_NO_SETUP",
                "reason": pressure_reason,
                "reject_reasons": [pressure_reason],
                "details": {
                    "current_price": current_price,
                    "atr": atr_val,
                    "pressure": pressure_info,
                    "continuation_guard": guard_details,
                },
            })
            return result

        entry_info = _build_pullback_entry(
            side=side,
            d=d,
            atr_val=atr_val,
        )

        entry = _safe_float(entry_info.get("entry"))
        if entry is None:
            result.update({
                "setup_type": "CRASH_NO_ENTRY_LEVEL",
                "status": "CRASH_WAIT_PULLBACK",
                "reason": entry_info.get("reason") or "Could not build pullback entry.",
                "reject_reasons": [entry_info.get("reason") or "Could not build pullback entry."],
            })
            return result

        stop_info = _build_stop_loss(
            side=side,
            d=d,
            entry=entry,
            atr_val=atr_val,
        )

        stop = _safe_float(stop_info.get("stop_loss"))
        if stop is None:
            result.update({
                "setup_type": "CRASH_NO_STOP",
                "status": "CRASH_WAIT_PULLBACK",
                "reason": stop_info.get("reason") or "Could not build stop.",
                "reject_reasons": [stop_info.get("reason") or "Could not build stop."],
            })
            return result

        original_stop = stop

        stop = _apply_minimum_stop_distance(
            side=side,
            entry=entry,
            stop=stop,
            atr_val=atr_val,
        )

        stop_adjusted = abs(float(stop) - float(original_stop)) > 0

        targets = _build_targets(
            side=side,
            entry=entry,
            stop=stop,
            atr_val=atr_val,
        )

        tp1 = _safe_float(targets.get("tp1"))
        tp2 = _safe_float(targets.get("tp2"))
        risk = _safe_float(targets.get("risk"))
        rr = _safe_float(targets.get("rr"))

        if tp1 is None or tp2 is None or risk is None or rr is None:
            result.update({
                "setup_type": "CRASH_NO_TARGETS",
                "status": "CRASH_WAIT_PULLBACK",
                "reason": targets.get("reason") or "Could not build targets.",
                "reject_reasons": [targets.get("reason") or "Could not build targets."],
            })
            return result

        valid, rejects, warnings, score = _validate_setup(
            side=side,
            interval=interval,
            current_price=current_price,
            entry=entry,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            risk=risk,
            atr_val=atr_val,
        )

        if guard_warnings:
            warnings.extend(guard_warnings)

        if guard_penalty:
            score = max(0, int(score) - int(guard_penalty))

        setup_type = (
            "CRASH_CONTINUATION_SHORT"
            if side == "SHORT"
            else "SQUEEZE_CONTINUATION_LONG"
        )

        if not valid:
            result.update({
                "setup_found": False,
                "setup_type": setup_type,
                "status": "CRASH_WAIT_PULLBACK",
                "order_type": "NONE",
                "can_enter_now": False,
                "is_executable_signal": False,
                "entry": entry,
                "entry_zone_low": entry_info.get("entry_zone_low"),
                "entry_zone_high": entry_info.get("entry_zone_high"),
                "stop_loss": stop,
                "tp1": tp1,
                "tp2": tp2,
                "risk": risk,
                "rr": rr,
                "quality_score": score,
                "risk_level": "HIGH",
                "advice": (
                    "Crash continuation pressure exists, but the calculated setup did not pass "
                    "risk/RR/chase validation. Do not enter."
                ),
                "reason": "Calculated crash continuation setup failed validation.",
                "reject_reasons": rejects,
                "warnings": warnings,
                "details": {
                    "symbol": symbol,
                    "interval": interval,
                    "current_price": current_price,
                    "atr": atr_val,
                    "entry_source": entry_info.get("entry_source"),
                    "entry_reason": entry_info.get("reason"),
                    "stop_source": stop_info.get("source"),
                    "original_stop_loss": original_stop,
                    "stop_adjusted_for_min_atr": stop_adjusted,
                    "stop_reason": stop_info.get("reason"),
                    "target_reason": targets.get("reason"),
                    "pressure_reason": pressure_reason,
                    "pressure": pressure_info,
                    "continuation_guard": guard_details,
                },
            })
            return result

        if score >= 78:
            risk_level = "MEDIUM"
        elif score >= 68:
            risk_level = "MEDIUM_HIGH"
        else:
            risk_level = "HIGH"

        if side == "SHORT":
            advice = (
                f"SHORT crash-continuation setup found. Do not chase current price. "
                f"Wait for pullback into {_fmt_price(entry_info.get('entry_zone_low'))} - "
                f"{_fmt_price(entry_info.get('entry_zone_high'))}, then require bearish rejection / failed reclaim."
            )
        else:
            advice = (
                f"LONG squeeze-continuation setup found. Do not chase current price. "
                f"Wait for pullback into {_fmt_price(entry_info.get('entry_zone_low'))} - "
                f"{_fmt_price(entry_info.get('entry_zone_high'))}, then require bullish reclaim / rejection."
            )

        result.update({
            "setup_found": True,
            "side": side,
            "setup_type": setup_type,
            "status": "WAIT_FOR_RETEST_CONFIRMATION",
            "order_type": "CONDITIONAL_LIMIT",
            "can_enter_now": False,
            "is_executable_signal": False,
            "entry": float(entry),
            "entry_zone_low": entry_info.get("entry_zone_low"),
            "entry_zone_high": entry_info.get("entry_zone_high"),
            "stop_loss": float(stop),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "risk": float(risk),
            "rr": float(rr),
            "quality_score": int(score),
            "risk_level": risk_level,
            "advice": advice,
            "reason": (
                f"{pressure_reason} "
                f"Entry source: {entry_info.get('entry_source')}. "
                "This is conditional; confirmation is required before entry."
            ),
            "reject_reasons": [],
            "warnings": warnings,
            "details": {
                "symbol": symbol,
                "interval": interval,
                "current_price": current_price,
                "atr": atr_val,
                "entry_source": entry_info.get("entry_source"),
                "entry_reason": entry_info.get("reason"),
                "stop_source": stop_info.get("source"),
                "stop_reason": stop_info.get("reason"),
                "target_reason": targets.get("reason"),
                "pressure_reason": pressure_reason,
                "pressure": pressure_info,
                "continuation_guard": guard_details,
            },
        })

        return result

    except Exception as exc:
        result.update({
            "enabled": False,
            "setup_found": False,
            "setup_type": "CRASH_ERROR",
            "status": "CRASH_ERROR",
            "advice": "Crash continuation setup failed.",
            "reason": str(exc),
            "reject_reasons": [str(exc)],
        })
        return result


# ==========================================================
# Scenario conversion helpers
# ==========================================================

def build_crash_continuation_scenario(
    *,
    base_wait_scenario: Dict[str, Any],
    crash_setup: Dict[str, Any],
) -> Dict[str, Any]:
    scenario = dict(base_wait_scenario or {})

    side = _normalize_side(crash_setup.get("side"))
    targets = [
        _fmt_price(crash_setup.get("tp1")),
        _fmt_price(crash_setup.get("tp2")),
    ]

    entry_zone_low = _fmt_price(crash_setup.get("entry_zone_low"))
    entry_zone_high = _fmt_price(crash_setup.get("entry_zone_high"))

    if entry_zone_low and entry_zone_high:
        entry_zone = f"{entry_zone_low} - {entry_zone_high}"
    else:
        entry_zone = _fmt_price(crash_setup.get("entry"))

    scenario.update({
        "name": "Crash Continuation Setup",
        "trade_type": side,
        "side": side,
        "status": "WAIT_FOR_RETEST_CONFIRMATION",
        "order_type": "CONDITIONAL_LIMIT",
        "probability": "Medium-High" if int(crash_setup.get("quality_score") or 0) >= 72 else "Medium",
        "quality_score": int(crash_setup.get("quality_score") or 0),
        "summary": crash_setup.get("advice") or "",
        "generated_advice": crash_setup.get("advice") or "",
        "entry_zone": entry_zone,
        "entry": _fmt_price(crash_setup.get("entry")),
        "targets": targets,
        "target": targets[-1] if targets else "",
        "invalidation": _fmt_price(crash_setup.get("stop_loss")),
        "stop_loss": _fmt_price(crash_setup.get("stop_loss")),
        "setup_type": crash_setup.get("setup_type"),
        "setup_source": "CRASH_CONTINUATION_SETUP",
        "backend_selected": True,
        "ai_can_modify_prices": False,
        "is_executable_signal": False,
        "can_enter_now": False,
        "watch_setup": True,
        "watch_type": "CRASH_CONTINUATION_RETEST",
        "watch_side": side,
        "watch_level": crash_setup.get("entry"),
        "watch_condition": (
            "Wait for price to pull back into the entry zone and confirm with lower-timeframe "
            "rejection/reclaim before entry."
        ),
        "crash_continuation": crash_setup,
        "raw_values": {
            "entry": crash_setup.get("entry"),
            "stop_loss": crash_setup.get("stop_loss"),
            "tp1": crash_setup.get("tp1"),
            "tp2": crash_setup.get("tp2"),
            "risk": crash_setup.get("risk"),
            "rr": crash_setup.get("rr"),
            "current_price": (crash_setup.get("details") or {}).get("current_price"),
            "atr": (crash_setup.get("details") or {}).get("atr"),
        },
    })

    return scenario


def add_crash_continuation_to_ai_data(
    *,
    ai_data: Dict[str, Any],
    interval: str,
    df_selected: Optional[pd.DataFrame],
    df_fast: Optional[pd.DataFrame] = None,
    backend_setup_plan: Optional[Dict[str, Any]] = None,
    crash_regime: Optional[Dict[str, Any]] = None,
    trend_regime: Optional[Dict[str, Any]] = None,
    symbol: str = "",
    allow_wait_fallback: bool = True,
) -> Dict[str, Any]:
    try:
        if not isinstance(ai_data, dict):
            return ai_data

        scenarios = ai_data.get("trade_scenarios") or []
        if not scenarios:
            return ai_data

        scenario = scenarios[0] if isinstance(scenarios[0], dict) else {}
        side = _normalize_side(scenario.get("trade_type") or scenario.get("side"))

        # Do not replace normal valid backend signals.
        if side in ["LONG", "SHORT"]:
            return ai_data

        if not allow_wait_fallback:
            return ai_data

        crash_setup = evaluate_crash_continuation_setup(
            df_selected=df_selected,
            df_fast=df_fast,
            interval=interval,
            scenario=scenario,
            backend_setup_plan=backend_setup_plan or ai_data.get("backend_setup_plan") or {},
            crash_regime=crash_regime or scenario.get("crash_regime") or ai_data.get("crash_regime") or {},
            trend_regime=(
                trend_regime
                or scenario.get("trend_regime")
                or scenario.get("backend_trend_regime")
                or scenario.get("trend_regime_observer")
                or ai_data.get("trend_regime")
                or ai_data.get("backend_trend_regime")
                or ai_data.get("trend_regime_observer")
                or {}
            ),
            symbol=symbol,
        )

        scenario["crash_continuation"] = crash_setup

        if crash_setup.get("setup_found"):
            crash_scenario = build_crash_continuation_scenario(
                base_wait_scenario=scenario,
                crash_setup=crash_setup,
            )

            ai_data["trade_scenarios"] = [crash_scenario]
            ai_data["selected_scenario"] = crash_scenario
            ai_data["trade_status_message"] = (
                f"WAIT | {crash_setup.get('setup_type')} | Retest Confirmation Required"
            )
            ai_data["action_advice"] = crash_scenario.get("generated_advice")

        else:
            scenarios[0] = scenario
            ai_data["trade_scenarios"] = scenarios
            ai_data["selected_scenario"] = scenario

        return ai_data

    except Exception as exc:
        try:
            scenario = (ai_data.get("trade_scenarios") or [{}])[0]
            if isinstance(scenario, dict):
                scenario["crash_continuation"] = {
                    "enabled": False,
                    "setup_found": False,
                    "setup_type": "CRASH_ERROR",
                    "status": "CRASH_ERROR",
                    "reason": str(exc),
                    "advice": "Crash continuation fallback failed.",
                }
                ai_data["trade_scenarios"] = [scenario]
                ai_data["selected_scenario"] = scenario
        except Exception:
            pass

        return ai_data