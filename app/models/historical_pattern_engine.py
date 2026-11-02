# app/models/historical_pattern_engine.py

"""
Historical Analog Engine (training-free)
========================================

Purpose
-------
Use candles that are already fetched by the ElliotIQ analysis flow to:

1. Build a normalized description of the current candle pattern.
2. Find independent, same-regime historical patterns inside the same dataframe.
3. Inspect what happened after those old patterns.
4. Return statistical evidence for Gemini / the final signal engine:
   - likely direction
   - expected favorable distance
   - expected adverse distance
   - likely market path
   - expected timing
   - trap / reversal risk
   - sample and prediction confidence
5. Optionally keep compact internal forward paths so a later AI-generated
   entry/SL/TP plan can be tested against the same historical analogs.

Important
---------
- No network/API calls are made here.
- No model training is required.
- No signal is created or blocked here.
- The caller should pass CLOSED candles only.
- Raw prices are normalized by ATR and candle shape, so different price levels
  can still be compared.
- Results are statistical evidence, not guaranteed forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ENGINE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ROLE_WINDOWS: Dict[str, int] = {
    "lower": 48,
    "main": 64,
    "confirm": 48,
    "swing": 36,
    "macro": 24,
}

DEFAULT_ROLE_WEIGHTS: Dict[str, float] = {
    "lower": 0.18,
    "main": 0.46,
    "confirm": 0.27,
    "swing": 0.07,
    "macro": 0.02,
}

DEFAULT_TIMEFRAME_HORIZONS: Dict[str, int] = {
    "5m": 36,
    "15m": 32,
    "30m": 28,
    "1h": 24,
    "2h": 20,
    "4h": 18,
    "6h": 16,
    "8h": 14,
    "12h": 12,
    "1d": 12,
    "1w": 8,
    "1mo": 6,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "role_windows": DEFAULT_ROLE_WINDOWS,
    "role_weights": DEFAULT_ROLE_WEIGHTS,
    "timeframe_horizons": DEFAULT_TIMEFRAME_HORIZONS,
    "resample_points": 24,
    "top_k": 20,
    "representative_matches": 5,
    "minimum_matches": 6,
    "good_matches": 10,
    "high_quality_matches": 16,
    "minimum_similarity": 0.58,
    "strong_similarity": 0.72,
    "stride_fraction": 0.125,
    "minimum_stride": 4,
    "minimum_match_separation_fraction": 0.55,
    "regime_filter": True,
    "allow_relaxed_regime_fallback": True,
    "recency_weight_floor": 0.72,
    "similarity_power": 2.2,
    "direction_threshold_atr": 1.0,
    "move_start_threshold_atr": 0.50,
    "deep_retest_threshold_atr": 0.75,
    "fake_breakout_initial_atr": 0.50,
    "fake_breakout_reversal_atr": 1.00,
    "max_feature_clip": 6.0,
}


FEATURE_WEIGHTS: Dict[str, float] = {
    "log_return": 1.50,
    "body_atr": 1.10,
    "range_atr": 0.95,
    "upper_wick_ratio": 0.70,
    "lower_wick_ratio": 0.70,
    "close_location": 0.75,
    "volume_zscore": 0.55,
    "ema20_distance_atr": 0.90,
    "ema50_distance_atr": 0.75,
}


@dataclass(frozen=True)
class _RegimeSignature:
    trend: str
    volatility: str
    structure: str
    momentum: str
    efficiency: float
    slope_atr: float
    atr_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": self.trend,
            "volatility": self.volatility,
            "structure": self.structure,
            "momentum": self.momentum,
            "efficiency": round(float(self.efficiency), 4),
            "slope_atr": round(float(self.slope_atr), 4),
            "atr_percent": round(float(self.atr_percent), 6),
        }


# ---------------------------------------------------------------------------
# Safe helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").replace("$", "").strip()
            if not value:
                return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return None
    denom = float(w[mask].sum())
    if denom <= 0:
        return None
    return float(np.dot(arr[mask], w[mask]) / denom)


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None

    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return None

    arr = arr[mask]
    w = w[mask]
    order = np.argsort(arr)
    arr = arr[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    if total <= 0:
        return None

    target = _clamp(quantile, 0.0, 1.0) * total
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 0), len(arr) - 1)
    return float(arr[idx])


def _weighted_rate(flags: Sequence[bool], weights: Sequence[float]) -> Optional[float]:
    if not flags or not weights or len(flags) != len(weights):
        return None
    values = [1.0 if bool(x) else 0.0 for x in flags]
    return _weighted_mean(values, weights)


def _first_index_at_or_above(values: Sequence[float], threshold: float) -> Optional[int]:
    for index, value in enumerate(values, start=1):
        number = _safe_float(value)
        if number is not None and number >= threshold:
            return index
    return None


def _first_index_at_or_below(values: Sequence[float], threshold: float) -> Optional[int]:
    for index, value in enumerate(values, start=1):
        number = _safe_float(value)
        if number is not None and number <= threshold:
            return index
    return None


def _iso_timestamp(value: Any) -> Optional[str]:
    try:
        if value is None or pd.isna(value):
            return None
        ts = pd.Timestamp(value)
        return ts.isoformat()
    except Exception:
        return str(value) if value is not None else None


def _normalize_timeframe(value: Any) -> str:
    tf = str(value or "").lower().strip()
    aliases = {
        "60m": "1h",
        "1hr": "1h",
        "1hour": "1h",
        "240m": "4h",
        "4hr": "4h",
        "4hour": "4h",
        "d": "1d",
        "daily": "1d",
        "1day": "1d",
        "1wk": "1w",
        "weekly": "1w",
        "1month": "1mo",
        "1mth": "1mo",
        "monthly": "1mo",
    }
    return aliases.get(tf, tf)


def _timeframe_minutes(value: Any) -> Optional[int]:
    tf = _normalize_timeframe(value)
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
        "1w": 10080,
        "1mo": 43200,
    }
    return mapping.get(tf)


def _deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    if not override:
        return result

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# OHLCV and indicator preparation
# ---------------------------------------------------------------------------


def _normalize_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return sorted, numeric OHLCV data using project-standard column names."""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        out = df.copy()
        rename_map = {
            "date": "Date",
            "datetime": "Date",
            "time": "Date",
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        out = out.rename(columns={column: rename_map.get(str(column).lower(), column) for column in out.columns})

        required = ["Open", "High", "Low", "Close"]
        if any(column not in out.columns for column in required):
            return pd.DataFrame()

        if "Date" not in out.columns:
            # ElliotIQ main.py normally moves Date into the DataFrame index.
            # Preserve that real candle timestamp instead of replacing it
            # with a synthetic numeric RangeIndex.
            if isinstance(out.index, pd.DatetimeIndex):
                out["Date"] = out.index
            elif str(out.index.name or "").lower() in {
                "date",
                "datetime",
                "time",
                "timestamp",
            }:
                out["Date"] = out.index
            else:
                out["Date"] = pd.RangeIndex(
                    start=0,
                    stop=len(out),
                    step=1,
                )

        if "Volume" not in out.columns:
            out["Volume"] = 0.0

        for column in ["Open", "High", "Low", "Close", "Volume"]:
            out[column] = pd.to_numeric(out[column], errors="coerce")

        try:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        except Exception:
            pass

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=required)
        out = out.loc[
            (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1))
            & (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))
            & (out["Close"] > 0)
            & (out["Open"] > 0)
        ]

        if "Date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["Date"]):
            out = out.sort_values("Date")

        out = out.drop_duplicates(subset=["Date"], keep="last")
        return out.reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


def _prepare_feature_frame(df: pd.DataFrame, max_clip: float) -> pd.DataFrame:
    d = _normalize_ohlcv(df)
    if d.empty:
        return d

    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    close = d["Close"].astype(float)
    open_ = d["Open"].astype(float)
    volume = d["Volume"].astype(float).fillna(0.0)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    atr = atr.replace(0, np.nan)

    candle_range = (high - low).replace(0, np.nan)
    body = close - open_
    body_abs = body.abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low

    ema20 = close.ewm(span=20, adjust=False, min_periods=10).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=25).mean()

    volume_mean = volume.rolling(30, min_periods=10).mean()
    volume_std = volume.rolling(30, min_periods=10).std(ddof=0).replace(0, np.nan)

    out = d.copy()
    out["ATR"] = atr
    out["ATRPercent"] = atr / close.replace(0, np.nan)
    out["EMA20"] = ema20
    out["EMA50"] = ema50
    out["log_return"] = np.log(close / previous_close.replace(0, np.nan))
    out["body_atr"] = body / atr
    out["range_atr"] = candle_range / atr
    out["upper_wick_ratio"] = upper_wick / candle_range
    out["lower_wick_ratio"] = lower_wick / candle_range
    out["close_location"] = (close - low) / candle_range
    out["volume_zscore"] = (volume - volume_mean) / volume_std
    out["ema20_distance_atr"] = (close - ema20) / atr
    out["ema50_distance_atr"] = (close - ema50) / atr

    feature_columns = list(FEATURE_WEIGHTS.keys())
    for column in feature_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out[column] = out[column].replace([np.inf, -np.inf], np.nan)

    # Neutral values are safer than discarding large early parts of the dataframe.
    neutral_fill = {
        "log_return": 0.0,
        "body_atr": 0.0,
        "range_atr": 1.0,
        "upper_wick_ratio": 0.0,
        "lower_wick_ratio": 0.0,
        "close_location": 0.5,
        "volume_zscore": 0.0,
        "ema20_distance_atr": 0.0,
        "ema50_distance_atr": 0.0,
    }
    for column, neutral in neutral_fill.items():
        out[column] = out[column].fillna(neutral).clip(-max_clip, max_clip)

    out["ATR"] = out["ATR"].ffill().bfill()
    out["ATRPercent"] = out["ATRPercent"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out["EMA20"] = out["EMA20"].ffill().bfill()
    out["EMA50"] = out["EMA50"].ffill().bfill()

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pattern encoding and regime detection
# ---------------------------------------------------------------------------


def _resample_sequence(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(points, dtype=float)
    if len(values) == 1:
        return np.repeat(values[0], points).astype(float)

    source_x = np.linspace(0.0, 1.0, num=len(values))
    target_x = np.linspace(0.0, 1.0, num=points)
    return np.interp(target_x, source_x, values)


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    median = float(np.nanmedian(arr)) if len(arr) else 0.0
    q75 = float(np.nanpercentile(arr, 75)) if len(arr) else 0.0
    q25 = float(np.nanpercentile(arr, 25)) if len(arr) else 0.0
    scale = q75 - q25
    if not math.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(arr))
    if not math.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    standardized = (arr - median) / scale
    return np.clip(standardized, -6.0, 6.0)


def _regime_signature(window: pd.DataFrame) -> _RegimeSignature:
    close = window["Close"].astype(float).to_numpy()
    atr = _safe_float(window["ATR"].iloc[-1], 0.0) or 0.0
    atr_percent = _safe_float(window["ATRPercent"].iloc[-1], 0.0) or 0.0
    ema20 = _safe_float(window["EMA20"].iloc[-1], close[-1]) or close[-1]
    ema50 = _safe_float(window["EMA50"].iloc[-1], close[-1]) or close[-1]

    span = min(10, max(2, len(window) // 4))
    previous_ema20 = _safe_float(window["EMA20"].iloc[-span], ema20) or ema20
    denominator = max(atr * max(span - 1, 1), 1e-12)
    slope_atr = (ema20 - previous_ema20) / denominator

    total_path = float(np.abs(np.diff(close)).sum()) if len(close) > 1 else 0.0
    efficiency = abs(float(close[-1] - close[0])) / total_path if total_path > 1e-12 else 0.0

    recent_atr_percent = window["ATRPercent"].tail(min(50, len(window))).replace([np.inf, -np.inf], np.nan).dropna()
    median_atr_percent = float(recent_atr_percent.median()) if not recent_atr_percent.empty else atr_percent
    volatility_ratio = atr_percent / median_atr_percent if median_atr_percent > 1e-12 else 1.0

    if volatility_ratio >= 1.28:
        volatility = "HIGH"
    elif volatility_ratio <= 0.78:
        volatility = "LOW"
    else:
        volatility = "NORMAL"

    if efficiency < 0.22:
        structure = "RANGE"
    elif efficiency < 0.38:
        structure = "MIXED"
    else:
        structure = "DIRECTIONAL"

    if ema20 > ema50 and slope_atr > 0.015:
        trend = "BULL_TREND"
    elif ema20 < ema50 and slope_atr < -0.015:
        trend = "BEAR_TREND"
    elif structure == "RANGE":
        trend = "RANGE"
    else:
        trend = "TRANSITION"

    lookback = min(6, len(close) - 1)
    if lookback > 0 and atr > 1e-12:
        momentum_atr = (close[-1] - close[-1 - lookback]) / atr
    else:
        momentum_atr = 0.0

    if momentum_atr >= 1.0:
        momentum = "STRONG_BULL"
    elif momentum_atr >= 0.25:
        momentum = "BULL"
    elif momentum_atr <= -1.0:
        momentum = "STRONG_BEAR"
    elif momentum_atr <= -0.25:
        momentum = "BEAR"
    else:
        momentum = "NEUTRAL"

    return _RegimeSignature(
        trend=trend,
        volatility=volatility,
        structure=structure,
        momentum=momentum,
        efficiency=float(efficiency),
        slope_atr=float(slope_atr),
        atr_percent=float(atr_percent),
    )


def _regime_compatibility(current: _RegimeSignature, candidate: _RegimeSignature) -> Tuple[bool, bool, float]:
    """Return strict match, relaxed match and a compatibility score from 0 to 1."""
    strict = (
        current.trend == candidate.trend
        and current.volatility == candidate.volatility
        and current.structure == candidate.structure
    )

    same_direction_family = (
        current.trend == candidate.trend
        or current.trend == "TRANSITION"
        or candidate.trend == "TRANSITION"
        or (current.trend == "RANGE" and candidate.structure in {"RANGE", "MIXED"})
    )
    volatility_close = current.volatility == candidate.volatility or "NORMAL" in {
        current.volatility,
        candidate.volatility,
    }
    structure_close = current.structure == candidate.structure or "MIXED" in {
        current.structure,
        candidate.structure,
    }
    relaxed = bool(same_direction_family and volatility_close and structure_close)

    score = 0.0
    score += 0.40 if current.trend == candidate.trend else (0.20 if same_direction_family else 0.0)
    score += 0.22 if current.volatility == candidate.volatility else (0.10 if volatility_close else 0.0)
    score += 0.20 if current.structure == candidate.structure else (0.10 if structure_close else 0.0)

    current_momentum_side = "BULL" if "BULL" in current.momentum else ("BEAR" if "BEAR" in current.momentum else "NEUTRAL")
    candidate_momentum_side = "BULL" if "BULL" in candidate.momentum else ("BEAR" if "BEAR" in candidate.momentum else "NEUTRAL")
    score += 0.10 if current_momentum_side == candidate_momentum_side else 0.0

    efficiency_gap = abs(current.efficiency - candidate.efficiency)
    score += 0.08 * max(0.0, 1.0 - efficiency_gap / 0.50)
    return strict, relaxed, _clamp(score, 0.0, 1.0)


def _pattern_vector(window: pd.DataFrame, points: int) -> np.ndarray:
    pieces: List[np.ndarray] = []

    for column, weight in FEATURE_WEIGHTS.items():
        values = window[column].to_numpy(dtype=float)
        sampled = _resample_sequence(values, points)
        standardized = _robust_standardize(sampled)
        pieces.append(standardized * float(weight))

    close = window["Close"].astype(float).to_numpy()
    atr = _safe_float(window["ATR"].iloc[-1], 0.0) or 0.0
    if len(close) > 1 and atr > 1e-12:
        normalized_close = (close - close[0]) / atr
    else:
        normalized_close = np.zeros(len(close), dtype=float)
    pieces.append(_robust_standardize(_resample_sequence(normalized_close, points)) * 1.35)

    regime = _regime_signature(window)
    summary = np.asarray(
        [
            regime.efficiency * 2.0,
            np.tanh(regime.slope_atr),
            np.tanh(regime.atr_percent * 100.0),
            1.0 if regime.trend == "BULL_TREND" else 0.0,
            1.0 if regime.trend == "BEAR_TREND" else 0.0,
            1.0 if regime.structure == "RANGE" else 0.0,
        ],
        dtype=float,
    )
    pieces.append(summary)

    vector = np.concatenate(pieces).astype(float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector = vector / norm
    return vector


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.dot(left, right) / denominator)
    return _clamp((cosine + 1.0) / 2.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Historical outcomes
# ---------------------------------------------------------------------------


def _first_touch_side(
    up_each: Sequence[float],
    down_each: Sequence[float],
    threshold: float,
) -> Tuple[str, Optional[int]]:
    for candle_number, (up_value, down_value) in enumerate(zip(up_each, down_each), start=1):
        up_hit = float(up_value) >= threshold
        down_hit = float(down_value) >= threshold

        if up_hit and down_hit:
            return "AMBIGUOUS", candle_number
        if up_hit:
            return "LONG", candle_number
        if down_hit:
            return "SHORT", candle_number

    return "NONE", None


def _path_flags(
    up_each: Sequence[float],
    down_each: Sequence[float],
    side: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    side = str(side or "").upper()
    if side == "LONG":
        favorable_each = list(up_each)
        adverse_each = list(down_each)
    else:
        favorable_each = list(down_each)
        adverse_each = list(up_each)

    move_start = float(config["move_start_threshold_atr"])
    deep_retest = float(config["deep_retest_threshold_atr"])
    fake_initial = float(config["fake_breakout_initial_atr"])
    fake_reversal = float(config["fake_breakout_reversal_atr"])

    time_to_start = _first_index_at_or_above(favorable_each, move_start)
    time_to_one = _first_index_at_or_above(favorable_each, 1.0)
    time_to_one_five = _first_index_at_or_above(favorable_each, 1.5)
    time_to_adverse_half = _first_index_at_or_above(adverse_each, 0.50)
    time_to_adverse_one = _first_index_at_or_above(adverse_each, 1.0)

    if time_to_start is None:
        adverse_before_start = max(adverse_each) if adverse_each else 0.0
    else:
        adverse_before_start = max(adverse_each[:time_to_start]) if time_to_start > 0 else 0.0

    immediate_continuation = bool(
        time_to_start is not None
        and time_to_start <= 3
        and adverse_before_start < 0.35
    )
    retest_then_continuation = bool(
        time_to_one is not None
        and time_to_adverse_half is not None
        and time_to_adverse_half < time_to_one
    )
    deep_retest_before_move = bool(
        time_to_one is not None
        and max(adverse_each[:time_to_one]) >= deep_retest
    )

    initial_favorable_time = _first_index_at_or_above(favorable_each, fake_initial)
    reversal_time = _first_index_at_or_above(adverse_each, fake_reversal)
    fake_breakout = bool(
        initial_favorable_time is not None
        and reversal_time is not None
        and initial_favorable_time < reversal_time
        and (time_to_one_five is None or reversal_time < time_to_one_five)
    )

    immediate_reversal = bool(
        time_to_adverse_one is not None
        and (time_to_start is None or time_to_adverse_one < time_to_start)
    )
    range_or_unclear = bool(time_to_one is None and time_to_adverse_one is None)

    if immediate_reversal:
        category = "IMMEDIATE_REVERSAL"
    elif fake_breakout:
        category = "FAKE_BREAKOUT_REVERSAL"
    elif retest_then_continuation:
        category = "RETEST_THEN_CONTINUATION"
    elif immediate_continuation:
        category = "IMMEDIATE_CONTINUATION"
    elif time_to_one is not None:
        category = "GRADUAL_CONTINUATION"
    else:
        category = "RANGE_OR_UNCLEAR"

    return {
        "category": category,
        "time_to_move_start": time_to_start,
        "time_to_1atr": time_to_one,
        "time_to_1_5atr": time_to_one_five,
        "adverse_before_move_atr": float(adverse_before_start),
        "immediate_continuation": immediate_continuation,
        "retest_then_continuation": retest_then_continuation,
        "deep_retest_before_move": deep_retest_before_move,
        "fake_breakout": fake_breakout,
        "immediate_reversal": immediate_reversal,
        "range_or_unclear": range_or_unclear,
    }


def _calculate_forward_outcome(
    prepared: pd.DataFrame,
    end_index: int,
    horizon: int,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if end_index < 0 or end_index + horizon >= len(prepared):
        return None

    entry_row = prepared.iloc[end_index]
    future = prepared.iloc[end_index + 1:end_index + 1 + horizon]
    if future.empty:
        return None

    entry = _safe_float(entry_row["Close"])
    atr = _safe_float(entry_row["ATR"])
    if entry is None or atr is None or atr <= 1e-12:
        return None

    highs = future["High"].astype(float).to_numpy()
    lows = future["Low"].astype(float).to_numpy()
    closes = future["Close"].astype(float).to_numpy()

    up_each = np.maximum((highs - entry) / atr, 0.0)
    down_each = np.maximum((entry - lows) / atr, 0.0)
    up_cumulative = np.maximum.accumulate(up_each)
    down_cumulative = np.maximum.accumulate(down_each)

    direction_threshold = float(config["direction_threshold_atr"])
    first_side, first_touch_candle = _first_touch_side(up_each, down_each, direction_threshold)

    terminal_return_atr = float((closes[-1] - entry) / atr)
    terminal_return_percent = float((closes[-1] - entry) / entry * 100.0)
    up_mfe_atr = float(up_cumulative[-1])
    down_mfe_atr = float(down_cumulative[-1])

    if first_side == "LONG":
        realized_direction = "LONG"
    elif first_side == "SHORT":
        realized_direction = "SHORT"
    elif first_side == "AMBIGUOUS":
        realized_direction = "AMBIGUOUS"
    elif terminal_return_atr >= 0.25:
        realized_direction = "LONG"
    elif terminal_return_atr <= -0.25:
        realized_direction = "SHORT"
    else:
        realized_direction = "RANGE"

    return {
        "entry_price": float(entry),
        "atr": float(atr),
        "horizon": int(horizon),
        "first_touch_side": first_side,
        "first_touch_candle": first_touch_candle,
        "realized_direction": realized_direction,
        "up_mfe_atr": up_mfe_atr,
        "down_mfe_atr": down_mfe_atr,
        "up_mfe_percent": float((highs.max() - entry) / entry * 100.0),
        "down_mfe_percent": float((entry - lows.min()) / entry * 100.0),
        "terminal_return_atr": terminal_return_atr,
        "terminal_return_percent": terminal_return_percent,
        "up_each_atr": up_each.tolist(),
        "down_each_atr": down_each.tolist(),
        "up_cumulative_atr": up_cumulative.tolist(),
        "down_cumulative_atr": down_cumulative.tolist(),
    }


# ---------------------------------------------------------------------------
# One-timeframe analog analysis
# ---------------------------------------------------------------------------


def _empty_timeframe_result(
    *,
    timeframe: str,
    role: str,
    reason: str,
    candles_available: int = 0,
    window_size: int = 0,
    horizon: int = 0,
) -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "INSUFFICIENT_HISTORY",
        "timeframe": timeframe,
        "role": role,
        "reason": reason,
        "candles_available": int(candles_available),
        "window_size": int(window_size),
        "forward_horizon_candles": int(horizon),
        "current_atr": None,
        "current_regime": {},
        "direction": {
            "likely_direction": "UNCLEAR",
            "long_probability": 0.0,
            "short_probability": 0.0,
            "range_probability": 1.0,
            "direction_edge": 0.0,
        },
        "expected_distance": {},
        "expected_adverse_move": {},
        "expected_path": {},
        "timing": {},
        "risk_forecast": {},
        "sample_quality": {
            "matches_found": 0,
            "independent_matches": 0,
            "average_similarity": 0.0,
            "confidence_score": 0,
            "confidence": "INSUFFICIENT_HISTORY",
            "regime_filter_mode": "NONE",
        },
        "representative_matches": [],
    }


def _candidate_end_indices(
    *,
    length: int,
    window_size: int,
    horizon: int,
    stride: int,
) -> Iterable[int]:
    first_end = window_size - 1
    last_end = length - window_size - horizon - 1
    if last_end < first_end:
        return []
    return range(first_end, last_end + 1, max(1, stride))


def _select_independent_matches(
    candidates: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    minimum_separation: int,
    minimum_similarity: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["similarity"], reverse=True):
        if candidate["similarity"] < minimum_similarity:
            continue
        if any(abs(int(candidate["end_index"]) - int(chosen["end_index"])) < minimum_separation for chosen in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _match_weight(
    *,
    similarity: float,
    end_index: int,
    maximum_history_end: int,
    regime_score: float,
    config: Mapping[str, Any],
) -> float:
    recency_floor = float(config["recency_weight_floor"])
    if maximum_history_end <= 0:
        recency_weight = 1.0
    else:
        recency_fraction = _clamp(end_index / maximum_history_end, 0.0, 1.0)
        recency_weight = recency_floor + (1.0 - recency_floor) * recency_fraction

    similarity_weight = max(similarity, 0.0) ** float(config["similarity_power"])
    regime_weight = 0.75 + 0.25 * _clamp(regime_score, 0.0, 1.0)
    return max(1e-9, similarity_weight * recency_weight * regime_weight)


def _direction_probabilities(matches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {"LONG": 0.0, "SHORT": 0.0, "RANGE": 0.0, "AMBIGUOUS": 0.0}
    for match in matches:
        direction = str(match["outcome"]["realized_direction"]).upper()
        weight = float(match["weight"])
        if direction not in totals:
            direction = "RANGE"
        totals[direction] += weight

    effective_range = totals["RANGE"] + totals["AMBIGUOUS"]
    denominator = totals["LONG"] + totals["SHORT"] + effective_range
    if denominator <= 0:
        return {
            "likely_direction": "UNCLEAR",
            "long_probability": 0.0,
            "short_probability": 0.0,
            "range_probability": 1.0,
            "direction_edge": 0.0,
        }

    long_probability = totals["LONG"] / denominator
    short_probability = totals["SHORT"] / denominator
    range_probability = effective_range / denominator

    if long_probability >= short_probability and long_probability >= range_probability:
        likely = "LONG"
    elif short_probability >= long_probability and short_probability >= range_probability:
        likely = "SHORT"
    else:
        likely = "RANGE"

    direction_edge = abs(long_probability - short_probability)
    if max(long_probability, short_probability) < 0.45:
        likely = "UNCLEAR"

    return {
        "likely_direction": likely,
        "long_probability": round(long_probability, 4),
        "short_probability": round(short_probability, 4),
        "range_probability": round(range_probability, 4),
        "direction_edge": round(direction_edge, 4),
    }


def _side_series(match: Mapping[str, Any], side: str) -> Tuple[List[float], List[float]]:
    outcome = match["outcome"]
    if side == "LONG":
        return list(outcome["up_each_atr"]), list(outcome["down_each_atr"])
    return list(outcome["down_each_atr"]), list(outcome["up_each_atr"])


def _aggregate_side_statistics(
    matches: Sequence[Dict[str, Any]],
    side: str,
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    favorable_atr: List[float] = []
    adverse_atr: List[float] = []
    favorable_percent: List[float] = []
    adverse_percent: List[float] = []
    weights: List[float] = []
    path_rows: List[Dict[str, Any]] = []

    for match in matches:
        outcome = match["outcome"]
        weight = float(match["weight"])
        if side == "LONG":
            favorable = float(outcome["up_mfe_atr"])
            adverse = float(outcome["down_mfe_atr"])
            favorable_pct = float(outcome["up_mfe_percent"])
            adverse_pct = float(outcome["down_mfe_percent"])
        else:
            favorable = float(outcome["down_mfe_atr"])
            adverse = float(outcome["up_mfe_atr"])
            favorable_pct = float(outcome["down_mfe_percent"])
            adverse_pct = float(outcome["up_mfe_percent"])

        favorable_each, adverse_each = _side_series(match, side)
        path = _path_flags(favorable_each, adverse_each, side, config)

        favorable_atr.append(favorable)
        adverse_atr.append(adverse)
        favorable_percent.append(favorable_pct)
        adverse_percent.append(adverse_pct)
        weights.append(weight)
        path_rows.append(path)

    expected_distance = {
        "side": side,
        "median_favorable_move_atr": _round_or_none(_weighted_quantile(favorable_atr, weights, 0.50), 4),
        "weak_case_p25_favorable_move_atr": _round_or_none(_weighted_quantile(favorable_atr, weights, 0.25), 4),
        "strong_case_p75_favorable_move_atr": _round_or_none(_weighted_quantile(favorable_atr, weights, 0.75), 4),
        "median_favorable_move_percent": _round_or_none(_weighted_quantile(favorable_percent, weights, 0.50), 4),
        "probability_reach_0_5atr": _round_or_none(_weighted_rate([value >= 0.50 for value in favorable_atr], weights), 4),
        "probability_reach_1atr": _round_or_none(_weighted_rate([value >= 1.00 for value in favorable_atr], weights), 4),
        "probability_reach_1_5atr": _round_or_none(_weighted_rate([value >= 1.50 for value in favorable_atr], weights), 4),
        "probability_reach_2atr": _round_or_none(_weighted_rate([value >= 2.00 for value in favorable_atr], weights), 4),
        "probability_reach_3atr": _round_or_none(_weighted_rate([value >= 3.00 for value in favorable_atr], weights), 4),
    }

    expected_adverse = {
        "side": side,
        "median_adverse_move_atr": _round_or_none(_weighted_quantile(adverse_atr, weights, 0.50), 4),
        "p75_adverse_move_atr": _round_or_none(_weighted_quantile(adverse_atr, weights, 0.75), 4),
        "median_adverse_move_percent": _round_or_none(_weighted_quantile(adverse_percent, weights, 0.50), 4),
        "probability_adverse_0_5atr": _round_or_none(_weighted_rate([value >= 0.50 for value in adverse_atr], weights), 4),
        "probability_adverse_1atr": _round_or_none(_weighted_rate([value >= 1.00 for value in adverse_atr], weights), 4),
    }

    categories = [row["category"] for row in path_rows]
    category_names = [
        "IMMEDIATE_CONTINUATION",
        "RETEST_THEN_CONTINUATION",
        "GRADUAL_CONTINUATION",
        "FAKE_BREAKOUT_REVERSAL",
        "IMMEDIATE_REVERSAL",
        "RANGE_OR_UNCLEAR",
    ]
    category_rates = {
        category: float(_weighted_rate([value == category for value in categories], weights) or 0.0)
        for category in category_names
    }
    deep_retest_rate = _weighted_rate([row["deep_retest_before_move"] for row in path_rows], weights) or 0.0
    most_likely_path = max(category_rates, key=category_rates.get) if category_rates else "UNCLEAR"

    expected_path = {
        "side": side,
        "most_likely_path": most_likely_path,
        "immediate_continuation_probability": round(category_rates["IMMEDIATE_CONTINUATION"], 4),
        "retest_then_continuation_probability": round(category_rates["RETEST_THEN_CONTINUATION"], 4),
        "gradual_continuation_probability": round(category_rates["GRADUAL_CONTINUATION"], 4),
        "deep_retest_before_move_probability": round(deep_retest_rate, 4),
        "fake_breakout_probability": round(category_rates["FAKE_BREAKOUT_REVERSAL"], 4),
        "immediate_reversal_probability": round(category_rates["IMMEDIATE_REVERSAL"], 4),
        "range_or_unclear_probability": round(category_rates["RANGE_OR_UNCLEAR"], 4),
    }

    move_start_times = [row["time_to_move_start"] for row in path_rows if row["time_to_move_start"] is not None]
    one_atr_times = [row["time_to_1atr"] for row in path_rows if row["time_to_1atr"] is not None]
    one_five_times = [row["time_to_1_5atr"] for row in path_rows if row["time_to_1_5atr"] is not None]

    move_start_weights = [weights[index] for index, row in enumerate(path_rows) if row["time_to_move_start"] is not None]
    one_atr_weights = [weights[index] for index, row in enumerate(path_rows) if row["time_to_1atr"] is not None]
    one_five_weights = [weights[index] for index, row in enumerate(path_rows) if row["time_to_1_5atr"] is not None]

    timing = {
        "side": side,
        "median_candles_to_move_start": _round_or_none(_weighted_quantile(move_start_times, move_start_weights, 0.50), 2),
        "median_candles_to_1atr": _round_or_none(_weighted_quantile(one_atr_times, one_atr_weights, 0.50), 2),
        "median_candles_to_1_5atr": _round_or_none(_weighted_quantile(one_five_times, one_five_weights, 0.50), 2),
        "p75_candles_to_move_start": _round_or_none(_weighted_quantile(move_start_times, move_start_weights, 0.75), 2),
    }

    return expected_distance, expected_adverse, expected_path, timing


def _confidence_label(score: int, matches: int, minimum_matches: int) -> str:
    if matches < minimum_matches:
        return "INSUFFICIENT_HISTORY"
    if score >= 78:
        return "HIGH"
    if score >= 62:
        return "MEDIUM_HIGH"
    if score >= 46:
        return "MEDIUM"
    if score >= 30:
        return "LOW"
    return "VERY_LOW"


def _sample_confidence(
    *,
    matches: Sequence[Dict[str, Any]],
    direction: Mapping[str, Any],
    regime_filter_mode: str,
    config: Mapping[str, Any],
) -> Tuple[int, str]:
    count = len(matches)
    if count == 0:
        return 0, "INSUFFICIENT_HISTORY"

    average_similarity = float(np.mean([float(match["similarity"]) for match in matches]))
    minimum_matches = int(config["minimum_matches"])
    good_matches = int(config["good_matches"])
    high_matches = int(config["high_quality_matches"])

    if count >= high_matches:
        sample_score = 100.0
    elif count >= good_matches:
        sample_score = 70.0 + 30.0 * (count - good_matches) / max(high_matches - good_matches, 1)
    else:
        sample_score = 70.0 * count / max(good_matches, 1)

    similarity_floor = float(config["minimum_similarity"])
    strong_similarity = float(config["strong_similarity"])
    similarity_score = 100.0 * _clamp(
        (average_similarity - similarity_floor) / max(strong_similarity - similarity_floor, 1e-6),
        0.0,
        1.0,
    )

    edge = float(direction.get("direction_edge") or 0.0)
    edge_score = 100.0 * _clamp(edge / 0.45, 0.0, 1.0)

    regime_multiplier = {
        "STRICT": 1.0,
        "RELAXED": 0.86,
        "UNFILTERED_FALLBACK": 0.66,
    }.get(regime_filter_mode, 0.55)

    raw_score = (0.38 * sample_score + 0.37 * similarity_score + 0.25 * edge_score) * regime_multiplier
    score = int(round(_clamp(raw_score, 0.0, 100.0)))
    label = _confidence_label(score, count, minimum_matches)
    return score, label


def _representative_matches(matches: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for match in list(matches)[:max(0, count)]:
        outcome = match["outcome"]
        output.append(
            {
                "match_id": f"{match['timeframe']}_{int(match['end_index'])}",
                "timeframe": match["timeframe"],
                "pattern_end_time": match["pattern_end_time"],
                "similarity": round(float(match["similarity"]), 4),
                "regime_compatibility": round(float(match["regime_score"]), 4),
                "regime": match["regime"],
                "realized_direction": outcome["realized_direction"],
                "up_mfe_atr": round(float(outcome["up_mfe_atr"]), 4),
                "down_mfe_atr": round(float(outcome["down_mfe_atr"]), 4),
                "terminal_return_atr": round(float(outcome["terminal_return_atr"]), 4),
                "first_touch_candle": outcome["first_touch_candle"],
            }
        )
    return output


def analyze_timeframe_analogs(
    *,
    df: Optional[pd.DataFrame],
    timeframe: str,
    role: str,
    config: Optional[Mapping[str, Any]] = None,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """Analyze current pattern versus historical patterns in one timeframe."""
    cfg = _deep_merge(DEFAULT_CONFIG, config)
    timeframe = _normalize_timeframe(timeframe)
    role = str(role or "main").lower().strip()

    prepared = _prepare_feature_frame(
        _normalize_ohlcv(df),
        max_clip=float(cfg["max_feature_clip"]),
    )

    role_windows = cfg["role_windows"]
    timeframe_horizons = cfg["timeframe_horizons"]
    window_size = int(role_windows.get(role, role_windows.get("main", 72)))
    horizon = int(timeframe_horizons.get(timeframe, 18))
    candles_available = len(prepared)

    minimum_required = window_size * 2 + horizon + 20
    if candles_available < minimum_required:
        return _empty_timeframe_result(
            timeframe=timeframe,
            role=role,
            reason=(
                f"Need at least {minimum_required} closed candles for a {window_size}-candle "
                f"pattern and {horizon}-candle forward test; received {candles_available}."
            ),
            candles_available=candles_available,
            window_size=window_size,
            horizon=horizon,
        )

    current_window = prepared.tail(window_size).copy()
    current_vector = _pattern_vector(current_window, int(cfg["resample_points"]))
    current_regime = _regime_signature(current_window)
    current_atr = _safe_float(current_window["ATR"].iloc[-1])
    current_close = _safe_float(current_window["Close"].iloc[-1])

    stride = max(
        int(cfg["minimum_stride"]),
        int(round(window_size * float(cfg["stride_fraction"]))),
    )
    minimum_separation = max(
        horizon,
        int(round(window_size * float(cfg["minimum_match_separation_fraction"]))),
    )

    candidate_rows: List[Dict[str, Any]] = []
    maximum_history_end = candles_available - window_size - horizon - 1

    for end_index in _candidate_end_indices(
        length=candles_available,
        window_size=window_size,
        horizon=horizon,
        stride=stride,
    ):
        start_index = end_index - window_size + 1
        candidate_window = prepared.iloc[start_index:end_index + 1]
        if len(candidate_window) != window_size:
            continue

        candidate_regime = _regime_signature(candidate_window)
        strict, relaxed, regime_score = _regime_compatibility(current_regime, candidate_regime)
        candidate_vector = _pattern_vector(candidate_window, int(cfg["resample_points"]))
        similarity = _cosine_similarity(current_vector, candidate_vector)

        outcome = _calculate_forward_outcome(prepared, end_index, horizon, cfg)
        if outcome is None:
            continue

        candidate_rows.append(
            {
                "timeframe": timeframe,
                "role": role,
                "start_index": int(start_index),
                "end_index": int(end_index),
                "pattern_end_time": _iso_timestamp(prepared.iloc[end_index]["Date"]),
                "similarity": float(similarity),
                "strict_regime": bool(strict),
                "relaxed_regime": bool(relaxed),
                "regime_score": float(regime_score),
                "regime": candidate_regime.to_dict(),
                "outcome": outcome,
            }
        )

    if not candidate_rows:
        return _empty_timeframe_result(
            timeframe=timeframe,
            role=role,
            reason="No valid historical candidate windows were available.",
            candles_available=candles_available,
            window_size=window_size,
            horizon=horizon,
        )

    minimum_similarity = float(cfg["minimum_similarity"])
    top_k = int(cfg["top_k"])
    minimum_matches = int(cfg["minimum_matches"])

    strict_candidates = [row for row in candidate_rows if row["strict_regime"]]
    selected = _select_independent_matches(
        strict_candidates,
        top_k=top_k,
        minimum_separation=minimum_separation,
        minimum_similarity=minimum_similarity,
    )
    regime_filter_mode = "STRICT"

    if len(selected) < minimum_matches and bool(cfg["allow_relaxed_regime_fallback"]):
        relaxed_candidates = [row for row in candidate_rows if row["relaxed_regime"]]
        selected = _select_independent_matches(
            relaxed_candidates,
            top_k=top_k,
            minimum_separation=minimum_separation,
            minimum_similarity=minimum_similarity,
        )
        regime_filter_mode = "RELAXED"

    if len(selected) < minimum_matches:
        selected = _select_independent_matches(
            candidate_rows,
            top_k=top_k,
            minimum_separation=minimum_separation,
            minimum_similarity=minimum_similarity,
        )
        regime_filter_mode = "UNFILTERED_FALLBACK"

    for match in selected:
        match["weight"] = _match_weight(
            similarity=float(match["similarity"]),
            end_index=int(match["end_index"]),
            maximum_history_end=max(maximum_history_end, 1),
            regime_score=float(match["regime_score"]),
            config=cfg,
        )

    direction = _direction_probabilities(selected)
    likely_direction = str(direction["likely_direction"])
    statistical_side = likely_direction if likely_direction in {"LONG", "SHORT"} else (
        "LONG" if float(direction["long_probability"]) >= float(direction["short_probability"]) else "SHORT"
    )

    expected_distance, expected_adverse, expected_path, timing = _aggregate_side_statistics(
        selected,
        statistical_side,
        cfg,
    )

    # Convert normalized ATR statistics into current-price distances for Gemini.
    if current_atr is not None and current_atr > 0 and current_close is not None:
        median_move_atr = _safe_float(expected_distance.get("median_favorable_move_atr"))
        weak_move_atr = _safe_float(expected_distance.get("weak_case_p25_favorable_move_atr"))
        strong_move_atr = _safe_float(expected_distance.get("strong_case_p75_favorable_move_atr"))
        median_adverse_atr = _safe_float(expected_adverse.get("median_adverse_move_atr"))
        p75_adverse_atr = _safe_float(expected_adverse.get("p75_adverse_move_atr"))

        direction_sign = 1.0 if statistical_side == "LONG" else -1.0
        if median_move_atr is not None:
            expected_distance["median_favorable_move_price_distance"] = round(median_move_atr * current_atr, 8)
            expected_distance["median_projected_price"] = round(current_close + direction_sign * median_move_atr * current_atr, 8)
        if weak_move_atr is not None:
            expected_distance["weak_case_price_distance"] = round(weak_move_atr * current_atr, 8)
            expected_distance["weak_case_projected_price"] = round(current_close + direction_sign * weak_move_atr * current_atr, 8)
        if strong_move_atr is not None:
            expected_distance["strong_case_price_distance"] = round(strong_move_atr * current_atr, 8)
            expected_distance["strong_case_projected_price"] = round(current_close + direction_sign * strong_move_atr * current_atr, 8)
        if median_adverse_atr is not None:
            expected_adverse["median_adverse_move_price_distance"] = round(median_adverse_atr * current_atr, 8)
            expected_adverse["median_adverse_projected_price"] = round(current_close - direction_sign * median_adverse_atr * current_atr, 8)
        if p75_adverse_atr is not None:
            expected_adverse["p75_adverse_move_price_distance"] = round(p75_adverse_atr * current_atr, 8)
            expected_adverse["p75_adverse_projected_price"] = round(current_close - direction_sign * p75_adverse_atr * current_atr, 8)

    timeframe_minutes = _timeframe_minutes(timeframe)
    if timeframe_minutes:
        for source_key, output_key in [
            ("median_candles_to_move_start", "median_time_to_move_start_hours"),
            ("median_candles_to_1atr", "median_time_to_1atr_hours"),
            ("median_candles_to_1_5atr", "median_time_to_1_5atr_hours"),
            ("p75_candles_to_move_start", "p75_time_to_move_start_hours"),
        ]:
            candle_count = _safe_float(timing.get(source_key))
            if candle_count is not None:
                timing[output_key] = round(candle_count * timeframe_minutes / 60.0, 2)

    weights = [float(match["weight"]) for match in selected]
    path_rows = []
    for match in selected:
        favorable_each, adverse_each = _side_series(match, statistical_side)
        path_rows.append(_path_flags(favorable_each, adverse_each, statistical_side, cfg))

    risk_forecast = {
        "side": statistical_side,
        "fake_breakout_probability": _round_or_none(
            _weighted_rate([row["fake_breakout"] for row in path_rows], weights),
            4,
        ),
        "immediate_reversal_probability": _round_or_none(
            _weighted_rate([row["immediate_reversal"] for row in path_rows], weights),
            4,
        ),
        "deep_retest_probability": _round_or_none(
            _weighted_rate([row["deep_retest_before_move"] for row in path_rows], weights),
            4,
        ),
        "entry_touch_then_adverse_1atr_probability": _round_or_none(
            _weighted_rate(
                [
                    max(_side_series(match, statistical_side)[1]) >= 1.0
                    for match in selected
                ],
                weights,
            ),
            4,
        ),
    }

    confidence_score, confidence_label = _sample_confidence(
        matches=selected,
        direction=direction,
        regime_filter_mode=regime_filter_mode,
        config=cfg,
    )

    average_similarity = float(np.mean([float(match["similarity"]) for match in selected])) if selected else 0.0
    current_time = _iso_timestamp(current_window.iloc[-1]["Date"])

    result: Dict[str, Any] = {
        "enabled": len(selected) >= minimum_matches,
        "status": "OK" if len(selected) >= minimum_matches else "LOW_SAMPLE",
        "timeframe": timeframe,
        "role": role,
        "reason": (
            "Historical analog statistics calculated from independent candle windows."
            if selected
            else "No sufficiently similar historical windows were found."
        ),
        "candles_available": int(candles_available),
        "window_size": int(window_size),
        "forward_horizon_candles": int(horizon),
        "current_pattern_end_time": current_time,
        "current_close": _round_or_none(current_close, 8),
        "current_atr": _round_or_none(current_atr, 8),
        "current_regime": current_regime.to_dict(),
        "direction": direction,
        "expected_distance": expected_distance,
        "expected_adverse_move": expected_adverse,
        "expected_path": expected_path,
        "timing": timing,
        "risk_forecast": risk_forecast,
        "sample_quality": {
            "matches_found": int(len(selected)),
            "independent_matches": int(len(selected)),
            "candidate_windows_examined": int(len(candidate_rows)),
            "average_similarity": round(average_similarity, 4),
            "minimum_similarity_used": round(minimum_similarity, 4),
            "confidence_score": int(confidence_score),
            "confidence": confidence_label,
            "regime_filter_mode": regime_filter_mode,
            "stride_candles": int(stride),
            "minimum_match_separation_candles": int(minimum_separation),
        },
        "representative_matches": _representative_matches(
            selected,
            int(cfg["representative_matches"]),
        ),
    }

    if include_internal:
        result["_internal"] = {
            "selected_matches": selected,
            "statistical_side": statistical_side,
            "config": cfg,
        }

    return result


# ---------------------------------------------------------------------------
# Multi-timeframe combination
# ---------------------------------------------------------------------------


def _combined_confidence(
    timeframe_results: Mapping[str, Mapping[str, Any]],
    probabilities: Mapping[str, float],
    alignment: str,
) -> Tuple[int, str]:
    valid = [
        result
        for result in timeframe_results.values()
        if isinstance(result, Mapping) and result.get("sample_quality")
    ]
    if not valid:
        return 0, "INSUFFICIENT_HISTORY"

    scores = [float(result["sample_quality"].get("confidence_score") or 0.0) for result in valid]
    base = float(np.mean(scores)) if scores else 0.0
    direction_edge = abs(float(probabilities.get("long", 0.0)) - float(probabilities.get("short", 0.0)))
    edge_bonus = min(15.0, direction_edge * 35.0)
    alignment_adjustment = {
        "STRONG": 10.0,
        "MODERATE": 4.0,
        "MIXED": -8.0,
        "CONFLICT": -16.0,
        "INSUFFICIENT": -20.0,
    }.get(alignment, -5.0)

    score = int(round(_clamp(base + edge_bonus + alignment_adjustment, 0.0, 100.0)))
    if score >= 78:
        label = "HIGH"
    elif score >= 62:
        label = "MEDIUM_HIGH"
    elif score >= 46:
        label = "MEDIUM"
    elif score >= 30:
        label = "LOW"
    else:
        label = "VERY_LOW"
    return score, label


def _alignment_from_results(results: Mapping[str, Mapping[str, Any]]) -> str:
    important_roles = [role for role in ["main", "confirm", "lower"] if role in results]
    directions = []
    for role in important_roles:
        result = results[role]
        if not result.get("enabled"):
            continue
        direction = str(result.get("direction", {}).get("likely_direction") or "UNCLEAR")
        if direction in {"LONG", "SHORT"}:
            directions.append((role, direction))

    if len(directions) < 2:
        return "INSUFFICIENT"

    direction_values = [direction for _, direction in directions]
    if len(set(direction_values)) == 1:
        if {role for role, _ in directions}.issuperset({"main", "confirm"}):
            return "STRONG"
        return "MODERATE"

    main_direction = next((direction for role, direction in directions if role == "main"), None)
    confirm_direction = next((direction for role, direction in directions if role == "confirm"), None)
    if main_direction and confirm_direction and main_direction != confirm_direction:
        return "CONFLICT"
    return "MIXED"


def _clean_for_external(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if key != "_internal"}


def analyze_historical_analogs(
    *,
    selected_df: Optional[pd.DataFrame],
    selected_timeframe: str,
    lower_df: Optional[pd.DataFrame] = None,
    lower_timeframe: Optional[str] = None,
    confirm_df: Optional[pd.DataFrame] = None,
    confirm_timeframe: Optional[str] = None,
    swing_df: Optional[pd.DataFrame] = None,
    swing_timeframe: Optional[str] = None,
    macro_df: Optional[pd.DataFrame] = None,
    macro_timeframe: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    include_internal: bool = False,
) -> Dict[str, Any]:
    """
    Analyze the current multi-timeframe market against old patterns.

    The caller should pass the same already-fetched CLOSED candle dataframes used
    by the main analysis flow. No extra exchange request is made here.
    """
    cfg = _deep_merge(DEFAULT_CONFIG, config)

    inputs = {
        "lower": (lower_df, lower_timeframe),
        "main": (selected_df, selected_timeframe),
        "confirm": (confirm_df, confirm_timeframe),
        "swing": (swing_df, swing_timeframe),
        "macro": (macro_df, macro_timeframe),
    }

    timeframe_results: Dict[str, Dict[str, Any]] = {}
    for role, (frame, timeframe) in inputs.items():
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or not timeframe:
            continue
        timeframe_results[role] = analyze_timeframe_analogs(
            df=frame,
            timeframe=str(timeframe),
            role=role,
            config=cfg,
            include_internal=include_internal,
        )

    if "main" not in timeframe_results:
        return {
            "enabled": False,
            "status": "NO_MAIN_TIMEFRAME_DATA",
            "engine_version": ENGINE_VERSION,
            "selected_timeframe": _normalize_timeframe(selected_timeframe),
            "reason": "Main selected-timeframe candles were not available.",
            "combined_forecast": {},
            "timeframes": {},
        }

    role_weights = cfg["role_weights"]
    long_sum = 0.0
    short_sum = 0.0
    range_sum = 0.0
    weight_sum = 0.0

    for role, result in timeframe_results.items():
        sample = result.get("sample_quality", {})
        confidence_factor = _clamp(float(sample.get("confidence_score") or 0.0) / 100.0, 0.05, 1.0)
        enabled_factor = 1.0 if result.get("enabled") else 0.35
        role_weight = float(role_weights.get(role, 0.0))
        effective_weight = role_weight * confidence_factor * enabled_factor
        direction = result.get("direction", {})

        long_sum += effective_weight * float(direction.get("long_probability") or 0.0)
        short_sum += effective_weight * float(direction.get("short_probability") or 0.0)
        range_sum += effective_weight * float(direction.get("range_probability") or 0.0)
        weight_sum += effective_weight

    if weight_sum <= 0:
        combined_probabilities = {"long": 0.0, "short": 0.0, "range": 1.0}
    else:
        combined_probabilities = {
            "long": long_sum / weight_sum,
            "short": short_sum / weight_sum,
            "range": range_sum / weight_sum,
        }
        total = sum(combined_probabilities.values())
        if total > 0:
            combined_probabilities = {
                key: value / total for key, value in combined_probabilities.items()
            }

    if combined_probabilities["long"] >= max(combined_probabilities["short"], combined_probabilities["range"]):
        combined_direction = "LONG"
    elif combined_probabilities["short"] >= max(combined_probabilities["long"], combined_probabilities["range"]):
        combined_direction = "SHORT"
    else:
        combined_direction = "RANGE"

    if max(combined_probabilities["long"], combined_probabilities["short"]) < 0.45:
        combined_direction = "UNCLEAR"

    alignment = _alignment_from_results(timeframe_results)
    confidence_score, confidence_label = _combined_confidence(
        timeframe_results,
        combined_probabilities,
        alignment,
    )

    main_result = timeframe_results["main"]
    lower_result = timeframe_results.get("lower") or {}
    confirm_result = timeframe_results.get("confirm") or {}

    # Main TF owns distance. Lower TF is more relevant to entry path/trap/timing.
    expected_distance = main_result.get("expected_distance", {})
    expected_adverse = main_result.get("expected_adverse_move", {})
    expected_path = lower_result.get("expected_path") or main_result.get("expected_path", {})
    timing = lower_result.get("timing") or main_result.get("timing", {})

    main_risk = main_result.get("risk_forecast", {})
    lower_risk = lower_result.get("risk_forecast", {})
    risk_forecast = {
        "fake_breakout_probability": lower_risk.get("fake_breakout_probability", main_risk.get("fake_breakout_probability")),
        "immediate_reversal_probability": lower_risk.get("immediate_reversal_probability", main_risk.get("immediate_reversal_probability")),
        "deep_retest_probability": lower_risk.get("deep_retest_probability", main_risk.get("deep_retest_probability")),
        "entry_touch_then_adverse_1atr_probability": lower_risk.get(
            "entry_touch_then_adverse_1atr_probability",
            main_risk.get("entry_touch_then_adverse_1atr_probability"),
        ),
    }

    result: Dict[str, Any] = {
        "enabled": bool(main_result.get("enabled")),
        "status": "OK" if main_result.get("enabled") else "LOW_CONFIDENCE",
        "engine_version": ENGINE_VERSION,
        "selected_timeframe": _normalize_timeframe(selected_timeframe),
        "reason": (
            "Multi-timeframe historical analog forecast is available."
            if main_result.get("enabled")
            else "Main-timeframe analog sample is weak; treat this as low-weight evidence."
        ),
        "combined_forecast": {
            "likely_direction": combined_direction,
            "long_probability": round(float(combined_probabilities["long"]), 4),
            "short_probability": round(float(combined_probabilities["short"]), 4),
            "range_probability": round(float(combined_probabilities["range"]), 4),
            "multi_timeframe_alignment": alignment,
            "confidence_score": int(confidence_score),
            "confidence": confidence_label,
            "expected_distance": expected_distance,
            "expected_adverse_move": expected_adverse,
            "expected_path": expected_path,
            "timing": timing,
            "risk_forecast": risk_forecast,
            "main_timeframe_direction": main_result.get("direction", {}),
            "confirm_timeframe_direction": confirm_result.get("direction", {}),
        },
        "timeframes": {
            role: _clean_for_external(frame_result)
            for role, frame_result in timeframe_results.items()
        },
    }

    if include_internal:
        result["_internal"] = {
            "timeframes": {
                role: frame_result.get("_internal", {})
                for role, frame_result in timeframe_results.items()
            },
            "config": cfg,
        }

    return result


# ---------------------------------------------------------------------------
# Optional exact AI-trade validation using selected analog paths
# ---------------------------------------------------------------------------


def _trade_first_touch(
    *,
    favorable_each: Sequence[float],
    adverse_each: Sequence[float],
    stop_atr: float,
    target_atr: float,
) -> str:
    for favorable, adverse in zip(favorable_each, adverse_each):
        target_hit = float(favorable) >= target_atr
        stop_hit = float(adverse) >= stop_atr
        if target_hit and stop_hit:
            return "AMBIGUOUS"
        if stop_hit:
            return "SL"
        if target_hit:
            return "TP"
    return "NEITHER"


def validate_trade_with_historical_analogs(
    *,
    analog_result: Mapping[str, Any],
    side: str,
    entry: float,
    stop_loss: float,
    tp2: Optional[float] = None,
    tp1_r: float = 1.5,
) -> Dict[str, Any]:
    """
    Test an AI-generated trade against the MAIN timeframe analog paths.

    `analyze_historical_analogs(..., include_internal=True)` must have been used.
    TP1 is calculated as exact `tp1_r` from entry and stop-loss.
    """
    normalized_side = str(side or "").upper().strip()
    if normalized_side not in {"LONG", "SHORT"}:
        return {
            "enabled": False,
            "status": "INVALID_SIDE",
            "reason": "Side must be LONG or SHORT.",
        }

    entry_value = _safe_float(entry)
    stop_value = _safe_float(stop_loss)
    tp2_value = _safe_float(tp2)
    if entry_value is None or stop_value is None:
        return {
            "enabled": False,
            "status": "INVALID_PRICES",
            "reason": "Entry and stop-loss must be valid numeric prices.",
        }

    if normalized_side == "LONG" and stop_value >= entry_value:
        return {"enabled": False, "status": "INVALID_GEOMETRY", "reason": "LONG stop-loss must be below entry."}
    if normalized_side == "SHORT" and stop_value <= entry_value:
        return {"enabled": False, "status": "INVALID_GEOMETRY", "reason": "SHORT stop-loss must be above entry."}

    main_external = analog_result.get("timeframes", {}).get("main", {})
    current_atr = _safe_float(main_external.get("current_atr"))
    internal_matches = (
        analog_result.get("_internal", {})
        .get("timeframes", {})
        .get("main", {})
        .get("selected_matches", [])
    )

    if current_atr is None or current_atr <= 1e-12 or not internal_matches:
        return {
            "enabled": False,
            "status": "INTERNAL_ANALOG_PATHS_UNAVAILABLE",
            "reason": "Run analyze_historical_analogs with include_internal=True before trade validation.",
        }

    risk_distance = abs(entry_value - stop_value)
    risk_atr = risk_distance / current_atr
    tp1_distance = risk_distance * float(tp1_r)
    tp1_atr = tp1_distance / current_atr

    if normalized_side == "LONG":
        tp1_price = entry_value + tp1_distance
        tp2_atr = ((tp2_value - entry_value) / current_atr) if tp2_value is not None else None
    else:
        tp1_price = entry_value - tp1_distance
        tp2_atr = ((entry_value - tp2_value) / current_atr) if tp2_value is not None else None

    if tp2_atr is not None and tp2_atr <= tp1_atr:
        return {
            "enabled": False,
            "status": "INVALID_TP2",
            "reason": "TP2 must be beyond TP1 in the trade direction.",
            "calculated_tp1": round(tp1_price, 8),
        }

    tp1_results: List[str] = []
    tp2_results: List[str] = []
    weights: List[float] = []
    time_to_tp1: List[int] = []
    time_to_tp1_weights: List[float] = []

    for match in internal_matches:
        favorable_each, adverse_each = _side_series(match, normalized_side)
        weight = float(match.get("weight") or 1.0)
        result_tp1 = _trade_first_touch(
            favorable_each=favorable_each,
            adverse_each=adverse_each,
            stop_atr=risk_atr,
            target_atr=tp1_atr,
        )
        tp1_results.append(result_tp1)
        weights.append(weight)

        if result_tp1 == "TP":
            hit_time = _first_index_at_or_above(favorable_each, tp1_atr)
            if hit_time is not None:
                time_to_tp1.append(hit_time)
                time_to_tp1_weights.append(weight)

        if tp2_atr is not None:
            tp2_results.append(
                _trade_first_touch(
                    favorable_each=favorable_each,
                    adverse_each=adverse_each,
                    stop_atr=risk_atr,
                    target_atr=tp2_atr,
                )
            )

    def result_rate(results: Sequence[str], wanted: str) -> float:
        return float(_weighted_rate([value == wanted for value in results], weights) or 0.0)

    output: Dict[str, Any] = {
        "enabled": True,
        "status": "OK",
        "side": normalized_side,
        "entry": round(entry_value, 8),
        "stop_loss": round(stop_value, 8),
        "calculated_tp1": round(tp1_price, 8),
        "tp1_r": round(float(tp1_r), 4),
        "risk_distance_atr": round(risk_atr, 4),
        "tp1_distance_atr": round(tp1_atr, 4),
        "sample_size": len(internal_matches),
        "tp1_before_sl_probability": round(result_rate(tp1_results, "TP"), 4),
        "sl_before_tp1_probability": round(result_rate(tp1_results, "SL"), 4),
        "ambiguous_tp1_sl_probability": round(result_rate(tp1_results, "AMBIGUOUS"), 4),
        "neither_tp1_nor_sl_probability": round(result_rate(tp1_results, "NEITHER"), 4),
        "median_candles_to_tp1": _round_or_none(
            _weighted_quantile(time_to_tp1, time_to_tp1_weights, 0.50),
            2,
        ),
    }

    if tp2_atr is not None and tp2_value is not None:
        output.update(
            {
                "tp2": round(tp2_value, 8),
                "tp2_distance_atr": round(tp2_atr, 4),
                "tp2_before_sl_probability": round(result_rate(tp2_results, "TP"), 4),
                "sl_before_tp2_probability": round(result_rate(tp2_results, "SL"), 4),
                "ambiguous_tp2_sl_probability": round(result_rate(tp2_results, "AMBIGUOUS"), 4),
                "neither_tp2_nor_sl_probability": round(result_rate(tp2_results, "NEITHER"), 4),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Compact payload for Gemini / logs
# ---------------------------------------------------------------------------


def build_gemini_analog_payload(result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return only the compact evidence Gemini needs; internal arrays are removed."""
    if not isinstance(result, Mapping):
        return {
            "available": False,
            "confidence": "UNAVAILABLE",
            "instruction": "Historical analog evidence is unavailable; do not invent it.",
        }

    combined = result.get("combined_forecast", {})
    timeframes = result.get("timeframes", {})

    compact_timeframes: Dict[str, Any] = {}
    for role in ["lower", "main", "confirm", "swing", "macro"]:
        frame = timeframes.get(role)
        if not isinstance(frame, Mapping):
            continue
        compact_timeframes[role] = {
            "timeframe": frame.get("timeframe"),
            "role": role,
            "direction": frame.get("direction"),
            "expected_distance": frame.get("expected_distance"),
            "expected_adverse_move": frame.get("expected_adverse_move"),
            "expected_path": frame.get("expected_path"),
            "timing": frame.get("timing"),
            "risk_forecast": frame.get("risk_forecast"),
            "sample_quality": frame.get("sample_quality"),
            "representative_matches": frame.get("representative_matches", [])[:3],
        }

    return {
        "available": bool(result.get("enabled")),
        "engine_version": result.get("engine_version"),
        "selected_timeframe": result.get("selected_timeframe"),
        "combined_forecast": combined,
        "timeframes": compact_timeframes,
        "instruction": (
            "Treat historical analog results as statistical evidence, not certainty. "
            "Combine them with current candles, Elliott, SMC, trend, SR, Fibonacci, "
            "liquidity, imbalance, volume and volatility. If sample confidence is low, "
            "give analog evidence low weight and do not manufacture probabilities."
        ),
    }