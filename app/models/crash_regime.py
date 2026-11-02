# app/models/crash_regime.py

"""
Crash / High-Momentum Regime Detector

Purpose:
- Detect fast bearish crash / bullish squeeze pressure after or during signal generation.
- Shadow/report-only module.
- Does NOT block signals.
- Does NOT change entry / SL / TP.
- Does NOT modify Firestore locks.
- Does NOT fetch candles by itself.

Expected use:
- User selected 15m -> selected_df only
- User selected 1h  -> selected_df + 15m fast_df
- User selected 4h  -> selected_df + 15m fast_df

The caller/main.py should pass already-fetched candles.
"""

import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ==========================================================
# Basic helpers
# ==========================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").replace("$", "").strip()
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

        if pd.isna(val):
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


def _mean(values) -> Optional[float]:
    try:
        vals = [
            _safe_float(v)
            for v in values
            if _safe_float(v) is not None
        ]
        if not vals:
            return None
        return float(sum(vals) / len(vals))
    except Exception:
        return None


def _pct(value: float, base: float) -> float:
    try:
        if base == 0:
            return 0.0
        return float((value / base) * 100.0)
    except Exception:
        return 0.0


def _body_size(row) -> float:
    try:
        return abs(float(row["Close"]) - float(row["Open"]))
    except Exception:
        return 0.0


def _range_size(row) -> float:
    try:
        return max(float(row["High"]) - float(row["Low"]), 0.0)
    except Exception:
        return 0.0


def _close_location(row) -> float:
    """
    Returns where close is inside candle range:
    0.0 = near low
    1.0 = near high
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


def _is_bearish(row) -> bool:
    try:
        return float(row["Close"]) < float(row["Open"])
    except Exception:
        return False


def _is_bullish(row) -> bool:
    try:
        return float(row["Close"]) > float(row["Open"])
    except Exception:
        return False


def _safe_tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    try:
        d = _normalize_ohlc(df)
        if d.empty:
            return d
        return d.tail(n).copy()
    except Exception:
        return pd.DataFrame()


# ==========================================================
# Timeframe analysis
# ==========================================================

def _support_resistance_breaks(df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
    """
    Detect simple recent structure break using previous lookback candles.

    Important:
    - This is not full S/R detection.
    - This is only a crash/squeeze regime signal.
    """
    out = {
        "support_break": False,
        "resistance_break": False,
        "broken_support": None,
        "broken_resistance": None,
        "reason": "",
    }

    try:
        d = _normalize_ohlc(df)

        if len(d) < lookback + 2:
            out["reason"] = "Not enough candles for structure break detection."
            return out

        prev = d.iloc[-(lookback + 1):-1]
        last = d.iloc[-1]

        prev_low = _safe_float(prev["Low"].min())
        prev_high = _safe_float(prev["High"].max())
        close = _safe_float(last["Close"])

        if prev_low is not None and close is not None and close < prev_low:
            out["support_break"] = True
            out["broken_support"] = float(prev_low)

        if prev_high is not None and close is not None and close > prev_high:
            out["resistance_break"] = True
            out["broken_resistance"] = float(prev_high)

        if out["support_break"]:
            out["reason"] = "Closed below recent support range."
        elif out["resistance_break"]:
            out["reason"] = "Closed above recent resistance range."
        else:
            out["reason"] = "No fresh structure break."

        return out

    except Exception as exc:
        out["reason"] = f"Structure break detection failed: {exc}"
        return out


def _volume_spike(df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
    out = {
        "available": False,
        "volume_spike": False,
        "ratio": 0.0,
        "last_volume": None,
        "avg_volume": None,
    }

    try:
        d = _normalize_ohlc(df)

        if d.empty or "Volume" not in d.columns or len(d) < lookback + 1:
            return out

        vols = pd.to_numeric(d["Volume"], errors="coerce").dropna()
        if len(vols) < lookback + 1:
            return out

        last_vol = _safe_float(vols.iloc[-1])
        avg_vol = _safe_float(vols.iloc[-(lookback + 1):-1].mean())

        if last_vol is None or avg_vol is None or avg_vol <= 0:
            return out

        ratio = float(last_vol / avg_vol)

        out.update({
            "available": True,
            "volume_spike": bool(ratio >= 1.6),
            "ratio": round(ratio, 2),
            "last_volume": float(last_vol),
            "avg_volume": float(avg_vol),
        })

        return out

    except Exception:
        return out


def _streak_info(df: pd.DataFrame, n: int = 4) -> Dict[str, Any]:
    out = {
        "bearish_streak": 0,
        "bullish_streak": 0,
        "last_n_bearish": 0,
        "last_n_bullish": 0,
    }

    try:
        d = _safe_tail(df, n)
        if d.empty:
            return out

        bearish_count = 0
        bullish_count = 0

        for _, row in d.iterrows():
            if _is_bearish(row):
                bearish_count += 1
            elif _is_bullish(row):
                bullish_count += 1

        # Consecutive streak from latest candle backwards.
        bear_streak = 0
        bull_streak = 0

        for _, row in d.iloc[::-1].iterrows():
            if _is_bearish(row):
                if bull_streak == 0:
                    bear_streak += 1
                else:
                    break
            elif _is_bullish(row):
                if bear_streak == 0:
                    bull_streak += 1
                else:
                    break
            else:
                break

        out.update({
            "bearish_streak": int(bear_streak),
            "bullish_streak": int(bull_streak),
            "last_n_bearish": int(bearish_count),
            "last_n_bullish": int(bullish_count),
        })

        return out

    except Exception:
        return out


def _analyze_single_timeframe(
    df: Optional[pd.DataFrame],
    *,
    timeframe: str = "",
    label: str = "selected",
) -> Dict[str, Any]:
    """
    Analyze one timeframe and return bullish/bearish crash pressure score.

    Score meaning:
    0-29  = calm/neutral
    30-49 = pressure
    50-69 = strong pressure
    70+   = crash/squeeze regime
    """
    result = {
        "enabled": True,
        "label": label,
        "timeframe": str(timeframe or ""),
        "direction": "NONE",
        "signal": "NEUTRAL",
        "score": 0,
        "bearish_score": 0,
        "bullish_score": 0,
        "atr": None,
        "last_close": None,
        "move_atr_5": 0.0,
        "move_pct_5": 0.0,
        "large_bearish_displacement": False,
        "large_bullish_displacement": False,
        "multi_candle_bearish_pressure": False,
        "multi_candle_bullish_pressure": False,
        "support_break": False,
        "resistance_break": False,
        "broken_support": None,
        "broken_resistance": None,
        "volume_spike": False,
        "volume_ratio": 0.0,
        "close_near_low": False,
        "close_near_high": False,
        "ema50": None,
        "ema200": None,
        "below_ema50": False,
        "above_ema50": False,
        "below_ema200": False,
        "above_ema200": False,
        "reasons": [],
        "error": "",
    }

    try:
        d = _normalize_ohlc(df)

        if d.empty or len(d) < 30:
            result["enabled"] = False
            result["error"] = "Not enough candles for crash regime detection."
            return result

        d = d.tail(250).copy()

        last = d.iloc[-1]
        close = _safe_float(last["Close"])
        open_ = _safe_float(last["Open"])
        high = _safe_float(last["High"])
        low = _safe_float(last["Low"])

        atr_val = _atr(d, period=14)
        ema50 = _ema(d["Close"], 50)
        ema200 = _ema(d["Close"], 200)

        if close is None or open_ is None or high is None or low is None:
            result["enabled"] = False
            result["error"] = "Invalid latest candle."
            return result

        result["last_close"] = float(close)
        result["atr"] = float(atr_val) if atr_val else None
        result["ema50"] = float(ema50) if ema50 is not None else None
        result["ema200"] = float(ema200) if ema200 is not None else None

        bearish_score = 0
        bullish_score = 0
        reasons = []

        # --------------------------------------------------
        # 1) Last candle displacement vs ATR
        # --------------------------------------------------
        body = _body_size(last)
        rng = _range_size(last)
        close_loc = _close_location(last)

        body_atr = 0.0
        range_atr = 0.0

        if atr_val and atr_val > 0:
            body_atr = body / atr_val
            range_atr = rng / atr_val

        last_bearish = _is_bearish(last)
        last_bullish = _is_bullish(last)

        close_near_low = close_loc <= 0.25
        close_near_high = close_loc >= 0.75

        result["close_near_low"] = bool(close_near_low)
        result["close_near_high"] = bool(close_near_high)

        if last_bearish and body_atr >= 1.15 and close_near_low:
            bearish_score += 26
            result["large_bearish_displacement"] = True
            reasons.append(f"{label}: large bearish displacement candle ({body_atr:.2f} ATR body).")

        elif last_bearish and range_atr >= 1.40 and close_near_low:
            bearish_score += 18
            reasons.append(f"{label}: bearish wide-range candle closed near low ({range_atr:.2f} ATR range).")

        if last_bullish and body_atr >= 1.15 and close_near_high:
            bullish_score += 26
            result["large_bullish_displacement"] = True
            reasons.append(f"{label}: large bullish displacement candle ({body_atr:.2f} ATR body).")

        elif last_bullish and range_atr >= 1.40 and close_near_high:
            bullish_score += 18
            reasons.append(f"{label}: bullish wide-range candle closed near high ({range_atr:.2f} ATR range).")

        # --------------------------------------------------
        # 2) Last 5 candle move vs ATR
        # --------------------------------------------------
        if len(d) >= 6:
            prev_close = _safe_float(d["Close"].iloc[-6])
            if prev_close is not None:
                move = close - prev_close
                move_pct = _pct(move, prev_close)
                move_atr = abs(move) / atr_val if atr_val and atr_val > 0 else 0.0

                result["move_atr_5"] = round(float(move_atr), 2)
                result["move_pct_5"] = round(float(move_pct), 2)

                if move < 0 and move_atr >= 2.0:
                    bearish_score += 22
                    reasons.append(f"{label}: strong 5-candle bearish move ({move_atr:.2f} ATR).")

                elif move < 0 and move_atr >= 1.25:
                    bearish_score += 12
                    reasons.append(f"{label}: bearish 5-candle pressure ({move_atr:.2f} ATR).")

                if move > 0 and move_atr >= 2.0:
                    bullish_score += 22
                    reasons.append(f"{label}: strong 5-candle bullish move ({move_atr:.2f} ATR).")

                elif move > 0 and move_atr >= 1.25:
                    bullish_score += 12
                    reasons.append(f"{label}: bullish 5-candle pressure ({move_atr:.2f} ATR).")

        # --------------------------------------------------
        # 3) Volume spike
        # --------------------------------------------------
        vol = _volume_spike(d, lookback=30)
        result["volume_spike"] = bool(vol.get("volume_spike"))
        result["volume_ratio"] = float(vol.get("ratio") or 0.0)

        if vol.get("volume_spike"):
            if last_bearish:
                bearish_score += 14
                reasons.append(f"{label}: bearish candle with volume spike ({vol.get('ratio')}x).")
            elif last_bullish:
                bullish_score += 14
                reasons.append(f"{label}: bullish candle with volume spike ({vol.get('ratio')}x).")
            else:
                # Neutral candle volume spike adds only small directional pressure
                # if the close is near one side of range.
                if close_near_low:
                    bearish_score += 6
                    reasons.append(f"{label}: volume spike with close near low.")
                elif close_near_high:
                    bullish_score += 6
                    reasons.append(f"{label}: volume spike with close near high.")

        # --------------------------------------------------
        # 4) Structure break
        # --------------------------------------------------
        breaks = _support_resistance_breaks(d, lookback=30)
        result["support_break"] = bool(breaks.get("support_break"))
        result["resistance_break"] = bool(breaks.get("resistance_break"))
        result["broken_support"] = breaks.get("broken_support")
        result["broken_resistance"] = breaks.get("broken_resistance")

        if result["support_break"]:
            bearish_score += 22
            reasons.append(f"{label}: closed below recent support range.")

        if result["resistance_break"]:
            bullish_score += 22
            reasons.append(f"{label}: closed above recent resistance range.")

        # --------------------------------------------------
        # 5) EMA regime pressure
        # --------------------------------------------------
        if ema50 is not None:
            result["below_ema50"] = bool(close < ema50)
            result["above_ema50"] = bool(close > ema50)

            if close < ema50 and last_bearish:
                bearish_score += 6
                reasons.append(f"{label}: bearish close below EMA50.")
            elif close > ema50 and last_bullish:
                bullish_score += 6
                reasons.append(f"{label}: bullish close above EMA50.")

        if ema200 is not None:
            result["below_ema200"] = bool(close < ema200)
            result["above_ema200"] = bool(close > ema200)

            if close < ema200 and last_bearish:
                bearish_score += 6
                reasons.append(f"{label}: bearish close below EMA200.")
            elif close > ema200 and last_bullish:
                bullish_score += 6
                reasons.append(f"{label}: bullish close above EMA200.")

        # --------------------------------------------------
        # 6) Multi-candle pressure / streak
        # --------------------------------------------------
        streak = _streak_info(d, n=4)

        if streak.get("last_n_bearish", 0) >= 3:
            bearish_score += 10
            result["multi_candle_bearish_pressure"] = True
            reasons.append(f"{label}: at least 3 of last 4 candles are bearish.")

        if streak.get("last_n_bullish", 0) >= 3:
            bullish_score += 10
            result["multi_candle_bullish_pressure"] = True
            reasons.append(f"{label}: at least 3 of last 4 candles are bullish.")

        if streak.get("bearish_streak", 0) >= 3:
            bearish_score += 6
            reasons.append(f"{label}: bearish candle streak is active.")

        if streak.get("bullish_streak", 0) >= 3:
            bullish_score += 6
            reasons.append(f"{label}: bullish candle streak is active.")

        # --------------------------------------------------
        # Normalize direction
        # --------------------------------------------------
        bearish_score = int(max(0, min(100, bearish_score)))
        bullish_score = int(max(0, min(100, bullish_score)))

        if bearish_score >= bullish_score + 8:
            direction = "SHORT"
            score = bearish_score
        elif bullish_score >= bearish_score + 8:
            direction = "LONG"
            score = bullish_score
        else:
            direction = "NONE"
            score = max(bearish_score, bullish_score)

        if direction == "SHORT":
            if score >= 70:
                signal = "BEARISH_CRASH"
            elif score >= 50:
                signal = "BEARISH_STRONG_PRESSURE"
            elif score >= 35:
                signal = "BEARISH_PRESSURE"
            else:
                signal = "NEUTRAL"
        elif direction == "LONG":
            if score >= 70:
                signal = "BULLISH_SQUEEZE"
            elif score >= 50:
                signal = "BULLISH_STRONG_PRESSURE"
            elif score >= 35:
                signal = "BULLISH_PRESSURE"
            else:
                signal = "NEUTRAL"
        else:
            signal = "NEUTRAL"

        result.update({
            "direction": direction,
            "signal": signal,
            "score": int(score),
            "bearish_score": bearish_score,
            "bullish_score": bullish_score,
            "reasons": reasons[:12],
        })

        return result

    except Exception as exc:
        result["enabled"] = False
        result["error"] = f"Crash timeframe analysis failed: {exc}"
        return result


# ==========================================================
# Combined regime
# ==========================================================

def _combine_timeframe_results(
    selected_result: Dict[str, Any],
    fast_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        selected_result = selected_result or {}
        fast_result = fast_result if isinstance(fast_result, dict) else None

        selected_enabled = bool(selected_result.get("enabled"))
        fast_enabled = bool(fast_result and fast_result.get("enabled"))

        if not selected_enabled and not fast_enabled:
            return {
                "direction": "NONE",
                "score": 0,
                "regime": "UNKNOWN",
                "agreement": "NONE",
                "reason": "No valid timeframe data for crash regime.",
            }

        selected_dir = _upper(selected_result.get("direction") or "NONE")
        selected_score = int(_safe_float(selected_result.get("score"), 0) or 0)

        if not fast_enabled:
            direction = selected_dir if selected_dir in ["LONG", "SHORT"] else "NONE"
            score = selected_score

            return {
                "direction": direction,
                "score": score,
                "regime": _regime_name(direction, score),
                "agreement": "SELECTED_ONLY",
                "reason": "Crash regime based on selected timeframe only.",
            }

        fast_dir = _upper(fast_result.get("direction") or "NONE")
        fast_score = int(_safe_float(fast_result.get("score"), 0) or 0)

        # Same direction: combine strongly.
        if selected_dir == fast_dir and selected_dir in ["LONG", "SHORT"]:
            score = int(round((selected_score * 0.55) + (fast_score * 0.45)))
            score = min(100, score + 6)  # small agreement bonus

            return {
                "direction": selected_dir,
                "score": score,
                "regime": _regime_name(selected_dir, score),
                "agreement": "ALIGNED",
                "reason": "Selected timeframe and fast timeframe agree.",
            }

        # Fast TF detects strong crash before selected TF confirms.
        # This is useful for post-lock warning.
        if fast_dir in ["LONG", "SHORT"] and fast_score >= 70 and selected_dir not in ["LONG", "SHORT"]:
            score = int(round((fast_score * 0.70) + (selected_score * 0.30)))

            return {
                "direction": fast_dir,
                "score": score,
                "regime": _regime_name(fast_dir, score),
                "agreement": "FAST_LEADS",
                "reason": "Fast timeframe shows strong regime pressure before selected timeframe confirms.",
            }

        # Selected TF strong, fast TF neutral.
        if selected_dir in ["LONG", "SHORT"] and selected_score >= 65 and fast_dir not in ["LONG", "SHORT"]:
            score = int(round((selected_score * 0.70) + (fast_score * 0.30)))

            return {
                "direction": selected_dir,
                "score": score,
                "regime": _regime_name(selected_dir, score),
                "agreement": "SELECTED_DOMINANT",
                "reason": "Selected timeframe shows stronger regime pressure than fast timeframe.",
            }

        # Conflict: lower confidence, warning only.
        if selected_dir in ["LONG", "SHORT"] and fast_dir in ["LONG", "SHORT"] and selected_dir != fast_dir:
            stronger_dir = selected_dir if selected_score >= fast_score else fast_dir
            score = int(round(abs(selected_score - fast_score) * 0.75))

            return {
                "direction": stronger_dir if score >= 35 else "NONE",
                "score": score,
                "regime": "CONFLICTING_CRASH_PRESSURE" if score >= 35 else "NEUTRAL",
                "agreement": "CONFLICT",
                "reason": "Selected timeframe and fast timeframe conflict.",
            }

        # Weak/neutral.
        score = max(selected_score, fast_score)
        direction = selected_dir if selected_score >= fast_score else fast_dir
        if score < 35:
            direction = "NONE"

        return {
            "direction": direction,
            "score": score,
            "regime": _regime_name(direction, score),
            "agreement": "WEAK_OR_NEUTRAL",
            "reason": "No strong aligned crash regime.",
        }

    except Exception as exc:
        return {
            "direction": "NONE",
            "score": 0,
            "regime": "UNKNOWN",
            "agreement": "ERROR",
            "reason": f"Crash regime combine failed: {exc}",
        }


def _regime_name(direction: str, score: int) -> str:
    direction = _upper(direction)
    score = int(score or 0)

    if direction == "SHORT":
        if score >= 70:
            return "BEARISH_CRASH"
        if score >= 50:
            return "BEARISH_STRONG_PRESSURE"
        if score >= 35:
            return "BEARISH_PRESSURE"
        return "NEUTRAL"

    if direction == "LONG":
        if score >= 70:
            return "BULLISH_SQUEEZE"
        if score >= 50:
            return "BULLISH_STRONG_PRESSURE"
        if score >= 35:
            return "BULLISH_PRESSURE"
        return "NEUTRAL"

    return "NEUTRAL"


def _trade_side_from_scenario(scenario: Optional[dict]) -> str:
    try:
        if not isinstance(scenario, dict):
            return "UNKNOWN"

        side = _upper(
            scenario.get("trade_type")
            or scenario.get("side")
            or ""
        )

        if "LONG" in side or side in ["BUY", "BULL", "BULLISH"]:
            return "LONG"

        if "SHORT" in side or side in ["SELL", "BEAR", "BEARISH"]:
            return "SHORT"

        return "UNKNOWN"

    except Exception:
        return "UNKNOWN"


def _scenario_status(scenario: Optional[dict]) -> str:
    try:
        if not isinstance(scenario, dict):
            return "UNKNOWN"

        return _upper(
            scenario.get("status")
            or scenario.get("trade_status")
            or scenario.get("execution_status")
            or ""
        )

    except Exception:
        return "UNKNOWN"


def _build_advice(
    *,
    direction: str,
    regime: str,
    score: int,
    trade_side: str,
    status: str,
    agreement: str,
) -> str:
    direction = _upper(direction)
    regime = _upper(regime)
    trade_side = _upper(trade_side)
    status = _upper(status)
    agreement = _upper(agreement)

    active = (
        "RUNNING" in status
        or "TP1" in status
        or "ACTIVE" in status
    )

    pending = (
        "LIMIT" in status
        or "PENDING" in status
        or "WAIT" in status
        or "READY" in status
        or not active
    )

    if direction == "SHORT" and score >= 70:
        if trade_side == "LONG":
            if active:
                return (
                    "Bearish crash pressure is active against this LONG trade. "
                    "Do not add more. Protect risk and watch for reclaim/stabilization. "
                    "This is a warning only, not an automatic exit."
                )
            return (
                "Bearish crash pressure is active. New/pending LONG entries are risky. "
                "Wait for stabilization or reclaim before trusting the old long idea."
            )

        if trade_side == "SHORT":
            if active:
                return (
                    "Bearish crash pressure favors the active SHORT direction. "
                    "Do not add more blindly; manage profit and watch TP/reversal zones."
                )
            return (
                "Bearish crash pressure may cause pullback SHORT limit entries to be missed. "
                "Do not chase. Watch breakdown/retest continuation instead."
            )

        return (
            "Bearish crash pressure detected. Avoid new longs and do not chase late shorts. "
            "Watch breakdown/retest continuation."
        )

    if direction == "LONG" and score >= 70:
        if trade_side == "SHORT":
            if active:
                return (
                    "Bullish squeeze pressure is active against this SHORT trade. "
                    "Do not add more. Protect risk and watch for rejection/stabilization. "
                    "This is a warning only, not an automatic exit."
                )
            return (
                "Bullish squeeze pressure is active. New/pending SHORT entries are risky. "
                "Wait for stabilization or rejection before trusting the old short idea."
            )

        if trade_side == "LONG":
            if active:
                return (
                    "Bullish squeeze pressure favors the active LONG direction. "
                    "Do not add more blindly; manage profit and watch TP/reversal zones."
                )
            return (
                "Bullish squeeze pressure may cause pullback LONG limit entries to be missed. "
                "Do not chase. Watch breakout/retest continuation instead."
            )

        return (
            "Bullish squeeze pressure detected. Avoid new shorts and do not chase late longs. "
            "Watch breakout/retest continuation."
        )

    if score >= 50:
        if direction == "SHORT":
            return (
                "Bearish high-momentum pressure is present, but crash confirmation is not full. "
                "Use caution with longs and avoid chasing shorts."
            )

        if direction == "LONG":
            return (
                "Bullish high-momentum pressure is present, but squeeze confirmation is not full. "
                "Use caution with shorts and avoid chasing longs."
            )

    if agreement == "CONFLICT":
        return (
            "Crash regime signals are mixed between timeframes. Treat this as caution only."
        )

    return "No strong crash/squeeze regime detected."


def _watch_levels(selected_result: Dict[str, Any], fast_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        out = {
            "broken_support": None,
            "broken_resistance": None,
            "source": "",
        }

        for src_name, res in [
            ("fast", fast_result),
            ("selected", selected_result),
        ]:
            if not isinstance(res, dict):
                continue

            if res.get("broken_support") is not None:
                out["broken_support"] = res.get("broken_support")
                out["source"] = src_name
                return out

            if res.get("broken_resistance") is not None:
                out["broken_resistance"] = res.get("broken_resistance")
                out["source"] = src_name
                return out

        return out

    except Exception:
        return {}


# ==========================================================
# Public API
# ==========================================================

def evaluate_crash_regime(
    *,
    selected_df: Optional[pd.DataFrame],
    selected_timeframe: str,
    fast_df: Optional[pd.DataFrame] = None,
    fast_timeframe: str = "15m",
    scenario: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate crash/squeeze regime in shadow mode.

    This function does not modify the scenario.
    """
    try:
        selected_tf = _tf_key(selected_timeframe)
        fast_tf = _tf_key(fast_timeframe)

        selected_result = _analyze_single_timeframe(
            selected_df,
            timeframe=selected_tf,
            label="selected_tf",
        )

        # 15m selected -> selected only.
        fast_result = None
        if fast_df is not None and not _tf_key(selected_tf) in ["15m"]:
            fast_result = _analyze_single_timeframe(
                fast_df,
                timeframe=fast_tf or "15m",
                label="fast_tf",
            )

        combined = _combine_timeframe_results(selected_result, fast_result)

        direction = _upper(combined.get("direction") or "NONE")
        score = int(_safe_float(combined.get("score"), 0) or 0)
        regime = _upper(combined.get("regime") or "NEUTRAL")
        agreement = _upper(combined.get("agreement") or "NONE")

        trade_side = _trade_side_from_scenario(scenario)
        status = _scenario_status(scenario)

        bearish_crash = direction == "SHORT" and score >= 70
        bullish_squeeze = direction == "LONG" and score >= 70

        bearish_pressure = direction == "SHORT" and score >= 50
        bullish_pressure = direction == "LONG" and score >= 50

        running_longs_need_protection = bool(
            bearish_pressure and trade_side == "LONG" and (
                "RUNNING" in status or "ACTIVE" in status or "TP1" in status
            )
        )

        running_shorts_need_protection = bool(
            bullish_pressure and trade_side == "SHORT" and (
                "RUNNING" in status or "ACTIVE" in status or "TP1" in status
            )
        )

        running_longs_favorable = bool(
            bullish_pressure and trade_side == "LONG" and (
                "RUNNING" in status or "ACTIVE" in status or "TP1" in status
            )
        )

        running_shorts_favorable = bool(
            bearish_pressure and trade_side == "SHORT" and (
                "RUNNING" in status or "ACTIVE" in status or "TP1" in status
            )
        )

        pending_longs_should_pause = bool(bearish_pressure and trade_side == "LONG")
        pending_shorts_should_pause = bool(bullish_pressure and trade_side == "SHORT")

        pending_longs_may_miss_limit = bool(bullish_pressure and trade_side == "LONG")
        pending_shorts_may_miss_limit = bool(bearish_pressure and trade_side == "SHORT")

        advice = _build_advice(
            direction=direction,
            regime=regime,
            score=score,
            trade_side=trade_side,
            status=status,
            agreement=agreement,
        )

        selected_signal = selected_result.get("signal") if isinstance(selected_result, dict) else "UNKNOWN"
        fast_signal = fast_result.get("signal") if isinstance(fast_result, dict) else "NOT_USED"

        return {
            "enabled": True,
            "mode": "SHADOW_WARNING_ONLY",
            "does_block_signal": False,
            "does_change_entry_sl_tp": False,

            "regime": regime,
            "direction": direction,
            "score": score,
            "agreement": agreement,
            "reason": combined.get("reason", ""),
            "advice": advice,

            "selected_timeframe": selected_tf,
            "fast_timeframe": fast_tf if fast_result else None,
            "selected_tf_signal": selected_signal,
            "fast_tf_signal": fast_signal,

            "bearish_crash": bearish_crash,
            "bullish_squeeze": bullish_squeeze,
            "bearish_pressure": bearish_pressure,
            "bullish_pressure": bullish_pressure,

            "new_longs_risky": bool(bearish_pressure),
            "new_shorts_risky": bool(bullish_pressure),

            "pending_longs_should_pause": pending_longs_should_pause,
            "pending_shorts_should_pause": pending_shorts_should_pause,
            "pending_longs_may_miss_limit": pending_longs_may_miss_limit,
            "pending_shorts_may_miss_limit": pending_shorts_may_miss_limit,

            "running_longs_need_protection": running_longs_need_protection,
            "running_shorts_need_protection": running_shorts_need_protection,
            "running_longs_favorable": running_longs_favorable,
            "running_shorts_favorable": running_shorts_favorable,

            "watch_levels": _watch_levels(selected_result, fast_result),

            "selected_tf_details": selected_result,
            "fast_tf_details": fast_result or {},
        }

    except Exception as exc:
        return {
            "enabled": True,
            "mode": "SHADOW_WARNING_ONLY",
            "does_block_signal": False,
            "does_change_entry_sl_tp": False,
            "regime": "UNKNOWN",
            "direction": "NONE",
            "score": 0,
            "agreement": "ERROR",
            "reason": f"Crash regime evaluation failed: {exc}",
            "advice": "Crash regime unavailable because evaluation failed.",
            "selected_timeframe": str(selected_timeframe or ""),
            "fast_timeframe": str(fast_timeframe or ""),
            "selected_tf_signal": "UNKNOWN",
            "fast_tf_signal": "UNKNOWN",
            "bearish_crash": False,
            "bullish_squeeze": False,
            "bearish_pressure": False,
            "bullish_pressure": False,
            "new_longs_risky": False,
            "new_shorts_risky": False,
            "pending_longs_should_pause": False,
            "pending_shorts_should_pause": False,
            "pending_longs_may_miss_limit": False,
            "pending_shorts_may_miss_limit": False,
            "running_longs_need_protection": False,
            "running_shorts_need_protection": False,
            "running_longs_favorable": False,
            "running_shorts_favorable": False,
            "watch_levels": {},
            "selected_tf_details": {},
            "fast_tf_details": {},
        }


def add_crash_regime_to_scenario(
    *,
    scenario: dict,
    interval: str,
    df_selected: Optional[pd.DataFrame],
    df_fast: Optional[pd.DataFrame] = None,
    fast_timeframe: str = "15m",
) -> dict:
    """
    Attach crash regime report to one scenario.

    Warning-only:
    - Does not block signal.
    - Does not change entry/SL/TP.
    - Does not change status.
    """
    try:
        if not isinstance(scenario, dict):
            return scenario

        scenario["crash_regime"] = evaluate_crash_regime(
            selected_df=df_selected,
            selected_timeframe=interval,
            fast_df=df_fast,
            fast_timeframe=fast_timeframe,
            scenario=scenario,
        )

        return scenario

    except Exception as exc:
        try:
            scenario["crash_regime"] = {
                "enabled": True,
                "mode": "SHADOW_WARNING_ONLY",
                "does_block_signal": False,
                "does_change_entry_sl_tp": False,
                "regime": "UNKNOWN",
                "direction": "NONE",
                "score": 0,
                "agreement": "ERROR",
                "reason": str(exc),
                "advice": "Crash regime unavailable because scenario attach failed.",
            }
        except Exception:
            pass

        return scenario


def add_crash_regime_to_ai_data(
    *,
    ai_data: dict,
    interval: str,
    df_selected: Optional[pd.DataFrame],
    df_fast: Optional[pd.DataFrame] = None,
    fast_timeframe: str = "15m",
) -> dict:
    """
    Attach crash regime to first trade scenario and selected_scenario.

    Warning-only:
    - Does not block signal.
    - Does not change entry/SL/TP.
    - Does not alter status/order type.
    """
    try:
        if not isinstance(ai_data, dict):
            return ai_data

        scenarios = ai_data.get("trade_scenarios") or []
        if not scenarios:
            return ai_data

        scenario = scenarios[0]
        if not isinstance(scenario, dict):
            return ai_data

        scenario = add_crash_regime_to_scenario(
            scenario=scenario,
            interval=interval,
            df_selected=df_selected,
            df_fast=df_fast,
            fast_timeframe=fast_timeframe,
        )

        scenarios[0] = scenario
        ai_data["trade_scenarios"] = scenarios

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            selected_side = _upper(
                selected.get("trade_type")
                or selected.get("side")
                or ""
            )

            scenario_side = _upper(
                scenario.get("trade_type")
                or scenario.get("side")
                or ""
            )

            selected_entry = _safe_float(
                selected.get("entry")
                or selected.get("entry_zone")
            )

            scenario_entry = _safe_float(
                scenario.get("entry")
                or scenario.get("entry_zone")
            )

            entry_matches = (
                selected_entry is None
                or scenario_entry is None
                or abs(selected_entry - scenario_entry) <= max(abs(scenario_entry) * 0.000001, 1e-12)
            )

            if selected_side == scenario_side and entry_matches:
                ai_data["selected_scenario"] = scenario

        return ai_data

    except Exception as exc:
        try:
            ai_data["crash_regime_attach_error"] = str(exc)
        except Exception:
            pass

        return ai_data


def format_crash_regime_text(crash_regime: Optional[dict]) -> str:
    """
    Optional formatter for CLI / signal.py output.
    """
    try:
        if not isinstance(crash_regime, dict) or not crash_regime:
            return ""

        lines = [
            "--- Crash Regime / Momentum Pressure ---",
            f"Mode: {crash_regime.get('mode', 'UNKNOWN')}",
            f"Regime: {crash_regime.get('regime', 'UNKNOWN')}",
            f"Direction: {crash_regime.get('direction', 'NONE')}",
            f"Score: {crash_regime.get('score', 0)}/100",
            f"Agreement: {crash_regime.get('agreement', 'UNKNOWN')}",
            f"Selected TF Signal: {crash_regime.get('selected_tf_signal', 'UNKNOWN')}",
            f"Fast TF Signal: {crash_regime.get('fast_tf_signal', 'NOT_USED')}",
            f"New Longs Risky: {crash_regime.get('new_longs_risky', False)}",
            f"New Shorts Risky: {crash_regime.get('new_shorts_risky', False)}",
            f"Pending Longs Should Pause: {crash_regime.get('pending_longs_should_pause', False)}",
            f"Pending Shorts Should Pause: {crash_regime.get('pending_shorts_should_pause', False)}",
            f"Pending Longs May Miss Limit: {crash_regime.get('pending_longs_may_miss_limit', False)}",
            f"Pending Shorts May Miss Limit: {crash_regime.get('pending_shorts_may_miss_limit', False)}",
            f"Running Longs Need Protection: {crash_regime.get('running_longs_need_protection', False)}",
            f"Running Shorts Need Protection: {crash_regime.get('running_shorts_need_protection', False)}",
            f"Running Longs Favorable: {crash_regime.get('running_longs_favorable', False)}",
            f"Running Shorts Favorable: {crash_regime.get('running_shorts_favorable', False)}",
            f"Watch Levels: {crash_regime.get('watch_levels', {})}",
            f"Advice: {crash_regime.get('advice', '')}",
            f"Reason: {crash_regime.get('reason', '')}",
        ]

        return "\n".join(lines)

    except Exception:
        return ""