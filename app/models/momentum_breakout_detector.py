# app/models/momentum_breakout_detector.py

"""
Momentum Breakout Detector

Purpose:
- Detect early breakout / breakdown momentum on lower execution timeframe.
- This does NOT call AI.
- This does NOT override higher timeframe trend.
- setup_planner.py can use this context to create a cautious early continuation candidate.
- suggested_entry/suggested_stop are context suggestions only; planner validation must pass before any signal.

Typical use:
- User TF 1h -> lower execution TF 15m
- User TF 4h -> lower execution TF 1h
"""

from typing import Any, Dict, Optional
import math

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


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

        return out.dropna(subset=required)

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


def _volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    try:
        if "Volume" not in df.columns or len(df) < lookback + 2:
            return 1.0

        current_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].iloc[-lookback - 1:-1].mean())

        if avg_vol <= 0:
            return 1.0

        return current_vol / avg_vol

    except Exception:
        return 1.0

def _candle_quality(open_p: float, high_p: float, low_p: float, close_p: float, side: str) -> Dict[str, Any]:
    """
    Check candle body/wick quality to avoid fake breakout candles.

    LONG:
      - strong body
      - upper wick should not dominate

    SHORT:
      - strong body
      - lower wick should not dominate
    """
    try:
        candle_range = max(high_p - low_p, 1e-12)
        body = abs(close_p - open_p)

        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p

        body_ratio = body / candle_range
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        side = str(side or "").upper()

        wick_ok = True
        wick_warning = ""

        if side == "LONG":
            if upper_wick_ratio > 0.45:
                wick_ok = False
                wick_warning = "Large upper wick after breakout; possible bull trap."
        elif side == "SHORT":
            if lower_wick_ratio > 0.45:
                wick_ok = False
                wick_warning = "Large lower wick after breakdown; possible bear trap."

        body_ok = body_ratio >= 0.40

        return {
            "body_ratio": round(body_ratio, 3),
            "upper_wick_ratio": round(upper_wick_ratio, 3),
            "lower_wick_ratio": round(lower_wick_ratio, 3),
            "body_ok": body_ok,
            "wick_ok": wick_ok,
            "wick_warning": wick_warning,
        }

    except Exception:
        return {
            "body_ratio": 0,
            "upper_wick_ratio": 0,
            "lower_wick_ratio": 0,
            "body_ok": False,
            "wick_ok": False,
            "wick_warning": "Candle quality check failed.",
        }


def _detect_close_back_inside(
    df: pd.DataFrame,
    side: str,
    breakout_level: float,
    lookahead_candles: int = 2,
) -> bool:
    """
    Detect fake breakout only after the breakout candle.

    Important:
    - Do not treat candles before the breakout as fake close-back-inside.
    - Only recent closes AFTER a breakout should be checked.
    """
    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < 4:
            return False

        side = str(side or "").upper()

        closes = list(d["Close"].values)

        breakout_idx = None
        for i in range(len(closes) - 1, -1, -1):
            c = float(closes[i])

            if side == "LONG" and c > breakout_level:
                breakout_idx = i
                break

            if side == "SHORT" and c < breakout_level:
                breakout_idx = i
                break

        if breakout_idx is None:
            return False

        after = d.iloc[breakout_idx + 1: breakout_idx + 1 + lookahead_candles]

        # If breakout is the latest candle, there is no later candle to prove fake yet.
        if after.empty:
            return False

        if side == "LONG":
            return bool((after["Close"] < breakout_level).any())

        if side == "SHORT":
            return bool((after["Close"] > breakout_level).any())

    except Exception:
        pass

    return False


def _detect_retest_confirmation(
    df: pd.DataFrame,
    side: str,
    breakout_level: float,
    atr_val: float,
    max_retest_candles: int = 6,
) -> Dict[str, Any]:
    """
    Check whether breakout level was retested and held/rejected.

    LONG:
      price retests near breakout_level and closes back above it.

    SHORT:
      price retests near breakdown level and closes back below it.
    """
    result = {
        "retest_seen": False,
        "retest_confirmed": False,
        "retest_reason": "",
    }

    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < max_retest_candles + 2:
            result["retest_reason"] = "Not enough candles for retest confirmation."
            return result

        side = str(side or "").upper()
        tolerance = max(float(atr_val) * 0.35, abs(float(breakout_level)) * 0.001)

        recent = d.tail(max_retest_candles)

        for row in recent.itertuples():
            low = float(row.Low)
            high = float(row.High)
            close = float(row.Close)
            open_p = float(row.Open)

            if side == "LONG":
                touched_level = low <= breakout_level + tolerance
                held_close = close > breakout_level
                bullish_close = close >= open_p

                if touched_level:
                    result["retest_seen"] = True

                    if held_close and bullish_close:
                        result["retest_confirmed"] = True
                        result["retest_reason"] = (
                            "Breakout level retested and held with bullish close."
                        )
                        return result

            elif side == "SHORT":
                touched_level = high >= breakout_level - tolerance
                held_close = close < breakout_level
                bearish_close = close <= open_p

                if touched_level:
                    result["retest_seen"] = True

                    if held_close and bearish_close:
                        result["retest_confirmed"] = True
                        result["retest_reason"] = (
                            "Breakdown level retested and rejected with bearish close."
                        )
                        return result

        if result["retest_seen"]:
            result["retest_reason"] = "Retest seen, but hold/rejection not confirmed yet."
        else:
            result["retest_reason"] = "No retest of breakout/breakdown level yet."

        return result

    except Exception as exc:
        result["retest_reason"] = f"Retest confirmation failed: {exc}"
        return result


def detect_momentum_breakout(
    df_lower: Optional[pd.DataFrame],
    side_hint: str = "ANY",
    timeframe: str = "15m",
    range_lookback: int = 24,
    max_breakout_age: int = 3,
) -> Dict[str, Any]:
    """
    Detect early breakout/breakdown on lower timeframe.

    Returns:
    {
      "enabled": True/False,
      "momentum_breakout": True/False,
      "side": "LONG"/"SHORT"/"UNKNOWN",
      "score": 0-100,
      "strength": "STRONG"/"MEDIUM"/"WEAK"/"NONE",
      "breakout_level": float,
      "current_price": float,
      "atr": float,
      "volume_ratio": float,
      "breakout_age": int,
      "is_extended": bool,
      "needs_retest": bool,
      "suggested_entry": float,
      "suggested_stop": float,
      "reason": "..."
    }
    """
    result = {
        "enabled": False,
        "momentum_breakout": False,
        "side": "UNKNOWN",
        "score": 0,
        "strength": "NONE",
        "timeframe": timeframe,
        "breakout_level": None,
        "current_price": None,
        "atr": None,
        "volume_ratio": None,
        "breakout_age": None,
        "is_extended": False,
        "needs_retest": False,
        "suggested_entry": None,
        "suggested_stop": None,
        "reason": "Lower timeframe data unavailable.",
    }

    try:
        d = _normalize_ohlc(df_lower)
        if len(d) < max(range_lookback + 10, 40):
            result["reason"] = "Not enough lower timeframe candles for momentum breakout detection."
            return result

        result["enabled"] = True

        side_hint = str(side_hint or "ANY").upper()
        atr_val = _atr(d, 14) or 0.0
        if atr_val <= 0:
            result["reason"] = "ATR unavailable on lower timeframe."
            return result

        current_price = float(d["Close"].iloc[-1])
        result["current_price"] = current_price
        result["atr"] = atr_val

        # Use previous candles only for range, excluding latest candle.
        recent = d.iloc[-range_lookback - 1:-1].copy()
        range_high = float(recent["High"].max())
        range_low = float(recent["Low"].min())

        last = d.iloc[-1]
        last_open = float(last["Open"])
        last_high = float(last["High"])
        last_low = float(last["Low"])
        last_close = float(last["Close"])

        candle_quality = _candle_quality(
            open_p=last_open,
            high_p=last_high,
            low_p=last_low,
            close_p=last_close,
            side="LONG" if last_close > range_high else "SHORT" if last_close < range_low else "UNKNOWN",
        )

        body_ratio = float(candle_quality.get("body_ratio") or 0)

        vol_ratio = _volume_ratio(d, lookback=20)
        result["volume_ratio"] = round(vol_ratio, 2)

        long_break = last_close > range_high
        short_break = last_close < range_low

        if side_hint == "LONG":
            short_break = False
        elif side_hint == "SHORT":
            long_break = False

        if not long_break and not short_break:
            result["reason"] = "No fresh range breakout/breakdown on lower timeframe."
            return result

        side = "LONG" if long_break else "SHORT"
        breakout_level = range_high if side == "LONG" else range_low

        # Find how many candles ago the first close beyond level happened.
        breakout_age = None
        tail = d.tail(max_breakout_age + 6).copy()

        closes = list(tail["Close"].values)
        for i in range(len(closes) - 1, -1, -1):
            c = float(closes[i])
            if side == "LONG" and c <= breakout_level:
                breakout_age = len(closes) - 1 - i
                break
            if side == "SHORT" and c >= breakout_level:
                breakout_age = len(closes) - 1 - i
                break

        if breakout_age is None:
            breakout_age = max_breakout_age + 5

        distance_from_breakout = (
            current_price - breakout_level if side == "LONG"
            else breakout_level - current_price
        )

        fake_close_back_inside = _detect_close_back_inside(
            df=d,
            side=side,
            breakout_level=breakout_level,
            lookahead_candles=2,
        )

        retest_info = _detect_retest_confirmation(
            df=d,
            side=side,
            breakout_level=breakout_level,
            atr_val=atr_val,
            max_retest_candles=6,
        )

        retest_seen = bool(retest_info.get("retest_seen"))
        retest_confirmed = bool(retest_info.get("retest_confirmed"))

        if retest_confirmed:
            breakout_state = "RETEST_CONFIRMED"
        elif breakout_age <= 2:
            breakout_state = "FRESH_BREAKOUT"
        elif retest_seen:
            breakout_state = "RETEST_WAITING_CONFIRMATION"
        else:
            breakout_state = "EXTENDED_OR_UNCONFIRMED"

        distance_atr = distance_from_breakout / atr_val if atr_val > 0 else 999
        is_extended = bool(distance_atr > 1.8 or breakout_age > max_breakout_age)

        score = 45

        if candle_quality.get("body_ok"):
            if body_ratio >= 0.55:
                score += 15
            elif body_ratio >= 0.40:
                score += 8
        else:
            score -= 10

        if candle_quality.get("wick_ok"):
            score += 6
        else:
            score -= 18

        if vol_ratio >= 1.8:
            score += 18
        elif vol_ratio >= 1.3:
            score += 10

        if distance_atr <= 0.8:
            score += 14
        elif distance_atr <= 1.3:
            score += 8
        elif distance_atr > 1.8:
            score -= 18

        if breakout_age <= 1:
            score += 10
        elif breakout_age <= 3:
            score += 4
        else:
            score -= 15

        if fake_close_back_inside:
            score -= 35

        if retest_confirmed:
            score += 18
        elif retest_seen:
            score += 4

        if breakout_state == "FRESH_BREAKOUT":
            score += 6
        elif breakout_state == "RETEST_CONFIRMED":
            score += 14
        elif breakout_state == "RETEST_WAITING_CONFIRMATION":
            score -= 4
        else:
            score -= 10

        score = max(0, min(100, int(score)))

        if score >= 78:
            strength = "STRONG"
        elif score >= 62:
            strength = "MEDIUM"
        elif score >= 48:
            strength = "WEAK"
        else:
            strength = "NONE"

        # Suggested entry is not a market chase.
        # It waits for a shallow pullback toward breakout level.
        if side == "LONG":
            suggested_entry = current_price - min(distance_from_breakout * 0.35, atr_val * 0.45)
            suggested_entry = max(suggested_entry, breakout_level)
            suggested_stop = breakout_level - (atr_val * 0.75)
        else:
            suggested_entry = current_price + min(distance_from_breakout * 0.35, atr_val * 0.45)
            suggested_entry = min(suggested_entry, breakout_level)
            suggested_stop = breakout_level + (atr_val * 0.75)

        allow_momentum = (
            strength in ["STRONG", "MEDIUM"]
            and not fake_close_back_inside
            and candle_quality.get("body_ok")
            and candle_quality.get("wick_ok")
            and breakout_state in ["FRESH_BREAKOUT", "RETEST_CONFIRMED"]
        )

        result.update({
            "momentum_breakout": allow_momentum,
            "side": side,
            "score": score,
            "strength": strength,
            "breakout_level": float(breakout_level),
            "breakout_age": int(breakout_age),
            "is_extended": is_extended,
            "needs_retest": is_extended,
            "suggested_entry": float(suggested_entry),
            "suggested_stop": float(suggested_stop),
            "breakout_state": breakout_state,
            "fake_close_back_inside": fake_close_back_inside,
            "retest_seen": retest_seen,
            "retest_confirmed": retest_confirmed,
            "retest_reason": retest_info.get("retest_reason", ""),
            "body_ratio": candle_quality.get("body_ratio"),
            "upper_wick_ratio": candle_quality.get("upper_wick_ratio"),
            "lower_wick_ratio": candle_quality.get("lower_wick_ratio"),
            "candle_body_ok": candle_quality.get("body_ok"),
            "candle_wick_ok": candle_quality.get("wick_ok"),
            "wick_warning": candle_quality.get("wick_warning", ""),
            "reason": (
                f"{side} lower-TF breakout detected on {timeframe}. "
                f"Level={breakout_level:.8f}, current={current_price:.8f}, "
                f"distance={distance_atr:.2f} ATR, age={breakout_age} candles, "
                f"volume_ratio={vol_ratio:.2f}, body_ratio={body_ratio:.2f}, "
                f"extended={is_extended}."
            ),
        })

        if is_extended and not retest_confirmed:
            result["momentum_breakout"] = False
            result["needs_retest"] = True
            result["reason"] += " Move is already extended; wait for retest confirmation."

        if fake_close_back_inside:
            result["momentum_breakout"] = False
            result["reason"] += " Close returned inside previous range; possible fake breakout."

        if not candle_quality.get("wick_ok"):
            result["momentum_breakout"] = False
            result["reason"] += f" {candle_quality.get('wick_warning', '')}"

        return result

    except Exception as exc:
        result["reason"] = f"Momentum breakout detection failed: {exc}"
        return result


def format_momentum_breakout_context_text(ctx: Dict[str, Any]) -> str:
    if not isinstance(ctx, dict):
        return "--- MOMENTUM BREAKOUT CONTEXT ---\nUnavailable."

    return "\n".join([
        "--- MOMENTUM BREAKOUT CONTEXT ---",
        f"Enabled: {ctx.get('enabled')}",
        f"Momentum Breakout: {ctx.get('momentum_breakout')}",
        f"Timeframe: {ctx.get('timeframe')}",
        f"Side: {ctx.get('side')}",
        f"Strength: {ctx.get('strength')}",
        f"Score: {ctx.get('score')}",
        f"Breakout Level: {ctx.get('breakout_level')}",
        f"Current Price: {ctx.get('current_price')}",
        f"ATR: {ctx.get('atr')}",
        f"Volume Ratio: {ctx.get('volume_ratio')}",
        f"Breakout Age: {ctx.get('breakout_age')}",
        f"Extended: {ctx.get('is_extended')}",
        f"Needs Retest: {ctx.get('needs_retest')}",
        f"Breakout State: {ctx.get('breakout_state')}",
        f"Fake Close Back Inside: {ctx.get('fake_close_back_inside')}",
        f"Retest Seen: {ctx.get('retest_seen')}",
        f"Retest Confirmed: {ctx.get('retest_confirmed')}",
        f"Retest Reason: {ctx.get('retest_reason')}",
        f"Body Ratio: {ctx.get('body_ratio')}",
        f"Upper Wick Ratio: {ctx.get('upper_wick_ratio')}",
        f"Lower Wick Ratio: {ctx.get('lower_wick_ratio')}",
        f"Candle Body OK: {ctx.get('candle_body_ok')}",
        f"Candle Wick OK: {ctx.get('candle_wick_ok')}",
        f"Wick Warning: {ctx.get('wick_warning')}",
        f"Suggested Entry: {ctx.get('suggested_entry')}",
        f"Suggested Stop: {ctx.get('suggested_stop')}",
        f"Reason: {ctx.get('reason')}",
    ])