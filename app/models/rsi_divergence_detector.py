# app/models/rsi_divergence_detector.py

"""
RSI Divergence Detector

Purpose:
- Detect regular and hidden RSI divergences.
- Use as confirmation / warning only.
- NOT a standalone signal generator.

Recommended usage:
1. Fresh setup generation:
   - Bullish divergence supports LONG near support/Fib/retest.
   - Bearish divergence supports SHORT near resistance/Fib/retest.
   - Opposite divergence warns against wrong-side trades.

2. Trade guidance:
   - LONG running near TP + bearish divergence = caution/protect profit.
   - SHORT running near TP + bullish divergence = caution/protect profit.

Output is designed to be passed into:
- market_context_text
- context_bundle
- setup_planner confluence/warning logic
- trade_guidance_manager warning flags
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import pandas as pd


# ==========================================================
# Basic helpers
# ==========================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, (int, float)):
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


def _format_price(value: Any) -> str:
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

        out = out.rename(
            columns={c: rename_map.get(str(c).lower(), c) for c in out.columns}
        )

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


# ==========================================================
# RSI
# ==========================================================

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder-style RSI calculation using pandas.
    No external dependency required.
    """
    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < period + 5:
            return pd.Series(dtype=float)

        close = d["Close"].astype(float)

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.where(avg_loss.ne(0))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask(
            avg_loss.eq(0) & avg_gain.gt(0),
            100.0,
        )

        rsi = rsi.fillna(50.0)
        return rsi

    except Exception:
        return pd.Series(dtype=float)


# ==========================================================
# Pivot detection
# ==========================================================

def _find_pivot_lows(
    df: pd.DataFrame,
    rsi: pd.Series,
    left: int = 2,
    right: int = 2,
) -> List[Dict[str, Any]]:
    """
    Find local swing lows.
    """
    pivots: List[Dict[str, Any]] = []

    try:
        d = _normalize_ohlc(df)
        if d.empty or rsi.empty:
            return pivots

        lows = d["Low"].values

        for i in range(left, len(d) - right):
            current = lows[i]
            window = lows[i - left:i + right + 1]

            if current == min(window):
                try:
                    pivots.append({
                        "index": i,
                        "time": str(d.index[i]),
                        "price": float(d.iloc[i]["Low"]),
                        "close": float(d.iloc[i]["Close"]),
                        "rsi": float(rsi.iloc[i]),
                    })
                except Exception:
                    continue

    except Exception:
        pass

    return pivots


def _find_pivot_highs(
    df: pd.DataFrame,
    rsi: pd.Series,
    left: int = 2,
    right: int = 2,
) -> List[Dict[str, Any]]:
    """
    Find local swing highs.
    """
    pivots: List[Dict[str, Any]] = []

    try:
        d = _normalize_ohlc(df)
        if d.empty or rsi.empty:
            return pivots

        highs = d["High"].values

        for i in range(left, len(d) - right):
            current = highs[i]
            window = highs[i - left:i + right + 1]

            if current == max(window):
                try:
                    pivots.append({
                        "index": i,
                        "time": str(d.index[i]),
                        "price": float(d.iloc[i]["High"]),
                        "close": float(d.iloc[i]["Close"]),
                        "rsi": float(rsi.iloc[i]),
                    })
                except Exception:
                    continue

    except Exception:
        pass

    return pivots


def _recent_pair(
    pivots: List[Dict[str, Any]],
    max_bars_between: int = 80,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Pick the most recent pivot pair that is not too far apart.
    """
    try:
        if len(pivots) < 2:
            return None, None

        recent = pivots[-8:]

        for j in range(len(recent) - 1, 0, -1):
            p2 = recent[j]
            for i in range(j - 1, -1, -1):
                p1 = recent[i]

                bars_between = int(p2["index"] - p1["index"])
                if 3 <= bars_between <= max_bars_between:
                    return p1, p2

    except Exception:
        pass

    return None, None


# ==========================================================
# Divergence classification
# ==========================================================

def _classify_bullish_divergence(
    low1: Dict[str, Any],
    low2: Dict[str, Any],
    min_price_diff_pct: float,
    min_rsi_diff: float,
) -> Optional[Dict[str, Any]]:
    """
    Bullish regular:
      Price lower low, RSI higher low.

    Hidden bullish:
      Price higher low, RSI lower low.
    """
    try:
        p1 = float(low1["price"])
        p2 = float(low2["price"])
        r1 = float(low1["rsi"])
        r2 = float(low2["rsi"])

        if p1 <= 0:
            return None

        price_diff_pct = ((p2 - p1) / p1) * 100.0
        rsi_diff = r2 - r1

        # Regular bullish: price lower low, RSI higher low.
        if price_diff_pct <= -min_price_diff_pct and rsi_diff >= min_rsi_diff:
            return {
                "has_divergence": True,
                "divergence_type": "BULLISH_REGULAR",
                "side": "BULLISH",
                "signal_role": "REVERSAL_WARNING",
                "confluence_for": "LONG",
                "warning_for": "SHORT",
                "price_relation": "Price made a lower low while RSI made a higher low.",
                "price_diff_pct": round(price_diff_pct, 3),
                "rsi_diff": round(rsi_diff, 3),
                "pivot_1": low1,
                "pivot_2": low2,
            }

        # Hidden bullish: price higher low, RSI lower low.
        if price_diff_pct >= min_price_diff_pct and rsi_diff <= -min_rsi_diff:
            return {
                "has_divergence": True,
                "divergence_type": "BULLISH_HIDDEN",
                "side": "BULLISH",
                "signal_role": "TREND_CONTINUATION",
                "confluence_for": "LONG",
                "warning_for": "SHORT",
                "price_relation": "Price made a higher low while RSI made a lower low.",
                "price_diff_pct": round(price_diff_pct, 3),
                "rsi_diff": round(rsi_diff, 3),
                "pivot_1": low1,
                "pivot_2": low2,
            }

    except Exception:
        return None

    return None


def _classify_bearish_divergence(
    high1: Dict[str, Any],
    high2: Dict[str, Any],
    min_price_diff_pct: float,
    min_rsi_diff: float,
) -> Optional[Dict[str, Any]]:
    """
    Bearish regular:
      Price higher high, RSI lower high.

    Hidden bearish:
      Price lower high, RSI higher high.
    """
    try:
        p1 = float(high1["price"])
        p2 = float(high2["price"])
        r1 = float(high1["rsi"])
        r2 = float(high2["rsi"])

        if p1 <= 0:
            return None

        price_diff_pct = ((p2 - p1) / p1) * 100.0
        rsi_diff = r2 - r1

        # Regular bearish: price higher high, RSI lower high.
        if price_diff_pct >= min_price_diff_pct and rsi_diff <= -min_rsi_diff:
            return {
                "has_divergence": True,
                "divergence_type": "BEARISH_REGULAR",
                "side": "BEARISH",
                "signal_role": "REVERSAL_WARNING",
                "confluence_for": "SHORT",
                "warning_for": "LONG",
                "price_relation": "Price made a higher high while RSI made a lower high.",
                "price_diff_pct": round(price_diff_pct, 3),
                "rsi_diff": round(rsi_diff, 3),
                "pivot_1": high1,
                "pivot_2": high2,
            }

        # Hidden bearish: price lower high, RSI higher high.
        if price_diff_pct <= -min_price_diff_pct and rsi_diff >= min_rsi_diff:
            return {
                "has_divergence": True,
                "divergence_type": "BEARISH_HIDDEN",
                "side": "BEARISH",
                "signal_role": "TREND_CONTINUATION",
                "confluence_for": "SHORT",
                "warning_for": "LONG",
                "price_relation": "Price made a lower high while RSI made a higher high.",
                "price_diff_pct": round(price_diff_pct, 3),
                "rsi_diff": round(rsi_diff, 3),
                "pivot_1": high1,
                "pivot_2": high2,
            }

    except Exception:
        return None

    return None


def _strength_from_divergence(div: Dict[str, Any]) -> str:
    try:
        price_diff = abs(float(div.get("price_diff_pct") or 0))
        rsi_diff = abs(float(div.get("rsi_diff") or 0))

        score = 0

        if price_diff >= 0.35:
            score += 1
        if price_diff >= 0.75:
            score += 1

        if rsi_diff >= 5:
            score += 1
        if rsi_diff >= 9:
            score += 1

        role = str(div.get("signal_role") or "")
        if role == "REVERSAL_WARNING":
            score += 1

        if score >= 4:
            return "STRONG"
        if score >= 2:
            return "MEDIUM"

    except Exception:
        pass

    return "WEAK"


# ==========================================================
# Zone proximity helpers
# ==========================================================

def _extract_numbers(text: str) -> List[float]:
    try:
        import re
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
        out = []
        for n in nums:
            v = _safe_float(n)
            if v is not None:
                out.append(float(v))
        return out
    except Exception:
        return []


def _extract_context_line(market_context_text: str, label: str) -> str:
    try:
        if not market_context_text:
            return ""

        import re
        pattern = rf"{re.escape(label)}:\s*(.+)"
        match = re.search(pattern, market_context_text)
        if match:
            return match.group(1).strip()
    except Exception:
        pass

    return ""


def _extract_zone_from_context(market_context_text: str, label: str) -> List[float]:
    try:
        line = _extract_context_line(market_context_text, label)
        return _extract_numbers(line)
    except Exception:
        return []


def _near_any_level(
    price: Optional[float],
    levels: List[float],
    tolerance_pct: float = 0.40,
) -> bool:
    try:
        if price is None or not levels:
            return False

        for lv in levels:
            if lv is None or lv <= 0:
                continue

            dist_pct = abs(price - lv) / lv * 100.0
            if dist_pct <= tolerance_pct:
                return True

    except Exception:
        pass

    return False


def _zone_proximity(
    div: Dict[str, Any],
    market_context_text: str = "",
    tolerance_pct: float = 0.40,
) -> Dict[str, Any]:
    """
    Check if divergence pivot is near S/R zones from market_context_text.
    This is optional. If market_context_text is not supplied, returns unknown.
    """
    out = {
        "near_key_zone": False,
        "near_support": False,
        "near_resistance": False,
        "zone_note": "No market context supplied for zone proximity.",
    }

    try:
        pivot_2 = div.get("pivot_2") or {}
        pivot_price = _safe_float(pivot_2.get("price"))

        if pivot_price is None or not market_context_text:
            return out

        support_levels: List[float] = []
        resistance_levels: List[float] = []

        for label in [
            "Nearest Support",
            "Major Support Zone",
            "Long Lookback Major Support Zone",
        ]:
            support_levels.extend(_extract_zone_from_context(market_context_text, label))

        for label in [
            "Nearest Resistance",
            "Major Resistance Zone",
            "Long Lookback Major Resistance Zone",
        ]:
            resistance_levels.extend(_extract_zone_from_context(market_context_text, label))

        near_support = _near_any_level(pivot_price, support_levels, tolerance_pct)
        near_resistance = _near_any_level(pivot_price, resistance_levels, tolerance_pct)

        out["near_support"] = near_support
        out["near_resistance"] = near_resistance
        out["near_key_zone"] = bool(near_support or near_resistance)

        if near_support and near_resistance:
            out["zone_note"] = "Divergence pivot is near both support and resistance/range zone."
        elif near_support:
            out["zone_note"] = "Divergence pivot is near support."
        elif near_resistance:
            out["zone_note"] = "Divergence pivot is near resistance."
        else:
            out["zone_note"] = "Divergence pivot is not close to a printed key S/R zone."

    except Exception:
        pass

    return out


# ==========================================================
# Main detection
# ==========================================================

def detect_rsi_divergence(
    df: Optional[pd.DataFrame],
    market_context_text: str = "",
    rsi_period: int = 14,
    pivot_left: int = 2,
    pivot_right: int = 2,
    lookback: int = 120,
    min_price_diff_pct: float = 0.15,
    min_rsi_diff: float = 3.0,
    max_bars_between_pivots: int = 80,
) -> Dict[str, Any]:
    """
    Detect most recent RSI divergence.

    Notes:
    - Uses closed-candle dataframe.
    - Use as confluence/warning only.
    - Does not create trades.
    """
    result: Dict[str, Any] = {
        "has_divergence": False,
        "divergence_type": "NONE",
        "side": "NONE",
        "strength": "NONE",
        "signal_role": "NONE",
        "confluence_for": "NONE",
        "warning_for": "NONE",
        "near_key_zone": False,
        "near_support": False,
        "near_resistance": False,
        "reason": "No RSI divergence detected.",
        "details": {},
    }

    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < max(rsi_period + 10, 30):
            result["reason"] = "Not enough data for RSI divergence detection."
            return result

        d = d.tail(lookback).copy()

        rsi = calculate_rsi(d, period=rsi_period)
        if rsi.empty or len(rsi) != len(d):
            result["reason"] = "RSI calculation unavailable."
            return result

        pivot_lows = _find_pivot_lows(d, rsi, left=pivot_left, right=pivot_right)
        pivot_highs = _find_pivot_highs(d, rsi, left=pivot_left, right=pivot_right)

        low1, low2 = _recent_pair(pivot_lows, max_bars_between=max_bars_between_pivots)
        high1, high2 = _recent_pair(pivot_highs, max_bars_between=max_bars_between_pivots)

        candidates: List[Dict[str, Any]] = []

        if low1 and low2:
            bull = _classify_bullish_divergence(
                low1,
                low2,
                min_price_diff_pct=min_price_diff_pct,
                min_rsi_diff=min_rsi_diff,
            )
            if bull:
                candidates.append(bull)

        if high1 and high2:
            bear = _classify_bearish_divergence(
                high1,
                high2,
                min_price_diff_pct=min_price_diff_pct,
                min_rsi_diff=min_rsi_diff,
            )
            if bear:
                candidates.append(bear)

        if not candidates:
            result["details"] = {
                "pivot_lows_found": len(pivot_lows),
                "pivot_highs_found": len(pivot_highs),
                "rsi_latest": round(float(rsi.iloc[-1]), 2),
            }
            return result

        # Pick most recent divergence by second pivot index.
        candidates.sort(key=lambda x: int((x.get("pivot_2") or {}).get("index", 0)), reverse=True)
        div = candidates[0]

        div["strength"] = _strength_from_divergence(div)

        proximity = _zone_proximity(
            div,
            market_context_text=market_context_text,
            tolerance_pct=0.40,
        )

        div.update(proximity)

        pivot_2 = div.get("pivot_2") or {}
        reason = (
            f"{div.get('divergence_type')} detected. "
            f"{div.get('price_relation')} "
            f"Second pivot: price {_format_price(pivot_2.get('price'))}, "
            f"RSI {round(float(pivot_2.get('rsi', 0)), 2)}. "
            f"{div.get('zone_note', '')}"
        )

        result.update({
            "has_divergence": True,
            "divergence_type": div.get("divergence_type", "UNKNOWN"),
            "side": div.get("side", "UNKNOWN"),
            "strength": div.get("strength", "WEAK"),
            "signal_role": div.get("signal_role", ""),
            "confluence_for": div.get("confluence_for", "NONE"),
            "warning_for": div.get("warning_for", "NONE"),
            "near_key_zone": div.get("near_key_zone", False),
            "near_support": div.get("near_support", False),
            "near_resistance": div.get("near_resistance", False),
            "reason": reason,
            "details": div,
        })

        return result

    except Exception as exc:
        result["reason"] = f"RSI divergence detection failed: {exc}"
        return result


# ==========================================================
# Formatting
# ==========================================================

def format_rsi_divergence_context_text(divergence: Dict[str, Any]) -> str:
    """
    Human/AI-readable text to append into market_context_text.
    """
    try:
        d = divergence or {}

        lines = [
            "--- RSI DIVERGENCE CONTEXT ---",
            f"Has Divergence: {d.get('has_divergence', False)}",
            f"Type: {d.get('divergence_type', 'NONE')}",
            f"Side: {d.get('side', 'NONE')}",
            f"Strength: {d.get('strength', 'NONE')}",
            f"Signal Role: {d.get('signal_role', 'NONE')}",
            f"Confluence For: {d.get('confluence_for', 'NONE')}",
            f"Warning For: {d.get('warning_for', 'NONE')}",
            f"Near Key Zone: {d.get('near_key_zone', False)}",
            f"Near Support: {d.get('near_support', False)}",
            f"Near Resistance: {d.get('near_resistance', False)}",
            f"Reason: {d.get('reason', '')}",
            "Usage Rule: RSI divergence is confirmation/warning only. Do not create a trade from RSI divergence alone.",
        ]

        return "\n".join(lines)

    except Exception:
        return (
            "--- RSI DIVERGENCE CONTEXT ---\n"
            "RSI divergence context unavailable.\n"
            "Usage Rule: RSI divergence is confirmation/warning only."
        )


def summarize_rsi_divergence_for_candidate(
    divergence: Dict[str, Any],
    candidate_side: str,
) -> Dict[str, Any]:
    """
    Helper for setup_planner later.

    Returns:
    - supports_candidate
    - warns_candidate
    - should_block
    - note

    IMPORTANT:
    This function is intentionally soft.
    It should not block good trades by itself unless STRONG + near key zone.
    """
    out = {
        "supports_candidate": False,
        "warns_candidate": False,
        "should_block": False,
        "note": "",
    }

    try:
        d = divergence or {}
        side = str(candidate_side or "").upper()

        if not d.get("has_divergence") or side not in ["LONG", "SHORT"]:
            out["note"] = "No RSI divergence effect."
            return out

        confluence_for = str(d.get("confluence_for") or "").upper()
        warning_for = str(d.get("warning_for") or "").upper()
        strength = str(d.get("strength") or "WEAK").upper()
        near_key_zone = bool(d.get("near_key_zone"))

        if side == confluence_for:
            out["supports_candidate"] = True
            out["note"] = f"RSI divergence supports {side}: {d.get('divergence_type')}."

        if side == warning_for:
            out["warns_candidate"] = True
            out["note"] = f"RSI divergence warns against {side}: {d.get('divergence_type')}."

            # Soft block only when high quality warning.
            if strength == "STRONG" and near_key_zone:
                out["should_block"] = True
                out["note"] += " Strong divergence near key zone; safer to WAIT."

        return out

    except Exception as exc:
        out["note"] = f"RSI divergence candidate summary failed: {exc}"
        return out


def summarize_rsi_divergence_for_guidance(
    divergence: Dict[str, Any],
    active_side: str,
) -> Dict[str, Any]:
    """
    Helper for trade_guidance_manager later.

    Existing trade:
    - LONG active + bearish divergence = caution/protect profit.
    - SHORT active + bullish divergence = caution/protect profit.
    """
    out = {
        "has_guidance_warning": False,
        "risk_add": 0,
        "flag": "",
        "note": "",
    }

    try:
        d = divergence or {}
        side = str(active_side or "").upper()

        if not d.get("has_divergence") or side not in ["LONG", "SHORT"]:
            out["note"] = "No RSI divergence guidance warning."
            return out

        warning_for = str(d.get("warning_for") or "").upper()
        strength = str(d.get("strength") or "WEAK").upper()
        near_key_zone = bool(d.get("near_key_zone"))

        if side == warning_for:
            out["has_guidance_warning"] = True
            out["flag"] = f"RSI_{d.get('divergence_type')}"
            out["note"] = (
                f"RSI divergence warning for active {side}: "
                f"{d.get('reason', '')}"
            )

            if strength == "STRONG":
                out["risk_add"] = 25
            elif strength == "MEDIUM":
                out["risk_add"] = 15
            else:
                out["risk_add"] = 8

            if near_key_zone:
                out["risk_add"] += 5

        return out

    except Exception as exc:
        out["note"] = f"RSI divergence guidance summary failed: {exc}"
        return out