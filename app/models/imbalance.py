import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
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


def _extract_price(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)):
            return _safe_float(value)

        text = str(value or "").replace(",", "").strip()
        if not text:
            return None

        import re
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None

        return _safe_float(match.group(0))
    except Exception:
        return None


def _normalize_side(value: Any) -> str:
    side = _upper(value)

    if "LONG" in side or side in ["BUY", "BULL", "BULLISH"]:
        return "LONG"

    if "SHORT" in side or side in ["SELL", "BEAR", "BEARISH"]:
        return "SHORT"

    return "UNKNOWN"


def _normalize_ohlc(df: Any) -> pd.DataFrame:
    """
    Normalize any OHLC dataframe into columns:
    Open, High, Low, Close, Volume(optional)

    This module must never break the main backend.
    If input is not usable, returns empty DataFrame.
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
            "timestamp": "Date",
        }

        out = out.rename(
            columns={
                c: rename_map.get(str(c).lower().strip(), c)
                for c in out.columns
            }
        )

        required = ["Open", "High", "Low", "Close"]
        if not set(required).issubset(set(out.columns)):
            return pd.DataFrame()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=required)

        if out.empty:
            return pd.DataFrame()

        return out

    except Exception:
        return pd.DataFrame()


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < 3:
            return pd.Series(dtype="float64")

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

        return tr.rolling(period).mean()
    except Exception:
        return pd.Series(dtype="float64")


def _latest_atr(df: pd.DataFrame, fallback: Optional[float] = None) -> Optional[float]:
    try:
        fb = _safe_float(fallback)
        if fb is not None and fb > 0:
            return fb

        atr_s = _atr_series(df)
        if atr_s.empty:
            return None

        val = _safe_float(atr_s.dropna().iloc[-1] if not atr_s.dropna().empty else None)
        return val if val and val > 0 else None
    except Exception:
        return None


def _candle_time(df: pd.DataFrame, idx: int) -> str:
    try:
        index_value = df.index[idx]
        return str(index_value)
    except Exception:
        return ""


def _detect_displacement(
    df: pd.DataFrame,
    *,
    atr: Optional[float],
    lookback: int = 12,
) -> Dict[str, Any]:
    """
    Detect recent displacement candle.

    Bullish displacement:
    - big candle body
    - close > open
    - range large compared with ATR

    Bearish displacement:
    - big candle body
    - close < open
    - range large compared with ATR
    """
    result = {
        "detected": False,
        "direction": "NONE",
        "range_atr": 0.0,
        "body_ratio": 0.0,
        "age_candles": None,
        "reason": "",
    }

    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < 5:
            result["reason"] = "Not enough candles for displacement check."
            return result

        atr_value = _safe_float(atr) or _latest_atr(d)
        if not atr_value or atr_value <= 0:
            result["reason"] = "ATR unavailable for displacement check."
            return result

        scan = d.tail(max(3, int(lookback or 12)))

        best = None

        for pos in range(len(scan)):
            row = scan.iloc[pos]

            o = _safe_float(row.get("Open"))
            h = _safe_float(row.get("High"))
            l = _safe_float(row.get("Low"))
            c = _safe_float(row.get("Close"))

            if None in [o, h, l, c]:
                continue

            candle_range = max(h - l, 1e-12)
            body = abs(c - o)
            body_ratio = body / candle_range
            range_atr = candle_range / atr_value

            if body_ratio >= 0.62 and range_atr >= 1.35:
                direction = "BULLISH" if c > o else "BEARISH"

                candidate = {
                    "direction": direction,
                    "range_atr": round(range_atr, 3),
                    "body_ratio": round(body_ratio, 3),
                    "age_candles": (len(scan) - 1) - pos,
                    "score": (range_atr * 10) + (body_ratio * 20),
                }

                if best is None or candidate["score"] > best["score"]:
                    best = candidate

        if not best:
            result["reason"] = "No recent displacement candle found."
            return result

        result.update({
            "detected": True,
            "direction": best["direction"],
            "range_atr": best["range_atr"],
            "body_ratio": best["body_ratio"],
            "age_candles": best["age_candles"],
            "reason": (
                f"Recent {best['direction'].lower()} displacement candle detected "
                f"with range {best['range_atr']} ATR and body ratio {best['body_ratio']}."
            ),
        })

        return result

    except Exception as e:
        result["reason"] = f"Displacement check failed: {e}"
        return result


def _make_gap(
    *,
    gap_type: str,
    bottom: float,
    top: float,
    origin_index: int,
    origin_time: str,
    atr: Optional[float],
) -> Dict[str, Any]:
    size = max(top - bottom, 0.0)
    atr_value = _safe_float(atr)

    return {
        "type": gap_type,
        "bottom": bottom,
        "top": top,
        "mid": (bottom + top) / 2.0,
        "size": size,
        "size_atr": round(size / atr_value, 3) if atr_value and atr_value > 0 else None,
        "origin_index": origin_index,
        "origin_time": origin_time,
        "age_candles": None,
        "is_mitigated": False,
        "is_partially_filled": False,
        "remaining_bottom": bottom,
        "remaining_top": top,
        "fill_pct": 0.0,
    }


def _find_fvgs(
    df: pd.DataFrame,
    *,
    atr: Optional[float],
    lookback: int = 200,
    include_mitigated: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fair Value Gap / Imbalance detection using 3-candle structure.

    Bullish FVG:
      candle1.High < candle3.Low
      zone = candle1.High -> candle3.Low

    Bearish FVG:
      candle1.Low > candle3.High
      zone = candle3.High -> candle1.Low

    Notes:
    - This is not used as direct entry signal.
    - It identifies unfilled/partially-filled imbalance zones that can act as magnets/risk zones.
    """
    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < 5:
            return []

        d = d.tail(max(10, int(lookback or 200))).copy()
        atr_value = _safe_float(atr) or _latest_atr(d)

        gaps: List[Dict[str, Any]] = []

        # i is third candle in the 3-candle FVG pattern
        for i in range(2, len(d)):
            c1 = d.iloc[i - 2]
            c3 = d.iloc[i]

            c1_high = _safe_float(c1.get("High"))
            c1_low = _safe_float(c1.get("Low"))
            c3_high = _safe_float(c3.get("High"))
            c3_low = _safe_float(c3.get("Low"))

            if None in [c1_high, c1_low, c3_high, c3_low]:
                continue

            # Bullish FVG / upward imbalance
            if c1_high < c3_low:
                gap = _make_gap(
                    gap_type="BULLISH",
                    bottom=c1_high,
                    top=c3_low,
                    origin_index=i,
                    origin_time=_candle_time(d, i),
                    atr=atr_value,
                )
                gaps.append(gap)

            # Bearish FVG / downward imbalance
            if c1_low > c3_high:
                gap = _make_gap(
                    gap_type="BEARISH",
                    bottom=c3_high,
                    top=c1_low,
                    origin_index=i,
                    origin_time=_candle_time(d, i),
                    atr=atr_value,
                )
                gaps.append(gap)

        # Mitigation / fill check
        final_gaps: List[Dict[str, Any]] = []

        for gap in gaps:
            origin_index = int(gap.get("origin_index") or 0)
            gtype = str(gap.get("type") or "").upper()
            bottom = _safe_float(gap.get("bottom"))
            top = _safe_float(gap.get("top"))

            if bottom is None or top is None or top <= bottom:
                continue

            after = d.iloc[origin_index + 1:]
            age = max(0, len(d) - 1 - origin_index)
            gap["age_candles"] = age

            is_mitigated = False
            is_partially_filled = False
            remaining_bottom = bottom
            remaining_top = top
            fill_pct = 0.0

            if not after.empty:
                if gtype == "BULLISH":
                    # Bullish gap is fully mitigated if later low trades down through bottom.
                    min_low = _safe_float(after["Low"].min())

                    if min_low is not None:
                        if min_low <= bottom:
                            is_mitigated = True
                            fill_pct = 100.0
                        elif min_low < top:
                            is_partially_filled = True
                            remaining_top = min_low
                            filled_amount = top - min_low
                            fill_pct = max(0.0, min(100.0, filled_amount / (top - bottom) * 100.0))

                elif gtype == "BEARISH":
                    # Bearish gap is fully mitigated if later high trades up through top.
                    max_high = _safe_float(after["High"].max())

                    if max_high is not None:
                        if max_high >= top:
                            is_mitigated = True
                            fill_pct = 100.0
                        elif max_high > bottom:
                            is_partially_filled = True
                            remaining_bottom = max_high
                            filled_amount = max_high - bottom
                            fill_pct = max(0.0, min(100.0, filled_amount / (top - bottom) * 100.0))

            gap["is_mitigated"] = bool(is_mitigated)
            gap["is_partially_filled"] = bool(is_partially_filled)
            gap["remaining_bottom"] = remaining_bottom
            gap["remaining_top"] = remaining_top
            gap["fill_pct"] = round(fill_pct, 2)

            remaining_size = max(
                remaining_top
                - remaining_bottom,
                0.0,
            )

            remaining_size_atr = (
                remaining_size
                / atr_value
                if (
                    atr_value is not None
                    and atr_value > 0
                )
                else None
            )

            gap["remaining_size"] = (
                remaining_size
            )

            gap["remaining_size_atr"] = (
                round(
                    remaining_size_atr,
                    3,
                )
                if remaining_size_atr is not None
                else None
            )

            gap["unfilled_pct"] = round(
                max(
                    0.0,
                    min(
                        100.0,
                        100.0 - fill_pct,
                    ),
                ),
                2,
            )

            if include_mitigated or not is_mitigated:
                final_gaps.append(gap)

        # Recent + large gaps first
        final_gaps.sort(
            key=lambda g: (
                _safe_float(g.get("is_mitigated"), 0) or 0,
                _safe_float(g.get("age_candles"), 999999) or 999999,
                -(_safe_float(g.get("size_atr"), 0) or 0),
            )
        )

        return final_gaps

    except Exception:
        return []


def _gap_distance_from_entry(
    *,
    side: str,
    entry: float,
    gap: Dict[str, Any],
) -> Optional[float]:
    try:
        bottom = _safe_float(gap.get("remaining_bottom") or gap.get("bottom"))
        top = _safe_float(gap.get("remaining_top") or gap.get("top"))

        if bottom is None or top is None:
            return None

        if side == "LONG":
            # Downside draw risk: bullish FVG below entry
            if top <= entry:
                return entry - top

        if side == "SHORT":
            # Upside draw risk: bearish FVG above entry
            if bottom >= entry:
                return bottom - entry

        return None

    except Exception:
        return None


def _is_gap_between_entry_and_stop(
    *,
    side: str,
    entry: float,
    stop: Optional[float],
    gap: Dict[str, Any],
) -> bool:
    try:
        if stop is None:
            return False

        bottom = _safe_float(gap.get("remaining_bottom") or gap.get("bottom"))
        top = _safe_float(gap.get("remaining_top") or gap.get("top"))

        if bottom is None or top is None:
            return False

        if side == "LONG":
            # LONG stop usually below entry. Risk gap below entry and above/before stop.
            low_bound = min(stop, entry)
            high_bound = max(stop, entry)
            return bottom >= low_bound and top <= high_bound

        if side == "SHORT":
            # SHORT stop usually above entry. Risk gap above entry and below/before stop.
            low_bound = min(entry, stop)
            high_bound = max(entry, stop)
            return bottom >= low_bound and top <= high_bound

        return False

    except Exception:
        return False


def _find_opposing_gap(
    *,
    side: str,
    entry: float,
    stop: Optional[float],
    gaps: List[Dict[str, Any]],
    atr: Optional[float],
) -> Optional[Dict[str, Any]]:
    """
    For LONG:
      downside risk usually comes from unfilled bullish FVG below entry.

    For SHORT:
      upside risk usually comes from unfilled bearish FVG above entry.
    """
    try:
        side = _normalize_side(side)

        candidates = []

        for gap in gaps or []:
            if gap.get("is_mitigated"):
                continue

            gtype = _upper(gap.get("type"))

            if side == "LONG" and gtype != "BULLISH":
                continue

            if side == "SHORT" and gtype != "BEARISH":
                continue

            distance = _gap_distance_from_entry(
                side=side,
                entry=entry,
                gap=gap,
            )

            if distance is None:
                continue

            atr_value = _safe_float(atr)
            distance_atr = distance / atr_value if atr_value and atr_value > 0 else None
            between_entry_stop = _is_gap_between_entry_and_stop(
                side=side,
                entry=entry,
                stop=stop,
                gap=gap,
            )

            candidate = dict(gap)
            candidate["distance_from_entry"] = distance
            candidate["distance_from_entry_atr"] = round(distance_atr, 3) if distance_atr is not None else None
            candidate["between_entry_and_stop"] = bool(between_entry_stop)

            candidates.append(candidate)

        if not candidates:
            return None

        # Prefer nearest gap between entry and stop. Otherwise nearest by ATR/distance.
        candidates.sort(
            key=lambda g: (
                0 if g.get("between_entry_and_stop") else 1,
                _safe_float(g.get("distance_from_entry_atr"), 999999) or 999999,
                _safe_float(g.get("age_candles"), 999999) or 999999,
            )
        )

        return candidates[0]

    except Exception:
        return None


def _build_imbalance_high_risk_gate(
    *,
    side: str,
    entry: float,
    stop: Optional[float],
    atr: Optional[float],
    opposing_gap: Optional[Dict[str, Any]],
    active_adverse_displacement: bool,
    active_supportive_displacement: bool,
) -> Dict[str, Any]:
    """
    Confirm whether a HIGH imbalance label is justified.

    An FVG being somewhere inside the Entry-SL range is
    location evidence only. HIGH requires at least two
    independent major risk factors:

    1. a meaningful remaining FVG,
    2. critical SL geometry,
    3. active adverse displacement.
    """
    result = {
        "meaningful_remaining_fvg": False,
        "unsafe_stop_geometry": False,
        "stop_relation": "UNKNOWN",
        "stop_buffer_atr": None,
        "remaining_size_atr": None,
        "unfilled_pct": None,
        "major_confluence_count": 0,
        "high_risk_confirmed": False,
    }

    try:
        if not opposing_gap:
            return result

        normalized_side = _normalize_side(
            side
        )

        stop_value = _safe_float(
            stop
        )

        atr_value = _safe_float(
            atr
        )

        remaining_bottom = _safe_float(
            opposing_gap.get(
                "remaining_bottom"
            )
        )

        if remaining_bottom is None:
            remaining_bottom = _safe_float(
                opposing_gap.get("bottom")
            )

        remaining_top = _safe_float(
            opposing_gap.get(
                "remaining_top"
            )
        )

        if remaining_top is None:
            remaining_top = _safe_float(
                opposing_gap.get("top")
            )

        remaining_size_atr = _safe_float(
            opposing_gap.get(
                "remaining_size_atr"
            )
        )

        if (
            remaining_size_atr is None
            and remaining_bottom is not None
            and remaining_top is not None
            and atr_value is not None
            and atr_value > 0
        ):
            remaining_size_atr = (
                max(
                    remaining_top
                    - remaining_bottom,
                    0.0,
                )
                / atr_value
            )

        fill_pct = (
            _safe_float(
                opposing_gap.get(
                    "fill_pct"
                ),
                0.0,
            )
            or 0.0
        )

        unfilled_pct = _safe_float(
            opposing_gap.get(
                "unfilled_pct"
            )
        )

        if unfilled_pct is None:
            unfilled_pct = max(
                0.0,
                min(
                    100.0,
                    100.0 - fill_pct,
                ),
            )

        meaningful_remaining_fvg = bool(
            remaining_size_atr is not None
            and remaining_size_atr >= 0.35
            and unfilled_pct >= 40.0
        )

        unsafe_stop_geometry = False
        stop_relation = "UNKNOWN"
        stop_buffer_atr = None

        valid_stop_geometry = bool(
            stop_value is not None
            and remaining_bottom is not None
            and remaining_top is not None
            and atr_value is not None
            and atr_value > 0
            and (
                (
                    normalized_side == "LONG"
                    and stop_value < entry
                )
                or (
                    normalized_side == "SHORT"
                    and stop_value > entry
                )
            )
        )

        if valid_stop_geometry:
            if normalized_side == "LONG":
                signed_stop_buffer = (
                    remaining_bottom
                    - stop_value
                )
            else:
                signed_stop_buffer = (
                    stop_value
                    - remaining_top
                )

            stop_buffer_atr = (
                signed_stop_buffer
                / atr_value
            )

            comparison_buffer_atr = round(
                stop_buffer_atr,
                8,
            )

            if comparison_buffer_atr < 0:
                if abs(
                    comparison_buffer_atr
                ) <= 0.50:
                    stop_relation = (
                        "FVG_CROSSES_OR_SITS_JUST_BEYOND_STOP"
                    )
                    unsafe_stop_geometry = True
                else:
                    stop_relation = (
                        "FVG_FAR_BEYOND_STOP"
                    )

            elif comparison_buffer_atr <= 0.20:
                stop_relation = "CRITICAL_BUFFER"
                unsafe_stop_geometry = True

            elif comparison_buffer_atr < 0.50:
                stop_relation = "THIN_BUFFER"

            else:
                stop_relation = "SAFE_BUFFER"

        major_confluence_count = sum(
            [
                bool(
                    meaningful_remaining_fvg
                ),
                bool(
                    unsafe_stop_geometry
                ),
                bool(
                    active_adverse_displacement
                ),
            ]
        )

        high_risk_confirmed = bool(
            major_confluence_count >= 2
            and not active_supportive_displacement
        )

        result.update(
            {
                "meaningful_remaining_fvg": bool(
                    meaningful_remaining_fvg
                ),
                "unsafe_stop_geometry": bool(
                    unsafe_stop_geometry
                ),
                "stop_relation": stop_relation,
                "stop_buffer_atr": (
                    round(
                        stop_buffer_atr,
                        4,
                    )
                    if stop_buffer_atr is not None
                    else None
                ),
                "remaining_size_atr": (
                    round(
                        remaining_size_atr,
                        4,
                    )
                    if remaining_size_atr is not None
                    else None
                ),
                "unfilled_pct": round(
                    unfilled_pct,
                    2,
                ),
                "major_confluence_count": int(
                    major_confluence_count
                ),
                "high_risk_confirmed": bool(
                    high_risk_confirmed
                ),
            }
        )

        return result

    except Exception:
        return result


def _risk_level_from_score(
    score: int,
    *,
    high_risk_confirmed: bool = False,
) -> str:
    if (
        score >= 65
        and high_risk_confirmed
    ):
        return "HIGH"

    if score >= 38:
        return "MEDIUM"

    return "LOW"


def _build_advice(
    *,
    side: str,
    risk_level: str,
    opposing_gap: Optional[Dict[str, Any]],
    displacement: Dict[str, Any],
) -> str:
    if risk_level == "LOW":
        return (
            "No critical imbalance-fill risk was found near the planned entry/stop path. "
            "Use existing confirmation and risk management rules."
        )

    if side == "SHORT":
        if opposing_gap:
            return (
                "SHORT setup has upside imbalance-fill risk. Price may continue higher to mitigate the bearish FVG before falling. "
                "Avoid blind short execution; prefer a higher retest/rejection or reduced risk."
            )

        if displacement.get("direction") == "BULLISH":
            return (
                "SHORT setup is facing recent bullish displacement pressure. "
                "Avoid adding into strength; wait for bearish rejection or use reduced risk."
            )

    if side == "LONG":
        if opposing_gap:
            return (
                "LONG setup has downside imbalance-fill risk. Price may continue lower to mitigate the bullish FVG before bouncing. "
                "Avoid blind long execution; prefer a lower retest/rejection or reduced risk."
            )

        if displacement.get("direction") == "BEARISH":
            return (
                "LONG setup is facing recent bearish displacement pressure. "
                "Avoid adding into weakness; wait for bullish rejection or use reduced risk."
            )

    return (
        "Imbalance risk is elevated. Treat this setup as conditional until price confirms around the entry zone."
    )


def evaluate_imbalance_risk(
    *,
    side: str,
    entry_price: float,
    stop_loss: Optional[float] = None,
    user_timeframe: str = "",
    df_analysis: Optional[pd.DataFrame] = None,
    atr: Optional[float] = None,
    current_price: Optional[float] = None,
    lookback_candles: int = 200,
) -> Dict[str, Any]:
    """
    Evaluate imbalance/FVG risk for a planned setup.

    Important:
    - This function does NOT fetch candles.
    - This function does NOT modify entry, SL, or TP.
    - This function does NOT block a signal.
    - It only returns structured imbalance risk context.

    Intended use:
    - Fresh setup: risk modifier / conditional status.
    - Running trade: active danger warning / reduce-risk signal.
    """
    try:
        normalized_side = _normalize_side(side)
        entry = _safe_float(entry_price)
        stop = _safe_float(stop_loss)
        current = _safe_float(current_price)

        d = _normalize_ohlc(df_analysis)

        if normalized_side not in ["LONG", "SHORT"] or entry is None:
            return {
                "enabled": True,
                "risk_level": "UNKNOWN",
                "imbalance_score": 0,
                "imbalance_risk_detected": False,
                "opposing_imbalance_risk": False,
                "nearest_opposing_gap": None,
                "unfilled_gaps": [],
                "displacement": {},
                "suggest_entry_adjustment": False,
                "suggest_sl_buffer_atr": 0.0,
                "suggested_entry_zone": None,
                "advice": "Imbalance risk unavailable because side or entry is missing.",
                "reason": "Invalid side/entry.",
                "signals": {},
            }

        if d.empty:
            return {
                "enabled": True,
                "risk_level": "UNKNOWN",
                "imbalance_score": 0,
                "imbalance_risk_detected": False,
                "opposing_imbalance_risk": False,
                "nearest_opposing_gap": None,
                "unfilled_gaps": [],
                "displacement": {},
                "suggest_entry_adjustment": False,
                "suggest_sl_buffer_atr": 0.0,
                "suggested_entry_zone": None,
                "advice": "Imbalance risk unavailable because candle data is missing.",
                "reason": "No usable OHLC candles.",
                "signals": {},
            }

        atr_value = _latest_atr(d, fallback=atr)
        gaps = _find_fvgs(
            d,
            atr=atr_value,
            lookback=lookback_candles,
            include_mitigated=False,
        )

        displacement = _detect_displacement(
            d,
            atr=atr_value,
            lookback=12,
        )

        opposing_gap = _find_opposing_gap(
            side=normalized_side,
            entry=entry,
            stop=stop,
            gaps=gaps,
            atr=atr_value,
        )

        score = 10
        reasons: List[str] = []

        if opposing_gap:
            score += 25
            reasons.append("opposing unfilled FVG exists near the setup path")

            remaining_size_atr = (
                _safe_float(
                    opposing_gap.get(
                        "remaining_size_atr"
                    ),
                    0,
                )
                or 0
            )
            distance_atr = _safe_float(opposing_gap.get("distance_from_entry_atr"), 999) or 999
            age = int(_safe_float(opposing_gap.get("age_candles"), 999) or 999)

            if remaining_size_atr >= 0.75:
                score += 15
                reasons.append(
                    "remaining opposing FVG is large relative to ATR"
                )

            elif remaining_size_atr >= 0.35:
                score += 8
                reasons.append(
                    "remaining opposing FVG has moderate ATR size"
                )

            if distance_atr <= 0.8:
                score += 15
                reasons.append("opposing FVG is very close to entry")
            elif distance_atr <= 1.6:
                score += 10
                reasons.append("opposing FVG is within nearby ATR range")

            if opposing_gap.get("between_entry_and_stop"):
                score += 15
                reasons.append("opposing FVG sits between entry and stop/invalidation")

            if age <= 20:
                score += 8
                reasons.append(
                    "opposing FVG is recent"
                )

            fill_pct = (
                _safe_float(
                    opposing_gap.get(
                        "fill_pct"
                    ),
                    0,
                )
                or 0
            )

            if fill_pct >= 75:
                score -= 12
                reasons.append(
                    "opposing FVG is mostly filled, "
                    "so remaining risk is reduced"
                )

            elif fill_pct >= 50:
                score -= 6
                reasons.append(
                    "opposing FVG is partially filled, "
                    "so remaining risk is reduced"
                )

        disp_dir = _upper(
            displacement.get("direction")
        )

        displacement_age_value = _safe_float(
            displacement.get(
                "age_candles"
            )
        )

        displacement_age = (
            int(
                displacement_age_value
            )
            if displacement_age_value is not None
            else 999999
        )

        adverse_displacement = bool(
            (
                normalized_side == "SHORT"
                and disp_dir == "BULLISH"
            )
            or (
                normalized_side == "LONG"
                and disp_dir == "BEARISH"
            )
        )

        supportive_displacement = bool(
            (
                normalized_side == "SHORT"
                and disp_dir == "BEARISH"
            )
            or (
                normalized_side == "LONG"
                and disp_dir == "BULLISH"
            )
        )

        active_adverse_displacement = bool(
            adverse_displacement
            and displacement_age <= 6
        )

        active_supportive_displacement = bool(
            supportive_displacement
            and displacement_age <= 6
        )

        if adverse_displacement:
            if displacement_age <= 3:
                score += 25

            elif displacement_age <= 6:
                score += 18

            else:
                score += 10

            reasons.append(
                "recent displacement direction is against the setup"
            )

        if supportive_displacement:
            if displacement_age <= 3:
                score -= 12

            elif displacement_age <= 6:
                score -= 8

            else:
                score -= 4

            reasons.append(
                "recent displacement supports the setup direction"
            )

        # Current price already moving against the idea.
        if current is not None:
            if normalized_side == "SHORT" and current > entry:
                score += 8
                reasons.append("current price is above short entry, showing adverse pressure")

            if normalized_side == "LONG" and current < entry:
                score += 8
                reasons.append("current price is below long entry, showing adverse pressure")

        raw_score = max(
            0,
            min(
                100,
                int(score),
            ),
        )

        high_risk_gate = (
            _build_imbalance_high_risk_gate(
                side=normalized_side,
                entry=entry,
                stop=stop,
                atr=atr_value,
                opposing_gap=opposing_gap,
                active_adverse_displacement=(
                    active_adverse_displacement
                ),
                active_supportive_displacement=(
                    active_supportive_displacement
                ),
            )
        )

        high_risk_confirmed = bool(
            high_risk_gate.get(
                "high_risk_confirmed"
            )
        )

        # Without two independent major confirmations,
        # a HIGH raw score is capped at MEDIUM.
        score = (
            raw_score
            if high_risk_confirmed
            else min(
                raw_score,
                64,
            )
        )

        if (
            raw_score >= 65
            and not high_risk_confirmed
        ):
            reasons.append(
                "HIGH imbalance risk was not confirmed because "
                "fewer than two independent major risk factors exist"
            )

        risk_level = _risk_level_from_score(
            score,
            high_risk_confirmed=(
                high_risk_confirmed
            ),
        )

        risk_detected = risk_level in ["MEDIUM", "HIGH"]
        high_risk = risk_level == "HIGH"

        suggested_entry_zone = None

        if opposing_gap:
            bottom = _safe_float(opposing_gap.get("remaining_bottom") or opposing_gap.get("bottom"))
            top = _safe_float(opposing_gap.get("remaining_top") or opposing_gap.get("top"))

            if bottom is not None and top is not None:
                mid = (bottom + top) / 2.0

                if normalized_side == "SHORT":
                    suggested_entry_zone = {
                        "type": "HIGHER_RETEST_FOR_SHORT",
                        "price": round(mid, 8),
                        "zone_bottom": round(bottom, 8),
                        "zone_top": round(top, 8),
                        "note": "For SHORT, safer entry may be closer to bearish FVG fill/rejection zone.",
                    }

                elif normalized_side == "LONG":
                    suggested_entry_zone = {
                        "type": "LOWER_RETEST_FOR_LONG",
                        "price": round(mid, 8),
                        "zone_bottom": round(bottom, 8),
                        "zone_top": round(top, 8),
                        "note": "For LONG, safer entry may be closer to bullish FVG fill/rejection zone.",
                    }

        sl_buffer = 0.0
        if high_risk:
            sl_buffer = 0.45
        elif risk_level == "MEDIUM":
            sl_buffer = 0.20

        advice = _build_advice(
            side=normalized_side,
            risk_level=risk_level,
            opposing_gap=opposing_gap,
            displacement=displacement,
        )

        return {
            "enabled": True,
            "user_timeframe": str(user_timeframe or "").lower().strip(),
            "side": normalized_side,
            "risk_level": risk_level,
            "imbalance_score": score,
            "raw_imbalance_score": raw_score,
            "imbalance_risk_detected": bool(
                risk_detected
            ),
            "opposing_imbalance_risk": bool(
                opposing_gap is not None
            ),
            "high_risk_confirmed": bool(
                high_risk_confirmed
            ),
            "high_risk_gate": high_risk_gate,
            "nearest_opposing_gap": opposing_gap,
            "unfilled_gaps": gaps[:20],
            "displacement": displacement,
            "suggest_entry_adjustment": bool(high_risk and suggested_entry_zone is not None),
            "suggest_sl_buffer_atr": sl_buffer,
            "suggested_entry_zone": suggested_entry_zone,
            "advice": advice,
            "reason": "; ".join(reasons) if reasons else "No critical opposing imbalance or displacement risk found.",
            "signals": {
                "unfilled_gap_count": len(gaps),
                "has_opposing_gap": opposing_gap is not None,
                "opposing_gap_between_entry_stop": bool(opposing_gap and opposing_gap.get("between_entry_and_stop")),
                "opposing_gap_distance_atr": opposing_gap.get("distance_from_entry_atr") if opposing_gap else None,
                "remaining_fvg_size_atr": (
                    high_risk_gate.get(
                        "remaining_size_atr"
                    )
                ),
                "unfilled_fvg_pct": (
                    high_risk_gate.get(
                        "unfilled_pct"
                    )
                ),
                "meaningful_remaining_fvg": bool(
                    high_risk_gate.get(
                        "meaningful_remaining_fvg"
                    )
                ),
                "unsafe_stop_geometry": bool(
                    high_risk_gate.get(
                        "unsafe_stop_geometry"
                    )
                ),
                "imbalance_stop_relation": (
                    high_risk_gate.get(
                        "stop_relation"
                    )
                ),
                "imbalance_stop_buffer_atr": (
                    high_risk_gate.get(
                        "stop_buffer_atr"
                    )
                ),
                "major_confluence_count": int(
                    high_risk_gate.get(
                        "major_confluence_count",
                        0,
                    )
                    or 0
                ),
                "high_risk_confirmed": bool(
                    high_risk_confirmed
                ),
                "active_adverse_displacement": bool(
                    active_adverse_displacement
                ),
                "active_supportive_displacement": bool(
                    active_supportive_displacement
                ),
                "recent_displacement_detected": bool(displacement.get("detected")),
                "recent_displacement_direction": displacement.get("direction"),
                "atr": atr_value,
                "current_price": current,
            },
        }

    except Exception as e:
        return {
            "enabled": True,
            "risk_level": "UNKNOWN",
            "imbalance_score": 0,
            "imbalance_risk_detected": False,
            "opposing_imbalance_risk": False,
            "nearest_opposing_gap": None,
            "unfilled_gaps": [],
            "displacement": {},
            "suggest_entry_adjustment": False,
            "suggest_sl_buffer_atr": 0.0,
            "suggested_entry_zone": None,
            "advice": "Imbalance risk evaluation failed.",
            "reason": str(e),
            "signals": {},
        }


def add_imbalance_to_scenario(
    scenario: Dict[str, Any],
    interval: str,
    df_analysis: Optional[pd.DataFrame],
    df_lower: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Decorate a scenario with imbalance risk.

    This is response-only / advisory by default.
    It does not modify entry, SL, or TP.
    """
    try:
        if not isinstance(scenario, dict):
            return scenario

        side = _normalize_side(
            scenario.get("trade_type")
            or scenario.get("side")
        )

        if side not in ["LONG", "SHORT"]:
            return scenario

        entry = _extract_price(
            scenario.get("entry")
            or scenario.get("entry_zone")
        )

        stop = _extract_price(
            scenario.get("stop_loss")
            or scenario.get("invalidation")
        )

        if entry is None:
            return scenario

        raw_values = scenario.get("raw_values") or {}
        atr = _safe_float(raw_values.get("atr") or scenario.get("atr"))
        current_price = _safe_float(raw_values.get("current_price"))

        # Prefer main analysis timeframe.
        # df_lower can be used later for more precise execution imbalance.
        analysis_df = df_analysis

        scenario["imbalance_risk"] = evaluate_imbalance_risk(
            side=side,
            entry_price=entry,
            stop_loss=stop,
            user_timeframe=interval,
            df_analysis=analysis_df,
            atr=atr,
            current_price=current_price,
        )

    except Exception as e:
        try:
            scenario["imbalance_risk"] = {
                "enabled": True,
                "risk_level": "UNKNOWN",
                "imbalance_score": 0,
                "imbalance_risk_detected": False,
                "opposing_imbalance_risk": False,
                "nearest_opposing_gap": None,
                "unfilled_gaps": [],
                "displacement": {},
                "suggest_entry_adjustment": False,
                "suggest_sl_buffer_atr": 0.0,
                "suggested_entry_zone": None,
                "advice": "Imbalance risk unavailable because scenario decoration failed.",
                "reason": str(e),
                "signals": {},
            }
        except Exception:
            pass

    return scenario


def add_imbalance_to_ai_data(
    ai_data: Dict[str, Any],
    interval: str,
    df_analysis: Optional[pd.DataFrame],
    df_lower: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Attach imbalance_risk to the first trade scenario and selected_scenario.
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

        scenario = add_imbalance_to_scenario(
            scenario=scenario,
            interval=interval,
            df_analysis=df_analysis,
            df_lower=df_lower,
        )

        scenarios[0] = scenario
        ai_data["trade_scenarios"] = scenarios

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            selected_side = _normalize_side(
                selected.get("trade_type")
                or selected.get("side")
            )
            scenario_side = _normalize_side(
                scenario.get("trade_type")
                or scenario.get("side")
            )

            selected_entry = _extract_price(
                selected.get("entry")
                or selected.get("entry_zone")
            )
            scenario_entry = _extract_price(
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

    except Exception as e:
        print(f"[IMBALANCE_RISK] attach skipped/error: {e}")

    return ai_data