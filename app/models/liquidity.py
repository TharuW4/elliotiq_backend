import math
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Safe helpers
# ============================================================

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


def _normalize_side(value: Any) -> str:
    side = _upper(value)

    if "LONG" in side or side in ["BUY", "BULL", "BULLISH"]:
        return "LONG"

    if "SHORT" in side or side in ["SELL", "BEAR", "BEARISH"]:
        return "SHORT"

    return "UNKNOWN"


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


def _normalize_ohlc(df: Any) -> pd.DataFrame:
    """
    Normalize OHLC dataframe into:
    Open, High, Low, Close, Volume(optional)

    This module never throws to caller.
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
            columns={c: rename_map.get(str(c).lower().strip(), c) for c in out.columns}
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

        clean = atr_s.dropna()
        if clean.empty:
            return None

        val = _safe_float(clean.iloc[-1])
        return val if val and val > 0 else None
    except Exception:
        return None


def _candle_time(df: pd.DataFrame, idx: int) -> str:
    try:
        return str(df.index[idx])
    except Exception:
        return ""


def _pct_distance(a: float, b: float) -> float:
    try:
        if b == 0:
            return 0.0
        return abs(a - b) / abs(b) * 100.0
    except Exception:
        return 0.0


# ============================================================
# Swing / Equal liquidity detection
# ============================================================

def _find_swing_points(
    df: pd.DataFrame,
    *,
    left: int = 2,
    right: int = 2,
    lookback: int = 250,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect local swing highs/lows.

    Swing high:
      high[i] > highs around it

    Swing low:
      low[i] < lows around it
    """
    result = {
        "swing_highs": [],
        "swing_lows": [],
    }

    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < (left + right + 5):
            return result

        d = d.tail(max(20, int(lookback or 250))).copy()

        highs: List[Dict[str, Any]] = []
        lows: List[Dict[str, Any]] = []

        for i in range(left, len(d) - right):
            row = d.iloc[i]
            h = _safe_float(row.get("High"))
            l = _safe_float(row.get("Low"))

            if h is None or l is None:
                continue

            left_high = d.iloc[i - left:i]["High"].max()
            right_high = d.iloc[i + 1:i + 1 + right]["High"].max()

            left_low = d.iloc[i - left:i]["Low"].min()
            right_low = d.iloc[i + 1:i + 1 + right]["Low"].min()

            if h > left_high and h > right_high:
                highs.append({
                    "type": "SWING_HIGH",
                    "price": float(h),
                    "index": i,
                    "time": _candle_time(d, i),
                    "age_candles": len(d) - 1 - i,
                    "strength": 1,
                })

            if l < left_low and l < right_low:
                lows.append({
                    "type": "SWING_LOW",
                    "price": float(l),
                    "index": i,
                    "time": _candle_time(d, i),
                    "age_candles": len(d) - 1 - i,
                    "strength": 1,
                })

        result["swing_highs"] = highs
        result["swing_lows"] = lows
        return result

    except Exception:
        return result


def _cluster_equal_levels(
    levels: List[Dict[str, Any]],
    *,
    tolerance_pct: float = 0.18,
    max_age: int = 180,
) -> List[Dict[str, Any]]:
    """
    Group equal highs / equal lows.

    Equal highs/lows are important because stops/liquidity often collect there.
    """
    try:
        clean = [
            dict(x)
            for x in levels or []
            if _safe_float(x.get("price")) is not None
            and int(_safe_float(x.get("age_candles"), 999999) or 999999) <= max_age
        ]

        if not clean:
            return []

        clean.sort(key=lambda x: _safe_float(x.get("price")) or 0)

        clusters: List[List[Dict[str, Any]]] = []

        for lvl in clean:
            price = _safe_float(lvl.get("price"))
            if price is None:
                continue

            placed = False

            for cluster in clusters:
                avg_price = sum(_safe_float(x.get("price"), 0) or 0 for x in cluster) / max(len(cluster), 1)
                if _pct_distance(price, avg_price) <= tolerance_pct:
                    cluster.append(lvl)
                    placed = True
                    break

            if not placed:
                clusters.append([lvl])

        out = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            prices = [_safe_float(x.get("price"), 0) or 0 for x in cluster]
            ages = [int(_safe_float(x.get("age_candles"), 999999) or 999999) for x in cluster]

            out.append({
                "price": sum(prices) / len(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "touches": len(cluster),
                "age_candles": min(ages),
                "members": cluster[:8],
                "strength": min(5, len(cluster)),
            })

        out.sort(key=lambda x: (-int(x.get("touches") or 0), int(x.get("age_candles") or 999999)))
        return out

    except Exception:
        return []


def _is_level_swept_after_origin(
    *,
    df: pd.DataFrame,
    level_price: float,
    level_type: str,
    origin_age: int,
) -> bool:
    """
    Check if a liquidity level was already swept after it formed.

    For highs:
      later high > level

    For lows:
      later low < level
    """
    try:
        d = _normalize_ohlc(df)
        if d.empty:
            return False

        origin_age = int(origin_age or 0)
        start_idx = max(0, len(d) - origin_age)
        after = d.iloc[start_idx:]

        if after.empty:
            return False

        level_type = _upper(level_type)

        if level_type == "HIGH":
            return bool(after["High"].max() > level_price)

        if level_type == "LOW":
            return bool(after["Low"].min() < level_price)

        return False

    except Exception:
        return False


def _build_liquidity_levels(
    df: pd.DataFrame,
    *,
    atr: Optional[float],
    current_price: Optional[float],
    lookback: int = 250,
) -> Dict[str, Any]:
    """
    Build unswept liquidity pools:
    - swing highs above/below
    - swing lows above/below
    - equal highs/equal lows clusters
    """
    result = {
        "swing_high_liquidity": [],
        "swing_low_liquidity": [],
        "equal_high_liquidity": [],
        "equal_low_liquidity": [],
    }

    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < 20:
            return result

        atr_value = _safe_float(atr) or _latest_atr(d)
        current = _safe_float(current_price)

        swings = _find_swing_points(d, lookback=lookback)
        swing_highs = swings.get("swing_highs") or []
        swing_lows = swings.get("swing_lows") or []

        equal_highs = _cluster_equal_levels(swing_highs, tolerance_pct=0.18, max_age=min(220, lookback))
        equal_lows = _cluster_equal_levels(swing_lows, tolerance_pct=0.18, max_age=min(220, lookback))

        # Single swing liquidity levels
        high_levels = []
        for h in swing_highs:
            price = _safe_float(h.get("price"))
            age = int(_safe_float(h.get("age_candles"), 999999) or 999999)
            if price is None:
                continue

            if _is_level_swept_after_origin(df=d, level_price=price, level_type="HIGH", origin_age=age):
                continue

            distance = abs(price - current) if current is not None else None
            distance_atr = distance / atr_value if distance is not None and atr_value and atr_value > 0 else None

            high_levels.append({
                "type": "SWING_HIGH_LIQUIDITY",
                "price": price,
                "age_candles": age,
                "strength": h.get("strength", 1),
                "distance_from_current": distance,
                "distance_from_current_atr": round(distance_atr, 3) if distance_atr is not None else None,
                "time": h.get("time", ""),
            })

        low_levels = []
        for l in swing_lows:
            price = _safe_float(l.get("price"))
            age = int(_safe_float(l.get("age_candles"), 999999) or 999999)
            if price is None:
                continue

            if _is_level_swept_after_origin(df=d, level_price=price, level_type="LOW", origin_age=age):
                continue

            distance = abs(price - current) if current is not None else None
            distance_atr = distance / atr_value if distance is not None and atr_value and atr_value > 0 else None

            low_levels.append({
                "type": "SWING_LOW_LIQUIDITY",
                "price": price,
                "age_candles": age,
                "strength": l.get("strength", 1),
                "distance_from_current": distance,
                "distance_from_current_atr": round(distance_atr, 3) if distance_atr is not None else None,
                "time": l.get("time", ""),
            })

        # Equal high clusters
        eq_high_levels = []
        for c in equal_highs:
            price = _safe_float(c.get("price"))
            age = int(_safe_float(c.get("age_candles"), 999999) or 999999)
            if price is None:
                continue

            if _is_level_swept_after_origin(df=d, level_price=price, level_type="HIGH", origin_age=age):
                continue

            distance = abs(price - current) if current is not None else None
            distance_atr = distance / atr_value if distance is not None and atr_value and atr_value > 0 else None

            eq_high_levels.append({
                "type": "EQUAL_HIGH_LIQUIDITY",
                "price": price,
                "min_price": c.get("min_price"),
                "max_price": c.get("max_price"),
                "touches": c.get("touches", 2),
                "age_candles": age,
                "strength": min(5, int(c.get("touches") or 2)),
                "distance_from_current": distance,
                "distance_from_current_atr": round(distance_atr, 3) if distance_atr is not None else None,
                "members": c.get("members", []),
            })

        # Equal low clusters
        eq_low_levels = []
        for c in equal_lows:
            price = _safe_float(c.get("price"))
            age = int(_safe_float(c.get("age_candles"), 999999) or 999999)
            if price is None:
                continue

            if _is_level_swept_after_origin(df=d, level_price=price, level_type="LOW", origin_age=age):
                continue

            distance = abs(price - current) if current is not None else None
            distance_atr = distance / atr_value if distance is not None and atr_value and atr_value > 0 else None

            eq_low_levels.append({
                "type": "EQUAL_LOW_LIQUIDITY",
                "price": price,
                "min_price": c.get("min_price"),
                "max_price": c.get("max_price"),
                "touches": c.get("touches", 2),
                "age_candles": age,
                "strength": min(5, int(c.get("touches") or 2)),
                "distance_from_current": distance,
                "distance_from_current_atr": round(distance_atr, 3) if distance_atr is not None else None,
                "members": c.get("members", []),
            })

        high_levels.sort(key=lambda x: (_safe_float(x.get("distance_from_current_atr"), 999999) or 999999, int(x.get("age_candles") or 999999)))
        low_levels.sort(key=lambda x: (_safe_float(x.get("distance_from_current_atr"), 999999) or 999999, int(x.get("age_candles") or 999999)))
        eq_high_levels.sort(key=lambda x: (_safe_float(x.get("distance_from_current_atr"), 999999) or 999999, -int(x.get("strength") or 0)))
        eq_low_levels.sort(key=lambda x: (_safe_float(x.get("distance_from_current_atr"), 999999) or 999999, -int(x.get("strength") or 0)))

        result["swing_high_liquidity"] = high_levels[:30]
        result["swing_low_liquidity"] = low_levels[:30]
        result["equal_high_liquidity"] = eq_high_levels[:20]
        result["equal_low_liquidity"] = eq_low_levels[:20]

        return result

    except Exception:
        return result


# ============================================================
# Sweep detection
# ============================================================

def _detect_recent_sweep(
    df: pd.DataFrame,
    *,
    atr: Optional[float],
    lookback_levels: int = 160,
    sweep_scan_candles: int = 12,
) -> Dict[str, Any]:
    """
    Detect recent liquidity sweep + rejection.

    Bearish sweep:
      candle high breaks previous liquidity high
      candle closes back below that level

    Bullish sweep:
      candle low breaks previous liquidity low
      candle closes back above that level
    """
    result = {
        "detected": False,
        "direction": "NONE",
        "sweep_type": "NONE",
        "swept_level": None,
        "sweep_candle_age": None,
        "sweep_candle_time": "",
        "sweep_high": None,
        "sweep_low": None,
        "pool_type": "NONE",
        "pool_touches": 0,
        "wick_size_atr": None,
        "close_reclaim": False,
        "bullish_event": {},
        "bearish_event": {},
        "reason": "",
    }

    try:
        d = _normalize_ohlc(df)
        if d.empty or len(d) < 30:
            result["reason"] = "Not enough candles for sweep detection."
            return result

        atr_value = _safe_float(atr) or _latest_atr(d)

        scan_n = max(3, int(sweep_scan_candles or 12))
        start = max(20, len(d) - scan_n)

        best_bullish = None
        best_bearish = None

        def _prefer_event(current, candidate):
            if current is None:
                return candidate

            current_age_value = _safe_float(
                current.get("sweep_candle_age"),
                999999,
            )
            candidate_age_value = _safe_float(
                candidate.get("sweep_candle_age"),
                999999,
            )
            current_age = int(
                999999
                if current_age_value is None
                else current_age_value
            )
            candidate_age = int(
                999999
                if candidate_age_value is None
                else candidate_age_value
            )

            if candidate_age < current_age:
                return candidate

            if (
                candidate_age == current_age
                and int(candidate.get("score") or 0)
                > int(current.get("score") or 0)
            ):
                return candidate

            return current

        for i in range(start, len(d)):
            row = d.iloc[i]
            high = _safe_float(row.get("High"))
            low = _safe_float(row.get("Low"))
            close = _safe_float(row.get("Close"))
            open_ = _safe_float(row.get("Open"))

            if None in [high, low, close, open_]:
                continue

            prior = d.iloc[max(0, i - lookback_levels):i]
            if prior.empty or len(prior) < 10:
                continue

            prior_close = _safe_float(prior["Close"].iloc[-1])
            local_pools = _build_liquidity_levels(
                prior,
                atr=atr_value,
                current_price=prior_close,
                lookback=lookback_levels,
            )

            high_pools = (
                list(local_pools.get("equal_high_liquidity") or [])
                + list(local_pools.get("swing_high_liquidity") or [])
            )[:24]
            low_pools = (
                list(local_pools.get("equal_low_liquidity") or [])
                + list(local_pools.get("swing_low_liquidity") or [])
            )[:24]

            for pool in high_pools:
                level = _safe_float(pool.get("price"))
                if level is None:
                    continue

                breach_buffer = max(
                    (atr_value or 0.0) * 0.03,
                    abs(level) * 0.00005,
                )
                reclaim_buffer = max(
                    (atr_value or 0.0) * 0.01,
                    abs(level) * 0.00002,
                )

                if not (
                    high > level + breach_buffer
                    and close < level - reclaim_buffer
                ):
                    continue

                wick = high - max(close, open_)
                wick_atr = (
                    wick / atr_value
                    if atr_value and atr_value > 0
                    else None
                )
                pool_type = _upper(pool.get("type"))
                touches = int(
                    _safe_float(
                        pool.get("touches"),
                        pool.get("strength", 1),
                    )
                    or 1
                )
                score = 40
                score += 15 if "EQUAL_" in pool_type else 0
                score += min(12, max(0, touches - 1) * 4)
                if wick_atr is not None:
                    score += 20 if wick_atr >= 0.70 else 10 if wick_atr >= 0.35 else 0
                if close < open_:
                    score += 10

                candidate = {
                    "detected": True,
                    "direction": "BEARISH",
                    "sweep_type": "HIGH_SWEEP_REJECTION",
                    "swept_level": level,
                    "sweep_candle_age": len(d) - 1 - i,
                    "sweep_candle_time": _candle_time(d, i),
                    "sweep_high": high,
                    "sweep_low": low,
                    "pool_type": pool_type,
                    "pool_touches": touches,
                    "wick_size_atr": round(wick_atr, 3) if wick_atr is not None else None,
                    "close_reclaim": True,
                    "score": score,
                    "reason": (
                        f"Recent candle swept local {pool_type} at {level} "
                        "and closed back below it."
                    ),
                }
                best_bearish = _prefer_event(
                    best_bearish,
                    candidate,
                )

            for pool in low_pools:
                level = _safe_float(pool.get("price"))
                if level is None:
                    continue

                breach_buffer = max(
                    (atr_value or 0.0) * 0.03,
                    abs(level) * 0.00005,
                )
                reclaim_buffer = max(
                    (atr_value or 0.0) * 0.01,
                    abs(level) * 0.00002,
                )

                if not (
                    low < level - breach_buffer
                    and close > level + reclaim_buffer
                ):
                    continue

                wick = min(close, open_) - low
                wick_atr = (
                    wick / atr_value
                    if atr_value and atr_value > 0
                    else None
                )
                pool_type = _upper(pool.get("type"))
                touches = int(
                    _safe_float(
                        pool.get("touches"),
                        pool.get("strength", 1),
                    )
                    or 1
                )
                score = 40
                score += 15 if "EQUAL_" in pool_type else 0
                score += min(12, max(0, touches - 1) * 4)
                if wick_atr is not None:
                    score += 20 if wick_atr >= 0.70 else 10 if wick_atr >= 0.35 else 0
                if close > open_:
                    score += 10

                candidate = {
                    "detected": True,
                    "direction": "BULLISH",
                    "sweep_type": "LOW_SWEEP_REJECTION",
                    "swept_level": level,
                    "sweep_candle_age": len(d) - 1 - i,
                    "sweep_candle_time": _candle_time(d, i),
                    "sweep_high": high,
                    "sweep_low": low,
                    "pool_type": pool_type,
                    "pool_touches": touches,
                    "wick_size_atr": round(wick_atr, 3) if wick_atr is not None else None,
                    "close_reclaim": True,
                    "score": score,
                    "reason": (
                        f"Recent candle swept local {pool_type} at {level} "
                        "and closed back above it."
                    ),
                }
                best_bullish = _prefer_event(
                    best_bullish,
                    candidate,
                )

        if not best_bullish and not best_bearish:
            result["reason"] = "No recent sweep/rejection pattern found."
            return result

        bullish_event = dict(best_bullish or {})
        bearish_event = dict(best_bearish or {})

        for event in (bullish_event, bearish_event):
            event.pop("score", None)

        result["bullish_event"] = bullish_event
        result["bearish_event"] = bearish_event

        candidates = [
            event
            for event in (best_bullish, best_bearish)
            if isinstance(event, dict)
        ]
        candidates.sort(
            key=lambda event: (
                int(
                    999999
                    if _safe_float(
                        event.get("sweep_candle_age"),
                        999999,
                    ) is None
                    else _safe_float(
                        event.get("sweep_candle_age"),
                        999999,
                    )
                ),
                -int(event.get("score") or 0),
            )
        )

        best = dict(candidates[0])
        result.update(best)
        result["bullish_event"] = bullish_event
        result["bearish_event"] = bearish_event
        result.pop("score", None)
        return result

    except Exception as e:
        result["reason"] = f"Sweep detection failed: {e}"
        return result


# ============================================================
# Risk analysis
# ============================================================

def _nearest_adverse_liquidity(
    *,
    side: str,
    entry: float,
    stop: Optional[float],
    liquidity: Dict[str, Any],
    atr: Optional[float],
) -> Optional[Dict[str, Any]]:
    """
    LONG adverse liquidity:
      unswept liquidity below entry, especially between entry and stop.

    SHORT adverse liquidity:
      unswept liquidity above entry, especially between entry and stop.
    """
    try:
        side = _normalize_side(side)
        atr_value = _safe_float(atr)

        candidates: List[Dict[str, Any]] = []

        if side == "LONG":
            pools = []
            pools.extend(liquidity.get("swing_low_liquidity") or [])
            pools.extend(liquidity.get("equal_low_liquidity") or [])

            for pool in pools:
                price = _safe_float(pool.get("price"))
                if price is None or price >= entry:
                    continue

                dist = entry - price
                dist_atr = dist / atr_value if atr_value and atr_value > 0 else None

                between_entry_stop = False
                if stop is not None:
                    between_entry_stop = price <= entry and price >= stop

                cand = dict(pool)
                cand["adverse_side"] = "DOWNSIDE"
                cand["distance_from_entry"] = dist
                cand["distance_from_entry_atr"] = round(dist_atr, 3) if dist_atr is not None else None
                cand["between_entry_and_stop"] = bool(between_entry_stop)
                candidates.append(cand)

        elif side == "SHORT":
            pools = []
            pools.extend(liquidity.get("swing_high_liquidity") or [])
            pools.extend(liquidity.get("equal_high_liquidity") or [])

            for pool in pools:
                price = _safe_float(pool.get("price"))
                if price is None or price <= entry:
                    continue

                dist = price - entry
                dist_atr = dist / atr_value if atr_value and atr_value > 0 else None

                between_entry_stop = False
                if stop is not None:
                    between_entry_stop = price >= entry and price <= stop

                cand = dict(pool)
                cand["adverse_side"] = "UPSIDE"
                cand["distance_from_entry"] = dist
                cand["distance_from_entry_atr"] = round(dist_atr, 3) if dist_atr is not None else None
                cand["between_entry_and_stop"] = bool(between_entry_stop)
                candidates.append(cand)

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (
                0 if x.get("between_entry_and_stop") else 1,
                _safe_float(x.get("distance_from_entry_atr"), 999999) or 999999,
                -int(x.get("strength") or 0),
                int(x.get("age_candles") or 999999),
            )
        )

        return candidates[0]

    except Exception:
        return None


def _supportive_sweep_for_side(side: str, sweep: Dict[str, Any]) -> bool:
    try:
        side = _normalize_side(side)
        direction = _upper(sweep.get("direction"))

        if side == "LONG" and direction == "BULLISH":
            return True

        if side == "SHORT" and direction == "BEARISH":
            return True

        return False
    except Exception:
        return False


def _adverse_sweep_for_side(side: str, sweep: Dict[str, Any]) -> bool:
    try:
        side = _normalize_side(side)
        direction = _upper(sweep.get("direction"))

        if side == "LONG" and direction == "BEARISH":
            return True

        if side == "SHORT" and direction == "BULLISH":
            return True

        return False
    except Exception:
        return False


def _build_liquidity_high_risk_gate(
    *,
    side: str,
    entry: float,
    stop: Optional[float],
    atr: Optional[float],
    adverse_liquidity: Optional[Dict[str, Any]],
    adverse_sweep: bool,
    supportive_sweep: bool,
) -> Dict[str, Any]:
    """
    Confirm whether a HIGH liquidity label is justified.

    A liquidity level being somewhere inside the Entry-SL
    range is location evidence only. HIGH requires at least
    two independent major risk factors:

    1. strong/repeated liquidity,
    2. a recent pool within 0.80 ATR of entry,
    3. critical SL geometry,
    4. an active adverse liquidity sweep.

    HIGH also requires unsafe stop geometry or an adverse sweep, so a nearby
    or repeated level by itself never becomes a hard execution gate.
    """
    result = {
        "strong_liquidity_evidence": False,
        "near_liquidity_evidence": False,
        "unsafe_stop_geometry": False,
        "stop_relation": "UNKNOWN",
        "stop_buffer_atr": None,
        "major_confluence_count": 0,
        "high_risk_confirmed": False,
    }

    try:
        if not adverse_liquidity:
            return result

        normalized_side = _normalize_side(side)
        stop_value = _safe_float(stop)
        atr_value = _safe_float(atr)

        level_price = _safe_float(
            adverse_liquidity.get("price")
        )

        level_type = _upper(
            adverse_liquidity.get("type")
        )

        strength = int(
            _safe_float(
                adverse_liquidity.get("strength"),
                1,
            )
            or 1
        )

        touches = int(
            _safe_float(
                adverse_liquidity.get("touches"),
                strength,
            )
            or strength
        )

        strong_liquidity_evidence = bool(
            "EQUAL_" in level_type
            or strength >= 2
            or touches >= 2
        )

        distance_from_entry_atr = _safe_float(
            adverse_liquidity.get("distance_from_entry_atr")
        )
        age_candles = int(
            _safe_float(
                adverse_liquidity.get("age_candles"),
                999999,
            )
            or 999999
        )
        near_liquidity_evidence = bool(
            distance_from_entry_atr is not None
            and distance_from_entry_atr <= 0.80
            and age_candles <= 80
        )

        unsafe_stop_geometry = False
        stop_relation = "UNKNOWN"
        stop_buffer_atr = None

        valid_stop_geometry = bool(
            stop_value is not None
            and level_price is not None
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
                    level_price - stop_value
                )
            else:
                signed_stop_buffer = (
                    stop_value - level_price
                )

            stop_buffer_atr = (
                signed_stop_buffer
                / atr_value
            )

            if stop_buffer_atr < 0:
                # The stop is reached before price reaches the adverse
                # liquidity pool. It is exposed regardless of distance.
                stop_relation = "STOP_BEFORE_LIQUIDITY_POOL"
                unsafe_stop_geometry = True

            elif stop_buffer_atr <= 0.20:
                stop_relation = "CRITICAL_BUFFER"
                unsafe_stop_geometry = True

            elif stop_buffer_atr < 0.50:
                stop_relation = "THIN_BUFFER"

            else:
                stop_relation = "SAFE_BUFFER"

        major_confluence_count = sum(
            [
                bool(strong_liquidity_evidence),
                bool(near_liquidity_evidence),
                bool(unsafe_stop_geometry),
                bool(adverse_sweep),
            ]
        )

        high_risk_confirmed = bool(
            major_confluence_count >= 2
            and (
                unsafe_stop_geometry
                or adverse_sweep
            )
            and not supportive_sweep
        )

        result.update(
            {
                "strong_liquidity_evidence": bool(
                    strong_liquidity_evidence
                ),
                "near_liquidity_evidence": bool(
                    near_liquidity_evidence
                ),
                "unsafe_stop_geometry": bool(
                    unsafe_stop_geometry
                ),
                "stop_relation": stop_relation,
                "stop_buffer_atr": (
                    round(stop_buffer_atr, 4)
                    if stop_buffer_atr is not None
                    else None
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
        score >= 68
        and high_risk_confirmed
    ):
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


def _build_advice(
    *,
    side: str,
    risk_level: str,
    adverse_liquidity: Optional[Dict[str, Any]],
    sweep: Dict[str, Any],
    supportive_sweep: bool,
    adverse_sweep: bool,
) -> str:
    side = _normalize_side(side)

    if supportive_sweep:
        if side == "LONG":
            return (
                "A recent downside liquidity sweep and bullish reclaim was detected. "
                "This can support a LONG setup if the entry zone also confirms."
            )

        if side == "SHORT":
            return (
                "A recent upside liquidity sweep and bearish reclaim was detected. "
                "This can support a SHORT setup if the entry zone also confirms."
            )

    if risk_level == "LOW":
        return (
            "No critical unswept liquidity risk was found near the planned entry/stop path. "
            "Use existing entry confirmation and risk management rules."
        )

    if side == "LONG":
        if adverse_liquidity:
            return (
                "LONG setup has downside liquidity sweep risk. "
                "Price may sweep a nearby swing/equal low before continuing. "
                "Avoid blind long execution; prefer a liquidity sweep plus bullish reclaim or use reduced risk."
            )

        if adverse_sweep:
            return (
                "Recent liquidity action is bearish against the LONG setup. "
                "Wait for bullish reclaim before treating the entry as clean."
            )

    if side == "SHORT":
        if adverse_liquidity:
            return (
                "SHORT setup has upside liquidity sweep risk. "
                "Price may sweep a nearby swing/equal high before falling. "
                "Avoid blind short execution; prefer a liquidity sweep plus bearish reclaim or use reduced risk."
            )

        if adverse_sweep:
            return (
                "Recent liquidity action is bullish against the SHORT setup. "
                "Wait for bearish reclaim before treating the entry as clean."
            )

    return (
        "Liquidity risk is elevated. Treat this setup as conditional until price confirms around the entry zone."
    )


# ============================================================
# Side-neutral Gemini context
# ============================================================

def build_side_neutral_liquidity_context(
    *,
    df_analysis: Optional[pd.DataFrame],
    df_lower: Optional[pd.DataFrame] = None,
    current_price: Optional[float] = None,
    lookback_candles: int = 250,
    lower_lookback_candles: int = 120,
) -> Dict[str, Any]:
    """
    Build compact liquidity facts before Gemini chooses a side or trade plan.

    This function is evidence-only. It does not select LONG/SHORT, create
    Entry/SL/TP prices, or block a signal.
    """
    result = {
        "available": False,
        "closed_candles_only": True,
        "decision_authority": "EVIDENCE_ONLY",
        "main_atr": None,
        "lower_atr": None,
        "nearest_unswept_swing_highs": [],
        "nearest_unswept_swing_lows": [],
        "nearest_unswept_equal_highs": [],
        "nearest_unswept_equal_lows": [],
        "recent_bullish_sweep_reclaim": {},
        "recent_bearish_sweep_reclaim": {},
        "reason": "Liquidity evidence is unavailable.",
    }

    try:
        d = _normalize_ohlc(df_analysis)
        lower = _normalize_ohlc(df_lower)

        if d.empty:
            return result

        current = _safe_float(current_price)
        if current is None:
            current = _safe_float(d["Close"].iloc[-1])

        main_atr = _latest_atr(d)
        pools = _build_liquidity_levels(
            d,
            atr=main_atr,
            current_price=current,
            lookback=lookback_candles,
        )

        sweep_df = (
            lower.tail(lower_lookback_candles)
            if not lower.empty
            else d.tail(80)
        )
        lower_atr = _latest_atr(sweep_df)
        sweeps = _detect_recent_sweep(
            sweep_df,
            atr=lower_atr,
            lookback_levels=80,
            sweep_scan_candles=12,
        )

        def _compact_pool(pool: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "type": str(pool.get("type") or ""),
                "price": _safe_float(pool.get("price")),
                "touches": int(
                    _safe_float(
                        pool.get("touches"),
                        pool.get("strength", 1),
                    )
                    or 1
                ),
                "strength": int(
                    _safe_float(pool.get("strength"), 1)
                    or 1
                ),
                "age_candles": int(
                    _safe_float(pool.get("age_candles"), 0)
                    or 0
                ),
                "distance_from_current_atr": _safe_float(
                    pool.get("distance_from_current_atr")
                ),
                "state": "UNSWEPT",
            }

        result.update({
            "available": True,
            "main_atr": main_atr,
            "lower_atr": lower_atr,
            "nearest_unswept_swing_highs": [
                _compact_pool(item)
                for item in (pools.get("swing_high_liquidity") or [])[:3]
            ],
            "nearest_unswept_swing_lows": [
                _compact_pool(item)
                for item in (pools.get("swing_low_liquidity") or [])[:3]
            ],
            "nearest_unswept_equal_highs": [
                _compact_pool(item)
                for item in (pools.get("equal_high_liquidity") or [])[:3]
            ],
            "nearest_unswept_equal_lows": [
                _compact_pool(item)
                for item in (pools.get("equal_low_liquidity") or [])[:3]
            ],
            "recent_bullish_sweep_reclaim": dict(
                sweeps.get("bullish_event") or {}
            ),
            "recent_bearish_sweep_reclaim": dict(
                sweeps.get("bearish_event") or {}
            ),
            "reason": (
                "Compact local liquidity pools and closed-candle "
                "sweep/reclaim events are available."
            ),
        })
        return result

    except Exception as exc:
        result["reason"] = (
            "Liquidity context error: "
            f"{str(exc)}"
        )
        return result


# ============================================================
# Public evaluator
# ============================================================

def evaluate_liquidity_risk(
    *,
    side: str,
    entry_price: float,
    stop_loss: Optional[float] = None,
    user_timeframe: str = "",
    df_analysis: Optional[pd.DataFrame] = None,
    df_lower: Optional[pd.DataFrame] = None,
    atr: Optional[float] = None,
    current_price: Optional[float] = None,
    lookback_candles: int = 250,
    lower_lookback_candles: int = 120,
) -> Dict[str, Any]:
    """
    Evaluate liquidity sweep / stop-hunt risk.

    Important:
    - Does NOT fetch candles.
    - Does NOT change entry/SL/TP.
    - Does NOT block signals.
    - Only returns structured liquidity risk context.

    Main idea:
    LONG risk:
      unswept swing/equal lows below entry.

    SHORT risk:
      unswept swing/equal highs above entry.

    Supportive confirmation:
      LONG likes recent low sweep + close back above.
      SHORT likes recent high sweep + close back below.
    """
    try:
        normalized_side = _normalize_side(side)
        entry = _safe_float(entry_price)
        stop = _safe_float(stop_loss)
        current = _safe_float(current_price)

        d = _normalize_ohlc(df_analysis)
        lower = _normalize_ohlc(df_lower)

        if normalized_side not in ["LONG", "SHORT"] or entry is None:
            return {
                "enabled": True,
                "risk_level": "UNKNOWN",
                "liquidity_score": 0,
                "liquidity_risk_detected": False,
                "adverse_liquidity_risk": False,
                "supportive_sweep_detected": False,
                "adverse_sweep_detected": False,
                "nearest_adverse_liquidity": None,
                "recent_sweep": {},
                "liquidity_pools": {},
                "suggest_wait_for_sweep": False,
                "suggest_reduced_risk": False,
                "suggested_sweep_level": None,
                "advice": "Liquidity risk unavailable because side or entry is missing.",
                "reason": "Invalid side/entry.",
                "signals": {},
            }

        if d.empty:
            return {
                "enabled": True,
                "risk_level": "UNKNOWN",
                "liquidity_score": 0,
                "liquidity_risk_detected": False,
                "adverse_liquidity_risk": False,
                "supportive_sweep_detected": False,
                "adverse_sweep_detected": False,
                "nearest_adverse_liquidity": None,
                "recent_sweep": {},
                "liquidity_pools": {},
                "suggest_wait_for_sweep": False,
                "suggest_reduced_risk": False,
                "suggested_sweep_level": None,
                "advice": "Liquidity risk unavailable because candle data is missing.",
                "reason": "No usable OHLC candles.",
                "signals": {},
            }

        atr_value = _latest_atr(d, fallback=atr)

        # Main timeframe liquidity pools
        liquidity = _build_liquidity_levels(
            d,
            atr=atr_value,
            current_price=current,
            lookback=lookback_candles,
        )

        adverse_liq = _nearest_adverse_liquidity(
            side=normalized_side,
            entry=entry,
            stop=stop,
            liquidity=liquidity,
            atr=atr_value,
        )

        # Recent sweep detection:
        # Prefer lower TF if available because sweep/reclaim often happens intraperiod.
        sweep_df = lower.tail(lower_lookback_candles) if not lower.empty else d.tail(80)

        # Sweep geometry belongs to the dataframe used for sweep detection.
        # Do not score 15m wicks with the much larger 1H ATR.
        sweep_atr = _latest_atr(sweep_df)

        recent_sweep = _detect_recent_sweep(
            sweep_df,
            atr=sweep_atr,
            lookback_levels=80,
            sweep_scan_candles=12,
        )

        supportive_event = (
            recent_sweep.get("bullish_event")
            if normalized_side == "LONG"
            else recent_sweep.get("bearish_event")
        ) or {}

        adverse_event = (
            recent_sweep.get("bearish_event")
            if normalized_side == "LONG"
            else recent_sweep.get("bullish_event")
        ) or {}

        supportive_level = _safe_float(
            supportive_event.get("swept_level")
        )
        supportive_distance_atr = (
            abs(entry - supportive_level) / atr_value
            if (
                supportive_level is not None
                and atr_value is not None
                and atr_value > 0
            )
            else None
        )
        supportive_level_on_risk_side = bool(
            supportive_level is not None
            and (
                (
                    normalized_side == "LONG"
                    and supportive_level < entry
                )
                or (
                    normalized_side == "SHORT"
                    and supportive_level > entry
                )
            )
        )
        supportive_sweep = bool(
            supportive_event.get("detected")
            and supportive_event.get("close_reclaim")
            and supportive_level_on_risk_side
            and supportive_distance_atr is not None
            and supportive_distance_atr <= 2.0
        )
        adverse_sweep = bool(
            adverse_event.get("detected")
            and adverse_event.get("close_reclaim")
        )

        score = 10
        reasons: List[str] = []

        if adverse_liq:
            score += 25
            reasons.append("adverse unswept liquidity exists near the setup path")

            strength = int(_safe_float(adverse_liq.get("strength"), 1) or 1)
            dist_atr = _safe_float(adverse_liq.get("distance_from_entry_atr"), 999) or 999

            if strength >= 3:
                score += 12
                reasons.append("adverse liquidity has multiple touches/equal-level strength")
            elif strength >= 2:
                score += 7
                reasons.append("adverse liquidity has cluster strength")

            if dist_atr <= 0.8:
                score += 15
                reasons.append("adverse liquidity is very close to entry")
            elif dist_atr <= 1.6:
                score += 9
                reasons.append("adverse liquidity is within nearby ATR range")

            if adverse_liq.get("between_entry_and_stop"):
                score += 18
                reasons.append("adverse liquidity sits between entry and stop/invalidation")

        if adverse_sweep:
            score += 30
            reasons.append(
                "recent liquidity sweep direction is against the setup"
            )

        if supportive_sweep:
            score -= 20
            reasons.append(
                "recent liquidity sweep/reclaim supports the setup direction"
            )

        if current is not None:
            if normalized_side == "LONG" and current < entry:
                score += 5
                reasons.append("current price is below long entry, so downside liquidity may be tested first")

            if normalized_side == "SHORT" and current > entry:
                score += 5
                reasons.append("current price is above short entry, so upside liquidity may be tested first")

        raw_score = max(
            0,
            min(
                100,
                int(score),
            ),
        )

        high_risk_gate = (
            _build_liquidity_high_risk_gate(
                side=normalized_side,
                entry=entry,
                stop=stop,
                atr=atr_value,
                adverse_liquidity=adverse_liq,
                adverse_sweep=adverse_sweep,
                supportive_sweep=supportive_sweep,
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
                67,
            )
        )

        if (
            raw_score >= 68
            and not high_risk_confirmed
        ):
            reasons.append(
                "HIGH liquidity risk was not confirmed because "
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

        suggested_sweep_level = None
        if adverse_liq:
            suggested_sweep_level = {
                "type": adverse_liq.get("type"),
                "price": round(float(adverse_liq.get("price")), 8),
                "adverse_side": adverse_liq.get("adverse_side"),
                "between_entry_and_stop": adverse_liq.get("between_entry_and_stop"),
                "note": (
                    "Watch this liquidity level for sweep/reclaim confirmation before treating the setup as clean."
                ),
            }

        advice = _build_advice(
            side=normalized_side,
            risk_level=risk_level,
            adverse_liquidity=adverse_liq,
            sweep=recent_sweep,
            supportive_sweep=supportive_sweep,
            adverse_sweep=adverse_sweep,
        )

        return {
            "enabled": True,
            "user_timeframe": str(user_timeframe or "").lower().strip(),
            "side": normalized_side,
            "risk_level": risk_level,
            "liquidity_score": score,
            "raw_liquidity_score": raw_score,
            "liquidity_risk_detected": bool(risk_detected),
            "adverse_liquidity_risk": bool(
                adverse_liq is not None
            ),
            "high_risk_confirmed": bool(
                high_risk_confirmed
            ),
            "high_risk_gate": high_risk_gate,
            "supportive_sweep_detected": bool(
                supportive_sweep
            ),
            "adverse_sweep_detected": bool(adverse_sweep),
            "nearest_adverse_liquidity": adverse_liq,
            "recent_sweep": recent_sweep,
            "supportive_sweep_event": supportive_event,
            "adverse_sweep_event": adverse_event,
            "liquidity_pools": {
                "swing_high_count": len(liquidity.get("swing_high_liquidity") or []),
                "swing_low_count": len(liquidity.get("swing_low_liquidity") or []),
                "equal_high_count": len(liquidity.get("equal_high_liquidity") or []),
                "equal_low_count": len(liquidity.get("equal_low_liquidity") or []),
                "nearest_swing_highs": (liquidity.get("swing_high_liquidity") or [])[:5],
                "nearest_swing_lows": (liquidity.get("swing_low_liquidity") or [])[:5],
                "nearest_equal_highs": (liquidity.get("equal_high_liquidity") or [])[:5],
                "nearest_equal_lows": (liquidity.get("equal_low_liquidity") or [])[:5],
            },
            "suggest_wait_for_sweep": bool(high_risk and adverse_liq is not None and not supportive_sweep),
            "suggest_reduced_risk": bool(high_risk or adverse_sweep),
            "suggested_sweep_level": suggested_sweep_level,
            "advice": advice,
            "reason": "; ".join(reasons) if reasons else "No critical liquidity sweep risk found.",
            "signals": {
                "atr": atr_value,
                "current_price": current,
                "has_adverse_liquidity": adverse_liq is not None,
                "adverse_liquidity_between_entry_stop": bool(adverse_liq and adverse_liq.get("between_entry_and_stop")),
                "adverse_liquidity_distance_atr": adverse_liq.get("distance_from_entry_atr") if adverse_liq else None,
                "strong_liquidity_evidence": bool(
                    high_risk_gate.get(
                        "strong_liquidity_evidence"
                    )
                ),
                "near_liquidity_evidence": bool(
                    high_risk_gate.get(
                        "near_liquidity_evidence"
                    )
                ),
                "unsafe_stop_geometry": bool(
                    high_risk_gate.get(
                        "unsafe_stop_geometry"
                    )
                ),
                "liquidity_stop_relation": (
                    high_risk_gate.get(
                        "stop_relation"
                    )
                ),
                "liquidity_stop_buffer_atr": (
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
                "recent_sweep_detected": bool(recent_sweep.get("detected")),
                "recent_sweep_direction": recent_sweep.get("direction"),
                "sweep_timeframe_atr": sweep_atr,
                "supportive_sweep": supportive_sweep,
                "supportive_sweep_setup_aligned": supportive_sweep,
                "supportive_sweep_distance_atr": (
                    round(supportive_distance_atr, 4)
                    if supportive_distance_atr is not None
                    else None
                ),
                "adverse_sweep": adverse_sweep,
            },
        }

    except Exception as e:
        return {
            "enabled": True,
            "risk_level": "UNKNOWN",
            "liquidity_score": 0,
            "liquidity_risk_detected": False,
            "adverse_liquidity_risk": False,
            "supportive_sweep_detected": False,
            "adverse_sweep_detected": False,
            "nearest_adverse_liquidity": None,
            "recent_sweep": {},
            "liquidity_pools": {},
            "suggest_wait_for_sweep": False,
            "suggest_reduced_risk": False,
            "suggested_sweep_level": None,
            "advice": "Liquidity risk evaluation failed.",
            "reason": str(e),
            "signals": {},
        }


# ============================================================
# Response attach helpers
# ============================================================

def add_liquidity_to_scenario(
    scenario: Dict[str, Any],
    interval: str,
    df_analysis: Optional[pd.DataFrame],
    df_lower: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Decorate scenario with liquidity_risk.

    Response-only / advisory by default.
    Does not modify entry, SL, TP, or status.
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

        scenario["liquidity_risk"] = evaluate_liquidity_risk(
            side=side,
            entry_price=entry,
            stop_loss=stop,
            user_timeframe=interval,
            df_analysis=df_analysis,
            df_lower=df_lower,
            atr=atr,
            current_price=current_price,
        )

    except Exception as e:
        try:
            scenario["liquidity_risk"] = {
                "enabled": True,
                "risk_level": "UNKNOWN",
                "liquidity_score": 0,
                "liquidity_risk_detected": False,
                "adverse_liquidity_risk": False,
                "supportive_sweep_detected": False,
                "adverse_sweep_detected": False,
                "nearest_adverse_liquidity": None,
                "recent_sweep": {},
                "liquidity_pools": {},
                "suggest_wait_for_sweep": False,
                "suggest_reduced_risk": False,
                "suggested_sweep_level": None,
                "advice": "Liquidity risk unavailable because scenario decoration failed.",
                "reason": str(e),
                "signals": {},
            }
        except Exception:
            pass

    return scenario


def add_liquidity_to_ai_data(
    ai_data: Dict[str, Any],
    interval: str,
    df_analysis: Optional[pd.DataFrame],
    df_lower: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Attach liquidity_risk to first trade scenario and selected_scenario.
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

        scenario = add_liquidity_to_scenario(
            scenario=scenario,
            interval=interval,
            df_analysis=df_analysis,
            df_lower=df_lower,
        )

        scenarios[0] = scenario
        ai_data["trade_scenarios"] = scenarios

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            selected_side = _normalize_side(selected.get("trade_type") or selected.get("side"))
            scenario_side = _normalize_side(scenario.get("trade_type") or scenario.get("side"))

            selected_entry = _extract_price(selected.get("entry") or selected.get("entry_zone"))
            scenario_entry = _extract_price(scenario.get("entry") or scenario.get("entry_zone"))

            entry_matches = (
                selected_entry is None
                or scenario_entry is None
                or abs(selected_entry - scenario_entry) <= max(abs(scenario_entry) * 0.000001, 1e-12)
            )

            if selected_side == scenario_side and entry_matches:
                ai_data["selected_scenario"] = scenario

    except Exception as e:
        print(f"[LIQUIDITY_RISK] attach skipped/error: {e}")

    return ai_data
