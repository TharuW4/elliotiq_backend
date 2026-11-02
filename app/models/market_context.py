import math
import pandas as pd
import numpy as np


# Added 0.236 for shallow pullbacks in strong trends.
FIB_RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSIONS = [1.0, 1.272, 1.618, 2.0]


def _safe_float(value):
    """
    Safely convert values into float.
    Handles None, NaN, Infinity, numpy numbers, and comma-formatted strings.
    """
    try:
        if value is None:
            return None

        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        cleaned = str(value).replace(",", "").strip()
        if cleaned == "":
            return None

        value = float(cleaned)
        if math.isnan(value) or math.isinf(value):
            return None

        return value
    except Exception:
        return None


def _format_price(value):
    value = _safe_float(value)
    if value is None:
        return "N/A"

    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.8f}"


def calculate_atr(df, period=14):
    """
    Calculate ATR from OHLC dataframe.
    Expected columns: High, Low, Close
    """
    try:
        if df is None or df.empty:
            return None

        required = {"High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            return None

        data = df.copy()

        high_low = data["High"] - data["Low"]
        high_close = (data["High"] - data["Close"].shift()).abs()
        low_close = (data["Low"] - data["Close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / period, adjust=False).mean()

        return _safe_float(atr.iloc[-1])
    except Exception:
        return None


def calculate_ema_context(df):
    """
    EMA50 / EMA200 supporting trend context.
    This is NOT a hard signal filter.
    It only helps AI understand trend background.
    """
    context = {
        "available": False,
        "state": "EMA context unavailable",
        "ema50": None,
        "ema200": None,
        "price_vs_ema50": "Unknown",
        "price_vs_ema200": "Unknown",
        "ema50_vs_ema200": "Unknown",
        "trend_bias": "Unknown",
    }

    try:
        if df is None or df.empty or "Close" not in df.columns:
            context["state"] = "EMA context unavailable: no close data"
            return context

        closes = pd.to_numeric(df["Close"], errors="coerce").dropna()

        if len(closes) < 50:
            context["state"] = "EMA context unavailable: not enough candles for EMA50"
            return context

        current_price = _safe_float(closes.iloc[-1])
        ema50 = _safe_float(closes.ewm(span=50, adjust=False).mean().iloc[-1])

        if current_price is None or ema50 is None:
            context["state"] = "EMA context unavailable: invalid price data"
            return context

        ema200 = None
        if len(closes) >= 200:
            ema200 = _safe_float(closes.ewm(span=200, adjust=False).mean().iloc[-1])

        context["available"] = True
        context["state"] = "EMA context available"
        context["ema50"] = ema50
        context["ema200"] = ema200

        context["price_vs_ema50"] = "Above EMA50" if current_price > ema50 else "Below EMA50"

        if ema200 is None:
            context["price_vs_ema200"] = "EMA200 unavailable"
            context["ema50_vs_ema200"] = "EMA200 unavailable"
            context["trend_bias"] = "Partial EMA context only"
            return context

        context["price_vs_ema200"] = "Above EMA200" if current_price > ema200 else "Below EMA200"
        context["ema50_vs_ema200"] = "EMA50 above EMA200" if ema50 > ema200 else "EMA50 below EMA200"

        if current_price > ema50 and ema50 > ema200:
            context["trend_bias"] = "Bullish EMA alignment"
        elif current_price < ema50 and ema50 < ema200:
            context["trend_bias"] = "Bearish EMA alignment"
        else:
            context["trend_bias"] = "Mixed EMA alignment"

        return context

    except Exception as e:
        context["state"] = f"EMA context error: {str(e)}"
        return context


def dynamic_sr_tolerance(current_price=None, atr=None):
    """
    Dynamic support/resistance grouping tolerance.

    For BTC/ETH/SOL/low-price coins, fixed tolerance can be too strict or too loose.
    This uses ATR% when available, otherwise falls back to 0.3%.
    """
    try:
        current_price = _safe_float(current_price)
        atr = _safe_float(atr)

        if current_price is None or current_price <= 0:
            return 0.003

        if atr is None or atr <= 0:
            return 0.003

        atr_pct = atr / current_price

        # Keep tolerance reasonable:
        # minimum 0.2%, maximum 0.8%
        return min(max(atr_pct * 0.75, 0.002), 0.008)
    except Exception:
        return 0.003


def get_fib_stale_threshold(interval=None, current_price=None, atr=None):
    """
    Return max stale distance percentage for Elliott Fib context.
    Example: 0.08 = 8%

    Lower timeframes become stale faster.
    Higher timeframes can keep wider swing context.
    """
    interval = str(interval or "").lower()

    if interval in ["1m", "3m", "5m"]:
        min_pct, max_pct = 0.02, 0.05
    elif interval in ["15m", "30m"]:
        min_pct, max_pct = 0.03, 0.08
    elif interval in ["1h", "2h"]:
        min_pct, max_pct = 0.05, 0.12
    elif interval in ["4h", "6h"]:
        min_pct, max_pct = 0.08, 0.18
    elif interval in ["1d", "daily", "1day"]:
        min_pct, max_pct = 0.12, 0.30
    else:
        min_pct, max_pct = 0.05, 0.12

    try:
        current_price = _safe_float(current_price)
        atr = _safe_float(atr)

        if current_price is not None and current_price != 0 and atr is not None and atr > 0:
            atr_pct = atr / current_price
            dynamic_pct = atr_pct * 8
            return max(min_pct, min(max_pct, dynamic_pct))
    except Exception:
        pass

    return max_pct

def _sr_distance_limits(interval=None):
    tf = str(interval or "").lower().strip()

    if tf in {"1m", "3m", "5m"}:
        return 0.03, 6.0

    if tf in {"15m", "30m"}:
        return 0.05, 8.0

    if tf in {"1h", "60m", "2h"}:
        return 0.08, 10.0

    if tf in {"4h", "240m", "6h"}:
        return 0.15, 14.0

    if tf in {"1d", "d", "daily", "1day"}:
        return 0.30, 20.0

    if tf in {"1w", "w", "weekly", "1mo", "monthly"}:
        return 0.60, 30.0

    return 0.08, 10.0


def _zone_break_confirmed(
    df,
    zone,
    direction,
    current_price,
    atr,
    tolerance_pct,
):
    """
    Backward-compatible helper.

    A polarity flip is fully confirmed only after:
    1. a close-based break
    2. a successful retest from the opposite side
    """
    result = _evaluate_zone_flip(
        df=df,
        zone=zone,
        direction=direction,
        current_price=current_price,
        atr=atr,
        tolerance_pct=tolerance_pct,
    )

    return (
        result.get("status")
        == "CONFIRMED_RETEST"
    )


def _evaluate_zone_flip(
    df,
    zone,
    direction,
    current_price,
    atr,
    tolerance_pct,
    lookback=24,
):
    """
    direction="above":
        old resistance may become support

    direction="below":
        old support may become resistance
    """
    result = {
        "status": "NO_BREAK",
        "break_confirmed": False,
        "retest_seen": False,
        "retest_held": False,
        "break_position": None,
    }

    try:
        required = {
            "High",
            "Low",
            "Close",
        }

        if (
            df is None
            or df.empty
            or not required.issubset(df.columns)
        ):
            return result

        zone_price = _safe_float(
            zone.get("price")
        )

        zone_min = _safe_float(
            zone.get("min")
        )

        zone_max = _safe_float(
            zone.get("max")
        )

        if zone_min is None:
            zone_min = zone_price

        if zone_max is None:
            zone_max = zone_price

        if (
            zone_min is None
            or zone_max is None
        ):
            return result

        current_price = (
            _safe_float(current_price)
            or (
                zone_min
                + zone_max
            )
            / 2.0
        )

        atr = _safe_float(atr)

        tolerance_pct = (
            _safe_float(tolerance_pct)
            or 0.003
        )

        data = df.tail(
            max(
                int(lookback or 24),
                8,
            )
        ).copy()

        for column in [
            "High",
            "Low",
            "Close",
        ]:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data = data.dropna(
            subset=[
                "High",
                "Low",
                "Close",
            ]
        )

        if len(data) < 3:
            return result

        break_buffer = max(
            abs(current_price)
            * tolerance_pct
            * 0.15,

            atr * 0.05
            if atr is not None
            and atr > 0
            else 0.0,

            1e-12,
        )

        retest_tolerance = max(
            abs(current_price)
            * tolerance_pct
            * 0.60,

            atr * 0.15
            if atr is not None
            and atr > 0
            else 0.0,

            break_buffer,
        )

        closes = data[
            "Close"
        ].to_numpy(
            dtype=float
        )

        if direction == "above":
            break_flags = (
                closes
                > zone_max
                + break_buffer
            )
        else:
            break_flags = (
                closes
                < zone_min
                - break_buffer
            )

        break_position = None

        for i in range(
            1,
            len(break_flags),
        ):
            recent_flags = break_flags[
                max(0, i - 2):
                i + 1
            ]

            if int(
                recent_flags.sum()
            ) >= 2:
                break_position = i
                break

        if break_position is None:
            return result

        result.update(
            {
                "status": (
                    "BREAK_CONFIRMED_RETEST_PENDING"
                ),
                "break_confirmed": True,
                "break_position": int(
                    break_position
                ),
            }
        )

        post_break = data.iloc[
            break_position + 1:
        ]

        if post_break.empty:
            return result

        if direction == "above":
            failed = bool(
                (
                    post_break["Close"]
                    < zone_min
                    - break_buffer
                ).any()
            )

            retest_rows = post_break[
                post_break["Low"]
                <= zone_max
                + retest_tolerance
            ]

            retest_held = bool(
                not retest_rows.empty
                and (
                    retest_rows["Close"]
                    >= zone_min
                    - break_buffer
                ).any()
            )

        else:
            failed = bool(
                (
                    post_break["Close"]
                    > zone_max
                    + break_buffer
                ).any()
            )

            retest_rows = post_break[
                post_break["High"]
                >= zone_min
                - retest_tolerance
            ]

            retest_held = bool(
                not retest_rows.empty
                and (
                    retest_rows["Close"]
                    <= zone_max
                    + break_buffer
                ).any()
            )

        result["retest_seen"] = bool(
            not retest_rows.empty
        )

        result["retest_held"] = (
            retest_held
        )

        if (
            failed
            and not retest_held
        ):
            result["status"] = (
                "FAILED_AFTER_BREAK"
            )

        elif retest_held:
            result["status"] = (
                "CONFIRMED_RETEST"
            )

        return result

    except Exception:
        return result


def _best_major_zone(
    zones,
    current_price,
):
    """
    Rank by zone score with a small distance penalty.
    """
    try:
        current_price = _safe_float(
            current_price
        )

        def rank(zone):
            score = (
                _safe_float(
                    zone.get("score")
                )
                or 0.0
            )

            price = _safe_float(
                zone.get("price")
            )

            if (
                current_price is None
                or current_price <= 0
                or price is None
            ):
                return score

            distance_pct = (
                abs(
                    price - current_price
                )
                / current_price
            )

            return (
                score
                - distance_pct * 8.0
            )

        return max(
            zones or [],
            key=rank,
            default=None,
        )

    except Exception:
        return None

def _merge_nearby_pivot_events(
    points,
    pivot_type,
    max_gap=3,
    price_tolerance_pct=0.0005,
):
    """
    Merge adjacent equal-high/equal-low plateau edges
    into one market reaction.

    Separate reactions farther apart remain
    separate touches.
    """
    try:
        ordered = sorted(
            points or [],
            key=lambda point: int(
                point.get(
                    "position",
                    0,
                )
            ),
        )

        if not ordered:
            return []

        merged = []

        for point in ordered:
            item = dict(point)

            if not merged:
                item["event_size"] = 1
                merged.append(item)
                continue

            previous = merged[-1]

            previous_price = _safe_float(
                previous.get("price")
            )

            current_price = _safe_float(
                item.get("price")
            )

            previous_position = int(
                previous.get(
                    "position",
                    0,
                )
            )

            current_position = int(
                item.get(
                    "position",
                    0,
                )
            )

            same_zone = False

            if (
                previous_price is not None
                and current_price is not None
            ):
                base = max(
                    abs(previous_price),
                    abs(current_price),
                    1e-12,
                )

                same_zone = (
                    abs(
                        current_price
                        - previous_price
                    )
                    / base
                    <= float(
                        price_tolerance_pct
                    )
                )

            nearby = (
                current_position
                - previous_position
                <= int(max_gap)
            )

            if same_zone and nearby:
                if pivot_type == "high":
                    previous["price"] = max(
                        previous_price,
                        current_price,
                    )
                else:
                    previous["price"] = min(
                        previous_price,
                        current_price,
                    )

                previous["position"] = max(
                    previous_position,
                    current_position,
                )

                previous["date"] = (
                    item.get("date")
                    or previous.get("date")
                )

                previous["recency"] = max(
                    _safe_float(
                        previous.get(
                            "recency"
                        )
                    )
                    or 0.0,

                    _safe_float(
                        item.get(
                            "recency"
                        )
                    )
                    or 0.0,
                )

                previous["rejection"] = max(
                    _safe_float(
                        previous.get(
                            "rejection"
                        )
                    )
                    or 0.0,

                    _safe_float(
                        item.get(
                            "rejection"
                        )
                    )
                    or 0.0,
                )

                previous["event_size"] = (
                    int(
                        previous.get(
                            "event_size",
                            1,
                        )
                    )
                    + 1
                )

            else:
                item["event_size"] = 1
                merged.append(item)

        return merged

    except Exception:
        return points or []


def detect_swing_levels(
    df,
    lookback=120,
    window=3,
):
    """
    Detect swing highs and swing lows.

    Equal-high/equal-low plateau edges are preserved.
    This allows liquidity levels and repeated S/R touches
    to be detected instead of silently ignored.
    """
    result = {
        "swing_highs": [],
        "swing_lows": [],
        "recent_swing_high": None,
        "recent_swing_low": None,
        "swing_points": [],
        "high_points": [],
        "low_points": [],
    }

    try:
        if (
            df is None
            or df.empty
            or not {
                "High",
                "Low",
                "Close",
            }.issubset(
                df.columns
            )
        ):
            return result

        data = df.tail(
            int(
                lookback
                or 120
            )
        ).copy()

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
        ]:
            if col in data.columns:
                data[col] = pd.to_numeric(
                    data[col],
                    errors="coerce",
                )

        data = data.dropna(
            subset=[
                "High",
                "Low",
                "Close",
            ]
        )

        window = max(
            1,
            int(
                window
                or 1
            ),
        )

        if len(data) < window * 2 + 5:
            return result

        high_points = []
        low_points = []

        for i in range(
            window,
            len(data) - window,
        ):
            curr_high = _safe_float(
                data["High"].iloc[i]
            )

            curr_low = _safe_float(
                data["Low"].iloc[i]
            )

            left_high = _safe_float(
                data["High"]
                .iloc[
                    i - window:i
                ]
                .max()
            )

            right_high = _safe_float(
                data["High"]
                .iloc[
                    i + 1:
                    i + 1 + window
                ]
                .max()
            )

            left_low = _safe_float(
                data["Low"]
                .iloc[
                    i - window:i
                ]
                .min()
            )

            right_low = _safe_float(
                data["Low"]
                .iloc[
                    i + 1:
                    i + 1 + window
                ]
                .min()
            )

            if None in [
                curr_high,
                curr_low,
                left_high,
                right_high,
                left_low,
                right_low,
            ]:
                continue

            high_eps = max(
                abs(curr_high)
                * 1e-10,
                1e-12,
            )

            low_eps = max(
                abs(curr_low)
                * 1e-10,
                1e-12,
            )

            is_high = (
                curr_high
                >= left_high - high_eps

                and curr_high
                >= right_high - high_eps

                and (
                    curr_high
                    > left_high + high_eps

                    or curr_high
                    > right_high + high_eps
                )
            )

            is_low = (
                curr_low
                <= left_low + low_eps

                and curr_low
                <= right_low + low_eps

                and (
                    curr_low
                    < left_low - low_eps

                    or curr_low
                    < right_low - low_eps
                )
            )

            row = data.iloc[i]

            high = _safe_float(
                row.get("High")
            )

            low = _safe_float(
                row.get("Low")
            )

            open_price = _safe_float(
                row.get("Open")
            )

            close = _safe_float(
                row.get("Close")
            )

            candle_range = (
                high - low
                if (
                    high is not None
                    and low is not None
                    and high > low
                )
                else None
            )

            recency = (
                1.0
                - (
                    (
                        len(data)
                        - 1
                        - i
                    )
                    / max(
                        len(data) - 1,
                        1,
                    )
                )
            )

            if is_high:
                rejection = 0.0

                if (
                    candle_range
                    and open_price
                    is not None
                    and close
                    is not None
                ):
                    rejection = max(
                        0.0,
                        min(
                            1.0,

                            (
                                high
                                - max(
                                    open_price,
                                    close,
                                )
                            )
                            / candle_range,
                        ),
                    )

                high_points.append(
                    {
                        "price": curr_high,
                        "type": "high",
                        "date": str(
                            data.index[i]
                        ),
                        "position": i,
                        "recency": recency,
                        "rejection": rejection,
                    }
                )

            if is_low:
                rejection = 0.0

                if (
                    candle_range
                    and open_price
                    is not None
                    and close
                    is not None
                ):
                    rejection = max(
                        0.0,
                        min(
                            1.0,

                            (
                                min(
                                    open_price,
                                    close,
                                )
                                - low
                            )
                            / candle_range,
                        ),
                    )

                low_points.append(
                    {
                        "price": curr_low,
                        "type": "low",
                        "date": str(
                            data.index[i]
                        ),
                        "position": i,
                        "recency": recency,
                        "rejection": rejection,
                    }
                )

        high_points = (
            _merge_nearby_pivot_events(
                high_points,
                pivot_type="high",
                max_gap=max(
                    2,
                    window,
                ),
            )
        )

        low_points = (
            _merge_nearby_pivot_events(
                low_points,
                pivot_type="low",
                max_gap=max(
                    2,
                    window,
                ),
            )
        )

        all_points = sorted(
            high_points
            + low_points,

            key=lambda point: (
                point["position"]
            ),
        )

        result["high_points"] = (
            high_points
        )

        result["low_points"] = (
            low_points
        )

        result["swing_highs"] = [
            point["price"]
            for point in high_points
        ]

        result["swing_lows"] = [
            point["price"]
            for point in low_points
        ]

        result["recent_swing_high"] = (
            high_points[-1]["price"]
            if high_points
            else None
        )

        result["recent_swing_low"] = (
            low_points[-1]["price"]
            if low_points
            else None
        )

        result["swing_points"] = (
            all_points[-60:]
        )

        return result

    except Exception:
        return result


def group_levels(
    levels,
    tolerance_pct=0.003,
    role=None,
):
    """
    Group nearby S/R levels.

    Zone ranking:
    1. touch count
    2. pivot recency
    3. rejection-wick quality

    Old float inputs remain supported.
    """
    try:
        clean = []

        for item in levels or []:
            if isinstance(
                item,
                dict,
            ):
                price = _safe_float(
                    item.get("price")
                )

                recency = (
                    _safe_float(
                        item.get(
                            "recency"
                        )
                    )
                    or 0.0
                )

                rejection = (
                    _safe_float(
                        item.get(
                            "rejection"
                        )
                    )
                    or 0.0
                )

            else:
                price = _safe_float(
                    item
                )

                recency = 0.0
                rejection = 0.0

            if price is not None:
                clean.append(
                    {
                        "price": price,
                        "recency": (
                            recency
                        ),
                        "rejection": (
                            rejection
                        ),
                    }
                )

        clean.sort(
            key=lambda item: (
                item["price"]
            )
        )

        if not clean:
            return []

        tolerance_pct = max(
            _safe_float(
                tolerance_pct
            )
            or 0.003,

            1e-8,
        )

        groups = [
            [
                clean[0]
            ]
        ]

        for item in clean[1:]:
            base = (
                sum(
                    point["price"]
                    for point
                    in groups[-1]
                )
                / len(
                    groups[-1]
                )
            )

            if (
                base
                and (
                    abs(
                        item["price"]
                        - base
                    )
                    / abs(base)
                )
                <= tolerance_pct
            ):
                groups[-1].append(
                    item
                )

            else:
                groups.append(
                    [
                        item
                    ]
                )

        zones = []

        for group in groups:
            prices = [
                item["price"]
                for item in group
            ]

            touches = len(
                group
            )

            avg_recency = (
                sum(
                    item[
                        "recency"
                    ]
                    for item in group
                )
                / touches
            )

            avg_rejection = (
                sum(
                    item[
                        "rejection"
                    ]
                    for item in group
                )
                / touches
            )

            weights = [
                (
                    1.0
                    + item[
                        "recency"
                    ]
                    * 0.5
                    + item[
                        "rejection"
                    ]
                    * 0.5
                )
                for item in group
            ]

            center = (
                sum(
                    group[i]["price"]
                    * weights[i]

                    for i
                    in range(
                        touches
                    )
                )
                / (
                    sum(weights)
                    or touches
                )
            )

            zones.append(
                {
                    "price": center,

                    "strength": (
                        touches
                    ),

                    "score": round(
                        (
                            touches
                            * 2.0
                        )
                        + avg_recency
                        + avg_rejection,

                        4,
                    ),

                    "min": min(
                        prices
                    ),

                    "max": max(
                        prices
                    ),

                    "role": str(
                        role
                        or "UNKNOWN"
                    ).upper(),

                    "origin_role": str(
                        role
                        or "UNKNOWN"
                    ).upper(),

                    "polarity_flipped": (
                        False
                    ),
                }
            )

        return sorted(
            zones,

            key=lambda zone: (
                zone["score"],
                zone["strength"],
            ),

            reverse=True,
        )

    except Exception:
        return []


def _filter_far_zones(
    zones,
    current_price,
    atr=None,
    max_pct=0.03,
    max_atr_mult=5,
):
    """
    Filter only wider major-zone candidates.

    Nearest support/resistance is selected before
    applying this distance filter.
    """
    try:
        current_price = _safe_float(
            current_price
        )

        atr = _safe_float(
            atr
        )

        if (
            current_price is None
            or current_price <= 0
        ):
            return zones or []

        result = []

        for zone in zones or []:
            price = _safe_float(
                zone.get("price")
            )

            if price is None:
                continue

            distance = abs(
                price
                - current_price
            )

            if (
                (
                    distance
                    / current_price
                )
                <= float(max_pct)

                or (
                    atr is not None
                    and atr > 0
                    and distance
                    <= atr
                    * float(
                        max_atr_mult
                    )
                )
            ):
                result.append(
                    zone
                )

        return result

    except Exception:
        return zones or []


def _decorate_flipped_zone(
    zone,
    new_role,
    evaluation,
):
    status = str(
        evaluation.get("status")
        or "NO_BREAK"
    ).upper()

    confirmed = (
        status
        == "CONFIRMED_RETEST"
    )

    flipped = dict(zone)

    flipped.update(
        {
            "origin_role": str(
                zone.get("origin_role")
                or zone.get("role")
                or "UNKNOWN"
            ).upper(),

            "role": str(
                new_role
            ).upper(),

            "polarity_flipped": (
                confirmed
            ),

            "potential_flip": (
                not confirmed
            ),

            "flip_status": status,

            "break_confirmed": bool(
                evaluation.get(
                    "break_confirmed"
                )
            ),

            "retest_seen": bool(
                evaluation.get(
                    "retest_seen"
                )
            ),

            "retest_held": bool(
                evaluation.get(
                    "retest_held"
                )
            ),

            "score": round(
                max(
                    0.0,

                    (
                        _safe_float(
                            zone.get("score")
                        )
                        or 0.0
                    )
                    + (
                        0.50
                        if confirmed
                        else -0.75
                    ),
                ),
                4,
            ),
        }
    )

    return flipped


def _classify_sr_zones(
    data,
    support_zones,
    resistance_zones,
    current_price,
    atr,
    tolerance_pct,
):
    """
    Break without retest:
        potential S/R only

    Break plus successful retest:
        confirmed polarity flip
    """
    supports = []
    resistances = []
    flips = []

    def native_zone(zone):
        item = dict(zone)

        item.setdefault(
            "flip_status",
            "NATIVE",
        )

        item.setdefault(
            "potential_flip",
            False,
        )

        item.setdefault(
            "break_confirmed",
            False,
        )

        item.setdefault(
            "retest_seen",
            False,
        )

        item.setdefault(
            "retest_held",
            False,
        )

        return item

    for zone in support_zones or []:
        price = _safe_float(
            zone.get("price")
        )

        if (
            price is None
            or current_price is None
        ):
            continue

        if price < current_price:
            supports.append(
                native_zone(zone)
            )
            continue

        evaluation = _evaluate_zone_flip(
            df=data,
            zone=zone,
            direction="below",
            current_price=current_price,
            atr=atr,
            tolerance_pct=tolerance_pct,
        )

        status = str(
            evaluation.get("status")
            or "NO_BREAK"
        ).upper()

        if status in {
            "CONFIRMED_RETEST",
            "BREAK_CONFIRMED_RETEST_PENDING",
        }:
            flipped = _decorate_flipped_zone(
                zone,
                "RESISTANCE",
                evaluation,
            )

            resistances.append(
                flipped
            )

            flips.append(
                flipped
            )

    for zone in resistance_zones or []:
        price = _safe_float(
            zone.get("price")
        )

        if (
            price is None
            or current_price is None
        ):
            continue

        if price > current_price:
            resistances.append(
                native_zone(zone)
            )
            continue

        evaluation = _evaluate_zone_flip(
            df=data,
            zone=zone,
            direction="above",
            current_price=current_price,
            atr=atr,
            tolerance_pct=tolerance_pct,
        )

        status = str(
            evaluation.get("status")
            or "NO_BREAK"
        ).upper()

        if status in {
            "CONFIRMED_RETEST",
            "BREAK_CONFIRMED_RETEST_PENDING",
        }:
            flipped = _decorate_flipped_zone(
                zone,
                "SUPPORT",
                evaluation,
            )

            supports.append(
                flipped
            )

            flips.append(
                flipped
            )

    return (
        supports,
        resistances,
        flips,
    )


def get_nearest_sr(
    df,
    lookback=120,
    interval=None,
):
    """
    Calculate scored support and resistance
    around current price.
    """
    context = {
        "current_price": None,
        "atr": None,

        "nearest_support": None,
        "nearest_resistance": None,

        "major_support": None,
        "major_resistance": None,

        "recent_swing_high": None,
        "recent_swing_low": None,

        "tolerance_pct": None,

        "support_zones": [],
        "resistance_zones": [],
        "polarity_flips": [],
    }

    try:
        if (
            df is None
            or df.empty
            or not {
                "High",
                "Low",
                "Close",
            }.issubset(
                df.columns
            )
        ):
            return context

        current_price = _safe_float(
            df["Close"].iloc[-1]
        )

        atr = calculate_atr(
            df,
            period=14,
        )

        tolerance_pct = (
            dynamic_sr_tolerance(
                current_price,
                atr,
            )
        )

        swings = detect_swing_levels(
            df,
            lookback=lookback,
            window=3,
        )

        native_support = group_levels(
            swings[
                "low_points"
            ],
            tolerance_pct,
            role="SUPPORT",
        )

        native_resistance = group_levels(
            swings[
                "high_points"
            ],
            tolerance_pct,
            role="RESISTANCE",
        )

        (
            supports,
            resistances,
            flips,
        ) = _classify_sr_zones(
            df,
            native_support,
            native_resistance,
            current_price,
            atr,
            tolerance_pct,
        )

        supports.sort(
            key=lambda zone: abs(
                current_price
                - zone["price"]
            )
        )

        resistances.sort(
            key=lambda zone: abs(
                zone["price"]
                - current_price
            )
        )

        (
            max_pct,
            max_atr_mult,
        ) = _sr_distance_limits(
            interval
        )

        context.update(
            {
                "current_price": (
                    current_price
                ),

                "atr": atr,

                "tolerance_pct": (
                    tolerance_pct
                ),

                "recent_swing_high": (
                    swings.get(
                        "recent_swing_high"
                    )
                ),

                "recent_swing_low": (
                    swings.get(
                        "recent_swing_low"
                    )
                ),

                "nearest_support": (
                    supports[0]
                    if supports
                    else None
                ),

                "nearest_resistance": (
                    resistances[0]
                    if resistances
                    else None
                ),

                "major_support": (
                    _best_major_zone(
                        _filter_far_zones(
                            supports,
                            current_price,
                            atr,
                            max_pct,
                            max_atr_mult,
                        ),

                        current_price,
                    )
                ),

                "major_resistance": (
                    _best_major_zone(
                        _filter_far_zones(
                            resistances,
                            current_price,
                            atr,
                            max_pct,
                            max_atr_mult,
                        ),

                        current_price,
                    )
                ),

                "support_zones": (
                    supports
                ),

                "resistance_zones": (
                    resistances
                ),

                "polarity_flips": (
                    flips
                ),
            }
        )

        return context

    except Exception:
        return context


def _append_range_extreme_if_unique(
    points,
    price,
    position,
    recency,
    pivot_type,
    window,
    tolerance_pct,
):
    """
    Add the absolute lookback extreme only when the
    same candle-area is not already represented by
    a confirmed swing.
    """
    try:
        price = _safe_float(price)

        if price is None:
            return points

        position = int(position)

        event_tolerance = min(
            max(
                (
                    _safe_float(
                        tolerance_pct
                    )
                    or 0.003
                )
                * 0.25,

                1e-8,
            ),
            0.0005,
        )

        for point in points or []:
            point_price = _safe_float(
                point.get("price")
            )

            point_position = int(
                point.get(
                    "position",
                    position,
                )
            )

            if point_price is None:
                continue

            base = max(
                abs(price),
                abs(point_price),
                1e-12,
            )

            same_price_area = (
                abs(
                    point_price
                    - price
                )
                / base
                <= event_tolerance
            )

            same_candle_area = (
                abs(
                    point_position
                    - position
                )
                <= max(
                    int(window),
                    2,
                )
            )

            if (
                same_price_area
                and same_candle_area
            ):
                return points

        points.append(
            {
                "price": price,
                "type": pivot_type,
                "date": "",
                "position": position,
                "recency": (
                    _safe_float(recency)
                    or 0.0
                ),
                "rejection": 0.10,
                "confirmed": False,
                "source": "RANGE_EXTREME",
                "event_size": 1,
            }
        )

        return points

    except Exception:
        return points


def get_major_structure_context(
    df,
    lookback=300,
    window=5,
    interval=None,
):
    """
    Calculate wider market structure.

    All pivots inside the requested lookback remain available.
    The absolute lookback high/low are also added explicitly.
    """
    context = {
        "major_swing_high": None,
        "major_swing_low": None,

        "major_resistance": None,
        "major_support": None,

        "major_range_high": None,
        "major_range_low": None,

        "lookback": lookback,

        "support_zones": [],
        "resistance_zones": [],
        "polarity_flips": [],
    }

    try:
        if (
            df is None
            or df.empty
            or not {
                "High",
                "Low",
                "Close",
            }.issubset(
                df.columns
            )
        ):
            return context

        data = df.tail(
            int(
                lookback
                or 300
            )
        ).copy()

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
        ]:
            if col in data.columns:
                data[col] = pd.to_numeric(
                    data[col],
                    errors="coerce",
                )

        data = data.dropna(
            subset=[
                "High",
                "Low",
                "Close",
            ]
        )

        if data.empty:
            return context

        current_price = _safe_float(
            data["Close"].iloc[-1]
        )

        atr = calculate_atr(
            data,
            period=14,
        )

        tolerance_pct = (
            dynamic_sr_tolerance(
                current_price,
                atr,
            )
        )

        range_high = _safe_float(
            data["High"].max()
        )

        range_low = _safe_float(
            data["Low"].min()
        )

        swings = detect_swing_levels(
            data,
            lookback=lookback,
            window=window,
        )

        high_points = list(
            swings[
                "high_points"
            ]
        )

        low_points = list(
            swings[
                "low_points"
            ]
        )

        confirmed_high_prices = [
            _safe_float(
                point.get("price")
            )
            for point in high_points
            if _safe_float(
                point.get("price")
            ) is not None
        ]

        confirmed_low_prices = [
            _safe_float(
                point.get("price")
            )
            for point in low_points
            if _safe_float(
                point.get("price")
            ) is not None
        ]

        major_swing_high = (
            max(
                confirmed_high_prices
            )
            if confirmed_high_prices
            else None
        )

        major_swing_low = (
            min(
                confirmed_low_prices
            )
            if confirmed_low_prices
            else None
        )

        high_i = int(
            np.nanargmax(
                data[
                    "High"
                ].to_numpy(
                    dtype=float
                )
            )
        )

        low_i = int(
            np.nanargmin(
                data[
                    "Low"
                ].to_numpy(
                    dtype=float
                )
            )
        )

        high_recency = (
            1.0
            - (
                (
                    len(data)
                    - 1
                    - high_i
                )
                / max(
                    len(data) - 1,
                    1,
                )
            )
        )

        low_recency = (
            1.0
            - (
                (
                    len(data)
                    - 1
                    - low_i
                )
                / max(
                    len(data) - 1,
                    1,
                )
            )
        )

        high_points = (
            _append_range_extreme_if_unique(
                points=high_points,
                price=range_high,
                position=high_i,
                recency=high_recency,
                pivot_type="high",
                window=window,
                tolerance_pct=tolerance_pct,
            )
        )

        low_points = (
            _append_range_extreme_if_unique(
                points=low_points,
                price=range_low,
                position=low_i,
                recency=low_recency,
                pivot_type="low",
                window=window,
                tolerance_pct=tolerance_pct,
            )
        )

        native_support = group_levels(
            low_points,
            tolerance_pct,
            role="SUPPORT",
        )

        native_resistance = group_levels(
            high_points,
            tolerance_pct,
            role="RESISTANCE",
        )

        (
            supports,
            resistances,
            flips,
        ) = _classify_sr_zones(
            data,
            native_support,
            native_resistance,
            current_price,
            atr,
            tolerance_pct,
        )

        (
            max_pct,
            max_atr_mult,
        ) = _sr_distance_limits(
            interval
        )

        context.update(
            {
                "major_swing_high": (
                    major_swing_high
                ),

                "major_swing_low": (
                    major_swing_low
                ),

                "major_range_high": (
                    range_high
                ),

                "major_range_low": (
                    range_low
                ),

                "major_support": (
                    _best_major_zone(
                        _filter_far_zones(
                            supports,
                            current_price,
                            atr,
                            max_pct,
                            max_atr_mult,
                        ),

                        current_price,
                    )
                ),

                "major_resistance": (
                    _best_major_zone(
                        _filter_far_zones(
                            resistances,
                            current_price,
                            atr,
                            max_pct,
                            max_atr_mult,
                        ),

                        current_price,
                    )
                ),

                "support_zones": (
                    supports
                ),

                "resistance_zones": (
                    resistances
                ),

                "polarity_flips": (
                    flips
                ),
            }
        )

        return context

    except Exception:
        return context

def extract_wave_prices(waves):
    """
    Extract price values from existing micro_waves format:
    {"lbl": "...", "p": {"price": ..., "date": ..., "type": "..."}}
    """
    prices = []

    try:
        if not waves:
            return prices

        for w in waves:
            p = w.get("p", {}) if isinstance(w, dict) else {}
            price = _safe_float(p.get("price"))

            if price is not None:
                prices.append(price)

        return prices
    except Exception:
        return prices


def calculate_fib_context_from_waves(waves):
    """
    Build simple Fibonacci context from latest detected micro swing.
    This is supporting context only, not a standalone Elliott target engine.
    """
    context = {
        "last_move_start": None,
        "last_move_end": None,
        "direction": "Unknown",
        "retracements": {},
        "extensions": {},
        "shallow_pullback_zone": None,
        "pullback_zone": None,
        "deep_pullback": None,
    }

    try:
        prices = extract_wave_prices(waves)

        if len(prices) < 2:
            return context

        start = prices[-2]
        end = prices[-1]
        move = end - start

        if abs(move) <= 0:
            return context

        direction = "Bullish latest swing" if move > 0 else "Bearish latest swing"

        context["last_move_start"] = start
        context["last_move_end"] = end
        context["direction"] = direction

        retracements = {}
        extensions = {}

        for fib in FIB_RETRACEMENTS:
            if move > 0:
                level = end - (abs(move) * fib)
            else:
                level = end + (abs(move) * fib)
            retracements[str(fib)] = level

        for fib in FIB_EXTENSIONS:
            # Standard extension measured from swing start.
            # 1.0 equals swing end.
            level = start + (
                move * fib
            )

            extensions[
                str(fib)
            ] = level

        context["retracements"] = retracements
        context["extensions"] = extensions

        # Strong-trend shallow pullback zone: 0.236 - 0.382
        r236 = retracements.get("0.236")
        r382 = retracements.get("0.382")

        if r236 is not None and r382 is not None:
            context["shallow_pullback_zone"] = {
                "min": min(r236, r382),
                "max": max(r236, r382),
                "label": "0.236 - 0.382"
            }

        # Normal pullback zone: 0.5 - 0.618
        r50 = retracements.get("0.5")
        r618 = retracements.get("0.618")

        if r50 is not None and r618 is not None:
            context["pullback_zone"] = {
                "min": min(r50, r618),
                "max": max(r50, r618),
                "label": "0.5 - 0.618"
            }

        # Golden Pocket (0.618)
        r618 = retracements.get("0.618")
        if r618 is not None:
            context["golden_pocket"] = {
                "price": r618,
                "label": "0.618 Golden Pocket"
            }

        # Deep pullback: 0.786
        r786 = retracements.get("0.786")

        if r786 is not None:
            context["deep_pullback"] = {
                "price": r786,
                "label": "0.786"
            }

        return context
    except Exception:
        return context


def calculate_fib_context_from_visible_swings(df, lookback=120, window=3):
    """
    Build Fibonacci context from the actual latest visible swing.
    Uses chronological swing points:
    - low -> high = Bullish visible swing
    - high -> low = Bearish visible swing

    This prevents bearish setups from incorrectly receiving bullish-only visible Fib context.
    """
    context = {
        "swing_start": None,
        "swing_end": None,
        "direction": "Unknown",
        "retracements": {},
        "extensions": {},
        "shallow_pullback_zone": None,
        "pullback_zone": None,
        "deep_pullback": None,
    }

    try:
        if df is None or df.empty:
            return context

        required = {"High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            return context

        swings = detect_swing_levels(df, lookback=lookback, window=window)
        swing_points = swings.get("swing_points", [])

        if not swing_points or len(swing_points) < 2:
            return context

        # Use latest swing point and nearest previous opposite-type swing point.
        last_point = swing_points[-1]
        previous_point = None

        for p in reversed(swing_points[:-1]):
            if p.get("type") != last_point.get("type"):
                previous_point = p
                break

        if previous_point is None:
            return context

        start = _safe_float(previous_point.get("price"))
        end = _safe_float(last_point.get("price"))

        if start is None or end is None or start == end:
            return context

        start_type = previous_point.get("type")
        end_type = last_point.get("type")

        if start_type == "low" and end_type == "high":
            direction = "Bullish visible swing"
        elif start_type == "high" and end_type == "low":
            direction = "Bearish visible swing"
        else:
            direction = "Unknown"

        move = end - start
        if abs(move) <= 0:
            return context

        context["swing_start"] = start
        context["swing_end"] = end
        context["direction"] = direction

        retracements = {}
        extensions = {}

        for fib in FIB_RETRACEMENTS:
            if move > 0:
                # Bullish swing: retracement levels are below swing high.
                level = end - (abs(move) * fib)
            else:
                # Bearish swing: retracement/bounce levels are above swing low.
                level = end + (abs(move) * fib)

            retracements[str(fib)] = level

        for fib in FIB_EXTENSIONS:
            # Standard extension measured from swing start.
            # 1.0 equals swing end.
            level = start + (
                move * fib
            )

            extensions[
                str(fib)
            ] = level

        context["retracements"] = retracements
        context["extensions"] = extensions

        r236 = retracements.get("0.236")
        r382 = retracements.get("0.382")
        if r236 is not None and r382 is not None:
            context["shallow_pullback_zone"] = {
                "min": min(r236, r382),
                "max": max(r236, r382),
                "label": "0.236 - 0.382"
            }

        r50 = retracements.get("0.5")
        r618 = retracements.get("0.618")
        if r50 is not None and r618 is not None:
            context["pullback_zone"] = {
                "min": min(r50, r618),
                "max": max(r50, r618),
                "label": "0.5 - 0.618"
            }

        # Golden Pocket (0.618)
        r618 = retracements.get("0.618")
        if r618 is not None:
            context["golden_pocket"] = {
                "price": r618,
                "label": "0.618 Golden Pocket"
            }

        r786 = retracements.get("0.786")
        if r786 is not None:
            context["deep_pullback"] = {
                "price": r786,
                "label": "0.786"
            }

        return context

    except Exception:
        return context


def calculate_volume_context(df, lookback=30):
    """
    Simple volume state.
    Current/latest candle is excluded from average calculation,
    so latest volume is compared against previous average.
    """
    try:
        if df is None or df.empty or "Volume" not in df.columns:
            return {
                "available": False,
                "state": "Volume data unavailable"
            }

        data = df.tail(lookback + 1).copy()

        if len(data) < 6:
            return {
                "available": False,
                "state": "Insufficient volume data"
            }

        # --- NEW: Smoothed Volume (3-Candle Average) ---
        # තනි කැන්ඩල් එකක බොරු Spike එකකට රැවටෙන්නේ නැති වෙන්න අන්තිම කැන්ඩල් 3ක සාමාන්‍යය ගමු.
        last_3_vol_avg = _safe_float(data["Volume"].iloc[-3:].mean())
        
        # ඒ කැන්ඩල් 3ට කලින් තිබුණු කැන්ඩල් 30ක සාමාන්‍යය
        prev_avg_vol = _safe_float(data["Volume"].iloc[:-3].mean())

        if last_3_vol_avg is None or prev_avg_vol is None or prev_avg_vol == 0:
            return {
                "available": False,
                "state": "Invalid volume data"
            }

        ratio = last_3_vol_avg / prev_avg_vol
        last_vol = last_3_vol_avg # ඉතුරු කෝඩ් එක වැඩ කරන්න මෙයාව පාවිච්චි වේවි.

        if ratio >= 1.8:
            state = "Very high volume"
        elif ratio >= 1.3:
            state = "High volume"
        elif ratio >= 1.1:
            state = "Above average volume"
        elif ratio <= 0.6:
            state = "Low volume"
        else:
            state = "Normal volume"

        # Lightweight health hint only. AI should not treat this as a hard rule.
        try:
            recent_closes = data["Close"].iloc[-6:]
            recent_vols = data["Volume"].iloc[-6:]

            price_change = _safe_float(recent_closes.iloc[-1] - recent_closes.iloc[0])

            vol_first_avg = _safe_float(recent_vols.iloc[:3].mean())
            vol_second_avg = _safe_float(recent_vols.iloc[3:].mean())

            volume_change = None
            if vol_first_avg is not None and vol_second_avg is not None:
                volume_change = vol_second_avg - vol_first_avg

            if price_change is not None and volume_change is not None:
                if price_change > 0 and volume_change > 0:
                    health = "Price rising with increasing volume"
                elif price_change > 0 and volume_change < 0:
                    health = "Price rising but volume weakening"
                elif price_change < 0 and volume_change > 0:
                    health = "Price falling with increasing volume"
                elif price_change < 0 and volume_change < 0:
                    health = "Price falling but selling volume weakening"
                else:
                    health = "Volume trend neutral"
            else:
                health = "Volume health unavailable"
        except Exception:
            health = "Volume health unavailable"

        return {
            "available": True,
            "last_volume": last_vol,
            "average_volume": prev_avg_vol,
            "ratio": ratio,
            "state": state,
            "health": health
        }
    except Exception:
        return {
            "available": False,
            "state": "Volume context error"
        }

def _zone_metadata_lines(
    label,
    zone,
):
    if not isinstance(
        zone,
        dict,
    ):
        return []

    score = _safe_float(
        zone.get("score")
    )

    origin = str(
        zone.get("origin_role")
        or zone.get("role")
        or "UNKNOWN"
    ).upper()

    flip_status = str(
        zone.get("flip_status")
        or (
            "CONFIRMED_RETEST"
            if zone.get(
                "polarity_flipped"
            )
            else "NATIVE"
        )
    ).upper()

    output = [
        (
            f"{label} Score: "
            f"{score:.4f}"
            if score is not None
            else f"{label} Score: N/A"
        ),

        f"{label} Origin: {origin}",

        (
            f"{label} Flip Status: "
            f"{flip_status}"
        ),
    ]

    if zone.get(
        "potential_flip"
    ):
        output.append(
            f"{label} Reliability: "
            "POTENTIAL ONLY — break confirmed, "
            "retest not yet confirmed."
        )

    return output


def _fib_overlaps_zone(
    fib_price,
    zone,
    current_price,
    atr,
    tolerance_pct,
):
    """
    Compare a Fib level with the full S/R zone width
    plus an ATR/dynamic-SR buffer.
    """
    try:
        fib_price = _safe_float(
            fib_price
        )

        if (
            fib_price is None
            or not isinstance(
                zone,
                dict,
            )
        ):
            return False

        zone_price = _safe_float(
            zone.get("price")
        )

        zone_min = _safe_float(
            zone.get("min")
        )

        zone_max = _safe_float(
            zone.get("max")
        )

        if zone_min is None:
            zone_min = zone_price

        if zone_max is None:
            zone_max = zone_price

        if (
            zone_min is None
            or zone_max is None
        ):
            return False

        current_price = _safe_float(
            current_price
        )

        atr = _safe_float(atr)

        tolerance_pct = (
            _safe_float(
                tolerance_pct
            )
            or 0.003
        )

        base_price = max(
            abs(
                current_price
                or fib_price
            ),
            1e-12,
        )

        buffer_value = max(
            base_price
            * tolerance_pct
            * 0.35,

            atr * 0.15
            if atr is not None
            and atr > 0
            else 0.0,

            1e-12,
        )

        lower = min(
            zone_min,
            zone_max,
        ) - buffer_value

        upper = max(
            zone_min,
            zone_max,
        ) + buffer_value

        return bool(
            lower
            <= fib_price
            <= upper
        )

    except Exception:
        return False

def build_market_context_text(df_micro, micro_waves, interval=None):
    """
    Final text block to inject into AI prompt.
    This gives AI better context without forcing final trade decisions.
    """
    try:
        if df_micro is None or df_micro.empty:
            return "Market context unavailable."

        current_price = _safe_float(df_micro["Close"].iloc[-1])
        atr = calculate_atr(df_micro, period=14)
        atr_pct = None
        if atr is not None and current_price is not None and current_price != 0:
            atr_pct = atr / current_price * 100

        sr = get_nearest_sr(
            df_micro,
            lookback=120,
            interval=interval,
        )
        major_structure = get_major_structure_context(
            df_micro,
            lookback=300,
            window=5,
            interval=interval,
        )
        fib = calculate_fib_context_from_waves(micro_waves)
        fib_status = "VALID"
        fib_warning = ""

        fib_start = _safe_float(fib.get("last_move_start"))
        fib_end = _safe_float(fib.get("last_move_end"))

        if current_price is not None and fib_start is not None and fib_end is not None:
            fib_move = fib_end - fib_start

            dist_start_pct = abs(fib_start - current_price) / current_price if current_price != 0 else 0
            dist_end_pct = abs(fib_end - current_price) / current_price if current_price != 0 else 0

            # Use timeframe + ATR based stale threshold.
            # If interval/atr are not available in this function scope, fallback values will be used.
            ctx_interval = interval if "interval" in locals() else None
            ctx_atr = atr if "atr" in locals() else None

            max_stale_pct = get_fib_stale_threshold(
                interval=ctx_interval,
                current_price=current_price,
                atr=ctx_atr
            )

            # If bullish swing start is broken, Elliott Fib is weak/invalid for long entry.
            if fib_move > 0 and current_price < fib_start:
                fib_status = "WEAK"
                fib_warning = "Price moved below bullish Elliott swing start. Do not use this Elliott Fib for long entry."

            # If bearish swing start is broken upward, Elliott Fib is weak/invalid for short entry.
            elif fib_move < 0 and current_price > fib_start:
                fib_status = "WEAK"
                fib_warning = "Price moved above bearish Elliott swing start. Do not use this Elliott Fib for short entry."

            # If both swing start and end are far from current price, this Fib context is stale.
            elif dist_start_pct > max_stale_pct and dist_end_pct > max_stale_pct:
                fib_status = "STALE"
                fib_warning = (
                    "Elliott Fib Context is STALE / far from current price "
                    "for this timeframe. "
                    f"Threshold used: {str(max_stale_pct * 100)}%. "
                    "Do not use for entry."
                )

        visible_fib = (
            calculate_fib_context_from_visible_swings(
                df_micro,
                lookback=120,
                window=3,
            )
        )

        visible_fib_status = (
            "UNAVAILABLE"
        )

        visible_fib_warning = ""

        visible_start = _safe_float(
            visible_fib.get(
                "swing_start"
            )
        )

        visible_end = _safe_float(
            visible_fib.get(
                "swing_end"
            )
        )

        if (
            current_price is not None
            and current_price > 0
            and visible_start is not None
            and visible_end is not None
        ):
            visible_move = (
                visible_end
                - visible_start
            )

            visible_stale_pct = (
                get_fib_stale_threshold(
                    interval=interval,
                    current_price=current_price,
                    atr=atr,
                )
            )

            visible_start_distance = (
                abs(
                    current_price
                    - visible_start
                )
                / current_price
            )

            visible_end_distance = (
                abs(
                    current_price
                    - visible_end
                )
                / current_price
            )

            if (
                visible_move > 0
                and current_price
                < visible_start
            ):
                visible_fib_status = (
                    "BROKEN"
                )

                visible_fib_warning = (
                    "Price moved below the bullish "
                    "visible-swing start. Ignore "
                    "these Fib levels for long entry."
                )

            elif (
                visible_move < 0
                and current_price
                > visible_start
            ):
                visible_fib_status = (
                    "BROKEN"
                )

                visible_fib_warning = (
                    "Price moved above the bearish "
                    "visible-swing start. Ignore "
                    "these Fib levels for short entry."
                )

            elif (
                visible_start_distance
                > visible_stale_pct
                and visible_end_distance
                > visible_stale_pct
            ):
                visible_fib_status = (
                    "STALE"
                )

                visible_fib_warning = (
                    "Visible-swing Fib is too far "
                    "from current price for this "
                    "timeframe."
                )

            else:
                visible_fib_status = (
                    "VALID"
                )

        visible_fib["status"] = (
            visible_fib_status
        )

        visible_fib["warning"] = (
            visible_fib_warning
        )

        volume = calculate_volume_context(
            df_micro,
            lookback=30,
        )
        ema_context = calculate_ema_context(df_micro)

        nearest_support = sr.get("nearest_support")
        nearest_resistance = sr.get("nearest_resistance")
        major_support = sr.get("major_support")
        major_resistance = sr.get("major_resistance")
        major_swing_high = major_structure.get("major_swing_high")
        major_swing_low = major_structure.get("major_swing_low")
        long_range_high = major_structure.get("major_range_high")
        long_range_low = major_structure.get("major_range_low")
        long_major_support = major_structure.get("major_support")
        long_major_resistance = major_structure.get("major_resistance")

        fib_shallow = fib.get("shallow_pullback_zone")
        fib_pullback = fib.get("pullback_zone")
        fib_deep = fib.get("deep_pullback")
        fib_golden = fib.get("golden_pocket")
        visible_fib_shallow = visible_fib.get("shallow_pullback_zone")
        visible_fib_pullback = visible_fib.get("pullback_zone")
        visible_fib_golden = visible_fib.get("golden_pocket")
        visible_fib_deep = visible_fib.get("deep_pullback")

        lines = []

        lines.append("--- MARKET STRUCTURE CONTEXT ---")
        lines.append(f"Current Price: {_format_price(current_price)}")

        if atr is not None:
            if atr_pct is not None:
                lines.append(f"ATR(14): {_format_price(atr)} ({atr_pct:.2f}% of price)")
            else:
                lines.append(f"ATR(14): {_format_price(atr)}")
        else:
            lines.append("ATR(14): N/A")

        if sr.get("tolerance_pct") is not None:
            lines.append(f"Dynamic S/R Grouping Tolerance: {sr.get('tolerance_pct') * 100:.2f}%")

        lines.append(f"Recent Swing High: {_format_price(sr.get('recent_swing_high'))}")
        lines.append(f"Recent Swing Low: {_format_price(sr.get('recent_swing_low'))}")

        if nearest_support:
            lines.append(
                f"Nearest Support: {_format_price(nearest_support.get('price'))} "
                f"(strength {nearest_support.get('strength')})"
            )
        else:
            lines.append("Nearest Support: N/A")

        if nearest_resistance:
            lines.append(
                f"Nearest Resistance: {_format_price(nearest_resistance.get('price'))} "
                f"(strength {nearest_resistance.get('strength')})"
            )
        else:
            lines.append("Nearest Resistance: N/A")

        if major_support:
            lines.append(
                f"Major Support Zone: {_format_price(major_support.get('min'))} - {_format_price(major_support.get('max'))} "
                f"(strength {major_support.get('strength')})"
            )
        else:
            lines.append("Major Support Zone: N/A")

        if major_resistance:
            lines.append(
                f"Major Resistance Zone: {_format_price(major_resistance.get('min'))} - {_format_price(major_resistance.get('max'))} "
                f"(strength {major_resistance.get('strength')})"
            )
        else:
            lines.append("Major Resistance Zone: N/A")


        lines.append("")
        lines.append("--- LARGER STRUCTURE CONTEXT ---")
        lines.append(f"Long Lookback Range High: {_format_price(long_range_high)}")
        lines.append(f"Long Lookback Range Low: {_format_price(long_range_low)}")
        lines.append(f"Major Swing High / Possible Impulse Peak: {_format_price(major_swing_high)}")
        lines.append(f"Major Swing Low / Possible Correction Base: {_format_price(major_swing_low)}")

        if long_major_support:
            lines.append(
                f"Long Lookback Major Support Zone: "
                f"{_format_price(long_major_support.get('min'))} - {_format_price(long_major_support.get('max'))} "
                f"(strength {long_major_support.get('strength')})"
            )
        else:
            lines.append("Long Lookback Major Support Zone: N/A")

        if long_major_resistance:
            lines.append(
                f"Long Lookback Major Resistance Zone: "
                f"{_format_price(long_major_resistance.get('min'))} - {_format_price(long_major_resistance.get('max'))} "
                f"(strength {long_major_resistance.get('strength')})"
            )
        else:
            lines.append("Long Lookback Major Resistance Zone: N/A")

        lines.append("")
        lines.append(
            "--- STRUCTURAL ZONE QUALITY ---"
        )

        zone_metadata_items = [
            (
                "Nearest Support",
                nearest_support,
            ),
            (
                "Nearest Resistance",
                nearest_resistance,
            ),
            (
                "Major Support",
                major_support,
            ),
            (
                "Major Resistance",
                major_resistance,
            ),
            (
                "Long Lookback Major Support",
                long_major_support,
            ),
            (
                "Long Lookback Major Resistance",
                long_major_resistance,
            ),
        ]

        metadata_found = False

        for (
            zone_label,
            zone_value,
        ) in zone_metadata_items:
            metadata_lines = (
                _zone_metadata_lines(
                    zone_label,
                    zone_value,
                )
            )

            if metadata_lines:
                metadata_found = True

                lines.extend(
                    metadata_lines
                )

        if not metadata_found:
            lines.append(
                "Structural zone quality: N/A"
            )

        lines.append("")
        lines.append("--- FIBONACCI CONTEXT FROM LATEST MICRO SWING ---")
        lines.append(f"Elliott Fib Context Status: {fib_status}")
        if fib_warning:
            lines.append(f"Warning: {fib_warning}")
        lines.append(f"Last Swing Direction: {fib.get('direction', 'Unknown')}")
        lines.append(f"Last Swing Start: {_format_price(fib.get('last_move_start'))}")
        lines.append(f"Last Swing End: {_format_price(fib.get('last_move_end'))}")

        if fib_shallow:
            lines.append(
                f"Shallow Pullback Zone ({fib_shallow.get('label')}): "
                f"{_format_price(fib_shallow.get('min'))} - {_format_price(fib_shallow.get('max'))}"
            )
        else:
            lines.append("Shallow Pullback Zone: N/A")

        if fib_pullback:
            lines.append(
                f"Normal Pullback Zone ({fib_pullback.get('label')}): "
                f"{_format_price(fib_pullback.get('min'))} - {_format_price(fib_pullback.get('max'))}"
            )
            
        else:
            lines.append("Normal Pullback Zone: N/A")

        if fib_golden:
            lines.append(f"Golden Pocket ({fib_golden.get('label')}): {_format_price(fib_golden.get('price'))}")
        else:
            lines.append("Golden Pocket: N/A")

        if fib_deep:
            lines.append(f"Deep Pullback ({fib_deep.get('label')}): {_format_price(fib_deep.get('price'))}")
        else:
            lines.append("Deep Pullback: N/A")

        extensions = fib.get("extensions", {})

        if extensions:
            ext_text = []
            for k in ["1.0", "1.272", "1.618", "2.0"]:
                if k in extensions:
                    ext_text.append(f"{k}: {_format_price(extensions[k])}")

            lines.append("Continuation Extension Levels From Last Swing End: " + " | ".join(ext_text))
        else:
            lines.append("Continuation Extension Levels From Last Swing End: N/A")


        lines.append("")
        lines.append(
            "--- FIBONACCI CONTEXT FROM "
            "RECENT VISIBLE SWING HIGH/LOW ---"
        )

        lines.append(
            f"Visible Fib Context Status: "
            f"{visible_fib_status}"
        )

        if visible_fib_warning:
            lines.append(
                f"Visible Fib Warning: "
                f"{visible_fib_warning}"
            )

        lines.append(
            f"Visible Swing Direction: "
            f"{visible_fib.get('direction', 'Unknown')}"
        )
        lines.append(f"Visible Swing Start: {_format_price(visible_fib.get('swing_start'))}")
        lines.append(f"Visible Swing End: {_format_price(visible_fib.get('swing_end'))}")

        if visible_fib_shallow:
            lines.append(
                f"Visible Shallow Pullback Zone ({visible_fib_shallow.get('label')}): "
                f"{_format_price(visible_fib_shallow.get('min'))} - {_format_price(visible_fib_shallow.get('max'))}"
            )
        else:
            lines.append("Visible Shallow Pullback Zone: N/A")

        if visible_fib_pullback:
            lines.append(
                f"Visible Normal Pullback Zone ({visible_fib_pullback.get('label')}): "
                f"{_format_price(visible_fib_pullback.get('min'))} - {_format_price(visible_fib_pullback.get('max'))}"
            )
            
        else:
            lines.append("Visible Normal Pullback Zone: N/A")

        if visible_fib_golden:
            lines.append(f"Visible Golden Pocket ({visible_fib_golden.get('label')}): {_format_price(visible_fib_golden.get('price'))}")
        else:
            lines.append("Visible Golden Pocket: N/A")

        if visible_fib_deep:
            lines.append(f"Visible Deep Pullback ({visible_fib_deep.get('label')}): {_format_price(visible_fib_deep.get('price'))}")
        else:
            lines.append("Visible Deep Pullback: N/A")

        visible_extensions = visible_fib.get("extensions", {})
        if visible_extensions:
            visible_ext_text = []
            for k in ["1.0", "1.272", "1.618", "2.0"]:
                if k in visible_extensions:
                    visible_ext_text.append(f"{k}: {_format_price(visible_extensions[k])}")

            lines.append("Visible Continuation Extension Levels: " + " | ".join(visible_ext_text))
        else:
            lines.append("Visible Continuation Extension Levels: N/A")

        # Volume context is intentionally NOT sent to AI for fresh trade decisions.
        # Locked/running trades use last-3-candle volume in main.generate_dynamic_management_strategy().

        lines.append("")
        # --- NEW: ENTRY DEPTH CONTEXT FOR AI ---
        impulse_size = None
        atr_multiple = None
        shallow_risk = "LOW"
        pref_short = "Any valid structural resistance"
        pref_long = "Any valid structural support"

        recent_high = _safe_float(sr.get("recent_swing_high"))
        recent_low = _safe_float(sr.get("recent_swing_low"))

        if recent_high is not None and recent_low is not None and atr and atr > 0:
            impulse_size = abs(recent_high - recent_low)
            atr_multiple = impulse_size / atr
            
            if atr_multiple > 2.0:
                shallow_risk = "HIGH (Relief Bounce Trap likely)"
                pref_short = "Wait for Normal/Deep pullback or Major Resistance ONLY"
                pref_long = "Wait for Normal/Deep pullback or Major Support ONLY"
            elif atr_multiple > 1.5:
                shallow_risk = "MEDIUM"
                pref_short = "Prefer Normal Pullback or Strong Confluence"
                pref_long = "Prefer Normal Pullback or Strong Confluence"

        lines.append("")
        lines.append("--- ENTRY DEPTH CONTEXT ---")
        if atr_multiple is not None:
            lines.append(f"Recent Impulse Size: {_format_price(impulse_size)}")
            lines.append(f"Recent Impulse ATR Multiple: {atr_multiple:.2f}x")
            lines.append(f"Shallow Entry Risk: {shallow_risk}")
            lines.append(f"Preferred SHORT Entry Depth: {pref_short}")
            lines.append(f"Preferred LONG Entry Depth: {pref_long}")
        else:
            lines.append("Recent Impulse Data: Unavailable")
        lines.append("--- EMA CONTEXT ---")

        if ema_context.get("available"):
            lines.append(f"EMA50: {_format_price(ema_context.get('ema50'))}")

            if ema_context.get("ema200") is not None:
                lines.append(f"EMA200: {_format_price(ema_context.get('ema200'))}")
            else:
                lines.append("EMA200: Unavailable")

            lines.append(f"Price vs EMA50: {ema_context.get('price_vs_ema50', 'Unknown')}")
            lines.append(f"Price vs EMA200: {ema_context.get('price_vs_ema200', 'Unknown')}")
            lines.append(f"EMA50 vs EMA200: {ema_context.get('ema50_vs_ema200', 'Unknown')}")
            lines.append(f"EMA Trend Bias: {ema_context.get('trend_bias', 'Unknown')}")
        else:
            lines.append(f"EMA Context: {ema_context.get('state', 'Unavailable')}")
        
        # --- DYNAMIC S/R + FIB CONFLUENCE CHECK ---
        confluence_found = False
        confluence_notes = []

        major_sr_zones = [
            (
                "Major Support",
                major_support,
            ),
            (
                "Major Resistance",
                major_resistance,
            ),
            (
                "Long Lookback Major Support",
                long_major_support,
            ),
            (
                "Long Lookback Major Resistance",
                long_major_resistance,
            ),
        ]

        fib_levels_to_check = []

        if (
            visible_fib_status
            == "VALID"
        ):
            if visible_fib.get(
                "deep_pullback"
            ):
                fib_levels_to_check.append(
                    (
                        "0.786",
                        visible_fib[
                            "deep_pullback"
                        ]["price"],
                    )
                )

            if visible_fib.get(
                "golden_pocket"
            ):
                fib_levels_to_check.append(
                    (
                        "0.618",
                        visible_fib[
                            "golden_pocket"
                        ]["price"],
                    )
                )

        for (
            fib_name,
            fib_price,
        ) in fib_levels_to_check:
            for (
                zone_name,
                zone,
            ) in major_sr_zones:
                if _fib_overlaps_zone(
                    fib_price=fib_price,
                    zone=zone,
                    current_price=current_price,
                    atr=atr,
                    tolerance_pct=sr.get(
                        "tolerance_pct"
                    ),
                ):
                    confluence_found = True

                    confluence_notes.append(
                        f"{zone_name} overlaps "
                        f"the {fib_name} visible Fib "
                        f"near {_format_price(fib_price)}."
                    )

        lines.append("")
        lines.append(
            "--- AUTOMATED CONFLUENCE CHECK ---"
        )

        if confluence_found:
            lines.append(
                "STATUS: HIGH CONFLUENCE DETECTED"
            )

            for note in confluence_notes:
                lines.append(
                    f"- {note}"
                )

        elif visible_fib_status in {
            "BROKEN",
            "STALE",
            "UNAVAILABLE",
        }:
            lines.append(
                "STATUS: Visible Fib is not "
                "valid enough for automated "
                "confluence."
            )

        else:
            lines.append(
                "STATUS: No strong S/R + "
                "Fibonacci overlap detected."
            )

        lines.append("")
        lines.append("IMPORTANT USAGE INSTRUCTION:")
        lines.append(
            "- In price discovery, use extension "
            "levels only when momentum, historical "
            "path, multi-timeframe structure, and "
            "validator risk limits also support them."
        )
        lines.append(
            "- Elliott structure is one major evidence "
            "source. Do not let it override stronger "
            "contradictory multi-timeframe structure, "
            "S/R, momentum, liquidity, or historical "
            "evidence."
        )
        lines.append(
            "- BREAK_CONFIRMED_RETEST_PENDING means "
            "potential support/resistance only. Treat "
            "CONFIRMED_RETEST as the stronger polarity "
            "flip evidence."
        )
        lines.append(
            "- Do not use visible Fib levels marked "
            "BROKEN or STALE as entry anchors."
        )
        lines.append("- Do not create a trade only from this market context.")
        lines.append("- Use these levels as supporting context, not forced trade signals.")
        lines.append("- Use Larger Structure Context to identify major impulse peaks, major resistance, and larger correction zones.")
        lines.append("- Entry should be near support/resistance or one of the printed Fibonacci pullback zones.")
        lines.append("- Do not claim Fibonacci confluence unless entry is close to a printed Fibonacci level/zone.")
        lines.append("- If Elliott-based Fibonacci and visible-swing Fibonacci conflict, prioritize the one that matches current price action and explain it.")
        lines.append("- Stop loss should be beyond nearest valid structural invalidation, not exactly on the level.")
        lines.append("- ATR can be used as a buffer/check for stop loss placement, often around 1.5x to 2x ATR when appropriate.")
        lines.append("- Use ATR to judge whether the stop loss is too wide or too tight; do not set SL mechanically from ATR alone.")
        lines.append("- TP2 should respect nearby resistance/support or continuation extension levels.")
        lines.append("- If current price is already close to TP1, return WAIT / NO CHASE.")
        lines.append("- Latest candle may be incomplete depending on data source; treat current price/volume context carefully.")
        lines.append("- If Elliott Fib Context Status is STALE or WEAK, ignore it for entry and prioritize visible swing / larger structure context.")
        lines.append("- EMA50/EMA200 context is only a supporting trend filter; do not create or reject trades from EMA alone.")

        return "\n".join(lines)

    except Exception as e:
        return f"Market context error: {str(e)}"