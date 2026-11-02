import re
import math


def _extract_float(val):
    """
    Safely extract first numeric value from int/float/string.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        try:
            if math.isnan(float(val)):
                return None
        except Exception:
            pass
        return float(val)

    if isinstance(val, str):
        try:
            text = val.replace(",", "").strip()
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if match:
                return float(match.group(0))
        except Exception:
            return None

    return None


def _format_price(value):
    """
    Dynamic price formatting for BTC/ETH and small coins.
    """
    value = _extract_float(value)
    if value is None:
        return "N/A"

    if abs(value) >= 1000:
        return f"{value:.2f}"
    elif abs(value) >= 100:
        return f"{value:.2f}"
    elif abs(value) >= 10:
        return f"{value:.3f}"
    elif abs(value) >= 1:
        return f"{value:.4f}"
    elif abs(value) >= 0.01:
        return f"{value:.4f}"
    elif abs(value) >= 0.0001:
        return f"{value:.6f}"
    else:
        return f"{value:.8f}"


def _extract_context_line(market_context_text, label):
    """
    Extract full text after a market context label.
    Example:
      Major Resistance Zone: 0.12460000 - 0.12520000 (strength 4)
    """
    try:
        if not market_context_text:
            return ""

        pattern = rf"{re.escape(label)}:\s*(.+)"
        match = re.search(pattern, str(market_context_text))

        if match:
            return match.group(1).strip()
    except Exception:
        pass

    return ""


def _extract_context_number(market_context_text, label):
    """
    Extract numeric value after a label.
    Example:
      ATR(14): 0.00055381
    """
    try:
        if not market_context_text:
            return None

        pattern = rf"{re.escape(label)}:\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, str(market_context_text))

        if match:
            return float(match.group(1))
    except Exception:
        pass

    return None


def _extract_context_zone_bounds(market_context_text, label):
    """
    Extract low/high numeric bounds from context.
    Handles:
      Major Resistance Zone: 0.12460000 - 0.12520000 (strength 4)
      Nearest Resistance: 0.12620000 (strength 1)
    """
    try:
        line = _extract_context_line(market_context_text, label)
        if not line:
            return None

        # Prefer explicit range before reading any other numbers like strength.
        range_match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)",
            line
        )
        if range_match:
            a = _extract_float(range_match.group(1))
            b = _extract_float(range_match.group(2))
            if a is not None and b is not None:
                return (min(a, b), max(a, b))

        single_match = re.search(r"(-?\d+(?:\.\d+)?)", line)
        if single_match:
            v = _extract_float(single_match.group(1))
            if v is not None:
                return (v, v)

    except Exception:
        pass

    return None


def _get_first_zone_bounds(market_context_text, labels):
    """
    Try labels in priority order and return first valid zone.
    """
    try:
        for label in labels:
            zone = _extract_context_zone_bounds(market_context_text, label)
            if zone is not None:
                return zone
    except Exception:
        pass

    return None


def _format_zone(zone):
    if not zone:
        return "N/A"

    low, high = zone

    if abs(low - high) <= 0:
        return _format_price(low)

    return f"{_format_price(low)} - {_format_price(high)}"


def _get_row_value(row, column_name):
    """
    Get value from pandas Series row using case-insensitive column name.
    Supports Open/High/Low/Close or open/high/low/close.
    """
    try:
        if column_name in row:
            return row[column_name]

        wanted = str(column_name).lower()
        for col in getattr(row, "index", []):
            if str(col).lower() == wanted:
                return row[col]
    except Exception:
        pass

    return None


def _get_row_time(row, idx):
    """
    Prefer Date column if exists, otherwise dataframe index.
    """
    try:
        date_value = _get_row_value(row, "Date")
        if date_value is not None:
            return str(date_value)
    except Exception:
        pass

    try:
        return str(idx)
    except Exception:
        return "Unknown"


def _safe_ohlc(row):
    """
    Extract O/H/L/C floats from a row.
    """
    try:
        o = _extract_float(_get_row_value(row, "Open"))
        h = _extract_float(_get_row_value(row, "High"))
        l = _extract_float(_get_row_value(row, "Low"))
        c = _extract_float(_get_row_value(row, "Close"))

        if o is None or h is None or l is None or c is None:
            return None

        # Safety normalize
        high = max(h, o, c, l)
        low = min(l, o, c, h)

        return {
            "open": o,
            "high": high,
            "low": low,
            "close": c
        }
    except Exception:
        return None


def _candle_metrics(ohlc):
    """
    Calculate candle body/wick details.
    """
    try:
        o = ohlc["open"]
        h = ohlc["high"]
        l = ohlc["low"]
        c = ohlc["close"]

        candle_range = max(h - l, 0)
        body = abs(c - o)

        upper_wick = max(h - max(o, c), 0)
        lower_wick = max(min(o, c) - l, 0)

        if candle_range > 0:
            close_pos = (c - l) / candle_range
        else:
            close_pos = 0.5

        if c > o:
            direction = "Bullish"
        elif c < o:
            direction = "Bearish"
        else:
            direction = "Doji"

        if close_pos >= 0.67:
            close_position = "near high"
        elif close_pos <= 0.33:
            close_position = "near low"
        else:
            close_position = "middle"

        return {
            "range": candle_range,
            "body": body,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "close_position_ratio": close_pos,
            "close_position": close_position,
            "direction": direction
        }
    except Exception:
        return {
            "range": 0,
            "body": 0,
            "upper_wick": 0,
            "lower_wick": 0,
            "close_position_ratio": 0.5,
            "close_position": "middle",
            "direction": "Unknown"
        }


def _zone_margin(current_price, atr, zone=None):
    """
    Small practical tolerance around zones.

    Uses:
    - 10% of ATR
    - or 0.05% of price
    Whichever is larger.

    Keeps zone checks practical for low-price coins.
    """
    try:
        price = _extract_float(current_price) or 0
        atr_val = _extract_float(atr) or 0

        atr_margin = atr_val * 0.10
        price_margin = abs(price) * 0.0005

        margin = max(atr_margin, price_margin)

        # Avoid making the margin too huge compared to a real zone width.
        if zone is not None:
            low, high = zone
            zone_width = abs(high - low)

            if zone_width > 0:
                max_reasonable = max(zone_width * 0.40, abs(price) * 0.0002)
                margin = min(margin, max_reasonable)

        return max(margin, 0)
    except Exception:
        return 0


def _candle_overlaps_zone(ohlc, zone, margin=0):
    """
    True if candle wick/body range overlaps a zone.
    """
    try:
        if not ohlc or zone is None:
            return False

        low, high = zone
        candle_high = ohlc["high"]
        candle_low = ohlc["low"]

        return candle_high >= (low - margin) and candle_low <= (high + margin)
    except Exception:
        return False


def _price_in_or_near_zone(price, zone, margin=0):
    try:
        price = _extract_float(price)
        if price is None or zone is None:
            return False

        low, high = zone
        return (low - margin) <= price <= (high + margin)
    except Exception:
        return False


def _price_broke_above_zone(price, zone, margin=0):
    try:
        price = _extract_float(price)
        if price is None or zone is None:
            return False

        return price > zone[1] + margin
    except Exception:
        return False


def _price_broke_below_zone(price, zone, margin=0):
    try:
        price = _extract_float(price)
        if price is None or zone is None:
            return False

        return price < zone[0] - margin
    except Exception:
        return False


def _detect_resistance_rejection(ohlc, metrics, resistance_zone, margin=0):
    """
    Bearish rejection confirmation from resistance.

    Early reversal allowed when:
    - candle touches/overlaps resistance zone
    - candle does NOT close clearly above the resistance zone
    - candle closes below zone midpoint/lower area
    - candle shape shows bearish rejection OR upper wick pressure
    """
    try:
        if resistance_zone is None or not ohlc:
            return False

        if not _candle_overlaps_zone(ohlc, resistance_zone, margin):
            return False

        zone_low, zone_high = resistance_zone
        zone_mid = (zone_low + zone_high) / 2.0
        close = ohlc["close"]
        open_ = ohlc["open"]

        # If it closed clearly above resistance, rejection failed.
        if close > zone_high + margin:
            return False

        bearish_close = close < open_
        closes_low_enough = close <= zone_mid + margin
        strong_upper_wick = metrics["upper_wick"] >= max(metrics["body"] * 0.50, metrics["range"] * 0.12)
        close_near_low = metrics["close_position_ratio"] <= 0.40

        return closes_low_enough and (bearish_close or strong_upper_wick or close_near_low)

    except Exception:
        return False


def _detect_support_rejection(ohlc, metrics, support_zone, margin=0):
    """
    Bullish rejection confirmation from support.

    Early reversal allowed when:
    - candle touches/overlaps support zone
    - candle does NOT close clearly below the support zone
    - candle closes above zone midpoint/upper area
    - candle shape shows bullish rejection OR lower wick pressure
    """
    try:
        if support_zone is None or not ohlc:
            return False

        if not _candle_overlaps_zone(ohlc, support_zone, margin):
            return False

        zone_low, zone_high = support_zone
        zone_mid = (zone_low + zone_high) / 2.0
        close = ohlc["close"]
        open_ = ohlc["open"]

        # If it closed clearly below support, rejection failed.
        if close < zone_low - margin:
            return False

        bullish_close = close > open_
        closes_high_enough = close >= zone_mid - margin
        strong_lower_wick = metrics["lower_wick"] >= max(metrics["body"] * 0.50, metrics["range"] * 0.12)
        close_near_high = metrics["close_position_ratio"] >= 0.60

        return closes_high_enough and (bullish_close or strong_lower_wick or close_near_high)

    except Exception:
        return False


def _zone_state_for_price(price, zone, margin=0, zone_type="resistance"):
    """
    Describe current/latest available price position vs zone.
    """
    try:
        price = _extract_float(price)
        if price is None or zone is None:
            return "UNKNOWN"

        if _price_in_or_near_zone(price, zone, margin):
            return "INSIDE_OR_NEAR"

        if zone_type == "resistance":
            if _price_broke_above_zone(price, zone, margin):
                return "BROKE_ABOVE"
            return "BELOW_ZONE"

        if zone_type == "support":
            if _price_broke_below_zone(price, zone, margin):
                return "BROKE_BELOW"
            return "ABOVE_ZONE"

    except Exception:
        pass

    return "UNKNOWN"


def _yes_no(value):
    return "YES" if bool(value) else "NO"


def build_recent_candle_rejection_flags(
    df,
    market_context_text="",
    use_last_closed=True
):
    """
    Main machine-readable function.

    Returns a dict with:
    - latest available price
    - last closed candle OHLC
    - wick/body metrics
    - support/resistance zones
    - rejection confirmation flags

    use_last_closed=True:
      If df contains an incomplete latest candle, use df.iloc[-2] as confirmation candle
      and df.iloc[-1] as latest available/current candle.
    """
    result = {
        "ok": False,
        "reason": "",
        "last_closed_time": "Unknown",
        "latest_available_time": "Unknown",
        "latest_available_price": None,

        "last_open": None,
        "last_high": None,
        "last_low": None,
        "last_close": None,

        "direction": "Unknown",
        "close_position": "Unknown",
        "upper_wick": None,
        "lower_wick": None,
        "body": None,
        "range": None,

        "support_zone": None,
        "resistance_zone": None,
        "support_zone_text": "N/A",
        "resistance_zone_text": "N/A",

        "support_rejection_confirmed": False,
        "resistance_rejection_confirmed": False,

        "support_zone_state": "UNKNOWN",
        "resistance_zone_state": "UNKNOWN",

        "both_rejections": False,
        "no_rejection": True,

        "margin": 0,
        "notes": []
    }

    try:
        if df is None or getattr(df, "empty", True):
            result["reason"] = "No dataframe supplied."
            return result

        if len(df) >= 2 and use_last_closed:
            closed_idx = df.index[-2]
            closed_row = df.iloc[-2]

            latest_idx = df.index[-1]
            latest_row = df.iloc[-1]
        else:
            closed_idx = df.index[-1]
            closed_row = df.iloc[-1]

            latest_idx = df.index[-1]
            latest_row = df.iloc[-1]

        closed_ohlc = _safe_ohlc(closed_row)
        latest_ohlc = _safe_ohlc(latest_row)

        if not closed_ohlc or not latest_ohlc:
            result["reason"] = "Could not read OHLC columns."
            return result

        current_price = _extract_context_number(market_context_text, "Current Price")
        if current_price is None:
            current_price = latest_ohlc["close"]

        atr = _extract_context_number(market_context_text, "ATR(14)")

        support_zone = _get_first_zone_bounds(
            market_context_text,
            ["Major Support Zone", "Long Lookback Major Support Zone", "Nearest Support"]
        )

        resistance_zone = _get_first_zone_bounds(
            market_context_text,
            ["Major Resistance Zone", "Long Lookback Major Resistance Zone", "Nearest Resistance"]
        )

        metrics = _candle_metrics(closed_ohlc)

        support_margin = _zone_margin(current_price, atr, support_zone)
        resistance_margin = _zone_margin(current_price, atr, resistance_zone)
        margin = max(support_margin, resistance_margin)

        support_rejection = _detect_support_rejection(
            closed_ohlc,
            metrics,
            support_zone,
            support_margin
        )

        resistance_rejection = _detect_resistance_rejection(
            closed_ohlc,
            metrics,
            resistance_zone,
            resistance_margin
        )

        support_state = _zone_state_for_price(
            current_price,
            support_zone,
            support_margin,
            zone_type="support"
        )

        resistance_state = _zone_state_for_price(
            current_price,
            resistance_zone,
            resistance_margin,
            zone_type="resistance"
        )

        notes = []

        if resistance_zone is not None:
            if resistance_rejection:
                notes.append("Bearish rejection from resistance confirmed on last closed candle.")
            elif _candle_overlaps_zone(closed_ohlc, resistance_zone, resistance_margin):
                notes.append("Resistance was touched, but bearish rejection is not confirmed.")

        if support_zone is not None:
            if support_rejection:
                notes.append("Bullish rejection from support confirmed on last closed candle.")
            elif _candle_overlaps_zone(closed_ohlc, support_zone, support_margin):
                notes.append("Support was touched, but bullish rejection is not confirmed.")

        if resistance_state == "BROKE_ABOVE":
            notes.append("Latest price is above resistance zone; short rejection may have failed.")

        if support_state == "BROKE_BELOW":
            notes.append("Latest price is below support zone; long rejection may have failed.")

        result.update({
            "ok": True,
            "reason": "OK",

            "last_closed_time": _get_row_time(closed_row, closed_idx),
            "latest_available_time": _get_row_time(latest_row, latest_idx),
            "latest_available_price": current_price,

            "last_open": closed_ohlc["open"],
            "last_high": closed_ohlc["high"],
            "last_low": closed_ohlc["low"],
            "last_close": closed_ohlc["close"],

            "direction": metrics["direction"],
            "close_position": metrics["close_position"],
            "upper_wick": metrics["upper_wick"],
            "lower_wick": metrics["lower_wick"],
            "body": metrics["body"],
            "range": metrics["range"],

            "support_zone": support_zone,
            "resistance_zone": resistance_zone,
            "support_zone_text": _format_zone(support_zone),
            "resistance_zone_text": _format_zone(resistance_zone),

            "support_rejection_confirmed": support_rejection,
            "resistance_rejection_confirmed": resistance_rejection,

            "support_zone_state": support_state,
            "resistance_zone_state": resistance_state,

            "both_rejections": bool(support_rejection and resistance_rejection),
            "no_rejection": bool(not support_rejection and not resistance_rejection),

            "margin": margin,
            "notes": notes
        })

        return result

    except Exception as e:
        result["reason"] = f"Failed to build candle context: {e}"
        return result


def build_recent_candle_rejection_context(
    df,
    market_context_text="",
    use_last_closed=True
):
    """
    Human/AI-readable context text.

    Add this text to the AI prompt and/or append to market_context_text.
    Signal quality backend can parse:
      Resistance Rejection Confirmed: YES/NO
      Support Rejection Confirmed: YES/NO
      Resistance Zone State: ...
      Support Zone State: ...
    """
    flags = build_recent_candle_rejection_flags(
        df=df,
        market_context_text=market_context_text,
        use_last_closed=use_last_closed
    )

    if not flags.get("ok"):
        return (
            "--- RECENT CANDLE / REJECTION CONTEXT ---\n"
            f"Status: unavailable ({flags.get('reason', 'Unknown error')})\n"
        )

    notes = flags.get("notes") or []
    notes_text = " | ".join(notes) if notes else "No special rejection note."

    return f"""--- RECENT CANDLE / REJECTION CONTEXT ---
Last Closed Candle Time: {flags.get("last_closed_time")}
Last Closed Candle O/H/L/C: {_format_price(flags.get("last_open"))} / {_format_price(flags.get("last_high"))} / {_format_price(flags.get("last_low"))} / {_format_price(flags.get("last_close"))}
Latest Available Candle Time: {flags.get("latest_available_time")}
Latest Available Price: {_format_price(flags.get("latest_available_price"))}
Candle Direction: {flags.get("direction")}
Close Position: {flags.get("close_position")}
Upper Wick: {_format_price(flags.get("upper_wick"))}
Lower Wick: {_format_price(flags.get("lower_wick"))}
Candle Body: {_format_price(flags.get("body"))}
Candle Range: {_format_price(flags.get("range"))}
Support Zone Used: {flags.get("support_zone_text")}
Resistance Zone Used: {flags.get("resistance_zone_text")}
Support Zone State: {flags.get("support_zone_state")}
Resistance Zone State: {flags.get("resistance_zone_state")}
Support Rejection Confirmed: {_yes_no(flags.get("support_rejection_confirmed"))}
Resistance Rejection Confirmed: {_yes_no(flags.get("resistance_rejection_confirmed"))}
Both Rejections Confirmed: {_yes_no(flags.get("both_rejections"))}
No Rejection Confirmed: {_yes_no(flags.get("no_rejection"))}
Zone Check Margin: {_format_price(flags.get("margin"))}
Rejection Notes: {notes_text}
"""


__all__ = [
    "build_recent_candle_rejection_flags",
    "build_recent_candle_rejection_context",
]