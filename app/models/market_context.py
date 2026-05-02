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


def detect_swing_levels(df, lookback=120, window=3):
    """
    Basic swing high / swing low detection.
    Returns:
    - swing_highs / swing_lows for S/R grouping
    - recent_swing_high / recent_swing_low
    - swing_points with type + date + position for visible Fib direction detection
    """
    result = {
        "swing_highs": [],
        "swing_lows": [],
        "recent_swing_high": None,
        "recent_swing_low": None,
        "swing_points": [],
    }

    try:
        if df is None or df.empty:
            return result

        required = {"High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            return result

        data = df.tail(lookback).copy()

        if len(data) < window * 2 + 5:
            return result

        highs = []
        lows = []
        swing_points = []

        for i in range(window, len(data) - window):
            curr_high = data["High"].iloc[i]
            curr_low = data["Low"].iloc[i]
            candle_index = data.index[i]

            left_highs = data["High"].iloc[i - window:i]
            right_highs = data["High"].iloc[i + 1:i + 1 + window]

            left_lows = data["Low"].iloc[i - window:i]
            right_lows = data["Low"].iloc[i + 1:i + 1 + window]

            if curr_high > left_highs.max() and curr_high > right_highs.max():
                price = _safe_float(curr_high)
                if price is not None:
                    highs.append(price)
                    swing_points.append({
                        "price": price,
                        "type": "high",
                        "date": str(candle_index),
                        "position": i,
                    })

            if curr_low < left_lows.min() and curr_low < right_lows.min():
                price = _safe_float(curr_low)
                if price is not None:
                    lows.append(price)
                    swing_points.append({
                        "price": price,
                        "type": "low",
                        "date": str(candle_index),
                        "position": i,
                    })

        highs = [x for x in highs if x is not None]
        lows = [x for x in lows if x is not None]

        # Keep chronological order for visible swing direction.
        swing_points.sort(key=lambda x: x.get("position", 0))

        result["swing_highs"] = highs[-12:]
        result["swing_lows"] = lows[-12:]
        result["recent_swing_high"] = highs[-1] if highs else None
        result["recent_swing_low"] = lows[-1] if lows else None
        result["swing_points"] = swing_points[-20:]

        return result

    except Exception:
        return result


def group_levels(levels, tolerance_pct=0.003):
    """
    Group nearby support/resistance levels.
    tolerance_pct is dynamic when called from get_nearest_sr().
    """
    try:
        clean = []
        for x in levels:
            val = _safe_float(x)
            if val is not None:
                clean.append(val)

        clean = sorted(clean)

        if not clean:
            return []

        zones = []
        current_group = [clean[0]]

        for price in clean[1:]:
            base = sum(current_group) / len(current_group)

            if base != 0 and abs(price - base) / abs(base) <= tolerance_pct:
                current_group.append(price)
            else:
                zones.append({
                    "price": sum(current_group) / len(current_group),
                    "strength": len(current_group),
                    "min": min(current_group),
                    "max": max(current_group),
                })
                current_group = [price]

        zones.append({
            "price": sum(current_group) / len(current_group),
            "strength": len(current_group),
            "min": min(current_group),
            "max": max(current_group),
        })

        zones.sort(key=lambda z: z["strength"], reverse=True)
        return zones
    except Exception:
        return []


def _filter_far_zones(zones, current_price, atr=None, max_pct=0.03, max_atr_mult=5):
    """
    Filter major S/R zones that are too far from current price.
    Keeps context practical for intraday 15m analysis.

    Rule:
    - keep if within 3% OR within 5 ATR
    """
    try:
        current_price = _safe_float(current_price)
        atr = _safe_float(atr)

        if current_price is None or current_price <= 0:
            return zones or []

        filtered = []
        for z in zones or []:
            price = _safe_float(z.get("price"))
            if price is None:
                continue

            dist = abs(price - current_price)
            pct_ok = dist / current_price <= max_pct
            atr_ok = atr is not None and atr > 0 and dist <= atr * max_atr_mult

            if pct_ok or atr_ok:
                filtered.append(z)

        return filtered
    except Exception:
        return zones or []


def get_nearest_sr(df, lookback=120):
    """
    Find nearest support and resistance around current price.
    Uses dynamic tolerance based on ATR/current price.
    """
    context = {
        "current_price": None,
        "nearest_support": None,
        "nearest_resistance": None,
        "major_support": None,
        "major_resistance": None,
        "recent_swing_high": None,
        "recent_swing_low": None,
        "tolerance_pct": None,
    }

    try:
        if df is None or df.empty:
            return context

        data = df.copy()
        current_price = _safe_float(data["Close"].iloc[-1])
        atr = calculate_atr(data, period=14)
        tolerance_pct = dynamic_sr_tolerance(current_price, atr)

        context["current_price"] = current_price
        context["tolerance_pct"] = tolerance_pct

        swings = detect_swing_levels(data, lookback=lookback, window=3)
        highs = swings.get("swing_highs", [])
        lows = swings.get("swing_lows", [])

        context["recent_swing_high"] = swings.get("recent_swing_high")
        context["recent_swing_low"] = swings.get("recent_swing_low")

        resistance_zones = group_levels(highs, tolerance_pct=tolerance_pct)
        support_zones = group_levels(lows, tolerance_pct=tolerance_pct)

        supports_below = [
            z for z in support_zones
            if current_price is not None and _safe_float(z.get("price")) is not None and z["price"] < current_price
        ]

        resistances_above = [
            z for z in resistance_zones
            if current_price is not None and _safe_float(z.get("price")) is not None and z["price"] > current_price
        ]

        supports_below.sort(key=lambda z: abs(current_price - z["price"]))
        resistances_above.sort(key=lambda z: abs(current_price - z["price"]))

        context["nearest_support"] = supports_below[0] if supports_below else None
        context["nearest_resistance"] = resistances_above[0] if resistances_above else None

        # Far-away major levels can mislead AI, so filter them.

        support_zones = _filter_far_zones(support_zones, current_price, atr=atr)
        resistance_zones = _filter_far_zones(resistance_zones, current_price, atr=atr)

        major_support_candidates = [
            z for z in support_zones
            if current_price is not None and _safe_float(z.get("price")) is not None and z["price"] < current_price
        ]

        major_resistance_candidates = [
            z for z in resistance_zones
            if current_price is not None and _safe_float(z.get("price")) is not None and z["price"] > current_price
        ]

        major_support_candidates.sort(key=lambda z: z["strength"], reverse=True)
        major_resistance_candidates.sort(key=lambda z: z["strength"], reverse=True)

        context["major_support"] = major_support_candidates[0] if major_support_candidates else None
        context["major_resistance"] = major_resistance_candidates[0] if major_resistance_candidates else None

        return context
    except Exception:
        return context

def get_major_structure_context(df, lookback=300, window=5):
    """
    Detect larger structure levels from a wider candle window.
    Useful for finding impulse peaks / major support / major resistance
    that may be missed by short-term 120-candle S/R logic.
    """
    context = {
        "major_swing_high": None,
        "major_swing_low": None,
        "major_resistance": None,
        "major_support": None,
        "major_range_high": None,
        "major_range_low": None,
        "lookback": lookback,
    }

    try:
        if df is None or df.empty:
            return context

        required = {"High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            return context

        data = df.tail(lookback).copy()
        if data.empty:
            return context

        current_price = _safe_float(data["Close"].iloc[-1])
        atr = calculate_atr(data, period=14)
        tolerance_pct = dynamic_sr_tolerance(current_price, atr)

        swings = detect_swing_levels(data, lookback=lookback, window=window)

        highs = swings.get("swing_highs", [])
        lows = swings.get("swing_lows", [])

        clean_highs = [_safe_float(x) for x in highs if _safe_float(x) is not None]
        clean_lows = [_safe_float(x) for x in lows if _safe_float(x) is not None]

        if clean_highs:
            context["major_swing_high"] = max(clean_highs)
            resistance_zones = group_levels(clean_highs, tolerance_pct=tolerance_pct)

            # For major resistance, highest significant zone is more useful than nearest only.
            if resistance_zones:
                resistance_zones.sort(key=lambda z: z["price"], reverse=True)
                context["major_resistance"] = resistance_zones[0]

        if clean_lows:
            context["major_swing_low"] = min(clean_lows)
            support_zones = group_levels(clean_lows, tolerance_pct=tolerance_pct)

            # For major support, lowest significant zone is useful.
            if support_zones:
                support_zones.sort(key=lambda z: z["price"])
                context["major_support"] = support_zones[0]

        context["major_range_high"] = _safe_float(data["High"].max())
        context["major_range_low"] = _safe_float(data["Low"].min())

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
            if move > 0:
                level = end + (abs(move) * fib)
            else:
                level = end - (abs(move) * fib)
            extensions[str(fib)] = level

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

        r50 = retracements.get("0.5")
        if r50 is not None:
            context["pullback_zone"] = {
                "price": r50,
                "label": "0.5"
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
            if move > 0:
                # Bullish continuation above swing high.
                level = end + (abs(move) * fib)
            else:
                # Bearish continuation below swing low.
                level = end - (abs(move) * fib)

            extensions[str(fib)] = level

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
        if r50 is not None:
            context["pullback_zone"] = {
                "price": r50,
                "label": "0.5"
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

        sr = get_nearest_sr(df_micro, lookback=120)
        major_structure = get_major_structure_context(df_micro, lookback=300, window=5)
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

        visible_fib = calculate_fib_context_from_visible_swings(df_micro, lookback=120, window=3)
        volume = calculate_volume_context(df_micro, lookback=30)
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
            lines.append(f"Normal Pullback Zone ({fib_pullback.get('label')}): {_format_price(fib_pullback.get('price'))}")
            
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
        lines.append("--- FIBONACCI CONTEXT FROM RECENT VISIBLE SWING HIGH/LOW ---")
        lines.append(f"Visible Swing Direction: {visible_fib.get('direction', 'Unknown')}")
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
            lines.append(f"Visible Normal Pullback Zone ({visible_fib_pullback.get('label')}): {_format_price(visible_fib_pullback.get('price'))}")
            
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
        
        # --- NEW: CONFLUENCE CHECKER ---
        confluence_found = False
        confluence_notes = []
        
        # Check if 0.618 or 0.786 overlaps with a Major Support/Resistance
        major_sr_levels = []
        if major_support: major_sr_levels.append(major_support.get('min'))
        if major_resistance: major_sr_levels.append(major_resistance.get('min'))
        
        fib_levels_to_check = []
        if visible_fib.get("deep_pullback"): fib_levels_to_check.append(("0.786", visible_fib["deep_pullback"]["price"]))
        if visible_fib.get("golden_pocket"): fib_levels_to_check.append(("0.618", visible_fib["golden_pocket"]["price"]))
        
        tolerance = current_price * 0.005 # 0.5% tolerance for overlap
        
        for name, f_price in fib_levels_to_check:
            for sr_price in major_sr_levels:
                if f_price and sr_price and abs(f_price - sr_price) <= tolerance:
                    confluence_found = True
                    confluence_notes.append(f"Major S/R closely aligns with {name} Fibonacci level at {_format_price(f_price)}.")
        
        lines.append("")
        lines.append("--- AUTOMATED CONFLUENCE CHECK ---")
        if confluence_found:
            lines.append("STATUS: 🔥 HIGH CONFLUENCE DETECTED")
            for note in confluence_notes:
                lines.append(f"- {note}")
        else:
            lines.append("STATUS: No strong S/R + Fibonacci overlap detected.") 

        lines.append("")
        lines.append("IMPORTANT USAGE INSTRUCTION:")
        lines.append("- If the asset is in price discovery (no major S/R above/below), strictly use the 'Visible Continuation Extension Levels' (e.g., 1.272 or 1.618) as your TP2.")
        lines.append("- Elliott Wave structure remains the PRIMARY model. This context can only confirm, weaken, or reject Elliott-based scenarios.")
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