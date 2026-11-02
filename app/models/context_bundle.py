# app/models/context_bundle.py

"""
Context Bundle Builder

Purpose:
- Convert existing text-based market context into backend-readable structured data.
- This allows setup_planner.py to use the same rich information that the old AI-only
  prompt used: Fib, Elliott status, S/R, EMA, ATR, rejection, execution price,
  dominant trend, and wave metadata.

This module does NOT create signals.
This module does NOT call AI.
This module is a data adapter.

Expected usage from main.py:

    context_bundle = build_context_bundle(
        symbol=symbol,
        timeframe=interval,
        market_context_text=market_context_text,
        recent_candle_context_text=recent_candle_context_text,
        execution_price_context_text=execution_price_context_text,
        dominant_trend_context_text=dominant_trend_context_text,
        macro_waves=macro_waves,
        swing_waves=swing_waves,
        confirm_1h_waves=confirm_1h_waves,
        micro_waves=micro_waves,
    )

Then pass this into setup_planner:

    backend_setup_plan = plan_backend_setup(
        df_micro=df_micro_analysis,
        df_confirm=df_confirm_1h_analysis,
        df_swing=df_swing_analysis,
        df_live=df_micro_live,
        backend_trend_regime=backend_trend_regime,
        context_bundle=context_bundle,
        symbol=symbol,
        user_tf=interval
    )
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


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

        text = str(value).strip()
        if not text:
            return default

        # Remove commas in prices like 80,216.16
        text = text.replace(",", "")

        # Remove common wrappers
        text = text.replace("$", "").strip()

        v = float(text)
        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except Exception:
        return default


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _combine_texts(*texts: Optional[str]) -> str:
    parts = []
    for t in texts:
        t = _clean_text(t)
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _line_value(text: str, label: str) -> Optional[str]:
    """
    Find exact-ish line:
        Label: value
    Returns value string.
    """
    try:
        pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
        m = re.search(pattern, text)
        if not m:
            return None
        return m.group(1).strip()
    except Exception:
        return None


def _line_float(text: str, label: str, default: Optional[float] = None) -> Optional[float]:
    val = _line_value(text, label)
    if val is None:
        return default

    # If line contains extra text, grab first number.
    return _first_float(val, default=default)


def _first_float(text: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if text is None:
            return default

        s = str(text).replace(",", "")
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        if not m:
            return default

        return _safe_float(m.group(0), default=default)
    except Exception:
        return default


def _all_floats(text: Any) -> List[float]:
    try:
        if text is None:
            return []
        s = str(text).replace(",", "")
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        out = []
        for n in nums:
            v = _safe_float(n)
            if v is not None:
                out.append(v)
        return out
    except Exception:
        return []


def _parse_zone_value(value: Optional[str]) -> Dict[str, Any]:
    """
    Parse:
      "77752.00 - 78310.54 (strength 8)"
      "80635.51 (strength 1)"
      "N/A"
    """
    result = {
        "raw": value or "",
        "low": None,
        "high": None,
        "mid": None,
        "strength": None,
        "valid": False,
    }

    try:
        if not value:
            return result

        raw = str(value).strip()
        result["raw"] = raw

        nums = _all_floats(raw)
        if not nums:
            return result

        # Remove strength number if present separately by regex.
        strength = None
        sm = re.search(r"\(strength\s+(\d+)\)", raw, re.IGNORECASE)
        if sm:
            strength = int(sm.group(1))
            result["strength"] = strength
            # Use all nums except final strength number if it is included.
            # Example nums: [77752.00, 78310.54, 8]
            if len(nums) >= 2 and abs(nums[-1] - strength) < 1e-9:
                nums = nums[:-1]

        if len(nums) == 1:
            low = high = nums[0]
        else:
            low = min(nums[0], nums[1])
            high = max(nums[0], nums[1])

        result["low"] = low
        result["high"] = high
        result["mid"] = (low + high) / 2.0
        result["valid"] = True

        return result

    except Exception:
        return result


def _parse_bool_from_yes_no(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = str(value).strip().upper()
    if "YES" in v:
        return True
    if "NO" in v:
        return False
    return None


def _section_between(text: str, header: str, next_headers: Optional[List[str]] = None) -> str:
    """
    Extract section starting after a header until next known header.
    Header example: "--- EMA CONTEXT ---"
    """
    try:
        if not text or not header:
            return ""

        start = text.find(header)
        if start < 0:
            return ""

        section_start = start + len(header)
        rest = text[section_start:]

        if not next_headers:
            # Stop at next line beginning with ---
            m = re.search(r"\n---\s+.+?\s+---", rest)
            if m:
                return rest[:m.start()].strip()
            return rest.strip()

        stop_positions = []
        for h in next_headers:
            idx = rest.find(h)
            if idx >= 0:
                stop_positions.append(idx)

        if stop_positions:
            stop = min(stop_positions)
            return rest[:stop].strip()

        return rest.strip()

    except Exception:
        return ""


def _distance_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None or b == 0:
            return None
        return abs(a - b) / abs(b) * 100.0
    except Exception:
        return None


def _price_in_zone(price: Optional[float], zone: Dict[str, Any], buffer: float = 0.0) -> bool:
    try:
        if price is None or not zone or not zone.get("valid"):
            return False
        low = zone.get("low")
        high = zone.get("high")
        if low is None or high is None:
            return False
        return (low - buffer) <= price <= (high + buffer)
    except Exception:
        return False


def _distance_to_zone(price: Optional[float], zone: Dict[str, Any]) -> Optional[float]:
    """
    Absolute distance from price to zone.
    0 if inside zone.
    """
    try:
        if price is None or not zone or not zone.get("valid"):
            return None

        low = zone.get("low")
        high = zone.get("high")

        if low is None or high is None:
            return None

        if low <= price <= high:
            return 0.0

        if price < low:
            return low - price

        return price - high

    except Exception:
        return None


# ==========================================================
# Market structure parser
# ==========================================================

def _parse_market_structure(text: str) -> Dict[str, Any]:
    ms = {
        "current_price": None,
        "atr": None,
        "atr_pct": None,
        "dynamic_sr_tolerance_pct": None,
        "recent_swing_high": None,
        "recent_swing_low": None,
        "nearest_support": None,
        "nearest_resistance": None,
        "major_support_zone": {},
        "major_resistance_zone": {},
        "long_lookback_high": None,
        "long_lookback_low": None,
        "major_swing_high": None,
        "major_swing_low": None,
        "long_lookback_major_support_zone": {},
        "long_lookback_major_resistance_zone": {},
    }

    try:
        ms["current_price"] = _line_float(text, "Current Price")

        atr_line = _line_value(text, "ATR(14)")
        if atr_line:
            nums = _all_floats(atr_line)
            if len(nums) >= 1:
                ms["atr"] = nums[0]
            if len(nums) >= 2:
                ms["atr_pct"] = nums[1]

        tol_line = _line_value(text, "Dynamic S/R Grouping Tolerance")
        if tol_line:
            ms["dynamic_sr_tolerance_pct"] = _first_float(tol_line)

        ms["recent_swing_high"] = _line_float(text, "Recent Swing High")
        ms["recent_swing_low"] = _line_float(text, "Recent Swing Low")

        ns = _line_value(text, "Nearest Support")
        nr = _line_value(text, "Nearest Resistance")
        ms["nearest_support"] = _first_float(ns)
        ms["nearest_resistance"] = _first_float(nr)

        ms["major_support_zone"] = _parse_zone_value(_line_value(text, "Major Support Zone"))
        ms["major_resistance_zone"] = _parse_zone_value(_line_value(text, "Major Resistance Zone"))

        ms["long_lookback_high"] = _line_float(text, "Long Lookback Range High")
        ms["long_lookback_low"] = _line_float(text, "Long Lookback Range Low")
        ms["major_swing_high"] = _line_float(text, "Major Swing High / Possible Impulse Peak")
        ms["major_swing_low"] = _line_float(text, "Major Swing Low / Possible Correction Base")

        ms["long_lookback_major_support_zone"] = _parse_zone_value(
            _line_value(text, "Long Lookback Major Support Zone")
        )
        ms["long_lookback_major_resistance_zone"] = _parse_zone_value(
            _line_value(text, "Long Lookback Major Resistance Zone")
        )

        return ms

    except Exception:
        return ms


# ==========================================================
# Fibonacci parser
# ==========================================================

def _parse_range_line(value: Optional[str]) -> Dict[str, Any]:
    result = {
        "raw": value or "",
        "low": None,
        "high": None,
        "mid": None,
        "valid": False,
    }

    try:
        if not value:
            return result

        nums = _all_floats(value)
        if not nums:
            return result

        if len(nums) == 1:
            low = high = nums[0]
        else:
            low = min(nums[0], nums[1])
            high = max(nums[0], nums[1])

        result.update({
            "low": low,
            "high": high,
            "mid": (low + high) / 2.0,
            "valid": True,
        })

        return result

    except Exception:
        return result


def _parse_extension_line(value: Optional[str]) -> Dict[str, float]:
    """
    Parse:
    "1.0: 75768.49 | 1.272: 75106.58 | 1.618: 74264.58 | 2.0: 73334.98"
    """
    out: Dict[str, float] = {}

    try:
        if not value:
            return out

        for part in str(value).split("|"):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            key = k.strip()
            val = _first_float(v)
            if val is not None:
                out[key] = val

        return out

    except Exception:
        return out


def _parse_fib_block(text: str, prefix: str = "") -> Dict[str, Any]:
    """
    prefix "" for Elliott latest:
      Last Swing Direction
      Shallow Pullback Zone ...
      Continuation Extension Levels From Last Swing End

    prefix "Visible " for visible swing:
      Visible Swing Direction
      Visible Shallow Pullback Zone ...
      Visible Continuation Extension Levels
    """
    fib = {
        "status": None,
        "warning": None,
        "direction": None,
        "start": None,
        "end": None,
        "shallow_zone": {},
        "normal_zone": {},
        "golden_pocket": None,
        "deep_pullback": None,
        "extensions": {},
        "usable_for_entry": True,
        "raw": "",
    }

    try:
        fib["raw"] = text or ""

        if prefix == "":
            fib["status"] = _line_value(text, "Elliott Fib Context Status")
            fib["warning"] = _line_value(text, "Warning")
            fib["direction"] = _line_value(text, "Last Swing Direction")
            fib["start"] = _line_float(text, "Last Swing Start")
            fib["end"] = _line_float(text, "Last Swing End")
            fib["shallow_zone"] = _parse_range_line(_line_value(text, "Shallow Pullback Zone (0.236 - 0.382)"))
            fib["normal_zone"] = _parse_range_line(_line_value(text, "Normal Pullback Zone (0.5 - 0.618)"))
            fib["golden_pocket"] = _line_float(text, "Golden Pocket (0.618 Golden Pocket)")
            fib["deep_pullback"] = _line_float(text, "Deep Pullback (0.786)")
            fib["extensions"] = _parse_extension_line(
                _line_value(text, "Continuation Extension Levels From Last Swing End")
            )

            status = str(fib.get("status") or "").upper()
            warning = str(fib.get("warning") or "").upper()
            if "STALE" in status or "WEAK" in status or "DO NOT USE" in warning:
                fib["usable_for_entry"] = False

        else:
            fib["status"] = "VISIBLE"
            fib["direction"] = _line_value(text, "Visible Swing Direction")
            fib["start"] = _line_float(text, "Visible Swing Start")
            fib["end"] = _line_float(text, "Visible Swing End")
            fib["shallow_zone"] = _parse_range_line(_line_value(text, "Visible Shallow Pullback Zone (0.236 - 0.382)"))
            fib["normal_zone"] = _parse_range_line(_line_value(text, "Visible Normal Pullback Zone (0.5 - 0.618)"))
            fib["golden_pocket"] = _line_float(text, "Visible Golden Pocket (0.618 Golden Pocket)")
            fib["deep_pullback"] = _line_float(text, "Visible Deep Pullback (0.786)")
            fib["extensions"] = _parse_extension_line(
                _line_value(text, "Visible Continuation Extension Levels")
            )
            fib["usable_for_entry"] = True

        return fib

    except Exception:
        return fib


def _parse_fibonacci_context(text: str) -> Dict[str, Any]:
    out = {
        "elliott": {},
        "visible": {},
        "best_entry_fib_source": "visible",
        "notes": [],
    }

    try:
        elliott_section = _section_between(
            text,
            "--- FIBONACCI CONTEXT FROM LATEST MICRO SWING ---"
        )
        visible_section = _section_between(
            text,
            "--- FIBONACCI CONTEXT FROM RECENT VISIBLE SWING HIGH/LOW ---"
        )

        out["elliott"] = _parse_fib_block(elliott_section, prefix="")
        out["visible"] = _parse_fib_block(visible_section, prefix="Visible ")

        if out["elliott"].get("usable_for_entry"):
            out["best_entry_fib_source"] = "elliott"
        else:
            out["best_entry_fib_source"] = "visible"
            out["notes"].append("Elliott fib is weak/stale/unusable; prefer visible swing fib.")

        return out

    except Exception:
        return out


# ==========================================================
# EMA / Entry depth / Confluence parser
# ==========================================================

def _parse_ema_context(text: str) -> Dict[str, Any]:
    ema = {
        "ema50": None,
        "ema200": None,
        "price_vs_ema50": None,
        "price_vs_ema200": None,
        "ema50_vs_ema200": None,
        "ema_trend_bias": None,
    }

    try:
        ema["ema50"] = _line_float(text, "EMA50")
        ema["ema200"] = _line_float(text, "EMA200")
        ema["price_vs_ema50"] = _line_value(text, "Price vs EMA50")
        ema["price_vs_ema200"] = _line_value(text, "Price vs EMA200")
        ema["ema50_vs_ema200"] = _line_value(text, "EMA50 vs EMA200")
        ema["ema_trend_bias"] = _line_value(text, "EMA Trend Bias")

        return ema

    except Exception:
        return ema


def _parse_entry_depth_context(text: str) -> Dict[str, Any]:
    ed = {
        "recent_impulse_size": None,
        "recent_impulse_atr_multiple": None,
        "shallow_entry_risk": None,
        "preferred_short_entry_depth": None,
        "preferred_long_entry_depth": None,
    }

    try:
        ed["recent_impulse_size"] = _line_float(text, "Recent Impulse Size")

        atr_mult = _line_value(text, "Recent Impulse ATR Multiple")
        if atr_mult:
            ed["recent_impulse_atr_multiple"] = _first_float(atr_mult)

        ed["shallow_entry_risk"] = _line_value(text, "Shallow Entry Risk")
        ed["preferred_short_entry_depth"] = _line_value(text, "Preferred SHORT Entry Depth")
        ed["preferred_long_entry_depth"] = _line_value(text, "Preferred LONG Entry Depth")

        return ed

    except Exception:
        return ed


def _parse_automated_confluence(text: str) -> Dict[str, Any]:
    conf = {
        "status": None,
        "high_confluence": False,
        "items": [],
        "raw": "",
    }

    try:
        section = _section_between(text, "--- AUTOMATED CONFLUENCE CHECK ---")
        conf["raw"] = section

        status = _line_value(section, "STATUS")
        conf["status"] = status

        if status and "HIGH CONFLUENCE" in status.upper():
            conf["high_confluence"] = True

        items = []
        for line in section.splitlines():
            line = line.strip()
            if line.startswith("-"):
                items.append(line.lstrip("-").strip())

        conf["items"] = items

        return conf

    except Exception:
        return conf


# ==========================================================
# Candle / rejection parser
# ==========================================================

def _parse_recent_candle_context(text: str) -> Dict[str, Any]:
    c = {
        "last_closed_candle_time": None,
        "last_closed_ohlc": {},
        "latest_available_candle_time": None,
        "latest_available_price": None,
        "candle_direction": None,
        "close_position": None,
        "upper_wick": None,
        "lower_wick": None,
        "candle_body": None,
        "candle_range": None,
        "support_zone_used": {},
        "resistance_zone_used": {},
        "support_zone_state": None,
        "resistance_zone_state": None,
        "support_rejection_confirmed": None,
        "resistance_rejection_confirmed": None,
        "zone_check_margin": None,
        "rejection_notes": None,
        "both_rejections": False,
        "no_rejection": False,
    }

    try:
        c["last_closed_candle_time"] = _line_value(text, "Last Closed Candle Time")
        ohlc_line = _line_value(text, "Last Closed Candle O/H/L/C")
        nums = _all_floats(ohlc_line)
        if len(nums) >= 4:
            c["last_closed_ohlc"] = {
                "open": nums[0],
                "high": nums[1],
                "low": nums[2],
                "close": nums[3],
            }

        c["latest_available_candle_time"] = _line_value(text, "Latest Available Candle Time")
        c["latest_available_price"] = _line_float(text, "Latest Available Price")

        c["candle_direction"] = _line_value(text, "Candle Direction")
        c["close_position"] = _line_value(text, "Close Position")
        c["upper_wick"] = _line_float(text, "Upper Wick")
        c["lower_wick"] = _line_float(text, "Lower Wick")
        c["candle_body"] = _line_float(text, "Candle Body")
        c["candle_range"] = _line_float(text, "Candle Range")

        c["support_zone_used"] = _parse_zone_value(_line_value(text, "Support Zone Used"))
        c["resistance_zone_used"] = _parse_zone_value(_line_value(text, "Resistance Zone Used"))

        c["support_zone_state"] = _line_value(text, "Support Zone State")
        c["resistance_zone_state"] = _line_value(text, "Resistance Zone State")

        c["support_rejection_confirmed"] = _parse_bool_from_yes_no(
            _line_value(text, "Support Rejection Confirmed")
        )
        c["resistance_rejection_confirmed"] = _parse_bool_from_yes_no(
            _line_value(text, "Resistance Rejection Confirmed")
        )

        c["zone_check_margin"] = _line_float(text, "Zone Check Margin")
        c["rejection_notes"] = _line_value(text, "Rejection Notes")

        c["both_rejections"] = bool(c["support_rejection_confirmed"] and c["resistance_rejection_confirmed"])
        c["no_rejection"] = bool(
            c["support_rejection_confirmed"] is False
            and c["resistance_rejection_confirmed"] is False
        )

        return c

    except Exception:
        return c


# ==========================================================
# Execution context parser
# ==========================================================

def _parse_execution_context(text: str) -> Dict[str, Any]:
    ex = {
        "live_price": None,
        "live_candle_time": None,
        "analysis_closed_price": None,
        "analysis_closed_time": None,
        "execution_note": None,
    }

    try:
        ex["live_price"] = _line_float(text, "Live Price For Execution")
        ex["live_candle_time"] = _line_value(text, "Live Candle Time")
        ex["analysis_closed_price"] = _line_float(text, "Analysis Closed Candle Price")
        ex["analysis_closed_time"] = _line_value(text, "Analysis Closed Candle Time")
        ex["execution_note"] = _line_value(text, "Execution Note")

        return ex

    except Exception:
        return ex


# ==========================================================
# Dominant trend parser
# ==========================================================

def _parse_dominant_trend_context(text: str) -> Dict[str, Any]:
    dt = {
        "dominant_direction": None,
        "confirm_tf_structure": None,
        "swing_tf_structure": None,
        "preferred_trade_side": None,
        "ema200_filter_confirm_tf": None,
        "confirm_tf_ema200_value": None,
        "current_price": None,
        "choch": None,
        "dominant_confidence_score": None,
        "confirm_tf_elliott_bias": None,
        "swing_tf_elliott_bias": None,
    }

    try:
        dt["dominant_direction"] = _line_value(text, "Dominant Direction")
        dt["confirm_tf_structure"] = _line_value(text, "Confirm TF Structure")
        dt["swing_tf_structure"] = _line_value(text, "Swing TF Structure")
        dt["preferred_trade_side"] = _line_value(text, "Preferred Trade Side")
        dt["ema200_filter_confirm_tf"] = _line_value(text, "EMA 200 Filter (Confirm TF)")
        dt["confirm_tf_ema200_value"] = _line_float(text, "Confirm TF EMA200 Value")
        dt["current_price"] = _line_float(text, "Current Price")
        dt["choch"] = _parse_bool_from_yes_no(_line_value(text, "Structure Broken (CHoCH)"))

        conf_line = _line_value(text, "Dominant Confidence Score")
        if conf_line:
            dt["dominant_confidence_score"] = _first_float(conf_line)

        dt["confirm_tf_elliott_bias"] = _line_value(text, "Confirm TF Elliott Bias")
        dt["swing_tf_elliott_bias"] = _line_value(text, "Swing TF Elliott Bias")

        return dt

    except Exception:
        return dt


# ==========================================================
# Wave metadata normalizer
# ==========================================================

def _safe_wave_summary(waves: Any) -> Dict[str, Any]:
    """
    Keep this defensive because WaveAnalyzer output may be dict/list/custom.
    We do not rely on exact schema yet.
    """
    out = {
        "available": False,
        "type": type(waves).__name__ if waves is not None else None,
        "raw": None,
        "bias": None,
        "wave_count_status": None,
        "phase": None,
        "confidence": None,
    }

    try:
        if waves is None:
            return out

        out["available"] = True

        if isinstance(waves, dict):
            out["raw"] = waves

            for key in ["bias", "trend", "direction", "elliott_bias"]:
                if key in waves:
                    out["bias"] = waves.get(key)
                    break

            for key in ["wave_count_status", "status", "count_status"]:
                if key in waves:
                    out["wave_count_status"] = waves.get(key)
                    break

            for key in ["phase", "wave_phase", "market_phase"]:
                if key in waves:
                    out["phase"] = waves.get(key)
                    break

            for key in ["confidence", "score", "probability"]:
                if key in waves:
                    out["confidence"] = waves.get(key)
                    break

        else:
            # Avoid huge custom objects. Store string preview only.
            s = str(waves)
            out["raw"] = s[:1000]

        return out

    except Exception:
        return out


def _build_wave_context(
    macro_waves: Any = None,
    swing_waves: Any = None,
    confirm_1h_waves: Any = None,
    micro_waves: Any = None,
) -> Dict[str, Any]:
    return {
        "macro": _safe_wave_summary(macro_waves),
        "swing": _safe_wave_summary(swing_waves),
        "confirm_1h": _safe_wave_summary(confirm_1h_waves),
        "micro": _safe_wave_summary(micro_waves),
    }


# ==========================================================
# Derived context
# ==========================================================

def _derive_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create practical flags setup_planner can use.
    """
    d = {
        "current_price": None,
        "live_price": None,
        "execution_price": None,
        "atr": None,
        "is_choppy": False,
        "has_support_rejection": False,
        "has_resistance_rejection": False,
        "both_support_and_resistance_rejection": False,
        "dominant_wait": False,
        "choch": False,
        "preferred_side_from_dominant": None,
        "fib_entry_source": "visible",
        "elliott_fib_usable": False,
        "visible_fib_available": False,
        "nearest_support_distance_pct": None,
        "nearest_resistance_distance_pct": None,
        "near_major_support": False,
        "near_major_resistance": False,
        "notes": [],
    }

    try:
        ms = bundle.get("market_structure", {})
        ex = bundle.get("execution", {})
        candle = bundle.get("rejection", {})
        dom = bundle.get("dominant_trend", {})
        fib = bundle.get("fibonacci", {})

        current_price = ms.get("current_price")
        live_price = ex.get("live_price")
        execution_price = live_price if live_price is not None else current_price

        atr = ms.get("atr") or 0.0

        d["current_price"] = current_price
        d["live_price"] = live_price
        d["execution_price"] = execution_price
        d["atr"] = atr

        d["has_support_rejection"] = bool(candle.get("support_rejection_confirmed"))
        d["has_resistance_rejection"] = bool(candle.get("resistance_rejection_confirmed"))
        d["both_support_and_resistance_rejection"] = bool(candle.get("both_rejections"))

        if d["both_support_and_resistance_rejection"]:
            d["is_choppy"] = True
            d["notes"].append("Both support and resistance rejection detected; market may be choppy.")

        d["choch"] = bool(dom.get("choch"))
        if d["choch"]:
            d["notes"].append("Dominant trend context says CHoCH is active.")

        preferred = dom.get("preferred_trade_side")
        d["preferred_side_from_dominant"] = preferred
        if preferred and str(preferred).upper() == "WAIT":
            d["dominant_wait"] = True
            d["notes"].append("Dominant trend context preferred side is WAIT.")

        elliott = fib.get("elliott", {})
        visible = fib.get("visible", {})

        d["elliott_fib_usable"] = bool(elliott.get("usable_for_entry"))
        d["visible_fib_available"] = bool(visible.get("direction"))

        d["fib_entry_source"] = fib.get("best_entry_fib_source", "visible")

        ns = ms.get("nearest_support")
        nr = ms.get("nearest_resistance")

        d["nearest_support_distance_pct"] = _distance_pct(execution_price, ns)
        d["nearest_resistance_distance_pct"] = _distance_pct(execution_price, nr)

        major_support = ms.get("major_support_zone", {})
        major_resistance = ms.get("major_resistance_zone", {})

        buffer = atr * 0.25 if atr else 0.0
        d["near_major_support"] = _price_in_zone(execution_price, major_support, buffer=buffer)
        d["near_major_resistance"] = _price_in_zone(execution_price, major_resistance, buffer=buffer)

        return d

    except Exception as e:
        d["notes"].append(f"Derived context error: {str(e)}")
        return d


# ==========================================================
# Public API
# ==========================================================

def build_context_bundle(
    symbol: str = "",
    timeframe: str = "",
    market_context_text: Optional[str] = None,
    recent_candle_context_text: Optional[str] = None,
    execution_price_context_text: Optional[str] = None,
    dominant_trend_context_text: Optional[str] = None,
    backend_trend_regime: Optional[Dict[str, Any]] = None,
    macro_waves: Any = None,
    swing_waves: Any = None,
    confirm_1h_waves: Any = None,
    micro_waves: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main builder.

    Accepts the text blocks already produced in main.py and converts them
    into structured context for backend setup planning.
    """
    combined = _combine_texts(
        market_context_text,
        recent_candle_context_text,
        execution_price_context_text,
        dominant_trend_context_text,
    )

    bundle: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "version": "context_bundle_v1",
        "raw_text_available": bool(combined),
        "market_structure": {},
        "fibonacci": {},
        "ema": {},
        "entry_depth": {},
        "automated_confluence": {},
        "rejection": {},
        "execution": {},
        "dominant_trend": {},
        "backend_trend_regime": backend_trend_regime or {},
        "waves": {},
        "derived": {},
        "extra": extra or {},
        "parser_warnings": [],
    }

    try:
        bundle["market_structure"] = _parse_market_structure(combined)
        bundle["fibonacci"] = _parse_fibonacci_context(combined)
        bundle["ema"] = _parse_ema_context(combined)
        bundle["entry_depth"] = _parse_entry_depth_context(combined)
        bundle["automated_confluence"] = _parse_automated_confluence(combined)
        bundle["rejection"] = _parse_recent_candle_context(combined)
        bundle["execution"] = _parse_execution_context(combined)
        bundle["dominant_trend"] = _parse_dominant_trend_context(combined)

        bundle["waves"] = _build_wave_context(
            macro_waves=macro_waves,
            swing_waves=swing_waves,
            confirm_1h_waves=confirm_1h_waves,
            micro_waves=micro_waves,
        )

        bundle["derived"] = _derive_context(bundle)

        # Basic parser warnings
        if bundle["market_structure"].get("current_price") is None:
            bundle["parser_warnings"].append("Current price not parsed from market context.")

        if bundle["market_structure"].get("atr") is None:
            bundle["parser_warnings"].append("ATR not parsed from market context.")

        if not bundle["fibonacci"].get("visible", {}).get("direction"):
            bundle["parser_warnings"].append("Visible fibonacci context not parsed.")

        return bundle

    except Exception as e:
        bundle["parser_warnings"].append(f"Context bundle build error: {str(e)}")
        bundle["derived"] = _derive_context(bundle)
        return bundle


def summarize_context_bundle(bundle: Dict[str, Any]) -> str:
    """
    Human-readable debug block for market_context_text.
    Add this to main.py output if you want to see what backend parsed.
    """
    try:
        ms = bundle.get("market_structure", {})
        fib = bundle.get("fibonacci", {})
        ema = bundle.get("ema", {})
        rej = bundle.get("rejection", {})
        ex = bundle.get("execution", {})
        dom = bundle.get("dominant_trend", {})
        derived = bundle.get("derived", {})

        lines = [
            "--- STRUCTURED CONTEXT BUNDLE ---",
            f"Symbol: {bundle.get('symbol', '')}",
            f"Timeframe: {bundle.get('timeframe', '')}",
            f"Current Price: {ms.get('current_price')}",
            f"Live/Execution Price: {ex.get('live_price')}",
            f"ATR: {ms.get('atr')}",
            f"Nearest Support: {ms.get('nearest_support')}",
            f"Nearest Resistance: {ms.get('nearest_resistance')}",
            f"Major Support Zone: {ms.get('major_support_zone', {}).get('raw')}",
            f"Major Resistance Zone: {ms.get('major_resistance_zone', {}).get('raw')}",
            f"EMA50: {ema.get('ema50')}",
            f"EMA200: {ema.get('ema200')}",
            f"EMA Trend Bias: {ema.get('ema_trend_bias')}",
            f"Elliott Fib Status: {fib.get('elliott', {}).get('status')}",
            f"Elliott Fib Usable: {derived.get('elliott_fib_usable')}",
            f"Visible Swing Direction: {fib.get('visible', {}).get('direction')}",
            f"Best Fib Entry Source: {derived.get('fib_entry_source')}",
            f"Support Rejection: {rej.get('support_rejection_confirmed')}",
            f"Resistance Rejection: {rej.get('resistance_rejection_confirmed')}",
            f"Both Rejections / Chop: {derived.get('both_support_and_resistance_rejection')}",
            f"Dominant Preferred Side: {dom.get('preferred_trade_side')}",
            f"CHoCH: {dom.get('choch')}",
            f"Derived Notes: {derived.get('notes', [])}",
            f"Parser Warnings: {bundle.get('parser_warnings', [])}",
        ]

        return "\n".join(lines)

    except Exception as e:
        return (
            "--- STRUCTURED CONTEXT BUNDLE ---\n"
            f"Context bundle summary error: {str(e)}"
        )


# ==========================================================
# Utility functions for setup_planner.py
# ==========================================================

def get_context_execution_price(context_bundle: Optional[Dict[str, Any]]) -> Optional[float]:
    try:
        if not context_bundle:
            return None

        derived = context_bundle.get("derived", {})
        v = derived.get("execution_price")
        if v is not None:
            return _safe_float(v)

        ex = context_bundle.get("execution", {})
        if ex.get("live_price") is not None:
            return _safe_float(ex.get("live_price"))

        ms = context_bundle.get("market_structure", {})
        return _safe_float(ms.get("current_price"))

    except Exception:
        return None


def get_context_atr(context_bundle: Optional[Dict[str, Any]]) -> Optional[float]:
    try:
        if not context_bundle:
            return None

        derived = context_bundle.get("derived", {})
        if derived.get("atr") is not None:
            return _safe_float(derived.get("atr"))

        return _safe_float(context_bundle.get("market_structure", {}).get("atr"))

    except Exception:
        return None


def get_context_major_zones(context_bundle: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    try:
        if not context_bundle:
            return {"support": {}, "resistance": {}}

        ms = context_bundle.get("market_structure", {})
        return {
            "support": ms.get("major_support_zone", {}) or {},
            "resistance": ms.get("major_resistance_zone", {}) or {},
            "long_support": ms.get("long_lookback_major_support_zone", {}) or {},
            "long_resistance": ms.get("long_lookback_major_resistance_zone", {}) or {},
        }

    except Exception:
        return {"support": {}, "resistance": {}}


def get_context_fib_zones(context_bundle: Optional[Dict[str, Any]], source: str = "best") -> Dict[str, Any]:
    try:
        if not context_bundle:
            return {}

        fib = context_bundle.get("fibonacci", {})
        derived = context_bundle.get("derived", {})

        if source == "best":
            source = derived.get("fib_entry_source", "visible")

        if source == "elliott":
            return fib.get("elliott", {}) or {}

        if source == "visible":
            return fib.get("visible", {}) or {}

        return {}

    except Exception:
        return {}


def is_context_choppy(context_bundle: Optional[Dict[str, Any]]) -> bool:
    try:
        if not context_bundle:
            return False

        derived = context_bundle.get("derived", {})
        return bool(derived.get("is_choppy"))

    except Exception:
        return False


def context_prefers_wait(context_bundle: Optional[Dict[str, Any]]) -> bool:
    try:
        if not context_bundle:
            return False

        derived = context_bundle.get("derived", {})
        return bool(derived.get("dominant_wait") or derived.get("choch"))

    except Exception:
        return False