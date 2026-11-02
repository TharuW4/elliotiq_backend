# app/models/abc_c_ending_setup.py

"""
ABC C-ending Setup Builder

Purpose:
- Convert Elliott phase context "ABC_C_ENDING" into a conservative setup idea.
- This file does NOT create final TP1/TP2/RR.
- setup_planner.py must convert this idea into a normal backend candidate using _make_candidate().
- Normal planner validation still decides whether this becomes a real signal.

Design:
- Bullish ABC C-ending -> LONG near support / Fib / EMA / rejection zone.
- Bearish ABC C-ending -> SHORT near resistance / Fib / EMA / rejection zone.
- No signal if phase is CONFLICT.
- No signal if setup is in the middle of nowhere.
"""

from typing import Any, Dict, List, Optional, Tuple
import math


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(str(value).replace(",", "").replace("$", "").strip())
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _tf_key(user_tf: str) -> str:
    return str(user_tf or "").lower().strip()


def _max_entry_distance_pct(user_tf: str) -> float:
    """
    Avoid ABC entries too far from current price.
    ABC C-ending should be near structure, not a deep random limit far away.
    """
    tf = _tf_key(user_tf)

    if tf in ["1m", "3m", "5m"]:
        return 1.0
    if tf in ["15m", "30m"]:
        return 2.0
    if tf in ["1h", "60m"]:
        return 3.5
    if tf in ["4h", "240m"]:
        return 5.5
    if tf in ["1d", "d", "daily"]:
        return 8.0

    return 3.5


def _zone_valid(zone: Any) -> bool:
    return (
        isinstance(zone, dict)
        and bool(zone.get("valid"))
        and zone.get("low") is not None
        and zone.get("high") is not None
    )


def _zone_mid(zone: Dict[str, Any]) -> Optional[float]:
    try:
        mid = _safe_float(zone.get("mid"))
        if mid is not None:
            return mid

        low = _safe_float(zone.get("low"))
        high = _safe_float(zone.get("high"))

        if low is None or high is None:
            return None

        return (low + high) / 2.0
    except Exception:
        return None


def _context_dict(context_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return context_bundle if isinstance(context_bundle, dict) else {}


def _add_unique_anchor(
    anchors: List[Tuple[str, float]],
    label: str,
    value: Any,
    side: str,
    current_price: float,
) -> None:
    fv = _safe_float(value)
    if fv is None or fv <= 0:
        return

    side = str(side or "").upper()

    if side == "LONG" and fv <= current_price:
        anchors.append((label, float(fv)))

    elif side == "SHORT" and fv >= current_price:
        anchors.append((label, float(fv)))


def _collect_abc_entry_anchors(
    side: str,
    current_price: float,
    pivots: Dict[str, Any],
    ema50: Optional[float],
    ema200: Optional[float],
    context_bundle: Optional[Dict[str, Any]],
) -> List[Tuple[str, float]]:
    """
    Find possible ABC C-ending entry anchors.

    LONG:
      support / major support / recent lows / EMA support below current price

    SHORT:
      resistance / major resistance / recent highs / EMA resistance above current price
    """
    anchors: List[Tuple[str, float]] = []

    side = str(side or "").upper()
    cb = _context_dict(context_bundle)
    ms = cb.get("market_structure", {}) or {}

    if side == "LONG":
        for label, value in [
            ("nearest_support", ms.get("nearest_support")),
            ("recent_low", pivots.get("recent_low")),
            ("previous_low", pivots.get("previous_low")),
            ("lowest", pivots.get("lowest")),
            ("ema50", ema50),
            ("ema200", ema200),
        ]:
            _add_unique_anchor(anchors, label, value, side, current_price)

        for zone_key in ["major_support_zone", "long_lookback_major_support_zone"]:
            z = ms.get(zone_key, {}) or {}
            if _zone_valid(z):
                # For LONG, upper/mid of support zone can react first.
                for label, value in [
                    (zone_key + "_high", z.get("high")),
                    (zone_key + "_mid", _zone_mid(z)),
                    (zone_key + "_low", z.get("low")),
                ]:
                    _add_unique_anchor(anchors, label, value, side, current_price)

    elif side == "SHORT":
        for label, value in [
            ("nearest_resistance", ms.get("nearest_resistance")),
            ("recent_high", pivots.get("recent_high")),
            ("previous_high", pivots.get("previous_high")),
            ("highest", pivots.get("highest")),
            ("ema50", ema50),
            ("ema200", ema200),
        ]:
            _add_unique_anchor(anchors, label, value, side, current_price)

        for zone_key in ["major_resistance_zone", "long_lookback_major_resistance_zone"]:
            z = ms.get(zone_key, {}) or {}
            if _zone_valid(z):
                # For SHORT, lower/mid of resistance zone can react first.
                for label, value in [
                    (zone_key + "_low", z.get("low")),
                    (zone_key + "_mid", _zone_mid(z)),
                    (zone_key + "_high", z.get("high")),
                ]:
                    _add_unique_anchor(anchors, label, value, side, current_price)

    # remove duplicates
    out: List[Tuple[str, float]] = []
    seen = set()

    for label, value in anchors:
        key = round(float(value), 8)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, float(value)))

    # closest to current price first
    if side == "LONG":
        out.sort(key=lambda x: current_price - x[1])
    elif side == "SHORT":
        out.sort(key=lambda x: x[1] - current_price)

    return out


def _build_abc_confluences(
    side: str,
    entry: float,
    entry_label: str,
    pivots: Dict[str, Any],
    ema50: Optional[float],
    ema200: Optional[float],
    atr_val: Optional[float],
    context_bundle: Optional[Dict[str, Any]],
    phase_alignment: str,
    phase_score: float,
) -> List[str]:
    confluences: List[str] = []

    side = str(side or "").upper()
    cb = _context_dict(context_bundle)
    ms = cb.get("market_structure", {}) or {}
    fib = cb.get("fibonacci", {}) or {}
    rejection = cb.get("rejection", {}) or {}
    auto_conf = cb.get("automated_confluence", {}) or {}

    tolerance = max(abs(entry) * 0.0035, (atr_val or 0) * 0.75)

    confluences.append(f"ABC C-ending phase {phase_alignment}")
    confluences.append(f"ABC C-ending score {int(phase_score)}")
    confluences.append(f"ABC anchor {entry_label}")

    if bool(auto_conf.get("high_confluence")):
        confluences.append("High S/R + Fib confluence")

    if side == "LONG":
        if bool(rejection.get("support_rejection_confirmed")):
            confluences.append("Support rejection confirmed")

        nearest_support = _safe_float(ms.get("nearest_support"))
        if nearest_support is not None and abs(entry - nearest_support) <= tolerance:
            confluences.append("Nearest support confluence")

        for zone_key in ["major_support_zone", "long_lookback_major_support_zone"]:
            z = ms.get(zone_key, {}) or {}
            if _zone_valid(z):
                low = _safe_float(z.get("low"))
                high = _safe_float(z.get("high"))
                if low is not None and high is not None and low - tolerance <= entry <= high + tolerance:
                    confluences.append(f"{zone_key} confluence")

    elif side == "SHORT":
        if bool(rejection.get("resistance_rejection_confirmed")):
            confluences.append("Resistance rejection confirmed")

        nearest_resistance = _safe_float(ms.get("nearest_resistance"))
        if nearest_resistance is not None and abs(entry - nearest_resistance) <= tolerance:
            confluences.append("Nearest resistance confluence")

        for zone_key in ["major_resistance_zone", "long_lookback_major_resistance_zone"]:
            z = ms.get(zone_key, {}) or {}
            if _zone_valid(z):
                low = _safe_float(z.get("low"))
                high = _safe_float(z.get("high"))
                if low is not None and high is not None and low - tolerance <= entry <= high + tolerance:
                    confluences.append(f"{zone_key} confluence")

    # EMA support/resistance confluence
    if ema50 is not None and abs(entry - float(ema50)) <= tolerance:
        confluences.append("EMA50 confluence")

    if ema200 is not None and abs(entry - float(ema200)) <= tolerance:
        confluences.append("EMA200 confluence")

    # Fib confluence from context bundle
    for src in ["visible", "elliott"]:
        f = fib.get(src, {}) or {}

        for key in ["normal_zone", "shallow_zone"]:
            zone = f.get(key, {}) or {}
            if _zone_valid(zone):
                low = _safe_float(zone.get("low"))
                high = _safe_float(zone.get("high"))
                if low is not None and high is not None and low - tolerance <= entry <= high + tolerance:
                    confluences.append(f"{src} Fib {key} confluence")

        gp = _safe_float(f.get("golden_pocket"))
        if gp is not None and abs(entry - gp) <= tolerance:
            confluences.append(f"{src} Fib golden pocket confluence")

        deep = _safe_float(f.get("deep_pullback"))
        if deep is not None and abs(entry - deep) <= tolerance:
            confluences.append(f"{src} Fib deep pullback confluence")

    return list(dict.fromkeys(confluences))


def _build_abc_stop(
    side: str,
    entry: float,
    pivots: Dict[str, Any],
    atr_val: Optional[float],
    context_bundle: Optional[Dict[str, Any]],
) -> Optional[float]:
    """
    Conservative structural stop for ABC C-ending.
    setup_planner.py will still add its own candidate validation after this.
    """
    side = str(side or "").upper()

    if atr_val is None or atr_val <= 0:
        return None

    cb = _context_dict(context_bundle)
    ms = cb.get("market_structure", {}) or {}

    atr_buffer = atr_val * 0.80

    if side == "LONG":
        lows: List[float] = []

        for value in [
            pivots.get("recent_low"),
            pivots.get("previous_low"),
            pivots.get("lowest"),
            ms.get("nearest_support"),
        ]:
            fv = _safe_float(value)
            if fv is not None and fv < entry:
                lows.append(fv)

        for zone_key in ["major_support_zone", "long_lookback_major_support_zone"]:
            z = ms.get(zone_key, {}) or {}
            if _zone_valid(z):
                low = _safe_float(z.get("low"))
                if low is not None and low < entry:
                    lows.append(low)

        if lows:
            base = min(lows)
            return float(base - atr_buffer)

        return float(entry - (atr_val * 1.5))

    if side == "SHORT":
        highs: List[float] = []

        for value in [
            pivots.get("recent_high"),
            pivots.get("previous_high"),
            pivots.get("highest"),
            ms.get("nearest_resistance"),
        ]:
            fv = _safe_float(value)
            if fv is not None and fv > entry:
                highs.append(fv)

        for zone_key in ["major_resistance_zone", "long_lookback_major_resistance_zone"]:
            z = ms.get(zone_key, {}) or {}
            if _zone_valid(z):
                high = _safe_float(z.get("high"))
                if high is not None and high > entry:
                    highs.append(high)

        if highs:
            base = max(highs)
            return float(base + atr_buffer)

        return float(entry + (atr_val * 1.5))

    return None


def make_abc_c_ending_idea(
    side: str,
    current_price: float,
    pivots: Dict[str, Any],
    ema50: Optional[float],
    ema200: Optional[float],
    atr_val: Optional[float],
    backend_trend_regime: Dict[str, Any],
    user_tf: str,
    context_bundle: Optional[Dict[str, Any]],
    elliott_phase_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return a raw setup idea.

    setup_planner.py must convert this using _make_candidate().
    """
    try:
        side = str(side or "").upper()

        if side not in ["LONG", "SHORT"]:
            return None

        if current_price is None or current_price <= 0:
            return None

        if atr_val is None or atr_val <= 0:
            return None

        ctx = elliott_phase_context if isinstance(elliott_phase_context, dict) else {}
        execution = ctx.get("execution", {}) or {}

        execution_phase = str(execution.get("elliott_phase") or "").upper()
        execution_side = str(execution.get("elliott_side") or "").upper()
        phase_alignment = str(ctx.get("elliott_phase_alignment") or "UNKNOWN").upper()
        phase_score = _safe_float(ctx.get("elliott_phase_score"), 0.0) or 0.0
        phase_reason = str(ctx.get("elliott_phase_reason") or "")

        if execution_phase != "ABC_C_ENDING":
            return None

        if phase_alignment == "CONFLICT":
            return None

        if execution_side in ["LONG", "SHORT"] and execution_side != side:
            return None

        cb = _context_dict(context_bundle)
        rejection = cb.get("rejection", {}) or {}
        auto_conf = cb.get("automated_confluence", {}) or {}

        support_rejection = bool(rejection.get("support_rejection_confirmed"))
        resistance_rejection = bool(rejection.get("resistance_rejection_confirmed"))
        high_confluence = bool(auto_conf.get("high_confluence"))

        has_rejection = support_rejection if side == "LONG" else resistance_rejection

        # WEAK/UNKNOWN ABC phase should require real rejection.
        if phase_alignment in ["WEAK", "UNKNOWN"] and not has_rejection:
            return None

        anchors = _collect_abc_entry_anchors(
            side=side,
            current_price=float(current_price),
            pivots=pivots or {},
            ema50=ema50,
            ema200=ema200,
            context_bundle=context_bundle,
        )

        if not anchors:
            return None

        entry_label, entry = anchors[0]

        # Avoid entry too far from current price.
        distance_pct = abs(float(current_price) - float(entry)) / float(current_price) * 100.0
        if distance_pct > _max_entry_distance_pct(user_tf):
            return None

        confluences = _build_abc_confluences(
            side=side,
            entry=entry,
            entry_label=entry_label,
            pivots=pivots or {},
            ema50=ema50,
            ema200=ema200,
            atr_val=atr_val,
            context_bundle=context_bundle,
            phase_alignment=phase_alignment,
            phase_score=phase_score,
        )

        # Need actual structure confirmation.
        # STRONG/MEDIUM can pass with high confluence.
        # WEAK/UNKNOWN already required rejection above.
        if not has_rejection and not high_confluence and len(confluences) < 4:
            return None

        stop = _build_abc_stop(
            side=side,
            entry=entry,
            pivots=pivots or {},
            atr_val=atr_val,
            context_bundle=context_bundle,
        )

        if stop is None:
            return None

        if side == "LONG" and stop >= entry:
            return None

        if side == "SHORT" and stop <= entry:
            return None

        entry_type = "abc_c_ending_long" if side == "LONG" else "abc_c_ending_short"

        return {
            "valid": True,
            "side": side,
            "entry": float(entry),
            "stop_loss": float(stop),
            "entry_type": entry_type,
            "source": "elliott_phase_detector",
            "template_priority": 18,
            "confluences": confluences,
            "abc_c_ending": True,
            "abc_status": "C_ENDING",
            "elliott_phase_alignment": phase_alignment,
            "elliott_phase_score": phase_score,
            "elliott_phase_reason": phase_reason,
            "execution_elliott_phase": execution_phase,
            "execution_elliott_side": execution_side,
            "phase_setup_note": (
                "ABC C-ending setup idea generated from Elliott phase context. "
                "Final approval depends on setup_planner SL/RR/no-chase/HTF validation."
            ),
            "entry_label": entry_label,
        }

    except Exception:
        return None