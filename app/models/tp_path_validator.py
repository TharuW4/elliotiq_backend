import math
import re
from typing import Any, Dict, List, Optional, Tuple


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


def _normalize_side(value: Any) -> str:
    text = str(value or "").upper().strip()

    if "LONG" in text or text in ["BUY", "BULL", "BULLISH"]:
        return "LONG"

    if "SHORT" in text or text in ["SELL", "BEAR", "BEARISH"]:
        return "SHORT"

    if text in ["WAIT", "NONE", "NEUTRAL", "NO_TRADE"]:
        return "WAIT"

    return "UNKNOWN"


def _normalize_timeframe(value: Any) -> str:
    tf = str(value or "").lower().strip()

    if tf in ["1hr", "1hour", "60"]:
        return "1h"

    if tf in ["4hr", "4hour", "240"]:
        return "4h"

    return tf


def _base_result(
    *,
    enabled: bool,
    user_timeframe: str,
    side: str,
    path_status: str,
    path_score: int,
    sr_strength: str,
    blocking_level: Optional[float],
    blocking_type: str,
    original_entry_valid: bool,
    recommended_trade_mode: str,
    required_action: str,
    exit_if_rejected: bool,
    continue_if_breaks: bool,
    advice: str,
    reason: str,
    levels: Optional[List[Dict[str, Any]]] = None,
    signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "user_timeframe": user_timeframe,
        "side": side,
        "path_status": str(path_status or "UNKNOWN").upper(),
        "path_score": max(0, min(100, int(path_score or 0))),
        "sr_strength": str(sr_strength or "UNKNOWN").upper(),
        "blocking_level": blocking_level,
        "blocking_type": str(blocking_type or "NONE").upper(),
        "original_entry_valid": bool(original_entry_valid),
        "recommended_trade_mode": str(recommended_trade_mode or "UNKNOWN").upper(),
        "required_action": str(required_action or "NONE").upper(),
        "exit_if_rejected": bool(exit_if_rejected),
        "continue_if_breaks": bool(continue_if_breaks),
        "advice": str(advice or ""),
        "reason": str(reason or ""),
        "levels": levels or [],
        "signals": signals or {},
    }


def _extract_numbers(text: Any) -> List[float]:
    try:
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
        out = []

        for n in nums:
            v = _safe_float(n)
            if v is not None:
                out.append(v)

        return out
    except Exception:
        return []


def _extract_context_line(market_context_text: Any, label: str) -> str:
    try:
        if not market_context_text:
            return ""

        pattern = rf"{re.escape(label)}:\s*(.+)"
        match = re.search(pattern, str(market_context_text), flags=re.IGNORECASE)

        if match:
            return match.group(1).strip()

    except Exception:
        pass

    return ""


def _parse_strength_from_line(line: str) -> Optional[float]:
    try:
        match = re.search(r"strength\s*[:=]?\s*(\d+(?:\.\d+)?)", str(line), flags=re.IGNORECASE)
        if not match:
            return None

        return _safe_float(match.group(1))
    except Exception:
        return None


def _add_level(
    levels: List[Dict[str, Any]],
    *,
    price: Optional[float],
    level_type: str,
    source: str,
    strength: Optional[float] = None,
) -> None:
    price = _safe_float(price)

    if price is None or price <= 0:
        return

    levels.append({
        "price": price,
        "type": str(level_type or "UNKNOWN").upper(),
        "source": str(source or "unknown"),
        "strength": _safe_float(strength, 1.0) or 1.0,
    })


def _collect_levels_from_market_context_text(market_context_text: Any) -> List[Dict[str, Any]]:
    """
    Supports text lines like:
    Nearest Support: 85.470 (strength 4)
    Major Support Zone: 85.000 - 85.500 (strength 3)
    Nearest Resistance: 664.20 (strength 4)
    Major Resistance Zone: 663.00 - 665.00 (strength 5)
    """
    levels: List[Dict[str, Any]] = []

    if not market_context_text:
        return levels

    support_labels = [
        "Nearest Support",
        "Major Support Zone",
        "Long Lookback Major Support Zone",
        "Support Zone",
    ]

    resistance_labels = [
        "Nearest Resistance",
        "Major Resistance Zone",
        "Long Lookback Major Resistance Zone",
        "Resistance Zone",
    ]

    for label in support_labels:
        line = _extract_context_line(market_context_text, label)
        if not line:
            continue

        nums = _extract_numbers(line)
        strength = _parse_strength_from_line(line)

        if not nums:
            continue

        # For zones, use both boundaries and midpoint.
        if "ZONE" in label.upper() and len(nums) >= 2:
            a, b = nums[0], nums[1]
            midpoint = (a + b) / 2.0

            _add_level(levels, price=a, level_type="SUPPORT", source=label, strength=strength)
            _add_level(levels, price=b, level_type="SUPPORT", source=label, strength=strength)
            _add_level(levels, price=midpoint, level_type="SUPPORT", source=f"{label} midpoint", strength=strength)
        else:
            _add_level(levels, price=nums[0], level_type="SUPPORT", source=label, strength=strength)

    for label in resistance_labels:
        line = _extract_context_line(market_context_text, label)
        if not line:
            continue

        nums = _extract_numbers(line)
        strength = _parse_strength_from_line(line)

        if not nums:
            continue

        if "ZONE" in label.upper() and len(nums) >= 2:
            a, b = nums[0], nums[1]
            midpoint = (a + b) / 2.0

            _add_level(levels, price=a, level_type="RESISTANCE", source=label, strength=strength)
            _add_level(levels, price=b, level_type="RESISTANCE", source=label, strength=strength)
            _add_level(levels, price=midpoint, level_type="RESISTANCE", source=f"{label} midpoint", strength=strength)
        else:
            _add_level(levels, price=nums[0], level_type="RESISTANCE", source=label, strength=strength)

    return levels


def _collect_levels_from_lists(
    *,
    support_levels: Optional[List[Any]] = None,
    resistance_levels: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    levels: List[Dict[str, Any]] = []

    for item in support_levels or []:
        if isinstance(item, dict):
            price = item.get("price") or item.get("level") or item.get("value")
            strength = item.get("strength") or item.get("score") or item.get("touches")
            source = item.get("source") or "support_levels"
        else:
            price = item
            strength = 1.0
            source = "support_levels"

        _add_level(levels, price=price, level_type="SUPPORT", source=source, strength=strength)

    for item in resistance_levels or []:
        if isinstance(item, dict):
            price = item.get("price") or item.get("level") or item.get("value")
            strength = item.get("strength") or item.get("score") or item.get("touches")
            source = item.get("source") or "resistance_levels"
        else:
            price = item
            strength = 1.0
            source = "resistance_levels"

        _add_level(levels, price=price, level_type="RESISTANCE", source=source, strength=strength)

    return levels


def _dedupe_levels(levels: List[Dict[str, Any]], tolerance_pct: float = 0.0015) -> List[Dict[str, Any]]:
    """
    Merge nearby levels of same type.
    Keeps highest strength/source info.
    """
    cleaned = []

    for lvl in levels or []:
        price = _safe_float(lvl.get("price"))
        lvl_type = str(lvl.get("type") or "").upper()
        strength = _safe_float(lvl.get("strength"), 1.0) or 1.0

        if price is None or price <= 0 or lvl_type not in ["SUPPORT", "RESISTANCE"]:
            continue

        merged = False

        for existing in cleaned:
            existing_price = existing["price"]
            existing_type = existing["type"]

            if existing_type != lvl_type:
                continue

            tolerance = max(abs(existing_price) * tolerance_pct, 1e-12)

            if abs(price - existing_price) <= tolerance:
                # Weighted average biased toward stronger level.
                existing_strength = _safe_float(existing.get("strength"), 1.0) or 1.0
                total_strength = existing_strength + strength

                existing["price"] = (
                    (existing_price * existing_strength) + (price * strength)
                ) / max(total_strength, 1e-12)

                existing["strength"] = max(existing_strength, strength)

                source = str(existing.get("source") or "")
                new_source = str(lvl.get("source") or "")
                if new_source and new_source not in source:
                    existing["source"] = f"{source}, {new_source}" if source else new_source

                merged = True
                break

        if not merged:
            cleaned.append({
                "price": price,
                "type": lvl_type,
                "source": lvl.get("source") or "unknown",
                "strength": strength,
            })

    cleaned.sort(key=lambda x: x["price"])
    return cleaned


def _is_between_for_path(side: str, price: float, entry: float, tp1: float) -> bool:
    """
    LONG path: entry < blocking resistance < tp1
    SHORT path: tp1 < blocking support < entry
    """
    if side == "LONG":
        return entry < price < tp1

    if side == "SHORT":
        return tp1 < price < entry

    return False


def _find_blocking_levels(side: str, entry: float, tp1: float, levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocking = []

    for lvl in levels or []:
        price = _safe_float(lvl.get("price"))
        lvl_type = str(lvl.get("type") or "").upper()

        if price is None:
            continue

        if not _is_between_for_path(side, price, entry, tp1):
            continue

        # Relevant wall type:
        # LONG is blocked mainly by resistance.
        # SHORT is blocked mainly by support.
        if side == "LONG" and lvl_type == "RESISTANCE":
            blocking.append(lvl)

        elif side == "SHORT" and lvl_type == "SUPPORT":
            blocking.append(lvl)

    # Closest-to-entry wall matters first.
    if side == "LONG":
        blocking.sort(key=lambda x: abs((_safe_float(x.get("price")) or 0) - entry))
    else:
        blocking.sort(key=lambda x: abs(entry - (_safe_float(x.get("price")) or 0)))

    return blocking


def _distance_pct(a: float, b: float) -> float:
    try:
        if a == 0:
            return 0.0
        return abs(a - b) / abs(a) * 100.0
    except Exception:
        return 0.0


def _score_sr_strength(
    *,
    blocking_level: Dict[str, Any],
    side: str,
    entry: float,
    tp1: float,
    tp2: Optional[float],
    direction_confidence: Optional[dict],
    entry_confirmation: Optional[dict],
    atr: Optional[float],
) -> Tuple[int, List[str], Dict[str, Any]]:
    """
    This score does NOT decide reject/block.
    It estimates how much attention the blocking S/R deserves.
    """
    reasons: List[str] = []
    signals: Dict[str, Any] = {}

    level = _safe_float(blocking_level.get("price"))
    raw_strength = _safe_float(blocking_level.get("strength"), 1.0) or 1.0

    score = 30

    # Raw source strength from market_context/support system.
    if raw_strength >= 5:
        score += 25
        reasons.append("blocking S/R has high source strength")
    elif raw_strength >= 3:
        score += 15
        reasons.append("blocking S/R has medium source strength")
    elif raw_strength >= 2:
        score += 8
        reasons.append("blocking S/R has some source strength")
    else:
        score += 2
        reasons.append("blocking S/R has low/unknown source strength")

    # Path location: if wall is close to entry, early rejection risk is higher.
    path_distance = abs(tp1 - entry)
    level_from_entry = abs((level or entry) - entry)
    level_progress_to_tp1 = (
        level_from_entry / path_distance * 100.0
        if path_distance > 0
        else 0.0
    )

    signals["level_progress_to_tp1_pct"] = round(level_progress_to_tp1, 2)

    if level_progress_to_tp1 <= 30:
        score += 15
        reasons.append("blocking level is close to entry, so rejection can happen early")
    elif level_progress_to_tp1 <= 65:
        score += 10
        reasons.append("blocking level is in the middle of entry-to-TP1 path")
    else:
        score += 4
        reasons.append("blocking level is near TP1, less dangerous for early entry")

    # ATR proximity: if wall is within about 1 ATR, it is more relevant.
    atr_value = _safe_float(atr)

    if atr_value and atr_value > 0 and level is not None:
        atr_distance = abs(level - entry) / atr_value
        signals["blocking_distance_atr"] = round(atr_distance, 2)

        if atr_distance <= 0.7:
            score += 12
            reasons.append("blocking level is within 0.7 ATR from entry")
        elif atr_distance <= 1.3:
            score += 8
            reasons.append("blocking level is within 1.3 ATR from entry")
        elif atr_distance <= 2.0:
            score += 4
            reasons.append("blocking level is within 2 ATR from entry")
    else:
        signals["blocking_distance_atr"] = None

    # Direction confidence helps decide whether to tolerate a wall.
    direction_score = None
    if isinstance(direction_confidence, dict):
        direction_score = _safe_float(direction_confidence.get("direction_confidence"))

    signals["direction_confidence"] = direction_score

    if direction_score is not None:
        if direction_score >= 80:
            score -= 8
            reasons.append("strong direction confidence lowers wall risk")
        elif direction_score >= 65:
            score -= 4
            reasons.append("acceptable direction confidence slightly lowers wall risk")
        elif direction_score < 50:
            score += 12
            reasons.append("weak direction confidence increases wall risk")

    # Entry confirmation helps decide whether the market is already reacting correctly.
    entry_status = ""
    entry_score = None

    if isinstance(entry_confirmation, dict):
        entry_status = str(entry_confirmation.get("confirmation_status") or "").upper()
        entry_score = _safe_float(entry_confirmation.get("confirmation_score"))

    signals["entry_confirmation_status"] = entry_status
    signals["entry_confirmation_score"] = entry_score

    if entry_status == "CONFIRMED":
        score -= 8
        reasons.append("entry confirmation is already strong")
    elif entry_status == "WAITING":
        score += 5
        reasons.append("entry confirmation is still waiting")
    elif entry_status in ["WEAK", "INVALIDATED"]:
        score += 12
        reasons.append("entry confirmation is weak or invalidated")

    score = max(0, min(100, int(round(score))))

    return score, reasons, signals


def _strength_label(score: int) -> str:
    if score >= 70:
        return "STRONG"

    if score >= 40:
        return "MEDIUM"

    return "WEAK"


def _mode_from_strength(score: int, side: str) -> Tuple[str, str, bool, bool, str]:
    """
    Returns:
    path_status, recommended_trade_mode, exit_if_rejected, continue_if_breaks, required_action
    """
    if score >= 70:
        if side == "LONG":
            return (
                "STRONG_SR_BEFORE_TP1",
                "REDUCED_RISK_OR_RECLAIM_CONFIRMATION",
                True,
                True,
                "WATCH_RECLAIM_OR_REJECTION",
            )

        if side == "SHORT":
            return (
                "STRONG_SR_BEFORE_TP1",
                "REDUCED_RISK_OR_BREAKDOWN_CONFIRMATION",
                True,
                True,
                "WATCH_BREAKDOWN_OR_REJECTION",
            )

    if score >= 40:
        return (
            "WATCH_SR_REACTION",
            "NORMAL_ENTRY_WITH_SR_WATCH",
            True,
            True,
            "WATCH_SR_REACTION",
        )

    return (
        "MINOR_SR_BEFORE_TP1",
        "NORMAL_ENTRY",
        False,
        True,
        "WATCH_MINOR_SR",
    )


def _build_advice(
    *,
    side: str,
    entry: float,
    tp1: float,
    blocking_level: Optional[float],
    blocking_type: str,
    sr_strength: str,
    path_status: str,
    recommended_trade_mode: str,
) -> str:
    if blocking_level is None:
        return (
            "TP1 path looks clean. No obvious blocking S/R was found between entry and TP1. "
            "Use normal entry confirmation and standard risk management."
        )

    if side == "LONG":
        if sr_strength == "WEAK":
            return (
                f"LONG path has a minor resistance near {blocking_level}. "
                "The signal is not blocked. Entry can remain valid, but watch how price reacts at that level before expecting TP1/TP2."
            )

        if sr_strength == "MEDIUM":
            return (
                f"LONG path has resistance before TP1 near {blocking_level}. "
                "Do not treat this as a completely clean path. If price rejects from that resistance, consider reducing risk, closing early, or tightening SL. "
                "If candle body reclaims above it and holds, continuation toward TP1/TP2 becomes healthier."
            )

        return (
            f"LONG path has strong resistance before TP1 near {blocking_level}. "
            "The signal is not blocked, but avoid blind full-risk execution. Use reduced risk or wait for reclaim/hold. "
            "If price rejects from this resistance, close early or tighten SL. If price closes above and retests/holds, continuation is more reliable."
        )

    if side == "SHORT":
        if sr_strength == "WEAK":
            return (
                f"SHORT path has a minor support near {blocking_level}. "
                "The signal is not blocked. Entry can remain valid, but watch how price reacts at that level before expecting TP1/TP2."
            )

        if sr_strength == "MEDIUM":
            return (
                f"SHORT path has support before TP1 near {blocking_level}. "
                "Do not treat this as a completely clean path. If price rejects from that support, consider reducing risk, closing early, or tightening SL. "
                "If candle body breaks below it and retests/rejects, continuation toward TP1/TP2 becomes healthier."
            )

        return (
            f"SHORT path has strong support before TP1 near {blocking_level}. "
            "The signal is not blocked, but avoid blind full-risk execution. Use reduced risk or wait for breakdown/retest rejection. "
            "If price rejects from this support, close early or tighten SL. If price closes below and retests/rejects, continuation is more reliable."
        )

    return "TP path risk advice unavailable."


def evaluate_tp_path_quality(
    *,
    side: str,
    entry_price: float,
    tp1: float,
    tp2: float | None = None,
    stop_loss: float | None = None,
    user_timeframe: str = "",
    market_context_text: str | None = None,
    support_levels: Optional[List[Any]] = None,
    resistance_levels: Optional[List[Any]] = None,
    entry_confirmation: Optional[dict] = None,
    direction_confidence: Optional[dict] = None,
    atr: float | None = None,
    current_price: float | None = None,
) -> dict:
    """
    Shadow-only TP path validator.

    IMPORTANT:
    - This function does NOT block a signal.
    - This function does NOT reject a setup.
    - This function does NOT change entry, SL, TP1, or TP2.
    - It only returns advisory risk/mode information.

    Purpose:
    - Detect S/R between entry and TP1.
    - Estimate whether the S/R is weak/medium/strong.
    - Provide practical advice:
      normal entry, watch reaction, reduced risk, break/reclaim confirmation.
    """
    tf = _normalize_timeframe(user_timeframe)
    normalized_side = _normalize_side(side)

    if normalized_side not in ["LONG", "SHORT"]:
        return _base_result(
            enabled=False,
            user_timeframe=tf,
            side=normalized_side,
            path_status="NOT_APPLICABLE",
            path_score=0,
            sr_strength="UNKNOWN",
            blocking_level=None,
            blocking_type="NONE",
            original_entry_valid=True,
            recommended_trade_mode="NOT_APPLICABLE",
            required_action="NONE",
            exit_if_rejected=False,
            continue_if_breaks=False,
            advice="TP path validation is not applicable because candidate side is not LONG/SHORT.",
            reason="Invalid or missing trade side.",
        )

    entry = _safe_float(entry_price)
    first_tp = _safe_float(tp1)
    second_tp = _safe_float(tp2)
    stop = _safe_float(stop_loss)

    if entry is None or first_tp is None:
        return _base_result(
            enabled=True,
            user_timeframe=tf,
            side=normalized_side,
            path_status="UNKNOWN",
            path_score=0,
            sr_strength="UNKNOWN",
            blocking_level=None,
            blocking_type="NONE",
            original_entry_valid=True,
            recommended_trade_mode="UNKNOWN",
            required_action="NONE",
            exit_if_rejected=False,
            continue_if_breaks=False,
            advice="TP path validation unavailable because entry or TP1 is missing.",
            reason="Missing entry_price or tp1.",
        )

    # Validate that TP1 is on correct side.
    if normalized_side == "LONG" and first_tp <= entry:
        return _base_result(
            enabled=True,
            user_timeframe=tf,
            side=normalized_side,
            path_status="INVALID_TP1_SIDE",
            path_score=0,
            sr_strength="UNKNOWN",
            blocking_level=None,
            blocking_type="NONE",
            original_entry_valid=False,
            recommended_trade_mode="CHECK_LEVELS",
            required_action="CHECK_LEVELS",
            exit_if_rejected=False,
            continue_if_breaks=False,
            advice="TP1 is not above entry for LONG. Check setup levels.",
            reason="Invalid LONG TP1 placement.",
        )

    if normalized_side == "SHORT" and first_tp >= entry:
        return _base_result(
            enabled=True,
            user_timeframe=tf,
            side=normalized_side,
            path_status="INVALID_TP1_SIDE",
            path_score=0,
            sr_strength="UNKNOWN",
            blocking_level=None,
            blocking_type="NONE",
            original_entry_valid=False,
            recommended_trade_mode="CHECK_LEVELS",
            required_action="CHECK_LEVELS",
            exit_if_rejected=False,
            continue_if_breaks=False,
            advice="TP1 is not below entry for SHORT. Check setup levels.",
            reason="Invalid SHORT TP1 placement.",
        )

    levels = []
    levels.extend(_collect_levels_from_market_context_text(market_context_text))
    levels.extend(_collect_levels_from_lists(
        support_levels=support_levels,
        resistance_levels=resistance_levels,
    ))

    levels = _dedupe_levels(levels)

    blocking_levels = _find_blocking_levels(
        normalized_side,
        entry,
        first_tp,
        levels,
    )

    signals = {
        "levels_found": len(levels),
        "blocking_levels_found": len(blocking_levels),
        "entry": entry,
        "tp1": first_tp,
        "tp2": second_tp,
        "stop_loss": stop,
        "current_price": _safe_float(current_price),
    }

    if not blocking_levels:
        return _base_result(
            enabled=True,
            user_timeframe=tf,
            side=normalized_side,
            path_status="CLEAN_TO_TP1",
            path_score=20,
            sr_strength="NONE",
            blocking_level=None,
            blocking_type="NONE",
            original_entry_valid=True,
            recommended_trade_mode="NORMAL_ENTRY",
            required_action="USE_ENTRY_CONFIRMATION",
            exit_if_rejected=False,
            continue_if_breaks=True,
            advice=(
                "TP1 path looks clean. No obvious blocking S/R was found between entry and TP1. "
                "Use normal entry confirmation and standard risk management."
            ),
            reason="No relevant support/resistance found between entry and TP1.",
            levels=levels,
            signals=signals,
        )

    nearest = blocking_levels[0]
    blocking_level = _safe_float(nearest.get("price"))
    blocking_type = str(nearest.get("type") or "UNKNOWN").upper()

    path_score, reasons, strength_signals = _score_sr_strength(
        blocking_level=nearest,
        side=normalized_side,
        entry=entry,
        tp1=first_tp,
        tp2=second_tp,
        direction_confidence=direction_confidence,
        entry_confirmation=entry_confirmation,
        atr=atr,
    )

    sr_strength = _strength_label(path_score)

    path_status, recommended_mode, exit_if_rejected, continue_if_breaks, required_action = _mode_from_strength(
        path_score,
        normalized_side,
    )

    signals.update(strength_signals)
    signals["blocking_source"] = nearest.get("source")
    signals["blocking_raw_strength"] = nearest.get("strength")
    signals["all_blocking_levels"] = blocking_levels[:5]

    advice = _build_advice(
        side=normalized_side,
        entry=entry,
        tp1=first_tp,
        blocking_level=blocking_level,
        blocking_type=blocking_type,
        sr_strength=sr_strength,
        path_status=path_status,
        recommended_trade_mode=recommended_mode,
    )

    return _base_result(
        enabled=True,
        user_timeframe=tf,
        side=normalized_side,
        path_status=path_status,
        path_score=path_score,
        sr_strength=sr_strength,
        blocking_level=blocking_level,
        blocking_type=blocking_type,
        original_entry_valid=True,
        recommended_trade_mode=recommended_mode,
        required_action=required_action,
        exit_if_rejected=exit_if_rejected,
        continue_if_breaks=continue_if_breaks,
        advice=advice,
        reason="; ".join(reasons),
        levels=levels,
        signals=signals,
    )