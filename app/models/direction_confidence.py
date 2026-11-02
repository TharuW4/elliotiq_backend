import math
from typing import Any, Dict, Optional


SUPPORTED_TIMEFRAMES = {"1h", "60m", "4h", "240m"}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            if not value:
                return default

        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default

        return v
    except Exception:
        return default


def _normalize_timeframe(value: Any) -> str:
    tf = str(value or "").lower().strip()

    if tf in ["1hr", "1hour", "60"]:
        return "1h"

    if tf in ["4hr", "4hour", "240"]:
        return "4h"

    return tf


def _normalize_side(value: Any) -> str:
    side = str(value or "").upper().strip()

    if "LONG" in side or side in ["BUY", "BULL", "BULLISH"]:
        return "LONG"

    if "SHORT" in side or side in ["SELL", "BEAR", "BEARISH"]:
        return "SHORT"

    if side in ["WAIT", "NONE", "NEUTRAL", "NO_TRADE"]:
        return "WAIT"

    return "UNKNOWN"


def _text_upper(value: Any) -> str:
    return str(value or "").upper().strip()


def _contains_any(text: Any, words) -> bool:
    t = _text_upper(text)
    return any(str(w).upper() in t for w in words)


def _base_result(
    *,
    enabled: bool,
    user_timeframe: str,
    candidate_side: str,
    primary_direction: str,
    direction_confidence: int,
    trend_state: str,
    htf_alignment: str,
    allowed_trade_mode: str,
    advice: str,
    reason: str,
    signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    score = max(0, min(100, int(direction_confidence or 0)))

    return {
        "enabled": bool(enabled),
        "user_timeframe": user_timeframe,
        "candidate_side": candidate_side,
        "primary_direction": primary_direction,
        "direction_confidence": score,
        "trend_state": trend_state,
        "htf_alignment": htf_alignment,
        "allowed_trade_mode": allowed_trade_mode,
        "advice": str(advice or ""),
        "reason": str(reason or ""),
        "signals": signals or {},
    }


def _extract_regime_fields(backend_trend_regime: Optional[dict]) -> Dict[str, Any]:
    r = backend_trend_regime or {}

    preferred_side = _normalize_side(
        r.get("preferred_side")
        or r.get("setup_bias")
        or r.get("market_bias")
        or r.get("bias")
    )

    blocked_side = _normalize_side(r.get("blocked_side"))

    regime = str(r.get("regime") or r.get("market_phase") or "").upper().strip()
    market_bias = str(r.get("market_bias") or r.get("bias") or "").upper().strip()
    market_phase = str(r.get("market_phase") or "").upper().strip()

    confidence = _safe_float(
        r.get("confidence")
        or r.get("regime_confidence")
        or r.get("trend_confidence"),
        0.0,
    )

    # Some modules return confidence 0-1; normalize to 0-100.
    if confidence is not None and confidence <= 1.0:
        confidence = confidence * 100.0

    return {
        "preferred_side": preferred_side,
        "blocked_side": blocked_side,
        "regime": regime,
        "market_bias": market_bias,
        "market_phase": market_phase,
        "confidence": max(0.0, min(100.0, confidence or 0.0)),
        "raw": r,
    }


def _extract_mtf_elliott_fields(setup: Optional[dict], mtf_elliott_context: Optional[dict]) -> Dict[str, Any]:
    s = setup or {}
    m = mtf_elliott_context or {}

    alignment = str(
        s.get("mtf_elliott_alignment")
        or m.get("mtf_elliott_alignment")
        or m.get("alignment")
        or "UNKNOWN"
    ).upper().strip()

    wave_side = _normalize_side(
        s.get("mtf_wave_side")
        or m.get("mtf_wave_side")
        or m.get("wave_side")
    )

    wave_phase = str(
        s.get("mtf_wave_phase")
        or m.get("mtf_wave_phase")
        or m.get("wave_phase")
        or ""
    ).upper().strip()

    score = _safe_float(
        s.get("mtf_confirmation_score")
        or m.get("mtf_confirmation_score")
        or m.get("score"),
        0.0,
    )

    reason = str(
        s.get("mtf_confirmation_reason")
        or m.get("mtf_confirmation_reason")
        or m.get("reason")
        or ""
    )

    return {
        "alignment": alignment,
        "wave_side": wave_side,
        "wave_phase": wave_phase,
        "score": max(0.0, min(100.0, score or 0.0)),
        "reason": reason,
    }


def _extract_wave_fields(setup: Optional[dict]) -> Dict[str, Any]:
    s = setup or {}

    wave3_status = str(s.get("wave3_status") or "").upper().strip()
    wave3_confirmed = bool(s.get("wave3_confirmed") is True)
    wave_template = str(s.get("wave_template") or s.get("wave_template_name") or "").upper().strip()
    setup_type = str(s.get("setup_type") or s.get("entry_type") or "").upper().strip()

    return {
        "wave3_status": wave3_status,
        "wave3_confirmed": wave3_confirmed,
        "wave_template": wave_template,
        "setup_type": setup_type,
    }


def _infer_primary_direction(preferred_side: str, market_bias: str, candidate_side: str) -> str:
    if preferred_side in ["LONG", "SHORT"]:
        return preferred_side

    if "BULL" in market_bias or "LONG" in market_bias:
        return "LONG"

    if "BEAR" in market_bias or "SHORT" in market_bias:
        return "SHORT"

    if candidate_side in ["LONG", "SHORT"]:
        return candidate_side

    return "WAIT"


def _infer_trend_state(
    *,
    regime: str,
    market_phase: str,
    market_bias: str,
    wave_phase: str,
) -> str:
    joined = " ".join([regime, market_phase, market_bias, wave_phase]).upper()

    if any(x in joined for x in ["CHOP", "RANGE", "SIDEWAYS", "CONSOLIDATION"]):
        return "RANGE_OR_CHOP"

    if any(x in joined for x in ["REVERSAL", "CHOCH", "CHANGE_OF_CHARACTER"]):
        return "REVERSAL_FORMING"

    if any(x in joined for x in ["EXHAUST", "WAVE_5", "ENDING", "DISTRIBUTION"]):
        return "EXHAUSTION_RISK"

    if any(x in joined for x in ["PULLBACK", "RETEST", "CORRECTION"]):
        return "PULLBACK_IN_TREND"

    if any(x in joined for x in ["TREND", "IMPULSE", "BULLISH", "BEARISH"]):
        return "TRENDING"

    return "MIXED_OR_UNKNOWN"


def _mode_from_score_and_flags(
    *,
    score: int,
    conflict: bool,
    candidate_blocked: bool,
    trend_state: str,
    risky_anticipation: bool,
) -> str:
    if candidate_blocked:
        return "WAIT"

    if conflict:
        if score >= 55:
            return "CONFIRMATION_REQUIRED"
        return "WAIT_FOR_REVERSAL_CONFIRMATION"

    if risky_anticipation:
        return "CONFIRMATION_REQUIRED"

    if trend_state in ["RANGE_OR_CHOP", "REVERSAL_FORMING", "EXHAUSTION_RISK"]:
        if score >= 70:
            return "CONFIRMATION_REQUIRED"
        if score >= 50:
            return "TP1_ONLY"
        return "WAIT"

    if score >= 72:
        return "FULL_TRADE"

    if score >= 55:
        return "CONFIRMATION_REQUIRED"

    if score >= 40:
        return "TP1_ONLY"

    return "WAIT"


def evaluate_direction_confidence(
    *,
    candidate_side: str,
    user_timeframe: str,
    backend_trend_regime: Optional[dict] = None,
    setup: Optional[dict] = None,
    mtf_elliott_context: Optional[dict] = None,
    momentum_context: Optional[dict] = None,
    rsi_divergence_context: Optional[dict] = None,
    market_context: Optional[dict] = None,
) -> dict:
    """
    Shadow-only direction confidence evaluator.

    IMPORTANT:
    - This function does NOT approve/reject a signal.
    - This function does NOT change entry, SL, TP1, or TP2.
    - This function does NOT override setup_planner.
    - This function only returns advisory direction-quality information.

    Intended usage:
    - Attach result to API response as "direction_confidence".
    - Later compare this shadow output with actual trade outcomes.
    - Only after enough data, optionally use allowed_trade_mode as a soft mode selector.
    """
    tf = _normalize_timeframe(user_timeframe)
    side = _normalize_side(candidate_side)

    if tf not in SUPPORTED_TIMEFRAMES:
        return _base_result(
            enabled=False,
            user_timeframe=tf,
            candidate_side=side,
            primary_direction="UNKNOWN",
            direction_confidence=0,
            trend_state="NOT_APPLICABLE",
            htf_alignment="UNKNOWN",
            allowed_trade_mode="NOT_APPLICABLE",
            advice="Direction confidence shadow layer is currently enabled only for 1H and 4H signals.",
            reason=f"Unsupported user timeframe: {tf or 'unknown'}",
            signals={},
        )

    if side not in ["LONG", "SHORT"]:
        return _base_result(
            enabled=True,
            user_timeframe=tf,
            candidate_side=side,
            primary_direction="WAIT",
            direction_confidence=0,
            trend_state="NO_TRADE",
            htf_alignment="UNKNOWN",
            allowed_trade_mode="WAIT",
            advice="No LONG/SHORT candidate side was provided, so direction confidence is not actionable.",
            reason="Candidate side is missing or not tradable.",
            signals={},
        )

    regime_fields = _extract_regime_fields(backend_trend_regime)
    mtf_fields = _extract_mtf_elliott_fields(setup, mtf_elliott_context)
    wave_fields = _extract_wave_fields(setup)

    preferred_side = regime_fields["preferred_side"]
    blocked_side = regime_fields["blocked_side"]
    regime_confidence = regime_fields["confidence"]
    market_bias = regime_fields["market_bias"]
    regime = regime_fields["regime"]
    market_phase = regime_fields["market_phase"]

    primary_direction = _infer_primary_direction(
        preferred_side=preferred_side,
        market_bias=market_bias,
        candidate_side=side,
    )

    trend_state = _infer_trend_state(
        regime=regime,
        market_phase=market_phase,
        market_bias=market_bias,
        wave_phase=mtf_fields["wave_phase"],
    )

    score = 50
    reasons = []
    signals = {
        "candidate_matches_preferred_side": False,
        "candidate_is_blocked_side": False,
        "mtf_elliott_aligned": False,
        "mtf_elliott_conflict": False,
        "wave3_anticipation": False,
        "fresh_breakout_risk": False,
        "range_or_chop": False,
        "reversal_or_exhaustion_risk": False,
        "regime_confidence": regime_confidence,
        "mtf_confirmation_score": mtf_fields["score"],
    }

    # Backend trend regime / preferred side.
    if preferred_side in ["LONG", "SHORT"]:
        if side == preferred_side:
            score += 22
            signals["candidate_matches_preferred_side"] = True
            reasons.append(f"candidate side matches backend preferred side ({preferred_side})")
        else:
            score -= 28
            reasons.append(f"candidate side conflicts with backend preferred side ({preferred_side})")
    else:
        reasons.append("backend preferred side is not clearly LONG/SHORT")

    # Blocked side should be taken seriously, but this is still advisory.
    candidate_blocked = blocked_side in ["LONG", "SHORT"] and side == blocked_side
    if candidate_blocked:
        score -= 35
        signals["candidate_is_blocked_side"] = True
        reasons.append(f"candidate side is backend blocked side ({blocked_side})")

    # Backend regime confidence.
    if regime_confidence >= 75:
        score += 10
        reasons.append("backend trend regime confidence is strong")
    elif regime_confidence >= 55:
        score += 4
        reasons.append("backend trend regime confidence is moderate")
    elif regime_confidence > 0:
        score -= 8
        reasons.append("backend trend regime confidence is weak")
    else:
        reasons.append("backend trend regime confidence unavailable")

    # MTF Elliott alignment.
    mtf_alignment = mtf_fields["alignment"]
    mtf_side = mtf_fields["wave_side"]
    mtf_score = mtf_fields["score"]

    mtf_conflict = False

    if mtf_alignment in ["STRONG", "ALIGNED", "MEDIUM"]:
        if mtf_side in ["LONG", "SHORT"]:
            if mtf_side == side:
                bonus = 14 if mtf_alignment in ["STRONG", "ALIGNED"] else 8
                score += bonus
                signals["mtf_elliott_aligned"] = True
                reasons.append(f"MTF Elliott alignment supports {side} ({mtf_alignment})")
            else:
                score -= 20
                mtf_conflict = True
                signals["mtf_elliott_conflict"] = True
                reasons.append(f"MTF Elliott side conflicts with candidate side ({mtf_side} vs {side})")
        else:
            score += 4
            reasons.append(f"MTF Elliott alignment is {mtf_alignment}, but side is unknown")

    elif mtf_alignment in ["WEAK"]:
        score -= 5
        reasons.append("MTF Elliott alignment is weak")

    elif mtf_alignment in ["CONFLICT", "OPPOSITE"]:
        score -= 22
        mtf_conflict = True
        signals["mtf_elliott_conflict"] = True
        reasons.append("MTF Elliott alignment conflicts with candidate")

    if mtf_score >= 75:
        score += 5
        reasons.append("MTF Elliott confirmation score is strong")
    elif 0 < mtf_score < 40:
        score -= 5
        reasons.append("MTF Elliott confirmation score is low")

    # Wave 3 anticipation should not be rejected here; it should require confirmation.
    wave3_status = wave_fields["wave3_status"]
    risky_anticipation = False

    if wave3_status == "ANTICIPATION":
        score -= 8
        risky_anticipation = True
        signals["wave3_anticipation"] = True
        reasons.append("Wave 3 setup is in anticipation mode, so confirmation is preferred")

    elif wave3_status == "CONFIRMED" or wave_fields["wave3_confirmed"]:
        score += 7
        reasons.append("Wave 3 continuation appears confirmed")

    # Market state risk.
    if trend_state == "RANGE_OR_CHOP":
        score -= 12
        signals["range_or_chop"] = True
        reasons.append("market appears range/chop; direction quality is lower")

    elif trend_state == "REVERSAL_FORMING":
        score -= 10
        signals["reversal_or_exhaustion_risk"] = True
        reasons.append("market may be forming a reversal; confirmation is required")

    elif trend_state == "EXHAUSTION_RISK":
        score -= 12
        signals["reversal_or_exhaustion_risk"] = True
        reasons.append("trend exhaustion risk detected")

    elif trend_state == "PULLBACK_IN_TREND":
        score += 5
        reasons.append("setup appears to be a pullback inside trend")

    elif trend_state == "TRENDING":
        score += 7
        reasons.append("market appears trending")

    # Momentum / breakout risk if provided as dict or text.
    momentum_text = momentum_context
    if isinstance(momentum_context, dict):
        momentum_text = " ".join(str(v) for v in momentum_context.values())

    if _contains_any(momentum_text, ["FRESH_BREAKOUT"]):
        score -= 6
        signals["fresh_breakout_risk"] = True
        reasons.append("fresh breakout context prefers retest/confirmation before full trade")

    if _contains_any(momentum_text, ["RETEST_CONFIRMED", "RETEST HOLD", "RETEST_HOLD"]):
        score += 7
        reasons.append("momentum context includes retest confirmation")

    if _contains_any(momentum_text, ["FAKEOUT", "FAILED BREAKOUT", "CLOSE BACK INSIDE"]):
        score -= 12
        reasons.append("momentum context warns about fakeout/failed breakout")

    # RSI divergence as caution, not hard direction override.
    divergence_text = rsi_divergence_context
    if isinstance(rsi_divergence_context, dict):
        divergence_text = " ".join(str(v) for v in rsi_divergence_context.values())

    if side == "LONG" and _contains_any(divergence_text, ["BEARISH DIVERGENCE", "REGULAR BEARISH"]):
        score -= 8
        reasons.append("bearish RSI divergence conflicts with LONG continuation")

    if side == "SHORT" and _contains_any(divergence_text, ["BULLISH DIVERGENCE", "REGULAR BULLISH"]):
        score -= 8
        reasons.append("bullish RSI divergence conflicts with SHORT continuation")

    score = max(0, min(100, int(round(score))))

    conflict = (
        mtf_conflict
        or candidate_blocked
        or (
            preferred_side in ["LONG", "SHORT"]
            and side != preferred_side
        )
    )

    if conflict:
        htf_alignment = "CONFLICT"
    elif signals["candidate_matches_preferred_side"] or signals["mtf_elliott_aligned"]:
        htf_alignment = "ALIGNED"
    else:
        htf_alignment = "NEUTRAL"

    allowed_trade_mode = _mode_from_score_and_flags(
        score=score,
        conflict=conflict,
        candidate_blocked=candidate_blocked,
        trend_state=trend_state,
        risky_anticipation=risky_anticipation,
    )

    if allowed_trade_mode == "FULL_TRADE":
        advice = (
            "Direction quality is strong. Candidate side is aligned with the main context. "
            "This is advisory only; live signal behavior is unchanged."
        )

    elif allowed_trade_mode == "CONFIRMATION_REQUIRED":
        advice = (
            "Direction quality is acceptable but not clean enough for blind execution. "
            "Use lower-timeframe entry confirmation before acting. "
            "This is advisory only; live signal behavior is unchanged."
        )

    elif allowed_trade_mode == "TP1_ONLY":
        advice = (
            "Direction quality is mixed. Treat this as a cautious setup; TP1-first management is safer than expecting full TP2. "
            "This is advisory only; live signal behavior is unchanged."
        )

    elif allowed_trade_mode == "WAIT_FOR_REVERSAL_CONFIRMATION":
        advice = (
            "Direction context is conflicted. Do not assume reversal/continuation until structure confirms. "
            "This is advisory only; live signal behavior is unchanged."
        )

    else:
        advice = (
            "Direction quality is weak or blocked. Waiting for clearer structure is safer. "
            "This is advisory only; live signal behavior is unchanged."
        )

    return _base_result(
        enabled=True,
        user_timeframe=tf,
        candidate_side=side,
        primary_direction=primary_direction,
        direction_confidence=score,
        trend_state=trend_state,
        htf_alignment=htf_alignment,
        allowed_trade_mode=allowed_trade_mode,
        advice=advice,
        reason="; ".join(reasons),
        signals=signals,
    )