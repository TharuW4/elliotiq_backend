# app/models/mtf_elliott_confirmation.py

from typing import Any, Dict, Optional, Tuple


def _safe_upper(value: Any, default: str = "UNKNOWN") -> str:
    if value is None:
        return default
    text = str(value).strip().upper()
    return text if text else default

def _flatten_recent_wave_text(wave_result: Any, limit: int = 8) -> str:
    """
    Converts dict/list wave data into a compact text string.
    Helps side/phase detection work with macro_waves, swing_waves, confirm_waves.
    """
    try:
        if isinstance(wave_result, list):
            parts = []

            for w in wave_result[-limit:]:
                if isinstance(w, dict):
                    parts.append(str(w.get("pattern_type", "")))
                    parts.append(str(w.get("lbl", "")))

                    p = w.get("p", {}) or {}
                    parts.append(str(p.get("type", "")))
                    parts.append(str(p.get("price", "")))
                    parts.append(str(p.get("date", "")))
                else:
                    parts.append(str(w))

            return " ".join(parts).upper()

        if isinstance(wave_result, dict):
            return str(wave_result).upper()

        return str(wave_result).upper()

    except Exception:
        return str(wave_result).upper()

def _extract_wave_side(wave_result: Any) -> str:
    """
    Try to infer higher-timeframe Elliott side.
    Returns: LONG / SHORT / MIXED / UNKNOWN

    Important:
    Works for both dict and list wave data.
    BearishImpulse / BearishCorrection must be treated as SHORT.
    """
    if not wave_result:
        return "UNKNOWN"

    labels_text = _flatten_recent_wave_text(wave_result)

    # Explicit bearish labels first. This must work for both dict and list waves.
    if "BEARISHIMPULSE" in labels_text or "BEARISHCORRECTION" in labels_text:
        return "SHORT"

    # Dict may already contain explicit side/bias fields.
    if isinstance(wave_result, dict):
        for key in [
            "preferred_side",
            "wave_side",
            "side",
            "direction",
            "trend",
            "bias",
            "structure_bias",
        ]:
            value = wave_result.get(key)
            if value:
                text = _safe_upper(value)
                if text in ("LONG", "BULLISH", "UP", "UPTREND"):
                    return "LONG"
                if text in ("SHORT", "BEARISH", "DOWN", "DOWNTREND"):
                    return "SHORT"
                if text in ("MIXED", "NEUTRAL", "RANGE", "CHOPPY"):
                    return "MIXED"

    bullish_hits = 0
    bearish_hits = 0

    bullish_words = [
        "BULLISH",
        "UPTREND",
        "IMPULSE UP",
        "WAVE 3 UP",
        "WAVE 5 UP",
        "LONG",
    ]

    bearish_words = [
        "BEARISH",
        "DOWNTREND",
        "IMPULSE DOWN",
        "WAVE 3 DOWN",
        "WAVE 5 DOWN",
        "SHORT",
        "BEARISHIMPULSE",
        "BEARISHCORRECTION",
    ]

    for word in bullish_words:
        if word in labels_text:
            bullish_hits += 1

    for word in bearish_words:
        if word in labels_text:
            bearish_hits += 1

    # Plain "Impulse" without bearish wording is treated as bullish.
    if bullish_hits == 0 and bearish_hits == 0:
        if "IMPULSE" in labels_text and "BEARISH" not in labels_text:
            bullish_hits += 1

    if bullish_hits > bearish_hits:
        return "LONG"
    if bearish_hits > bullish_hits:
        return "SHORT"
    if bullish_hits and bearish_hits:
        return "MIXED"

    return "UNKNOWN"


def _extract_wave_phase(wave_result: Any) -> str:
    """
    Try to infer higher timeframe wave phase.
    Returns: IMPULSE / CORRECTION / PULLBACK / UNKNOWN
    """
    if not wave_result:
        return "UNKNOWN"

    text = _flatten_recent_wave_text(wave_result)

    if any(word in text for word in ["ABC", "CORRECTION", "CORRECTIVE", "WAVE 2", "WAVE 4"]):
        return "CORRECTION"

    if any(word in text for word in ["WAVE 3", "WAVE 5", "IMPULSE", "IMPULSIVE"]):
        return "IMPULSE"

    if any(word in text for word in ["PULLBACK", "RETRACE", "RETRACEMENT"]):
        return "PULLBACK"

    return "UNKNOWN"


def _side_score(local_side: str, htf_side: str, weight: int) -> Tuple[int, str]:
    local_side = _safe_upper(local_side)
    htf_side = _safe_upper(htf_side)

    if htf_side == "UNKNOWN":
        return int(weight * 0.35), "unknown"
    if htf_side == "MIXED":
        return int(weight * 0.45), "mixed"
    if htf_side == local_side:
        return weight, "aligned"

    return 0, "conflict"


def _phase_score(phase: str, weight: int) -> Tuple[int, str]:
    phase = _safe_upper(phase)

    if phase == "IMPULSE":
        return weight, "impulse"
    if phase in ("PULLBACK", "CORRECTION"):
        return int(weight * 0.60), "corrective/pullback"
    return int(weight * 0.35), "unknown"


def _alignment_from_score(score: int, has_direct_conflict: bool) -> str:
    if has_direct_conflict and score < 55:
        return "CONFLICT"
    if score >= 85:
        return "STRONG"
    if score >= 65:
        return "MEDIUM"
    if score >= 45:
        return "WEAK"
    if has_direct_conflict:
        return "CONFLICT"
    return "WEAK"


def confirm_mtf_elliott_alignment(
    candidate: Dict[str, Any],
    mtf_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Adds MTF Elliott confirmation fields for a local Elliott candidate.

    Expected candidate fields:
      - side / direction: LONG or SHORT
      - wave3_status: ANTICIPATION / CONFIRMED
      - template_type / pattern_type optional

    Expected mtf_context fields:
      - confirm_waves
      - swing_waves
      - macro_waves
      - confirm_tf
      - swing_tf
      - macro_tf
    """
    mtf_context = mtf_context or {}

    local_side = _safe_upper(
        candidate.get("side")
        or candidate.get("direction")
        or candidate.get("trade_side")
        or candidate.get("preferred_side")
    )

    if local_side not in ("LONG", "SHORT"):
        return {
            "mtf_elliott_alignment": "UNKNOWN",
            "mtf_wave_side": "UNKNOWN",
            "mtf_wave_phase": "UNKNOWN",
            "mtf_confirmation_score": 0,
            "mtf_confirmation_reason": "Local Elliott candidate side is unknown, so MTF confirmation cannot be calculated.",
        }

    confirm_waves = mtf_context.get("confirm_waves")
    swing_waves = mtf_context.get("swing_waves")
    macro_waves = mtf_context.get("macro_waves")

    confirm_tf = mtf_context.get("confirm_tf", "Confirm TF")
    swing_tf = mtf_context.get("swing_tf", "Swing TF")
    macro_tf = mtf_context.get("macro_tf", "Macro TF")

    confirm_side = _extract_wave_side(confirm_waves)
    swing_side = _extract_wave_side(swing_waves)
    macro_side = _extract_wave_side(macro_waves)

    confirm_phase = _extract_wave_phase(confirm_waves)
    swing_phase = _extract_wave_phase(swing_waves)
    macro_phase = _extract_wave_phase(macro_waves)

    score = 0
    reasons = []
    has_direct_conflict = False

    # Confirm TF side alignment: 35
    part, state = _side_score(local_side, confirm_side, 35)
    score += part
    reasons.append(f"{confirm_tf}: side={confirm_side}, phase={confirm_phase}, state={state}")
    if state == "conflict":
        has_direct_conflict = True

    # Swing TF side alignment: 35
    part, state = _side_score(local_side, swing_side, 35)
    score += part
    reasons.append(f"{swing_tf}: side={swing_side}, phase={swing_phase}, state={state}")
    if state == "conflict":
        has_direct_conflict = True

    # Macro TF should not strongly conflict: 15
    if macro_side == local_side:
        score += 15
        macro_state = "aligned"
    elif macro_side in ("UNKNOWN", "MIXED"):
        score += 8
        macro_state = "neutral/unknown"
    else:
        score += 0
        macro_state = "conflict"
        # Macro conflict alone should be caution, not always full hard conflict.
        # But it matters when confirm/swing are also weak.
        if confirm_side != local_side and swing_side != local_side:
            has_direct_conflict = True

    reasons.append(f"{macro_tf}: side={macro_side}, phase={macro_phase}, state={macro_state}")

    # Local wave status: 15
    wave3_status = _safe_upper(candidate.get("wave3_status"))
    if wave3_status == "CONFIRMED":
        score += 15
        reasons.append("Local Wave 3 status is CONFIRMED.")
    elif wave3_status == "ANTICIPATION":
        score += 9
        reasons.append("Local Wave 3 status is ANTICIPATION.")
    else:
        score += 5
        reasons.append("Local Wave 3 status is unknown/early.")

    score = max(0, min(100, int(score)))

    alignment = _alignment_from_score(score, has_direct_conflict)

    # Overall MTF side
    sides = [confirm_side, swing_side, macro_side]
    long_count = sides.count("LONG")
    short_count = sides.count("SHORT")

    if long_count > short_count:
        mtf_wave_side = "LONG"
    elif short_count > long_count:
        mtf_wave_side = "SHORT"
    elif long_count == short_count and long_count > 0:
        mtf_wave_side = "MIXED"
    else:
        mtf_wave_side = "UNKNOWN"

    # Overall MTF phase
    phases = [confirm_phase, swing_phase, macro_phase]
    if phases.count("IMPULSE") >= 2:
        mtf_wave_phase = "IMPULSE"
    elif phases.count("CORRECTION") >= 2:
        mtf_wave_phase = "CORRECTION"
    elif phases.count("PULLBACK") >= 1:
        mtf_wave_phase = "PULLBACK"
    else:
        mtf_wave_phase = "UNKNOWN"

    return {
        "mtf_elliott_alignment": alignment,
        "mtf_wave_side": mtf_wave_side,
        "mtf_wave_phase": mtf_wave_phase,
        "mtf_confirmation_score": score,
        "mtf_confirmation_reason": " | ".join(reasons),
    }