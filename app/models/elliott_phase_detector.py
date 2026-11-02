# app/models/elliott_phase_detector.py

"""
Elliott Phase Detector

Goal:
- Convert detected Elliott wave labels into a structured phase context.
- This file does NOT create trades.
- This file does NOT create entry/SL/TP.
- setup_planner.py can later use this as a permission/filter layer.

Supported broad phases:
- WAVE_2_PULLBACK
- WAVE_3_RUNNING
- WAVE_4_PULLBACK
- WAVE_5_RUNNING
- WAVE_5_ENDING
- ABC_CORRECTION
- ABC_C_ENDING
- UNKNOWN

Supports bullish and bearish pattern labels:
- Impulse
- BearishImpulse
- Correction
- BearishCorrection
"""

from typing import Any, Dict, List, Optional


def _safe_upper(value: Any, default: str = "UNKNOWN") -> str:
    if value is None:
        return default
    text = str(value).strip().upper()
    return text if text else default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        return v
    except Exception:
        return default


def _normalize_waves(waves: Any) -> List[Dict[str, Any]]:
    """
    Expected wave shape:
    {
        "lbl": "1/2/3/4/5/A/B/C",
        "pattern_type": "Impulse/BearishImpulse/Correction/BearishCorrection",
        "p": {"price": float, "date": "...", "type": "high/low"}
    }
    """
    if not waves:
        return []

    if isinstance(waves, list):
        return [w for w in waves if isinstance(w, dict)]

    return []


def _latest_pattern_group(waves: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract recent pattern information from wave labels.

    Since current wave list is flattened, we infer using the latest detected labels.
    This is intentionally defensive and simple for the first version.
    """
    result = {
        "pattern_type": "UNKNOWN",
        "latest_label": "UNKNOWN",
        "latest_price": None,
        "latest_date": "",
        "labels_seen": [],
        "impulse_labels": [],
        "correction_labels": [],
    }

    if not waves:
        return result

    try:
        last = waves[-1]
        pattern_type = _safe_upper(last.get("pattern_type"))
        latest_label = _safe_upper(last.get("lbl"))
        p = last.get("p", {}) or {}

        labels_seen = []
        impulse_labels = []
        correction_labels = []

        for w in waves[-12:]:
            lbl = _safe_upper(w.get("lbl"))
            pt = _safe_upper(w.get("pattern_type"))

            if lbl and lbl != "UNKNOWN":
                labels_seen.append(lbl)

            if pt in ("IMPULSE", "BEARISHIMPULSE") and lbl in ("1", "2", "3", "4", "5"):
                impulse_labels.append(lbl)

            if pt in ("CORRECTION", "BEARISHCORRECTION") and lbl in ("A", "B", "C"):
                correction_labels.append(lbl)

        result.update({
            "pattern_type": pattern_type,
            "latest_label": latest_label,
            "latest_price": _safe_float(p.get("price")),
            "latest_date": str(p.get("date", "")),
            "labels_seen": labels_seen,
            "impulse_labels": impulse_labels,
            "correction_labels": correction_labels,
        })

    except Exception:
        pass

    return result


def _side_from_pattern(pattern_type: str) -> str:
    pt = _safe_upper(pattern_type)

    if pt == "IMPULSE":
        return "LONG"

    if pt == "BEARISHIMPULSE":
        return "SHORT"

    # Normal Correction means A down, B up, C down after bullish impulse.
    # It is not automatically LONG/SHORT; side depends on ending confirmation.
    if pt == "CORRECTION":
        return "LONG"

    # BearishCorrection means A up, B down, C up after bearish impulse.
    if pt == "BEARISHCORRECTION":
        return "SHORT"

    return "UNKNOWN"


def _phase_from_latest_group(group: Dict[str, Any]) -> Dict[str, Any]:
    pattern_type = _safe_upper(group.get("pattern_type"))
    latest_label = _safe_upper(group.get("latest_label"))
    impulse_labels = group.get("impulse_labels") or []
    correction_labels = group.get("correction_labels") or []

    side = _side_from_pattern(pattern_type)

    phase = "UNKNOWN"
    abc_status = "NONE"
    trade_permission = "WAIT"
    confidence = 35
    reason = "No clear Elliott phase detected."

    if pattern_type in ("IMPULSE", "BEARISHIMPULSE"):
        if latest_label == "1":
            phase = "WAVE_2_PULLBACK"
            trade_permission = "CAUTION"
            confidence = 45
            reason = "Wave 1 detected. Waiting for Wave 2 pullback before Wave 3 continuation."

        elif latest_label == "2":
            phase = "WAVE_2_PULLBACK"
            trade_permission = "ALLOW_CONTINUATION"
            confidence = 75
            reason = "Wave 2 pullback detected. This supports a possible Wave 3 continuation setup."

        elif latest_label == "3":
            phase = "WAVE_3_RUNNING"
            trade_permission = "CAUTION"
            confidence = 65
            reason = "Wave 3 appears to be running. Avoid chasing unless backend has a clean retest setup."

        elif latest_label == "4":
            phase = "WAVE_4_PULLBACK"
            trade_permission = "ALLOW_CONTINUATION"
            confidence = 70
            reason = "Wave 4 pullback detected. This supports a possible Wave 5 continuation setup."

        elif latest_label == "5":
            phase = "WAVE_5_ENDING"
            trade_permission = "CAUTION"
            confidence = 70
            reason = "Wave 5 detected. Trend may be mature; avoid late continuation without strong confirmation."

        else:
            if "2" in impulse_labels and "3" not in impulse_labels:
                phase = "WAVE_2_PULLBACK"
                trade_permission = "ALLOW_CONTINUATION"
                confidence = 65
                reason = "Recent impulse labels suggest Wave 2 pullback before Wave 3."
            elif "4" in impulse_labels and "5" not in impulse_labels:
                phase = "WAVE_4_PULLBACK"
                trade_permission = "ALLOW_CONTINUATION"
                confidence = 60
                reason = "Recent impulse labels suggest Wave 4 pullback before Wave 5."

    elif pattern_type in ("CORRECTION", "BEARISHCORRECTION"):
        abc_status = "ACTIVE"

        if latest_label == "A":
            phase = "ABC_CORRECTION"
            trade_permission = "WAIT"
            confidence = 50
            reason = "A-wave correction detected. Correction is early; wait for B/C development."

        elif latest_label == "B":
            phase = "ABC_CORRECTION"
            trade_permission = "WAIT"
            confidence = 55
            reason = "B-wave correction detected. Avoid assuming correction is complete before C-wave."

        elif latest_label == "C":
            phase = "ABC_C_ENDING"
            abc_status = "C_ENDING"
            trade_permission = "ALLOW_ABC_ENDING_WITH_CONFIRMATION"
            confidence = 68
            reason = "C-wave area detected. Possible correction ending, but requires rejection/support-resistance confirmation."

        else:
            if "C" in correction_labels:
                phase = "ABC_C_ENDING"
                abc_status = "C_ENDING"
                trade_permission = "ALLOW_ABC_ENDING_WITH_CONFIRMATION"
                confidence = 60
                reason = "Recent correction labels include C-wave. Possible correction ending."
            else:
                phase = "ABC_CORRECTION"
                trade_permission = "WAIT"
                confidence = 45
                reason = "Correction structure detected, but C-wave ending is not clear."

    return {
        "elliott_phase": phase,
        "elliott_side": side,
        "elliott_pattern": pattern_type,
        "latest_label": latest_label,
        "phase_confidence": confidence,
        "phase_reason": reason,
        "abc_status": abc_status,
        "trade_permission": trade_permission,
    }


def detect_single_tf_phase(
    waves: Any,
    timeframe: str = "unknown",
) -> Dict[str, Any]:
    normalized = _normalize_waves(waves)
    group = _latest_pattern_group(normalized)
    phase_info = _phase_from_latest_group(group)

    phase_info.update({
        "timeframe": timeframe,
        "wave_count": len(normalized),
        "latest_price": group.get("latest_price"),
        "latest_date": group.get("latest_date"),
        "labels_seen": group.get("labels_seen", []),
    })

    return phase_info


def _score_tf_alignment(local_side: str, htf_side: str, weight: int) -> int:
    local_side = _safe_upper(local_side)
    htf_side = _safe_upper(htf_side)

    if local_side not in ("LONG", "SHORT"):
        return int(weight * 0.35)

    if htf_side == local_side:
        return weight

    if htf_side in ("UNKNOWN", "MIXED"):
        return int(weight * 0.45)

    return 0


def _combine_phase_context(
    execution_phase: Dict[str, Any],
    confirm_phase: Dict[str, Any],
    swing_phase: Dict[str, Any],
    macro_phase: Dict[str, Any],
) -> Dict[str, Any]:
    local_side = _safe_upper(execution_phase.get("elliott_side"))
    confirm_side = _safe_upper(confirm_phase.get("elliott_side"))
    swing_side = _safe_upper(swing_phase.get("elliott_side"))
    macro_side = _safe_upper(macro_phase.get("elliott_side"))

    score = 0
    score += _score_tf_alignment(local_side, confirm_side, 35)
    score += _score_tf_alignment(local_side, swing_side, 35)
    score += _score_tf_alignment(local_side, macro_side, 15)

    local_permission = _safe_upper(execution_phase.get("trade_permission"))
    local_phase = _safe_upper(execution_phase.get("elliott_phase"))

    if "ALLOW" in local_permission:
        score += 15
    elif local_permission == "CAUTION":
        score += 8
    else:
        score += 4

    score = max(0, min(100, int(score)))

    if score >= 80:
        phase_alignment = "STRONG"
    elif score >= 62:
        phase_alignment = "MEDIUM"
    elif score >= 42:
        phase_alignment = "WEAK"
    else:
        phase_alignment = "CONFLICT"

    # Conservative permission.
    if phase_alignment == "CONFLICT":
        final_permission = "WAIT"
    elif local_phase in ("WAVE_5_ENDING", "ABC_CORRECTION"):
        final_permission = "CAUTION"
    elif local_phase == "ABC_C_ENDING":
        final_permission = "ALLOW_ABC_ENDING_WITH_CONFIRMATION"
    elif "ALLOW" in local_permission and phase_alignment in ("STRONG", "MEDIUM", "WEAK"):
        final_permission = "ALLOW_CONTINUATION"
    else:
        final_permission = "CAUTION"

    reason = (
        f"Execution phase={execution_phase.get('elliott_phase')} side={local_side}; "
        f"Confirm phase={confirm_phase.get('elliott_phase')} side={confirm_side}; "
        f"Swing phase={swing_phase.get('elliott_phase')} side={swing_side}; "
        f"Macro phase={macro_phase.get('elliott_phase')} side={macro_side}; "
        f"Phase alignment={phase_alignment} score={score}."
    )

    return {
        "elliott_phase_alignment": phase_alignment,
        "elliott_phase_score": score,
        "elliott_phase_permission": final_permission,
        "elliott_phase_reason": reason,
    }


def detect_elliott_phase_context(
    execution_waves: Any,
    confirm_waves: Any = None,
    swing_waves: Any = None,
    macro_waves: Any = None,
    execution_tf: str = "execution",
    confirm_tf: str = "confirm",
    swing_tf: str = "swing",
    macro_tf: str = "macro",
) -> Dict[str, Any]:
    """
    Main public function.

    Returns phase context for each timeframe + combined alignment.
    """
    execution_phase = detect_single_tf_phase(execution_waves, execution_tf)
    confirm_phase = detect_single_tf_phase(confirm_waves, confirm_tf)
    swing_phase = detect_single_tf_phase(swing_waves, swing_tf)
    macro_phase = detect_single_tf_phase(macro_waves, macro_tf)

    combined = _combine_phase_context(
        execution_phase=execution_phase,
        confirm_phase=confirm_phase,
        swing_phase=swing_phase,
        macro_phase=macro_phase,
    )

    return {
        "execution": execution_phase,
        "confirm": confirm_phase,
        "swing": swing_phase,
        "macro": macro_phase,
        **combined,
    }


def format_elliott_phase_context_text(phase_context: Dict[str, Any]) -> str:
    """
    Human-readable text for market_context/debug/AI explanation.
    """
    if not isinstance(phase_context, dict):
        return "--- ELLIOTT PHASE CONTEXT ---\nUnavailable."

    def line(label: str, data: Dict[str, Any]) -> str:
        data = data or {}
        return (
            f"{label}: "
            f"phase={data.get('elliott_phase', 'UNKNOWN')}, "
            f"side={data.get('elliott_side', 'UNKNOWN')}, "
            f"pattern={data.get('elliott_pattern', 'UNKNOWN')}, "
            f"label={data.get('latest_label', 'UNKNOWN')}, "
            f"confidence={data.get('phase_confidence', 0)}, "
            f"permission={data.get('trade_permission', 'WAIT')}"
        )

    lines = [
        "--- ELLIOTT PHASE CONTEXT ---",
        line("Execution", phase_context.get("execution", {})),
        line("Confirm", phase_context.get("confirm", {})),
        line("Swing", phase_context.get("swing", {})),
        line("Macro", phase_context.get("macro", {})),
        f"Phase Alignment: {phase_context.get('elliott_phase_alignment', 'UNKNOWN')}",
        f"Phase Score: {phase_context.get('elliott_phase_score', 0)}",
        f"Phase Permission: {phase_context.get('elliott_phase_permission', 'WAIT')}",
        f"Phase Reason: {phase_context.get('elliott_phase_reason', '')}",
    ]

    return "\n".join(lines)