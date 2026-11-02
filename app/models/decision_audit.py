import math
from typing import Any, Dict, List, Optional


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").replace("x", "").strip()
            if not value:
                return default

        v = float(value)

        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        v = _safe_float(value, None)
        if v is None:
            return default
        return int(round(v))
    except Exception:
        return default


def _text(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        return str(value).strip()
    except Exception:
        return default


def _upper(value: Any) -> str:
    return _text(value).upper().strip()


def _bool(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:
        return False


def _get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    try:
        cur = data
        for key in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(key)
        return cur if cur is not None else default
    except Exception:
        return default


def _normalize_side(value: Any) -> str:
    side = _upper(value)

    if side in ["LONG", "BUY", "BULL", "BULLISH"]:
        return "LONG"

    if side in ["SHORT", "SELL", "BEAR", "BEARISH"]:
        return "SHORT"

    if "LONG" in side:
        return "LONG"

    if "SHORT" in side:
        return "SHORT"

    return "UNKNOWN"


def _score_bucket(score: Optional[float]) -> str:
    s = _safe_float(score, None)

    if s is None:
        return "UNKNOWN"

    if s >= 80:
        return "STRONG"

    if s >= 65:
        return "GOOD"

    if s >= 45:
        return "MIXED"

    if s >= 25:
        return "WEAK"

    return "VERY_WEAK"


def _risk_rank(level: Any) -> int:
    lvl = _upper(level)

    if lvl in ["VERY_HIGH", "EXTREME"]:
        return 4

    if lvl == "HIGH":
        return 3

    if lvl in ["MEDIUM_HIGH", "MED-HIGH"]:
        return 2

    if lvl == "MEDIUM":
        return 1

    if lvl == "LOW":
        return 0

    return 0


def _risk_label_from_rank(rank: int) -> str:
    if rank >= 4:
        return "VERY_HIGH"

    if rank == 3:
        return "HIGH"

    if rank == 2:
        return "MEDIUM_HIGH"

    if rank == 1:
        return "MEDIUM"

    return "LOW"


def _append_unique(items: List[str], value: str) -> None:
    value = _text(value)
    if value and value not in items:
        items.append(value)


def _compact_dict(d: Optional[Dict[str, Any]], allowed_keys: List[str]) -> Dict[str, Any]:
    try:
        if not isinstance(d, dict):
            return {}

        return {k: d.get(k) for k in allowed_keys if k in d}
    except Exception:
        return {}


def _shorten(value: Any, limit: int = 220) -> str:
    text = _text(value)

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def build_decision_audit(
    scenario: Dict[str, Any],
    *,
    backend_status: str = "",
) -> Dict[str, Any]:
    """
    Build a response-only decision audit.

    This function does NOT:
    - change entry/SL/TP
    - change status
    - block or approve trades
    - write Firestore
    - call AI
    - fetch candles

    It only reads scenario fields and explains:
    - final execution state
    - which modules supported the setup
    - which modules warned against it
    - where conflicts exist
    - what user should watch
    """
    try:
        if not isinstance(scenario, dict):
            return {
                "enabled": True,
                "audit_status": "UNAVAILABLE",
                "summary": "Decision audit unavailable because scenario is missing.",
                "final_action": "UNKNOWN",
                "entry_allowed_now": False,
                "confidence_view": "UNKNOWN",
                "risk_view": "UNKNOWN",
                "supporting_factors": [],
                "risk_factors": [],
                "conflicts": [],
                "watch_levels": [],
                "module_roles": {},
                "module_outputs": {},
            }

        side = _normalize_side(
            scenario.get("trade_type")
            or scenario.get("side")
        )

        entry = (
            scenario.get("entry")
            or scenario.get("entry_zone")
            or ""
        )

        tp1 = None
        tp2 = None
        targets = scenario.get("targets") or []

        if isinstance(targets, list) and targets:
            tp1 = targets[0]
            if len(targets) > 1:
                tp2 = targets[1]

        tp1 = tp1 or scenario.get("tp1")
        tp2 = tp2 or scenario.get("tp2") or scenario.get("target")

        stop = (
            scenario.get("stop_loss")
            or scenario.get("invalidation")
            or ""
        )

        quality_score = _safe_int(scenario.get("quality_score"), 0)

        entry_conf = scenario.get("entry_confirmation") or {}
        direction_conf = scenario.get("direction_confidence") or {}
        tp_path = scenario.get("tp_path_quality") or {}
        imbalance = scenario.get("imbalance_risk") or {}
        liquidity = scenario.get("liquidity_risk") or {}
        crash = scenario.get("crash_regime") or {}
        execution = scenario.get("execution_readiness") or {}

        status = _upper(
            backend_status
            or execution.get("backend_tracking_status")
            or scenario.get("status")
        )

        execution_status = _upper(
            execution.get("execution_status")
            or scenario.get("execution_status")
            or status
        )

        execution_mode = _upper(
            execution.get("execution_mode")
            or scenario.get("execution_mode")
        )

        final_action = _text(
            execution.get("user_action")
            or scenario.get("execution_user_action")
            or scenario.get("generated_advice")
            or scenario.get("summary")
        )

        can_enter_now = _bool(execution.get("can_enter_now"))
        can_hold = _bool(execution.get("can_hold"))
        should_wait = _bool(execution.get("should_wait"))
        should_reduce_risk = _bool(execution.get("should_reduce_risk"))
        is_executable_signal = _bool(execution.get("is_executable_signal"))

        readiness_score = _safe_int(execution.get("readiness_score"), 0)
        execution_risk_level = _upper(execution.get("risk_level") or "UNKNOWN")
        execution_risk_mode = _upper(execution.get("risk_mode") or "UNKNOWN")

        entry_status = _upper(entry_conf.get("confirmation_status"))
        entry_score = _safe_int(entry_conf.get("confirmation_score"), 0)
        entry_touched = _bool(entry_conf.get("entry_zone_touched"))
        would_activate = _bool(entry_conf.get("would_activate_by_confirmation"))

        direction_score = _safe_int(direction_conf.get("direction_confidence"), 0)
        direction_mode = _upper(direction_conf.get("allowed_trade_mode"))
        trend_state = _upper(direction_conf.get("trend_state"))
        htf_alignment = _upper(direction_conf.get("htf_alignment"))
        primary_direction = _normalize_side(direction_conf.get("primary_direction"))
        candidate_side = _normalize_side(direction_conf.get("candidate_side") or side)

        tp_path_status = _upper(tp_path.get("path_status"))
        tp_path_score = _safe_int(tp_path.get("path_score"), 0)
        tp_sr_strength = _upper(tp_path.get("sr_strength"))
        tp_blocking_level = tp_path.get("blocking_level")
        tp_blocking_type = _upper(tp_path.get("blocking_type"))
        tp_required_action = _upper(tp_path.get("required_action"))
        tp_exit_if_rejected = _bool(tp_path.get("exit_if_rejected"))

        imbalance_level = _upper(imbalance.get("risk_level"))
        imbalance_score = _safe_int(imbalance.get("imbalance_score"), 0)
        imbalance_opposing = _bool(imbalance.get("opposing_imbalance_risk"))
        imbalance_adjust = _bool(imbalance.get("suggest_entry_adjustment"))
        imbalance_zone = imbalance.get("suggested_entry_zone")

        liquidity_level = _upper(liquidity.get("risk_level"))
        liquidity_score = _safe_int(liquidity.get("liquidity_score"), 0)
        adverse_liquidity = _bool(liquidity.get("adverse_liquidity_risk"))
        supportive_sweep = _bool(liquidity.get("supportive_sweep_detected"))
        adverse_sweep = _bool(liquidity.get("adverse_sweep_detected"))
        wait_for_sweep = _bool(liquidity.get("suggest_wait_for_sweep"))
        reduced_liquidity_risk = _bool(liquidity.get("suggest_reduced_risk"))
        sweep_level = liquidity.get("suggested_sweep_level")
        recent_sweep = liquidity.get("recent_sweep") or {}

        crash_regime = _upper(crash.get("regime"))
        crash_direction = _normalize_side(crash.get("direction"))
        crash_score = _safe_int(crash.get("score"), 0)
        crash_agreement = _upper(crash.get("agreement"))
        bearish_crash = _bool(crash.get("bearish_crash"))
        bullish_squeeze = _bool(crash.get("bullish_squeeze"))
        bearish_pressure = _bool(crash.get("bearish_pressure"))
        bullish_pressure = _bool(crash.get("bullish_pressure"))

        new_longs_risky = _bool(crash.get("new_longs_risky"))
        new_shorts_risky = _bool(crash.get("new_shorts_risky"))

        pending_longs_should_pause = _bool(crash.get("pending_longs_should_pause"))
        pending_shorts_should_pause = _bool(crash.get("pending_shorts_should_pause"))
        pending_longs_may_miss_limit = _bool(crash.get("pending_longs_may_miss_limit"))
        pending_shorts_may_miss_limit = _bool(crash.get("pending_shorts_may_miss_limit"))

        running_longs_need_protection = _bool(crash.get("running_longs_need_protection"))
        running_shorts_need_protection = _bool(crash.get("running_shorts_need_protection"))
        running_longs_favorable = _bool(crash.get("running_longs_favorable"))
        running_shorts_favorable = _bool(crash.get("running_shorts_favorable"))

        crash_watch_levels = crash.get("watch_levels") or {}

        supporting_factors: List[str] = []
        risk_factors: List[str] = []
        conflicts: List[str] = []
        watch_levels: List[Dict[str, Any]] = []
        action_notes: List[str] = []

        # --------------------------------------------------
        # Direction role
        # --------------------------------------------------
        if side in ["LONG", "SHORT"]:
            if candidate_side == side and primary_direction == side:
                _append_unique(
                    supporting_factors,
                    f"Direction module supports {side}; score {direction_score}/100."
                )
            elif primary_direction in ["LONG", "SHORT"] and primary_direction != side:
                _append_unique(
                    conflicts,
                    f"Direction conflict: setup side is {side}, but primary direction is {primary_direction}."
                )

        if direction_score >= 80:
            _append_unique(supporting_factors, "Direction confidence is strong.")
        elif 60 <= direction_score < 80:
            _append_unique(supporting_factors, "Direction confidence is acceptable but not elite.")
        elif direction_score > 0:
            _append_unique(risk_factors, "Direction confidence is weak/mixed.")

        if direction_mode in ["TP1_ONLY", "SCALP_ONLY", "REDUCED_RISK"]:
            _append_unique(
                risk_factors,
                f"Direction module recommends limited mode: {direction_mode}."
            )

        if trend_state in ["RANGE_OR_CHOP", "CHOP", "SIDEWAYS"]:
            _append_unique(
                risk_factors,
                "Market state is range/chop; full TP2 continuation is less reliable."
            )

        if htf_alignment in ["CONFLICT", "OPPOSED"]:
            _append_unique(
                conflicts,
                "Higher-timeframe alignment conflicts with the setup."
            )

        # --------------------------------------------------
        # Entry confirmation role
        # --------------------------------------------------
        if entry_status == "CONFIRMED":
            _append_unique(
                supporting_factors,
                f"Entry confirmation is valid on lower timeframe; score {entry_score}/100."
            )

            if not can_enter_now and "WAIT_ENTRY_TOUCH" in execution_status:
                _append_unique(
                    conflicts,
                    "Entry confirmation came from historical candles, but post-lock entry activation is not detected yet."
                )

        elif entry_status in ["WAITING", "UNKNOWN", ""]:
            _append_unique(
                risk_factors,
                "Entry confirmation is not ready yet."
            )

        elif entry_status == "INVALIDATED":
            _append_unique(
                conflicts,
                "Entry confirmation is invalidated."
            )

        elif entry_status == "ALREADY_ACTIVE":
            _append_unique(
                action_notes,
                "Entry confirmation is no longer actionable because trade is active/closed."
            )

        if entry_touched and not can_enter_now and "WAIT_ENTRY_TOUCH" in execution_status:
            _append_unique(
                conflicts,
                "Entry was touched in recent/historical confirmation data, but not after the current lock."
            )

        if would_activate and not can_enter_now and "WAIT_ENTRY_TOUCH" in execution_status:
            _append_unique(
                conflicts,
                "Lower-timeframe confirmation looks good, but execution is still waiting for fresh post-lock activation."
            )

        # --------------------------------------------------
        # TP path role
        # --------------------------------------------------
        if tp_path_status in ["CLEAN_TO_TP1"]:
            _append_unique(
                supporting_factors,
                "TP path to TP1 is clean; no major S/R wall detected."
            )

        elif tp_path_status in ["MINOR_SR_BEFORE_TP1"]:
            _append_unique(
                risk_factors,
                f"Minor {tp_blocking_type or 'S/R'} before TP1 at {tp_blocking_level}."
            )

        elif tp_path_status in ["WATCH_SR_REACTION", "BLOCKING_SR_BEFORE_TP1"]:
            _append_unique(
                risk_factors,
                f"Important {tp_blocking_type or 'S/R'} before TP1 at {tp_blocking_level}; reaction must be watched."
            )

        if tp_blocking_level:
            watch_levels.append({
                "type": "TP_PATH_BLOCKING_LEVEL",
                "price": tp_blocking_level,
                "side": tp_blocking_type,
                "importance": tp_sr_strength or "UNKNOWN",
                "what_to_watch": (
                    "If price rejects here before TP1, reduce risk or wait. "
                    "If candle body breaks and retests, continuation becomes healthier."
                ),
            })

        if tp_exit_if_rejected:
            _append_unique(
                action_notes,
                "TP path module says rejection at the blocking level should trigger defensive management."
            )

        # --------------------------------------------------
        # Imbalance role
        # --------------------------------------------------
        if imbalance_level in ["HIGH", "MEDIUM_HIGH"] or imbalance_adjust:
            _append_unique(
                risk_factors,
                f"Imbalance/FVG risk is high; score {imbalance_score}/100."
            )

            if imbalance_adjust:
                _append_unique(
                    conflicts,
                    "Original entry may be early because imbalance module suggests a better retest zone."
                )

        elif imbalance_level == "MEDIUM":
            _append_unique(
                risk_factors,
                f"Imbalance/FVG risk is medium; score {imbalance_score}/100."
            )

        elif imbalance_level == "LOW":
            _append_unique(
                supporting_factors,
                "No critical imbalance/FVG risk detected."
            )

        if isinstance(imbalance_zone, dict):
            watch_levels.append({
                "type": "IMBALANCE_RETEST_ZONE",
                "price": imbalance_zone.get("price"),
                "zone_bottom": imbalance_zone.get("zone_bottom"),
                "zone_top": imbalance_zone.get("zone_top"),
                "importance": imbalance_level or "UNKNOWN",
                "what_to_watch": imbalance_zone.get("note")
                or "Watch for retest/rejection from this FVG/imbalance zone.",
            })

        # --------------------------------------------------
        # Liquidity role
        # --------------------------------------------------
        if liquidity_level in ["HIGH", "MEDIUM_HIGH"] or wait_for_sweep:
            _append_unique(
                risk_factors,
                f"Liquidity sweep risk is high; score {liquidity_score}/100."
            )

            if wait_for_sweep:
                _append_unique(
                    conflicts,
                    "Liquidity module suggests waiting for sweep/reclaim instead of blind entry."
                )

        elif liquidity_level == "MEDIUM":
            _append_unique(
                risk_factors,
                f"Liquidity sweep risk is medium; score {liquidity_score}/100."
            )

        elif liquidity_level == "LOW":
            _append_unique(
                supporting_factors,
                "No critical liquidity sweep risk detected."
            )

        if supportive_sweep:
            _append_unique(
                supporting_factors,
                "Recent liquidity sweep/reclaim supports the setup direction."
            )

        if adverse_sweep:
            _append_unique(
                risk_factors,
                "Recent liquidity sweep is against the setup direction."
            )

        if isinstance(sweep_level, dict):
            watch_levels.append({
                "type": "LIQUIDITY_SWEEP_LEVEL",
                "price": sweep_level.get("price"),
                "adverse_side": sweep_level.get("adverse_side"),
                "between_entry_and_stop": sweep_level.get("between_entry_and_stop"),
                "importance": liquidity_level or "UNKNOWN",
                "what_to_watch": sweep_level.get("note")
                or "Watch for liquidity sweep and reclaim/rejection at this level.",
            })

        if isinstance(recent_sweep, dict) and recent_sweep.get("detected"):
            watch_levels.append({
                "type": "RECENT_SWEEP_LEVEL",
                "price": recent_sweep.get("swept_level"),
                "direction": recent_sweep.get("direction"),
                "sweep_type": recent_sweep.get("sweep_type"),
                "importance": "SUPPORTIVE" if supportive_sweep else "ADVERSE" if adverse_sweep else "NEUTRAL",
                "what_to_watch": (
                    "This level was recently swept. Watch whether price holds the reclaim/rejection."
                ),
            })

        # --------------------------------------------------
        # Crash regime / momentum pressure role
        # --------------------------------------------------
        if crash_regime and crash_regime not in ["UNKNOWN", "NEUTRAL", ""]:
            if crash_direction == side:
                _append_unique(
                    supporting_factors,
                    f"Crash/momentum regime supports {side}; regime {crash_regime}, score {crash_score}/100."
                )

                if side == "LONG" and pending_longs_may_miss_limit:
                    _append_unique(
                        action_notes,
                        "Bullish squeeze pressure may cause pullback LONG limit entries to be missed. Do not chase; watch breakout/retest."
                    )

                if side == "SHORT" and pending_shorts_may_miss_limit:
                    _append_unique(
                        action_notes,
                        "Bearish crash pressure may cause pullback SHORT limit entries to be missed. Do not chase; watch breakdown/retest."
                    )

            elif crash_direction in ["LONG", "SHORT"] and side in ["LONG", "SHORT"] and crash_direction != side:
                _append_unique(
                    risk_factors,
                    f"Crash/momentum regime is against this {side} setup; regime {crash_regime}, score {crash_score}/100."
                )

                if side == "LONG" and new_longs_risky:
                    _append_unique(
                        conflicts,
                        "Bearish crash pressure makes new/pending LONG entries risky until stabilization/reclaim."
                    )

                if side == "SHORT" and new_shorts_risky:
                    _append_unique(
                        conflicts,
                        "Bullish squeeze pressure makes new/pending SHORT entries risky until stabilization/rejection."
                    )

            elif crash_score >= 50:
                _append_unique(
                    risk_factors,
                    f"Market has high-momentum pressure; regime {crash_regime}, score {crash_score}/100."
                )

        if running_longs_need_protection:
            _append_unique(
                action_notes,
                "Crash regime warns active LONG needs defensive risk protection. This is warning-only, not automatic exit."
            )

        if running_shorts_need_protection:
            _append_unique(
                action_notes,
                "Crash regime warns active SHORT needs defensive risk protection. This is warning-only, not automatic exit."
            )

        if running_longs_favorable:
            _append_unique(
                supporting_factors,
                "Crash/squeeze pressure currently favors the active LONG direction."
            )

        if running_shorts_favorable:
            _append_unique(
                supporting_factors,
                "Crash pressure currently favors the active SHORT direction."
            )

        if crash_agreement == "CONFLICT":
            _append_unique(
                conflicts,
                "Crash regime signals conflict between selected timeframe and fast timeframe."
            )

        if isinstance(crash_watch_levels, dict):
            broken_support = crash_watch_levels.get("broken_support")
            broken_resistance = crash_watch_levels.get("broken_resistance")

            if broken_support is not None:
                watch_levels.append({
                    "type": "CRASH_BROKEN_SUPPORT",
                    "price": broken_support,
                    "importance": "HIGH" if bearish_crash else "MEDIUM",
                    "what_to_watch": (
                        "Watch whether price retests and rejects this broken support. "
                        "Holding below it supports bearish continuation; reclaim reduces crash pressure."
                    ),
                })

            if broken_resistance is not None:
                watch_levels.append({
                    "type": "SQUEEZE_BROKEN_RESISTANCE",
                    "price": broken_resistance,
                    "importance": "HIGH" if bullish_squeeze else "MEDIUM",
                    "what_to_watch": (
                        "Watch whether price retests and holds this broken resistance. "
                        "Holding above it supports bullish continuation; losing it reduces squeeze pressure."
                    ),
                })

        # --------------------------------------------------
        # Execution role
        # --------------------------------------------------
        if can_enter_now and is_executable_signal:
            final_state = "EXECUTABLE"
            _append_unique(
                action_notes,
                "Execution readiness allows entry now."
            )

        elif can_hold and "ACTIVE" in execution_status:
            final_state = "ACTIVE_MANAGEMENT"
            _append_unique(
                action_notes,
                "Trade is already active; manage according to active trade guidance."
            )

        elif should_wait or "WAIT" in execution_status:
            final_state = "WAIT"
            _append_unique(
                action_notes,
                "Execution readiness says wait; this is not executable now."
            )

        elif "CLOSED" in execution_status or status in ["WON", "LOST", "MISSED", "CANCELLED", "EXPIRED"]:
            final_state = "CLOSED"
            _append_unique(
                action_notes,
                "Trade is closed or no longer actionable."
            )

        else:
            final_state = "CONDITIONAL"
            _append_unique(
                action_notes,
                "Setup is conditional; use confirmation before acting."
            )

        if should_reduce_risk:
            _append_unique(
                risk_factors,
                "Execution readiness recommends reducing/protecting risk."
            )

        crash_risk_level = "LOW"
        if crash_score >= 75 and crash_direction in ["LONG", "SHORT"] and crash_direction != side:
            crash_risk_level = "HIGH"
        elif crash_score >= 55 and crash_direction in ["LONG", "SHORT"] and crash_direction != side:
            crash_risk_level = "MEDIUM"
        elif crash_agreement == "CONFLICT":
            crash_risk_level = "MEDIUM"

        if execution_risk_level and execution_risk_level != "UNKNOWN":
            max_risk_rank = max(
                _risk_rank(execution_risk_level),
                _risk_rank(imbalance_level),
                _risk_rank(liquidity_level),
                _risk_rank(crash_risk_level),
            )
        else:
            max_risk_rank = max(
                _risk_rank(imbalance_level),
                _risk_rank(liquidity_level),
                _risk_rank(crash_risk_level),
            )

        risk_view = _risk_label_from_rank(max_risk_rank)

        # --------------------------------------------------
        # Overall confidence view
        # --------------------------------------------------
        support_count = len(supporting_factors)
        risk_count = len(risk_factors)
        conflict_count = len(conflicts)

        confidence_score = 50

        confidence_score += min(quality_score, 100) * 0.12
        confidence_score += min(direction_score, 100) * 0.14
        confidence_score += min(entry_score, 100) * 0.10
        confidence_score += min(readiness_score, 100) * 0.10

        if tp_path_status == "CLEAN_TO_TP1":
            confidence_score += 8
        elif tp_path_status in ["WATCH_SR_REACTION", "BLOCKING_SR_BEFORE_TP1"]:
            confidence_score -= 8
        elif tp_path_status == "MINOR_SR_BEFORE_TP1":
            confidence_score -= 3

        if imbalance_level == "HIGH":
            confidence_score -= 10
        elif imbalance_level == "MEDIUM":
            confidence_score -= 5

        if liquidity_level == "HIGH":
            confidence_score -= 10
        elif liquidity_level == "MEDIUM":
            confidence_score -= 5

        if supportive_sweep:
            confidence_score += 8

        if adverse_sweep:
            confidence_score -= 8

        if crash_direction in ["LONG", "SHORT"] and side in ["LONG", "SHORT"]:
            if crash_direction == side:
                if crash_score >= 70:
                    confidence_score += 6
                elif crash_score >= 50:
                    confidence_score += 3
            else:
                if crash_score >= 70:
                    confidence_score -= 10
                elif crash_score >= 50:
                    confidence_score -= 5

        if crash_agreement == "CONFLICT":
            confidence_score -= 5

        if direction_mode in ["TP1_ONLY", "SCALP_ONLY", "REDUCED_RISK"]:
            confidence_score -= 6

        if conflict_count:
            confidence_score -= min(18, conflict_count * 6)

        if not can_enter_now and final_state == "WAIT":
            confidence_score -= 8

        confidence_score = max(0, min(100, int(round(confidence_score))))

        confidence_view = _score_bucket(confidence_score)

        # --------------------------------------------------
        # Main reason
        # --------------------------------------------------
        main_reason = ""

        if execution_status:
            main_reason = f"Final execution status is {execution_status}."

        if "IMBALANCE" in execution_status:
            main_reason = "Final decision is waiting because imbalance/FVG retest risk is high."

        elif "LIQUIDITY" in execution_status:
            main_reason = "Final decision is waiting because liquidity sweep/reclaim risk is high."

        elif "WAIT_ENTRY_TOUCH" in execution_status:
            main_reason = "Final decision is waiting because post-lock entry touch has not been detected."

        elif "WAIT_ENTRY_CONFIRMATION" in execution_status:
            main_reason = "Final decision is waiting because lower-timeframe entry confirmation is not ready."

        elif "WATCH_SR" in execution_status or execution_risk_mode == "WATCH_SR_REACTION":
            main_reason = "Final decision is conditional because TP path has S/R reaction risk."

        elif "ACTIVE_TP1" in execution_status:
            main_reason = "Trade has reached TP1; focus is profit protection and runner management."

        elif "ACTIVE" in execution_status:
            main_reason = "Trade is active; entry logic is no longer actionable."

        elif "CLOSED" in execution_status:
            main_reason = "Trade is closed or no longer actionable."

        # --------------------------------------------------
        # Human summary
        # --------------------------------------------------
        if final_state == "EXECUTABLE":
            summary = (
                f"{side} setup is executable now. "
                f"Confidence view: {confidence_view}. Risk view: {risk_view}."
            )

        elif final_state == "WAIT":
            summary = (
                f"{side} setup is not executable now. "
                f"{main_reason} Confidence view: {confidence_view}. Risk view: {risk_view}."
            )

        elif final_state == "ACTIVE_MANAGEMENT":
            summary = (
                f"{side} trade is in active management. "
                f"{main_reason} Confidence view: {confidence_view}. Risk view: {risk_view}."
            )

        elif final_state == "CLOSED":
            summary = (
                f"{side} setup is closed/no longer actionable. "
                f"{main_reason}"
            )

        else:
            summary = (
                f"{side} setup is conditional. "
                f"{main_reason} Confidence view: {confidence_view}. Risk view: {risk_view}."
            )

        return {
            "enabled": True,
            "audit_status": "OK",
            "side": side,
            "entry": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "backend_status": status,
            "execution_status": execution_status,
            "execution_mode": execution_mode,
            "final_state": final_state,
            "final_action": final_action,
            "entry_allowed_now": bool(can_enter_now and is_executable_signal),
            "can_hold": bool(can_hold),
            "should_wait": bool(should_wait or final_state == "WAIT"),
            "should_reduce_risk": bool(should_reduce_risk),
            "confidence_score": confidence_score,
            "confidence_view": confidence_view,
            "risk_view": risk_view,
            "main_reason": main_reason,
            "summary": summary,
            "supporting_factors": supporting_factors[:12],
            "risk_factors": risk_factors[:12],
            "conflicts": conflicts[:12],
            "action_notes": action_notes[:12],
            "watch_levels": watch_levels[:12],
            "module_roles": {
                "trend_direction": "Direction/regime only. Should not create entry by itself.",
                "setup_planner": "Builds backend-owned entry, SL, TP and selected setup.",
                "entry_confirmation": "Lower-timeframe timing/reaction only. Should not change backend prices.",
                "tp_path_quality": "Checks S/R obstacle between entry and TP1.",
                "imbalance_risk": "Checks FVG/imbalance fill risk and safer retest zones.",
                "liquidity_risk": "Checks sweep/stop-hunt risk around swing/equal highs/lows.",
                "crash_regime": "Checks market-wide crash/squeeze momentum pressure. Warning-only; should not change entry/SL/TP.",
                "execution_readiness": "Combines module outputs into user action: enter/wait/hold/protect.",
            },
            "module_outputs": {
                "quality_score": quality_score,
                "direction": {
                    "score": direction_score,
                    "bucket": _score_bucket(direction_score),
                    "primary_direction": primary_direction,
                    "candidate_side": candidate_side,
                    "trend_state": trend_state,
                    "htf_alignment": htf_alignment,
                    "allowed_trade_mode": direction_mode,
                },
                "entry_confirmation": {
                    "status": entry_status,
                    "score": entry_score,
                    "entry_zone_touched": entry_touched,
                    "would_activate_by_confirmation": would_activate,
                },
                "tp_path": {
                    "status": tp_path_status,
                    "score": tp_path_score,
                    "sr_strength": tp_sr_strength,
                    "blocking_level": tp_blocking_level,
                    "blocking_type": tp_blocking_type,
                    "required_action": tp_required_action,
                    "exit_if_rejected": tp_exit_if_rejected,
                },
                "imbalance": {
                    "risk_level": imbalance_level,
                    "score": imbalance_score,
                    "opposing_imbalance": imbalance_opposing,
                    "suggest_entry_adjustment": imbalance_adjust,
                    "suggested_zone": imbalance_zone,
                },
                "liquidity": {
                    "risk_level": liquidity_level,
                    "score": liquidity_score,
                    "adverse_liquidity": adverse_liquidity,
                    "supportive_sweep": supportive_sweep,
                    "adverse_sweep": adverse_sweep,
                    "suggest_wait_for_sweep": wait_for_sweep,
                    "suggest_reduced_risk": reduced_liquidity_risk,
                    "suggested_sweep_level": sweep_level,
                    "recent_sweep": _compact_dict(
                        recent_sweep,
                        [
                            "detected",
                            "direction",
                            "sweep_type",
                            "swept_level",
                            "sweep_candle_age",
                            "wick_size_atr",
                        ],
                    ),
                },
                "crash_regime": {
                    "regime": crash_regime,
                    "direction": crash_direction,
                    "score": crash_score,
                    "agreement": crash_agreement,
                    "bearish_crash": bearish_crash,
                    "bullish_squeeze": bullish_squeeze,
                    "bearish_pressure": bearish_pressure,
                    "bullish_pressure": bullish_pressure,
                    "new_longs_risky": new_longs_risky,
                    "new_shorts_risky": new_shorts_risky,
                    "pending_longs_should_pause": pending_longs_should_pause,
                    "pending_shorts_should_pause": pending_shorts_should_pause,
                    "pending_longs_may_miss_limit": pending_longs_may_miss_limit,
                    "pending_shorts_may_miss_limit": pending_shorts_may_miss_limit,
                    "running_longs_need_protection": running_longs_need_protection,
                    "running_shorts_need_protection": running_shorts_need_protection,
                    "running_longs_favorable": running_longs_favorable,
                    "running_shorts_favorable": running_shorts_favorable,
                    "watch_levels": crash_watch_levels,
                },
                "execution": {
                    "status": execution_status,
                    "mode": execution_mode,
                    "risk_mode": execution_risk_mode,
                    "risk_level": execution_risk_level,
                    "readiness_score": readiness_score,
                    "can_enter_now": can_enter_now,
                    "can_hold": can_hold,
                    "should_wait": should_wait,
                    "should_reduce_risk": should_reduce_risk,
                    "is_executable_signal": is_executable_signal,
                },
            },
            "raw_notes": {
                "entry_reason": _shorten(entry_conf.get("reason")),
                "direction_reason": _shorten(direction_conf.get("reason")),
                "tp_path_reason": _shorten(tp_path.get("reason")),
                "imbalance_reason": _shorten(imbalance.get("reason")),
                "liquidity_reason": _shorten(liquidity.get("reason")),
                "crash_reason": _shorten(crash.get("reason")),
                "crash_advice": _shorten(crash.get("advice")),
                "execution_reason": _shorten(execution.get("reason")),
            },
        }

    except Exception as e:
        return {
            "enabled": True,
            "audit_status": "ERROR",
            "summary": "Decision audit failed.",
            "final_action": "UNKNOWN",
            "entry_allowed_now": False,
            "confidence_view": "UNKNOWN",
            "risk_view": "UNKNOWN",
            "supporting_factors": [],
            "risk_factors": [],
            "conflicts": [],
            "watch_levels": [],
            "module_roles": {},
            "module_outputs": {},
            "error": str(e),
        }


def add_decision_audit_to_scenario(
    scenario: Dict[str, Any],
    *,
    backend_status: str = "",
) -> Dict[str, Any]:
    """
    Attach decision_audit to one scenario.
    """
    try:
        if not isinstance(scenario, dict):
            return scenario

        scenario["decision_audit"] = build_decision_audit(
            scenario,
            backend_status=backend_status,
        )

    except Exception as e:
        try:
            scenario["decision_audit"] = {
                "enabled": True,
                "audit_status": "ERROR",
                "summary": "Decision audit attach failed.",
                "error": str(e),
            }
        except Exception:
            pass

    return scenario


def add_decision_audit_to_ai_data(
    ai_data: Dict[str, Any],
    *,
    backend_status: str = "",
) -> Dict[str, Any]:
    """
    Attach decision_audit to first trade_scenario and selected_scenario.
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

        scenario = add_decision_audit_to_scenario(
            scenario,
            backend_status=backend_status,
        )

        scenarios[0] = scenario
        ai_data["trade_scenarios"] = scenarios

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            selected_side = _normalize_side(
                selected.get("trade_type")
                or selected.get("side")
            )

            scenario_side = _normalize_side(
                scenario.get("trade_type")
                or scenario.get("side")
            )

            selected_entry = _safe_float(
                selected.get("entry")
                or selected.get("entry_zone"),
                None,
            )

            scenario_entry = _safe_float(
                scenario.get("entry")
                or scenario.get("entry_zone"),
                None,
            )

            entry_matches = (
                selected_entry is None
                or scenario_entry is None
                or abs(selected_entry - scenario_entry) <= max(abs(scenario_entry) * 0.000001, 1e-12)
            )

            if selected_side == scenario_side and entry_matches:
                ai_data["selected_scenario"] = scenario

    except Exception as e:
        print(f"[DECISION_AUDIT] attach skipped/error: {e}")

    return ai_data