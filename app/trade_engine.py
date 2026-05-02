import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# PRICE / TEXT PARSING HELPERS
# ============================================================

def extract_price(value: Any) -> Optional[float]:
    """
    Extract the first valid price from a value.

    Examples:
    - "65000" -> 65000.0
    - "65,000 - 66,000" -> 65000.0
    - "Entry around 0.5678" -> 0.5678

    Important:
    This does NOT merge ranges into wrong numbers.
    """
    try:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).replace(",", "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)

        if match:
            return float(match.group(0))

    except Exception:
        return None

    return None


def parse_entry_zone(entry_zone: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse entry zone into low/high.

    Examples:
    - 65000 -> (65000, 65000)
    - "65000 - 65500" -> (65000, 65500)
    - "Entry Zone: 0.5678 - 0.5720" -> (0.5678, 0.5720)
    """
    if entry_zone is None:
        return None, None

    if isinstance(entry_zone, (int, float)):
        val = float(entry_zone)
        return val, val

    text = str(entry_zone).replace(",", "").strip()

    # Handles ranges like "2360 - 2380"
    dash_parts = [p.strip() for p in re.split(r"\s*-\s*", text) if p.strip()]
    if len(dash_parts) >= 2:
        low = extract_price(dash_parts[0])
        high = extract_price(dash_parts[1])

        if low is not None and high is not None:
            return (low, high) if low <= high else (high, low)

    # Fallback single value
    val = extract_price(text)
    if val is not None:
        return val, val

    return None, None


def _round_price(value: Optional[float]) -> Optional[float]:
    """
    Dynamic rounding for large and small coins.
    Keeps low-price coins like TRX/APE/DOGE more precise.
    """
    if value is None:
        return None

    try:
        value = float(value)

        if abs(value) >= 100:
            return round(value, 2)
        if abs(value) >= 1:
            return round(value, 4)
        return round(value, 8)

    except Exception:
        return None


# ============================================================
# MARKET DATA HELPERS
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate ATR using EWM smoothing.

    This is closer to the market_context.py ATR style than simple rolling mean.
    Expected columns: High, Low, Close
    """
    try:
        if df is None or df.empty or len(df) < period + 2:
            return None

        required = {"High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            return None

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)

        prev_close = close.shift(1)

        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]

        if pd.isna(atr):
            return None

        return float(atr)

    except Exception:
        return None


def get_current_price(df: pd.DataFrame) -> Optional[float]:
    try:
        if df is None or df.empty:
            return None

        return float(df.iloc[-1]["Close"])

    except Exception:
        return None


# ============================================================
# TRADE TYPE / ENTRY / SL / TP
# ============================================================

def normalize_trade_type(trade_type: Any) -> str:
    text = str(trade_type or "").upper()

    if "SHORT" in text or "SELL" in text:
        return "SHORT"
    if "LONG" in text or "BUY" in text:
        return "LONG"
    if "WAIT" in text:
        return "WAIT"
    if "NEUTRAL" in text:
        return "NEUTRAL"

    return text or "UNKNOWN"


def choose_entry_from_zone(
    trade_type: str,
    zone_low: Optional[float],
    zone_high: Optional[float],
    current_price: Optional[float],
) -> Optional[float]:
    """
    Choose a practical entry from AI-provided zone.

    LONG:
    - Prefer lower side of the zone.
    SHORT:
    - Prefer upper side of the zone.

    If current price is already inside the zone:
    - use current price, but status logic can still classify READY / NO CHASE later.
    """
    if zone_low is None or zone_high is None:
        return None

    if zone_low == zone_high:
        return _round_price(zone_low)

    zone_low = float(zone_low)
    zone_high = float(zone_high)

    if trade_type == "LONG":
        # Better execution closer to support side
        base_entry = zone_low + (zone_high - zone_low) * 0.35

    elif trade_type == "SHORT":
        # Better execution closer to resistance side
        base_entry = zone_high - (zone_high - zone_low) * 0.35

    else:
        base_entry = (zone_low + zone_high) / 2.0

    if current_price is not None and zone_low <= current_price <= zone_high:
        return _round_price(current_price)

    return _round_price(base_entry)


def build_stop_loss(
    trade_type: str,
    invalidation: Optional[float],
    atr_value: Optional[float],
    current_price: Optional[float],
) -> Optional[float]:
    """
    Build stop loss beyond structural invalidation.

    Important:
    - Invalidation is still primary.
    - ATR is only a small buffer beyond the structure.
    - Do NOT mechanically set SL from ATR alone.

    Buffer default:
    - 0.25 ATR beyond invalidation.
    """
    if invalidation is None:
        return None

    invalidation = float(invalidation)

    if atr_value is None or atr_value <= 0:
        atr_value = 0.0

    # Practical wick/noise buffer.
    # 0.25 ATR is safer than tiny 0.10/0.15 ATR,
    # but not too wide like full 1.5x/2x ATR.
    buffer_mult = 0.25
    buffer_val = atr_value * buffer_mult

    if trade_type == "LONG":
        return _round_price(invalidation - buffer_val)

    if trade_type == "SHORT":
        return _round_price(invalidation + buffer_val)

    return _round_price(invalidation)


def build_tp_levels(
    trade_type: str,
    entry: Optional[float],
    stop_loss: Optional[float],
    ai_targets: Optional[List[Any]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Build TP1/TP2.

    Hard rule:
    - TP1 is always exact 1.5R from entry.
    - TP2 defaults to 3R.
    - AI targets may only help align TP2, never reduce TP1 below 1.5R.

    LONG:
    - TP1 > entry
    - TP2 > TP1

    SHORT:
    - TP1 < entry
    - TP2 < TP1
    """
    if entry is None or stop_loss is None:
        return None, None, None

    entry = float(entry)
    stop_loss = float(stop_loss)

    risk = abs(entry - stop_loss)

    if risk <= 0:
        return None, None, None

    if trade_type == "LONG":
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 3.0

    elif trade_type == "SHORT":
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 3.0

    else:
        return None, None, None

    # Parse AI targets as optional structural guidance for TP2 only.
    parsed_ai_targets: List[float] = []
    if ai_targets:
        for target in ai_targets:
            val = extract_price(target)
            if val is not None:
                parsed_ai_targets.append(float(val))

    if parsed_ai_targets:
        if trade_type == "LONG":
            # Only use AI targets that are beyond TP1.
            valid_targets = [t for t in parsed_ai_targets if t > tp1]

            if valid_targets:
                # Keep TP2 within system default 3R if AI gives a nearer structure target.
                # If AI gives very far target, system 3R remains safer.
                structure_tp2 = max(valid_targets)
                tp2 = min(tp2, structure_tp2)

            # Safety: TP2 must remain above TP1.
            if tp2 <= tp1:
                tp2 = entry + risk * 3.0

        elif trade_type == "SHORT":
            # Only use AI targets that are beyond TP1 in short direction.
            valid_targets = [t for t in parsed_ai_targets if t < tp1]

            if valid_targets:
                structure_tp2 = min(valid_targets)
                tp2 = max(tp2, structure_tp2)

            # Safety: TP2 must remain below TP1.
            if tp2 >= tp1:
                tp2 = entry - risk * 3.0

    rr_tp1 = abs(tp1 - entry) / risk if risk else None

    return (
        _round_price(tp1),
        _round_price(tp2),
        round(rr_tp1, 4) if rr_tp1 is not None else None,
    )


# ============================================================
# STATUS / SCORE
# ============================================================

def determine_trade_status(
    trade_type: str,
    current_price: Optional[float],
    zone_low: Optional[float],
    zone_high: Optional[float],
    entry: Optional[float],
) -> str:
    """
    Decide whether the setup is ready, pending, or no-chase.
    """
    if trade_type in ["WAIT", "NEUTRAL", "UNKNOWN"]:
        return "WAIT"

    if current_price is None:
        return "LIMIT ORDER"

    if zone_low is not None and zone_high is not None:
        zone_low = float(zone_low)
        zone_high = float(zone_high)

        if zone_low <= current_price <= zone_high:
            return "READY"

        if trade_type == "LONG" and current_price < zone_low:
            return "LIMIT ORDER"

        if trade_type == "SHORT" and current_price > zone_high:
            return "LIMIT ORDER"

        return "NO CHASE"

    if entry is not None:
        return "READY"

    return "WAIT"


def score_scenario(
    scenario: Dict[str, Any],
    macro_trend: str,
    micro_status: str,
    current_price: Optional[float],
    entry: Optional[float],
    stop_loss: Optional[float],
    tp1: Optional[float],
    trade_status: str,
) -> int:
    """
    Lightweight score for ranking AI scenarios.
    This should NOT replace Elliott/AI decision.
    It is only a post-processing quality hint.
    """
    score = 0

    trade_type = normalize_trade_type(scenario.get("trade_type"))
    name = str(scenario.get("name", "")).lower()
    prob = str(scenario.get("probability", "")).lower()

    macro_trend = str(macro_trend or "").lower()
    micro_status = str(micro_status or "").lower()

    # Macro trend alignment
    if "bear" in macro_trend and trade_type == "SHORT":
        score += 30
    elif "bull" in macro_trend and trade_type == "LONG":
        score += 30
    elif "bear" in macro_trend and trade_type == "LONG":
        score -= 15
    elif "bull" in macro_trend and trade_type == "SHORT":
        score -= 15

    # Micro alignment.
    # Keep this light because Elliott status text may vary.
    if any(k in micro_status for k in ["impulse", "up", "bull"]):
        if trade_type == "LONG":
            score += 12

    if any(k in micro_status for k in ["correction", "down", "bear"]):
        if trade_type == "SHORT":
            score += 12

    # Probability hints from AI
    if "high" in prob:
        score += 10
    elif "low" in prob:
        score -= 10

    # Alternative scenario penalty
    if "alternative" in name:
        score -= 8

    # Trade status
    if trade_status == "READY":
        score += 15
    elif trade_status == "LIMIT ORDER":
        score += 8
    elif trade_status == "NO CHASE":
        score -= 20
    elif trade_status == "WAIT":
        score -= 25

    # RR quality based on TP1
    if entry is not None and stop_loss is not None and tp1 is not None:
        risk = abs(entry - stop_loss)
        reward = abs(tp1 - entry)

        rr = reward / risk if risk else 0.0

        if rr >= 1.5:
            score += 15
        elif rr >= 1.2:
            score += 8
        else:
            score -= 15

    # Distance from current price
    if current_price is not None and entry is not None and current_price > 0:
        dist_pct = abs(entry - current_price) / current_price * 100.0

        if dist_pct <= 0.5:
            score += 10
        elif dist_pct <= 1.5:
            score += 5
        elif dist_pct > 3.0:
            score -= 8

    return int(score)


# ============================================================
# SCENARIO ENRICHMENT
# ============================================================

def enrich_scenario(
    scenario: Dict[str, Any],
    macro_trend: str,
    micro_status: str,
    df_micro: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Enrich one AI scenario with system-side math:
    - normalized trade type
    - parsed entry zone
    - selected entry
    - ATR buffered SL
    - TP1 exact 1.5R
    - TP2 system/structure aligned
    - trade status
    - score
    """
    enriched = dict(scenario)

    trade_type = normalize_trade_type(enriched.get("trade_type"))
    current_price = get_current_price(df_micro)
    atr_value = calc_atr(df_micro)

    zone_low, zone_high = parse_entry_zone(
        enriched.get("entry_zone") or enriched.get("entry")
    )

    invalidation = extract_price(
        enriched.get("invalidation") or enriched.get("stop_loss")
    )

    final_entry = choose_entry_from_zone(
        trade_type=trade_type,
        zone_low=zone_low,
        zone_high=zone_high,
        current_price=current_price,
    )

    final_sl = build_stop_loss(
        trade_type=trade_type,
        invalidation=invalidation,
        atr_value=atr_value,
        current_price=current_price,
    )

    ai_targets = enriched.get("targets", [])
    if not ai_targets and enriched.get("target") is not None:
        ai_targets = [enriched.get("target")]

    tp1, tp2, rr_tp1 = build_tp_levels(
        trade_type=trade_type,
        entry=final_entry,
        stop_loss=final_sl,
        ai_targets=ai_targets,
    )

    trade_status = determine_trade_status(
        trade_type=trade_type,
        current_price=current_price,
        zone_low=zone_low,
        zone_high=zone_high,
        entry=final_entry,
    )

    score = score_scenario(
        scenario=enriched,
        macro_trend=macro_trend,
        micro_status=micro_status,
        current_price=current_price,
        entry=final_entry,
        stop_loss=final_sl,
        tp1=tp1,
        trade_status=trade_status,
    )

    enriched["trade_type"] = trade_type

    enriched["entry_zone_low"] = _round_price(zone_low)
    enriched["entry_zone_high"] = _round_price(zone_high)
    enriched["entry"] = final_entry

    enriched["stop_loss"] = final_sl
    enriched["invalidation"] = final_sl

    enriched["tp1"] = tp1
    enriched["tp2"] = tp2

    if tp1 is not None and tp2 is not None:
        enriched["targets"] = [tp1, tp2]
        enriched["target"] = tp2
    else:
        enriched["targets"] = enriched.get("targets", [])

    enriched["rr_tp1"] = rr_tp1
    enriched["atr"] = _round_price(atr_value)
    enriched["current_price"] = _round_price(current_price)

    enriched["trade_status"] = trade_status
    enriched["score"] = score

    return enriched


def enrich_ai_prediction(ai_data: Dict[str, Any], df_micro: pd.DataFrame) -> Dict[str, Any]:
    """
    Enrich all AI scenarios and select best scenario by score.

    Important:
    - This runs AFTER AI response.
    - It does not create a trade by itself.
    - It only cleans/ranks/calculates risk levels.
    """
    if not ai_data:
        return ai_data

    macro_trend = str(ai_data.get("macro_analysis", {}).get("trend", ""))
    micro_status = str(ai_data.get("micro_analysis", {}).get("wave_count_status", ""))

    scenarios = ai_data.get("trade_scenarios", [])
    enriched_scenarios: List[Dict[str, Any]] = []

    for scenario in scenarios:
        try:
            enriched_scenarios.append(
                enrich_scenario(
                    scenario=scenario,
                    macro_trend=macro_trend,
                    micro_status=micro_status,
                    df_micro=df_micro,
                )
            )
        except Exception:
            enriched_scenarios.append(scenario)

    try:
        enriched_scenarios.sort(
            key=lambda x: x.get("score", -9999),
            reverse=True,
        )
    except Exception:
        pass

    ai_data["trade_scenarios"] = enriched_scenarios

    if enriched_scenarios:
        ai_data["selected_scenario"] = enriched_scenarios[0]

    return ai_data


def should_take_trade(scenario: Dict[str, Any], min_score: int = 30) -> bool:
    """
    Final lightweight permission check.

    This should be used only as a safety check.
    It should not override a WAIT/NO CHASE decision.
    """
    trade_type = normalize_trade_type(scenario.get("trade_type"))

    if trade_type in ["WAIT", "NEUTRAL", "UNKNOWN"]:
        return False

    status = str(scenario.get("trade_status", ""))

    if status == "NO CHASE":
        return False

    if status == "WAIT":
        return False

    try:
        score = int(scenario.get("score", 0))
    except Exception:
        score = 0

    return score >= min_score