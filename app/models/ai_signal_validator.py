# app/models/ai_signal_validator.py

"""
ElliotIQ AI Signal Validator
============================

Deterministically validates an AI-generated trade draft before it can be
published or saved.

Validation includes:

- Decision / side consistency
- Entry and stop-loss geometry
- Exact TP1 calculation at 1.5R
- TP2 direction and realism
- ATR-based stop distance
- Entry chase protection
- Market-packet evidence paths
- Entry / SL / TP2 anchor paths
- Historical Analog Engine trade-path validation
- Frontend status and advice formatting
- Safe NO_TRADE fallback generation

This module does not call Gemini and does not fetch market data.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from .historical_pattern_engine import (
    validate_trade_with_historical_analogs,
)


VALIDATOR_VERSION = "1.2.0"
TP1_R_MULTIPLE = 1.5


ALLOWED_DECISIONS = {
    "LONG",
    "SHORT",
    "WAIT",
    "NO_TRADE",
}


ALLOWED_SIDES = {
    "LONG",
    "SHORT",
    "NONE",
}


ALLOWED_EXECUTION_STATES = {
    "ENTER_NOW",
    "PLACE_LIMIT",
    "WAIT_FOR_ENTRY",
    "WAIT_FOR_RETEST",
    "WAIT_FOR_RECLAIM",
    "WAIT_FOR_BREAKOUT",
    "WAIT_FOR_FAILED_RECLAIM",
    "NO_TRADE",
}


ALLOWED_ORDER_TYPES = {
    "MARKET",
    "LIMIT",
    "STOP",
    "NONE",
}

WAIT_EXECUTION_STATES = {
    "WAIT_FOR_ENTRY",
    "WAIT_FOR_RETEST",
    "WAIT_FOR_RECLAIM",
    "WAIT_FOR_BREAKOUT",
    "WAIT_FOR_FAILED_RECLAIM",
}


DEFAULT_VALIDATION_CONFIG: Dict[str, Any] = {
    # TP1 is always backend-owned at exactly 1.5R.
    "tp1_r_multiple": TP1_R_MULTIPLE,

    # Stop-loss hard limits.
    # 1.30 ATR is the minimum floor, not a fixed SL distance.
    "hard_min_stop_atr": 1.30,
    "hard_max_stop_atr_15m_30m": 2.80,
    "hard_max_stop_atr_1h": 3.50,
    "hard_max_stop_atr_4h": 4.60,
    "hard_max_stop_atr_1d": 5.80,

    # TP2 hard policy.
    # Below 2R is rejected. Above the effective maximum is capped.
    "hard_min_tp2_r": 2.00,
    "hard_max_tp2_r": 5.00,

    # Timeframe-specific final TP2 distance caps.
    "hard_max_tp2_atr_15m_30m": 5.00,
    "hard_max_tp2_atr_1h": 7.00,
    "hard_max_tp2_atr_4h": 8.00,
    "hard_max_tp2_atr_1d": 10.00,

    # No-chase protection.
    "max_entry_chase_fraction": 0.70,

    # Evidence contract.
    "minimum_evidence_paths": 4,

    # Anchor validation.
    "require_numeric_anchor_support": True,
    "anchor_tolerance_atr": 0.75,
    "anchor_tolerance_fraction": 0.005,
    "max_anchor_numeric_values": 3,

    # Order-price validation.
    "market_entry_tolerance_atr": 0.35,
    "market_entry_tolerance_fraction": 0.003,
    "pending_order_tolerance_atr": 0.10,
    "pending_order_tolerance_fraction": 0.001,

    # Historical analog thresholds.
    "analog_min_sample_for_hard_reject": 10,
    "analog_min_confidence_for_hard_reject": 60.0,
    "analog_hard_min_tp1_probability": 0.25,
    "analog_hard_min_sl_probability": 0.55,
    "analog_warning_min_tp1_probability": 0.40,
    "analog_warning_min_tp2_probability": 0.20,
}


ARCHITECTURE_LOCKED_CONFIG_KEYS = {
    "tp1_r_multiple",
    "hard_min_stop_atr",
    "hard_max_stop_atr_15m_30m",
    "hard_max_stop_atr_1h",
    "hard_max_stop_atr_4h",
    "hard_max_stop_atr_1d",
    "hard_min_tp2_r",
    "hard_max_tp2_r",
    "hard_max_tp2_atr_15m_30m",
    "hard_max_tp2_atr_1h",
    "hard_max_tp2_atr_4h",
    "hard_max_tp2_atr_1d",
}

# ==========================================================
# BASIC HELPERS
# ==========================================================

def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            cleaned = (
                value
                .replace(",", "")
                .replace("$", "")
                .replace("%", "")
                .strip()
            )

            if not cleaned:
                return default

            value = cleaned

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except Exception:
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    number = _safe_float(value)

    if number is None:
        return default

    try:
        return int(number)
    except Exception:
        return default


def _upper_text(
    value: Any,
    default: str = "",
) -> str:
    text = str(
        value or default
    ).strip().upper()

    return text.replace(" ", "_")


def _clean_text(
    value: Any,
    max_length: int = 500,
) -> str:
    text = str(
        value or ""
    ).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text[:max_length].strip()


def _merge_config(
    custom: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(
        DEFAULT_VALIDATION_CONFIG
    )

    if isinstance(custom, Mapping):
        for key, value in custom.items():
            key_text = str(key)

            if (
                key_text not in merged
                or key_text
                in ARCHITECTURE_LOCKED_CONFIG_KEYS
            ):
                continue

            default_value = merged[key_text]

            if isinstance(default_value, bool):
                if isinstance(value, bool):
                    merged[key_text] = value

                continue

            if isinstance(value, bool):
                continue

            number = _safe_float(
                value
            )

            if number is None:
                continue

            if (
                isinstance(default_value, int)
                and not isinstance(
                    default_value,
                    bool,
                )
            ):
                merged[key_text] = int(
                    number
                )

            else:
                merged[key_text] = float(
                    number
                )

    # Architecture-owned risk limits cannot be weakened
    # through caller-provided validation configuration.
    for key in ARCHITECTURE_LOCKED_CONFIG_KEYS:
        if key in DEFAULT_VALIDATION_CONFIG:
            merged[key] = (
                DEFAULT_VALIDATION_CONFIG[key]
            )

    merged["tp1_r_multiple"] = (
        TP1_R_MULTIPLE
    )

    return merged


def _append_unique(
    collection: list[str],
    message: str,
) -> None:
    cleaned = _clean_text(
        message,
        600,
    )

    if (
        cleaned
        and cleaned not in collection
    ):
        collection.append(cleaned)


def _round_price(
    value: Optional[float],
) -> Optional[float]:
    if value is None:
        return None

    absolute = abs(value)

    if absolute >= 1000:
        digits = 2
    elif absolute >= 1:
        digits = 6
    else:
        digits = 10

    return round(
        value,
        digits,
    )


# ==========================================================
# MARKET PACKET PATH RESOLUTION
# ==========================================================

def _normalize_packet_path(
    path: Any,
) -> str:
    text = str(
        path or ""
    ).strip()

    prefixes = (
        "MARKET_PACKET_JSON.",
        "market_packet.",
        "packet.",
        "$.",
    )

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]

    return text.strip(".")


def _resolve_packet_path(
    packet: Any,
    path: Any,
) -> tuple[bool, Any]:
    """
    Resolves paths such as:

    timeframes.main.summary.atr14

    timeframes.main.recent_closed_candles.12.low
    """

    normalized = _normalize_packet_path(
        path
    )

    if not normalized:
        return False, None

    current = packet

    for part in normalized.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return False, None

            current = current[part]
            continue

        if isinstance(
            current,
            (
                list,
                tuple,
            ),
        ):
            try:
                index = int(part)
            except Exception:
                return False, None

            if (
                index < 0
                or index >= len(current)
            ):
                return False, None

            current = current[index]
            continue

        return False, None

    return True, current

def _path_segment_signature(
    value: Any,
) -> str:
    """
    Normalize case and separators in one path segment.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value or "").lower(),
    )


def _canonicalize_packet_path(
    packet: Any,
    path: Any,
) -> str:
    """
    Recover an exact packet path only when every segment has
    one unique case/separator-insensitive match.

    Ambiguous or invented paths remain unchanged so the normal
    deterministic validator can reject them.
    """

    normalized = _normalize_packet_path(
        path
    )

    if not normalized:
        return ""

    current = packet
    canonical_parts: list[str] = []

    for part in normalized.split("."):
        if isinstance(current, Mapping):
            if part in current:
                actual_key = part

            else:
                signature = (
                    _path_segment_signature(
                        part
                    )
                )

                matches = [
                    key
                    for key in current.keys()
                    if (
                        signature
                        and _path_segment_signature(
                            key
                        ) == signature
                    )
                ]

                # Repair only one unique real packet match.
                if len(matches) != 1:
                    return normalized

                actual_key = matches[0]

            canonical_parts.append(
                str(actual_key)
            )

            current = current[
                actual_key
            ]

            continue

        if isinstance(
            current,
            (
                list,
                tuple,
            ),
        ):
            try:
                index = int(part)
            except Exception:
                return normalized

            if (
                index < 0
                or index >= len(current)
            ):
                return normalized

            canonical_parts.append(
                str(index)
            )

            current = current[index]
            continue

        return normalized

    return ".".join(
        canonical_parts
    )

def _validate_catalog_reference(
    packet: Mapping[str, Any],
    path: Any,
    *,
    expected_types: Optional[
        Sequence[str]
    ] = None,
) -> Dict[str, Any]:
    """
    Validate an exact evidence_catalog
    price reference.
    """

    normalized = _normalize_packet_path(
        path
    )

    result: Dict[str, Any] = {
        "is_catalog_reference": False,
        "valid": True,
        "catalog_id": None,
        "type": None,
        "timeframe": None,
        "reason": "",
    }

    if not normalized.startswith(
        "evidence_catalog."
    ):
        return result

    result["is_catalog_reference"] = True
    result["valid"] = False

    parts = normalized.split(".")

    if (
        len(parts) != 4
        or parts[0] != "evidence_catalog"
        or parts[1] != "items"
        or parts[3] != "price"
    ):
        result["reason"] = (
            "Catalog reference must use the "
            "exact stored item price path."
        )
        return result

    catalog_id = parts[2]
    result["catalog_id"] = catalog_id

    catalog = packet.get(
        "evidence_catalog",
        {},
    )

    items = (
        catalog.get("items", {})
        if isinstance(catalog, Mapping)
        else {}
    )

    item = (
        items.get(catalog_id)
        if isinstance(items, Mapping)
        else None
    )

    if not isinstance(item, Mapping):
        result["reason"] = (
            "Catalog evidence ID does not exist."
        )
        return result

    stored_path = _normalize_packet_path(
        item.get("path")
    )

    if stored_path != normalized:
        result["reason"] = (
            "Catalog item path does not match "
            "the cited path."
        )
        return result

    price = _safe_float(
        item.get("price")
    )

    if price is None or price <= 0:
        result["reason"] = (
            "Catalog item price is invalid."
        )
        return result

    evidence_type = str(
        item.get("type") or ""
    ).upper().strip()

    allowed_types = {
        "MARKET_PRICE",
        "CLOSED_PRICE",
        "SUPPORT",
        "RESISTANCE",
        "SWING_HIGH",
        "SWING_LOW",
    }

    result["type"] = (
        evidence_type or None
    )

    if evidence_type not in allowed_types:
        result["reason"] = (
            "Catalog item has an unsupported "
            "evidence type."
        )
        return result

    if expected_types is not None:
        normalized_expected = {
            str(value or "").upper().strip()
            for value in expected_types
        }

        if evidence_type not in (
            normalized_expected
        ):
            result["reason"] = (
                "Catalog evidence type is not "
                "valid for this anchor."
            )
            return result

    timeframe = str(
        item.get("tf") or ""
    ).lower().strip()

    result["timeframe"] = (
        timeframe or None
    )

    role_key = catalog_id.split(
        "_",
        1,
    )[0].lower()

    if role_key == "market":
        market = packet.get(
            "market",
            {},
        )

        expected_timeframe = str(
            market.get("selected_timeframe")
            if isinstance(market, Mapping)
            else ""
        ).lower().strip()

    elif role_key in {
        "lower",
        "main",
        "confirm",
        "swing",
        "macro",
    }:
        timeframes = packet.get(
            "timeframes",
            {},
        )

        role_block = (
            timeframes.get(role_key, {})
            if isinstance(
                timeframes,
                Mapping,
            )
            else {}
        )

        expected_timeframe = str(
            role_block.get("timeframe")
            if isinstance(
                role_block,
                Mapping,
            )
            else ""
        ).lower().strip()

    else:
        result["reason"] = (
            "Catalog evidence ID has an "
            "unknown timeframe role."
        )
        return result

    if (
        not timeframe
        or not expected_timeframe
        or timeframe != expected_timeframe
    ):
        result["reason"] = (
            "Catalog item timeframe does not "
            "match its packet role."
        )
        return result

    result["valid"] = True
    return result

def _collect_numeric_values(
    value: Any,
    *,
    max_values: int = 250,
) -> list[float]:
    values: list[float] = []

    def walk(item: Any) -> None:
        if len(values) >= max_values:
            return

        if isinstance(item, bool):
            return

        if isinstance(item, Mapping):
            for nested in item.values():
                walk(nested)

                if len(values) >= max_values:
                    break

            return

        if isinstance(
            item,
            (
                list,
                tuple,
                set,
            ),
        ):
            for nested in item:
                walk(nested)

                if len(values) >= max_values:
                    break

            return

        number = _safe_float(item)

        if number is not None:
            values.append(number)

    walk(value)

    return values


def _node_supports_price(
    node: Any,
    target_price: float,
    tolerance: float,
) -> bool:
    """
    Checks whether an anchor node contains a price near the
    proposed Entry / SL / TP2 level.

    A full candle object also supports any price inside its
    High-Low range.
    """

    if isinstance(node, Mapping):
        low = _safe_float(
            node.get("low")
        )

        if low is None:
            low = _safe_float(
                node.get("Low")
            )

        high = _safe_float(
            node.get("high")
        )

        if high is None:
            high = _safe_float(
                node.get("High")
            )

        if (
            low is not None
            and high is not None
        ):
            lower_bound = min(
                low,
                high,
            ) - tolerance

            upper_bound = max(
                low,
                high,
            ) + tolerance

            if (
                lower_bound
                <= target_price
                <= upper_bound
            ):
                return True

    numeric_values = _collect_numeric_values(
        node
    )

    return any(
        abs(value - target_price)
        <= tolerance

        for value in numeric_values
    )


# ==========================================================
# PACKET VALUE HELPERS
# ==========================================================

def _packet_number(
    packet: Mapping[str, Any],
    path: str,
) -> Optional[float]:
    exists, value = _resolve_packet_path(
        packet,
        path,
    )

    if not exists:
        return None

    return _safe_float(value)


def _packet_text(
    packet: Mapping[str, Any],
    path: str,
) -> str:
    exists, value = _resolve_packet_path(
        packet,
        path,
    )

    if not exists:
        return ""

    return _clean_text(
        value,
        200,
    )


def _get_main_atr(
    packet: Mapping[str, Any],
) -> Optional[float]:
    return _packet_number(
        packet,
        "timeframes.main.summary.atr14",
    )


def _get_current_price(
    packet: Mapping[str, Any],
) -> Optional[float]:
    current_price = _packet_number(
        packet,
        "market.current_live_price",
    )

    if current_price is not None:
        return current_price

    return _packet_number(
        packet,
        "market.last_closed_price",
    )


def _get_analog_confidence_score(
    analog_result: Mapping[str, Any],
) -> float:
    try:
        main_result = (
            analog_result
            .get("timeframes", {})
            .get("main", {})
        )

        sample_quality = (
            main_result.get(
                "sample_quality",
                {},
            )
        )

        return float(
            sample_quality.get(
                "confidence_score",
                0.0,
            )
            or 0.0
        )

    except Exception:
        return 0.0


# ==========================================================
# FRONTEND TEXT VALIDATION
# ==========================================================

def _count_sentences(
    text: str,
) -> int:
    cleaned = _clean_text(
        text,
        1000,
    )

    if not cleaned:
        return 0

    parts = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    return len(
        [
            part
            for part in parts
            if part.strip()
        ]
    )


def _contains_markdown(
    text: str,
) -> bool:
    markers = (
        "```",
        "**",
        "__",
        "##",
        "- ",
        "* ",
        "•",
    )

    return any(
        marker in text
        for marker in markers
    )


def _extract_numbers_from_text(
    text: str,
) -> list[float]:
    matches = re.findall(
        r"(?<![A-Za-z])"
        r"[-+]?"
        r"(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?",
        str(text or ""),
    )

    values: list[float] = []

    for match in matches:
        number = _safe_float(match)

        if number is not None:
            values.append(number)

    return values


def _text_mentions_price(
    text: str,
    price: Optional[float],
) -> bool:
    if price is None or price <= 0:
        return False

    numbers = _extract_numbers_from_text(
        text
    )

    tolerance = max(
        abs(price) * 0.00005,
        1e-8,
    )

    return any(
        abs(number - price)
        <= tolerance

        for number in numbers
    )


def _validate_frontend_text(
    signal: Mapping[str, Any],
    *,
    tp1: Optional[float],
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    raw_status = str(
        signal.get("status_text")
        or ""
    )

    raw_advice = str(
        signal.get("advice")
        or ""
    )

    status_text = _clean_text(
        raw_status,
        200,
    )

    advice = _clean_text(
        raw_advice,
        1000,
    )

    decision = _upper_text(
        signal.get("decision"),
        "NO_TRADE",
    )

    execution_state = _upper_text(
        signal.get("execution_state"),
        "NO_TRADE",
    )

    order_type = _upper_text(
        signal.get("order_type"),
        "NONE",
    )

    if not status_text:
        _append_unique(
            errors,
            "Missing frontend status_text.",
        )

    if "\n" in raw_status or "\r" in raw_status:
        _append_unique(
            errors,
            "status_text must contain one line only.",
        )

    if len(status_text) > 60:
        _append_unique(
            errors,
            (
                "status_text exceeds the hard "
                "60-character limit."
            ),
        )

    elif len(status_text) > 40:
        _append_unique(
            warnings,
            (
                "status_text is longer than the "
                "preferred 40-character limit."
            ),
        )

    if not advice:
        _append_unique(
            errors,
            "Missing frontend advice.",
        )

    if "\n" in raw_advice or "\r" in raw_advice:
        _append_unique(
            errors,
            (
                "Frontend advice must be one compact "
                "paragraph without line breaks."
            ),
        )

    if len(advice) > 500:
        _append_unique(
            errors,
            (
                "Frontend advice exceeds the "
                "500-character hard limit."
            ),
        )

    sentence_count = _count_sentences(
        advice
    )

    if sentence_count > 2:
        _append_unique(
            errors,
            "Frontend advice must contain no more than two sentences.",
        )

    if _contains_markdown(
        status_text
    ):
        _append_unique(
            errors,
            "status_text must not contain markdown.",
        )

    if _contains_markdown(
        advice
    ):
        _append_unique(
            errors,
            "Frontend advice must not contain markdown or bullet formatting.",
        )

    status_upper = status_text.upper()

    if decision == "NO_TRADE":
        if not status_upper.startswith(
            "NO TRADE |"
        ):
            _append_unique(
                errors,
                (
                    "NO_TRADE status_text must start "
                    "with 'NO TRADE |'."
                ),
            )

    elif decision == "WAIT":
        if not status_upper.startswith(
            "WAIT |"
        ):
            _append_unique(
                errors,
                (
                    "WAIT status_text must start "
                    "with 'WAIT |'."
                ),
            )

    elif decision in {
        "LONG",
        "SHORT",
    }:
        allowed_prefixes = {
            f"{decision} |",
        }

        if (
            order_type == "LIMIT"
            or execution_state == "PLACE_LIMIT"
        ):
            allowed_prefixes.add(
                "LIMIT ORDER |"
            )

        if not any(
            status_upper.startswith(prefix)
            for prefix in allowed_prefixes
        ):
            _append_unique(
                errors,
                (
                    f"{decision} status_text is inconsistent "
                    "with the trade decision."
                ),
            )

    if (
        advice
        and not advice.upper().startswith(
            "ACTION:"
        )
    ):
        _append_unique(
            errors,
            (
                "Frontend advice must begin with "
                "'ACTION:' and state the immediate action first."
            ),
        )

    entry_price = _safe_float(
        signal.get("entry_price")
    )

    if (
        decision in {
            "LONG",
            "SHORT",
        }
        and execution_state == "ENTER_NOW"
        and _text_mentions_price(
            advice,
            entry_price,
        )
    ):
        _append_unique(
            errors,
            (
                "Frontend advice must not repeat the Entry "
                "price after the trade is active."
            ),
        )

    stop_loss = _safe_float(
        signal.get("stop_loss")
    )

    tp2 = _safe_float(
        signal.get("tp2")
    )

    if _text_mentions_price(
        advice,
        stop_loss,
    ):
        _append_unique(
            errors,
            "Frontend advice repeats the stop-loss price.",
        )

    if _text_mentions_price(
        advice,
        tp1,
    ):
        _append_unique(
            errors,
            "Frontend advice repeats the TP1 price.",
        )

    if _text_mentions_price(
        advice,
        tp2,
    ):
        _append_unique(
            errors,
            "Frontend advice repeats the TP2 price.",
        )

    generic_phrases = (
        "trade carefully",
        "use risk management",
        "be cautious",
        "looks good",
        "seems strong",
        "manage your risk",
    )

    advice_lower = advice.lower()

    if any(
        phrase in advice_lower
        for phrase in generic_phrases
    ):
        _append_unique(
            errors,
            (
                "Frontend advice contains generic wording "
                "instead of market-specific guidance."
            ),
        )

    # Formatting defects are diagnostic warnings only, but malformed
    # text must not be persisted or shown to the frontend. Replace only
    # the presentation text; never change decision, side or trade prices.
    canonicalized = bool(errors)

    if canonicalized:
        if decision == "NO_TRADE":
            status_text = (
                "NO TRADE | No Valid Setup"
            )
            advice = (
                "ACTION: STAY OUT. No validated trade "
                "is available."
            )

        elif decision == "WAIT":
            status_text = (
                "WAIT | Confirmation Required"
            )
            advice = (
                "ACTION: WAIT. Required confirmation "
                "is incomplete."
            )

        elif (
            order_type == "MARKET"
            and execution_state == "ENTER_NOW"
        ):
            status_text = (
                f"{decision} | Entry Confirmed"
            )
            advice = (
                "ACTION: ENTER NOW. Execute only the "
                "validated trade plan shown."
            )

        elif (
            order_type == "LIMIT"
            or execution_state == "PLACE_LIMIT"
        ):
            status_text = (
                "LIMIT ORDER | Fresh Setup"
            )
            advice = (
                "ACTION: PLACE LIMIT. Wait for the "
                "validated entry condition."
            )

        else:
            status_text = (
                f"{decision} | Setup Validated"
            )
            advice = (
                "ACTION: WAIT. Follow the validated "
                "execution state before entering."
            )

        sentence_count = _count_sentences(
            advice
        )

    return {
        "status_text": status_text,
        "advice": advice,
        "sentence_count": sentence_count,
        "canonicalized": canonicalized,
    }


# ==========================================================
# EVIDENCE AND ANCHOR VALIDATION
# ==========================================================

def _normalize_paths(
    value: Any,
) -> list[str]:
    if not isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return []

    paths: list[str] = []

    for item in value:
        path = _normalize_packet_path(
            item
        )

        if (
            path
            and path not in paths
        ):
            paths.append(path)

    return paths

def _is_precise_market_evidence(
    path: str,
    node: Any,
) -> bool:
    normalized = _normalize_packet_path(
        path
    )

    parts = [
        part
        for part in normalized.split(".")
        if part
    ]

    if not parts:
        return False

    root = parts[0]

    if root == "market":
        if len(parts) != 2:
            return False

        if parts[1] not in {
            "current_live_price",
            "last_closed_price",
            "selected_timeframe",
            "market_type",
        }:
            return False

    elif root == "timeframes":
        if len(parts) < 4:
            return False

    elif root in {
        "quantitative_context",
        "historical_analog_engine",
    }:
        if len(parts) < 2:
            return False

    elif root == "evidence_catalog":
        if (
            len(parts) != 4
            or parts[1] != "items"
            or parts[3] != "price"
        ):
            return False

    else:
        return False

    # Evidence must point to a specific leaf value.
    return not isinstance(
        node,
        (
            Mapping,
            list,
            tuple,
            set,
        ),
    )


def _is_specific_anchor_node(
    path: str,
    node: Any,
    *,
    max_numeric_values: int,
) -> bool:
    normalized = _normalize_packet_path(
        path
    )

    parts = [
        part
        for part in normalized.split(".")
        if part
    ]

    if not parts:
        return False

    if len(parts) == 1:
        return False

    if (
        parts[0] == "timeframes"
        and len(parts) <= 3
    ):
        return False

    if not isinstance(node, bool):
        direct_number = _safe_float(
            node
        )

        if direct_number is not None:
            return True

    if isinstance(node, Mapping):
        low = _safe_float(
            node.get("low")
        )

        if low is None:
            low = _safe_float(
                node.get("Low")
            )

        high = _safe_float(
            node.get("high")
        )

        if high is None:
            high = _safe_float(
                node.get("High")
            )

        if (
            low is not None
            and high is not None
        ):
            return True

    numeric_values = _collect_numeric_values(
        node,
        max_values=(
            max_numeric_values + 1
        ),
    )

    return (
        1
        <= len(numeric_values)
        <= max_numeric_values
    )


def _validate_evidence_paths(
    signal: Mapping[str, Any],
    packet: Mapping[str, Any],
    *,
    concrete_plan: bool,
    minimum_required: int,
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    evidence_paths = _normalize_paths(
        signal.get("evidence_paths")
    )

    counter_paths = _normalize_paths(
        signal.get(
            "counter_evidence_paths"
        )
    )

    valid_evidence: list[str] = []
    invalid_evidence: list[str] = []
    imprecise_evidence: list[str] = []

    valid_counter: list[str] = []
    invalid_counter: list[str] = []
    imprecise_counter: list[str] = []

    for path in evidence_paths:
        exists, node = _resolve_packet_path(
            packet,
            path,
        )

        catalog_check = (
            _validate_catalog_reference(
                packet,
                path,
            )
        )

        if not exists:
            invalid_evidence.append(
                path
            )

        elif (
            catalog_check.get(
                "is_catalog_reference"
            )
            and not catalog_check.get("valid")
        ):
            invalid_evidence.append(
                path
            )

        elif not _is_precise_market_evidence(
            path,
            node,
        ):
            imprecise_evidence.append(
                path
            )

        else:
            valid_evidence.append(
                path
            )

    for path in counter_paths:
        exists, node = _resolve_packet_path(
            packet,
            path,
        )

        catalog_check = (
            _validate_catalog_reference(
                packet,
                path,
            )
        )

        if not exists:
            invalid_counter.append(
                path
            )

        elif (
            catalog_check.get(
                "is_catalog_reference"
            )
            and not catalog_check.get("valid")
        ):
            invalid_counter.append(
                path
            )

        elif not _is_precise_market_evidence(
            path,
            node,
        ):
            imprecise_counter.append(
                path
            )

        else:
            valid_counter.append(
                path
            )

    if concrete_plan:
        if len(valid_evidence) < minimum_required:
            _append_unique(
                errors,
                (
                    "Concrete trade plan requires at least "
                    f"{minimum_required} precise market-evidence "
                    f"paths; only {len(valid_evidence)} were valid."
                ),
            )

        if invalid_evidence:
            _append_unique(
                errors,
                (
                    "Some evidence_paths do not exist in the supplied "
                    "market packet: "
                    + ", ".join(
                        invalid_evidence[:6]
                    )
                ),
            )

        if imprecise_evidence:
            _append_unique(
                errors,
                (
                    "Some evidence_paths are administrative or too broad. "
                    "Use exact leaf market-evidence paths: "
                    + ", ".join(
                        imprecise_evidence[:6]
                    )
                ),
            )

        if not valid_counter:
            _append_unique(
                warnings,
                "The signal contains no precise counter-evidence path.",
            )

    else:
        if invalid_evidence:
            _append_unique(
                warnings,
                (
                    "Some non-trade evidence_paths were invalid: "
                    + ", ".join(
                        invalid_evidence[:6]
                    )
                ),
            )

        if imprecise_evidence:
            _append_unique(
                warnings,
                (
                    "Some non-trade evidence_paths were too broad: "
                    + ", ".join(
                        imprecise_evidence[:6]
                    )
                ),
            )

    if invalid_counter:
        _append_unique(
            warnings,
            (
                "Some counter_evidence_paths do not exist: "
                + ", ".join(
                    invalid_counter[:6]
                )
            ),
        )

    if imprecise_counter:
        _append_unique(
            warnings,
            (
                "Some counter_evidence_paths are too broad: "
                + ", ".join(
                    imprecise_counter[:6]
                )
            ),
        )

    return {
        "valid_evidence_paths": valid_evidence,
        "invalid_evidence_paths": invalid_evidence,
        "imprecise_evidence_paths": (
            imprecise_evidence
        ),
        "valid_counter_evidence_paths": valid_counter,
        "invalid_counter_evidence_paths": invalid_counter,
        "imprecise_counter_evidence_paths": (
            imprecise_counter
        ),
    }

def _validate_single_anchor(
    *,
    anchor_name: str,
    path: Any,
    target_price: Optional[float],
    packet: Mapping[str, Any],
    atr: Optional[float],
    config: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
    expected_catalog_types: Optional[
        Sequence[str]
    ] = None,
    ) -> Dict[str, Any]:
    normalized_path = _normalize_packet_path(
        path
    )

    result = {
        "path": normalized_path or "NONE",
        "exists": False,
        "specific_path": False,
        "numeric_price_support": False,
        "catalog_validation": {},
        "target_price": _round_price(
            target_price
        ),
    }

    if (
        not normalized_path
        or normalized_path.upper() == "NONE"
    ):
        _append_unique(
            errors,
            f"Missing {anchor_name} anchor path.",
        )

        return result

    exists, node = _resolve_packet_path(
        packet,
        normalized_path,
    )

    result["exists"] = exists

    if not exists:
        _append_unique(
            errors,
            (
                f"{anchor_name} anchor path does not exist "
                f"in the market packet: {normalized_path}"
            ),
        )

        return result

    catalog_validation = (
        _validate_catalog_reference(
            packet,
            normalized_path,
            expected_types=(
                expected_catalog_types
            ),
        )
    )

    result["catalog_validation"] = (
        catalog_validation
    )

    if (
        catalog_validation.get(
            "is_catalog_reference"
        )
        and not catalog_validation.get("valid")
    ):
        _append_unique(
            errors,
            (
                f"{anchor_name} catalog anchor is invalid: "
                f"{catalog_validation.get('reason') or 'UNKNOWN_REASON'}"
            ),
        )

        return result

    max_numeric_values = max(
        1,
        int(
            _safe_float(
                config.get(
                    "max_anchor_numeric_values",
                    3,
                ),
                3.0,
            )
            or 3
        ),
    )

    specific_path = (
        _is_specific_anchor_node(
            normalized_path,
            node,
            max_numeric_values=(
                max_numeric_values
            ),
        )
    )

    result["specific_path"] = (
        specific_path
    )

    if not specific_path:
        _append_unique(
            errors,
            (
                f"{anchor_name} anchor path is too broad. "
                "Use a specific price field, zone, or candle path: "
                f"{normalized_path}"
            ),
        )

        return result

    if (
        target_price is None
        or target_price <= 0
    ):
        _append_unique(
            errors,
            f"{anchor_name} target price is invalid.",
        )

        return result

    atr_tolerance = (
        _safe_float(
            config.get(
                "anchor_tolerance_atr",
                0.75,
            ),
            0.75,
        )
        or 0.75
    )

    fraction_tolerance = (
        _safe_float(
            config.get(
                "anchor_tolerance_fraction",
                0.005,
            ),
            0.005,
        )
        or 0.005
    )

    tolerance = max(
        (
            atr_tolerance * atr
        )
        if atr is not None and atr > 0
        else 0.0,

        abs(target_price)
        * fraction_tolerance,

        1e-8,
    )

    numeric_support = _node_supports_price(
        node,
        target_price,
        tolerance,
    )

    result["numeric_price_support"] = (
        numeric_support
    )

    result["tolerance"] = _round_price(
        tolerance
    )

    if not numeric_support:
        _append_unique(
            errors,
            (
                f"{anchor_name} anchor exists but does not contain "
                "a numeric market level near the proposed price."
            ),
        )

    return result


# ==========================================================
# DIRECTION AND GEOMETRY
# ==========================================================

def _calculate_tp1(
    side: str,
    entry: float,
    stop_loss: float,
    tp1_r: float,
) -> float:
    risk_distance = abs(
        entry - stop_loss
    )

    if side == "LONG":
        return entry + (
            risk_distance * tp1_r
        )

    return entry - (
        risk_distance * tp1_r
    )


def _validate_decision_consistency(
    signal: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    decision = _upper_text(
        signal.get("decision"),
        "NO_TRADE",
    )

    side = _upper_text(
        signal.get("side"),
        "NONE",
    )

    execution_state = _upper_text(
        signal.get("execution_state"),
        "NO_TRADE",
    )

    order_type = _upper_text(
        signal.get("order_type"),
        "NONE",
    )

    trade_quality = _upper_text(
        signal.get("trade_quality"),
        "NO_TRADE",
    )

    setup_type = _upper_text(
        signal.get("setup_type"),
        "NO_TRADE",
    )

    entry = _safe_float(
        signal.get("entry_price"),
        0.0,
    ) or 0.0

    stop_loss = _safe_float(
        signal.get("stop_loss"),
        0.0,
    ) or 0.0

    tp2 = _safe_float(
        signal.get("tp2"),
        0.0,
    ) or 0.0

    if decision not in ALLOWED_DECISIONS:
        _append_unique(
            errors,
            f"Invalid decision value: {decision}",
        )

    if side not in ALLOWED_SIDES:
        _append_unique(
            errors,
            f"Invalid side value: {side}",
        )

    if (
        execution_state
        not in ALLOWED_EXECUTION_STATES
    ):
        _append_unique(
            errors,
            (
                "Invalid execution_state value: "
                f"{execution_state}"
            ),
        )

    if order_type not in ALLOWED_ORDER_TYPES:
        _append_unique(
            errors,
            f"Invalid order_type value: {order_type}",
        )

    has_any_price = any(
        value > 0
        for value in (
            entry,
            stop_loss,
            tp2,
        )
    )

    has_complete_prices = all(
        value > 0
        for value in (
            entry,
            stop_loss,
            tp2,
        )
    )

    if decision in {
        "LONG",
        "SHORT",
    }:
        if side != decision:
            _append_unique(
                errors,
                (
                    f"Decision {decision} requires "
                    f"side {decision}."
                ),
            )

        if not has_complete_prices:
            _append_unique(
                errors,
                (
                    f"{decision} requires valid Entry, "
                    "Stop Loss and TP2 prices."
                ),
            )

        if order_type == "NONE":
            _append_unique(
                errors,
                (
                    f"{decision} requires MARKET, LIMIT "
                    "or STOP order_type."
                ),
            )

        if execution_state == "NO_TRADE":
            _append_unique(
                errors,
                (
                    f"{decision} cannot use "
                    "execution_state NO_TRADE."
                ),
            )

        if (
            execution_state == "ENTER_NOW"
            and order_type != "MARKET"
        ):
            _append_unique(
                errors,
                (
                    "ENTER_NOW requires order_type MARKET "
                    "for an active LONG/SHORT signal."
                ),
            )

        if (
            order_type == "MARKET"
            and execution_state != "ENTER_NOW"
        ):
            _append_unique(
                errors,
                (
                    "MARKET order_type requires "
                    "execution_state ENTER_NOW."
                ),
            )

        if (
            execution_state == "PLACE_LIMIT"
            and order_type != "LIMIT"
        ):
            _append_unique(
                errors,
                "PLACE_LIMIT requires order_type LIMIT.",
            )

        if (
            order_type == "LIMIT"
            and execution_state == "ENTER_NOW"
        ):
            _append_unique(
                errors,
                (
                    "LIMIT order_type cannot use "
                    "execution_state ENTER_NOW."
                ),
            )

        if trade_quality == "NO_TRADE":
            _append_unique(
                errors,
                (
                    f"{decision} cannot use "
                    "trade_quality NO_TRADE."
                ),
            )

        if setup_type == "NO_TRADE":
            _append_unique(
                errors,
                (
                    f"{decision} cannot use "
                    "setup_type NO_TRADE."
                ),
            )

    elif decision == "WAIT":
        if (
            execution_state
            not in WAIT_EXECUTION_STATES
        ):
            _append_unique(
                errors,
                (
                    "WAIT requires a waiting execution_state "
                    "such as WAIT_FOR_ENTRY, WAIT_FOR_RETEST, "
                    "WAIT_FOR_RECLAIM, WAIT_FOR_BREAKOUT, or "
                    "WAIT_FOR_FAILED_RECLAIM."
                ),
            )

        if order_type == "MARKET":
            _append_unique(
                errors,
                "WAIT cannot use order_type MARKET.",
            )

        if side == "NONE":
            if has_any_price:
                _append_unique(
                    errors,
                    (
                        "WAIT with side NONE must not contain "
                        "trade prices."
                    ),
                )

            if order_type != "NONE":
                _append_unique(
                    errors,
                    (
                        "WAIT with side NONE requires "
                        "order_type NONE."
                    ),
                )

        elif side in {
            "LONG",
            "SHORT",
        }:
            if not has_complete_prices:
                _append_unique(
                    errors,
                    (
                        "Directional WAIT plan requires complete "
                        "Entry, Stop Loss and TP2 prices."
                    ),
                )

    elif decision == "NO_TRADE":
        if side != "NONE":
            _append_unique(
                errors,
                "NO_TRADE requires side NONE.",
            )

        if order_type != "NONE":
            _append_unique(
                errors,
                "NO_TRADE requires order_type NONE.",
            )

        if execution_state != "NO_TRADE":
            _append_unique(
                errors,
                (
                    "NO_TRADE requires "
                    "execution_state NO_TRADE."
                ),
            )

        if has_any_price:
            _append_unique(
                errors,
                (
                    "NO_TRADE must use zero Entry, "
                    "Stop Loss and TP2 prices."
                ),
            )

        if trade_quality != "NO_TRADE":
            _append_unique(
                errors,
                (
                    "NO_TRADE requires "
                    "trade_quality NO_TRADE."
                ),
            )

        if setup_type != "NO_TRADE":
            _append_unique(
                warnings,
                (
                    "NO_TRADE normally should use "
                    "setup_type NO_TRADE."
                ),
            )

    concrete_plan = (
        side in {
            "LONG",
            "SHORT",
        }
        and has_complete_prices
        and decision in {
            "LONG",
            "SHORT",
            "WAIT",
        }
    )

    return {
        "decision": decision,
        "side": side,
        "execution_state": execution_state,
        "order_type": order_type,
        "entry": entry,
        "stop_loss": stop_loss,
        "tp2": tp2,
        "trade_quality": trade_quality,
        "setup_type": setup_type,
        "concrete_plan": concrete_plan,
    }


# ==========================================================
# STOP AND TARGET VALIDATION
# ==========================================================

def _normalize_risk_timeframe(
    packet: Mapping[str, Any],
) -> str:
    """
    Normalize the selected timeframe used by the hard risk policy.
    """

    timeframe = str(
        _packet_text(
            packet,
            "market.selected_timeframe",
        )
        or ""
    ).lower().strip().replace(
        " ",
        "",
    )

    aliases = {
        "15": "15m",
        "15min": "15m",
        "15mins": "15m",
        "15minute": "15m",
        "15minutes": "15m",

        "30": "30m",
        "30min": "30m",
        "30mins": "30m",
        "30minute": "30m",
        "30minutes": "30m",

        "60": "1h",
        "60m": "1h",
        "60min": "1h",
        "1hr": "1h",
        "1hour": "1h",

        "240": "4h",
        "240m": "4h",
        "240min": "4h",
        "4hr": "4h",
        "4hour": "4h",

        "1440m": "1d",
        "1440min": "1d",
        "d": "1d",
        "day": "1d",
        "daily": "1d",
    }

    return aliases.get(
        timeframe,
        timeframe,
    )


def _timeframe_risk_limits(
    packet: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Resolve timeframe-specific SL and TP2 hard limits.

    Unknown timeframes use the strict 15m/30m policy.
    """

    timeframe = _normalize_risk_timeframe(
        packet
    )

    if timeframe in {
        "15m",
        "30m",
    }:
        policy = "15m_30m"

    elif timeframe == "1h":
        policy = "1h"

    elif timeframe == "4h":
        policy = "4h"

    elif timeframe == "1d":
        policy = "1d"

    else:
        policy = "15m_30m"

    return {
        "selected_timeframe": (
            timeframe or "UNKNOWN"
        ),

        "policy": policy,

        "used_fallback_policy": (
            timeframe
            not in {
                "15m",
                "30m",
                "1h",
                "4h",
                "1d",
            }
        ),

        "hard_min_stop_atr": float(
            config.get(
                "hard_min_stop_atr",
                1.30,
            )
        ),

        "hard_max_stop_atr": float(
            config.get(
                f"hard_max_stop_atr_{policy}",
                2.80,
            )
        ),

        "hard_min_tp2_r": float(
            config.get(
                "hard_min_tp2_r",
                2.00,
            )
        ),

        "hard_max_tp2_r": float(
            config.get(
                "hard_max_tp2_r",
                5.00,
            )
        ),

        "hard_max_tp2_atr": float(
            config.get(
                f"hard_max_tp2_atr_{policy}",
                5.00,
            )
        ),
    }


def _validate_geometry_and_targets(
    *,
    side: str,
    entry: float,
    stop_loss: float,
    tp2: float,
    tp1_r: float,
    packet: Mapping[str, Any],
    config: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    starting_error_count = len(
        errors
    )

    result: Dict[str, Any] = {
        "entry": _round_price(entry),
        "stop_loss": _round_price(
            stop_loss
        ),
        "tp1": None,

        # Final TP2 may be capped below the AI request.
        "tp2": _round_price(tp2),
        "requested_tp2": _round_price(
            tp2
        ),

        "risk_distance": None,
        "risk_distance_atr": None,

        "tp2_r_multiple": None,
        "tp2_distance_atr": None,

        "requested_tp2_r_multiple": None,
        "requested_tp2_distance_atr": None,

        "tp2_was_capped": False,
        "tp2_cap_reason": "",
        "effective_max_tp2_r": None,

        "current_price": _round_price(
            _get_current_price(packet)
        ),

        "entry_chase_fraction": None,
        "risk_policy": {},
    }

    if side == "LONG":
        if stop_loss >= entry:
            _append_unique(
                errors,
                (
                    "LONG stop loss must be below "
                    "the entry price."
                ),
            )

        if tp2 <= entry:
            _append_unique(
                errors,
                (
                    "LONG TP2 must be above "
                    "the entry price."
                ),
            )

    elif side == "SHORT":
        if stop_loss <= entry:
            _append_unique(
                errors,
                (
                    "SHORT stop loss must be above "
                    "the entry price."
                ),
            )

        if tp2 >= entry:
            _append_unique(
                errors,
                (
                    "SHORT TP2 must be below "
                    "the entry price."
                ),
            )

    if len(errors) > starting_error_count:
        return result

    risk_distance = abs(
        entry - stop_loss
    )

    result["risk_distance"] = (
        _round_price(
            risk_distance
        )
    )

    if risk_distance <= 0:
        _append_unique(
            errors,
            (
                "Trade risk distance must be "
                "greater than zero."
            ),
        )

        return result

    tp1 = _calculate_tp1(
        side,
        entry,
        stop_loss,
        tp1_r,
    )

    result["tp1"] = _round_price(
        tp1
    )

    requested_tp2 = tp2

    requested_tp2_distance = (
        requested_tp2 - entry
        if side == "LONG"
        else entry - requested_tp2
    )

    requested_tp2_r = (
        requested_tp2_distance
        / risk_distance
    )

    result[
        "requested_tp2_r_multiple"
    ] = round(
        requested_tp2_r,
        4,
    )

    limits = _timeframe_risk_limits(
        packet,
        config,
    )

    result["risk_policy"] = dict(
        limits
    )

    if limits.get(
        "used_fallback_policy"
    ):
        _append_unique(
            warnings,
            (
                "Unknown selected timeframe; strict "
                "15m/30m risk limits were applied."
            ),
        )

    hard_min_stop_atr = float(
        limits["hard_min_stop_atr"]
    )

    hard_max_stop_atr = float(
        limits["hard_max_stop_atr"]
    )

    hard_min_tp2_r = float(
        limits["hard_min_tp2_r"]
    )

    hard_max_tp2_r = float(
        limits["hard_max_tp2_r"]
    )

    hard_max_tp2_atr = float(
        limits["hard_max_tp2_atr"]
    )

    if (
        requested_tp2_r
        < hard_min_tp2_r
        and not math.isclose(
            requested_tp2_r,
            hard_min_tp2_r,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        _append_unique(
            errors,
            (
                f"TP2 is too close at "
                f"{requested_tp2_r:.2f}R; "
                f"minimum is "
                f"{hard_min_tp2_r:.2f}R."
            ),
        )

    atr = _get_main_atr(
        packet
    )

    result["main_atr"] = _round_price(
        atr
    )

    maximum_by_r = (
        risk_distance
        * hard_max_tp2_r
    )

    maximum_tp2_distance = (
        maximum_by_r
    )

    cap_reason = (
        f"{hard_max_tp2_r:.2f}R maximum"
    )

    if atr is None or atr <= 0:
        _append_unique(
            errors,
            (
                "Main-timeframe ATR is required for "
                "timeframe-specific SL and TP2 validation."
            ),
        )

    else:
        risk_atr = (
            risk_distance / atr
        )

        requested_tp2_atr = (
            requested_tp2_distance / atr
        )

        result[
            "risk_distance_atr"
        ] = round(
            risk_atr,
            4,
        )

        result[
            "requested_tp2_distance_atr"
        ] = round(
            requested_tp2_atr,
            4,
        )

        if (
            risk_atr < hard_min_stop_atr
            and not math.isclose(
                risk_atr,
                hard_min_stop_atr,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            _append_unique(
                errors,
                (
                    f"Stop is too tight at "
                    f"{risk_atr:.2f} ATR; "
                    f"minimum is "
                    f"{hard_min_stop_atr:.2f} ATR."
                ),
            )

        elif (
            risk_atr > hard_max_stop_atr
            and not math.isclose(
                risk_atr,
                hard_max_stop_atr,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            _append_unique(
                errors,
                (
                    f"Stop is too wide at "
                    f"{risk_atr:.2f} ATR for "
                    f"{limits['selected_timeframe']}; "
                    f"maximum is "
                    f"{hard_max_stop_atr:.2f} ATR."
                ),
            )

        maximum_by_atr = (
            atr
            * hard_max_tp2_atr
        )

        maximum_tp2_distance = min(
            maximum_by_r,
            maximum_by_atr,
        )

        if (
            maximum_by_atr
            < maximum_by_r
        ):
            cap_reason = (
                f"{hard_max_tp2_atr:.2f} ATR "
                "timeframe maximum"
            )

    effective_max_tp2_r = (
        maximum_tp2_distance
        / risk_distance
    )

    result[
        "effective_max_tp2_r"
    ] = round(
        effective_max_tp2_r,
        4,
    )

    if (
        effective_max_tp2_r
        < hard_min_tp2_r
        and not math.isclose(
            effective_max_tp2_r,
            hard_min_tp2_r,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        _append_unique(
            errors,
            (
                "No valid TP2 range exists for this "
                "stop width: the timeframe ATR cap "
                f"allows only {effective_max_tp2_r:.2f}R, "
                f"below the {hard_min_tp2_r:.2f}R minimum."
            ),
        )

    final_tp2_distance = (
        requested_tp2_distance
    )

    final_tp2 = requested_tp2

    if (
        requested_tp2_distance
        > maximum_tp2_distance
        and not math.isclose(
            requested_tp2_distance,
            maximum_tp2_distance,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        final_tp2_distance = (
            maximum_tp2_distance
        )

        final_tp2 = (
            entry + final_tp2_distance
            if side == "LONG"
            else entry - final_tp2_distance
        )

        result["tp2_was_capped"] = True
        result["tp2_cap_reason"] = (
            cap_reason
        )

        _append_unique(
            warnings,
            (
                f"TP2 was capped from "
                f"{requested_tp2_r:.2f}R to "
                f"{effective_max_tp2_r:.2f}R by the "
                f"{cap_reason}."
            ),
        )

    final_tp2_r = (
        final_tp2_distance
        / risk_distance
    )

    tp2_beyond_tp1 = (
        final_tp2 > tp1
        if side == "LONG"
        else final_tp2 < tp1
    )

    if not tp2_beyond_tp1:
        _append_unique(
            errors,
            (
                "Final TP2 must be beyond the exact "
                "1.5R TP1 in the trade direction."
            ),
        )

    result["tp2"] = _round_price(
        final_tp2
    )

    result["tp2_r_multiple"] = round(
        final_tp2_r,
        4,
    )

    if atr is not None and atr > 0:
        result[
            "tp2_distance_atr"
        ] = round(
            final_tp2_distance / atr,
            4,
        )

    return result

def _validate_order_and_market_state(
    *,
    side: str,
    entry: float,
    stop_loss: float,
    tp2: float,
    order_type: str,
    packet: Mapping[str, Any],
    config: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    current_price = _get_current_price(
        packet
    )

    atr = _get_main_atr(
        packet
    )

    result: Dict[str, Any] = {
        "current_price": _round_price(
            current_price
        ),
        "market_entry_tolerance": None,
        "pending_order_tolerance": None,
    }

    if (
        current_price is None
        or current_price <= 0
    ):
        _append_unique(
            warnings,
            (
                "Current market price is unavailable; "
                "order-price validation is limited."
            ),
        )

        return result

    market_atr_fraction = (
        _safe_float(
            config.get(
                "market_entry_tolerance_atr",
                0.35,
            ),
            0.35,
        )
        or 0.35
    )

    market_price_fraction = (
        _safe_float(
            config.get(
                "market_entry_tolerance_fraction",
                0.003,
            ),
            0.003,
        )
        or 0.003
    )

    pending_atr_fraction = (
        _safe_float(
            config.get(
                "pending_order_tolerance_atr",
                0.10,
            ),
            0.10,
        )
        or 0.10
    )

    pending_price_fraction = (
        _safe_float(
            config.get(
                "pending_order_tolerance_fraction",
                0.001,
            ),
            0.001,
        )
        or 0.001
    )

    market_tolerance = max(
        (
            atr * market_atr_fraction
        )
        if atr is not None and atr > 0
        else 0.0,

        abs(current_price)
        * market_price_fraction,

        1e-8,
    )

    pending_tolerance = max(
        (
            atr * pending_atr_fraction
        )
        if atr is not None and atr > 0
        else 0.0,

        abs(current_price)
        * pending_price_fraction,

        1e-8,
    )

    result[
        "market_entry_tolerance"
    ] = _round_price(
        market_tolerance
    )

    result[
        "pending_order_tolerance"
    ] = _round_price(
        pending_tolerance
    )

    if side == "LONG":
        if current_price <= stop_loss:
            _append_unique(
                errors,
                (
                    "Current price is already at or below "
                    "the LONG stop-loss invalidation."
                ),
            )

        if current_price >= tp2:
            _append_unique(
                errors,
                (
                    "Current price is already at or beyond "
                    "the LONG TP2 target."
                ),
            )

    elif side == "SHORT":
        if current_price >= stop_loss:
            _append_unique(
                errors,
                (
                    "Current price is already at or above "
                    "the SHORT stop-loss invalidation."
                ),
            )

        if current_price <= tp2:
            _append_unique(
                errors,
                (
                    "Current price is already at or beyond "
                    "the SHORT TP2 target."
                ),
            )

    if order_type == "MARKET":
        if (
            abs(entry - current_price)
            > market_tolerance
        ):
            _append_unique(
                errors,
                (
                    "MARKET entry is too far from the "
                    "current market price."
                ),
            )

    elif order_type == "LIMIT":
        if (
            side == "LONG"
            and entry > (
                current_price
                + pending_tolerance
            )
        ):
            _append_unique(
                errors,
                (
                    "LONG LIMIT entry must not be materially "
                    "above the current market price."
                ),
            )

        elif (
            side == "SHORT"
            and entry < (
                current_price
                - pending_tolerance
            )
        ):
            _append_unique(
                errors,
                (
                    "SHORT LIMIT entry must not be materially "
                    "below the current market price."
                ),
            )

    elif order_type == "STOP":
        if (
            side == "LONG"
            and entry < (
                current_price
                - pending_tolerance
            )
        ):
            _append_unique(
                errors,
                (
                    "LONG STOP entry must not be materially "
                    "below the current market price."
                ),
            )

        elif (
            side == "SHORT"
            and entry > (
                current_price
                + pending_tolerance
            )
        ):
            _append_unique(
                errors,
                (
                    "SHORT STOP entry must not be materially "
                    "above the current market price."
                ),
            )

    return result

# ==========================================================
# ENTRY CHASE VALIDATION
# ==========================================================

def _validate_entry_chase(
    *,
    side: str,
    entry: float,
    tp1: Optional[float],
    current_price: Optional[float],
    order_type: str,
    execution_state: str,
    maximum_fraction: float,
    errors: list[str],
    warnings: list[str],
) -> Optional[float]:
    if (
        tp1 is None
        or current_price is None
    ):
        _append_unique(
            warnings,
            (
                "Entry chase could not be fully checked "
                "because current price or TP1 is unavailable."
            ),
        )

        return None

    tp1_distance = abs(
        tp1 - entry
    )

    if tp1_distance <= 0:
        return None

    if side == "LONG":
        progress = (
            current_price - entry
        ) / tp1_distance

    else:
        progress = (
            entry - current_price
        ) / tp1_distance

    if progress >= 1.0:
        _append_unique(
            errors,
            (
                "Current price is already at or beyond TP1 "
                "relative to the proposed entry. "
                "The setup is stale and must not be published."
            ),
        )

    elif progress >= maximum_fraction:
        message = (
            f"Current price has already travelled "
            f"{progress * 100:.1f}% of the Entry-to-TP1 path."
        )

        if (
            order_type == "MARKET"
            or execution_state == "ENTER_NOW"
        ):
            _append_unique(
                errors,
                (
                    message
                    + " Immediate entry violates the no-chase rule."
                ),
            )

        else:
            _append_unique(
                warnings,
                (
                    message
                    + " Keep the pending entry only if the original "
                    "level has not already been filled or invalidated."
                ),
            )

    return round(
        progress,
        4,
    )


# ==========================================================
# MULTI-TIMEFRAME DIRECTION CHECK
# ==========================================================

def _validate_simple_timeframe_alignment(
    *,
    side: str,
    packet: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    main_direction = _upper_text(
        _packet_text(
            packet,
            (
                "timeframes.main.summary."
                "short_term_direction"
            ),
        ),
        "UNKNOWN",
    )

    confirm_direction = _upper_text(
        _packet_text(
            packet,
            (
                "timeframes.confirm.summary."
                "short_term_direction"
            ),
        ),
        "UNKNOWN",
    )

    opposing = (
        "BEARISH"
        if side == "LONG"
        else "BULLISH"
    )

    aligned = (
        "BULLISH"
        if side == "LONG"
        else "BEARISH"
    )

    if (
        main_direction == opposing
        and confirm_direction == opposing
    ):
        _append_unique(
            warnings,
            (
                f"Both main and confirmation candle slopes "
                f"currently oppose the proposed {side} side."
            ),
        )

    elif (
        main_direction == aligned
        and confirm_direction == aligned
    ):
        pass

    elif (
        main_direction == opposing
        or confirm_direction == opposing
    ):
        _append_unique(
            warnings,
            (
                f"One important timeframe currently opposes "
                f"the proposed {side} side."
            ),
        )

    return {
        "main_short_term_direction": (
            main_direction
        ),
        "confirm_short_term_direction": (
            confirm_direction
        ),
    }


# ==========================================================
# HISTORICAL ANALOG VALIDATION
# ==========================================================

def _validate_historical_trade(
    *,
    analog_result: Optional[
        Mapping[str, Any]
    ],
    side: str,
    entry: float,
    stop_loss: float,
    tp2: float,
    config: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    if not isinstance(
        analog_result,
        Mapping,
    ):
        _append_unique(
            warnings,
            (
                "Full Historical Analog result is unavailable; "
                "exact path validation was skipped."
            ),
        )

        return {
            "enabled": False,
            "status": (
                "ANALOG_RESULT_UNAVAILABLE"
            ),
        }

    try:
        validation = (
            validate_trade_with_historical_analogs(
                analog_result=analog_result,
                side=side,
                entry=entry,
                stop_loss=stop_loss,
                tp2=tp2,
                tp1_r=(
                    TP1_R_MULTIPLE
                ),
            )
        )

    except Exception as exc:
        _append_unique(
            warnings,
            (
                "Historical Analog trade validation failed: "
                f"{str(exc)}"
            ),
        )

        return {
            "enabled": False,
            "status": "ANALOG_VALIDATION_ERROR",
            "reason": str(exc),
        }

    if not validation.get(
        "enabled"
    ):
        _append_unique(
            warnings,
            (
                "Historical Analog trade-path validation "
                "was unavailable: "
                f"{validation.get('status', 'UNKNOWN')}"
            ),
        )

        return validation

    sample_size = _safe_int(
        validation.get("sample_size"),
        0,
    )

    confidence_score = (
        _get_analog_confidence_score(
            analog_result
        )
    )

    tp1_probability = _safe_float(
        validation.get(
            "tp1_before_sl_probability"
        ),
        0.0,
    ) or 0.0

    sl_probability = _safe_float(
        validation.get(
            "sl_before_tp1_probability"
        ),
        0.0,
    ) or 0.0

    tp2_probability = _safe_float(
        validation.get(
            "tp2_before_sl_probability"
        )
    )

    hard_sample = int(
        config.get(
            "analog_min_sample_for_hard_reject",
            10,
        )
    )

    hard_confidence = float(
        config.get(
            (
                "analog_min_confidence_"
                "for_hard_reject"
            ),
            60.0,
        )
    )

    hard_tp1_minimum = float(
        config.get(
            "analog_hard_min_tp1_probability",
            0.25,
        )
    )

    hard_sl_minimum = float(
        config.get(
            "analog_hard_min_sl_probability",
            0.55,
        )
    )

    if (
        sample_size >= hard_sample
        and confidence_score >= hard_confidence
        and tp1_probability < hard_tp1_minimum
        and sl_probability > hard_sl_minimum
    ):
        _append_unique(
            errors,
            (
                "High-confidence historical analog paths strongly "
                "favour Stop Loss before TP1 "
                f"(TP1 first {tp1_probability:.0%}, "
                f"SL first {sl_probability:.0%})."
            ),
        )

    elif tp1_probability < float(
        config.get(
            "analog_warning_min_tp1_probability",
            0.40,
        )
    ):
        _append_unique(
            warnings,
            (
                "Historical analog TP1-before-SL probability "
                f"is weak at {tp1_probability:.0%}."
            ),
        )

    if (
        tp2_probability is not None
        and tp2_probability < float(
            config.get(
                (
                    "analog_warning_min_"
                    "tp2_probability"
                ),
                0.20,
            )
        )
    ):
        _append_unique(
            warnings,
            (
                "Historical analog TP2-before-SL probability "
                f"is weak at {tp2_probability:.0%}."
            ),
        )

    validation = dict(
        validation
    )

    validation["main_analog_confidence_score"] = (
        round(
            confidence_score,
            2,
        )
    )

    return validation


# ==========================================================
# ADVERSE-MOVE STOP CHECK
# ==========================================================

def _validate_expected_adverse_move(
    *,
    analog_result: Optional[
        Mapping[str, Any]
    ],
    risk_distance_atr: Optional[float],
    errors: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    output = {
        "p75_expected_adverse_atr": None,
        "risk_to_p75_ratio": None,
    }

    if (
        not isinstance(
            analog_result,
            Mapping,
        )
        or risk_distance_atr is None
    ):
        return output

    try:
        main_result = (
            analog_result
            .get("timeframes", {})
            .get("main", {})
        )

        adverse = (
            main_result.get(
                "expected_adverse_move",
                {},
            )
        )

        p75_adverse = _safe_float(
            adverse.get(
                "p75_adverse_move_atr"
            )
        )

        confidence_score = (
            _get_analog_confidence_score(
                analog_result
            )
        )

        if (
            p75_adverse is None
            or p75_adverse <= 0
        ):
            return output

        ratio = (
            risk_distance_atr
            / p75_adverse
        )

        output.update(
            {
                "p75_expected_adverse_atr": (
                    round(
                        p75_adverse,
                        4,
                    )
                ),

                "risk_to_p75_ratio": (
                    round(
                        ratio,
                        4,
                    )
                ),
            }
        )

        if (
            confidence_score >= 60
            and ratio < 0.55
        ):
            _append_unique(
                warnings,
                (
                    "Stop distance is much smaller than the "
                    "75th-percentile historical adverse move."
                ),
            )

        elif (
            confidence_score >= 40
            and ratio < 0.75
        ):
            _append_unique(
                warnings,
                (
                    "Stop may be vulnerable to historically normal "
                    "adverse movement before continuation."
                ),
            )

    except Exception:
        pass

    return output


# ==========================================================
# PUBLIC VALIDATION FUNCTION
# ==========================================================

def validate_ai_signal(
    *,
    signal: Mapping[str, Any],
    market_packet: Mapping[str, Any],
    historical_analog_result: Optional[
        Mapping[str, Any]
    ] = None,
    config: Optional[
        Mapping[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Validate one AI signal draft.

    Returns:

    {
        "valid": bool,
        "needs_correction": bool,
        "trade_publishable": bool,
        "watch_publishable": bool,
        "final_signal": {...},
        "errors": [...],
        "warnings": [...],
        "trade": {...},
        "evidence_validation": {...},
        "anchor_validation": {...},
        "historical_validation": {...},
    }
    """

    cfg = _merge_config(
        config
    )

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(
        signal,
        Mapping,
    ):
        return {
            "valid": False,
            "needs_correction": True,
            "trade_publishable": False,
            "watch_publishable": False,
            "validator_version": (
                VALIDATOR_VERSION
            ),
            "final_signal": None,
            "errors": [
                "AI signal is not a mapping/object."
            ],
            "warnings": [],
            "trade": {},
            "evidence_validation": {},
            "anchor_validation": {},
            "historical_validation": {},
        }

    if not isinstance(
        market_packet,
        Mapping,
    ):
        return {
            "valid": False,
            "needs_correction": True,
            "trade_publishable": False,
            "watch_publishable": False,
            "validator_version": (
                VALIDATOR_VERSION
            ),
            "final_signal": dict(signal),
            "errors": [
                "AI market packet is not a mapping/object."
            ],
            "warnings": [],
            "trade": {},
            "evidence_validation": {},
            "anchor_validation": {},
            "historical_validation": {},
        }

    normalized_signal = dict(
        signal
    )

    # Gemini may preserve all path words but remove underscores
    # or alter case. Repair only uniquely matching real paths.
    def canonicalize_signal_paths(
        value: Any,
    ) -> list[str]:
        canonical_paths: list[str] = []

        for path in _normalize_paths(
            value
        ):
            canonical_path = (
                _canonicalize_packet_path(
                    market_packet,
                    path,
                )
            )

            if (
                canonical_path
                and canonical_path
                not in canonical_paths
            ):
                canonical_paths.append(
                    canonical_path
                )

        return canonical_paths

    normalized_signal["evidence_paths"] = (
        canonicalize_signal_paths(
            normalized_signal.get(
                "evidence_paths"
            )
        )
    )

    normalized_signal[
        "counter_evidence_paths"
    ] = canonicalize_signal_paths(
        normalized_signal.get(
            "counter_evidence_paths"
        )
    )

    for anchor_field in (
        "entry_anchor_path",
        "sl_anchor_path",
        "tp2_anchor_path",
    ):
        normalized_signal[
            anchor_field
        ] = _canonicalize_packet_path(
            market_packet,
            normalized_signal.get(
                anchor_field
            ),
        )

    consistency = (
        _validate_decision_consistency(
            normalized_signal,
            errors,
            warnings,
        )
    )

    decision = consistency["decision"]
    side = consistency["side"]
    execution_state = consistency[
        "execution_state"
    ]
    order_type = consistency[
        "order_type"
    ]

    entry = consistency["entry"]
    stop_loss = consistency["stop_loss"]
    tp2 = consistency["tp2"]

    concrete_plan = consistency[
        "concrete_plan"
    ]

    supplied_tp1_r = _safe_float(
        normalized_signal.get(
            "tp1_r_multiple"
        )
    )

    required_tp1_r = (
        TP1_R_MULTIPLE
    )

    if (
        supplied_tp1_r is None
        or not math.isclose(
            supplied_tp1_r,
            required_tp1_r,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        _append_unique(
            errors,
            (
                "tp1_r_multiple must be exactly "
                f"{required_tp1_r:.1f}."
            ),
        )

    # Backend remains the final owner of TP1.
    normalized_signal["tp1_r_multiple"] = (
        required_tp1_r
    )

    # Evidence citation path mistakes must not invalidate an otherwise
    # structurally and mathematically valid trade. Entry, SL and TP2
    # anchor validation remains hard and is handled separately below.
    evidence_path_errors: list[str] = []

    evidence_validation = (
        _validate_evidence_paths(
            normalized_signal,
            market_packet,
            concrete_plan=concrete_plan,
            minimum_required=int(
                cfg.get(
                    "minimum_evidence_paths",
                    4,
                )
            ),
            errors=evidence_path_errors,
            warnings=warnings,
        )
    )

    # Preserve missing, invalid or imprecise evidence paths for
    # diagnostics without converting the trade into NO_TRADE.
    for evidence_issue in evidence_path_errors:
        _append_unique(
            warnings,
            (
                "Evidence path: "
                f"{evidence_issue}"
            ),
        )

    trade_validation: Dict[str, Any] = {}
    anchor_validation: Dict[str, Any] = {}
    historical_validation: Dict[str, Any] = {
        "enabled": False,
        "status": "NOT_APPLICABLE",
    }

    alignment_validation: Dict[str, Any] = {}
    order_market_validation: Dict[str, Any] = {}
    adverse_move_validation: Dict[str, Any] = {}

    tp1: Optional[float] = None

    decision_reason = _clean_text(
        normalized_signal.get(
            "decision_reason"
        ),
        1000,
    )

    if not decision_reason:
        _append_unique(
            errors,
            (
                "Every AI decision requires "
                "a concrete decision_reason."
            ),
        )

    if concrete_plan:
        trade_validation = (
            _validate_geometry_and_targets(
                side=side,
                entry=entry,
                stop_loss=stop_loss,
                tp2=tp2,
                tp1_r=required_tp1_r,
                packet=market_packet,
                config=cfg,
                errors=errors,
                warnings=warnings,
            )
        )

        tp1 = _safe_float(
            trade_validation.get("tp1")
        )

        requested_tp2 = _safe_float(
            trade_validation.get(
                "requested_tp2"
            ),
            tp2,
        ) or tp2

        tp2 = _safe_float(
            trade_validation.get("tp2"),
            tp2,
        ) or tp2

        # The backend owns the final capped TP2.
        normalized_signal["tp2"] = (
            _round_price(
                tp2
            )
        )

        normalized_signal[
            "requested_tp2"
        ] = _round_price(
            requested_tp2
        )

        normalized_signal[
            "tp2_was_capped"
        ] = bool(
            trade_validation.get(
                "tp2_was_capped"
            )
        )

        normalized_signal[
            "tp2_cap_reason"
        ] = _clean_text(
            trade_validation.get(
                "tp2_cap_reason"
            ),
            200,
        )

        normalized_signal[
            "effective_max_tp2_r"
        ] = trade_validation.get(
            "effective_max_tp2_r"
        )

        atr = _get_main_atr(
            market_packet
        )

        anchor_validation = {
            "entry": _validate_single_anchor(
                anchor_name="Entry",
                path=normalized_signal.get(
                    "entry_anchor_path"
                ),
                target_price=entry,
                packet=market_packet,
                atr=atr,
                config=cfg,
                errors=errors,
                warnings=warnings,
                expected_catalog_types=(
                    "MARKET_PRICE",
                    "CLOSED_PRICE",
                    "SUPPORT",
                    "RESISTANCE",
                    "SWING_HIGH",
                    "SWING_LOW",
                ),
            ),

            "stop_loss": _validate_single_anchor(
                anchor_name="Stop Loss",
                path=normalized_signal.get(
                    "sl_anchor_path"
                ),
                target_price=stop_loss,
                packet=market_packet,
                atr=atr,
                config=cfg,
                errors=errors,
                warnings=warnings,
                expected_catalog_types=(
                    (
                        "SUPPORT",
                        "SWING_LOW",
                    )
                    if side == "LONG"
                    else (
                        "RESISTANCE",
                        "SWING_HIGH",
                    )
                ),
            ),

            "tp2": _validate_single_anchor(
                anchor_name="TP2",
                path=normalized_signal.get(
                    "tp2_anchor_path"
                ),
                target_price=(
                    requested_tp2
                    if trade_validation.get(
                        "tp2_was_capped"
                    )
                    else tp2
                ),
                packet=market_packet,
                atr=atr,
                config=cfg,
                errors=errors,
                warnings=warnings,
                expected_catalog_types=(
                    (
                        "RESISTANCE",
                        "SWING_HIGH",
                    )
                    if side == "LONG"
                    else (
                        "SUPPORT",
                        "SWING_LOW",
                    )
                ),
            ),
        }

        order_market_validation = (
            _validate_order_and_market_state(
                side=side,
                entry=entry,
                stop_loss=stop_loss,
                tp2=tp2,
                order_type=order_type,
                packet=market_packet,
                config=cfg,
                errors=errors,
                warnings=warnings,
            )
        )

        current_price = _get_current_price(
            market_packet
        )

        chase_fraction = (
            _validate_entry_chase(
                side=side,
                entry=entry,
                tp1=tp1,
                current_price=current_price,
                order_type=order_type,
                execution_state=execution_state,
                maximum_fraction=float(
                    cfg.get(
                        (
                            "max_entry_chase_"
                            "fraction"
                        ),
                        0.70,
                    )
                ),
                errors=errors,
                warnings=warnings,
            )
        )

        trade_validation[
            "entry_chase_fraction"
        ] = chase_fraction

        alignment_validation = (
            _validate_simple_timeframe_alignment(
                side=side,
                packet=market_packet,
                errors=errors,
                warnings=warnings,
            )
        )

        historical_validation = (
            _validate_historical_trade(
                analog_result=(
                    historical_analog_result
                ),
                side=side,
                entry=entry,
                stop_loss=stop_loss,
                tp2=tp2,
                config=cfg,
                errors=errors,
                warnings=warnings,
            )
        )

        adverse_move_validation = (
            _validate_expected_adverse_move(
                analog_result=(
                    historical_analog_result
                ),
                risk_distance_atr=(
                    _safe_float(
                        trade_validation.get(
                            "risk_distance_atr"
                        )
                    )
                ),
                errors=errors,
                warnings=warnings,
            )
        )

        invalidation = _clean_text(
            normalized_signal.get(
                "invalidation_condition"
            ),
            600,
        )

        if (
            not invalidation
            or invalidation.upper() == "NONE"
        ):
            _append_unique(
                errors,
                (
                    "Concrete trade plan requires "
                    "a clear invalidation_condition."
                ),
            )

        normalized_signal["tp1"] = (
            _round_price(
                tp1
            )
        )

        normalized_signal[
            "risk_distance"
        ] = trade_validation.get(
            "risk_distance"
        )

        normalized_signal[
            "risk_distance_atr"
        ] = trade_validation.get(
            "risk_distance_atr"
        )

        normalized_signal[
            "tp2_r_multiple"
        ] = trade_validation.get(
            "tp2_r_multiple"
        )

    else:
        normalized_signal["tp1"] = 0.0
        normalized_signal[
            "risk_distance"
        ] = 0.0
        normalized_signal[
            "risk_distance_atr"
        ] = None
        normalized_signal[
            "tp2_r_multiple"
        ] = None
        normalized_signal[
            "requested_tp2"
        ] = 0.0

        normalized_signal[
            "tp2_was_capped"
        ] = False

        normalized_signal[
            "tp2_cap_reason"
        ] = ""

        normalized_signal[
            "effective_max_tp2_r"
        ] = None

    # Frontend text formatting must never invalidate an otherwise
    # structurally and mathematically valid trade.
    frontend_formatting_errors: list[str] = []

    frontend_validation = (
        _validate_frontend_text(
            normalized_signal,
            tp1=tp1,
            errors=frontend_formatting_errors,
            warnings=warnings,
        )
    )

    # Preserve frontend formatting findings for diagnostics,
    # but do not include them in the hard validation errors.
    for formatting_issue in frontend_formatting_errors:
        _append_unique(
            warnings,
            (
                "Frontend formatting: "
                f"{formatting_issue}"
            ),
        )

    normalized_signal["decision"] = decision
    normalized_signal["side"] = side
    normalized_signal[
        "execution_state"
    ] = execution_state
    normalized_signal[
        "order_type"
    ] = order_type

    normalized_signal["entry_price"] = (
        _round_price(entry)
        if entry > 0
        else 0.0
    )

    normalized_signal["stop_loss"] = (
        _round_price(stop_loss)
        if stop_loss > 0
        else 0.0
    )

    normalized_signal["tp2"] = (
        _round_price(tp2)
        if tp2 > 0
        else 0.0
    )

    normalized_signal["status_text"] = (
        frontend_validation.get(
            "status_text",
            "",
        )
    )

    normalized_signal["advice"] = (
        frontend_validation.get(
            "advice",
            "",
        )
    )

    valid = len(errors) == 0

    trade_publishable = (
        valid
        and decision in {
            "LONG",
            "SHORT",
        }
        and concrete_plan
    )

    watch_publishable = (
        valid
        and decision == "WAIT"
        and concrete_plan
    )

    return {
        "valid": valid,

        "needs_correction": (
            not valid
        ),

        "trade_publishable": (
            trade_publishable
        ),

        "watch_publishable": (
            watch_publishable
        ),

        "decision": decision,

        "validator_version": (
            VALIDATOR_VERSION
        ),

        "final_signal": (
            normalized_signal
        ),

        "errors": errors,

        "warnings": warnings,

        "trade": trade_validation,

        "decision_consistency": (
            consistency
        ),

        "evidence_validation": (
            evidence_validation
        ),

        "anchor_validation": (
            anchor_validation
        ),

        "order_market_validation": (
            order_market_validation
        ),

        "timeframe_alignment": (
            alignment_validation
        ),

        "historical_validation": (
            historical_validation
        ),

        "adverse_move_validation": (
            adverse_move_validation
        ),

        "frontend_validation": (
            frontend_validation
        ),
    }


# ==========================================================
# SAFE FINAL FALLBACK
# ==========================================================

def build_safe_no_trade_signal(
    *,
    reason: str,
    previous_signal: Optional[
        Mapping[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Used after:

    - Gemini request failure
    - Invalid AI response
    - First validation failure
    - Correction attempt failure

    The backend publishes no trade rather than allowing an
    invalid or unsupported signal.
    """

    previous = (
        previous_signal
        if isinstance(
            previous_signal,
            Mapping,
        )
        else {}
    )

    reason_text = (
        _clean_text(
            reason,
            500,
        )
        or "Signal could not pass deterministic validation."
    )

    evidence_paths = _normalize_paths(
        previous.get(
            "evidence_paths"
        )
    )

    counter_paths = _normalize_paths(
        previous.get(
            "counter_evidence_paths"
        )
    )

    return {
        "decision": "NO_TRADE",
        "side": "NONE",
        "execution_state": "NO_TRADE",
        "order_type": "NONE",

        "entry_price": 0.0,
        "stop_loss": 0.0,
        "tp1": 0.0,
        "tp1_r_multiple": TP1_R_MULTIPLE,
        "tp2": 0.0,

        "requested_tp2": 0.0,
        "tp2_was_capped": False,
        "tp2_cap_reason": "",
        "effective_max_tp2_r": None,

        "risk_distance": 0.0,
        "risk_distance_atr": None,
        "tp2_r_multiple": None,

        "setup_type": "NO_TRADE",
        "market_regime": (
            _clean_text(
                previous.get(
                    "market_regime"
                ),
                100,
            )
            or "UNCLEAR"
        ),

        "confidence": 0,
        "trade_quality": "NO_TRADE",
        "trap_risk": "HIGH",
        "sl_hit_risk": "HIGH",
        "tp2_quality": "NOT_APPLICABLE",
        "expected_path": "UNCLEAR",

        "timeframe_bias": (
            previous.get(
                "timeframe_bias"
            )
            if isinstance(
                previous.get(
                    "timeframe_bias"
                ),
                Mapping,
            )
            else {
                "lower": "UNCLEAR",
                "main": "UNCLEAR",
                "confirm": "UNCLEAR",
                "swing": "UNCLEAR",
                "macro": "UNCLEAR",
            }
        ),

        "entry_anchor_path": "NONE",
        "sl_anchor_path": "NONE",
        "tp2_anchor_path": "NONE",

        "evidence_paths": evidence_paths,
        "counter_evidence_paths": (
            counter_paths
        ),

        "reason_codes": [
            "DETERMINISTIC_VALIDATION_FAILED",
        ],

        "decision_reason": reason_text,

        "invalidation_condition": "NONE",

        "status_text": (
            "NO TRADE | Validation Failed"
        ),

        "advice": (
            "ACTION: STAY OUT. The proposed setup did not "
            "pass the required market-structure and risk checks."
        ),
    }