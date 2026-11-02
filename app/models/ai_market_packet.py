# app/models/ai_market_packet.py

"""
AI Market Packet Builder
========================

Gemini AI signal engine එකට යවන්න compact, structured සහ JSON-safe
market packet එකක් හදන module එක.

මේ module එක:

- API calls කරන්නේ නැහැ
- Signal generate කරන්නේ නැහැ
- Gemini call කරන්නේ නැහැ
- Closed candles පමණක් package කරනවා
- Existing quantitative results package කරනවා
- Historical Analog Engine result package කරනවා
- Frontend status/advice rules package කරනවා

Candles 1000ක් backend එකෙන් locally analyse කළත් Geminiට යවන්නේ
අවශ්‍ය recent candles සහ calculated summaries පමණයි.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from .market_context import get_nearest_sr

PACKET_VERSION = "1.4.0"
TRADE_RISK_POLICY_VERSION = "1.1.0"

# Global Gemini packet-size protection.
# Warning limit keeps normal prompts reasonably compact.
# Hard limit prevents oversized packets reaching Gemini.
PACKET_SIZE_WARNING_CHARACTERS = 80_000
PACKET_SIZE_HARD_LIMIT_CHARACTERS = 100_000

# Geminiට raw recent candles යවන ප්‍රමාණය.
# Historical Analog Engine එකට full available candles use වෙනවා.
DEFAULT_CANDLE_LIMITS: Dict[str, int] = {
    "lower": 48,
    "main": 80,
    "confirm": 56,
    "swing": 32,
    "macro": 20,
}

SUMMARY_CANDLE_LIMITS: Dict[str, int] = {
    "lower": 32,
    "main": 48,
    "confirm": 32,
    "swing": 20,
    "macro": 12,
}

# Every compact recent-candle row follows this exact order.
# Field names are sent once per timeframe instead of once per candle.
RECENT_CANDLE_FIELD_ORDER = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_percent",
    "body_percent_of_range",
    "upper_wick_percent_of_range",
    "lower_wick_percent_of_range",
    "range_atr",
)


ROLE_DESCRIPTIONS: Dict[str, str] = {
    "lower": (
        "Entry timing, confirmation and short-term trap detection"
    ),
    "main": (
        "Primary trade setup and execution structure"
    ),
    "confirm": (
        "Higher-timeframe direction and setup confirmation"
    ),
    "swing": (
        "Broader swing structure and major support/resistance context"
    ),
    "macro": (
        "Macro direction and long-horizon regime context"
    ),
}

# AI-visible trade-risk policy.
# The deterministic validator remains the final authority
# for calculations, TP2 capping and signal approval.
STOP_LOSS_MINIMUM_ATR = 1.30

STOP_LOSS_MAXIMUM_ATR_BY_TIMEFRAME: Dict[str, float] = {
    "15m": 2.80,
    "30m": 2.80,
    "1h": 3.50,
    "4h": 4.60,
    "1d": 5.80,
}


TP1_R_MULTIPLE = 1.50

TP2_MINIMUM_R = 2.00
TP2_PREFERRED_R = 3.00
TP2_MAXIMUM_R = 5.00

TP2_MAXIMUM_ATR_BY_TIMEFRAME: Dict[str, float] = {
    "15m": 5.00,
    "30m": 5.00,
    "1h": 7.00,
    "4h": 8.00,
    "1d": 10.00,
}


def _normalize_trade_policy_timeframe(
    value: Any,
) -> str:
    timeframe = str(
        value or ""
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


def _build_selected_trade_risk_policy(
    selected_timeframe: Any,
) -> Dict[str, Any]:
    normalized = (
        _normalize_trade_policy_timeframe(
            selected_timeframe
        )
    )

    supported = (
        normalized
        in STOP_LOSS_MAXIMUM_ATR_BY_TIMEFRAME
        and normalized
        in TP2_MAXIMUM_ATR_BY_TIMEFRAME
    )

    # Unknown timeframe එකක් ආවොත් strict 15m policy එක use කරනවා.
    policy_timeframe = (
        normalized
        if supported
        else "15m"
    )

    return {
        "requested_timeframe": (
            normalized or "UNKNOWN"
        ),

        "policy_timeframe": (
            policy_timeframe
        ),

        "used_strict_fallback_policy": (
            not supported
        ),

        # 1.30 ATR is only the floor.
        # The AI may choose any structural SL between
        # this floor and the timeframe maximum.
        "stop_loss_minimum_atr": (
            STOP_LOSS_MINIMUM_ATR
        ),

        "stop_loss_minimum_is_floor_not_fixed_distance": (
            True
        ),

        "stop_loss_maximum_atr": (
            STOP_LOSS_MAXIMUM_ATR_BY_TIMEFRAME[
                policy_timeframe
            ]
        ),

        "tp1_r_multiple": (
            TP1_R_MULTIPLE
        ),

        "tp2_minimum_r": (
            TP2_MINIMUM_R
        ),

        "tp2_preferred_r": (
            TP2_PREFERRED_R
        ),

        "tp2_below_preferred_action": (
            "ALLOW_ONLY_WITH_EXACT_STRUCTURE_ANCHOR"
        ),

        "tp2_maximum_r": (
            TP2_MAXIMUM_R
        ),

        "tp2_maximum_atr": (
            TP2_MAXIMUM_ATR_BY_TIMEFRAME[
                policy_timeframe
            ]
        ),

        "effective_tp2_maximum_rule": (
            "MIN_OF_GLOBAL_R_MAX_AND_TIMEFRAME_TP2_ATR_CAP"
        ),

        "tp2_above_effective_maximum_action": (
            "BACKEND_CAPS_DOWN_AND_CONTINUES_VALIDATION"
        ),

        "tp2_below_minimum_action": (
            "REJECT_FOR_AI_CORRECTION"
        ),
    }

# ==========================================================
# SAFE CONVERSION HELPERS
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
                .replace("%", "")
                .replace("$", "")
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


def _round_or_none(
    value: Any,
    digits: int = 8,
) -> Optional[float]:
    number = _safe_float(value)

    if number is None:
        return None

    return round(number, digits)


def _iso_timestamp(value: Any) -> Optional[str]:
    try:
        if value is None or pd.isna(value):
            return None

        return pd.Timestamp(value).isoformat()

    except Exception:
        return str(value) if value is not None else None


def _json_safe(value: Any) -> Any:
    """
    Pandas, numpy සහ project objects JSON-safe values වලට convert කරනවා.
    """

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value

        return None

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, (pd.Timestamp, datetime)):
        return _iso_timestamp(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not str(key).startswith("_internal")
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, pd.Series):
        return [
            _json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.DataFrame):
        return [
            _json_safe(row)
            for row in value.to_dict(orient="records")
        ]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


# ==========================================================
# AI PACKET SECURITY / SIZE SANITIZER
# ==========================================================

SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "api_secret",
    "secret",
    "secret_key",
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "authorization",
    "password",
    "passwd",
    "private_key",
    "private_key_id",
    "credentials",
    "credential",
    "service_account",
    "service_account_key",
    "firebase_credentials",
    "gemini_api_key",
    "gemini_paid_key",
    "gemini_api_key_pool",
}


SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_api_secret",
    "_secret_key",
    "_client_secret",
    "_access_token",
    "_refresh_token",
    "_password",
    "_passwd",
    "_private_key",
    "_credentials",
)


DEBUG_ONLY_KEY_NAMES = {
    "raw_response",
    "raw_gemini_response",
    "raw_provider_response",
    "raw_text",
    "full_prompt",
    "prompt_text",
    "system_prompt",
    "system_instruction",
    "stack_trace",
    "stacktrace",
    "traceback",
    "debug_dump",
    "request_headers",
    "response_headers",
}

FORBIDDEN_PLANNER_KEY_NAMES = {
    "backend_setup_plan",
    "backend_selected_setup",
    "selected_setup",
    "primary_setup",
    "setup_candidates",
    "candidate_setups",
    "ranked_candidates",
    "backend_candidate",
    "backend_candidates",
    "planner_selected_setup",
}


_GOOGLE_API_KEY_PATTERN = re.compile(
    r"\bAIza[0-9A-Za-z_-]{25,}\b"
)

_BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"
)

_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z ]*PRIVATE KEY-----",
    flags=re.DOTALL,
)

_CONTROL_CHARACTER_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
)


def _normalize_security_key(
    key: Any,
) -> str:
    text = str(
        key or ""
    ).strip().lower()

    text = re.sub(
        r"[^a-z0-9_]+",
        "_",
        text,
    )

    return text.strip("_")


def _should_remove_packet_key(
    key: Any,
) -> bool:
    normalized = _normalize_security_key(
        key
    )

    if not normalized:
        return False

    if normalized.startswith(
        "_internal"
    ):
        return True

    if normalized in SENSITIVE_KEY_NAMES:
        return True

    if normalized in DEBUG_ONLY_KEY_NAMES:
        return True

    return any(
        normalized.endswith(suffix)
        for suffix in SENSITIVE_KEY_SUFFIXES
    )


def _sanitize_packet_string(
    value: str,
    *,
    path: str,
    report: Dict[str, Any],
    max_string_length: int,
) -> str:
    cleaned = _CONTROL_CHARACTER_PATTERN.sub(
        "",
        str(value or ""),
    )

    replaced = False

    if _PRIVATE_KEY_PATTERN.search(cleaned):
        cleaned = _PRIVATE_KEY_PATTERN.sub(
            "[REDACTED_PRIVATE_KEY]",
            cleaned,
        )

        replaced = True

    if _GOOGLE_API_KEY_PATTERN.search(cleaned):
        cleaned = _GOOGLE_API_KEY_PATTERN.sub(
            "[REDACTED_API_KEY]",
            cleaned,
        )

        replaced = True

    if _BEARER_TOKEN_PATTERN.search(cleaned):
        cleaned = _BEARER_TOKEN_PATTERN.sub(
            "Bearer [REDACTED_TOKEN]",
            cleaned,
        )

        replaced = True

    if replaced:
        report[
            "redacted_string_paths"
        ].append(path)

    if len(cleaned) > max_string_length:
        removed_characters = (
            len(cleaned)
            - max_string_length
        )

        cleaned = (
            cleaned[:max_string_length]
            + (
                "\n[TRUNCATED_BY_AI_PACKET_SANITIZER: "
                f"{removed_characters} CHARACTERS REMOVED]"
            )
        )

        report[
            "truncated_string_paths"
        ].append(path)

    return cleaned


def _sanitize_packet_value(
    value: Any,
    *,
    path: str,
    depth: int,
    report: Dict[str, Any],
    max_depth: int,
    max_string_length: int,
    max_list_items: int,
    max_dict_items: int,
) -> Any:
    if depth > max_depth:
        report[
            "depth_limited_paths"
        ].append(path)

        return {
            "truncated": True,
            "reason": (
                "MAX_PACKET_DEPTH_REACHED"
            ),
        }

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        if isinstance(value, str):
            return _sanitize_packet_string(
                value,
                path=path,
                report=report,
                max_string_length=(
                    max_string_length
                ),
            )

        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value

        return None

    if isinstance(value, np.generic):
        return _sanitize_packet_value(
            value.item(),
            path=path,
            depth=depth,
            report=report,
            max_depth=max_depth,
            max_string_length=(
                max_string_length
            ),
            max_list_items=max_list_items,
            max_dict_items=max_dict_items,
        )

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return _iso_timestamp(value)

    if isinstance(value, Mapping):
        sanitized_mapping: Dict[str, Any] = {}

        item_count = 0

        for key, item in value.items():
            key_text = str(key)

            child_path = (
                f"{path}.{key_text}"
                if path
                else key_text
            )

            if _should_remove_packet_key(
                key_text
            ):
                report[
                    "removed_field_paths"
                ].append(child_path)

                continue

            if item_count >= max_dict_items:
                report[
                    "dict_limited_paths"
                ].append(path)

                break

            sanitized_mapping[key_text] = (
                _sanitize_packet_value(
                    item,
                    path=child_path,
                    depth=depth + 1,
                    report=report,
                    max_depth=max_depth,
                    max_string_length=(
                        max_string_length
                    ),
                    max_list_items=(
                        max_list_items
                    ),
                    max_dict_items=(
                        max_dict_items
                    ),
                )
            )

            item_count += 1

        return sanitized_mapping

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        source_items = list(value)

        if len(source_items) > max_list_items:
            report[
                "list_limited_paths"
            ].append(path)

            source_items = source_items[
                :max_list_items
            ]

        return [
            _sanitize_packet_value(
                item,
                path=(
                    f"{path}.{index}"
                    if path
                    else str(index)
                ),
                depth=depth + 1,
                report=report,
                max_depth=max_depth,
                max_string_length=(
                    max_string_length
                ),
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )

            for index, item
            in enumerate(source_items)
        ]

    if isinstance(value, pd.Series):
        return _sanitize_packet_value(
            value.tolist(),
            path=path,
            depth=depth,
            report=report,
            max_depth=max_depth,
            max_string_length=(
                max_string_length
            ),
            max_list_items=max_list_items,
            max_dict_items=max_dict_items,
        )

    if isinstance(value, pd.DataFrame):
        return _sanitize_packet_value(
            value.to_dict(
                orient="records"
            ),
            path=path,
            depth=depth,
            report=report,
            max_depth=max_depth,
            max_string_length=(
                max_string_length
            ),
            max_list_items=max_list_items,
            max_dict_items=max_dict_items,
        )

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return _sanitize_packet_string(
        str(value),
        path=path,
        report=report,
        max_string_length=(
            max_string_length
        ),
    )

def _find_forbidden_planner_paths(
    value: Any,
    *,
    path: str = "",
) -> list[str]:
    """
    Detect backend planner-owned trade plans inside the AI packet.

    Gemini may receive quantitative evidence, but it must not
    receive a backend-selected Entry/SL/TP plan to copy.
    """

    found_paths: list[str] = []

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)

            normalized_key = (
                _normalize_security_key(
                    key_text
                )
            )

            child_path = (
                f"{path}.{key_text}"
                if path
                else key_text
            )

            if (
                normalized_key
                in FORBIDDEN_PLANNER_KEY_NAMES
            ):
                found_paths.append(
                    child_path
                )

            found_paths.extend(
                _find_forbidden_planner_paths(
                    item,
                    path=child_path,
                )
            )

    elif isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        for index, item in enumerate(value):
            child_path = (
                f"{path}.{index}"
                if path
                else str(index)
            )

            found_paths.extend(
                _find_forbidden_planner_paths(
                    item,
                    path=child_path,
                )
            )

    return found_paths


def sanitize_ai_market_packet(
    packet: Mapping[str, Any],
    *,
    include_report: bool = False,
    max_depth: int = 14,
    max_string_length: int = 50000,
    max_list_items: int = 500,
    max_dict_items: int = 500,
) -> Any:
    """
    Remove sensitive/debug fields and bound packet size before
    the packet is passed to Gemini.

    This does not change market prices or calculated evidence.
    """

    report: Dict[str, Any] = {
        "removed_field_paths": [],
        "redacted_string_paths": [],
        "truncated_string_paths": [],
        "depth_limited_paths": [],
        "list_limited_paths": [],
        "dict_limited_paths": [],
    }

    safe_source = _json_safe(
        packet or {}
    )

    sanitized = _sanitize_packet_value(
        safe_source,
        path="",
        depth=0,
        report=report,
        max_depth=max_depth,
        max_string_length=max_string_length,
        max_list_items=max_list_items,
        max_dict_items=max_dict_items,
    )

    if not isinstance(
        sanitized,
        dict,
    ):
        sanitized = {}

    summary = {
        "removed_sensitive_fields": len(
            report[
                "removed_field_paths"
            ]
        ),

        "redacted_strings": len(
            report[
                "redacted_string_paths"
            ]
        ),

        "truncated_strings": len(
            report[
                "truncated_string_paths"
            ]
        ),

        "depth_limited_nodes": len(
            report[
                "depth_limited_paths"
            ]
        ),

        "limited_lists": len(
            report[
                "list_limited_paths"
            ]
        ),

        "limited_dictionaries": len(
            report[
                "dict_limited_paths"
            ]
        ),
    }

    report["summary"] = summary

    if include_report:
        return sanitized, report

    return sanitized

def _compact_json_character_count(
    value: Any,
) -> int:
    """
    Return the actual compact JSON character count.

    Uses strict JSON rules so NaN/Infinity values cannot
    silently pass through the size-validation boundary.
    """

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

    return len(serialized)


def _build_packet_size_metadata(
    packet: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build stable size metadata for the final sanitized packet.

    The final count includes the packet_size metadata itself.
    """

    working = dict(
        packet or {}
    )

    metadata: Dict[str, Any] = {
        "estimated_json_characters": 0,
        "rough_token_estimate": 0,

        "warning_threshold_characters": (
            PACKET_SIZE_WARNING_CHARACTERS
        ),

        "hard_limit_characters": (
            PACKET_SIZE_HARD_LIMIT_CHARACTERS
        ),

        "warning_threshold_exceeded": False,
        "hard_limit_exceeded": False,
    }

    working["packet_size"] = metadata

    previous_size = -1

    # Adding the numeric metadata slightly changes the JSON
    # length, so recalculate until the value becomes stable.
    for _ in range(4):
        current_size = (
            _compact_json_character_count(
                working
            )
        )

        metadata.update(
            {
                "estimated_json_characters": (
                    current_size
                ),

                "rough_token_estimate": max(
                    1,
                    current_size // 4,
                ),

                "warning_threshold_exceeded": (
                    current_size
                    > PACKET_SIZE_WARNING_CHARACTERS
                ),

                "hard_limit_exceeded": (
                    current_size
                    > PACKET_SIZE_HARD_LIMIT_CHARACTERS
                ),
            }
        )

        if current_size == previous_size:
            break

        previous_size = current_size

    return metadata

# ==========================================================
# CANDLE DATA PREPARATION
# ==========================================================

def _find_column(
    df: pd.DataFrame,
    wanted: str,
) -> Optional[str]:
    wanted_lower = wanted.lower()

    for column in df.columns:
        if str(column).lower() == wanted_lower:
            return str(column)

    return None


def _prepare_candle_frame(
    df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    ElliotIQ DataFrame එක Gemini packet එකට suitable format එකට හදනවා.

    Date column එක index එකට move කරලා තිබුණත් preserve කරනවා.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.DataFrame()

    out = df.copy()

    date_column = _find_column(out, "Date")

    if date_column is None:
        if isinstance(out.index, pd.DatetimeIndex):
            out["Date"] = out.index

        elif str(out.index.name or "").lower() in {
            "date",
            "datetime",
            "time",
            "timestamp",
        }:
            out["Date"] = out.index

        else:
            out["Date"] = pd.RangeIndex(
                start=0,
                stop=len(out),
                step=1,
            )

    elif date_column != "Date":
        out.rename(
            columns={
                date_column: "Date",
            },
            inplace=True,
        )

    rename_map: Dict[str, str] = {}

    for canonical in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        column = _find_column(out, canonical)

        if column is not None and column != canonical:
            rename_map[column] = canonical

    if rename_map:
        out.rename(
            columns=rename_map,
            inplace=True,
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if any(
        column not in out.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    out = out.dropna(
        subset=required_columns,
    ).copy()

    if out.empty:
        return out

    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    out = out.drop_duplicates(
        subset=["Date"],
        keep="last",
    )

    try:
        out = out.sort_values("Date")
    except Exception:
        pass

    return out.reset_index(drop=True)


# ==========================================================
# ATR / CANDLE FEATURES
# ==========================================================

def _true_range(
    frame: pd.DataFrame,
) -> pd.Series:
    previous_close = frame["Close"].shift(1)

    parts = pd.concat(
        [
            frame["High"] - frame["Low"],
            (
                frame["High"] - previous_close
            ).abs(),
            (
                frame["Low"] - previous_close
            ).abs(),
        ],
        axis=1,
    )

    return parts.max(axis=1)


def _atr_series(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    true_range = _true_range(frame)

    return true_range.rolling(
        period,
        min_periods=max(
            3,
            period // 3,
        ),
    ).mean()


def _classify_last_candle(
    row: pd.Series,
) -> str:
    open_price = _safe_float(row.get("Open"))
    high_price = _safe_float(row.get("High"))
    low_price = _safe_float(row.get("Low"))
    close_price = _safe_float(row.get("Close"))

    if None in {
        open_price,
        high_price,
        low_price,
        close_price,
    }:
        return "UNKNOWN"

    full_range = max(
        high_price - low_price,
        1e-12,
    )

    body = abs(
        close_price - open_price
    )

    body_ratio = body / full_range

    upper_wick = (
        high_price
        - max(
            open_price,
            close_price,
        )
    )

    lower_wick = (
        min(
            open_price,
            close_price,
        )
        - low_price
    )

    if body_ratio <= 0.15:
        return "DOJI_OR_INDECISION"

    if (
        lower_wick / full_range >= 0.50
        and close_price >= open_price
    ):
        return "BULLISH_LOWER_WICK_REJECTION"

    if (
        upper_wick / full_range >= 0.50
        and close_price <= open_price
    ):
        return "BEARISH_UPPER_WICK_REJECTION"

    if (
        close_price > open_price
        and body_ratio >= 0.65
    ):
        return "STRONG_BULLISH_BODY"

    if (
        close_price < open_price
        and body_ratio >= 0.65
    ):
        return "STRONG_BEARISH_BODY"

    if close_price > open_price:
        return "BULLISH_CANDLE"

    if close_price < open_price:
        return "BEARISH_CANDLE"

    return "NEUTRAL_CANDLE"


def _linear_slope_atr(
    frame: pd.DataFrame,
    atr_value: Optional[float],
) -> float:
    if len(frame) < 4:
        return 0.0

    closes = frame["Close"].to_numpy(
        dtype=float,
    )

    x_axis = np.arange(
        len(closes),
        dtype=float,
    )

    try:
        slope = float(
            np.polyfit(
                x_axis,
                closes,
                1,
            )[0]
        )

    except Exception:
        return 0.0

    denominator = (
        atr_value
        if atr_value is not None
        and atr_value > 0
        else max(
            abs(closes[-1]),
            1e-12,
        )
    )

    return slope / denominator


# ==========================================================
# TIMEFRAME SUMMARY
# ==========================================================

def _build_frame_summary(
    frame: pd.DataFrame,
    sent_limit: int,
) -> Dict[str, Any]:
    if frame.empty:
        return {
            "available": False,
            "candles_available": 0,
            "candles_sent": 0,
        }

    recent = frame.tail(
        max(
            2,
            sent_limit,
        )
    ).copy()

    atr_values = _atr_series(frame)

    atr_value = (
        _safe_float(atr_values.iloc[-1])
        if not atr_values.empty
        else None
    )

    first_close = _safe_float(
        recent["Close"].iloc[0]
    )

    last_close = _safe_float(
        recent["Close"].iloc[-1]
    )

    recent_high = _safe_float(
        recent["High"].max()
    )

    recent_low = _safe_float(
        recent["Low"].min()
    )

    change_percent = None

    if (
        first_close not in (None, 0.0)
        and last_close is not None
    ):
        change_percent = (
            (
                last_close / first_close
            )
            - 1.0
        ) * 100.0

    atr_percent = None

    if (
        atr_value is not None
        and last_close not in (None, 0.0)
    ):
        atr_percent = (
            atr_value / last_close
        ) * 100.0

    position_in_range = None

    if (
        last_close is not None
        and recent_high is not None
        and recent_low is not None
        and recent_high > recent_low
    ):
        position_in_range = (
            last_close - recent_low
        ) / (
            recent_high - recent_low
        )

    volume_ratio = None

    if recent["Volume"].notna().any():
        volume_mean = _safe_float(
            recent["Volume"]
            .tail(20)
            .mean()
        )

        last_volume = _safe_float(
            recent["Volume"].iloc[-1]
        )

        if (
            volume_mean not in (None, 0.0)
            and last_volume is not None
        ):
            volume_ratio = (
                last_volume / volume_mean
            )

    slope_atr = _linear_slope_atr(
        recent,
        atr_value,
    )

    if slope_atr >= 0.08:
        short_term_direction = "BULLISH"

    elif slope_atr <= -0.08:
        short_term_direction = "BEARISH"

    else:
        short_term_direction = "SIDEWAYS"

    if atr_percent is None:
        volatility_state = "UNKNOWN"

    elif atr_percent >= 3.0:
        volatility_state = "EXTREME"

    elif atr_percent >= 1.5:
        volatility_state = "HIGH"

    elif atr_percent <= 0.35:
        volatility_state = "LOW"

    else:
        volatility_state = "NORMAL"

    return {
        "available": True,

        "candles_available": int(
            len(frame)
        ),

        "candles_sent": int(
            len(recent)
        ),

        "last_closed_time": _iso_timestamp(
            recent["Date"].iloc[-1]
        ),

        "last_closed_price": _round_or_none(
            last_close
        ),

        "window_change_percent": _round_or_none(
            change_percent,
            4,
        ),

        "atr14": _round_or_none(
            atr_value
        ),

        "atr14_percent": _round_or_none(
            atr_percent,
            4,
        ),

        "volatility_state": volatility_state,

        "short_term_direction": (
            short_term_direction
        ),

        "close_slope_atr_per_candle": (
            _round_or_none(
                slope_atr,
                5,
            )
        ),

        "recent_high": _round_or_none(
            recent_high
        ),

        "recent_low": _round_or_none(
            recent_low
        ),

        "close_position_in_recent_range": (
            _round_or_none(
                position_in_range,
                4,
            )
        ),

        "last_volume_vs_20_candle_average": (
            _round_or_none(
                volume_ratio,
                4,
            )
        ),

        "last_candle_type": (
            _classify_last_candle(
                recent.iloc[-1]
            )
        ),
    }


# ==========================================================
# RECENT CANDLES FOR GEMINI
# ==========================================================

def _build_recent_candles(
    frame: pd.DataFrame,
    limit: int,
) -> list[Dict[str, Any]]:
    if frame.empty:
        return []

    recent = frame.tail(
        max(
            1,
            int(limit),
        )
    ).copy()

    atr_values = _atr_series(frame)

    recent_atr = atr_values.reindex(
        recent.index
    )

    candles: list[Dict[str, Any]] = []

    previous_close: Optional[float] = None

    for index, row in recent.iterrows():
        open_price = _safe_float(
            row.get("Open")
        )

        high_price = _safe_float(
            row.get("High")
        )

        low_price = _safe_float(
            row.get("Low")
        )

        close_price = _safe_float(
            row.get("Close")
        )

        volume = _safe_float(
            row.get("Volume")
        )

        atr_value = _safe_float(
            recent_atr.get(index)
        )

        if None in {
            open_price,
            high_price,
            low_price,
            close_price,
        }:
            continue

        full_range = max(
            high_price - low_price,
            1e-12,
        )

        body = abs(
            close_price - open_price
        )

        upper_wick = (
            high_price
            - max(
                open_price,
                close_price,
            )
        )

        lower_wick = (
            min(
                open_price,
                close_price,
            )
            - low_price
        )

        return_percent = None

        if previous_close not in (
            None,
            0.0,
        ):
            return_percent = (
                (
                    close_price
                    / previous_close
                )
                - 1.0
            ) * 100.0

        candle = {
            "time": _iso_timestamp(
                row.get("Date")
            ),

            "open": _round_or_none(
                open_price
            ),

            "high": _round_or_none(
                high_price
            ),

            "low": _round_or_none(
                low_price
            ),

            "close": _round_or_none(
                close_price
            ),

            "volume": _round_or_none(
                volume,
                4,
            ),

            "return_percent": (
                _round_or_none(
                    return_percent,
                    5,
                )
            ),

            "body_percent_of_range": (
                _round_or_none(
                    (
                        body
                        / full_range
                    )
                    * 100.0,
                    2,
                )
            ),

            "upper_wick_percent_of_range": (
                _round_or_none(
                    (
                        upper_wick
                        / full_range
                    )
                    * 100.0,
                    2,
                )
            ),

            "lower_wick_percent_of_range": (
                _round_or_none(
                    (
                        lower_wick
                        / full_range
                    )
                    * 100.0,
                    2,
                )
            ),

            "range_atr": _round_or_none(
                (
                    full_range
                    / atr_value
                )
                if atr_value not in (
                    None,
                    0.0,
                )
                else None,
                4,
            ),
        }

        candles.append(candle)

        previous_close = close_price

    return candles

def _encode_recent_candle_rows(
    candles: Any,
) -> list[list[Any]]:
    """
    Preserve every candle value while removing
    repeated JSON field names.
    """

    if not isinstance(
        candles,
        (
            list,
            tuple,
        ),
    ):
        return []

    rows: list[list[Any]] = []

    for candle in candles:
        if not isinstance(
            candle,
            Mapping,
        ):
            continue

        rows.append([
            candle.get(field_name)
            for field_name
            in RECENT_CANDLE_FIELD_ORDER
        ])

    return rows

# ==========================================================
# PUBLIC MARKET PACKET BUILDER
# ==========================================================

# ==========================================================
# STRUCTURED MULTI-TIMEFRAME SUPPORT / RESISTANCE
# ==========================================================

def _compact_sr_zone(
    zone: Any,
    current_price: Any,
    atr: Any,
) -> Optional[Dict[str, Any]]:
    """Return factual numeric S/R evidence without strength labels."""

    if not isinstance(zone, Mapping):
        return None

    price = _safe_float(zone.get("price"))
    current = _safe_float(current_price)
    atr_value = _safe_float(atr)

    if price is None:
        return None

    distance = (
        abs(price - current)
        if current is not None
        else None
    )

    role = str(
        zone.get("role") or "UNKNOWN"
    ).upper()

    origin_role = str(
        zone.get("origin_role") or role
    ).upper()

    flip_status = str(
        zone.get("flip_status") or "NATIVE"
    ).upper()

    item: Dict[str, Any] = {
        "price": _round_or_none(price),
        "zone_min": _round_or_none(zone.get("min")),
        "zone_max": _round_or_none(zone.get("max")),
        "role": role,

        "touch_count": int(
            _safe_float(zone.get("strength")) or 0
        ),

        "numeric_score": _round_or_none(
            zone.get("score"),
            4,
        ),

        "distance_atr": _round_or_none(
            (
                distance / atr_value
                if (
                    distance is not None
                    and atr_value not in (None, 0.0)
                )
                else None
            ),
            4,
        ),

        "distance_pct": _round_or_none(
            (
                distance / abs(current) * 100.0
                if (
                    distance is not None
                    and current not in (None, 0.0)
                )
                else None
            ),
            4,
        ),
    }

    if origin_role != role:
        item["origin_role"] = origin_role

    if flip_status != "NATIVE":
        item.update({
            "flip_status": flip_status,
            "polarity_flipped": bool(
                zone.get("polarity_flipped")
            ),
            "potential_flip": bool(
                zone.get("potential_flip")
            ),
            "break_confirmed": bool(
                zone.get("break_confirmed")
            ),
            "retest_seen": bool(
                zone.get("retest_seen")
            ),
            "retest_held": bool(
                zone.get("retest_held")
            ),
        })

    return item


def _build_structured_sr_context(
    frame: pd.DataFrame,
    timeframe: str,
) -> Dict[str, Any]:
    """Build compact, side-neutral S/R evidence from closed candles."""

    if frame.empty or len(frame) < 20:
        return {
            "available": False,
            "lookback_candles": 120,
            "reason": "INSUFFICIENT_OR_INVALID_CANDLE_DATA",
        }

    try:
        raw = get_nearest_sr(
            frame,
            lookback=120,
            interval=timeframe,
        )

        if not isinstance(raw, Mapping):
            raise ValueError("Invalid S/R result")

        current = _safe_float(
            raw.get("current_price")
        )

        atr_value = _safe_float(
            raw.get("atr")
        )

        def compact_many(
            key: str,
        ) -> list[Dict[str, Any]]:
            values = raw.get(key)

            if not isinstance(
                values,
                (list, tuple),
            ):
                return []

            result = []

            for zone in values:
                item = _compact_sr_zone(
                    zone,
                    current,
                    atr_value,
                )

                if item is not None:
                    result.append(item)

                if len(result) >= 2:
                    break

            return result

        supports = compact_many(
            "support_zones"
        )

        resistances = compact_many(
            "resistance_zones"
        )

        return {
            "available": bool(
                current is not None
                and (
                    supports
                    or resistances
                )
            ),

            "lookback_candles": 120,

            "reason": (
                None
                if (
                    supports
                    or resistances
                )
                else "NO_VALID_SR_ZONES_FOUND"
            ),

            "current_price": _round_or_none(
                current
            ),

            "atr": _round_or_none(
                atr_value
            ),

            "recent_swing_high": _round_or_none(
                raw.get("recent_swing_high")
            ),

            "recent_swing_low": _round_or_none(
                raw.get("recent_swing_low")
            ),

            "support_zones_nearest_first": (
                supports
            ),

            "resistance_zones_nearest_first": (
                resistances
            ),

            "major_support": _compact_sr_zone(
                raw.get("major_support"),
                current,
                atr_value,
            ),

            "major_resistance": _compact_sr_zone(
                raw.get("major_resistance"),
                current,
                atr_value,
            ),
        }

    except Exception:
        return {
            "available": False,
            "lookback_candles": 120,
            "reason": "SR_CALCULATION_FAILED",
        }

def _build_evidence_catalog(
    *,
    packet_timeframes: Mapping[str, Any],
    selected_timeframe: str,
    current_price: Any,
    last_closed_price: Any,
) -> Dict[str, Any]:
    """
    Build compact, stable price-evidence IDs
    for Gemini citations.
    """

    items: Dict[str, Any] = {}

    def add_item(
        *,
        key: str,
        evidence_type: str,
        timeframe: str,
        price: Any,
        zone: Optional[Mapping[str, Any]] = None,
    ) -> None:
        numeric_price = _safe_float(price)

        if (
            numeric_price is None
            or numeric_price <= 0
        ):
            return

        catalog_id = key.upper()

        item: Dict[str, Any] = {
            "path": (
                "evidence_catalog.items."
                f"{catalog_id}.price"
            ),
            "type": evidence_type,
            "tf": timeframe,
            "price": _round_or_none(
                numeric_price
            ),
        }

        if isinstance(zone, Mapping):
            score = _round_or_none(
                zone.get("numeric_score"),
                4,
            )

            if score is not None:
                item["score"] = score

        items[catalog_id] = item

    selected_tf = str(
        selected_timeframe or ""
    ).strip()

    add_item(
        key="market_current_live",
        evidence_type="MARKET_PRICE",
        timeframe=selected_tf,
        price=current_price,
    )

    add_item(
        key="market_last_closed",
        evidence_type="CLOSED_PRICE",
        timeframe=selected_tf,
        price=last_closed_price,
    )

    role_zone_limits = {
        "lower": 1,
        "main": 2,
        "confirm": 1,
        "swing": 1,
        "macro": 1,
    }

    for role, zone_limit in (
        role_zone_limits.items()
    ):
        timeframe_block = (
            packet_timeframes.get(
                role,
                {},
            )
        )

        if not isinstance(
            timeframe_block,
            Mapping,
        ):
            continue

        timeframe = str(
            timeframe_block.get(
                "timeframe"
            )
            or ""
        ).strip()

        structured_sr = (
            timeframe_block.get(
                "structured_sr",
                {},
            )
        )

        if not isinstance(
            structured_sr,
            Mapping,
        ):
            continue

        # Lower/main swing levels are useful
        # for Entry and SL anchors.
        if role in {"lower", "main"}:
            add_item(
                key=f"{role}_swing_high",
                evidence_type="SWING_HIGH",
                timeframe=timeframe,
                price=structured_sr.get(
                    "recent_swing_high"
                ),
            )

            add_item(
                key=f"{role}_swing_low",
                evidence_type="SWING_LOW",
                timeframe=timeframe,
                price=structured_sr.get(
                    "recent_swing_low"
                ),
            )

        for evidence_type, list_key in (
            (
                "SUPPORT",
                "support_zones_nearest_first",
            ),
            (
                "RESISTANCE",
                "resistance_zones_nearest_first",
            ),
        ):
            zones = structured_sr.get(
                list_key,
                [],
            )

            if not isinstance(
                zones,
                (list, tuple),
            ):
                continue

            for index, zone in enumerate(
                zones[:zone_limit],
                start=1,
            ):
                if not isinstance(
                    zone,
                    Mapping,
                ):
                    continue

                key = (
                    f"{role}_"
                    f"{evidence_type.lower()}_"
                    f"{index}"
                )

                add_item(
                    key=key,
                    evidence_type=evidence_type,
                    timeframe=timeframe,
                    price=zone.get("price"),
                    zone=zone,
                )

    return {
        "catalog_version": "1.0",
        "advisory_evidence_only": True,
        "item_key_is_evidence_id": True,
        "select_existing_id_and_path_only": (
            True
        ),
        "item_count": len(items),
        "items": items,
    }

def build_ai_market_packet(
    *,
    symbol: str,
    selected_timeframe: str,
    timeframe_data: Mapping[
        str,
        Mapping[str, Any],
    ],
    quantitative_context: Optional[
        Mapping[str, Any]
    ] = None,
    historical_analog: Optional[
        Mapping[str, Any]
    ] = None,
    market_type: str = "futures",
    current_price: Any = None,
    last_closed_price: Any = None,
    analysis_timestamp: Any = None,
    previous_trade_context: Optional[
        Mapping[str, Any]
    ] = None,
    candle_limits: Optional[
        Mapping[str, int]
    ] = None,
) -> Dict[str, Any]:
    """
    Gemini signal engine එකට compact market packet එක හදනවා.

    timeframe_data format:

    {
        "lower": {
            "timeframe": "15m",
            "df": df_lower_closed,
        },
        "main": {
            "timeframe": "1h",
            "df": df_main_closed,
        },
        "confirm": {
            "timeframe": "4h",
            "df": df_confirm_closed,
        },
        "swing": {
            "timeframe": "1d",
            "df": df_swing_closed,
        },
        "macro": {
            "timeframe": "1w",
            "df": df_macro_closed,
        },
    }
    """

    limits: Dict[str, int] = dict(
        DEFAULT_CANDLE_LIMITS
    )

    if isinstance(candle_limits, Mapping):
        for role, value in candle_limits.items():
            try:
                limits[str(role)] = max(
                    1,
                    min(
                        int(value),
                        150,
                    ),
                )

            except Exception:
                continue

    packet_timeframes: Dict[str, Any] = {}

    missing_required_roles: list[str] = []
    missing_optional_roles: list[str] = []

    for role in [
        "lower",
        "main",
        "confirm",
        "swing",
        "macro",
    ]:
        spec = (
            timeframe_data.get(
                role,
                {},
            )
            if isinstance(
                timeframe_data,
                Mapping,
            )
            else {}
        )

        timeframe = str(
            spec.get("timeframe")
            or ""
        ).strip()

        frame = _prepare_candle_frame(
            spec.get("df")
        )

        limit = int(
            limits.get(
                role,
                DEFAULT_CANDLE_LIMITS[role],
            )
        )

        if frame.empty:
            if role in {
                "main",
                "confirm",
            }:
                missing_required_roles.append(
                    role
                )

            else:
                missing_optional_roles.append(
                    role
                )

        packet_timeframes[role] = {
            "role": role,

            "role_description": (
                ROLE_DESCRIPTIONS.get(
                    role,
                    role,
                )
            ),

            "timeframe": timeframe,

            "closed_candles_only": True,

            "summary": _build_frame_summary(
                frame,
                int(
                    SUMMARY_CANDLE_LIMITS.get(
                        role,
                        limit,
                    )
                ),
            ),

            "structured_sr": _build_structured_sr_context(
                frame,
                timeframe,
            ),

            "recent_closed_candle_fields": list(
                RECENT_CANDLE_FIELD_ORDER
            ),

            "recent_closed_candles": (
                _encode_recent_candle_rows(
                    _build_recent_candles(
                        frame,
                        limit,
                    )
                )
            ),
        }

    main_summary = (
        packet_timeframes
        .get(
            "main",
            {},
        )
        .get(
            "summary",
            {},
        )
    )

    resolved_last_closed = _safe_float(
        last_closed_price
    )

    if resolved_last_closed is None:
        resolved_last_closed = _safe_float(
            main_summary.get(
                "last_closed_price"
            )
        )

    resolved_timestamp = _iso_timestamp(
        analysis_timestamp
    )

    if resolved_timestamp is None:
        resolved_timestamp = (
            datetime
            .now(timezone.utc)
            .isoformat()
        )

    selected_trade_risk_policy = (
        _build_selected_trade_risk_policy(
            selected_timeframe
        )
    )

    market_packet: Dict[str, Any] = {
        "packet_version": PACKET_VERSION,

        "purpose": (
            "Provide complete closed-candle market evidence "
            "for an AI-first crypto futures signal decision. "
            "The packet does not contain a backend-selected "
            "trade that the model must copy."
        ),

        "market": {
            "symbol": str(
                symbol or ""
            ).upper(),

            "market_type": str(
                market_type or "futures"
            ),

            "selected_timeframe": str(
                selected_timeframe or ""
            ),

            "analysis_timestamp": (
                resolved_timestamp
            ),

            "current_live_price": (
                _round_or_none(
                    current_price
                )
            ),

            "last_closed_price": (
                _round_or_none(
                    resolved_last_closed
                )
            ),
        },

        "data_integrity": {
            "closed_candles_only": True,

            "open_candle_excluded_from_analysis": (
                True
            ),

            "missing_required_timeframe_roles": (
                missing_required_roles
            ),

            "missing_optional_timeframe_roles": (
                missing_optional_roles
            ),

            "all_required_timeframes_available": (
                len(
                    missing_required_roles
                )
                == 0
            ),

            "recent_candle_encoding": (
                "FIELD_ORDERED_ROWS"
            ),

            "recent_candle_field_order_path": (
                "timeframes.<role>.recent_closed_candle_fields"
            ),

            "recent_candle_rows_match_field_order": True,

            "do_not_invent_missing_data": True,
        },

        "timeframes": packet_timeframes,

        "evidence_catalog": _build_evidence_catalog(
            packet_timeframes=packet_timeframes,
            selected_timeframe=selected_timeframe,
            current_price=current_price,
            last_closed_price=(
                resolved_last_closed
            ),
        ),

        "evidence_usage_rules": {
            "stable_evidence_catalog": {
                "use_catalog_ids_and_paths_when_available": (
                    True
                ),
                "do_not_invent_catalog_ids_or_paths": (
                    True
                ),
                "catalog_is_evidence_not_trade_direction": (
                    True
                ),
                "entry_sl_tp2_must_still_be_numerically_supported": (
                    True
                ),
            },
            "structured_multi_timeframe_sr": {
                "advisory_evidence_only": True,
                "price_levels_and_numeric_facts_only": True,
                "do_not_treat_touch_count_or_score_as_certainty": True,
                "prefer_higher_timeframe_levels_when_timeframes_conflict": True,
                "sr_before_tp1_alone_must_not_reject_a_trade": True,
                "evaluate_level_quality_distance_and_market_structure_together": True,
                "do_not_invent_missing_levels": True,
            },
        },

        "quantitative_context": _json_safe(
            quantitative_context or {}
        ),

        "historical_analog_engine": _json_safe(
            historical_analog or {}
        ),

        "previous_trade_context": _json_safe(
            previous_trade_context or {}
        ),

        "non_negotiable_trade_rules": {
            "risk_policy_version": (
                TRADE_RISK_POLICY_VERSION
            ),

            "allowed_decisions": [
                "LONG",
                "SHORT",
                "WAIT",
                "NO_TRADE",
            ],

            # Selected timeframe එකට අදාළ exact limits.
            "selected_timeframe_risk_policy": (
                selected_trade_risk_policy
            ),

            "stop_loss": {
                "must_be_structural": True,

                "must_be_on_correct_side_of_entry": (
                    True
                ),

                # This is a minimum floor, not a fixed SL.
                "minimum_atr": (
                    STOP_LOSS_MINIMUM_ATR
                ),

                "minimum_is_floor_not_fixed_distance": (
                    True
                ),

                "maximum_atr_by_timeframe": dict(
                    STOP_LOSS_MAXIMUM_ATR_BY_TIMEFRAME
                ),

                "oversized_stop_to_force_trade_is_forbidden": (
                    True
                ),

                "artificially_tightening_stop_to_pass_rules_is_forbidden": (
                    True
                ),
            },

            "tp1": {
                "r_multiple": (
                    TP1_R_MULTIPLE
                ),

                "mandatory": True,

                "recalculated_and_owned_by_backend": (
                    True
                ),
            },

            "tp2": {
                "minimum_r": (
                    TP2_MINIMUM_R
                ),

                "preferred_r": (
                    TP2_PREFERRED_R
                ),

                "below_preferred_action": (
                    "ALLOW_ONLY_WITH_EXACT_STRUCTURE_ANCHOR"
                ),

                "maximum_r": (
                    TP2_MAXIMUM_R
                ),

                "maximum_atr_by_timeframe": dict(
                    TP2_MAXIMUM_ATR_BY_TIMEFRAME
                ),

                "effective_maximum_rule": (
                    "MIN_OF_GLOBAL_R_MAX_AND_TIMEFRAME_TP2_ATR_CAP"
                ),

                "must_be_beyond_tp1_in_trade_direction": (
                    True
                ),

                "must_reference_supplied_market_structure": (
                    True
                ),

                "below_minimum_r_is_invalid": True,

                # Example:
                # AI TP2 is above the effective maximum.
                # The effective maximum is the lower of 5R
                # and the selected-timeframe TP2 ATR cap.
                # The backend caps TP2 and continues validation.
                "above_effective_maximum_is_not_rejected_for_excess_alone": (
                    True
                ),

                "above_effective_maximum_action": (
                    "BACKEND_CAPS_DOWN_AND_CONTINUES_VALIDATION"
                ),
            },

            "unsupported_or_invented_price_levels_are_forbidden": (
                True
            ),

            "maximum_entry_chase_fraction_toward_tp1": (
                0.70
            ),

            "insufficient_or_conflicting_evidence_means": (
                "WAIT_OR_NO_TRADE"
            ),

            "historical_analogs_are_evidence_not_certainty": (
                True
            ),
        },

        "frontend_output_rules": {
            "status_text": {
                "one_short_line_only": True,

                "examples": [
                    "LONG | Entry Confirmed",
                    "SHORT | Entry Confirmed",
                    "WAIT | Retest Required",
                    "LIMIT ORDER | Fresh Setup",
                    "NO TRADE | Weak Setup",
                ],
            },

            "advice": {
                "maximum_sentences": 2,

                "compact_single_paragraph": True,

                "state_immediate_action_first": True,

                "must_be_specific_to_current_market_evidence": (
                    True
                ),

                "generic_advice_is_forbidden": True,

                "mention_entry_price_only_before_entry_when_useful": (
                    True
                ),

                "do_not_repeat_stop_loss_tp1_tp2_prices": (
                    True
                ),

                "no_markdown_no_bullets_no_headings": (
                    True
                ),
            },
        },
    }

    # --------------------------------------------------
    # Final packet security and size sanitization.
    # This runs after every market section has been built
    # and immediately before the packet can reach Gemini.
    # --------------------------------------------------
    sanitized_packet, sanitizer_report = (
        sanitize_ai_market_packet(
            market_packet,
            include_report=True,
        )
    )

    data_integrity = (
        sanitized_packet.setdefault(
            "data_integrity",
            {},
        )
    )

    if not isinstance(
        data_integrity,
        dict,
    ):
        data_integrity = {}

        sanitized_packet[
            "data_integrity"
        ] = data_integrity

    sanitizer_summary = (
        sanitizer_report.get(
            "summary",
            {},
        )
    )

    data_integrity.update(
        {
            "packet_sanitizer_applied": True,

            "removed_sensitive_field_count": (
                sanitizer_summary.get(
                    "removed_sensitive_fields",
                    0,
                )
            ),

            "redacted_string_count": (
                sanitizer_summary.get(
                    "redacted_strings",
                    0,
                )
            ),

            "truncated_string_count": (
                sanitizer_summary.get(
                    "truncated_strings",
                    0,
                )
            ),

            "depth_limited_node_count": (
                sanitizer_summary.get(
                    "depth_limited_nodes",
                    0,
                )
            ),

            "limited_list_count": (
                sanitizer_summary.get(
                    "limited_lists",
                    0,
                )
            ),

            "limited_dictionary_count": (
                sanitizer_summary.get(
                    "limited_dictionaries",
                    0,
                )
            ),
        }
    )

    # Packet size is calculated only after sanitization.
    # The hard limit is enforced again at the final JSON
    # serialization boundary before Gemini can receive it.
    try:
        sanitized_packet[
            "packet_size"
        ] = _build_packet_size_metadata(
            sanitized_packet
        )

    except Exception as exc:
        sanitized_packet[
            "packet_size"
        ] = {
            "estimated_json_characters": None,
            "rough_token_estimate": None,

            "warning_threshold_characters": (
                PACKET_SIZE_WARNING_CHARACTERS
            ),

            "hard_limit_characters": (
                PACKET_SIZE_HARD_LIMIT_CHARACTERS
            ),

            "warning_threshold_exceeded": None,
            "hard_limit_exceeded": None,

            "size_calculation_error": str(
                exc
            )[:500],
        }

    return sanitized_packet


# ==========================================================
# JSON SERIALIZATION
# ==========================================================

def market_packet_to_json(
    packet: Mapping[str, Any],
    *,
    pretty: bool = False,
) -> str:
    """
    Convert the market packet to JSON text for the Gemini prompt.

    Security rule:
    Always sanitize again at the final serialization boundary.
    This protects direct callers that did not use
    build_ai_market_packet().
    """

    safe_packet = sanitize_ai_market_packet(
        packet or {}
    )

    forbidden_planner_paths = (
        _find_forbidden_planner_paths(
            safe_packet
        )
    )

    if forbidden_planner_paths:
        raise ValueError(
            "Backend planner-owned setup data cannot be "
            "serialized into the AI market packet: "
            + ", ".join(
                forbidden_planner_paths[:10]
            )
        )

    compact_size = (
        _compact_json_character_count(
            safe_packet
        )
    )

    if (
        compact_size
        > PACKET_SIZE_HARD_LIMIT_CHARACTERS
    ):
        raise ValueError(
            "AI market packet exceeds the hard JSON-size "
            f"limit: {compact_size} > "
            f"{PACKET_SIZE_HARD_LIMIT_CHARACTERS} characters."
        )

    return json.dumps(
        safe_packet,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=(
            None
            if pretty
            else (",", ":")
        ),
    )


# ==========================================================
# PACKET VALIDATION
# ==========================================================

def validate_ai_market_packet(
    packet: Any,
) -> Dict[str, Any]:
    """
    Gemini call එකට කලින් market packet structure එක validate කරනවා.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(packet, Mapping):
        return {
            "valid": False,
            "errors": [
                "Market packet is not a mapping/object."
            ],
            "warnings": [],
        }

    # --------------------------------------------------
    # SECURITY / INTEGRITY VALIDATION
    # --------------------------------------------------
    data_integrity = packet.get(
        "data_integrity"
    )

    if not isinstance(
        data_integrity,
        Mapping,
    ):
        errors.append(
            "Missing data_integrity section."
        )

    else:
        if (
            data_integrity.get(
                "packet_sanitizer_applied"
            )
            is not True
        ):
            errors.append(
                "AI market packet sanitizer was not applied."
            )

        if (
            data_integrity.get(
                "closed_candles_only"
            )
            is not True
        ):
            errors.append(
                "Market packet is not marked as closed-candle-only."
            )

        if (
            data_integrity.get(
                "open_candle_excluded_from_analysis"
            )
            is not True
        ):
            errors.append(
                "Open candle exclusion is not confirmed."
            )

    packet_size = packet.get(
        "packet_size"
    )

    if not isinstance(
        packet_size,
        Mapping,
    ):
        warnings.append(
            "Packet size metadata is unavailable."
        )

    # Recalculate actual JSON size instead of trusting
    # caller-provided packet_size metadata.
    actual_packet_characters: Optional[int] = None

    try:
        actual_packet_characters = (
            _compact_json_character_count(
                packet
            )
        )

        if (
            actual_packet_characters
            > PACKET_SIZE_HARD_LIMIT_CHARACTERS
        ):
            errors.append(
                "AI market packet exceeds the hard JSON-size "
                f"limit: {actual_packet_characters} > "
                f"{PACKET_SIZE_HARD_LIMIT_CHARACTERS} characters."
            )

        elif (
            actual_packet_characters
            > PACKET_SIZE_WARNING_CHARACTERS
        ):
            warnings.append(
                "AI market packet is above the preferred "
                f"size threshold: {actual_packet_characters} > "
                f"{PACKET_SIZE_WARNING_CHARACTERS} characters."
            )

    except Exception as exc:
        errors.append(
            "Market packet is not safely JSON serializable: "
            f"{exc}"
        )

    # --------------------------------------------------
    # AI-FIRST ARCHITECTURE VALIDATION
    # --------------------------------------------------
    forbidden_planner_paths = (
        _find_forbidden_planner_paths(
            packet
        )
    )

    if forbidden_planner_paths:
        errors.append(
            "Backend planner-owned setup data leaked into "
            "the AI market packet: "
            + ", ".join(
                forbidden_planner_paths[:10]
            )
        )

    market = packet.get("market")

    if not isinstance(market, Mapping):
        errors.append(
            "Missing market section."
        )

    else:
        if not str(
            market.get("symbol")
            or ""
        ).strip():
            errors.append(
                "Missing market.symbol."
            )

        if not str(
            market.get(
                "selected_timeframe"
            )
            or ""
        ).strip():
            errors.append(
                "Missing market.selected_timeframe."
            )

    timeframes = packet.get("timeframes")

    if not isinstance(timeframes, Mapping):
        errors.append(
            "Missing timeframes section."
        )

    else:
        for required_role in [
            "main",
            "confirm",
        ]:
            role_data = timeframes.get(
                required_role
            )

            if not isinstance(
                role_data,
                Mapping,
            ):
                errors.append(
                    "Missing required timeframe role: "
                    f"{required_role}."
                )

                continue

            candles = role_data.get(
                "recent_closed_candles"
            )

            if (
                not isinstance(candles, list)
                or len(candles) < 20
            ):
                errors.append(
                    "Timeframe role "
                    f"{required_role} "
                    "has fewer than 20 closed candles."
                )

        for optional_role in [
            "lower",
            "swing",
            "macro",
        ]:
            role_data = timeframes.get(
                optional_role
            )

            if not isinstance(
                role_data,
                Mapping,
            ):
                warnings.append(
                    "Optional timeframe role "
                    f"{optional_role} is unavailable."
                )

    trade_rules = packet.get(
        "non_negotiable_trade_rules"
    )

    if not isinstance(
        trade_rules,
        Mapping,
    ):
        errors.append(
            "Missing non_negotiable_trade_rules section."
        )

    else:
        selected_policy = trade_rules.get(
            "selected_timeframe_risk_policy"
        )

        if not isinstance(
            selected_policy,
            Mapping,
        ):
            errors.append(
                "Missing selected timeframe risk policy."
            )

        else:
            expected_policy = (
                _build_selected_trade_risk_policy(
                    (
                        market.get(
                            "selected_timeframe"
                        )
                        if isinstance(
                            market,
                            Mapping,
                        )
                        else ""
                    )
                )
            )

            policy_fields = (
                "policy_timeframe",
                "stop_loss_minimum_atr",
                "stop_loss_maximum_atr",
                "tp1_r_multiple",
                "tp2_minimum_r",
                "tp2_preferred_r",
                "tp2_below_preferred_action",
                "tp2_maximum_r",
                "tp2_maximum_atr",
                "effective_tp2_maximum_rule",
                "tp2_above_effective_maximum_action",
            )

            for field in policy_fields:
                actual_value = (
                    selected_policy.get(
                        field
                    )
                )

                expected_value = (
                    expected_policy.get(
                        field
                    )
                )

                if isinstance(
                    expected_value,
                    float,
                ):
                    actual_number = _safe_float(
                        actual_value
                    )

                    if (
                        actual_number is None
                        or not math.isclose(
                            actual_number,
                            expected_value,
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                    ):
                        errors.append(
                            "Selected risk policy has an "
                            f"invalid {field} value."
                        )

                elif actual_value != expected_value:
                    errors.append(
                        "Selected risk policy has an "
                        f"invalid {field} value."
                    )

        stop_rules = trade_rules.get(
            "stop_loss"
        )

        tp1_rules = trade_rules.get(
            "tp1"
        )

        tp2_rules = trade_rules.get(
            "tp2"
        )

        if not isinstance(
            stop_rules,
            Mapping,
        ):
            errors.append(
                "Missing stop_loss trade rules."
            )

        elif (
            stop_rules.get(
                "minimum_is_floor_not_fixed_distance"
            )
            is not True
        ):
            errors.append(
                "Stop-loss minimum must be marked as "
                "a floor rather than a fixed distance."
            )

        if not isinstance(
            tp1_rules,
            Mapping,
        ):
            errors.append(
                "Missing TP1 trade rules."
            )

        else:
            packet_tp1_r = _safe_float(
                tp1_rules.get(
                    "r_multiple"
                )
            )

            if (
                packet_tp1_r is None
                or not math.isclose(
                    packet_tp1_r,
                    TP1_R_MULTIPLE,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                errors.append(
                    "TP1 trade rule must be exactly 1.5R."
                )

        if not isinstance(
            tp2_rules,
            Mapping,
        ):
            errors.append(
                "Missing TP2 trade rules."
            )

        else:
            tp2_minimum = _safe_float(
                tp2_rules.get(
                    "minimum_r"
                )
            )

            tp2_maximum = _safe_float(
                tp2_rules.get(
                    "maximum_r"
                )
            )

            if (
                tp2_minimum is None
                or not math.isclose(
                    tp2_minimum,
                    TP2_MINIMUM_R,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                errors.append(
                    f"TP2 minimum trade rule must be {TP2_MINIMUM_R:.2f}R."
                )

            if (
                tp2_maximum is None
                or not math.isclose(
                    tp2_maximum,
                    TP2_MAXIMUM_R,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                errors.append(
                    f"TP2 maximum trade rule must be {TP2_MAXIMUM_R:.2f}R."
                )

            if (
                tp2_rules.get(
                    "above_effective_maximum_action"
                )
                != (
                    "BACKEND_CAPS_DOWN_AND_"
                    "CONTINUES_VALIDATION"
                )
            ):
                errors.append(
                    "TP2 excess-target cap action is missing."
                )

    analog = packet.get(
        "historical_analog_engine"
    )

    if (
        not isinstance(analog, Mapping)
        or not analog
    ):
        warnings.append(
            "Historical analog evidence is unavailable or empty."
        )

    elif analog.get("available") is False:
        warnings.append(
            "Historical analog engine marked its result unavailable."
        )

    quant_context = packet.get(
        "quantitative_context"
    )

    if (
        not isinstance(
            quant_context,
            Mapping,
        )
        or not quant_context
    ):
        warnings.append(
            "Quantitative context is empty."
        )

    return {
        "valid": len(errors) == 0,

        "errors": errors,
        "warnings": warnings,

        "actual_json_characters": (
            actual_packet_characters
        ),

        "warning_threshold_characters": (
            PACKET_SIZE_WARNING_CHARACTERS
        ),

        "hard_limit_characters": (
            PACKET_SIZE_HARD_LIMIT_CHARACTERS
        ),
    }