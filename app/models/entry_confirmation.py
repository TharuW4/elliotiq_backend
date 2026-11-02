import math
from typing import Any, Dict, Optional

import pandas as pd


SUPPORTED_TIMEFRAMES = {
    "1h": "15m",
    "60m": "15m",
    "4h": "1h",
    "240m": "1h",
}

CONFIRMATION_SETTINGS = {
    # 1H setup -> scan recent 15m candles.
    # Confirmation should appear within about 2 hours after entry touch.
    "1h": {
        "lookback_candles": 48,
        "confirmation_window_candles": 8,
    },
    "60m": {
        "lookback_candles": 48,
        "confirmation_window_candles": 8,
    },

    # 4H setup -> scan recent 1H candles.
    # Confirmation should appear within about 6 hours after entry touch.
    "4h": {
        "lookback_candles": 36,
        "confirmation_window_candles": 6,
    },
    "240m": {
        "lookback_candles": 36,
        "confirmation_window_candles": 6,
    },
}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").strip()
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


def _base_signals() -> Dict[str, Any]:
    return {
        "close_reclaim": False,
        "wick_rejection": False,
        "ema_hold_or_reclaim": False,
        "invalid_close": False,
        "stop_breached": False,
        "momentum_close": False,
        "displacement_close": False,
        "large_opposite_body": False,

        # New lifecycle scan fields
        "touch_found_in_lookback": False,
        "touch_candle_index_from_end": None,
        "confirmation_found_in_window": False,
        "confirmation_candle_index_from_end": None,
        "candles_since_entry_touch": None,
        "candles_after_touch_to_confirmation": None,
        "confirmation_window_candles": None,
        "confirmation_window_expired": False,
        "lookback_candles": None,
        "entry_armed": False,
        "activation_candle_age": None,
        "required_approach_direction": "NONE",
        "post_activation_touch": False,
        "touch_approach_confirmed": False,
    }


def _base_result(
    *,
    enabled: bool,
    user_timeframe: str,
    confirmation_timeframe: str,
    entry_zone_touched: bool,
    status: str,
    score: int = 0,
    would_activate: bool = False,
    advice: str = "",
    reason: str = "",
    signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged_signals = _base_signals()
    if isinstance(signals, dict):
        merged_signals.update(signals)

    return {
        "enabled": bool(enabled),
        "user_timeframe": user_timeframe,
        "confirmation_timeframe": confirmation_timeframe,
        "entry_zone_touched": bool(entry_zone_touched),
        "confirmation_status": str(status or "UNKNOWN").upper(),
        "confirmation_score": max(0, min(100, int(score or 0))),
        "would_activate_by_confirmation": bool(would_activate),
        "advice": str(advice or ""),
        "reason": str(reason or ""),
        "signals": merged_signals,
    }


def _normalize_ohlc(lower_df) -> pd.DataFrame:
    """
    Normalize OHLC dataframe into columns:
    Open, High, Low, Close, Volume(optional)

    This function is intentionally defensive because this module is only
    advisory/shadow logic and must never break the main backend response.
    """
    try:
        if lower_df is None or not isinstance(lower_df, pd.DataFrame) or lower_df.empty:
            return pd.DataFrame()

        out = lower_df.copy()

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "date": "Date",
            "time": "Date",
            "timestamp": "Date",
        }

        out = out.rename(
            columns={
                c: rename_map.get(str(c).lower().strip(), c)
                for c in out.columns
            }
        )

        required = ["Open", "High", "Low", "Close"]
        if not set(required).issubset(set(out.columns)):
            return pd.DataFrame()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=required)

        if out.empty:
            return pd.DataFrame()

        return out

    except Exception:
        return pd.DataFrame()


def _atr(df, period: int = 14) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)
        if len(d) < period + 2:
            return None

        high = d["High"]
        low = d["Low"]
        close = d["Close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        val = tr.rolling(period).mean().iloc[-1]
        return _safe_float(val)

    except Exception:
        return None


def _ema(df, span: int) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)
        if len(d) < span or "Close" not in d.columns:
            return None

        return _safe_float(
            d["Close"].ewm(span=span, adjust=False).mean().iloc[-1]
        )

    except Exception:
        return None


def _tf_wait_text(confirmation_timeframe: str) -> str:
    if confirmation_timeframe == "15m":
        return "Wait for 15m confirmation."

    if confirmation_timeframe == "1h":
        return "Wait for 1H confirmation."

    return "Wait for lower-timeframe confirmation."


def _direction_text(side: str) -> str:
    side = str(side or "").upper().strip()
    if side == "LONG":
        return "bullish"
    if side == "SHORT":
        return "bearish"
    return "directional"


def _status_from_score(score: int) -> tuple[str, bool]:
    if score >= 65:
        return "CONFIRMED", True

    if score >= 45:
        return "WAITING", False

    return "WEAK", False


def _get_scan_settings(tf: str) -> dict:
    return CONFIRMATION_SETTINGS.get(
        tf,
        {
            "lookback_candles": 48,
            "confirmation_window_candles": 8,
        },
    )


def _row_ohlc(row) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    return (
        _safe_float(row.get("Open")),
        _safe_float(row.get("High")),
        _safe_float(row.get("Low")),
        _safe_float(row.get("Close")),
    )


def _find_touch_indices(
    *,
    d: pd.DataFrame,
    side: str,
    entry: float,
    tolerance: float,
    start_index: int = 0,
    required_approach_direction: str = "",
) -> list[int]:
    touch_indices: list[int] = []

    try:
        start = max(
            0,
            int(start_index or 0),
        )

        approach = str(
            required_approach_direction or ""
        ).upper().strip()

        for pos in range(
            start,
            len(d),
        ):
            row = d.iloc[pos]

            _, high, low, _ = (
                _row_ohlc(row)
            )

            if (
                high is None
                or low is None
            ):
                continue

            previous_close = None

            if pos > 0:
                previous_close = _safe_float(
                    d.iloc[
                        pos - 1
                    ].get("Close")
                )

            approach_confirmed = True

            if approach == "FROM_ABOVE":
                approach_confirmed = bool(
                    previous_close is not None
                    and previous_close > entry
                )

            elif approach == "FROM_BELOW":
                approach_confirmed = bool(
                    previous_close is not None
                    and previous_close < entry
                )

            if not approach_confirmed:
                continue

            if side == "LONG":
                if low <= entry + tolerance:
                    touch_indices.append(pos)

            elif side == "SHORT":
                if high >= entry - tolerance:
                    touch_indices.append(pos)

    except Exception:
        return []

    return touch_indices


def _score_confirmation_candle(
    *,
    row,
    side: str,
    entry: float,
    stop: Optional[float],
    current_price: Optional[float],
    atr_val: Optional[float],
    ema50_val: Optional[float],
    ema200_val: Optional[float],
    tolerance: float,
) -> tuple[int, Dict[str, Any], list[str], bool]:
    """
    Score one lower-timeframe candle as possible confirmation.

    Returns:
    - score
    - signals
    - reasons
    - invalidated
    """
    signals = _base_signals()
    reasons: list[str] = []
    score = 0

    open_price, high, low, close = _row_ohlc(row)

    if None in [open_price, high, low, close]:
        return 0, signals, ["candidate candle has invalid OHLC values"], False

    assert open_price is not None
    assert high is not None
    assert low is not None
    assert close is not None

    # Do not score an older candle with a later live price.
    current = close

    candle_range = max(abs(high - low), abs(entry) * 0.000001)
    body = abs(close - open_price)
    lower_wick = max(min(open_price, close) - low, 0.0)
    upper_wick = max(high - max(open_price, close), 0.0)

    body_is_large = body >= max(
        candle_range * 0.55,
        (atr_val or 0.0) * 0.30,
    )

    clearly_against_ema = False

    if side == "LONG":
        signals["close_reclaim"] = close > entry

        signals["wick_rejection"] = (
            low <= entry + tolerance
            and close >= entry
            and lower_wick >= max(body * 0.60, candle_range * 0.20)
        )

        signals["invalid_close"] = (
            stop is not None
            and close < stop
        )

        signals["stop_breached"] = (
            stop is not None
            and low <= stop
        )

        signals["momentum_close"] = (
            close > open_price
        )

        signals["displacement_close"] = bool(
            signals["momentum_close"]
            and body_is_large
        )

        signals["large_opposite_body"] = (
            open_price >= entry
            and close < entry
            and body_is_large
        )

        if ema50_val is not None and ema200_val is not None:
            clearly_against_ema = (
                close < ema50_val
                and close < ema200_val
                and current < ema50_val
                and current < ema200_val
            )
            signals["ema_hold_or_reclaim"] = not clearly_against_ema

        elif ema50_val is not None:
            clearly_against_ema = close < ema50_val and current < ema50_val
            signals["ema_hold_or_reclaim"] = not clearly_against_ema

        else:
            signals["ema_hold_or_reclaim"] = False

    elif side == "SHORT":
        signals["close_reclaim"] = close < entry

        signals["wick_rejection"] = (
            high >= entry - tolerance
            and close <= entry
            and upper_wick >= max(body * 0.60, candle_range * 0.20)
        )

        signals["invalid_close"] = (
            stop is not None
            and close > stop
        )

        signals["stop_breached"] = (
            stop is not None
            and high >= stop
        )

        signals["momentum_close"] = (
            close < open_price
        )

        signals["displacement_close"] = bool(
            signals["momentum_close"]
            and body_is_large
        )

        signals["large_opposite_body"] = (
            open_price <= entry
            and close > entry
            and body_is_large
        )

        if ema50_val is not None and ema200_val is not None:
            clearly_against_ema = (
                close > ema50_val
                and close > ema200_val
                and current > ema50_val
                and current > ema200_val
            )
            signals["ema_hold_or_reclaim"] = not clearly_against_ema

        elif ema50_val is not None:
            clearly_against_ema = close > ema50_val and current > ema50_val
            signals["ema_hold_or_reclaim"] = not clearly_against_ema

        else:
            signals["ema_hold_or_reclaim"] = False

    if signals["stop_breached"]:
        return 0, signals, [
            (
                "candidate candle traded through "
                "the planned stop/invalidation side"
            )
        ], True

    if signals["invalid_close"]:
        return 0, signals, [
            (
                "candidate candle closed beyond "
                "stop/invalidation side"
            )
        ], True

    if signals["close_reclaim"]:
        score += 30
        reasons.append("lower-timeframe candle reclaimed the entry side")
    else:
        score -= 5
        reasons.append("lower-timeframe candle has not reclaimed the entry side")

    if signals["wick_rejection"]:
        score += 20
        reasons.append("wick rejection is visible around the entry zone")

    # Not closing beyond stop/invalidation is a basic positive condition.
    score += 10
    reasons.append("close is not beyond stop/invalidation side")

    if signals["displacement_close"]:
        score += 15
        reasons.append(
            "lower-timeframe displacement candle "
            "closed in the trade direction"
        )

    elif signals["momentum_close"]:
        score += 7
        reasons.append(
            "candle body closed in the trade "
            "direction without strong displacement"
        )

    if ema50_val is not None or ema200_val is not None:
        if signals["ema_hold_or_reclaim"]:
            score += 10
            reasons.append("close is not clearly on the wrong side of available EMA context")
        elif clearly_against_ema:
            score -= 10
            reasons.append("close is clearly on the wrong side of available EMA context")

    if signals["large_opposite_body"]:
        score -= 20
        reasons.append("large opposite body pushed through the entry zone")

    score = max(1, min(100, int(score)))

    return score, signals, reasons, False


def _confirmation_status_advice(
    *,
    status: str,
    side: str,
    confirmation_tf: str,
    candles_since_touch: Optional[int],
    candles_after_touch_to_confirmation: Optional[int],
) -> str:
    direction_word = _direction_text(side)

    if status == "CONFIRMED":
        if candles_after_touch_to_confirmation is not None:
            return (
                f"Entry zone reaction was confirmed by {confirmation_tf} candle behavior "
                f"{candles_after_touch_to_confirmation} candle(s) after the entry touch. "
                f"The lower-timeframe action supports {direction_word} continuation. "
                "This is shadow confirmation only; live trade status is unchanged."
            )

        return (
            f"Entry zone reaction is confirmed by {confirmation_tf} candle behavior. "
            f"The lower-timeframe action supports {direction_word} continuation. "
            "This is shadow confirmation only; live trade status is unchanged."
        )

    if status == "WAITING":
        return (
            "Entry zone touched, but confirmation is not strong enough yet. "
            "Do not enter immediately. "
            f"{_tf_wait_text(confirmation_tf)}"
        )

    if status == "STALE":
        if candles_since_touch is not None:
            return (
                f"Entry zone was touched {candles_since_touch} lower-timeframe candle(s) ago, "
                "but no clean confirmation was found inside the valid confirmation window. "
                "This is no longer a fresh entry decision. Do not add more based on the old touch. "
                "Use active trade management or wait for a new reclaim/retest confirmation."
            )

        return (
            "Entry zone was touched earlier, but no clean confirmation was found inside the valid confirmation window. "
            "This is no longer a fresh entry decision. Do not add more based on the old touch. "
            "Use active trade management or wait for a new reclaim/retest confirmation."
        )

    if status == "INVALIDATED":
        return (
            "Entry zone was touched, but a lower-timeframe candle closed beyond the stop-loss side. "
            "Confirmation is invalidated. This is advisory only; live trade status is unchanged."
        )

    return (
        "Entry zone touched, but lower-timeframe reaction is weak. "
        "Avoid entering immediately unless a stronger reclaim/rejection confirmation appears. "
        f"{_tf_wait_text(confirmation_tf)}"
    )


def evaluate_entry_confirmation(
    *,
    side: str,
    entry_price: float,
    stop_loss: float | None,
    user_timeframe: str,
    lower_df,
    current_price: float | None = None,
    entry_zone_touched: bool = False,
    atr: float | None = None,
    ema50: float | None = None,
    ema200: float | None = None,
    activation_candle_age: int | None = None,
    require_post_activation_touch: bool = False,
    required_approach_direction: str = "",
) -> dict:
    """
    Shadow-only lower-timeframe entry confirmation.

    IMPORTANT:
    - This function does NOT activate a trade.
    - This function does NOT reject a trade.
    - This function does NOT cancel a trade.
    - This function does NOT modify entry, SL, TP1, or TP2.
    - It only returns advisory confirmation evidence for the response payload.

    Supported mapping:
    - 1H setup -> 15m confirmation
    - 4H setup -> 1H confirmation

    New behavior:
    - Does not only inspect the latest candle.
    - Scans recent lower-timeframe candles.
    - Finds the entry-touch candle.
    - Looks for confirmation within a limited post-touch window.
    - Returns STALE if touch happened too long ago without valid confirmation.
    """
    tf = _normalize_timeframe(user_timeframe)
    confirmation_tf = SUPPORTED_TIMEFRAMES.get(tf, "unknown")

    if tf not in SUPPORTED_TIMEFRAMES:
        return _base_result(
            enabled=False,
            user_timeframe=tf,
            confirmation_timeframe=confirmation_tf,
            entry_zone_touched=entry_zone_touched,
            status="NOT_APPLICABLE",
            advice="Entry confirmation shadow layer is only enabled for 1H and 4H signals.",
            reason=f"Unsupported user timeframe: {tf or 'unknown'}",
        )

    normalized_side = str(side or "").upper().strip()
    entry = _safe_float(entry_price)
    stop = _safe_float(stop_loss)

    if normalized_side not in ["LONG", "SHORT"] or entry is None:
        return _base_result(
            enabled=True,
            user_timeframe=tf,
            confirmation_timeframe=confirmation_tf,
            entry_zone_touched=entry_zone_touched,
            status="UNKNOWN",
            advice="Entry confirmation unavailable because setup side or entry price is missing.",
            reason="Missing or invalid side/entry.",
        )

    d_all = _normalize_ohlc(lower_df)
    if d_all.empty:
        return _base_result(
            enabled=True,
            user_timeframe=tf,
            confirmation_timeframe=confirmation_tf,
            entry_zone_touched=entry_zone_touched,
            status="UNKNOWN",
            advice="Entry confirmation unavailable because lower-timeframe candle data is missing.",
            reason=f"No usable {confirmation_tf} candles available.",
        )

    atr_val = _safe_float(atr) or _atr(d_all)
    ema50_val = _safe_float(ema50) or _ema(d_all, 50)
    ema200_val = _safe_float(ema200) or _ema(d_all, 200)

    # Adaptive tolerance around the planned entry zone.
    tolerance = max(abs(entry) * 0.0015, (atr_val or 0.0) * 0.15)

    settings = _get_scan_settings(tf)
    lookback_candles = int(settings.get("lookback_candles") or 48)
    confirmation_window_candles = int(settings.get("confirmation_window_candles") or 8)

    d = d_all.tail(
        max(
            lookback_candles,
            confirmation_window_candles + 5,
        )
    ).copy()

    # Use indicator values available at each historical candle.
    prev_close = d_all["Close"].shift(1)

    true_range_all = pd.concat(
        [
            d_all["High"] - d_all["Low"],
            (
                d_all["High"]
                - prev_close
            ).abs(),
            (
                d_all["Low"]
                - prev_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = (
        true_range_all
        .rolling(14)
        .mean()
        .reindex(d.index)
    )

    ema50_series = (
        d_all["Close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
        .reindex(d.index)
    )

    ema200_series = (
        d_all["Close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
        .reindex(d.index)
    )

    base_scan_signals = _base_signals()

    base_scan_signals[
        "lookback_candles"
    ] = lookback_candles

    base_scan_signals[
        "confirmation_window_candles"
    ] = confirmation_window_candles

    candle_signal_keys = {
        "close_reclaim",
        "wick_rejection",
        "ema_hold_or_reclaim",
        "invalid_close",
        "stop_breached",
        "momentum_close",
        "displacement_close",
        "large_opposite_body",
    }

    def _merge_candle_signals(
        lifecycle_signals: Dict[str, Any],
        candle_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged_signals = dict(
            lifecycle_signals
        )

        merged_signals.update({
            key: value
            for key, value in (
                candle_signals or {}
            ).items()
            if key in candle_signal_keys
        })

        return merged_signals

    activation_age = None

    try:
        if activation_candle_age is not None:
            activation_age = max(
                0,
                int(
                    activation_candle_age
                ),
            )
    except Exception:
        activation_age = None

    activation_idx = None
    touch_start_index = 0

    approach_direction = str(
        required_approach_direction or ""
    ).upper().strip()

    if approach_direction not in {
        "FROM_ABOVE",
        "FROM_BELOW",
    }:
        approach_direction = "NONE"

    if (
        require_post_activation_touch
        and activation_age is not None
        and activation_age < len(d)
    ):
        activation_idx = (
            len(d)
            - 1
            - activation_age
        )

        # Sweep/reclaim candle cannot confirm
        # its own later entry retest.
        touch_start_index = (
            activation_idx + 1
        )

        base_scan_signals[
            "entry_armed"
        ] = True

        base_scan_signals[
            "activation_candle_age"
        ] = activation_age

        base_scan_signals[
            "required_approach_direction"
        ] = approach_direction

    touch_indices = _find_touch_indices(
        d=d,
        side=normalized_side,
        entry=entry,
        tolerance=tolerance,
        start_index=touch_start_index,
        required_approach_direction=(
            approach_direction
            if require_post_activation_touch
            else ""
        ),
    )

    if not touch_indices:
        if (
            require_post_activation_touch
            and activation_idx is not None
        ):
            return _base_result(
                enabled=True,
                user_timeframe=tf,
                confirmation_timeframe=(
                    confirmation_tf
                ),
                entry_zone_touched=False,
                status="WAITING_RETEST",
                score=0,
                would_activate=False,
                advice=(
                    "The liquidity sweep and "
                    "closed-candle reclaim are confirmed. "
                    "The entry is armed only for a later "
                    "retest from the correct side; do not "
                    "treat the original sweep move as an "
                    "entry touch."
                ),
                reason=(
                    "No post-reclaim entry retest "
                    "approached from "
                    f"{approach_direction.replace('_', ' ').lower()} "
                    "has occurred yet."
                ),
                signals=base_scan_signals,
            )

        # If backend says touched but it is not in our recent lower-TF lookback,
        # do not keep saying "wait for confirmation" forever.
        if entry_zone_touched:
            base_scan_signals["touch_found_in_lookback"] = False
            base_scan_signals["confirmation_window_expired"] = True

            return _base_result(
                enabled=True,
                user_timeframe=tf,
                confirmation_timeframe=confirmation_tf,
                entry_zone_touched=True,
                status="STALE",
                score=0,
                would_activate=False,
                advice=(
                    "Backend has marked the entry as touched, but the touch is not visible in the recent "
                    f"{confirmation_tf} confirmation lookback. This is no longer a fresh entry decision. "
                    "Do not add more based on the old touch. Use active trade management or wait for a new confirmation."
                ),
                reason=(
                    "Entry touch was reported by backend, but no entry-touch candle was found inside the recent "
                    f"{lookback_candles}-candle confirmation lookback."
                ),
                signals=base_scan_signals,
            )

        return _base_result(
            enabled=True,
            user_timeframe=tf,
            confirmation_timeframe=confirmation_tf,
            entry_zone_touched=False,
            status="WAITING",
            score=0,
            would_activate=False,
            advice=f"Entry zone has not been touched yet. {_tf_wait_text(confirmation_tf)}",
            reason="No lower-timeframe candle has touched the entry zone inside the recent confirmation lookback.",
            signals=base_scan_signals,
        )

    # Use the most recent touch in the lookback window.
    # This prevents very old touches from controlling a fresh retest.
    touch_idx = touch_indices[-1]
    candles_since_touch = (len(d) - 1) - touch_idx

    base_scan_signals[
        "touch_found_in_lookback"
    ] = True

    base_scan_signals[
        "post_activation_touch"
    ] = bool(
        activation_idx is not None
        and touch_idx > activation_idx
    )

    base_scan_signals[
        "touch_approach_confirmed"
    ] = bool(
        not require_post_activation_touch
        or base_scan_signals[
            "post_activation_touch"
        ]
    )

    base_scan_signals[
        "touch_candle_index_from_end"
    ] = candles_since_touch
    base_scan_signals["candles_since_entry_touch"] = candles_since_touch

    # Candidate confirmation candles start at the touch candle and extend
    # only up to the confirmation window.
    window_end_idx = min(
        len(d) - 1,
        touch_idx + confirmation_window_candles,
    )

    candidate_range = range(touch_idx, window_end_idx + 1)

    best_score = 0
    best_signals: Dict[str, Any] = {}
    best_reasons: list[str] = []
    best_idx: Optional[int] = None

    invalidated_idx: Optional[int] = None
    invalidated_signals: Dict[str, Any] = {}
    invalidated_reasons: list[str] = []

    for idx in candidate_range:
        row = d.iloc[idx]

        candle_atr = _safe_float(
            atr_series.iloc[idx],
            atr_val,
        )

        candle_ema50 = _safe_float(
            ema50_series.iloc[idx],
            ema50_val,
        )

        candle_ema200 = _safe_float(
            ema200_series.iloc[idx],
            ema200_val,
        )

        (
            score,
            candle_signals,
            candle_reasons,
            invalidated,
        ) = _score_confirmation_candle(
            row=row,
            side=normalized_side,
            entry=entry,
            stop=stop,
            current_price=_safe_float(
                row.get("Close")
            ),
            atr_val=candle_atr,
            ema50_val=candle_ema50,
            ema200_val=candle_ema200,
            tolerance=tolerance,
        )

        if invalidated:
            invalidated_idx = idx
            invalidated_signals = candle_signals
            invalidated_reasons = candle_reasons
            break

        if score > best_score:
            best_score = score
            best_signals = candle_signals
            best_reasons = candle_reasons
            best_idx = idx

    if invalidated_idx is not None:
        candles_after_touch = invalidated_idx - touch_idx
        merged = _merge_candle_signals(
            base_scan_signals,
            invalidated_signals,
        )
        merged["confirmation_candle_index_from_end"] = (len(d) - 1) - invalidated_idx
        merged["candles_after_touch_to_confirmation"] = candles_after_touch
        merged["confirmation_window_expired"] = candles_since_touch > confirmation_window_candles

        return _base_result(
            enabled=True,
            user_timeframe=tf,
            confirmation_timeframe=confirmation_tf,
            entry_zone_touched=True,
            status="INVALIDATED",
            score=0,
            would_activate=False,
            advice=_confirmation_status_advice(
                status="INVALIDATED",
                side=normalized_side,
                confirmation_tf=confirmation_tf,
                candles_since_touch=candles_since_touch,
                candles_after_touch_to_confirmation=candles_after_touch,
            ),
            reason="; ".join(invalidated_reasons),
            signals=merged,
        )

    status, would_activate = _status_from_score(best_score)

    # If no confirmation happened within the valid window and the touch is old,
    # mark it STALE instead of WAITING forever.
    window_expired = candles_since_touch > confirmation_window_candles

    if status != "CONFIRMED" and window_expired:
        merged = _merge_candle_signals(
            base_scan_signals,
            best_signals,
        )
        merged["confirmation_window_expired"] = True

        if best_idx is not None:
            merged["confirmation_candle_index_from_end"] = (len(d) - 1) - best_idx
            merged["candles_after_touch_to_confirmation"] = best_idx - touch_idx

        return _base_result(
            enabled=True,
            user_timeframe=tf,
            confirmation_timeframe=confirmation_tf,
            entry_zone_touched=True,
            status="STALE",
            score=best_score,
            would_activate=False,
            advice=_confirmation_status_advice(
                status="STALE",
                side=normalized_side,
                confirmation_tf=confirmation_tf,
                candles_since_touch=candles_since_touch,
                candles_after_touch_to_confirmation=None,
            ),
            reason=(
                f"Entry touch is {candles_since_touch} lower-timeframe candle(s) old. "
                f"No valid confirmation score >= 65 was found within the first "
                f"{confirmation_window_candles} candle(s) after touch. "
                + ("; ".join(best_reasons) if best_reasons else "")
            ),
            signals=merged,
        )

    candles_after_touch_to_confirmation = None
    confirmation_index_from_end = None

    if best_idx is not None:
        candles_after_touch_to_confirmation = best_idx - touch_idx
        confirmation_index_from_end = (len(d) - 1) - best_idx

    merged = _merge_candle_signals(
        base_scan_signals,
        best_signals,
    )
    merged["confirmation_found_in_window"] = bool(status == "CONFIRMED")
    merged["confirmation_candle_index_from_end"] = confirmation_index_from_end
    merged["candles_after_touch_to_confirmation"] = candles_after_touch_to_confirmation
    merged["confirmation_window_expired"] = window_expired

    advice = _confirmation_status_advice(
        status=status,
        side=normalized_side,
        confirmation_tf=confirmation_tf,
        candles_since_touch=candles_since_touch,
        candles_after_touch_to_confirmation=candles_after_touch_to_confirmation,
    )

    reason_parts = []

    if status == "CONFIRMED":
        reason_parts.append(
            f"Entry touch was found {candles_since_touch} lower-timeframe candle(s) ago."
        )
        if candles_after_touch_to_confirmation is not None:
            reason_parts.append(
                f"Confirmation was found {candles_after_touch_to_confirmation} candle(s) after touch."
            )

    if best_reasons:
        reason_parts.extend(best_reasons)

    return _base_result(
        enabled=True,
        user_timeframe=tf,
        confirmation_timeframe=confirmation_tf,
        entry_zone_touched=True,
        status=status,
        score=best_score,
        would_activate=would_activate,
        advice=advice,
        reason="; ".join(reason_parts),
        signals=merged,
    )