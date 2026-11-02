# app/models/smc_context.py

"""
SMC Context Layer

Uses the downloaded smart-money-concepts repo as a context/risk layer.

Purpose:
- Add SMC / ICT-style context to existing backend scenarios.
- Detect and summarize:
    - FVG / imbalance
    - Swing highs/lows
    - BOS / CHoCH
    - Order Blocks
    - Liquidity pools
    - Previous high/low
- Does NOT change entry/SL/TP.
- Does NOT create signals.
- Does NOT decide LONG/SHORT by itself.
- Does NOT block signals in v1.

Expected repo structure:
project_root/
  smartmoneyconcepts/
    __init__.py
    smc.py
  app/
    models/
      smc_context.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ==========================================================
# Optional import from downloaded repo
# ==========================================================

try:
    from smartmoneyconcepts import smc
except Exception:
    smc = None


# Conservative advisory defaults. These are relevance limits,
# not trade-entry or rejection rules.
SMC_MAX_ZONE_AGE_BARS = 120
SMC_NEAR_ZONE_ATR = 2.0
SMC_MAX_ZONE_DISTANCE_ATR = 6.0


# ==========================================================
# Helpers
# ==========================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            text = (
                value.replace(",", "")
                .replace("$", "")
                .replace("%", "")
                .strip()
            )
            if not text:
                return default
            value = text

        v = float(value)

        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except Exception:
        return default


def _upper(value: Any) -> str:
    return str(value or "").upper().strip()


def _normalize_side(value: Any) -> str:
    text = _upper(value)

    if text in ["LONG", "BUY", "BULL", "BULLISH"] or "LONG" in text:
        return "LONG"

    if text in ["SHORT", "SELL", "BEAR", "BEARISH"] or "SHORT" in text:
        return "SHORT"

    return "NONE"


def _fmt_price(value: Any) -> str:
    v = _safe_float(value)
    if v is None:
        return ""

    av = abs(v)

    if av >= 1000:
        return f"{v:.2f}"
    if av >= 100:
        return f"{v:.2f}"
    if av >= 10:
        return f"{v:.3f}"
    if av >= 1:
        return f"{v:.4f}"
    if av >= 0.01:
        return f"{v:.5f}"
    if av >= 0.0001:
        return f"{v:.7f}"

    return f"{v:.10f}"


def _normalize_ohlc(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    smart-money-concepts repo expects lowercase ohlc columns:
      open, high, low, close, volume
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        out = df.copy()

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Date": "date",
            "Time": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "date": "date",
            "time": "date",
        }

        out = out.rename(columns={c: rename_map.get(str(c), str(c).lower()) for c in out.columns})

        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in out.columns:
                return pd.DataFrame()

        if "volume" not in out.columns:
            out["volume"] = 0.0

        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=required)

        return out

    except Exception:
        return pd.DataFrame()


def _scenario_side(scenario: Optional[Dict[str, Any]]) -> str:
    try:
        s = scenario or {}
        return _normalize_side(s.get("trade_type") or s.get("side"))
    except Exception:
        return "NONE"


def _scenario_levels(scenario: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    try:
        s = scenario or {}

        entry = (
            _safe_float(s.get("entry"))
            or _safe_float(s.get("entry_price"))
            or _safe_float(s.get("entry_zone"))
        )

        stop = (
            _safe_float(s.get("stop_loss"))
            or _safe_float(s.get("invalidation"))
            or _safe_float(s.get("sl"))
        )

        tp1 = None
        tp2 = None

        targets = s.get("targets")
        if isinstance(targets, list):
            if len(targets) >= 1:
                tp1 = _safe_float(targets[0])
            if len(targets) >= 2:
                tp2 = _safe_float(targets[1])

        tp1 = tp1 or _safe_float(s.get("tp1"))
        tp2 = tp2 or _safe_float(s.get("tp2")) or _safe_float(s.get("target"))

        return {
            "entry": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
        }

    except Exception:
        return {
            "entry": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
        }


def _latest_price(df: pd.DataFrame) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)
        if d.empty:
            return None
        return float(d.iloc[-1]["close"])
    except Exception:
        return None

def _calculate_atr(
    df: Optional[pd.DataFrame],
    period: int = 14,
) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)

        if d.empty or len(d) < period + 1:
            return None

        previous_close = d["close"].shift(1)

        true_range = pd.concat(
            [
                d["high"] - d["low"],
                (d["high"] - previous_close).abs(),
                (d["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = _safe_float(true_range.tail(period).mean())

        if atr is None or atr <= 0:
            return None

        return atr

    except Exception:
        return None


def _row_age_bars(
    df: pd.DataFrame,
    row: pd.Series,
) -> Optional[int]:
    try:
        position = int(df.index.get_indexer([row.name])[0])

        if position < 0:
            return None

        return max(0, (len(df) - 1) - position)

    except Exception:
        return None


def _zone_position(
    price: Optional[float],
    top: Optional[float],
    bottom: Optional[float],
) -> str:
    try:
        if price is None or top is None or bottom is None:
            return "UNKNOWN"

        low = min(top, bottom)
        high = max(top, bottom)

        if low <= price <= high:
            return "AT_PRICE"

        if high < price:
            return "BELOW_PRICE"

        return "ABOVE_PRICE"

    except Exception:
        return "UNKNOWN"


def _zone_distance_atr(
    price: Optional[float],
    top: Optional[float],
    bottom: Optional[float],
    atr: Optional[float],
) -> Optional[float]:
    try:
        if (
            price is None
            or top is None
            or bottom is None
            or atr is None
            or atr <= 0
        ):
            return None

        low = min(top, bottom)
        high = max(top, bottom)

        if low <= price <= high:
            raw_distance = 0.0

        elif price < low:
            raw_distance = low - price

        else:
            raw_distance = price - high

        return round(raw_distance / atr, 3)

    except Exception:
        return None


def _active_zone_candidates(
    df: Optional[pd.DataFrame],
    signal_column: str,
    consumed_column: str,
    price: Optional[float],
    atr: Optional[float],
) -> list:
    candidates = []

    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return candidates

        required = {
            signal_column,
            consumed_column,
            "Top",
            "Bottom",
        }

        if not required.issubset(set(df.columns)):
            return candidates

        valid_rows = df[df[signal_column].notna()]

        for _, row in valid_rows.iterrows():
            signal_value = _safe_float(row.get(signal_column))
            consumed_value = _safe_float(row.get(consumed_column))
            top = _safe_float(row.get("Top"))
            bottom = _safe_float(row.get("Bottom"))

            if signal_value is None or signal_value == 0:
                continue

            # Positive mitigation/sweep index means consumed.
            if consumed_value is None or consumed_value > 0:
                continue

            if top is None or bottom is None:
                continue

            age_bars = _row_age_bars(df, row)

            distance_atr = _zone_distance_atr(
                price,
                top,
                bottom,
                atr,
            )

            candidates.append({
                "row": row,
                "age_bars": age_bars,
                "distance_atr": distance_atr,
            })

        # Prefer non-stale zones first, then the nearest zone.
        candidates.sort(
            key=lambda item: (
                0
                if (
                    item["age_bars"] is not None
                    and item["age_bars"] <= SMC_MAX_ZONE_AGE_BARS
                )
                else 1,
                item["distance_atr"]
                if item["distance_atr"] is not None
                else float("inf"),
                item["age_bars"]
                if item["age_bars"] is not None
                else float("inf"),
            )
        )

        return candidates

    except Exception:
        return []


def _last_valid_row(df: Optional[pd.DataFrame], column: str) -> Optional[pd.Series]:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None

        if column not in df.columns:
            return None

        x = df[df[column].notna()]
        if x.empty:
            return None

        return x.iloc[-1]

    except Exception:
        return None

# ==========================================================
# SMC function wrappers
# ==========================================================

def _run_smc_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "available": False,
        "error": "",
        "fvg": None,
        "swing_highs_lows": None,
        "bos_choch": None,
        "ob": None,
        "liquidity": None,
        "previous_high_low": None,
    }

    try:
        if smc is None:
            result["error"] = "smartmoneyconcepts package is not importable."
            return result

        d = _normalize_ohlc(df)

        if d.empty or len(d) < 80:
            result["error"] = "Not enough candles for SMC context."
            return result

        # Keep only recent candles to avoid heavy computation.
        d = d.tail(240).copy()

        swing_length = 20

        swings = smc.swing_highs_lows(d, swing_length=swing_length)

        result["fvg"] = smc.fvg(d, join_consecutive=True)
        result["swing_highs_lows"] = swings
        result["bos_choch"] = smc.bos_choch(d, swings, close_break=True)
        result["ob"] = smc.ob(d, swings)
        result["liquidity"] = smc.liquidity(d, swings, range_percent=0.01)

        try:
            result["previous_high_low"] = smc.previous_high_low(d, time_frame="1D")
        except Exception:
            result["previous_high_low"] = None

        result["available"] = True
        return result

    except Exception as exc:
        result["error"] = str(exc)
        return result

def build_side_neutral_smc_evidence(
    *,
    df_selected: Optional[pd.DataFrame],
    interval: str = "",
) -> Dict[str, Any]:
    """
    Build compact pre-Gemini SMC facts
    without LONG/SHORT, Entry or SL.
    """

    result: Dict[str, Any] = {
        "version": "1.0.0",
        "available": False,
        "mode": "SIDE_NEUTRAL_ADVISORY",
        "decision_authority": "NONE",
        "can_block_trade": False,
        "can_change_prices": False,
        "timeframe": str(interval or ""),
        "reference_price": None,
        "structure": {},
        "fvg": {},
        "order_blocks": {},
        "liquidity": {},
        "reason": "",
    }

    try:
        d = _normalize_ohlc(
            df_selected
        )

        if d.empty or len(d) < 80:
            result["reason"] = (
                "INSUFFICIENT_CANDLES"
            )
            return result

        price = _latest_price(d)
        atr = _calculate_atr(d)

        result["reference_price"] = (
            _safe_float(price)
        )

        smc_data = _run_smc_indicators(
            d
        )

        if not smc_data.get("available"):
            result["reason"] = (
                smc_data.get("error")
                or "SMC_INDICATORS_UNAVAILABLE"
            )
            return result

        def clean_index(
            value: Any,
        ) -> Optional[int]:
            number = _safe_float(value)

            if (
                number is None
                or number < 0
            ):
                return None

            return int(number)

        def compact_zone_pair(
            frame: Optional[pd.DataFrame],
            *,
            signal_column: str,
            consumed_column: str,
        ) -> Dict[str, Any]:
            zones: Dict[str, Any] = {
                "bullish": None,
                "bearish": None,
            }

            candidates = (
                _active_zone_candidates(
                    frame,
                    signal_column=signal_column,
                    consumed_column=(
                        consumed_column
                    ),
                    price=price,
                    atr=atr,
                )
            )

            for candidate in candidates:
                row = candidate.get("row")

                if not isinstance(
                    row,
                    pd.Series,
                ):
                    continue

                signal_value = _safe_float(
                    row.get(signal_column)
                )

                top = _safe_float(
                    row.get("Top")
                )

                bottom = _safe_float(
                    row.get("Bottom")
                )

                age_bars = candidate.get(
                    "age_bars"
                )

                distance_atr = candidate.get(
                    "distance_atr"
                )

                if (
                    signal_value is None
                    or signal_value == 0
                    or top is None
                    or bottom is None
                ):
                    continue

                if (
                    age_bars is not None
                    and age_bars
                    > SMC_MAX_ZONE_AGE_BARS
                ):
                    continue

                if (
                    distance_atr is not None
                    and distance_atr
                    > SMC_MAX_ZONE_DISTANCE_ATR
                ):
                    continue

                direction = (
                    "bullish"
                    if signal_value > 0
                    else "bearish"
                )

                if zones[direction] is not None:
                    continue

                low = min(top, bottom)
                high = max(top, bottom)

                item: Dict[str, Any] = {
                    "top": top,
                    "bottom": bottom,

                    "age_bars": (
                        int(age_bars)
                        if age_bars is not None
                        else None
                    ),

                    "distance_atr": (
                        distance_atr
                    ),

                    "position": _zone_position(
                        price,
                        top,
                        bottom,
                    ),
                }

                if signal_column == "FVG":
                    fill_pct = None

                    if (
                        price is not None
                        and high > low
                    ):
                        if direction == "bullish":
                            fill_pct = (
                                (high - price)
                                / (high - low)
                                * 100.0
                            )

                        else:
                            fill_pct = (
                                (price - low)
                                / (high - low)
                                * 100.0
                            )

                        fill_pct = round(
                            max(
                                0.0,
                                min(
                                    100.0,
                                    fill_pct,
                                ),
                            ),
                            2,
                        )

                    item["fill_pct"] = (
                        fill_pct
                    )

                if signal_column == "OB":
                    strength = _safe_float(
                        row.get("Percentage")
                    )

                    if strength is not None:
                        item["strength_pct"] = (
                            round(
                                strength,
                                2,
                            )
                        )

                zones[direction] = item

                if all(
                    value is not None
                    for value in zones.values()
                ):
                    break

            return zones

        structure_raw = (
            _extract_bos_choch_context(
                smc_data.get("bos_choch"),
                "NONE",
            )
        )

        structure: Dict[str, Any] = {}

        if structure_raw.get(
            "has_structure_event"
        ):
            event_side = str(
                structure_raw.get(
                    "event_side"
                )
                or "NONE"
            ).upper()

            structure = {
                "event": structure_raw.get(
                    "event_type"
                ),

                "direction": (
                    "BULLISH"
                    if event_side == "LONG"
                    else (
                        "BEARISH"
                        if event_side == "SHORT"
                        else "NONE"
                    )
                ),

                "level": _safe_float(
                    structure_raw.get("level")
                ),

                "broken_index": clean_index(
                    structure_raw.get(
                        "broken_index"
                    )
                ),
            }

        liquidity: Dict[str, Any] = {
            "nearest_unswept_upside": None,
            "nearest_unswept_downside": None,
            "recent_sweep": None,
        }

        liquidity_frame = smc_data.get(
            "liquidity"
        )

        if (
            isinstance(
                liquidity_frame,
                pd.DataFrame,
            )
            and not liquidity_frame.empty
            and "Liquidity"
            in liquidity_frame.columns
        ):
            unswept_candidates = []
            swept_candidates = []

            valid_rows = liquidity_frame[
                liquidity_frame[
                    "Liquidity"
                ].notna()
            ]

            for _, row in (
                valid_rows.iterrows()
            ):
                liquidity_value = (
                    _safe_float(
                        row.get("Liquidity")
                    )
                )

                level = _safe_float(
                    row.get("Level")
                )

                swept_index = clean_index(
                    row.get("Swept")
                )

                if (
                    liquidity_value is None
                    or liquidity_value == 0
                    or level is None
                ):
                    continue

                pool_side = (
                    "UPSIDE"
                    if liquidity_value > 0
                    else "DOWNSIDE"
                )

                distance_atr = None

                if (
                    price is not None
                    and atr is not None
                    and atr > 0
                ):
                    distance_atr = round(
                        abs(level - price) / atr,
                        3,
                    )

                item = {
                    "side": pool_side,
                    "level": level,
                    "distance_atr": (
                        distance_atr
                    ),
                    "end_index": clean_index(
                        row.get("End")
                    ),
                }

                if (
                    swept_index is None
                    or swept_index <= 0
                ):
                    if (
                        distance_atr is None
                        or distance_atr
                        <= SMC_MAX_ZONE_DISTANCE_ATR
                    ):
                        unswept_candidates.append(
                            item
                        )

                else:
                    item["swept_index"] = (
                        swept_index
                    )

                    if (
                        0
                        <= swept_index
                        < len(d)
                    ):
                        item[
                            "sweep_age_bars"
                        ] = max(
                            0,
                            (len(d) - 1)
                            - swept_index,
                        )

                    swept_candidates.append(
                        item
                    )

            for (
                pool_side,
                output_key,
            ) in (
                (
                    "UPSIDE",
                    "nearest_unswept_upside",
                ),
                (
                    "DOWNSIDE",
                    "nearest_unswept_downside",
                ),
            ):
                matches = [
                    item
                    for item
                    in unswept_candidates
                    if item.get("side")
                    == pool_side
                ]

                matches.sort(
                    key=lambda item: (
                        item.get("distance_atr")
                        if item.get(
                            "distance_atr"
                        ) is not None
                        else float("inf")
                    )
                )

                if matches:
                    liquidity[
                        output_key
                    ] = matches[0]

            recent_sweeps = [
                item
                for item in swept_candidates
                if (
                    item.get(
                        "sweep_age_bars"
                    )
                    is not None
                    and item.get(
                        "sweep_age_bars"
                    ) <= 20
                )
            ]

            recent_sweeps.sort(
                key=lambda item: item.get(
                    "sweep_age_bars",
                    999999,
                )
            )

            if recent_sweeps:
                liquidity[
                    "recent_sweep"
                ] = recent_sweeps[0]

        result.update({
            "available": True,

            "structure": structure,

            "fvg": compact_zone_pair(
                smc_data.get("fvg"),
                signal_column="FVG",
                consumed_column=(
                    "MitigatedIndex"
                ),
            ),

            "order_blocks": (
                compact_zone_pair(
                    smc_data.get("ob"),
                    signal_column="OB",
                    consumed_column=(
                        "MitigatedIndex"
                    ),
                )
            ),

            "liquidity": liquidity,

            "reason": (
                "SIDE_NEUTRAL_SMC_FACTS_READY"
            ),
        })

        return result

    except Exception as exc:
        result["reason"] = str(
            exc
        )[:300]

        return result

# ==========================================================
# Context extraction
# ==========================================================

def _extract_fvg_context(
    fvg_df: Optional[pd.DataFrame],
    side: str,
    price: Optional[float],
    atr: Optional[float],
) -> Dict[str, Any]:
    out = {
        "has_recent_fvg": False,
        "active": False,
        "recent_fvg_side": "NONE",
        "top": None,
        "bottom": None,
        "mitigated_index": None,
        "reference_price": price,
        "age_bars": None,
        "distance_atr": None,
        "zone_position": "UNKNOWN",
        "relevance": "NONE",
        "alignment": "NEUTRAL",
        "risk_note": "",
    }

    try:
        candidates = _active_zone_candidates(
            fvg_df,
            signal_column="FVG",
            consumed_column="MitigatedIndex",
            price=price,
            atr=atr,
        )

        if not candidates:
            return out

        selected = candidates[0]
        row = selected["row"]

        age_bars = selected.get("age_bars")
        distance_atr = selected.get("distance_atr")

        fvg_value = _safe_float(row.get("FVG"))
        top = _safe_float(row.get("Top"))
        bottom = _safe_float(row.get("Bottom"))

        if (
            fvg_value is None
            or fvg_value == 0
            or top is None
            or bottom is None
        ):
            return out

        fvg_side = "LONG" if fvg_value > 0 else "SHORT"

        position = _zone_position(
            price,
            top,
            bottom,
        )

        alignment = "NEUTRAL"
        relevance = "UNKNOWN"

        if (
            age_bars is not None
            and age_bars > SMC_MAX_ZONE_AGE_BARS
        ):
            relevance = "STALE"

        elif distance_atr is None:
            relevance = "UNKNOWN"

        elif distance_atr <= SMC_NEAR_ZONE_ATR:
            relevance = "NEAR"

        elif distance_atr <= SMC_MAX_ZONE_DISTANCE_ATR:
            relevance = "RELEVANT"

        else:
            relevance = "FAR"

        can_influence = relevance in [
            "NEAR",
            "RELEVANT",
        ]

        supportive_location = (
            (
                side == "LONG"
                and fvg_side == "LONG"
                and position in ["BELOW_PRICE", "AT_PRICE"]
            )
            or
            (
                side == "SHORT"
                and fvg_side == "SHORT"
                and position in ["ABOVE_PRICE", "AT_PRICE"]
            )
        )

        conflicting_location = (
            (
                side == "LONG"
                and fvg_side == "SHORT"
                and position in ["ABOVE_PRICE", "AT_PRICE"]
            )
            or
            (
                side == "SHORT"
                and fvg_side == "LONG"
                and position in ["BELOW_PRICE", "AT_PRICE"]
            )
        )

        if can_influence and supportive_location:
            alignment = "SUPPORTIVE"

            note = (
                f"Active {fvg_side} FVG is "
                f"{relevance.lower()} and correctly positioned "
                f"for this {side} setup."
            )

        elif can_influence and conflicting_location:
            alignment = "CONFLICT"

            note = (
                f"Active opposing FVG is {relevance.lower()} "
                f"in the proposed {side} trade path. "
                "Treat it as confirmation risk, "
                "not an automatic rejection."
            )

        elif relevance == "STALE":
            note = (
                "An active FVG exists, but it is too old "
                "to influence the SMC score."
            )

        elif relevance == "FAR":
            note = (
                "An active FVG exists, but it is too far "
                "from the reference price to influence the SMC score."
            )

        else:
            note = (
                "Active FVG exists but its direction/location "
                "is neutral for this setup."
            )

        out.update({
            "has_recent_fvg": bool(
                age_bars is None
                or age_bars <= SMC_MAX_ZONE_AGE_BARS
            ),
            "active": True,
            "recent_fvg_side": fvg_side,
            "top": top,
            "bottom": bottom,
            "mitigated_index": row.get("MitigatedIndex"),
            "age_bars": age_bars,
            "distance_atr": distance_atr,
            "zone_position": position,
            "relevance": relevance,
            "alignment": alignment,
            "risk_note": note,
        })

        return out

    except Exception as exc:
        out["risk_note"] = f"FVG context error: {exc}"
        return out


def _extract_bos_choch_context(bos_df: Optional[pd.DataFrame], side: str) -> Dict[str, Any]:
    out = {
        "has_structure_event": False,
        "event_type": "NONE",
        "event_side": "NONE",
        "level": None,
        "broken_index": None,
        "alignment": "NEUTRAL",
        "risk_note": "",
    }

    try:
        if bos_df is None or not isinstance(bos_df, pd.DataFrame) or bos_df.empty:
            return out

        event_row = None
        event_type = "NONE"

        choch_row = _last_valid_row(bos_df, "CHOCH")
        bos_row = _last_valid_row(bos_df, "BOS")

        # Prefer the latest by index if both exist.
        if choch_row is not None and bos_row is not None:
            if choch_row.name >= bos_row.name:
                event_row = choch_row
                event_type = "CHOCH"
            else:
                event_row = bos_row
                event_type = "BOS"
        elif choch_row is not None:
            event_row = choch_row
            event_type = "CHOCH"
        elif bos_row is not None:
            event_row = bos_row
            event_type = "BOS"

        if event_row is None:
            return out

        val = _safe_float(event_row.get(event_type), 0)
        if val is None or val == 0:
            return out

        event_side = "LONG" if val > 0 else "SHORT"
        level = _safe_float(event_row.get("Level"))
        broken_index = event_row.get("BrokenIndex")

        alignment = "NEUTRAL"
        if side in ["LONG", "SHORT"]:
            alignment = "SUPPORTIVE" if event_side == side else "CONFLICT"

        if event_type == "CHOCH" and alignment == "CONFLICT":
            note = f"SMC CHoCH conflicts with {side}; direction quality should be treated as risky."
        elif alignment == "SUPPORTIVE":
            note = f"SMC {event_type} supports {side} structure."
        else:
            note = f"SMC {event_type} is neutral/mixed for this setup."

        out.update({
            "has_structure_event": True,
            "event_type": event_type,
            "event_side": event_side,
            "level": level,
            "broken_index": broken_index,
            "alignment": alignment,
            "risk_note": note,
        })

        return out

    except Exception as exc:
        out["risk_note"] = f"BOS/CHoCH context error: {exc}"
        return out


def _extract_order_block_context(
    ob_df: Optional[pd.DataFrame],
    side: str,
    levels: Dict[str, Optional[float]],
    price: Optional[float],
    atr: Optional[float],
) -> Dict[str, Any]:
    out = {
        "has_order_block": False,
        "active": False,
        "ob_side": "NONE",
        "top": None,
        "bottom": None,
        "ob_volume": None,
        "percentage": None,
        "mitigated_index": None,
        "reference_price": None,
        "entry_inside_ob": False,
        "price_inside_ob": False,
        "age_bars": None,
        "distance_atr": None,
        "zone_position": "UNKNOWN",
        "relevance": "NONE",
        "alignment": "NEUTRAL",
        "risk_note": "",
    }

    try:
        entry = _safe_float(levels.get("entry"))

        reference_price = (
            entry
            if entry is not None
            else _safe_float(price)
        )

        candidates = _active_zone_candidates(
            ob_df,
            signal_column="OB",
            consumed_column="MitigatedIndex",
            price=reference_price,
            atr=atr,
        )

        if not candidates:
            return out

        selected = candidates[0]
        row = selected["row"]

        age_bars = selected.get("age_bars")
        distance_atr = selected.get("distance_atr")

        ob_value = _safe_float(row.get("OB"))
        top = _safe_float(row.get("Top"))
        bottom = _safe_float(row.get("Bottom"))

        if (
            ob_value is None
            or ob_value == 0
            or top is None
            or bottom is None
        ):
            return out

        ob_side = "LONG" if ob_value > 0 else "SHORT"

        low = min(top, bottom)
        high = max(top, bottom)

        entry_inside = bool(
            entry is not None
            and low <= entry <= high
        )

        price_inside = bool(
            price is not None
            and low <= price <= high
        )

        position = _zone_position(
            reference_price,
            top,
            bottom,
        )

        if (
            age_bars is not None
            and age_bars > SMC_MAX_ZONE_AGE_BARS
        ):
            relevance = "STALE"

        elif distance_atr is None:
            relevance = "UNKNOWN"

        elif distance_atr <= SMC_NEAR_ZONE_ATR:
            relevance = "NEAR"

        elif distance_atr <= SMC_MAX_ZONE_DISTANCE_ATR:
            relevance = "RELEVANT"

        else:
            relevance = "FAR"

        can_influence = relevance in [
            "NEAR",
            "RELEVANT",
        ]

        supportive_location = (
            (
                side == "LONG"
                and ob_side == "LONG"
                and position in ["BELOW_PRICE", "AT_PRICE"]
            )
            or
            (
                side == "SHORT"
                and ob_side == "SHORT"
                and position in ["ABOVE_PRICE", "AT_PRICE"]
            )
        )

        conflicting_location = (
            (
                side == "LONG"
                and ob_side == "SHORT"
                and position in ["ABOVE_PRICE", "AT_PRICE"]
            )
            or
            (
                side == "SHORT"
                and ob_side == "LONG"
                and position in ["BELOW_PRICE", "AT_PRICE"]
            )
        )

        alignment = "NEUTRAL"

        if (
            can_influence
            and ob_side == side
            and entry_inside
        ):
            alignment = "SUPPORTIVE"

            note = (
                f"Proposed entry is inside an active "
                f"{side} order block."
            )

        elif (
            can_influence
            and supportive_location
            and distance_atr is not None
            and distance_atr <= SMC_NEAR_ZONE_ATR
        ):
            alignment = "SUPPORTIVE"

            note = (
                f"Active {side} order block is near and "
                "correctly positioned for the setup."
            )

        elif can_influence and supportive_location:
            alignment = "WATCH"

            note = (
                f"Active {side} order block exists, but entry "
                "is outside the zone. Use it as advisory "
                "location evidence only."
            )

        elif can_influence and conflicting_location:
            alignment = "CONFLICT"

            note = (
                f"Active opposing order block is in the proposed "
                f"{side} trade path. Require confirmation; "
                "do not reject solely because of this zone."
            )

        elif relevance == "STALE":
            note = (
                "An active order block exists, but it is too old "
                "to influence the SMC score."
            )

        elif relevance == "FAR":
            note = (
                "An active order block exists, but it is too far "
                "away to influence the SMC score."
            )

        else:
            note = (
                "Active order block exists but is neutral "
                "for this setup."
            )

        out.update({
            "has_order_block": True,
            "active": True,
            "ob_side": ob_side,
            "top": top,
            "bottom": bottom,
            "ob_volume": _safe_float(row.get("OBVolume")),
            "percentage": _safe_float(row.get("Percentage")),
            "mitigated_index": row.get("MitigatedIndex"),
            "reference_price": reference_price,
            "entry_inside_ob": entry_inside,
            "price_inside_ob": price_inside,
            "age_bars": age_bars,
            "distance_atr": distance_atr,
            "zone_position": position,
            "relevance": relevance,
            "alignment": alignment,
            "risk_note": note,
        })

        return out

    except Exception as exc:
        out["risk_note"] = f"Order block context error: {exc}"
        return out

def _extract_liquidity_context(
    liq_df: Optional[pd.DataFrame],
    side: str,
    levels: Dict[str, Optional[float]],
    price: Optional[float],
) -> Dict[str, Any]:
    out = {
        "has_liquidity": False,
        "liquidity_side": "NONE",
        "level": None,
        "end": None,
        "swept": None,
        "alignment": "NEUTRAL",
        "position_vs_sl": "NONE",
        "inside_sl_range": False,
        "target_liquidity": None,
        "risk_liquidity": None,
        "risk_note": "",
    }

    try:
        if liq_df is None or not isinstance(liq_df, pd.DataFrame) or liq_df.empty:
            return out

        if "Liquidity" not in liq_df.columns or "Swept" not in liq_df.columns:
            return out

        if side not in ["LONG", "SHORT"]:
            return out

        entry = _safe_float(levels.get("entry"))
        stop = _safe_float(levels.get("stop_loss"))
        reference_price = entry if entry is not None else _safe_float(price)

        if reference_price is None:
            return out

        target_candidates = []
        risk_candidates = []

        valid_rows = liq_df[liq_df["Liquidity"].notna()]

        for _, row in valid_rows.iterrows():
            liquidity_value = _safe_float(row.get("Liquidity"))
            swept_value = _safe_float(row.get("Swept"))
            level = _safe_float(row.get("Level"))

            if liquidity_value is None or liquidity_value == 0 or level is None:
                continue

            # smartmoneyconcepts uses zero for an unswept pool.
            if swept_value is None or swept_value > 0:
                continue

            liquidity_side = "UPSIDE" if liquidity_value > 0 else "DOWNSIDE"
            distance_abs = abs(level - reference_price)
            distance_pct = None

            if reference_price != 0:
                distance_pct = (distance_abs / abs(reference_price)) * 100.0

            item = {
                "liquidity_side": liquidity_side,
                "level": level,
                "end": row.get("End"),
                "swept": swept_value,
                "distance_pct": distance_pct,
                "_distance_abs": distance_abs,
            }

            if side == "LONG":
                if liquidity_side == "UPSIDE" and level >= reference_price:
                    target_candidates.append(item)

                elif liquidity_side == "DOWNSIDE" and level <= reference_price:
                    risk_candidates.append(item)

            elif side == "SHORT":
                if liquidity_side == "DOWNSIDE" and level <= reference_price:
                    target_candidates.append(item)

                elif liquidity_side == "UPSIDE" and level >= reference_price:
                    risk_candidates.append(item)

        target = (
            min(target_candidates, key=lambda x: x["_distance_abs"])
            if target_candidates
            else None
        )

        risk = (
            min(risk_candidates, key=lambda x: x["_distance_abs"])
            if risk_candidates
            else None
        )

        if target is None and risk is None:
            return out

        position_vs_sl = "NONE"
        inside_sl_range = False

        if risk is not None:
            risk_level = _safe_float(risk.get("level"))

            if stop is None or risk_level is None:
                position_vs_sl = "UNKNOWN"

            elif side == "LONG":
                if stop <= risk_level <= reference_price:
                    position_vs_sl = "INSIDE_SL_RANGE"
                    inside_sl_range = True

                elif risk_level < stop:
                    position_vs_sl = "BEYOND_SL"

                else:
                    position_vs_sl = "OUTSIDE_EXPECTED_SIDE"

            elif side == "SHORT":
                if reference_price <= risk_level <= stop:
                    position_vs_sl = "INSIDE_SL_RANGE"
                    inside_sl_range = True

                elif risk_level > stop:
                    position_vs_sl = "BEYOND_SL"

                else:
                    position_vs_sl = "OUTSIDE_EXPECTED_SIDE"

        notes = []
        alignment = "NEUTRAL"

        if target is not None:
            notes.append(
                f"Unswept {target['liquidity_side'].lower()} liquidity at "
                f"{_fmt_price(target['level'])} may act as a {side} objective."
            )

        if risk is not None and position_vs_sl == "BEYOND_SL":
            alignment = "RISK"

            notes.append(
                f"Opposing liquidity at {_fmt_price(risk['level'])} is beyond "
                "the proposed SL; price may invalidate the setup before sweeping it."
            )

        elif risk is not None and position_vs_sl == "INSIDE_SL_RANGE":
            # Liquidity inside Entry-SL is awareness only.
            # It must not automatically create HIGH risk.
            alignment = "WATCH"

            notes.append(
                f"Opposing liquidity at {_fmt_price(risk['level'])} is inside "
                "the Entry-SL range. Treat it as drawdown/sweep awareness only, "
                "not automatic high risk."
            )

        elif risk is not None:
            alignment = "WATCH"

            notes.append(
                f"Opposing liquidity at {_fmt_price(risk['level'])} exists, "
                "but its SL relationship is not confirmed. "
                "Treat it as advisory only."
            )

        elif target is not None:
            alignment = "TARGET"

        primary_candidates = [
            item for item in [target, risk]
            if item is not None
        ]

        primary = min(
            primary_candidates,
            key=lambda x: x["_distance_abs"],
        )

        clean_target = None

        if target is not None:
            clean_target = {
                key: value
                for key, value in target.items()
                if not key.startswith("_")
            }

        clean_risk = None

        if risk is not None:
            clean_risk = {
                key: value
                for key, value in risk.items()
                if not key.startswith("_")
            }

        out.update({
            "has_liquidity": True,
            "liquidity_side": primary.get("liquidity_side") or "NONE",
            "level": primary.get("level"),
            "end": primary.get("end"),
            "swept": primary.get("swept"),
            "alignment": alignment,
            "position_vs_sl": position_vs_sl,
            "inside_sl_range": bool(inside_sl_range),
            "target_liquidity": clean_target,
            "risk_liquidity": clean_risk,
            "risk_note": " ".join(notes).strip(),
        })

        return out

    except Exception as exc:
        out["risk_note"] = f"Liquidity context error: {exc}"
        return out


def _extract_previous_high_low_context(prev_df: Optional[pd.DataFrame], side: str, price: Optional[float]) -> Dict[str, Any]:
    out = {
        "available": False,
        "previous_high": None,
        "previous_low": None,
        "broken_high": None,
        "broken_low": None,
        "context": "NEUTRAL",
        "risk_note": "",
    }

    try:
        if prev_df is None or not isinstance(prev_df, pd.DataFrame) or prev_df.empty:
            return out

        row = prev_df.iloc[-1]

        prev_high = _safe_float(row.get("PreviousHigh"))
        prev_low = _safe_float(row.get("PreviousLow"))
        broken_high = row.get("BrokenHigh")
        broken_low = row.get("BrokenLow")

        note = ""
        context = "NEUTRAL"

        if side == "SHORT" and prev_low is not None:
            context = "SHORT_OBJECTIVE"
            note = f"Previous low {_fmt_price(prev_low)} may act as bearish objective."

        elif side == "LONG" and prev_high is not None:
            context = "LONG_OBJECTIVE"
            note = f"Previous high {_fmt_price(prev_high)} may act as bullish objective."

        out.update({
            "available": True,
            "previous_high": prev_high,
            "previous_low": prev_low,
            "broken_high": broken_high,
            "broken_low": broken_low,
            "context": context,
            "risk_note": note,
        })

        return out

    except Exception as exc:
        out["risk_note"] = f"Previous high/low context error: {exc}"
        return out


# ==========================================================
# Scoring / summary
# ==========================================================

def _combine_smc_bias(
    *,
    side: str,
    fvg: Dict[str, Any],
    structure: Dict[str, Any],
    ob: Dict[str, Any],
    liquidity: Dict[str, Any],
) -> Dict[str, Any]:
    score = 50
    support = []
    risks = []

    try:
        if side not in ["LONG", "SHORT"]:
            return {
                "bias": "NEUTRAL",
                "score": 0,
                "supporting_factors": [],
                "risk_factors": ["No tradable setup side."],
                "risk_level": "NONE",
            }

        for name, ctx in [
            ("FVG", fvg),
            ("Structure", structure),
            ("Order Block", ob),
        ]:
            alignment = _upper(ctx.get("alignment"))

            if alignment == "SUPPORTIVE":
                score += 10
                support.append(ctx.get("risk_note") or f"{name} supports {side}.")
            elif alignment == "CONFLICT":
                score -= 12
                risks.append(ctx.get("risk_note") or f"{name} conflicts with {side}.")

        liq_alignment = _upper(liquidity.get("alignment"))
        if liq_alignment == "TARGET":
            score += 6
            support.append(liquidity.get("risk_note") or "Liquidity target supports setup objective.")
        elif liq_alignment == "RISK":
            score -= 8
            risks.append(liquidity.get("risk_note") or "Liquidity sweep risk exists.")

        score = max(0, min(100, score))

        if score >= 70 and not risks:
            bias = f"SUPPORTS_{side}"
            risk_level = "LOW"
        elif score >= 62:
            bias = f"LEAN_{side}"
            risk_level = "MEDIUM"
        elif score <= 38:
            bias = "CONFLICT"
            risk_level = "HIGH"
        else:
            bias = "MIXED"
            risk_level = "MEDIUM"

        return {
            "bias": bias,
            "score": int(score),
            "supporting_factors": support,
            "risk_factors": risks,
            "risk_level": risk_level,
        }

    except Exception as exc:
        return {
            "bias": "ERROR",
            "score": 0,
            "supporting_factors": [],
            "risk_factors": [str(exc)],
            "risk_level": "UNKNOWN",
        }


# ==========================================================
# Public API
# ==========================================================

def evaluate_smc_context(
    *,
    df_selected: Optional[pd.DataFrame],
    scenario: Optional[Dict[str, Any]] = None,
    interval: str = "",
    symbol: str = "",
) -> Dict[str, Any]:
    result = {
        "enabled": True,
        "available": False,
        "mode": "SMC_CONTEXT_ONLY",
        "decision_authority": "ADVISORY_ONLY",
        "can_block_trade": False,
        "can_change_prices": False,
        "score_is_accuracy": False,
        "symbol": symbol,
        "interval": interval,
        "side": "NONE",
        "bias": "NEUTRAL",
        "score": 0,
        "risk_level": "NONE",
        "fvg_context": {},
        "structure_context": {},
        "order_block_context": {},
        "liquidity_context": {},
        "previous_high_low_context": {},
        "supporting_factors": [],
        "risk_factors": [],
        "advice": "SMC context unavailable.",
        "reason": "",
        "raw_error": "",
    }

    try:
        d = _normalize_ohlc(df_selected)
        side = _scenario_side(scenario)
        levels = _scenario_levels(scenario)
        price = _latest_price(d)
        atr = _calculate_atr(d)

        entry = _safe_float(levels.get("entry"))
        reference_price = entry if entry is not None else price

        result["side"] = side

        if d.empty:
            result.update({
                "enabled": False,
                "reason": "No selected timeframe candles available for SMC context.",
            })
            return result

        if side not in ["LONG", "SHORT"]:
            result.update({
                "available": False,
                "reason": "Scenario side is not tradable, so SMC setup alignment is skipped.",
            })
            return result

        smc_data = _run_smc_indicators(d)

        if not smc_data.get("available"):
            result.update({
                "available": False,
                "reason": "SMC indicators unavailable.",
                "raw_error": smc_data.get("error") or "",
            })
            return result

        fvg_context = _extract_fvg_context(
            smc_data.get("fvg"),
            side,
            reference_price,
            atr,
        )
        structure_context = _extract_bos_choch_context(smc_data.get("bos_choch"), side)
        order_block_context = _extract_order_block_context(
            smc_data.get("ob"),
            side,
            levels,
            price,
            atr,
        )
        liquidity_context = _extract_liquidity_context(
            smc_data.get("liquidity"),
            side,
            levels,
            price,
        )
        previous_context = _extract_previous_high_low_context(smc_data.get("previous_high_low"), side, price)

        combined = _combine_smc_bias(
            side=side,
            fvg=fvg_context,
            structure=structure_context,
            ob=order_block_context,
            liquidity=liquidity_context,
        )

        advice = "SMC context is mixed/neutral. Keep normal system rules."

        if combined["bias"] == f"SUPPORTS_{side}":
            advice = f"SMC context supports this {side} setup. Use it as confirmation, not as a standalone entry reason."
        elif combined["bias"] == f"LEAN_{side}":
            advice = f"SMC context leans toward this {side} setup, but normal entry confirmation is still required."
        elif combined["bias"] == "CONFLICT":
            advice = f"SMC context conflicts with this {side} setup. Wait for stronger confirmation or use reduced risk."
        elif combined["bias"] == "MIXED":
            advice = "SMC context is mixed. Do not increase risk based on SMC."

        result.update({
            "available": True,
            "bias": combined.get("bias"),
            "score": combined.get("score"),
            "risk_level": combined.get("risk_level"),
            "fvg_context": fvg_context,
            "structure_context": structure_context,
            "order_block_context": order_block_context,
            "liquidity_context": liquidity_context,
            "previous_high_low_context": previous_context,
            "supporting_factors": combined.get("supporting_factors") or [],
            "risk_factors": combined.get("risk_factors") or [],
            "advice": advice,
            "reason": "SMC context evaluated as confirmation/risk layer only.",
        })

        return result

    except Exception as exc:
        result.update({
            "enabled": False,
            "available": False,
            "bias": "ERROR",
            "risk_level": "UNKNOWN",
            "advice": "SMC context evaluation failed.",
            "reason": str(exc),
        })
        return result


def add_smc_context_to_scenario(
    *,
    scenario: Dict[str, Any],
    interval: str,
    df_selected: Optional[pd.DataFrame],
    symbol: str = "",
) -> Dict[str, Any]:
    try:
        if not isinstance(scenario, dict):
            return scenario

        smc_context = evaluate_smc_context(
            df_selected=df_selected,
            scenario=scenario,
            interval=interval,
            symbol=symbol,
        )

        scenario["smc_context"] = smc_context

        # Compact note only. Do not change prices or status.
        side = _scenario_side(scenario)

        if side in ["LONG", "SHORT"] and smc_context.get("available"):
            bias = _upper(smc_context.get("bias"))
            advice = str(smc_context.get("advice") or "").strip()

            note = ""

            if bias == f"SUPPORTS_{side}":
                note = f" SMC context supports this {side} setup."
            elif bias == f"LEAN_{side}":
                note = f" SMC context leans toward this {side} setup."
            elif bias == "CONFLICT":
                note = f" SMC context conflicts with this {side} setup; use confirmation/reduced risk."
            elif bias == "MIXED":
                note = " SMC context is mixed; keep normal risk."

            if note:
                old_summary = str(scenario.get("summary") or "").strip()
                old_advice = str(scenario.get("generated_advice") or "").strip()

                if old_summary and "SMC context" not in old_summary:
                    scenario["summary"] = old_summary + note
                elif not old_summary:
                    scenario["summary"] = note.strip()

                if old_advice and "SMC context" not in old_advice:
                    scenario["generated_advice"] = old_advice + " " + advice
                elif not old_advice and advice:
                    scenario["generated_advice"] = advice

        return scenario

    except Exception as exc:
        try:
            scenario["smc_context"] = {
                "enabled": False,
                "available": False,
                "bias": "ERROR",
                "advice": "SMC context attach failed.",
                "reason": str(exc),
            }
        except Exception:
            pass

        return scenario


def add_smc_context_to_ai_data(
    *,
    ai_data: Dict[str, Any],
    interval: str,
    df_selected: Optional[pd.DataFrame],
    symbol: str = "",
) -> Dict[str, Any]:
    try:
        if not isinstance(ai_data, dict):
            return ai_data

        scenarios = ai_data.get("trade_scenarios") or []
        if not scenarios:
            return ai_data

        updated = []

        for s in scenarios:
            if isinstance(s, dict):
                updated.append(
                    add_smc_context_to_scenario(
                        scenario=s,
                        interval=interval,
                        df_selected=df_selected,
                        symbol=symbol,
                    )
                )
            else:
                updated.append(s)

        ai_data["trade_scenarios"] = updated

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            ai_data["selected_scenario"] = add_smc_context_to_scenario(
                scenario=selected,
                interval=interval,
                df_selected=df_selected,
                symbol=symbol,
            )

        return ai_data

    except Exception as exc:
        try:
            scenarios = ai_data.get("trade_scenarios") or [{}]
            if scenarios and isinstance(scenarios[0], dict):
                scenarios[0]["smc_context"] = {
                    "enabled": False,
                    "available": False,
                    "bias": "ERROR",
                    "advice": "SMC context AI attach failed.",
                    "reason": str(exc),
                }
                ai_data["trade_scenarios"] = scenarios
        except Exception:
            pass

        return ai_data