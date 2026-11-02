# app/models/trade_guidance_manager.py

"""
Trade Guidance Manager

Purpose:
- Give lightweight guidance for an EXISTING locked/global signal.
- Designed for refresh/status flow, not fresh signal generation.
- Does NOT create new entry/SL/TP.
- Does NOT modify backend prices.
- Can be used before optional AI explanation layer.

Main outputs:
- WAIT_ENTRY
- HOLD
- TAKE_PARTIAL
- HOLD_TP2_WITH_TRAIL
- CAUTION
- REDUCE_OR_CLOSE
- CLOSE_CONFIRMED
- MISSED
- EXPIRED
- CANCELLED
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import pandas as pd


# ==========================================================
# Helpers
# ==========================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, (int, float)):
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v

        text = str(value).replace(",", "").replace("$", "").strip()
        if not text:
            return default

        v = float(text)
        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except Exception:
        return default


def _normalize_ohlc(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        out = df.copy()

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "date": "Date",
            "time": "Date",
        }

        out = out.rename(
            columns={c: rename_map.get(str(c).lower(), c) for c in out.columns}
        )

        required = ["Open", "High", "Low", "Close"]
        for col in required:
            if col not in out.columns:
                return pd.DataFrame()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=required)
        return out

    except Exception:
        return pd.DataFrame()


def _extract_price(value: Any) -> Optional[float]:
    """
    Works with:
    - plain floats
    - formatted strings like "0.1234"
    - entry_zone strings
    """
    try:
        if isinstance(value, (int, float)):
            return _safe_float(value)

        text = str(value or "").replace(",", "").strip()
        if not text:
            return None

        import re
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None

        return _safe_float(match.group(0))

    except Exception:
        return None


def _format_price(value: Any) -> str:
    v = _extract_price(value)
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


def _tf_key(interval: str) -> str:
    return str(interval or "").lower().strip()


def _max_pending_candles(interval: str) -> int:
    tf = _tf_key(interval)

    if tf in ["1m", "3m", "5m"]:
        return 36

    if tf in ["15m"]:
        return 24

    if tf in ["30m"]:
        return 24

    if tf in ["1h", "60m"]:
        return 24

    if tf in ["4h", "240m"]:
        return 30

    if tf in ["1d", "d", "daily"]:
        return 14

    return 24


def _get_targets(scenario: Dict[str, Any]) -> List[float]:
    targets = scenario.get("targets") or []

    if not targets and scenario.get("target") is not None:
        targets = [scenario.get("target")]

    out: List[float] = []

    for t in targets:
        v = _extract_price(t)
        if v is not None:
            out.append(float(v))

    return out


def _scenario_prices(scenario: Dict[str, Any]) -> Dict[str, Optional[float]]:
    entry = _extract_price(
        scenario.get("entry")
        or scenario.get("entry_zone")
    )

    stop = _extract_price(
        scenario.get("stop_loss")
        or scenario.get("invalidation")
    )

    targets = _get_targets(scenario)

    trade_type = str(
        scenario.get("trade_type")
        or scenario.get("side")
        or ""
    ).upper()

    if trade_type == "LONG":
        targets = sorted(targets)
    elif trade_type == "SHORT":
        targets = sorted(targets, reverse=True)

    tp1 = targets[0] if len(targets) >= 1 else None
    tp2 = targets[1] if len(targets) >= 2 else (targets[0] if targets else None)

    return {
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
    }


def _latest_price(df_live: Optional[pd.DataFrame]) -> Optional[float]:
    try:
        d = _normalize_ohlc(df_live)
        if d.empty:
            return None
        return float(d.iloc[-1]["Close"])
    except Exception:
        return None


def _calculate_pnl_pct(side: str, entry: Optional[float], current_price: Optional[float]) -> Optional[float]:
    try:
        if entry is None or current_price is None or entry <= 0:
            return None

        side = str(side or "").upper()

        if side == "LONG":
            return ((current_price - entry) / entry) * 100.0

        if side == "SHORT":
            return ((entry - current_price) / entry) * 100.0

    except Exception:
        return None

    return None


def _progress_to_tp1_pct(side: str, entry: Optional[float], tp1: Optional[float], current_price: Optional[float]) -> float:
    try:
        if entry is None or tp1 is None or current_price is None:
            return 0.0

        side = str(side or "").upper()

        if side == "LONG":
            total = tp1 - entry
            moved = current_price - entry
        elif side == "SHORT":
            total = entry - tp1
            moved = entry - current_price
        else:
            return 0.0

        if total <= 0:
            return 0.0

        return max(0.0, min(100.0, (moved / total) * 100.0))

    except Exception:
        return 0.0


def _progress_to_tp2_pct(side: str, entry: Optional[float], tp2: Optional[float], current_price: Optional[float]) -> float:
    try:
        if entry is None or tp2 is None or current_price is None:
            return 0.0

        side = str(side or "").upper()

        if side == "LONG":
            total = tp2 - entry
            moved = current_price - entry
        elif side == "SHORT":
            total = entry - tp2
            moved = entry - current_price
        else:
            return 0.0

        if total <= 0:
            return 0.0

        return max(0.0, min(100.0, (moved / total) * 100.0))

    except Exception:
        return 0.0


def _entry_touched(side: str, df_since_lock: pd.DataFrame, entry: float) -> bool:
    try:
        d = _normalize_ohlc(df_since_lock)
        if d.empty:
            return False

        side = str(side or "").upper()

        if side == "LONG":
            return bool((d["Low"] <= entry).any())

        if side == "SHORT":
            return bool((d["High"] >= entry).any())

    except Exception:
        pass

    return False


def _tp_hit(side: str, df_since_lock: pd.DataFrame, tp: Optional[float]) -> bool:
    try:
        if tp is None:
            return False

        d = _normalize_ohlc(df_since_lock)
        if d.empty:
            return False

        side = str(side or "").upper()

        if side == "LONG":
            return bool((d["High"] >= tp).any())

        if side == "SHORT":
            return bool((d["Low"] <= tp).any())

    except Exception:
        pass

    return False


def _sl_hit(side: str, df_since_lock: pd.DataFrame, stop: Optional[float]) -> bool:
    try:
        if stop is None:
            return False

        d = _normalize_ohlc(df_since_lock)
        if d.empty:
            return False

        side = str(side or "").upper()

        if side == "LONG":
            return bool((d["Low"] <= stop).any())

        if side == "SHORT":
            return bool((d["High"] >= stop).any())

    except Exception:
        pass

    return False


def _price_moved_70pct_to_tp1_before_entry(
    side: str,
    df_since_lock: pd.DataFrame,
    entry: Optional[float],
    tp1: Optional[float],
) -> bool:
    try:
        if entry is None or tp1 is None:
            return False

        d = _normalize_ohlc(df_since_lock)
        if d.empty:
            return False

        side = str(side or "").upper()

        if side == "LONG" and tp1 > entry:
            no_chase_level = entry + ((tp1 - entry) * 0.70)

            # If price reached no-chase before touching entry.
            touched_entry = False
            for row in d.itertuples():
                if float(row.Low) <= entry:
                    touched_entry = True
                    break
                if float(row.High) >= no_chase_level and not touched_entry:
                    return True

        if side == "SHORT" and tp1 < entry:
            no_chase_level = entry - ((entry - tp1) * 0.70)

            touched_entry = False
            for row in d.itertuples():
                if float(row.High) >= entry:
                    touched_entry = True
                    break
                if float(row.Low) <= no_chase_level and not touched_entry:
                    return True

    except Exception:
        pass

    return False


def _last_n_closes(df_live: Optional[pd.DataFrame], n: int = 5) -> List[float]:
    try:
        d = _normalize_ohlc(df_live)
        if d.empty:
            return []
        return [float(x) for x in d["Close"].tail(n).values]
    except Exception:
        return []


def _detect_return_to_entry_after_profit(
    side: str,
    entry: Optional[float],
    tp1: Optional[float],
    df_since_lock: pd.DataFrame,
) -> bool:
    """
    Detect case:
    - trade moved meaningfully toward TP1
    - then returned near entry
    This is not automatically bad. It is a caution flag only.
    """
    try:
        if entry is None or tp1 is None:
            return False

        d = _normalize_ohlc(df_since_lock)
        if len(d) < 3:
            return False

        side = str(side or "").upper()

        if side == "LONG":
            total = tp1 - entry
            if total <= 0:
                return False

            strong_progress_level = entry + (total * 0.60)
            returned_level = entry + (total * 0.15)

            reached_progress = bool((d["High"] >= strong_progress_level).any())
            returned = bool((d["Close"].tail(3) <= returned_level).any())

            return reached_progress and returned

        if side == "SHORT":
            total = entry - tp1
            if total <= 0:
                return False

            strong_progress_level = entry - (total * 0.60)
            returned_level = entry - (total * 0.15)

            reached_progress = bool((d["Low"] <= strong_progress_level).any())
            returned = bool((d["Close"].tail(3) >= returned_level).any())

            return reached_progress and returned

    except Exception:
        pass

    return False


def _detect_close_back_beyond_entry(
    side: str,
    entry: Optional[float],
    df_live: Optional[pd.DataFrame],
) -> bool:
    try:
        if entry is None:
            return False

        closes = _last_n_closes(df_live, n=3)
        if not closes:
            return False

        last_close = closes[-1]
        side = str(side or "").upper()

        tolerance = abs(entry) * 0.001

        if side == "LONG":
            return last_close < entry - tolerance

        if side == "SHORT":
            return last_close > entry + tolerance

    except Exception:
        pass

    return False


def _simple_opposite_pressure(
    side: str,
    df_live: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Lightweight candle pressure check.
    Not a full trend analysis.
    """
    out = {
        "opposite_pressure": False,
        "reason": "",
    }

    try:
        d = _normalize_ohlc(df_live)
        if len(d) < 5:
            return out

        recent = d.tail(3)

        red_count = 0
        green_count = 0

        for row in recent.itertuples():
            o = float(row.Open)
            c = float(row.Close)

            if c < o:
                red_count += 1
            elif c > o:
                green_count += 1

        side = str(side or "").upper()

        if side == "LONG" and red_count >= 2:
            out["opposite_pressure"] = True
            out["reason"] = "Recent candles show bearish pressure against the LONG."

        elif side == "SHORT" and green_count >= 2:
            out["opposite_pressure"] = True
            out["reason"] = "Recent candles show bullish pressure against the SHORT."

    except Exception:
        pass

    return out

def _closed_back_through_tp1(
    side: str,
    tp1: Optional[float],
    df_live: Optional[pd.DataFrame],
) -> bool:
    """
    After TP1 hit, detect whether price has closed back through TP1.

    LONG:
      close below TP1 = weakness after TP1

    SHORT:
      close above TP1 = weakness after TP1
    """
    try:
        if tp1 is None:
            return False

        d = _normalize_ohlc(df_live)
        if len(d) < 2:
            return False

        last_close = float(d.iloc[-1]["Close"])
        side = str(side or "").upper()

        tolerance = abs(tp1) * 0.001

        if side == "LONG":
            return last_close < tp1 - tolerance

        if side == "SHORT":
            return last_close > tp1 + tolerance

    except Exception:
        pass

    return False


def _confirm_tf_pressure(
    side: str,
    df_confirm: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    One higher timeframe pressure check.

    This is only a warning/support input.
    It should not force a close by itself.
    """
    out = {
        "confirm_opposite_pressure": False,
        "confirm_pressure_reason": "",
    }

    try:
        d = _normalize_ohlc(df_confirm)
        if len(d) < 5:
            return out

        pressure = _simple_opposite_pressure(side, d)

        if pressure.get("opposite_pressure"):
            out["confirm_opposite_pressure"] = True
            out["confirm_pressure_reason"] = (
                "Confirm timeframe also shows pressure against the trade. "
                + str(pressure.get("reason") or "")
            )

    except Exception:
        pass

    return out

def _level_between_entry_and_stop(
    side: str,
    level: Optional[float],
    entry: Optional[float],
    stop: Optional[float],
) -> bool:
    """
    Check whether a liquidity / imbalance level is inside the SL-side range.

    SHORT:
      entry < level < stop

    LONG:
      stop < level < entry
    """
    try:
        if level is None or entry is None or stop is None:
            return False

        side = str(side or "").upper()

        if side == "SHORT":
            return entry < level < stop

        if side == "LONG":
            return stop < level < entry

    except Exception:
        pass

    return False


def _collect_sl_side_magnet_levels(
    side: str,
    scenario: Dict[str, Any],
    entry: Optional[float],
    stop: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Collect SL-side magnet levels from liquidity_risk and imbalance_risk.

    Magnet means:
    - liquidity level inside entry-to-SL range
    - imbalance/FVG retest zone inside entry-to-SL range

    This does NOT change entry/SL/TP.
    This is only for TP1 runner management.
    """
    levels: List[Dict[str, Any]] = []

    try:
        side = str(side or "").upper()
        if side not in ["LONG", "SHORT"]:
            return levels

        if entry is None or stop is None:
            return levels

        # -------------------------------
        # Liquidity level
        # -------------------------------
        liquidity = scenario.get("liquidity_risk") or {}
        if isinstance(liquidity, dict):
            suggested = (
                liquidity.get("suggested_sweep_level")
                or liquidity.get("sweep_level")
                or liquidity.get("liquidity_sweep_level")
                or {}
            )

            liq_price = None
            liq_type = ""

            if isinstance(suggested, dict):
                liq_price = _extract_price(
                    suggested.get("price")
                    or suggested.get("level")
                    or suggested.get("value")
                )
                liq_type = str(suggested.get("type") or "LIQUIDITY").upper()
            else:
                liq_price = _extract_price(suggested)
                liq_type = "LIQUIDITY"

            if _level_between_entry_and_stop(side, liq_price, entry, stop):
                levels.append({
                    "source": "LIQUIDITY",
                    "type": liq_type,
                    "price": liq_price,
                    "zone_bottom": None,
                    "zone_top": None,
                    "risk_level": str(liquidity.get("risk_level") or "").upper(),
                    "score": liquidity.get("liquidity_score"),
                })

        # -------------------------------
        # Imbalance / FVG level
        # -------------------------------
        imbalance = scenario.get("imbalance_risk") or {}
        if isinstance(imbalance, dict):
            zone = (
                imbalance.get("suggested_entry_zone")
                or imbalance.get("suggested_zone")
                or {}
            )

            imb_price = None
            zone_bottom = None
            zone_top = None
            imb_type = "IMBALANCE_FVG"

            if isinstance(zone, dict):
                imb_price = _extract_price(
                    zone.get("price")
                    or zone.get("level")
                    or zone.get("value")
                )
                zone_bottom = _extract_price(zone.get("zone_bottom"))
                zone_top = _extract_price(zone.get("zone_top"))
                imb_type = str(zone.get("type") or "IMBALANCE_FVG").upper()
            else:
                imb_price = _extract_price(zone)

            # If price is missing but zone exists, use zone midpoint.
            if imb_price is None and zone_bottom is not None and zone_top is not None:
                imb_price = (zone_bottom + zone_top) / 2.0

            # Check price first.
            inside = _level_between_entry_and_stop(side, imb_price, entry, stop)

            # If midpoint/price not inside, check whether zone overlaps entry-stop range.
            if not inside and zone_bottom is not None and zone_top is not None:
                z_low = min(zone_bottom, zone_top)
                z_high = max(zone_bottom, zone_top)

                if side == "SHORT":
                    inside = bool(z_high > entry and z_low < stop)
                elif side == "LONG":
                    inside = bool(z_low < entry and z_high > stop)

            if inside and imb_price is not None:
                levels.append({
                    "source": "IMBALANCE",
                    "type": imb_type,
                    "price": imb_price,
                    "zone_bottom": zone_bottom,
                    "zone_top": zone_top,
                    "risk_level": str(imbalance.get("risk_level") or "").upper(),
                    "score": imbalance.get("imbalance_score"),
                })

        return levels

    except Exception:
        return levels


def _tp1_sl_side_magnet_state(
    side: str,
    scenario: Dict[str, Any],
    entry: Optional[float],
    stop: Optional[float],
    current_price: Optional[float],
) -> Dict[str, Any]:
    """
    After TP1, detect if price is retesting unresolved SL-side liquidity/imbalance.

    Output meaning:
    - pending_magnet_test:
        Price is moving toward liquidity/imbalance, but has not clearly closed beyond it.
        Do NOT call CLOSE_FULL too early. Use defensive runner.

    - magnet_breached_against_trade:
        Price closed beyond the SL-side liquidity/imbalance level against the trade.
        Now CLOSE_FULL is safer.

    SHORT:
      magnet is above entry.
      pending while current close is below/near magnet.
      breached if current close is above magnet.

    LONG:
      magnet is below entry.
      pending while current close is above/near magnet.
      breached if current close is below magnet.
    """
    out = {
        "has_magnet": False,
        "pending_magnet_test": False,
        "magnet_breached_against_trade": False,
        "primary_level": None,
        "sources": [],
        "levels": [],
        "reason": "",
    }

    try:
        side = str(side or "").upper()
        if side not in ["LONG", "SHORT"]:
            return out

        if entry is None or stop is None or current_price is None:
            return out

        levels = _collect_sl_side_magnet_levels(
            side=side,
            scenario=scenario,
            entry=entry,
            stop=stop,
        )

        if not levels:
            return out

        # Choose the nearest magnet level from current price.
        levels_sorted = sorted(
            levels,
            key=lambda x: abs(float(x.get("price") or current_price) - current_price)
        )
        primary = levels_sorted[0]
        primary_level = _extract_price(primary.get("price"))

        if primary_level is None:
            return out

        tol = abs(primary_level) * 0.0015

        sources = []
        for item in levels_sorted:
            src = str(item.get("source") or "").upper()
            if src and src not in sources:
                sources.append(src)

        out.update({
            "has_magnet": True,
            "primary_level": primary_level,
            "sources": sources,
            "levels": levels_sorted,
        })

        if side == "SHORT":
            if current_price > primary_level + tol:
                out.update({
                    "magnet_breached_against_trade": True,
                    "reason": (
                        "Price has closed beyond SL-side liquidity/imbalance against the SHORT runner."
                    ),
                })
                return out

            out.update({
                "pending_magnet_test": True,
                "reason": (
                    "TP1 was hit, but price is retracing toward unresolved upside liquidity/imbalance "
                    "inside the entry-to-SL range."
                ),
            })
            return out

        if side == "LONG":
            if current_price < primary_level - tol:
                out.update({
                    "magnet_breached_against_trade": True,
                    "reason": (
                        "Price has closed beyond SL-side liquidity/imbalance against the LONG runner."
                    ),
                })
                return out

            out.update({
                "pending_magnet_test": True,
                "reason": (
                    "TP1 was hit, but price is retracing toward unresolved downside liquidity/imbalance "
                    "inside the entry-to-SL range."
                ),
            })
            return out

        return out

    except Exception:
        return out

def _simple_atr(df: Optional[pd.DataFrame], period: int = 14) -> Optional[float]:
    """
    Lightweight ATR for guidance caution only.
    Does not affect entry/SL/TP.
    """
    try:
        d = _normalize_ohlc(df)
        if len(d) < 3:
            return None

        prev_close = d["Close"].shift(1)
        tr1 = d["High"] - d["Low"]
        tr2 = (d["High"] - prev_close).abs()
        tr3 = (d["Low"] - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.tail(period).mean()

        if pd.isna(atr) or atr <= 0:
            return None

        return float(atr)

    except Exception:
        return None


def _latest_rejection_at_entry(
    side: str,
    df_live: Optional[pd.DataFrame],
    entry: Optional[float],
) -> bool:
    """
    Detect whether latest candle touched/retested entry and rejected.

    LONG:
      lower wick touched below/near entry and close is back above entry.

    SHORT:
      upper wick touched above/near entry and close is back below entry.
    """
    try:
        if entry is None:
            return False

        d = _normalize_ohlc(df_live)
        if d.empty:
            return False

        row = d.iloc[-1]
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

        side = str(side or "").upper()
        tol = abs(entry) * 0.0015

        candle_range = max(h - l, 0.0)
        if candle_range <= 0:
            return False

        body_low = min(o, c)
        body_high = max(o, c)

        lower_wick = body_low - l
        upper_wick = h - body_high

        if side == "LONG":
            touched = l <= entry + tol
            reclaimed = c >= entry - tol
            strong_lower_wick = lower_wick >= candle_range * 0.35
            return bool(touched and reclaimed and strong_lower_wick)

        if side == "SHORT":
            touched = h >= entry - tol
            reclaimed = c <= entry + tol
            strong_upper_wick = upper_wick >= candle_range * 0.35
            return bool(touched and reclaimed and strong_upper_wick)

    except Exception:
        pass

    return False


def _near_entry_caution(
    side: str,
    entry: Optional[float],
    stop: Optional[float],
    current_price: Optional[float],
    entry_hit: bool,
    df_live: Optional[pd.DataFrame],
    df_confirm: Optional[pd.DataFrame],
    opposite_pressure: bool,
    confirm_opposite_pressure: bool,
) -> Dict[str, Any]:
    """
    Adds a near-entry caution label/message only.

    Important:
    - Does NOT change trade status.
    - Does NOT cancel the setup.
    - Does NOT modify entry/SL/TP.
    - Only helps user manage near-entry volatility.
    """
    out = {
        "entry_caution_level": "NORMAL",
        "entry_caution_reason": "",
        "entry_caution_advice": "Setup remains valid. Wait for entry according to plan.",
    }

    try:
        if entry is None or stop is None or current_price is None:
            return out

        side = str(side or "").upper()
        if side not in ["LONG", "SHORT"]:
            return out

        risk_distance = abs(entry - stop)
        if risk_distance <= 0:
            return out

        d = _normalize_ohlc(df_live)
        if len(d) < 3:
            return out

        latest = d.iloc[-1]
        last_close = float(latest["Close"])
        last_open = float(latest["Open"])
        last_high = float(latest["High"])
        last_low = float(latest["Low"])

        atr = _simple_atr(d, period=14)

        recent = d.tail(3)
        first_close = float(recent.iloc[0]["Close"])
        last_recent_close = float(recent.iloc[-1]["Close"])

        # Is current price near entry zone?
        distance_to_entry = abs(current_price - entry)
        near_entry = distance_to_entry <= (risk_distance * 1.25)

        # Directional pressure against trade
        if side == "LONG":
            last_body_against = max(last_open - last_close, 0.0)
            net_move_against = max(first_close - last_recent_close, 0.0)
            close_failed_entry = last_close < entry - (abs(entry) * 0.001)
        else:
            last_body_against = max(last_close - last_open, 0.0)
            net_move_against = max(last_recent_close - first_close, 0.0)
            close_failed_entry = last_close > entry + (abs(entry) * 0.001)

        strong_last_candle = False
        strong_recent_move = False

        if atr is not None and atr > 0:
            strong_last_candle = last_body_against >= atr * 0.85
            strong_recent_move = net_move_against >= atr * 1.25

        rejection = _latest_rejection_at_entry(side, d, entry)

        # Not near entry and not entered: keep normal. Do not pollute advice.
        if not near_entry and not entry_hit:
            return out

        # Entry touched with rejection/reclaim = volatile but still acceptable.
        if entry_hit and rejection:
            out.update({
                "entry_caution_level": "VOLATILE_ENTRY",
                "entry_caution_reason": (
                    "Price reached the entry zone with volatility, but a rejection/reclaim reaction appeared."
                ),
                "entry_caution_advice": (
                    "Entry zone is volatile, but rejection appeared. Setup can still work. Manage risk carefully."
                ),
            })
            return out

        # Entry hit or very near entry, but failed to hold with pressure.
        if entry_hit and close_failed_entry and (
            opposite_pressure or confirm_opposite_pressure or strong_last_candle or strong_recent_move
        ):
            out.update({
                "entry_caution_level": "HIGH_RISK",
                "entry_caution_reason": (
                    "Entry/support failed to hold cleanly and recent candles show pressure against the setup."
                ),
                "entry_caution_advice": (
                    "High risk. Entry/support failed to hold. Consider reducing risk if next candle cannot reclaim."
                ),
            })
            return out

        # Near entry with pressure, but not fully failed.
        if near_entry and (
            opposite_pressure or confirm_opposite_pressure or strong_last_candle or strong_recent_move
        ):
            out.update({
                "entry_caution_level": "CAUTION",
                "entry_caution_reason": (
                    "Price is near the entry zone and recent candles show pressure against the setup."
                ),
                "entry_caution_advice": (
                    "Setup is still valid, but risk increased. Do not add more. Wait for reclaim/reaction."
                ),
            })
            return out

        return out

    except Exception:
        return out


def _append_entry_caution_text(user_guidance: str, caution: Dict[str, Any]) -> str:
    try:
        level = str(caution.get("entry_caution_level") or "NORMAL").upper()
        advice = str(caution.get("entry_caution_advice") or "").strip()

        if not advice or level == "NORMAL":
            return user_guidance

        if advice in str(user_guidance or ""):
            return user_guidance

        return str(user_guidance or "").strip() + " " + advice

    except Exception:
        return user_guidance

def _build_management_decision_after_tp1(
    side: str,
    tp1_lost_after_hit: bool,
    danger_score: int,
    opposite_pressure: bool,
    confirm_opposite_pressure: bool,
    returned_to_entry_after_profit: bool,
    close_back_beyond_entry: bool,
    pending_magnet_test_after_tp1: bool = False,
    magnet_breached_after_tp1: bool = False,
    magnet_test_level: Optional[float] = None,
    magnet_sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Decide management mode after TP1 is confirmed.

    Important rules:
    - TP1 hit means secure partial profit.
    - Hold only the remaining runner toward TP2.
    - Do NOT recommend automatic breakeven.
    - Only tighten or close the runner when reversal risk increases.
    - Full close only when reversal is confirmed.
    """
    decision = {
        "management_mode": "HOLD_FOR_TP2",
        "tp2_probability": "STRONG",
        "reversal_risk": "LOW",
        "profit_action": "SECURE_PARTIAL",
        "sl_management": "STRUCTURE_TRAIL_ONLY",
        "short_action_text": "ACTION: SECURE TP1 PARTIAL, HOLD RUNNER",
        "exit_trigger": "Exit only if price loses TP1/structure.",
        "action": "HOLD_FOR_TP2",
        "risk_level": "LOW",
        "guidance_status": "TP1_HIT",
    }

    # --------------------------------------------------
    # Strong reversal confirmation after TP1
    # --------------------------------------------------
    # Full close only when price loses entry/structure after TP1,
    # not just because TP1 was retested.
    # --------------------------------------------------
    # After-TP1 SL-side liquidity / imbalance refinement
    # --------------------------------------------------
    # Important:
    # - If liquidity/imbalance is still pending inside entry-to-SL range,
    #   do NOT immediately call CLOSE_FULL.
    # - If price closes beyond that liquidity/imbalance level against the trade,
    #   CLOSE_FULL becomes justified.
    sources_text = "/".join(magnet_sources or [])

    if magnet_breached_after_tp1:
        decision.update({
            "management_mode": "CLOSE_FULL",
            "tp2_probability": "WEAK",
            "reversal_risk": "CONFIRMED",
            "profit_action": "CLOSE_FULL",
            "sl_management": "CLOSE_FULL",
            "short_action_text": "ACTION: CLOSE FULL",
            "exit_trigger": (
                f"Price closed/held beyond SL-side {sources_text or 'liquidity/imbalance'} "
                f"near {_format_price(magnet_test_level)} after TP1."
            ),
            "action": "CLOSE_FULL",
            "risk_level": "HIGH",
            "guidance_status": "TP1_MAGNET_LEVEL_BREACHED",
            "magnet_test_level": magnet_test_level,
            "magnet_sources": magnet_sources or [],
        })
        return decision

    if pending_magnet_test_after_tp1:
        decision.update({
            "management_mode": "DEFENSIVE_RUNNER_TO_MAGNET_TEST",
            "tp2_probability": "MIXED",
            "reversal_risk": "HIGH",
            "profit_action": "PROTECT_PROFIT_WAIT_MAGNET_TEST",
            "sl_management": "STRUCTURE_TRAIL_ONLY",
            "short_action_text": "ACTION: DEFENSIVE RUNNER",
            "exit_trigger": (
                f"Watch SL-side {sources_text or 'liquidity/imbalance'} near {_format_price(magnet_test_level)}. "
                "Continue runner only if that level sweeps/fills and rejects in the trade direction; "
                "close remaining if price closes/holds beyond it."
            ),
            "action": "DEFENSIVE_RUNNER_TO_MAGNET_TEST",
            "risk_level": "HIGH",
            "guidance_status": "TP1_HIT_MAGNET_TEST_PENDING",
            "magnet_test_level": magnet_test_level,
            "magnet_sources": magnet_sources or [],
        })
        return decision
    if close_back_beyond_entry:
        decision.update({
            "management_mode": "CLOSE_FULL",
            "tp2_probability": "WEAK",
            "reversal_risk": "CONFIRMED",
            "profit_action": "CLOSE_FULL",
            "sl_management": "CLOSE_FULL",
            "short_action_text": "ACTION: CLOSE FULL",
            "exit_trigger": "Price lost entry/structure after TP1.",
            "action": "CLOSE_FULL",
            "risk_level": "HIGH",
            "guidance_status": "TP1_REVERSAL_CONFIRMED",
        })
        return decision

    # TP1 lost + returned close to entry = stronger reversal confirmation.
    if tp1_lost_after_hit and returned_to_entry_after_profit:
        decision.update({
            "management_mode": "CLOSE_FULL",
            "tp2_probability": "WEAK",
            "reversal_risk": "CONFIRMED",
            "profit_action": "CLOSE_FULL",
            "sl_management": "CLOSE_FULL",
            "short_action_text": "ACTION: CLOSE FULL",
            "exit_trigger": "Price returned toward entry after TP1.",
            "action": "CLOSE_FULL",
            "risk_level": "HIGH",
            "guidance_status": "TP1_REVERSAL_CONFIRMED",
        })
        return decision

    # TP1 lost + bearish pressure = protect profit, not full close yet.
    if tp1_lost_after_hit and (opposite_pressure or confirm_opposite_pressure):
        decision.update({
            "management_mode": "PROTECT_PROFIT",
            "tp2_probability": "WEAK",
            "reversal_risk": "HIGH",
            "profit_action": "SECURE_PARTIAL",
            "sl_management": "STRUCTURE_TRAIL_ONLY",
            "short_action_text": "ACTION: PROTECT PROFIT",
            "exit_trigger": "Hold remaining only while entry/structure stays valid.",
            "action": "PROTECT_PROFIT",
            "risk_level": "MEDIUM",
            "guidance_status": "TP1_HIT_PROTECT_PROFIT",
        })
        return decision

    # TP1 lost only = caution, not close full.
    if tp1_lost_after_hit:
        decision.update({
            "management_mode": "HOLD_WITH_CAUTION",
            "tp2_probability": "MIXED",
            "reversal_risk": "MEDIUM",
            "profit_action": "SECURE_PARTIAL",
            "sl_management": "STRUCTURE_TRAIL_ONLY",
            "short_action_text": "ACTION: HOLD WITH CAUTION",
            "exit_trigger": "Continue only while entry/structure holds.",
            "action": "HOLD_WITH_CAUTION",
            "risk_level": "MEDIUM",
            "guidance_status": "TP1_HIT_CAUTION",
        })
        return decision

    # --------------------------------------------------
    # High risk but not fully confirmed reversal
    # --------------------------------------------------
    high_risk_inputs = 0

    if opposite_pressure:
        high_risk_inputs += 1
    if confirm_opposite_pressure:
        high_risk_inputs += 1
    if returned_to_entry_after_profit:
        high_risk_inputs += 1
    if danger_score >= 45:
        high_risk_inputs += 1

    if high_risk_inputs >= 2:
        decision.update({
            "management_mode": "PROTECT_PROFIT",
            "tp2_probability": "WEAK",
            "reversal_risk": "HIGH",
            "profit_action": "SECURE_PARTIAL",
            "sl_management": "STRUCTURE_TRAIL_ONLY",
            "short_action_text": "ACTION: PROTECT PROFIT",
            "exit_trigger": "Hold remaining only while TP1/structure stays valid.",
            "action": "PROTECT_PROFIT",
            "risk_level": "MEDIUM",
            "guidance_status": "TP1_HIT_PROTECT_PROFIT",
        })
        return decision

    # --------------------------------------------------
    # Mixed / caution
    # --------------------------------------------------
    if danger_score >= 25 or opposite_pressure or confirm_opposite_pressure:
        decision.update({
            "management_mode": "HOLD_WITH_CAUTION",
            "tp2_probability": "MIXED",
            "reversal_risk": "MEDIUM",
            "profit_action": "SECURE_PARTIAL",
            "sl_management": "STRUCTURE_TRAIL_ONLY",
            "short_action_text": "ACTION: HOLD WITH CAUTION",
            "exit_trigger": "Continue only while TP1/structure holds.",
            "action": "HOLD_WITH_CAUTION",
            "risk_level": "MEDIUM",
            "guidance_status": "TP1_HIT_CAUTION",
        })
        return decision

    return decision

# ==========================================================
# Main guidance function
# ==========================================================

def build_trade_guidance_facts(
    scenario: Dict[str, Any],
    df_live: Optional[pd.DataFrame],
    df_since_lock: Optional[pd.DataFrame],
    interval: str = "15m",
    existing_status: str = "",
    existing_advice: str = "",
    df_confirm: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build deterministic backend facts/guidance for an existing locked/global signal.

    This does not create a new setup.
    This does not change entry/SL/TP.
    """
    scenario = scenario or {}
    df_since_lock = _normalize_ohlc(df_since_lock)
    df_live_norm = _normalize_ohlc(df_live)
    df_confirm_norm = _normalize_ohlc(df_confirm)

    result: Dict[str, Any] = {
        "guidance_status": "UNKNOWN",
        "risk_level": "UNKNOWN",
        "action": "WAIT",
        "backend_advice": "Unable to evaluate trade guidance.",
        "user_guidance": "Unable to evaluate trade guidance.",
        "hard_status": existing_status or "",
        "existing_advice": existing_advice or "",
        "danger_score": 0,
        "danger_flags": [],
        "management_mode": "UNKNOWN",
        "tp2_probability": "UNKNOWN",
        "reversal_risk": "UNKNOWN",
        "profit_action": "UNKNOWN",
        "sl_management": "UNKNOWN",
        "short_action_text": "",
        "exit_trigger": "",
        "entry_caution_level": "NORMAL",
        "entry_caution_reason": "",
        "entry_caution_advice": "",
        "facts": {},
    }

    try:
        side = str(
            scenario.get("trade_type")
            or scenario.get("side")
            or ""
        ).upper()

        if side not in ["LONG", "SHORT"]:
            result.update({
                "guidance_status": "NO_TRADE",
                "risk_level": "NONE",
                "action": "NO_ACTION",
                "backend_advice": "No active LONG/SHORT trade to guide.",
                "user_guidance": "No active LONG/SHORT trade to guide.",
            })
            return result

        prices = _scenario_prices(scenario)
        entry = prices.get("entry")
        stop = prices.get("stop")
        tp1 = prices.get("tp1")
        tp2 = prices.get("tp2")

        current_price = _latest_price(df_live_norm)
        pnl_pct = _calculate_pnl_pct(side, entry, current_price)
        tp1_progress_pct = _progress_to_tp1_pct(side, entry, tp1, current_price)
        tp2_progress_pct = _progress_to_tp2_pct(side, entry, tp2, current_price)

        if entry is None or stop is None or tp1 is None:
            result.update({
                "guidance_status": "INVALID_DATA",
                "risk_level": "HIGH",
                "action": "WAIT",
                "backend_advice": "Trade guidance unavailable because entry/SL/TP1 data is missing.",
                "user_guidance": "Trade guidance unavailable because entry/SL/TP1 data is missing.",
            })
            return result

        if df_since_lock.empty:
            df_since_lock = df_live_norm.tail(1)

        entry_hit = _entry_touched(side, df_since_lock, entry)
        tp1_hit = _tp_hit(side, df_since_lock, tp1)
        tp2_hit = _tp_hit(side, df_since_lock, tp2)
        sl_hit = _sl_hit(side, df_since_lock, stop)

        missed_no_chase = _price_moved_70pct_to_tp1_before_entry(
            side=side,
            df_since_lock=df_since_lock,
            entry=entry,
            tp1=tp1,
        )

        returned_to_entry_after_profit = _detect_return_to_entry_after_profit(
            side=side,
            entry=entry,
            tp1=tp1,
            df_since_lock=df_since_lock,
        )

        close_back_beyond_entry = _detect_close_back_beyond_entry(
            side=side,
            entry=entry,
            df_live=df_live_norm,
        )

        pressure = _simple_opposite_pressure(
            side=side,
            df_live=df_live_norm,
        )

        confirm_pressure = _confirm_tf_pressure(
            side=side,
            df_confirm=df_confirm_norm,
        )

        entry_caution = _near_entry_caution(
            side=side,
            entry=entry,
            stop=stop,
            current_price=current_price,
            entry_hit=entry_hit,
            df_live=df_live_norm,
            df_confirm=df_confirm_norm,
            opposite_pressure=bool(pressure.get("opposite_pressure", False)),
            confirm_opposite_pressure=bool(confirm_pressure.get("confirm_opposite_pressure", False)),
        )

        tp1_magnet_state = _tp1_sl_side_magnet_state(
            side=side,
            scenario=scenario,
            entry=entry,
            stop=stop,
            current_price=current_price,
        )

        tp1_lost_after_hit = False
        # Only meaningful after TP1 is confirmed.
        # This is used later in TP1 management decision.

        facts = {
            "side": side,
            "entry": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "current_price": current_price,
            "pnl_pct": round(pnl_pct, 3) if pnl_pct is not None else None,
            "entry_hit": entry_hit,
            "tp1_hit": tp1_hit,
            "tp2_hit": tp2_hit,
            "sl_hit": sl_hit,
            "tp1_progress_pct": round(tp1_progress_pct, 2),
            "tp2_progress_pct": round(tp2_progress_pct, 2),
            "missed_no_chase": missed_no_chase,
            "returned_to_entry_after_profit": returned_to_entry_after_profit,
            "close_back_beyond_entry": close_back_beyond_entry,
            "opposite_pressure": pressure.get("opposite_pressure", False),
            "opposite_pressure_reason": pressure.get("reason", ""),
            "confirm_opposite_pressure": confirm_pressure.get("confirm_opposite_pressure", False),
            "confirm_pressure_reason": confirm_pressure.get("confirm_pressure_reason", ""),
            "entry_caution_level": entry_caution.get("entry_caution_level", "NORMAL"),
            "entry_caution_reason": entry_caution.get("entry_caution_reason", ""),
            "entry_caution_advice": entry_caution.get("entry_caution_advice", ""),
            "tp1_lost_after_hit": tp1_lost_after_hit,
            "tp1_magnet_has_level": tp1_magnet_state.get("has_magnet", False),
            "tp1_pending_magnet_test": tp1_magnet_state.get("pending_magnet_test", False),
            "tp1_magnet_breached_against_trade": tp1_magnet_state.get("magnet_breached_against_trade", False),
            "tp1_magnet_level": tp1_magnet_state.get("primary_level"),
            "tp1_magnet_sources": tp1_magnet_state.get("sources", []),
            "tp1_magnet_reason": tp1_magnet_state.get("reason", ""),
        }

        result["facts"] = facts

        danger_score = 0
        flags: List[str] = []

        # ==================================================
        # Authoritative terminal status first
        # ==================================================
        # check_trade_status_historical() evaluates candles
        # chronologically using the conservative same-candle
        # SL-first rule. Trust that result here.
        hard_status = str(
            existing_status
            or ""
        ).upper().strip().replace(
            " ",
            "_",
        )

        hard_reason_upper = str(
            existing_advice
            or ""
        ).upper()

        if hard_status == "WON":
            partial_runner_stop = bool(
                "TP1" in hard_reason_upper
                and (
                    "STOP" in hard_reason_upper
                    or "SL" in hard_reason_upper
                )
            )

            if partial_runner_stop:
                result.update({
                    "guidance_status": (
                        "TP1_PARTIAL_WIN"
                    ),

                    "risk_level": "CLOSED",

                    "action": (
                        "PARTIAL_WIN_CONFIRMED"
                    ),

                    "backend_advice": (
                        "TP1 was secured before the "
                        "remaining runner later hit Stop "
                        "Loss. Record this as a partial win."
                    ),

                    "user_guidance": (
                        "TP1 profit was secured. The "
                        "remaining runner later hit SL, "
                        "so this trade is closed as a "
                        "partial win."
                    ),

                    "danger_score": 0,
                    "danger_flags": [],

                    "management_mode": (
                        "CLOSED_PARTIAL_WIN"
                    ),

                    "tp2_probability": "CLOSED",
                    "reversal_risk": "REALIZED",

                    "profit_action": (
                        "TP1_PARTIAL_SECURED"
                    ),

                    "sl_management": (
                        "TRADE_CLOSED"
                    ),

                    "short_action_text": (
                        "ACTION: PARTIAL WIN CLOSED"
                    ),

                    "exit_trigger": (
                        "Runner reached Stop Loss "
                        "after TP1."
                    ),
                })

                return result

            result.update({
                "guidance_status": "TP2_HIT",
                "risk_level": "CLOSED",

                "action": (
                    "TAKE_PROFIT_CONFIRMED"
                ),

                "backend_advice": (
                    "TP2/full target was reached."
                ),

                "user_guidance": (
                    "TP2 hit. This trade idea has "
                    "completed its main target."
                ),

                "danger_score": 0,
                "danger_flags": [],

                "management_mode": (
                    "CLOSED_FULL_WIN"
                ),

                "tp2_probability": "COMPLETED",
                "reversal_risk": "NONE",

                "profit_action": (
                    "TP2_FULL_SECURED"
                ),

                "sl_management": (
                    "TRADE_CLOSED"
                ),

                "short_action_text": (
                    "ACTION: FULL TARGET COMPLETE"
                ),

                "exit_trigger": (
                    "TP2/full target reached."
                ),
            })

            return result

        if hard_status == "LOST":
            result.update({
                "guidance_status": (
                    "CLOSE_CONFIRMED"
                ),

                "risk_level": "CLOSED",

                "action": (
                    "SL_HIT_CONFIRMED"
                ),

                "backend_advice": (
                    f"Stop loss / invalidation was hit "
                    f"for this {side} setup before a "
                    "secured TP1 outcome."
                ),

                "user_guidance": (
                    f"SL hit confirmed for this {side} "
                    "setup before TP1. Do not hold this "
                    "old trade idea."
                ),

                "danger_score": 100,

                "danger_flags": [
                    "SL_HIT_CONFIRMED",
                ],

                "management_mode": (
                    "CLOSED_LOSS"
                ),

                "tp2_probability": "CLOSED",
                "reversal_risk": "REALIZED",
                "profit_action": "NONE",

                "sl_management": (
                    "TRADE_CLOSED"
                ),

                "short_action_text": (
                    "ACTION: LOSS CLOSED"
                ),

                "exit_trigger": (
                    "Stop Loss hit before TP1."
                ),
            })

            return result

        # Fallback only when no authoritative terminal
        # status was supplied.
        if sl_hit:
            result.update({
                "guidance_status": (
                    "CLOSE_CONFIRMED"
                ),

                "risk_level": "CLOSED",

                "action": (
                    "SL_HIT_CONFIRMED"
                ),

                "backend_advice": (
                    f"Stop loss / invalidation appears "
                    f"to be hit for this {side} setup. "
                    "This global signal should be "
                    "treated as closed/invalidated."
                ),

                "user_guidance": (
                    f"SL hit confirmed for this {side} "
                    "setup. Do not hold this old "
                    "trade idea."
                ),

                "danger_score": 100,

                "danger_flags": [
                    "SL_HIT_CONFIRMED",
                ],
            })

            return result

        if tp2_hit:
            result.update({
                "guidance_status": "TP2_HIT",
                "risk_level": "CLOSED",

                "action": (
                    "TAKE_PROFIT_CONFIRMED"
                ),

                "backend_advice": (
                    "TP2 appears to be hit. "
                    "Full target reached."
                ),

                "user_guidance": (
                    "TP2 hit. This trade idea has "
                    "completed its main target."
                ),

                "danger_score": 0,
                "danger_flags": [],
            })

            return result

        # ==================================================
        # Pending limit order phase
        # ==================================================
        if not entry_hit:
            if missed_no_chase:
                result.update({
                    "guidance_status": "MISSED",
                    "risk_level": "HIGH",
                    "action": "DO_NOT_CHASE",
                    "backend_advice": (
                        "Entry was not reached, but price moved more than 70% toward TP1. "
                        "Treat this setup as missed. Do not chase."
                    ),
                    "user_guidance": (
                        "This setup is likely missed. Price moved too far toward TP1 before entry. Do not chase."
                    ),
                    "danger_score": 85,
                    "danger_flags": ["NO_CHASE"],
                })
                return result

            # Pending but current price is dangerously close to SL side before entry.
            if side == "LONG" and current_price is not None and current_price <= stop:
                result.update({
                    "guidance_status": "CANCELLED",
                    "risk_level": "HIGH",
                    "action": "CANCEL_PENDING",
                    "backend_advice": "Price is already beyond LONG invalidation before entry.",
                    "user_guidance": "Cancel this pending LONG setup. Price has moved beyond invalidation before safe entry.",
                    "danger_score": 95,
                    "danger_flags": ["INVALIDATION_BEFORE_ENTRY"],
                })
                return result

            if side == "SHORT" and current_price is not None and current_price >= stop:
                result.update({
                    "guidance_status": "CANCELLED",
                    "risk_level": "HIGH",
                    "action": "CANCEL_PENDING",
                    "backend_advice": "Price is already beyond SHORT invalidation before entry.",
                    "user_guidance": "Cancel this pending SHORT setup. Price has moved beyond invalidation before safe entry.",
                    "danger_score": 95,
                    "danger_flags": ["INVALIDATION_BEFORE_ENTRY"],
                })
                return result

            base_user_guidance = (
                f"Limit order still pending. Wait for entry around {_format_price(entry)}. "
                "Do not chase the market."
            )

            base_backend_advice = (
                f"Limit order is still pending. Wait for price to reach entry {_format_price(entry)}. "
                "Do not chase if price moves away from entry."
            )

            result.update({
                "guidance_status": "WAIT_ENTRY",
                "risk_level": "LOW",
                "action": "WAIT_FOR_ENTRY",
                "backend_advice": _append_entry_caution_text(base_backend_advice, entry_caution),
                "user_guidance": _append_entry_caution_text(base_user_guidance, entry_caution),
                "danger_score": 10,
                "danger_flags": [],
                "entry_caution_level": entry_caution.get("entry_caution_level", "NORMAL"),
                "entry_caution_reason": entry_caution.get("entry_caution_reason", ""),
                "entry_caution_advice": entry_caution.get("entry_caution_advice", ""),
            })
            return result

        # ==================================================
        # Active trade phase
        # ==================================================
        if close_back_beyond_entry:
            danger_score += 25
            flags.append("CLOSE_BACK_BEYOND_ENTRY")

        if returned_to_entry_after_profit:
            danger_score += 20
            flags.append("RETURNED_TO_ENTRY_AFTER_PROFIT")

        if pressure.get("opposite_pressure"):
            danger_score += 15
            flags.append("OPPOSITE_CANDLE_PRESSURE")

        if confirm_pressure.get("confirm_opposite_pressure"):
            danger_score += 10
            flags.append("CONFIRM_TF_OPPOSITE_PRESSURE")

        # TP1 hit but TP2 not hit.
        if tp1_hit and not tp2_hit:
            tp1_lost_after_hit = _closed_back_through_tp1(
                side=side,
                tp1=tp1,
                df_live=df_live_norm,
            )

            facts["tp1_lost_after_hit"] = tp1_lost_after_hit
            result["facts"] = facts

            if tp1_lost_after_hit:
                danger_score += 40
                flags.append("TP1_LOST_AFTER_HIT")

            decision = _build_management_decision_after_tp1(
                side=side,
                tp1_lost_after_hit=tp1_lost_after_hit,
                danger_score=danger_score,
                opposite_pressure=bool(pressure.get("opposite_pressure")),
                confirm_opposite_pressure=bool(confirm_pressure.get("confirm_opposite_pressure")),
                returned_to_entry_after_profit=returned_to_entry_after_profit,
                close_back_beyond_entry=close_back_beyond_entry,
                pending_magnet_test_after_tp1=bool(
                    tp1_magnet_state.get("pending_magnet_test", False)
                ),
                magnet_breached_after_tp1=bool(
                    tp1_magnet_state.get("magnet_breached_against_trade", False)
                ),
                magnet_test_level=tp1_magnet_state.get("primary_level"),
                magnet_sources=tp1_magnet_state.get("sources", []),
            )

            if decision["management_mode"] == "HOLD_FOR_TP2":
                user_guidance = (
                    "ACTION: SECURE TP1 PARTIAL, HOLD RUNNER. "
                    "TP1 is confirmed and structure is still healthy. "
                    "Secure partial profit at TP1, then hold the "
                    "remaining runner toward TP2 and trail only "
                    "by market structure."
                )

                backend_advice = (
                    "TP1 is confirmed. Secure partial profit at TP1 "
                    "and keep only the remaining runner toward TP2. "
                    "Do not move SL blindly."
                )

            elif decision["management_mode"] == "HOLD_WITH_CAUTION":
                user_guidance = (
                    "ACTION: SECURE TP1 PARTIAL, HOLD WITH CAUTION. "
                    "TP1 is confirmed, but momentum is not fully clean. "
                    "Keep TP1 partial profit secured and hold only "
                    "the remaining runner while TP1/structure stays valid."
                )

                backend_advice = (
                    "TP1 is confirmed and partial profit should be "
                    "secured. Hold only the runner with caution unless "
                    "TP1/structure is lost."
                )

            elif decision["management_mode"] == "PROTECT_PROFIT":
                user_guidance = (
                    "ACTION: PROTECT PROFIT. "
                    "TP1 is confirmed, but reversal risk is increasing. "
                    "Secure partial profit and keep the rest only if TP1/structure stays valid."
                )
                backend_advice = (
                    "TP1 is confirmed and reversal risk is increasing. "
                    "Protect profit, but avoid blind SL movement that can cut a normal TP2 retest."
                )

            elif decision["management_mode"] == "DEFENSIVE_RUNNER_TO_MAGNET_TEST":
                magnet_level_text = _format_price(decision.get("magnet_test_level"))
                magnet_sources_text = "/".join(decision.get("magnet_sources") or [])

                user_guidance = (
                    "ACTION: DEFENSIVE RUNNER. "
                    f"TP1 is confirmed, but price is retracing toward unresolved "
                    f"{magnet_sources_text or 'liquidity/imbalance'} near {magnet_level_text}. "
                    "Keep TP1 partial profit secured and do not add more. "
                    "Hold the runner only if that level sweeps/fills and rejects "
                    "in the trade direction. Close the remaining runner if price closes or holds beyond that level."
                )

                backend_advice = (
                    "TP1 is confirmed. Price is retracing toward unresolved SL-side "
                    f"{magnet_sources_text or 'liquidity/imbalance'} after TP1. "
                    "Do not force immediate full close before the magnet level is tested. "
                    "Use defensive runner management and close if that level holds against the trade."
                )

            elif decision["management_mode"] == "CLOSE_FULL":
                user_guidance = (
                    "ACTION: CLOSE REMAINING RUNNER. "
                    "TP1 partial profit was secured, but reversal "
                    "is now confirmed. Close the remaining runner "
                    "and do not wait for TP2."
                )

                backend_advice = (
                    "TP1 partial profit was secured, but price has "
                    "lost TP1/structure. Treat the remaining runner "
                    "as closed."
                )

            else:
                user_guidance = (
                    "ACTION: HOLD. "
                    "TP1 is confirmed. Continue monitoring the original TP2 plan."
                )
                backend_advice = "TP1 is confirmed. Continue monitoring."

            result.update({
                "guidance_status": decision.get("guidance_status", "TP1_HIT"),
                "risk_level": decision.get("risk_level", "LOW"),
                "action": decision.get("action", "HOLD_FOR_TP2"),
                "backend_advice": backend_advice,
                "user_guidance": user_guidance,
                "danger_score": danger_score,
                "danger_flags": flags,
                "management_mode": decision.get("management_mode", ""),
                "tp2_probability": decision.get("tp2_probability", ""),
                "reversal_risk": decision.get("reversal_risk", ""),
                "profit_action": decision.get("profit_action", ""),
                "sl_management": decision.get("sl_management", ""),
                "short_action_text": decision.get("short_action_text", ""),
                "exit_trigger": decision.get("exit_trigger", ""),
                "magnet_test_level": decision.get("magnet_test_level"),
                "magnet_sources": decision.get("magnet_sources", []),
                "entry_caution_level": entry_caution.get("entry_caution_level", "NORMAL"),
                "entry_caution_reason": entry_caution.get("entry_caution_reason", ""),
                "entry_caution_advice": entry_caution.get("entry_caution_advice", ""),
            })
            return result

        # TP1 not hit yet.
        if not tp1_hit:
            if danger_score >= 55:
                result.update({
                    "guidance_status": "HIGH_RISK_WARNING",
                    "risk_level": "HIGH",
                    "action": "REDUCE_OR_CLOSE",
                    "backend_advice": (
                        "Trade is active but has multiple warning signs before TP1. "
                        "Consider reducing risk or closing if your plan does not allow this drawdown."
                    ),
                    "user_guidance": _append_entry_caution_text(
                        "Warning: this trade has weakened before TP1. Consider reducing risk or closing if price action confirms against the setup.",
                        entry_caution
                    ),
                    "danger_score": danger_score,
                    "danger_flags": flags,
                })
                return result

            if danger_score >= 30:
                result.update({
                    "guidance_status": "CAUTION",
                    "risk_level": "MEDIUM",
                    "action": "HOLD_WITH_CAUTION",
                    "backend_advice": (
                        "Trade is active but showing caution signs before TP1. "
                        "Do not add size. Watch entry/invalidation closely."
                    ),
                    "user_guidance": _append_entry_caution_text(
                        "Trade is still active, but caution is needed. Do not add more size. Watch entry and invalidation closely.",
                        entry_caution
                    ),
                    "danger_score": danger_score,
                    "danger_flags": flags,
                })
                return result

            # Near TP1 but not hit: avoid wrong breakeven advice.
            if tp1_progress_pct >= 80:
                result.update({
                    "guidance_status": "NEAR_TP1",
                    "risk_level": "LOW",
                    "action": "HOLD_TO_TP1",
                    "backend_advice": (
                        "Price is near TP1 but TP1 is not confirmed yet. "
                        "Avoid moving SL blindly before target confirmation."
                    ),
                    "user_guidance": _append_entry_caution_text(
                        "Price is near TP1, but TP1 has not been confirmed yet. Avoid emotional exits or blind SL changes.",
                        entry_caution
                    ),
                    "danger_score": danger_score,
                    "danger_flags": flags,
                })
                return result

            result.update({
                "guidance_status": "RUNNING",
                "risk_level": "LOW",
                "action": "HOLD_TO_TP1",
                "backend_advice": (
                    "Trade is active and no major danger signs are detected. Hold toward TP1 according to the original plan."
                ),
                "user_guidance": _append_entry_caution_text(
                    "Trade is active. No major warning signs detected. Hold toward TP1 according to the original plan.",
                    entry_caution
                ),
                "danger_score": danger_score,
                "danger_flags": flags,
            })
            return result

        # Fallback active state.
        result.update({
            "guidance_status": "RUNNING",
            "risk_level": "MEDIUM" if danger_score >= 30 else "LOW",
            "action": "HOLD",
            "backend_advice": "Trade is active. Continue monitoring price action.",
            "user_guidance": "Trade is active. Continue monitoring price action.",
            "danger_score": danger_score,
            "danger_flags": flags,
        })
        return result

    except Exception as exc:
        result.update({
            "guidance_status": "ERROR",
            "risk_level": "UNKNOWN",
            "action": "WAIT",
            "backend_advice": f"Trade guidance failed: {exc}",
            "user_guidance": "Trade guidance failed. Use normal TP/SL rules.",
            "danger_score": 0,
            "danger_flags": ["GUIDANCE_ERROR"],
        })
        return result


def build_guidance_context_text(guidance: Dict[str, Any]) -> str:
    """
    Text block for AI prompt later, or debug output.
    """
    try:
        g = guidance or {}
        facts = g.get("facts") or {}

        lines = [
            "--- BACKEND TRADE GUIDANCE FACTS ---",
            f"Guidance Status: {g.get('guidance_status', '')}",
            f"Risk Level: {g.get('risk_level', '')}",
            f"Suggested Action: {g.get('action', '')}",
            f"Danger Score: {g.get('danger_score', 0)}",
            f"Danger Flags: {', '.join(g.get('danger_flags', []) or [])}",
            f"Backend Advice: {g.get('backend_advice', '')}",
            f"Management Mode: {g.get('management_mode', '')}",
            f"TP2 Probability: {g.get('tp2_probability', '')}",
            f"Reversal Risk: {g.get('reversal_risk', '')}",
            f"Profit Action: {g.get('profit_action', '')}",
            f"SL Management: {g.get('sl_management', '')}",
            f"Short Action Text: {g.get('short_action_text', '')}",
            f"Exit Trigger: {g.get('exit_trigger', '')}",
            f"Entry Caution Level: {g.get('entry_caution_level', facts.get('entry_caution_level', 'NORMAL'))}",
            f"Entry Caution Reason: {g.get('entry_caution_reason', facts.get('entry_caution_reason', ''))}",
            f"Entry Caution Advice: {g.get('entry_caution_advice', facts.get('entry_caution_advice', ''))}",
            "",
            f"Side: {facts.get('side', '')}",
            f"Entry: {_format_price(facts.get('entry'))}",
            f"Stop Loss: {_format_price(facts.get('stop_loss'))}",
            f"TP1: {_format_price(facts.get('tp1'))}",
            f"TP2: {_format_price(facts.get('tp2'))}",
            f"Current Price: {_format_price(facts.get('current_price'))}",
            f"Current PnL %: {facts.get('pnl_pct')}",
            f"Entry Hit: {facts.get('entry_hit')}",
            f"TP1 Hit: {facts.get('tp1_hit')}",
            f"TP2 Hit: {facts.get('tp2_hit')}",
            f"SL Hit: {facts.get('sl_hit')}",
            f"TP1 Progress %: {facts.get('tp1_progress_pct')}",
            f"TP2 Progress %: {facts.get('tp2_progress_pct')}",
            f"Returned To Entry After Profit: {facts.get('returned_to_entry_after_profit')}",
            f"Close Back Beyond Entry: {facts.get('close_back_beyond_entry')}",
            f"Opposite Pressure: {facts.get('opposite_pressure')}",
            f"Opposite Pressure Reason: {facts.get('opposite_pressure_reason')}",
            f"TP1 Lost After Hit: {facts.get('tp1_lost_after_hit')}",
            f"TP1 Magnet Has Level: {facts.get('tp1_magnet_has_level')}",
            f"TP1 Pending Magnet Test: {facts.get('tp1_pending_magnet_test')}",
            f"TP1 Magnet Breached Against Trade: {facts.get('tp1_magnet_breached_against_trade')}",
            f"TP1 Magnet Level: {_format_price(facts.get('tp1_magnet_level'))}",
            f"TP1 Magnet Sources: {', '.join(facts.get('tp1_magnet_sources', []) or [])}",
            f"TP1 Magnet Reason: {facts.get('tp1_magnet_reason')}",
            f"Confirm TF Opposite Pressure: {facts.get('confirm_opposite_pressure')}",
            f"Confirm TF Pressure Reason: {facts.get('confirm_pressure_reason')}",
            "",
            "Rules for explanation:",
            "- Do not create a new trade.",
            "- Do not change entry, SL, TP1, or TP2.",
            "- When TP1 is confirmed, secure partial profit and manage only the remaining runner.",
            "- Do not advise blind breakeven just because price is near TP1.",
            "- If TP1 is not confirmed, say TP1 is not confirmed.",
            "- If SL is hit, say the setup is invalidated/closed.",
        ]

        return "\n".join(lines)

    except Exception:
        return "--- BACKEND TRADE GUIDANCE FACTS ---\nGuidance unavailable."