"""
CRT / Range Sweep Model

Purpose:
- Detect range high/low sweeps, reclaim/rejection, breakdown retests,
  and breakout retests.
- Attach structured range-sweep context to an existing validated scenario.
- Support locked-trade monitoring and risk interpretation.

Important:
- Does NOT select trade direction.
- Does NOT change Entry, Stop Loss, TP1, or TP2.
- Does NOT create executable signals or watch setups.
- Does NOT mutate frontend summary/advice text.
- Does NOT save Firestore locks by itself.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import math
import pandas as pd


# ==========================================================
# Helpers
# ==========================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").replace("%", "").strip()
            if not value:
                return default

        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except Exception:
        return default


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


def _upper(value: Any) -> str:
    return str(value or "").upper().strip()


def _normalize_side(value: Any) -> str:
    s = _upper(value)

    if s in ["LONG", "BUY", "BULL", "BULLISH"] or "LONG" in s:
        return "LONG"

    if s in ["SHORT", "SELL", "BEAR", "BEARISH"] or "SHORT" in s:
        return "SHORT"

    return "NONE"


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

        out = out.rename(columns={c: rename_map.get(str(c).lower(), c) for c in out.columns})

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


def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        d = _normalize_ohlc(df)
        if len(d) < period + 2:
            return None

        prev_close = d["Close"].shift(1)
        tr1 = d["High"] - d["Low"]
        tr2 = (d["High"] - prev_close).abs()
        tr3 = (d["Low"] - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        val = tr.tail(period).mean()

        if pd.isna(val) or val <= 0:
            return None

        return float(val)

    except Exception:
        return None


def _scenario_side(scenario: Optional[Dict[str, Any]]) -> str:
    try:
        s = scenario or {}
        return _normalize_side(s.get("trade_type") or s.get("side"))
    except Exception:
        return "NONE"


def _alignment(crt_bias: str, setup_side: str) -> str:
    crt_bias = _normalize_side(crt_bias)
    setup_side = _normalize_side(setup_side)

    if crt_bias not in ["LONG", "SHORT"] or setup_side not in ["LONG", "SHORT"]:
        return "NEUTRAL"

    if crt_bias == setup_side:
        return "SUPPORTIVE"

    return "CONFLICT"

# ==========================================================
# Core CRT / Range detection
# ==========================================================

def evaluate_range_sweep(
    *,
    df_selected: Optional[pd.DataFrame],
    df_fast: Optional[pd.DataFrame] = None,
    user_timeframe: str = "",
    scenario: Optional[Dict[str, Any]] = None,
    lookback: int = 32,
) -> Dict[str, Any]:
    """
    Evaluate range sweep / breakdown-retest state.

    Uses selected TF as the primary structure source.
    Uses fast TF only as confirmation, if provided.
    """
    result: Dict[str, Any] = {
        "enabled": True,
        "mode": "CRT_RANGE_SWEEP",
        "user_timeframe": str(user_timeframe or ""),
        "state": "NO_CRT_SETUP",
        "bias": "NONE",
        "alignment_with_setup": "NEUTRAL",
        "entry_mode": "NO_CRT_ACTION",
        "confidence_adjustment": 0,
        "risk_adjustment": "NONE",
        "range_high": None,
        "range_low": None,
        "retest_level": None,
        "sweep_level": None,
        "sweep_type": "NONE",
        "chase_risk": "LOW",
        "fast_confirmation": "NOT_USED",
        "watch_only": True,
        "can_enter_now": False,
        "advice": "No useful CRT / range-sweep context detected.",
        "reason": "",
        "details": {},
    }

    try:
        d = _normalize_ohlc(df_selected)
        if len(d) < max(lookback + 5, 40):
            result.update({
                "enabled": False,
                "reason": "Not enough selected-timeframe candles for CRT / range-sweep detection.",
            })
            return result

        d = d.tail(max(lookback + 10, 80)).copy()

        # Use previous lookback candles as range, latest candle as decision candle.
        prev = d.iloc[-(lookback + 1):-1]
        last = d.iloc[-1]

        range_high = float(prev["High"].max())
        range_low = float(prev["Low"].min())

        last_open = float(last["Open"])
        last_high = float(last["High"])
        last_low = float(last["Low"])
        last_close = float(last["Close"])

        atr_val = _atr(d, period=14)
        range_size = max(range_high - range_low, 0.0)

        tol = 0.0
        if atr_val is not None:
            tol = atr_val * 0.08
        else:
            tol = abs(last_close) * 0.001

        swept_high = last_high > range_high + tol
        swept_low = last_low < range_low - tol

        closed_back_below_high = last_close < range_high
        closed_back_above_low = last_close > range_low

        close_below_range = last_close < range_low - tol
        close_above_range = last_close > range_high + tol

        body = abs(last_close - last_open)
        candle_range = max(last_high - last_low, 0.0)
        body_ratio = body / candle_range if candle_range > 0 else 0.0

        selected_state = "NO_CRT_SETUP"
        retest_level = None
        sweep_level = None
        sweep_type = "NONE"
        bias = "NONE"

        # 1) Clean sweep/reclaim/rejection
        if swept_high and closed_back_below_high:
            selected_state = "RANGE_HIGH_SWEEP_REJECTION"
            bias = "SHORT"
            retest_level = range_high
            sweep_level = last_high
            sweep_type = "HIGH_SWEEP_REJECTION"

        elif swept_low and closed_back_above_low:
            selected_state = "RANGE_LOW_SWEEP_RECLAIM"
            bias = "LONG"
            retest_level = range_low
            sweep_level = last_low
            sweep_type = "LOW_SWEEP_RECLAIM"

        # 2) Breakdown / breakout continuation watch
        elif close_below_range:
            selected_state = "BREAKDOWN_RETEST_SHORT"
            bias = "SHORT"
            retest_level = range_low
            sweep_level = range_low
            sweep_type = "BREAKDOWN"

        elif close_above_range:
            selected_state = "BREAKOUT_RETEST_LONG"
            bias = "LONG"
            retest_level = range_high
            sweep_level = range_high
            sweep_type = "BREAKOUT"

        # 3) Strong directional candle inside/near range
        else:
            # Avoid over-triggering. Internal range candles are neutral.
            selected_state = "NO_CRT_SETUP"
            bias = "NONE"

        # Fast confirmation
        fast_confirmation = "NOT_USED"
        try:
            f = _normalize_ohlc(df_fast)
            if not f.empty and len(f) >= 10 and bias in ["LONG", "SHORT"]:
                recent = f.tail(3)

                green = 0
                red = 0

                for _, r in recent.iterrows():
                    if float(r["Close"]) > float(r["Open"]):
                        green += 1
                    elif float(r["Close"]) < float(r["Open"]):
                        red += 1

                if bias == "SHORT":
                    if red >= 2:
                        fast_confirmation = "SUPPORTIVE"
                    elif green >= 2:
                        fast_confirmation = "CONFLICT"
                    else:
                        fast_confirmation = "NEUTRAL"

                elif bias == "LONG":
                    if green >= 2:
                        fast_confirmation = "SUPPORTIVE"
                    elif red >= 2:
                        fast_confirmation = "CONFLICT"
                    else:
                        fast_confirmation = "NEUTRAL"
        except Exception:
            fast_confirmation = "UNKNOWN"

        # Compare CRT bias only with the already validated scenario side.
        # This module must never infer a trade side from legacy planner data,
        # free-form WAIT text, or crash-regime hints.
        setup_side = _scenario_side(scenario)
        align = _alignment(bias, setup_side)

        confidence_adjustment = 0
        risk_adjustment = "NONE"
        entry_mode = "NO_CRT_ACTION"

        if selected_state == "NO_CRT_SETUP":
            entry_mode = "NORMAL_ENTRY_UNCHANGED"
            advice = "No strong CRT sweep/retest condition. Keep normal system rules."
            reason = "No range high/low sweep, breakdown retest, or breakout retest was detected."

        elif align == "SUPPORTIVE":
            confidence_adjustment = 6
            risk_adjustment = "LOWER_IF_ENTRY_CONFIRMS"

            if selected_state in ["RANGE_HIGH_SWEEP_REJECTION", "RANGE_LOW_SWEEP_RECLAIM"]:
                entry_mode = "CRT_SWEEP_CONFIRMS_SETUP"
                advice = (
                    f"CRT range sweep supports the planned {setup_side}. "
                    f"Watch { _fmt_price(retest_level) } as the reclaim/rejection decision level."
                )
            else:
                entry_mode = "WAIT_RETEST_CONFIRMATION"
                advice = (
                    f"CRT continuation supports {setup_side}, but current price may be extended. "
                    f"Do not chase. Wait for retest/rejection near { _fmt_price(retest_level) }."
                )

            reason = f"CRT bias {bias} aligns with setup/candidate side {setup_side}."

        elif align == "CONFLICT":
            confidence_adjustment = -8
            risk_adjustment = "HIGHER"
            entry_mode = "WAIT_FOR_RECONFIRMATION"
            advice = (
                f"CRT range behavior conflicts with the planned {setup_side}. "
                "Do not use full-risk entry until fresh confirmation appears."
            )
            reason = f"CRT bias {bias} conflicts with setup/candidate side {setup_side}."

        else:
            # No validated LONG/SHORT side is available for comparison.
            # Keep the detector output informational only; do not infer or
            # create a replacement direction/watch setup.
            entry_mode = "INFORMATION_ONLY"
            confidence_adjustment = 0
            risk_adjustment = "NONE"

            advice = (
                f"{bias} CRT/range behavior was detected near "
                f"{_fmt_price(retest_level)}, but there is no validated "
                "LONG/SHORT scenario side to compare against."
            )

            reason = (
                "Range-sweep state detected without an active validated "
                "directional scenario."
            )

        chase_risk = "LOW"
        try:
            if selected_state in ["BREAKDOWN_RETEST_SHORT", "BREAKOUT_RETEST_LONG"]:
                chase_risk = "HIGH"
            elif selected_state in ["RANGE_HIGH_SWEEP_REJECTION", "RANGE_LOW_SWEEP_RECLAIM"]:
                chase_risk = "MEDIUM"
        except Exception:
            chase_risk = "UNKNOWN"

        result.update({
            "state": selected_state,
            "bias": bias,
            "alignment_with_setup": align,
            "entry_mode": entry_mode,
            "confidence_adjustment": confidence_adjustment,
            "risk_adjustment": risk_adjustment,
            "range_high": range_high,
            "range_low": range_low,
            "retest_level": retest_level,
            "sweep_level": sweep_level,
            "sweep_type": sweep_type,
            "chase_risk": chase_risk,
            "fast_confirmation": fast_confirmation,
            "watch_only": True,
            "can_enter_now": False,
            "advice": advice,
            "reason": reason,
            "details": {
                "last_open": last_open,
                "last_high": last_high,
                "last_low": last_low,
                "last_close": last_close,
                "atr": atr_val,
                "range_size": range_size,
                "body_ratio": round(body_ratio, 3),
                "setup_side_used": setup_side,
            },
        })

        return result

    except Exception as exc:
        result.update({
            "enabled": False,
            "state": "ERROR",
            "bias": "NONE",
            "entry_mode": "ERROR",
            "advice": "CRT / Range Sweep check failed.",
            "reason": str(exc),
        })
        return result


# ==========================================================
# Scenario helper
# ==========================================================

def add_range_sweep_to_scenario(
    *,
    scenario: Dict[str, Any],
    interval: str,
    df_selected: Optional[pd.DataFrame],
    df_fast: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    try:
        if not isinstance(scenario, dict):
            return scenario

        crt = evaluate_range_sweep(
            df_selected=df_selected,
            df_fast=df_fast,
            user_timeframe=interval,
            scenario=scenario,
        )

        # Structured monitoring context only.
        # Execution-readiness and other downstream layers may read this,
        # but this module must not rewrite final frontend text.
        scenario["range_sweep"] = crt

        return scenario

    except Exception as exc:
        try:
            scenario["range_sweep"] = {
                "enabled": False,
                "state": "ERROR",
                "advice": "CRT / Range Sweep attach failed.",
                "reason": str(exc),
            }
        except Exception:
            pass
        return scenario