import sys
import json
import os
import re
import time
from copy import deepcopy
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# --- IMPORT CONFIG ---
from . import config

AI_ANALYSIS_DEADLINE_SECONDS = 210.0
AI_CORRECTION_MIN_REMAINING_SECONDS = 35.0

# Firebase Admin Imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Models imports
from .data_loader import get_data
from .models.WaveAnalyzer import WaveAnalyzer
from .models.WaveOptions import WaveOptionsGenerator5
from .models.WaveRules import Impulse, Correction
from .models.WavePattern import WavePattern
from .ai_engine import (
    analyze_scanner_candidates,
    generate_trade_guidance_advice,
)
from .models.trade_guidance_manager import (
    build_trade_guidance_facts,
    build_guidance_context_text,
)
from .scanner import scan_market
from .models.market_context import build_market_context_text
from .candle_context import build_recent_candle_rejection_context
from .models.trend_direction import build_dominant_trend_context
from .models.trend_regime import (
    detect_backend_trend_regime,
    format_backend_trend_regime_context
)
from .models.entry_confirmation import evaluate_entry_confirmation
from .models.direction_confidence import evaluate_direction_confidence
from .models.tp_path_validator import evaluate_tp_path_quality
from .models.imbalance import add_imbalance_to_scenario, add_imbalance_to_ai_data
from .models.liquidity import (
    add_liquidity_to_scenario,
    add_liquidity_to_ai_data,
    build_side_neutral_liquidity_context,
    evaluate_liquidity_risk,
)
from .models.crash_regime import add_crash_regime_to_scenario, add_crash_regime_to_ai_data
from .models.range_sweep_model import add_range_sweep_to_scenario
from .models.smc_context import (
    add_smc_context_to_scenario,
    add_smc_context_to_ai_data,
    build_side_neutral_smc_evidence,
)
from .models.decision_audit import add_decision_audit_to_scenario, add_decision_audit_to_ai_data
from .models.elliott_phase_detector import (
    detect_elliott_phase_context,
    format_elliott_phase_context_text,
)
from .models.momentum_breakout_detector import (
    detect_momentum_breakout,
)
from .models.rsi_divergence_detector import (
    detect_rsi_divergence,
    format_rsi_divergence_context_text,
)
from .models.context_bundle import (
    build_context_bundle,
    summarize_context_bundle
)
from .models.historical_pattern_engine import (
    analyze_historical_analogs,
    build_gemini_analog_payload,
)
from .models.ai_market_packet import (
    build_ai_market_packet,
    validate_ai_market_packet,
    sanitize_ai_market_packet,
)
from .models.ai_signal_engine import (
    generate_ai_signal_draft,
    regenerate_ai_signal,
)

from .models.ai_signal_validator import (
    validate_ai_signal,
    build_safe_no_trade_signal,
)
from .trade_management import (
    calculate_running_pnl_percent,
    format_running_pnl_advice,
)

from .trade_audit import audit_open_trade, audit_close_trade

# --- FIREBASE SETUP ---
FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    "serviceAccountKey.json",
)

db = None
FIREBASE_AVAILABLE = False

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            FIREBASE_CREDENTIALS_PATH
        )

        firebase_admin.initialize_app(
            cred
        )

    db = firestore.client()
    FIREBASE_AVAILABLE = True

except Exception as e:
    db = None
    FIREBASE_AVAILABLE = False

    print(
        f"[WARN] Firebase setup failed: {e}"
    )


COLLECTION_NAME = "global_analysis_locks"

# --- HELPER FUNCTIONS ---

def get_dynamic_timeframes(interval):
    """
    LEAD AI v2: Strict Timeframe Scaling (15m, 1H, 4H, 1D ONLY)
    Adjusts Confirm, Swing, and Macro timeframes based on the allowed UI intervals.
    """
    interval = str(interval).lower().strip()
    
    # 1. Base / Default Setup (15m)
    tf_map = {
        "micro": {"tf": "15m", "pd": "1mo"},
        "confirm": {"tf": "1h", "pd": "3mo"},
        "swing": {"tf": "4h", "pd": "6mo"},
        "macro": {"tf": "1d", "pd": "2y"}
    }
    
    # 2. If User Selects 1H
    if interval in ["1h", "60m"]:
        tf_map = {
            "micro": {"tf": "1h", "pd": "3mo"},
            "confirm": {"tf": "4h", "pd": "6mo"},
            "swing": {"tf": "1d", "pd": "2y"},
            "macro": {"tf": "1w", "pd": "5y"}
        }
        
    # 3. If User Selects 4H
    elif interval in ["4h", "240m"]:
        tf_map = {
            "micro": {"tf": "4h", "pd": "6mo"},
            "confirm": {"tf": "1d", "pd": "2y"},
            "swing": {"tf": "1w", "pd": "5y"},
            "macro": {"tf": "1mo", "pd": "max"}
        }
        
    # 4. If User Selects 1D
    elif interval in ["1d", "1D", "d", "daily"]:
        tf_map = {
            "micro": {"tf": "1d", "pd": "2y"},
            "confirm": {"tf": "1w", "pd": "5y"},
            "swing": {"tf": "1mo", "pd": "max"},
            "macro": {"tf": "1mo", "pd": "max"}
        }
        
    return tf_map

def get_lower_execution_timeframe(interval):
    """
    Lower timeframe used only for early momentum breakout timing.

    User TF 1h -> check 15m
    User TF 4h -> check 1h
    User TF 1d -> check 4h
    15m stays None for now. Later we can add 5m.
    """
    tf = str(interval or "").lower().strip()

    if tf in ["1h", "60m"]:
        return {"tf": "15m", "pd": "1mo"}

    if tf in ["4h", "240m"]:
        return {"tf": "1h", "pd": "3mo"}

    if tf in ["1d", "d", "daily"]:
        return {"tf": "4h", "pd": "6mo"}

    return None

def get_crash_fast_timeframe_from_existing_lower(interval):
    """
    Crash regime fast timeframe using only candles already fetched by analyze/status flow.

    Important:
    - No extra API calls.
    - Uses selected TF + existing lower execution TF.
    - Does NOT affect setup generation.
    - Does NOT affect entry/SL/TP.
    - Does NOT block signals.

    Current mapping follows get_lower_execution_timeframe():
      15m -> selected TF only
      1h  -> 1h + already fetched 15m
      4h  -> 4h + already fetched 1h
      1d  -> 1d + already fetched 4h
    """
    lower = get_lower_execution_timeframe(interval)

    if isinstance(lower, dict):
        return lower.get("tf")

    return None

def get_guidance_confirm_timeframe(interval):
    """
    Lightweight advice refresh timeframe mapping.
    This is ONLY for active trade guidance refresh.
    It does NOT affect fresh analyze / setup generation.

    15m trade -> 1h confirm
    1h trade  -> 4h confirm
    4h trade  -> 1d confirm
    """
    tf = str(interval or "").lower().strip()

    if tf in ["15m"]:
        return {"tf": "1h", "pd": "1mo"}

    if tf in ["1h", "60m"]:
        return {"tf": "4h", "pd": "3mo"}

    if tf in ["4h", "240m"]:
        return {"tf": "1d", "pd": "2y"}

    if tf in ["1d", "d", "daily"]:
        return {"tf": "1w", "pd": "5y"}

    return {"tf": "1h", "pd": "1mo"}

def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbols for consistent API/database usage.
    Examples:
      BTC/USDT -> BTCUSDT
      btcusdt  -> BTCUSDT
      BTC:USDT -> BTCUSDT
    """
    return str(symbol).replace("/", "").replace(":", "").replace(" ", "").upper()


def load_api_key():
    if not FIREBASE_AVAILABLE:
        print("[ERROR] Firebase not connected.")
        return False
    try:
        doc_ref = db.collection("app_secrets").document("config")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            pool = data.get("gemini_api_key_pool")
            if pool and isinstance(pool, list):
                config.API_KEY_POOL = pool
            else:
                config.API_KEY_POOL = []

            paid_key = data.get("gemini_paid_key")
            if paid_key:
                config.PAID_API_KEY = paid_key
            else:
                config.PAID_API_KEY = None

            if config.API_KEY_POOL or config.PAID_API_KEY:
                # Keys loaded successfully. Do not print key count/suffixes in normal logs.
                return True

            
        else:
            print("[ERROR] Secret config not found.")
    except Exception as e:
        print(f"[ERROR] Failed to load API Keys: {e}")
    return False


def get_firestore_lock(symbol, tf):
    if not FIREBASE_AVAILABLE:
        return None, None
    try:
        clean_symbol = normalize_symbol(symbol)
        doc_id = f"{clean_symbol}_{tf}"
        doc_ref = db.collection(COLLECTION_NAME).document(doc_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("scenario"), data.get("created_at") or data.get("updated_at")
    except Exception as e:
        print(f"[WARN] Failed to get Firebase lock: {e}")
    return None, None


def get_cross_timeframe_setups(symbol, current_tf):
    """
    Return structured same-symbol scenarios from other timeframes.
    Used as AI market evidence and for cross-timeframe risk warnings.

    Example:
    15m LONG SL is above 1H LONG entry
    -> 15m is still usable, but should be SCALP_ONLY / caution.
    """
    if not FIREBASE_AVAILABLE:
        return []

    try:
        clean_symbol = normalize_symbol(symbol)
        docs = db.collection(COLLECTION_NAME).where("symbol", "==", clean_symbol).stream()

        setups = []

        for doc in docs:
            data = doc.to_dict() or {}
            tf = data.get("timeframe")

            if tf == current_tf:
                continue

            scenario = data.get("scenario") or {}
            trade_type = str(
                scenario.get("trade_type") or scenario.get("side") or ""
            ).upper()

            if trade_type not in ["LONG", "SHORT"]:
                continue

            entry = extract_price(
                scenario.get("entry")
                or scenario.get("entry_zone")
            )
            stop = extract_price(
                scenario.get("stop_loss")
                or scenario.get("invalidation")
            )

            targets = scenario.get("targets") or []
            tp1 = extract_price(targets[0]) if targets else extract_price(scenario.get("target"))

            if entry is None or stop is None:
                continue

            setups.append({
                "timeframe": tf,
                "side": trade_type,
                "entry": entry,
                "stop_loss": stop,
                "tp1": tp1,
                "status": scenario.get("status"),
                "setup_type": scenario.get("setup_type"),
                "quality_score": scenario.get("quality_score"),
            })

        return setups

    except Exception as e:
        print(f"[WARN] Error fetching structured cross-tf setups: {e}")
        return []

def build_cross_timeframe_risk_warning(symbol, current_tf, current_setup):
    """
    Add a user-facing warning when a lower-timeframe setup overlaps
    with a saved higher-timeframe setup on the same symbol.

    This does NOT block the trade.
    It only adds risk-management context to summary/advice.
    """
    try:
        if not current_setup:
            return ""

        side = str(
            current_setup.get("trade_type")
            or current_setup.get("side")
            or ""
        ).upper()

        if side not in ["LONG", "SHORT"]:
            return ""

        current_sl = extract_price(
            current_setup.get("stop_loss")
            or current_setup.get("invalidation")
        )

        if current_sl is None:
            return ""

        tf = str(current_tf or "").lower().strip()

        # Main use-case: 15m setup vs saved 1h setup.
        if tf not in ["15m", "15"]:
            return ""

        higher_setups = get_cross_timeframe_setups(symbol, current_tf)

        # Prefer saved 1h setup with same direction.
        htf = None
        for s in higher_setups:
            s_tf = str(s.get("timeframe") or "").lower().strip()
            s_side = str(s.get("side") or "").upper()

            if s_tf in ["1h", "60m"] and s_side == side:
                htf = s
                break

        # If 15m is generated first and no 1h setup exists yet.
        if not htf:
            return (
                "⚠️ Lower-timeframe caution: No saved 1h setup is available for comparison. "
                "Treat this as a 15m trade only unless higher-timeframe confirmation is checked. "
                "Do not assume the 1h structure is aligned."
            )

        htf_entry = extract_price(htf.get("entry"))

        if htf_entry is None:
            return ""

        # LONG: saved 1h entry is below 15m stop.
        if side == "LONG" and htf_entry < current_sl:
            return (
                f"⚠️ Cross-timeframe warning: Saved 1h LONG entry "
                f"({format_signal_price(htf_entry)}) is below this 15m stop "
                f"({format_signal_price(current_sl)}). "
                "This 15m setup is aggressive/scalp-only. If 15m fails, price may still reach the 1h pullback zone. "
                "Do not double risk on both setups."
            )

        # SHORT: saved 1h entry is above 15m stop.
        if side == "SHORT" and htf_entry > current_sl:
            return (
                f"⚠️ Cross-timeframe warning: Saved 1h SHORT entry "
                f"({format_signal_price(htf_entry)}) is above this 15m stop "
                f"({format_signal_price(current_sl)}). "
                "This 15m setup is aggressive/scalp-only. If 15m fails, price may still reach the 1h pullback zone. "
                "Do not double risk on both setups."
            )

        # Same direction 1h exists, but not beyond 15m stop.
        return (
            f"ℹ️ Cross-timeframe note: A saved 1h {side} setup also exists. "
            "Manage total exposure as one trade idea; avoid stacking full risk on both timeframes."
        )

    except Exception as e:
        print(f"[XTF_WARNING] skipped/error: {e}")
        return ""


def save_firestore_lock(
    symbol,
    tf,
    scenario,
):
    if (
        not FIREBASE_AVAILABLE
        or db is None
    ):
        return (
            False,
            "FIREBASE_UNAVAILABLE",
        )

    try:
        clean_symbol = normalize_symbol(
            symbol
        )

        doc_id = (
            f"{clean_symbol}_{tf}"
        )

        doc_ref = (
            db
            .collection(
                COLLECTION_NAME
            )
            .document(
                doc_id
            )
        )

        doc_ref.set(
            {
                "scenario": scenario,
                "created_at": (
                    firestore.SERVER_TIMESTAMP
                ),
                "updated_at": (
                    firestore.SERVER_TIMESTAMP
                ),
                "symbol": clean_symbol,
                "timeframe": tf,
            }
        )

        return (
            True,
            "LOCK_SAVED",
        )

    except Exception as e:
        error_text = str(
            e
        )

        print(
            "[ERROR] Failed to save "
            f"Firebase lock: {error_text}"
        )

        return (
            False,
            error_text,
        )

def get_recent_locked_signals(limit=5):
    """
    Return latest locked global signals from global_analysis_locks.
    Used by frontend Recently tab.
    Only returns minimal card data:
    symbol | timeframe | side
    """
    if not FIREBASE_AVAILABLE:
        return {
            "signals": [],
            "error": "Firebase not connected."
        }

    try:
        limit = int(limit or 5)
        limit = max(1, min(limit, 10))

        docs = (
            db.collection(COLLECTION_NAME)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit * 3)  # read extra because we skip WAIT/unknown
            .stream()
        )

        signals = []

        for doc in docs:
            data = doc.to_dict() or {}
            scenario = data.get("scenario") or {}

            doc_id = doc.id

            symbol = (
                data.get("symbol")
                or scenario.get("symbol")
                or ""
            )

            timeframe = (
                data.get("timeframe")
                or scenario.get("timeframe")
                or ""
            )

            # fallback from doc id: BTCUSDT_1h
            if (not symbol or not timeframe) and "_" in doc_id:
                parts = doc_id.rsplit("_", 1)
                if len(parts) == 2:
                    symbol = symbol or parts[0]
                    timeframe = timeframe or parts[1]

            side = (
                scenario.get("trade_type")
                or scenario.get("side")
                or scenario.get("backend_preferred_side")
                or data.get("side")
                or ""
            )

            side = str(side or "").upper().strip()

            # Only show actual LONG/SHORT locked signals
            if side not in ["LONG", "SHORT"]:
                continue

            signals.append({
                "id": doc_id,
                "symbol": str(symbol or "").upper(),
                "timeframe": str(timeframe or ""),
                "side": side,
            })

            if len(signals) >= limit:
                break

        return {
            "signals": signals
        }

    except Exception as e:
        print(f"[RECENT_LOCKED_SIGNALS] Error: {e}")
        return {
            "signals": [],
            "error": str(e)
        }

def delete_firestore_lock(symbol, tf):
    if not FIREBASE_AVAILABLE:
        return
    try:
        clean_symbol = normalize_symbol(symbol)
        doc_id = f"{clean_symbol}_{tf}"
        db.collection(COLLECTION_NAME).document(doc_id).delete()
    except Exception as e:
        print(f"Error deleting lock: {e}")

def update_firestore_lock_guidance(symbol, tf, scenario, guidance_facts=None, ai_guidance=None):
    """
    Update only guidance fields on an existing global lock.
    Important: Do NOT reset created_at, because created_at is used as lock_time.
    """
    if not FIREBASE_AVAILABLE:
        return

    try:
        clean_symbol = normalize_symbol(symbol)
        doc_id = f"{clean_symbol}_{tf}"
        doc_ref = db.collection(COLLECTION_NAME).document(doc_id)

        payload = {
            "scenario.generated_advice": scenario.get("generated_advice", ""),
            "scenario.trade_guidance": guidance_facts or {},
            "scenario.ai_trade_guidance": ai_guidance or {},
            "updated_at": firestore.SERVER_TIMESTAMP,
        }

        doc_ref.update(payload)

    except Exception as e:
        print(f"[WARN] Failed to update guidance on Firebase lock: {e}")


def extract_price(price_str):
    try:
        if isinstance(price_str, (int, float)):
            return float(price_str)
        match = re.search(r"[\d,]+(\.\d+)?", str(price_str))
        if match:
            return float(match.group(0).replace(',', ''))
    except Exception:
        return None
    return None

def format_signal_price(value):
    """
    Dynamic price formatting for signal values.
    Prevents low-price coins like TRX/DOGE/PEPE from becoming 0.33 / 0.00.
    """
    value = extract_price(value)
    if value is None:
        return ""

    if abs(value) >= 1000:
        return f"{value:.2f}"
    elif abs(value) >= 100:
        return f"{value:.2f}"
    elif abs(value) >= 10:
        return f"{value:.3f}"
    elif abs(value) >= 1:
        return f"{value:.4f}"
    elif abs(value) >= 0.01:
        return f"{value:.4f}"
    elif abs(value) >= 0.0001:
        return f"{value:.6f}"
    else:
        return f"{value:.8f}"


def latest_close_price(df):
    try:
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return extract_price(df.iloc[-1]["Close"])
    except Exception:
        return None


def _entry_confirmation_analysis_df(lower_df, *, interval=None):
    try:
        if lower_df is None or lower_df.empty:
            return pd.DataFrame()

        d = lower_df.copy()
        if "Date" in d.columns:
            d.set_index("Date", inplace=True)

        _, analysis = split_live_and_analysis_df(
            d,
            min_rows=80,
            interval=interval,
        )
        return analysis

    except Exception:
        return pd.DataFrame()


def _entry_zone_touched_for_confirmation(side, entry_price, lower_df):
    try:
        side = str(side or "").upper().strip()
        entry = extract_price(entry_price)

        if entry is None or lower_df is None or lower_df.empty:
            return False

        d = lower_df.copy()
        rename_map = {"high": "High", "low": "Low"}
        d = d.rename(columns={c: rename_map.get(str(c).lower(), c) for c in d.columns})

        if side == "LONG" and "Low" in d.columns:
            low = extract_price(d.iloc[-1]["Low"])
            return bool(low is not None and low <= entry)

        if side == "SHORT" and "High" in d.columns:
            high = extract_price(d.iloc[-1]["High"])
            return bool(high is not None and high >= entry)

    except Exception:
        pass

    return False


def add_entry_confirmation_to_scenario(
    scenario,
    interval,
    lower_df,
    current_price=None,
    entry_zone_touched=None,
):
    """
    Attach shadow entry-confirmation advice to a response scenario only.
    This must not be used to alter status, save decisions, or trade prices.
    """
    try:
        if not isinstance(scenario, dict):
            return scenario

        side = str(scenario.get("trade_type") or scenario.get("side") or "").upper().strip()
        if side not in ["LONG", "SHORT"]:
            return scenario

        entry = extract_price(scenario.get("entry_zone") or scenario.get("entry"))
        stop = extract_price(scenario.get("invalidation") or scenario.get("stop_loss"))
        if entry is None:
            return scenario

        raw_values = scenario.get("raw_values") or {}
        atr = extract_price(raw_values.get("atr") or scenario.get("atr"))
        current = (
            extract_price(current_price)
            or extract_price(raw_values.get("current_price"))
            or latest_close_price(lower_df)
        )

        if entry_zone_touched is None:
            entry_zone_touched = _entry_zone_touched_for_confirmation(
                side,
                entry,
                lower_df,
            )

        scenario["entry_confirmation"] = evaluate_entry_confirmation(
            side=side,
            entry_price=entry,
            stop_loss=stop,
            user_timeframe=interval,
            lower_df=lower_df,
            current_price=current,
            entry_zone_touched=entry_zone_touched,
            atr=atr,
        )
    except Exception as e:
        try:
            scenario["entry_confirmation"] = {
                "enabled": True,
                "user_timeframe": str(interval or "").lower().strip(),
                "confirmation_timeframe": "unknown",
                "entry_zone_touched": False,
                "confirmation_status": "UNKNOWN",
                "confirmation_score": 0,
                "would_activate_by_confirmation": False,
                "advice": "Entry confirmation unavailable because shadow evaluation failed.",
                "reason": str(e),
                "signals": {
                    "close_reclaim": False,
                    "wick_rejection": False,
                    "ema_hold_or_reclaim": False,
                    "invalid_close": False,
                    "momentum_close": False,
                },
            }
        except Exception:
            pass

    return scenario


def add_entry_confirmation_to_ai_data(ai_data, interval, lower_df, current_price=None):
    try:
        if not isinstance(ai_data, dict):
            return ai_data

        scenarios = ai_data.get("trade_scenarios") or []
        if not scenarios:
            return ai_data

        scenario = scenarios[0]
        if not isinstance(scenario, dict):
            return ai_data

        scenario = add_entry_confirmation_to_scenario(
            scenario,
            interval,
            lower_df,
            current_price=current_price,
        )

        scenarios[0] = scenario
        ai_data["trade_scenarios"] = scenarios

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            selected_side = str(selected.get("trade_type") or selected.get("side") or "").upper().strip()
            scenario_side = str(scenario.get("trade_type") or scenario.get("side") or "").upper().strip()
            if selected_side == scenario_side:
                ai_data["selected_scenario"] = scenario
    except Exception as e:
        print(f"[ENTRY_CONFIRMATION] attach skipped/error: {e}")

    return ai_data

def apply_balanced_liquidity_execution_guard(
    *,
    signal,
    interval,
    df_analysis,
    df_lower,
    backend_trend_regime=None,
    current_price=None,
):
    """Attach liquidity evidence without overriding a validated AI decision."""
    candidate = deepcopy(signal) if isinstance(signal, dict) else {}
    regime = backend_trend_regime or {}

    context = {
        "enabled": True,
        "policy": "LIQUIDITY_ADVISORY_ONLY",
        "action": "UNCHANGED",
        "candidate_side": str(candidate.get("side") or "NONE").upper(),
        "trend_trade_mode": str(
            regime.get("allowed_trade_mode") or "FULL_TRADE"
        ).upper().strip(),
        "risk_level": "UNKNOWN",
        "high_risk_confirmed": False,
        "required_sweep_level": None,
        "supportive_sweep_event": {},
        "entry_confirmation": {},
        "reason": "No directional trade plan to assess.",
    }

    try:
        decision = str(candidate.get("decision") or "NO_TRADE").upper().strip()
        side = str(candidate.get("side") or "NONE").upper().strip()
        entry = extract_price(candidate.get("entry_price"))
        stop = extract_price(candidate.get("stop_loss"))

        if (
            decision not in {"LONG", "SHORT", "WAIT"}
            or side not in {"LONG", "SHORT"}
            or entry is None or entry <= 0
            or stop is None or stop <= 0
        ):
            return candidate, context

        liquidity = evaluate_liquidity_risk(
            side=side,
            entry_price=entry,
            stop_loss=stop,
            user_timeframe=interval,
            df_analysis=df_analysis,
            df_lower=df_lower,
            current_price=(
                extract_price(current_price)
                or latest_close_price(df_analysis)
            ),
        )

        context.update({
            "action": "ADVISORY_ONLY",
            "risk_level": str(
                liquidity.get("risk_level") or "UNKNOWN"
            ).upper(),
            "high_risk_confirmed": bool(
                liquidity.get("high_risk_confirmed")
            ),
            "supportive_sweep_event": dict(
                liquidity.get("supportive_sweep_event") or {}
            ),
            "liquidity_reason": str(liquidity.get("reason") or ""),
            "suggested_sweep_level": liquidity.get("suggested_sweep_level"),
            "reason": (
                "Liquidity evidence is advisory; the validated AI decision "
                "and trade prices are unchanged. No additional mandatory "
                "sweep/reclaim/retest sequence is imposed."
            ),
        })

    except Exception as exc:
        context["action"] = "ADVISORY_UNAVAILABLE"
        context["reason"] = str(exc)[:300]

    return candidate, context



def build_non_actionable_entry_confirmation(interval, status, entry_hit_fact=False):
    """
    Entry confirmation is only actionable while a setup is still pending.
    For RUNNING / TP1 HIT / WON / LOST / MISSED / CANCELLED / EXPIRED states,
    do not ask the user to wait for fresh entry confirmation.
    """
    tf = str(interval or "").lower().strip()

    if tf in ["1h", "60m"]:
        confirmation_tf = "15m"
    elif tf in ["4h", "240m"]:
        confirmation_tf = "1h"
    else:
        confirmation_tf = "unknown"

    status_text = str(status or "").upper().strip()

    return {
        "enabled": True,
        "user_timeframe": tf,
        "confirmation_timeframe": confirmation_tf,
        "entry_zone_touched": bool(entry_hit_fact),
        "confirmation_status": "ALREADY_ACTIVE",
        "confirmation_score": 0,
        "would_activate_by_confirmation": False,
        "advice": (
            "Trade is already active or no longer pending. "
            "Entry confirmation is not actionable now. "
            "Use active trade management, TP path guidance, and risk controls instead."
        ),
        "reason": f"Current backend tracking status is {status_text}.",
        "signals": {
            "close_reclaim": False,
            "wick_rejection": False,
            "ema_hold_or_reclaim": False,
            "invalid_close": False,
            "momentum_close": False,
            "large_opposite_body": False,
            "touch_found_in_lookback": None,
            "candles_since_entry_touch": None,
            "confirmation_window_expired": None,
        },
    }

def add_direction_confidence_to_scenario(
    scenario,
    interval,
    backend_trend_regime=None,
    mtf_elliott_context=None,
    momentum_context=None,
    rsi_divergence_context=None,
    market_context=None,
):
    """
    Attach shadow direction-confidence advice to a response scenario only.

    This must NOT:
    - approve/reject trades
    - change setup selection
    - change entry/SL/TP
    - change Firestore lock behavior
    - change AI/Gemini decisions
    """
    try:
        if not isinstance(scenario, dict):
            return scenario

        side = str(
            scenario.get("trade_type")
            or scenario.get("side")
            or ""
        ).upper().strip()

        if side not in ["LONG", "SHORT"]:
            return scenario

        # Prefer explicit function argument, then scenario-embedded backend regime.
        regime = (
            backend_trend_regime
            or scenario.get("backend_trend_regime")
            or {
                "preferred_side": scenario.get("backend_preferred_side"),
                "blocked_side": scenario.get("backend_blocked_side"),
                "confidence": scenario.get("backend_regime_confidence"),
                "market_bias": scenario.get("backend_market_bias"),
                "market_phase": scenario.get("backend_market_phase"),
                "regime": scenario.get("backend_regime"),
            }
        )

        # Scenario itself contains MTF Elliott fields in current schema.
        mtf_context = mtf_elliott_context or {
            "mtf_elliott_alignment": scenario.get("mtf_elliott_alignment"),
            "mtf_wave_side": scenario.get("mtf_wave_side"),
            "mtf_wave_phase": scenario.get("mtf_wave_phase"),
            "mtf_confirmation_score": scenario.get("mtf_confirmation_score"),
            "mtf_confirmation_reason": scenario.get("mtf_confirmation_reason"),
        }

        scenario["direction_confidence"] = evaluate_direction_confidence(
            candidate_side=side,
            user_timeframe=interval,
            backend_trend_regime=regime,
            setup=scenario,
            mtf_elliott_context=mtf_context,
            momentum_context=momentum_context,
            rsi_divergence_context=rsi_divergence_context,
            market_context=market_context,
        )

    except Exception as e:
        try:
            scenario["direction_confidence"] = {
                "enabled": True,
                "user_timeframe": str(interval or "").lower().strip(),
                "candidate_side": str(
                    scenario.get("trade_type")
                    or scenario.get("side")
                    or ""
                ).upper().strip(),
                "primary_direction": "UNKNOWN",
                "direction_confidence": 0,
                "trend_state": "UNKNOWN",
                "htf_alignment": "UNKNOWN",
                "allowed_trade_mode": "UNKNOWN",
                "advice": "Direction confidence unavailable because shadow evaluation failed.",
                "reason": str(e),
                "signals": {},
            }
        except Exception:
            pass

    return scenario


def add_direction_confidence_to_ai_data(
    ai_data,
    interval,
    backend_trend_regime=None,
    mtf_elliott_context=None,
    momentum_context=None,
    rsi_divergence_context=None,
    market_context=None,
):
    """
    Attach direction_confidence to the first trade scenario and selected_scenario.
    Response-only decoration. No live behavior change.
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

        scenario = add_direction_confidence_to_scenario(
            scenario=scenario,
            interval=interval,
            backend_trend_regime=backend_trend_regime or ai_data.get("backend_trend_regime"),
            mtf_elliott_context=mtf_elliott_context,
            momentum_context=momentum_context,
            rsi_divergence_context=rsi_divergence_context,
            market_context=market_context,
        )

        scenarios[0] = scenario
        ai_data["trade_scenarios"] = scenarios

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            selected_side = str(
                selected.get("trade_type")
                or selected.get("side")
                or ""
            ).upper().strip()

            scenario_side = str(
                scenario.get("trade_type")
                or scenario.get("side")
                or ""
            ).upper().strip()

            selected_entry = extract_price(
                selected.get("entry")
                or selected.get("entry_zone")
            )
            scenario_entry = extract_price(
                scenario.get("entry")
                or scenario.get("entry_zone")
            )

            # Safer than side-only match:
            # update selected_scenario when side matches and entry is same/unknown.
            entry_matches = (
                selected_entry is None
                or scenario_entry is None
                or abs(selected_entry - scenario_entry) <= max(abs(scenario_entry) * 0.000001, 1e-12)
            )

            if selected_side == scenario_side and entry_matches:
                ai_data["selected_scenario"] = scenario

    except Exception as e:
        print(f"[DIRECTION_CONFIDENCE] attach skipped/error: {e}")

    return ai_data

def add_tp_path_quality_to_scenario(
    scenario,
    interval,
    market_context_text=None,
):
    """
    Attach shadow TP path quality / S/R reaction advice to a response scenario only.

    This must NOT:
    - block signals
    - reject setups
    - change entry/SL/TP
    - change setup selection
    - change Firestore lock behavior
    - change AI/Gemini decisions
    """
    try:
        if not isinstance(scenario, dict):
            return scenario

        side = str(
            scenario.get("trade_type")
            or scenario.get("side")
            or ""
        ).upper().strip()

        if side not in ["LONG", "SHORT"]:
            return scenario

        entry = extract_price(
            scenario.get("entry")
            or scenario.get("entry_zone")
        )

        stop = extract_price(
            scenario.get("stop_loss")
            or scenario.get("invalidation")
        )

        targets = scenario.get("targets") or []

        tp1 = None
        tp2 = None

        if isinstance(targets, list) and targets:
            tp1 = extract_price(targets[0])
            if len(targets) > 1:
                tp2 = extract_price(targets[1])

        if tp1 is None:
            tp1 = extract_price(scenario.get("tp1"))

        if tp2 is None:
            tp2 = (
                extract_price(scenario.get("tp2"))
                or extract_price(scenario.get("target"))
            )

        if entry is None or tp1 is None:
            return scenario

        raw_values = scenario.get("raw_values") or {}
        atr = extract_price(raw_values.get("atr") or scenario.get("atr"))
        current_price = extract_price(raw_values.get("current_price"))

        entry_confirmation = scenario.get("entry_confirmation")
        direction_confidence = scenario.get("direction_confidence")

        scenario["tp_path_quality"] = evaluate_tp_path_quality(
            side=side,
            entry_price=entry,
            tp1=tp1,
            tp2=tp2,
            stop_loss=stop,
            user_timeframe=interval,
            market_context_text=market_context_text,
            entry_confirmation=entry_confirmation,
            direction_confidence=direction_confidence,
            atr=atr,
            current_price=current_price,
        )

    except Exception as e:
        try:
            scenario["tp_path_quality"] = {
                "enabled": True,
                "user_timeframe": str(interval or "").lower().strip(),
                "side": str(
                    scenario.get("trade_type")
                    or scenario.get("side")
                    or ""
                ).upper().strip(),
                "path_status": "UNKNOWN",
                "path_score": 0,
                "sr_strength": "UNKNOWN",
                "blocking_level": None,
                "blocking_type": "NONE",
                "original_entry_valid": True,
                "recommended_trade_mode": "UNKNOWN",
                "required_action": "NONE",
                "exit_if_rejected": False,
                "continue_if_breaks": False,
                "advice": "TP path quality unavailable because shadow evaluation failed.",
                "reason": str(e),
                "levels": [],
                "signals": {},
            }
        except Exception:
            pass

    return scenario


def add_tp_path_quality_to_ai_data(
    ai_data,
    interval,
    market_context_text=None,
):
    """
    Attach tp_path_quality to the first trade scenario and selected_scenario.
    Response-only decoration. No live behavior change.
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

        scenario = add_tp_path_quality_to_scenario(
            scenario=scenario,
            interval=interval,
            market_context_text=market_context_text,
        )

        scenarios[0] = scenario
        ai_data["trade_scenarios"] = scenarios

        selected = ai_data.get("selected_scenario")
        if isinstance(selected, dict):
            selected_side = str(
                selected.get("trade_type")
                or selected.get("side")
                or ""
            ).upper().strip()

            scenario_side = str(
                scenario.get("trade_type")
                or scenario.get("side")
                or ""
            ).upper().strip()

            selected_entry = extract_price(
                selected.get("entry")
                or selected.get("entry_zone")
            )
            scenario_entry = extract_price(
                scenario.get("entry")
                or scenario.get("entry_zone")
            )

            entry_matches = (
                selected_entry is None
                or scenario_entry is None
                or abs(selected_entry - scenario_entry) <= max(abs(scenario_entry) * 0.000001, 1e-12)
            )

            if selected_side == scenario_side and entry_matches:
                ai_data["selected_scenario"] = scenario

    except Exception as e:
        print(f"[TP_PATH_QUALITY] attach skipped/error: {e}")

    return ai_data

def split_live_and_analysis_df(
    df,
    min_rows=50,
    *,
    interval=None,
    now_utc=None,
):
    """Split Binance UTC candle-open data using actual candle end times."""
    if df is None or df.empty:
        return df, df

    df_live = df.copy()

    # Preserve legacy behavior for callers without a timeframe (e.g. forex).
    if interval is None:
        if len(df_live) <= min_rows:
            return df_live, df_live.copy()
        return df_live, df_live.iloc[:-1].copy()

    if "Date" in df_live.columns:
        raw_times = df_live["Date"]
    elif isinstance(df_live.index, pd.DatetimeIndex):
        raw_times = df_live.index
    else:
        raise ValueError("Candle splitting requires Date or a DatetimeIndex.")

    # Binance's naive timestamps represent UTC candle opening times.
    opens = pd.DatetimeIndex(
        pd.to_datetime(raw_times, utc=True, errors="raise")
    )
    if opens.hasnans or not opens.is_monotonic_increasing:
        raise ValueError("Candle timestamps must be valid and chronological.")

    raw_tf = str(interval).strip()
    tf = "1mo" if raw_tf == "1M" else raw_tf.lower()
    tf = {
        "daily": "1d",
        "d": "1d",
        "1wk": "1w",
        "monthly": "1mo",
        "1month": "1mo",
    }.get(tf, tf)

    match = re.fullmatch(r"([1-9]\d*)(mo|m|h|d|w)", tf)
    if match is None:
        raise ValueError(f"Unsupported candle timeframe: {interval}")

    count = int(match.group(1))
    unit = match.group(2)

    if unit == "mo":
        ends = pd.DatetimeIndex([
            stamp + pd.DateOffset(months=count)
            for stamp in opens
        ])
    else:
        seconds = {
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
        }
        ends = opens + pd.Timedelta(
            seconds=count * seconds[unit]
        )

    cutoff = (
        pd.Timestamp.now(tz="UTC")
        if now_utc is None else pd.Timestamp(now_utc)
    )
    cutoff = (
        cutoff.tz_localize("UTC")
        if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    )
    if pd.isna(cutoff):
        raise ValueError("Invalid candle analysis cutoff.")

    df_analysis = df_live.loc[ends <= cutoff].copy()
    return df_live, df_analysis

def build_execution_price_context(df_live, df_analysis):
    """
    Add live execution price separately.
    Main analysis uses closed candles, but no-chase/status uses live price.
    """
    try:
        live_price = None
        live_time = ""
        analysis_price = None
        analysis_time = ""

        if df_live is not None and not df_live.empty:
            live_price = float(df_live.iloc[-1]["Close"])
            live_time = str(df_live.index[-1])

        if df_analysis is not None and not df_analysis.empty:
            analysis_price = float(df_analysis.iloc[-1]["Close"])
            analysis_time = str(df_analysis.index[-1])

        lines = [
            "--- EXECUTION PRICE CONTEXT ---",
            f"Live Price For Execution: {format_signal_price(live_price)}",
            f"Live Candle Time: {live_time}",
            f"Analysis Closed Candle Price: {format_signal_price(analysis_price)}",
            f"Analysis Closed Candle Time: {analysis_time}",
            "Execution Note: Trend/Elliott/Fib/SR use closed candles. Live price is only for no-chase, entry touch, running PnL, TP/SL status."
        ]

        return "\n".join(lines)

    except Exception:
        return (
            "--- EXECUTION PRICE CONTEXT ---\n"
            "Execution Note: execution price context unavailable."
        )

def probability_from_ai_confidence(score):
    try:
        score = int(float(score or 0))
    except Exception:
        score = 0

    if score >= 82:
        return "High"

    if score >= 70:
        return "Medium-High"

    if score >= 58:
        return "Medium"

    return "Low"


def build_ai_first_scenario(
    signal,
    symbol,
    interval,
    market_packet=None,
    validation_result=None,
    provider=None,
    correction_used=False,
    backend_trend_regime=None,
):
    """
    Convert the final backend-validated AI signal into the
    existing frontend / Firestore / trade-audit scenario format.

    Important:
    - AI prices are accepted only after deterministic validation.
    - TP1 is the backend-calculated exact 1.5R value.
    - WAIT and NO_TRADE are not active trade locks.
    """

    signal = dict(signal or {})
    market_packet = market_packet or {}
    validation_result = validation_result or {}
    provider = provider or {}
    backend_trend_regime = backend_trend_regime or {}

    decision = str(
        signal.get("decision")
        or "NO_TRADE"
    ).upper().strip()

    side = str(
        signal.get("side")
        or "NONE"
    ).upper().strip()

    execution_state = str(
        signal.get("execution_state")
        or "NO_TRADE"
    ).upper().strip()

    order_type = str(
        signal.get("order_type")
        or "NONE"
    ).upper().strip()

    entry = extract_price(
        signal.get("entry_price")
    ) or 0.0

    stop_loss = extract_price(
        signal.get("stop_loss")
    ) or 0.0

    tp1 = extract_price(
        signal.get("tp1")
    ) or 0.0

    tp2 = extract_price(
        signal.get("tp2")
    ) or 0.0

    requested_tp2 = (
        extract_price(
            signal.get("requested_tp2")
        )
        or tp2
    )

    tp2_was_capped = bool(
        signal.get("tp2_was_capped")
    )

    tp2_cap_reason = str(
        signal.get("tp2_cap_reason")
        or ""
    ).strip()

    effective_max_tp2_r = extract_price(
        signal.get("effective_max_tp2_r")
    )

    has_complete_plan = bool(
        side in ["LONG", "SHORT"]
        and entry > 0
        and stop_loss > 0
        and tp1 > 0
        and tp2 > 0
    )

    status_map = {
        "ENTER_NOW": "RUNNING",
        "PLACE_LIMIT": "LIMIT_ORDER",
        "WAIT_FOR_ENTRY": "WAIT_FOR_ENTRY",
        "WAIT_FOR_RETEST": "WAIT_FOR_RETEST",
        "WAIT_FOR_RECLAIM": "WAIT_FOR_RECLAIM",
        "WAIT_FOR_BREAKOUT": "WAIT_FOR_BREAKOUT",
        "WAIT_FOR_FAILED_RECLAIM": (
            "WAIT_FOR_FAILED_RECLAIM"
        ),
        "NO_TRADE": "NO_TRADE",
    }


    if decision == "NO_TRADE" or not has_complete_plan:
        frontend_trade_type = "WAIT"

        scenario_status = (
            "NO_TRADE"
            if decision == "NO_TRADE"
            else "WAIT"
        )

        scenario_side = (
            "NONE"
            if decision == "NO_TRADE"
            else side
        )

        scenario_order_type = (
            "NONE"
            if decision == "NO_TRADE"
            else order_type
        )

    elif decision == "WAIT":
        frontend_trade_type = (
            side
            if (
                has_complete_plan
                and side in [
                    "LONG",
                    "SHORT",
                ]
            )
            else "WAIT"
        )

        scenario_status = status_map.get(
            execution_state,
            "WAIT",
        )

        scenario_side = side

        scenario_order_type = (
            order_type
            if has_complete_plan
            else "NONE"
        )

    else:
        frontend_trade_type = side

        scenario_status = status_map.get(
            execution_state,
            "LIMIT_ORDER",
        )

        scenario_side = side
        scenario_order_type = order_type


    market_section = (
        market_packet.get(
            "market",
            {},
        )
        if isinstance(
            market_packet,
            dict,
        )
        else {}
    )

    main_summary = (
        (
            (
                market_packet.get(
                    "timeframes"
                )
                or {}
            )
            .get(
                "main",
                {},
            )
            .get(
                "summary",
                {},
            )
        )
        if isinstance(
            market_packet,
            dict,
        )
        else {}
    )

    current_price = extract_price(
        market_section.get(
            "current_live_price"
        )
    ) or 0.0

    atr = extract_price(
        main_summary.get("atr14")
    ) or 0.0

    analysis_timestamp = (
        market_section.get(
            "analysis_timestamp"
        )
        or ""
    )

    risk_distance = (
        abs(entry - stop_loss)
        if has_complete_plan
        else 0.0
    )

    risk_pct = (
        (
            risk_distance / entry
        )
        * 100.0
        if has_complete_plan
        and entry > 0
        else 0.0
    )

    tp2_r = (
        abs(tp2 - entry)
        / risk_distance
        if has_complete_plan
        and risk_distance > 0
        else 0.0
    )

    try:
        confidence = int(
            float(
                signal.get(
                    "confidence"
                )
                or 0
            )
        )
    except Exception:
        confidence = 0

    setup_type = str(
        signal.get("setup_type")
        or "NO_TRADE"
    ).upper().strip()

    evidence_paths = (
        signal.get("evidence_paths")
        or []
    )

    elliott_evidence = any(
        (
            "elliott"
            in str(path).lower()
        )
        or (
            "wave"
            in str(path).lower()
        )
        for path in evidence_paths
    )

    elliott_confirmed = bool(
        elliott_evidence
        and (
            "WAVE" in setup_type
            or "ELLIOTT" in setup_type
        )
    )

    targets = (
        [
            format_signal_price(tp1),
            format_signal_price(tp2),
        ]
        if has_complete_plan
        else []
    )

    advice = str(
        signal.get("advice")
        or ""
    ).strip()

    status_text = str(
        signal.get("status_text")
        or ""
    ).strip()

    decision_reason = str(
        signal.get("decision_reason")
        or ""
    ).strip()

    scenario = {
        "name": "AI-First Signal",

        "trade_type": (
            frontend_trade_type
        ),

        "side": scenario_side,
        "decision": decision,
        "status": scenario_status,
        "status_text": status_text,

        "order_type": (
            scenario_order_type
        ),

        "execution_state": (
            execution_state
        ),

        "probability": (
            probability_from_ai_confidence(
                confidence
            )
        ),

        "quality_score": confidence,

        "summary": (
            advice
            or decision_reason
        ),

        "generated_advice": (
            advice
            or decision_reason
        ),

        "entry_zone": (
            format_signal_price(entry)
            if has_complete_plan
            else ""
        ),

        "entry": (
            format_signal_price(entry)
            if has_complete_plan
            else ""
        ),

        "invalidation": (
            format_signal_price(
                stop_loss
            )
            if has_complete_plan
            else ""
        ),

        "stop_loss": (
            format_signal_price(
                stop_loss
            )
            if has_complete_plan
            else ""
        ),

        "targets": targets,

        "target": (
            format_signal_price(tp2)
            if has_complete_plan
            else ""
        ),

        "tp1": (
            format_signal_price(tp1)
            if has_complete_plan
            else ""
        ),

        "tp2": (
            format_signal_price(tp2)
            if has_complete_plan
            else ""
        ),

        "risk_pct": round(
            risk_pct,
            4,
        ),

        "rr": (
            round(tp2_r, 4)
            if has_complete_plan
            else 0.0
        ),

        "tp1_r": 1.5,

        "tp2_r": (
            round(tp2_r, 4)
            if has_complete_plan
            else 0.0
        ),

        "requested_tp2": (
            format_signal_price(
                requested_tp2
            )
            if (
                has_complete_plan
                and requested_tp2 > 0
            )
            else ""
        ),

        "tp2_was_capped": (
            tp2_was_capped
        ),

        "tp2_cap_reason": (
            tp2_cap_reason
        ),

        "effective_max_tp2_r": (
            effective_max_tp2_r
        ),

        "setup_type": setup_type,

        "setup_source": (
            "AI_FIRST_GEMINI_VALIDATED"
        ),

        "entry_type": (
            execution_state
        ),

        "entry_source": signal.get(
            "entry_anchor_path"
        ),

        "market_regime": signal.get(
            "market_regime"
        ),

        "trade_quality": signal.get(
            "trade_quality"
        ),

        "trap_risk": signal.get(
            "trap_risk"
        ),

        "sl_hit_risk": signal.get(
            "sl_hit_risk"
        ),

        "tp2_quality": signal.get(
            "tp2_quality"
        ),

        "expected_path": signal.get(
            "expected_path"
        ),

        "timeframe_bias": (
            signal.get(
                "timeframe_bias"
            )
            or {}
        ),

        "invalidation_condition": (
            signal.get(
                "invalidation_condition"
            )
        ),

        "evidence_paths": (
            evidence_paths
        ),

        "counter_evidence_paths": (
            signal.get(
                "counter_evidence_paths"
            )
            or []
        ),

        "reason_codes": (
            signal.get(
                "reason_codes"
            )
            or []
        ),

        "decision_reason": (
            decision_reason
        ),

        "entry_anchor_path": (
            signal.get(
                "entry_anchor_path"
            )
        ),

        "sl_anchor_path": (
            signal.get(
                "sl_anchor_path"
            )
        ),

        "tp2_anchor_path": (
            signal.get(
                "tp2_anchor_path"
            )
        ),

        "ai_first": True,

        "backend_validated": bool(
            validation_result.get(
                "valid"
            )
        ),

        "backend_selected": False,
        "ai_can_modify_prices": False,

        "ai_correction_used": bool(
            correction_used
        ),

        "ai_provider": {
            "model": provider.get(
                "model"
            ),

            "key_source": provider.get(
                "key_source"
            ),

            "usage": (
                provider.get("usage")
                or {}
            ),
        },

        "ai_validation": {
            "valid": bool(
                validation_result.get(
                    "valid"
                )
            ),

            "trade_publishable": bool(
                validation_result.get(
                    "trade_publishable"
                )
            ),

            "watch_publishable": bool(
                validation_result.get(
                    "watch_publishable"
                )
            ),

            "errors": (
                validation_result.get(
                    "errors"
                )
                or []
            ),

            "warnings": (
                validation_result.get(
                    "warnings"
                )
                or []
            ),

            "validator_version": (
                validation_result.get(
                    "validator_version"
                )
            ),

            "historical_validation": (
                validation_result.get(
                    "historical_validation"
                )
                or {}
            ),
        },

        "elliott_confirmed": (
            elliott_confirmed
        ),

        "wave_template": (
            setup_type
            if elliott_confirmed
            else ""
        ),

        "wave_target": (
            format_signal_price(tp2)
            if elliott_confirmed
            else ""
        ),

        "wave_confidence": (
            confidence
            if elliott_confirmed
            else ""
        ),

        "mtf_elliott_alignment": (
            "UNKNOWN"
        ),

        "mtf_confirmation_score": 0,
        "mtf_confirmation_reason": "",

        "backend_trend_regime": (
            backend_trend_regime
        ),

        "backend_regime": (
            backend_trend_regime.get(
                "regime"
            )
        ),

        "backend_market_bias": (
            backend_trend_regime.get(
                "market_bias"
            )
        ),

        "backend_market_phase": (
            backend_trend_regime.get(
                "market_phase"
            )
        ),

        "backend_preferred_side": (
            backend_trend_regime.get(
                "preferred_side"
            )
        ),

        "backend_blocked_side": (
            backend_trend_regime.get(
                "blocked_side"
            )
        ),

        "backend_regime_confidence": (
            backend_trend_regime.get(
                "confidence"
            )
        ),

        "symbol": symbol,
        "timeframe": interval,

        "entry_filled_at_creation": bool(
            decision in [
                "LONG",
                "SHORT",
            ]
            and execution_state
            == "ENTER_NOW"
            and scenario_order_type
            == "MARKET"
        ),

        "entry_fill_time": (
            analysis_timestamp
        ),

        "raw_values": {
            "entry": entry,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "requested_tp2": requested_tp2,
            "tp2_was_capped": tp2_was_capped,
            "tp2_cap_reason": tp2_cap_reason,
            "effective_max_tp2_r": (
                effective_max_tp2_r
            ),
            "risk": risk_distance,
            "risk_pct": risk_pct,
            "rr": tp2_r,
            "current_price": (
                current_price
            ),
            "atr": atr,
        },
    }


    return scenario


def get_pending_expiry_candles(interval):
    """
    How long a pending limit setup can remain valid without entry.
    Conservative default values.
    """
    tf = str(interval or "").lower().strip()

    if tf in ["1m", "3m", "5m"]:
        return 36       # short scalps expire quickly
    if tf in ["15m", "30m"]:
        return 24       # about 6h for 15m, 12h for 30m
    if tf in ["1h", "60m"]:
        return 24       # about 1 day
    if tf in ["4h", "240m"]:
        return 30       # about 5 days
    if tf in ["1d", "d", "daily"]:
        return 14       # about 2 weeks

    return 24


def check_trade_status_historical(scenario, historical_df, interval="15m"):
    try:
        trade_type = scenario.get("trade_type", "").upper()
        targets = scenario.get("targets", [])

        if not targets and "target" in scenario:
            targets = [scenario.get("target")]

        raw_tps = []
        for t in targets:
            val = extract_price(t)
            if val is not None:
                raw_tps.append(val)

        tp_levels = sorted(raw_tps) if "LONG" in trade_type else sorted(raw_tps, reverse=True)

        entry_price = extract_price(scenario.get("entry_zone", "0") or scenario.get("entry", "0"))
        stop_loss = extract_price(scenario.get("invalidation", "0") or scenario.get("stop_loss", "0"))

        if not tp_levels or stop_loss is None or entry_price is None:
            return "UNKNOWN", "Data Missing"

        if "LONG" not in trade_type and "SHORT" not in trade_type:
            return "UNKNOWN", "Invalid trade type"

        scenario_status = str(
            scenario.get("status")
            or ""
        ).upper().strip()

        scenario_order_type = str(
            scenario.get("order_type")
            or ""
        ).upper().strip()

        is_active = bool(
            scenario.get(
                "entry_filled_at_creation"
            )
            or scenario.get(
                "entry_filled"
            )
            or scenario_order_type
            == "MARKET"
            or "RUNNING"
            in scenario_status
            or "TP 1 HIT"
            in scenario_status
            or "TP1_HIT"
            in scenario_status
        )
        highest_tp_hit_idx = -1
        required_win_index = 1 if len(tp_levels) > 1 else 0
        max_pending_candles = get_pending_expiry_candles(interval)

        tp1 = tp_levels[0]
        candles_seen = 0

        for row in historical_df.itertuples():
            candles_seen += 1

            curr_low = float(row.Low)
            curr_high = float(row.High)
            curr_close = float(row.Close)

            # ==================================================
            # PENDING LIMIT ORDER PHASE
            # ==================================================
            if not is_active:
                entry_touched = (
                    ("LONG" in trade_type and curr_low <= entry_price) or
                    ("SHORT" in trade_type and curr_high >= entry_price)
                )

                if entry_touched:
                    is_active = True

                else:
                    # -------------------------------
                    # MISSED / NO CHASE
                    # -------------------------------
                    try:
                        if "LONG" in trade_type and tp1 > entry_price:
                            no_chase_level = entry_price + ((tp1 - entry_price) * 0.70)

                            if curr_high >= no_chase_level:
                                return (
                                    "MISSED",
                                    "No chase. Price moved more than 70% toward TP1 before entry."
                                )

                        elif "SHORT" in trade_type and tp1 < entry_price:
                            no_chase_level = entry_price - ((entry_price - tp1) * 0.70)

                            if curr_low <= no_chase_level:
                                return (
                                    "MISSED",
                                    "No chase. Price moved more than 70% toward TP1 before entry."
                                )
                    except Exception:
                        pass

                    # -------------------------------
                    # CANCELLED BEFORE ENTRY
                    # This is a conservative safety check.
                    # In normal limit orders, entry usually touches before stop.
                    # But this catches malformed / invalid pending setups.
                    # -------------------------------
                    try:
                        if "LONG" in trade_type and curr_close <= stop_loss:
                            return (
                                "CANCELLED",
                                "Setup cancelled. Price closed beyond invalidation before safe entry."
                            )

                        elif "SHORT" in trade_type and curr_close >= stop_loss:
                            return (
                                "CANCELLED",
                                "Setup cancelled. Price closed beyond invalidation before safe entry."
                            )
                    except Exception:
                        pass

                    # -------------------------------
                    # EXPIRED
                    # -------------------------------
                    if max_pending_candles and candles_seen > max_pending_candles:
                        return (
                            "EXPIRED",
                            f"Setup expired after {candles_seen} candles without entry."
                        )

                    continue

            # ==================================================
            # ACTIVE TRADE PHASE
            # ==================================================
            if is_active:
                if "LONG" in trade_type:
                    if curr_low <= stop_loss:
                        # TP1 was reached on an earlier candle.
                        # The secured partial profit makes this
                        # an overall partial win.
                        if highest_tp_hit_idx >= 0:
                            return (
                                "WON",
                                "TP1 was secured earlier; "
                                "the remaining runner later "
                                "hit Stop Loss."
                            )

                        # Conservative same-candle rule:
                        # SL is checked before TP targets.
                        return (
                            "LOST",
                            "Stop Loss Hit before TP1"
                        )

                    for i, tp in enumerate(tp_levels):
                        if (
                            curr_high >= tp
                            and i > highest_tp_hit_idx
                        ):
                            highest_tp_hit_idx = i

                elif "SHORT" in trade_type:
                    if curr_high >= stop_loss:
                        # TP1 was reached on an earlier candle.
                        # The secured partial profit makes this
                        # an overall partial win.
                        if highest_tp_hit_idx >= 0:
                            return (
                                "WON",
                                "TP1 was secured earlier; "
                                "the remaining runner later "
                                "hit Stop Loss."
                            )

                        # Conservative same-candle rule:
                        # SL is checked before TP targets.
                        return (
                            "LOST",
                            "Stop Loss Hit before TP1"
                        )

                    for i, tp in enumerate(tp_levels):
                        if (
                            curr_low <= tp
                            and i > highest_tp_hit_idx
                        ):
                            highest_tp_hit_idx = i

                if highest_tp_hit_idx >= required_win_index:
                    return "WON", f"Target {highest_tp_hit_idx + 1} Hit!"

        # ==================================================
        # FINAL STATUS AFTER ALL CANDLES
        # ==================================================
        if is_active:
            if highest_tp_hit_idx == 0 and len(tp_levels) > 1:
                return (
                    "TP 1 HIT",
                    "TP1 hit. Secure partial profit and move SL near entry/breakeven, or trail using latest structure if holding for TP2."
                )

            last_price = None
            try:
                last_price = float(historical_df.iloc[-1]["Close"])
            except Exception:
                pass

            pnl_percent = calculate_running_pnl_percent(scenario, last_price)
            running_advice = format_running_pnl_advice(pnl_percent)

            return "RUNNING", running_advice

        return "LIMIT ORDER", "Limit order still pending. Wait for entry confirmation."

    except Exception as e:
        return "UNKNOWN", str(e)

def refresh_existing_signal_guidance(symbol, interval, market_type="crypto"):
    """
    Lightweight refresh for an existing global locked signal.

    This is used by:
    - action == "status"
    - action == "analyze" early return when a global lock already exists

    It does NOT run full MTF/Elliott/backend setup planning.
    It only:
    - loads locked global signal
    - loads selected timeframe candles
    - checks TP/SL/entry status
    - builds backend guidance facts
    - asks AI to write user-friendly advice from backend facts
    """
    locked_scenario, lock_time = get_firestore_lock(symbol, interval)

    if not locked_scenario:
        return {
            "ai_prediction": {
                "trade_status_message": "NO ACTIVE TRADE",
                "action_advice": "No active global signal for this coin/timeframe.",
                "macro_analysis": {"trend": "Unknown"},
                "micro_analysis": {"wave_personality": "Unknown"},
                "trade_scenarios": []
            }
        }

    if locked_scenario and not lock_time:
        delete_firestore_lock(symbol, interval)
        return {
            "ai_prediction": {
                "trade_status_message": "NO ACTIVE TRADE",
                "action_advice": "Old signal lock had no timestamp and was removed.",
                "macro_analysis": {"trend": "Unknown"},
                "micro_analysis": {"wave_personality": "Unknown"},
                "trade_scenarios": []
            }
        }

    # --------------------------------------------------
    # Guidance refresh data only.
    # Important:
    # - Do NOT use this for fresh analyze.
    # - Fresh analyze keeps its existing dynamic periods.
    # - Advice refresh uses selected TF + one higher confirm TF.
    # --------------------------------------------------

    guidance_confirm_tf = get_guidance_confirm_timeframe(interval)

    df_micro = get_data(
        symbol,
        interval=interval,
        period="1mo",
        market_type=market_type
    )

    if not df_micro.empty and "Date" in df_micro.columns:
        df_micro.set_index("Date", inplace=True)

    df_confirm_guidance = get_data(
        symbol,
        interval=guidance_confirm_tf["tf"],
        period=guidance_confirm_tf["pd"],
        market_type=market_type
    )

    if not df_confirm_guidance.empty and "Date" in df_confirm_guidance.columns:
        df_confirm_guidance.set_index("Date", inplace=True)

    if not df_confirm_guidance.empty and not isinstance(df_confirm_guidance.index, pd.DatetimeIndex):
        df_confirm_guidance.index = pd.to_datetime(df_confirm_guidance.index)

    if not isinstance(df_micro.index, pd.DatetimeIndex):
        df_micro.index = pd.to_datetime(df_micro.index)

    try:
        if df_micro.index.tz is None and hasattr(lock_time, "tzinfo") and lock_time.tzinfo:
            lock_time = lock_time.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        pass

    df_since_lock = df_micro[df_micro.index >= lock_time]

    if df_since_lock.empty:
        df_since_lock = df_micro.tail(1)

    # Use only recent candles for guidance pressure/advice.
    # df_since_lock still uses full selected TF data from lock time
    # so TP/SL/entry status tracking is not broken.
    df_guidance_live = df_micro.tail(350)
    df_confirm_guidance_live = df_confirm_guidance.tail(350)

    status, base_advice = check_trade_status_historical(
        locked_scenario,
        df_since_lock,
        interval=interval
    )

    # --------------------------------------------------
    # Pre-guidance risk decoration for active management.
    #
    # trade_guidance_manager needs liquidity_risk / imbalance_risk
    # BEFORE build_trade_guidance_facts(), especially after TP1.
    #
    # This lets guidance say:
    #   TP1 hit + SL-side liquidity/imbalance pending
    #   => DEFENSIVE RUNNER
    # instead of aggressive CLOSE_FULL.
    #
    # Important:
    # - does NOT change entry/SL/TP
    # - does NOT change lock status
    # - response-only / guidance-only context
    # --------------------------------------------------
    df_lower_exec_analysis = pd.DataFrame()

    try:
        status_upper_for_guidance = str(status or "").upper().strip()

        lower_context_relevant = (
            "LIMIT ORDER" in status_upper_for_guidance
            or "LIMIT_ORDER" in status_upper_for_guidance
            or "PENDING" in status_upper_for_guidance
            or "WAITING_FOR_ENTRY" in status_upper_for_guidance
            or "RUNNING" in status_upper_for_guidance
            or "TP 1 HIT" in status_upper_for_guidance
            or "TP1_HIT" in status_upper_for_guidance
        )

        if lower_context_relevant:
            lower_exec_tf = get_lower_execution_timeframe(interval)
            df_lower_exec = pd.DataFrame()

            if lower_exec_tf:
                df_lower_exec = get_data(
                    symbol,
                    interval=lower_exec_tf["tf"],
                    period=lower_exec_tf["pd"],
                    market_type=market_type
                )

            df_lower_exec_analysis = _entry_confirmation_analysis_df(
                df_lower_exec,
                interval=(
                    lower_exec_tf["tf"]
                    if market_type == "crypto" and lower_exec_tf else None
                ),
            )

    except Exception as e:
        print(f"[GUIDANCE_RISK_CONTEXT] lower TF fetch skipped/error: {e}")
        df_lower_exec_analysis = pd.DataFrame()

    guidance_scenario = dict(locked_scenario or {})

    try:
        guidance_scenario = add_imbalance_to_scenario(
            scenario=guidance_scenario,
            interval=interval,
            df_analysis=df_micro,
            df_lower=df_lower_exec_analysis if not df_lower_exec_analysis.empty else None,
        )

        guidance_scenario = add_liquidity_to_scenario(
            scenario=guidance_scenario,
            interval=interval,
            df_analysis=df_micro,
            df_lower=df_lower_exec_analysis if not df_lower_exec_analysis.empty else None,
        )

    except Exception as e:
        print(f"[GUIDANCE_RISK_CONTEXT] imbalance/liquidity attach skipped/error: {e}")
        guidance_scenario = dict(locked_scenario or {})

    guidance_facts = build_trade_guidance_facts(
        scenario=guidance_scenario,
        df_live=df_guidance_live,
        df_since_lock=df_since_lock,
        interval=interval,
        existing_status=status,
        existing_advice=base_advice,
        df_confirm=df_confirm_guidance_live,
    )

    management_strategy = {
        "phase": guidance_facts.get("guidance_status", status),
        "advice": guidance_facts.get("user_guidance", base_advice),
        "action": guidance_facts.get("action", ""),
        "risk_level": guidance_facts.get("risk_level", ""),
        "danger_score": guidance_facts.get("danger_score", 0),
        "danger_flags": guidance_facts.get("danger_flags", []),
        "management_mode": guidance_facts.get("management_mode", ""),
        "tp2_probability": guidance_facts.get("tp2_probability", ""),
        "reversal_risk": guidance_facts.get("reversal_risk", ""),
        "profit_action": guidance_facts.get("profit_action", ""),
        "sl_management": guidance_facts.get("sl_management", ""),
        "exit_trigger": guidance_facts.get("exit_trigger", ""),
        "confirm_tf": guidance_confirm_tf.get("tf", ""),
    }
    guidance_context_text = build_guidance_context_text(guidance_facts)

    ai_guidance = generate_trade_guidance_advice(
        scenario=guidance_scenario,
        guidance_facts=guidance_facts,
        guidance_context_text=guidance_context_text,
    )

    final_advice = (
        ai_guidance.get("ai_guidance")
        if isinstance(ai_guidance, dict) and ai_guidance.get("ai_guidance")
        else guidance_facts.get("user_guidance", base_advice)
    )

    # --------------------------------------------------
    # Safety guard:
    # If guidance facts prove the trade was already entered or TP1 was hit,
    # do NOT allow old status logic to mark it as MISSED/CANCELLED/EXPIRED.
    # This prevents deleting active trades incorrectly.
    # --------------------------------------------------
    facts = guidance_facts.get("facts") or {}
    entry_hit_fact = bool(facts.get("entry_hit"))
    tp1_hit_fact = bool(facts.get("tp1_hit"))
    tp2_hit_fact = bool(facts.get("tp2_hit"))
    sl_hit_fact = bool(facts.get("sl_hit"))

    if status in ["MISSED", "CANCELLED", "EXPIRED"] and entry_hit_fact:
        if tp2_hit_fact:
            status = "WON"
        elif sl_hit_fact:
            status = "LOST"
        elif tp1_hit_fact:
            status = "TP 1 HIT"
        else:
            status = "RUNNING"

    # --------------------------------------------------
    # Terminal trade states
    # --------------------------------------------------
    if status in ["WON", "LOST"]:
        last_price = None
        try:
            last_price = float(df_since_lock.iloc[-1]["Close"])
        except Exception:
            pass

        if (
            FIREBASE_AVAILABLE
            and db is not None
        ):
            try:
                ok, audit_msg = (
                    audit_close_trade(
                        db=db,
                        symbol=symbol,
                        timeframe=interval,
                        result=status,
                        reason=base_advice,
                        close_price=last_price,
                    )
                )

                if not ok:
                    print(
                        "[TRADE_AUDIT] close "
                        f"skipped/error: {audit_msg}"
                    )

            except Exception as e:
                print(
                    "[TRADE_AUDIT] close "
                    f"skipped/error: {e}"
                )

        else:
            print(
                "[TRADE_AUDIT] close skipped: "
                "Firebase is unavailable."
            )

        delete_firestore_lock(symbol, interval)
        status_msg = f"TRADE CLOSED ({status})"

    elif status in ["MISSED", "CANCELLED", "EXPIRED"]:
        delete_firestore_lock(symbol, interval)
        status_msg = f"{status} | {final_advice}"

    else:
        # Keep old UI behavior:
        # RUNNING status should still show PnL/profit text near the status badge.
        if str(status).upper() == "RUNNING":
            status_msg = f"{status} | {base_advice}"
        elif str(status).upper().startswith("TP 1 HIT"):
            status_msg = f"{status} | {guidance_facts.get('short_action_text') or guidance_facts.get('action', '')}"
        else:
            status_msg = status

        try:
            locked_scenario["generated_advice"] = final_advice
            locked_scenario["trade_guidance"] = guidance_facts
            locked_scenario["ai_trade_guidance"] = ai_guidance
            update_firestore_lock_guidance(
                symbol=symbol,
                tf=interval,
                scenario=locked_scenario,
                guidance_facts=guidance_facts,
                ai_guidance=ai_guidance,
            )
        except Exception as e:
            print(f"[GUIDANCE] Failed to update locked scenario guidance: {e}")

    response_scenario = dict(guidance_scenario or locked_scenario or {})
    try:
        status_upper = str(status or "").upper().strip()

        entry_confirmation_relevant = (
            "LIMIT ORDER" in status_upper
            or "LIMIT_ORDER" in status_upper
            or "PENDING" in status_upper
            or "WAITING_FOR_ENTRY" in status_upper
        )

        # Liquidity sweep detection benefits from lower TF even when the trade is already RUNNING/TP1_HIT.
        # This is still response-only/shadow context and does not change entry, SL, TP, or lock state.
        liquidity_lower_relevant = (
            entry_confirmation_relevant
            or "RUNNING" in status_upper
            or "TP 1 HIT" in status_upper
            or "TP1_HIT" in status_upper
        )

        if liquidity_lower_relevant and (
            df_lower_exec_analysis is None or df_lower_exec_analysis.empty
        ):
            lower_exec_tf = get_lower_execution_timeframe(interval)
            df_lower_exec = pd.DataFrame()

            if lower_exec_tf:
                df_lower_exec = get_data(
                    symbol,
                    interval=lower_exec_tf["tf"],
                    period=lower_exec_tf["pd"],
                    market_type=market_type
                )

            df_lower_exec_analysis = _entry_confirmation_analysis_df(
                df_lower_exec,
                interval=(
                    lower_exec_tf["tf"]
                    if market_type == "crypto" and lower_exec_tf else None
                ),
            )

        if entry_confirmation_relevant:
            response_scenario = add_entry_confirmation_to_scenario(
                response_scenario,
                interval,
                df_lower_exec_analysis,
                current_price=latest_close_price(df_micro),
                entry_zone_touched=True if entry_hit_fact else None,
            )
        else:
            response_scenario["entry_confirmation"] = build_non_actionable_entry_confirmation(
                interval=interval,
                status=status,
                entry_hit_fact=entry_hit_fact,
            )

        response_scenario = add_direction_confidence_to_scenario(
            scenario=response_scenario,
            interval=interval,
            backend_trend_regime=response_scenario.get("backend_trend_regime"),
        )

        response_scenario = add_tp_path_quality_to_scenario(
            scenario=response_scenario,
            interval=interval,
            market_context_text=response_scenario.get("market_context_text") or response_scenario.get("market_context") or "",
        )

        response_scenario = add_imbalance_to_scenario(
            scenario=response_scenario,
            interval=interval,
            df_analysis=df_micro,
            df_lower=df_lower_exec_analysis if not df_lower_exec_analysis.empty else None,
        )

        response_scenario = add_liquidity_to_scenario(
            scenario=response_scenario,
            interval=interval,
            df_analysis=df_micro,
            df_lower=df_lower_exec_analysis if not df_lower_exec_analysis.empty else None,
        )

        try:
            # --------------------------------------------------
            # Crash regime / momentum pressure shadow warning.
            # ZERO extra API version:
            # - selected TF uses df_micro
            # - fast TF uses already fetched df_lower_exec_analysis only
            # - no new get_data() calls here
            # --------------------------------------------------
            crash_fast_tf = get_crash_fast_timeframe_from_existing_lower(interval)

            df_crash_fast_analysis = pd.DataFrame()
            try:
                if (
                    crash_fast_tf
                    and df_lower_exec_analysis is not None
                    and not df_lower_exec_analysis.empty
                ):
                    df_crash_fast_analysis = df_lower_exec_analysis
            except Exception:
                df_crash_fast_analysis = pd.DataFrame()

            response_scenario = add_crash_regime_to_scenario(
                scenario=response_scenario,
                interval=interval,
                df_selected=df_micro,
                df_fast=df_crash_fast_analysis if not df_crash_fast_analysis.empty else None,
                fast_timeframe=crash_fast_tf or "",
            )

        except Exception as e:
            print(f"[CRASH_REGIME] locked response attach skipped/error: {e}")

        try:
            response_scenario = add_smc_context_to_scenario(
                scenario=response_scenario,
                interval=interval,
                df_selected=df_micro,
                symbol=symbol,
            )
        except Exception as e:
            print(f"[SMC_CONTEXT] locked response attach skipped/error: {e}")

        try:
            response_scenario = add_range_sweep_to_scenario(
                scenario=response_scenario,
                interval=interval,
                df_selected=df_micro,
                df_fast=(
                    df_lower_exec_analysis
                    if not df_lower_exec_analysis.empty
                    else None
                ),
            )
        except Exception as e:
            print(f"[RANGE_SWEEP] locked response attach skipped/error: {e}")


        response_scenario = add_decision_audit_to_scenario(
            scenario=response_scenario,
            backend_status=status,
        )

    except Exception as e:
        print(f"[SHADOW_ADVICE] locked response attach skipped/error: {e}")

    return {
        "meta": {
            "symbol": symbol,
            "timeframe": interval,
            "mode": "EXISTING_GLOBAL_SIGNAL_REFRESH"
        },
        "ai_prediction": {
            "trade_status_message": status_msg,
            "action_advice": final_advice,
            "management_strategy": management_strategy,
            "macro_analysis": {"trend": "LOCKED (Monitoring)"},
            "micro_analysis": {
                "wave_personality": "Monitoring Phase",
                "guidance_status": guidance_facts.get("guidance_status", ""),
                "risk_level": guidance_facts.get("risk_level", ""),
            },
            "trade_scenarios": [response_scenario]
        }
    }

# --- ANALYSIS HELPERS ---
class WaveOptionsGenerator3:
    def __init__(self, up_to=1):
        self.options_sorted = []
        for i in range(up_to + 1):
            for j in range(up_to + 1):
                for k in range(up_to + 1):
                    self.options_sorted.append(type('obj', (object,), {'values': [i, j, k]})())
        self.options_sorted.sort(key=lambda x: sum(x.values))


def find_best_fit(wa, start_idx, end_idx_limit, generators, rules):
    best_waves, best_type, max_len = None, "", 0

    # Bullish impulse search: up, down, up, down, up
    try:
        for opt in generators["impulse"].options_sorted:
            try:
                waves = wa.find_impulsive_wave(
                    idx_start=start_idx,
                    wave_config=opt.values
                )

                if waves and waves[-1].idx_end <= end_idx_limit + 10:
                    wp = WavePattern(waves, verbose=False)

                    if wp.check_rule(rules["impulse"]):
                        dur = waves[-1].idx_end - waves[0].idx_start

                        if dur > max_len:
                            max_len, best_waves, best_type = dur, waves, "Impulse"
            except Exception:
                continue
    except Exception:
        pass

    # Bearish impulse search: down, up, down, up, down
    try:
        for opt in generators["impulse"].options_sorted:
            try:
                if not hasattr(wa, "find_impulsive_wave_down"):
                    continue

                waves = wa.find_impulsive_wave_down(
                    idx_start=start_idx,
                    wave_config=opt.values
                )

                if waves and waves[-1].idx_end <= end_idx_limit + 10:
                    # Do not use normal Impulse() rules here.
                    # Impulse() rules are designed for bullish structures.
                    # Bearish validation is handled inside WaveAnalyzer.find_impulsive_wave_down().
                    dur = waves[-1].idx_end - waves[0].idx_start

                    if dur > max_len:
                        max_len, best_waves, best_type = dur, waves, "BearishImpulse"
            except Exception:
                continue
    except Exception:
        pass

    # Corrective wave search: down, up, down
    try:
        for opt in generators["correction"].options_sorted:
            try:
                waves = wa.find_corrective_wave(
                    idx_start=start_idx,
                    wave_config=opt.values
                )

                if waves and waves[-1].idx_end <= end_idx_limit + 10:
                    wp = WavePattern(waves, verbose=False)

                    if wp.check_rule(rules["correction"]):
                        dur = waves[-1].idx_end - waves[0].idx_start

                        if dur > max_len:
                            max_len, best_waves, best_type = dur, waves, "Correction"
            except Exception:
                continue
    except Exception:
        pass

    return best_type, best_waves


def find_best_bearish_fit(wa, start_idx, end_idx_limit, generators):
    """
    Search only for bearish impulse:
    down, up, down, up, down

    This is used as an extra parallel scan.
    It does not replace the existing bullish/correction scan.
    """
    best_waves, best_type, max_len = None, "", 0

    try:
        if not hasattr(wa, "find_impulsive_wave_down"):
            return "", None

        for opt in generators["impulse"].options_sorted:
            try:
                waves = wa.find_impulsive_wave_down(
                    idx_start=start_idx,
                    wave_config=opt.values
                )

                if waves and waves[-1].idx_end <= end_idx_limit + 10:
                    dur = waves[-1].idx_end - waves[0].idx_start

                    if dur > max_len:
                        max_len, best_waves, best_type = dur, waves, "BearishImpulse"

            except Exception:
                continue

    except Exception:
        pass

    return best_type, best_waves


def analyze_structure_phase1_only(df):
    if df.empty:
        return [], pd.DataFrame()

    df_proc = df.reset_index()

    if "Date" not in df_proc.columns and "index" in df_proc.columns:
        df_proc["Date"] = df_proc["index"]

    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    df_proc = df_proc.rename(columns=rename_map)

    lows, highs = df_proc["Low"].values, df_proc["High"].values
    idx_abs_low, idx_abs_high = np.argmin(lows), np.argmax(highs)

    # Keep the original bullish-oriented scan behavior.
    if idx_abs_low > idx_abs_high:
        idx_abs_high = len(df_proc) - 1

    wa = WaveAnalyzer(df=df_proc, verbose=False)

    gens = {
        "impulse": WaveOptionsGenerator5(up_to=3),
        "correction": WaveOptionsGenerator3(up_to=3)
    }

    rules = {
        "impulse": Impulse("impulse"),
        "correction": Correction("correction")
    }

    final_wave_objects = []
    seen_wave_keys = set()

    def add_wave_object(p_type, waves):
        """
        Add wave object only if it is not an exact duplicate.
        Keeps bullish/correction behavior and safely adds bearish results.
        """
        try:
            if not waves:
                return

            start = waves[0].idx_start
            end = waves[-1].idx_end
            key = (p_type, start, end)

            if key in seen_wave_keys:
                return

            seen_wave_keys.add(key)
            final_wave_objects.append((p_type, waves))

        except Exception:
            pass

    # ======================================================
    # 1) Existing original scan:
    #    Bullish impulse / bearish impulse / correction
    #    from the original low-based start area.
    # ======================================================
    curr_idx = idx_abs_low

    while curr_idx < idx_abs_high - 5:
        p_type, waves = find_best_fit(wa, curr_idx, idx_abs_high, gens, rules)

        if waves:
            add_wave_object(p_type, waves)

            next_idx = waves[-1].idx_end

            if next_idx <= curr_idx:
                curr_idx += 2
            else:
                curr_idx = next_idx

            if curr_idx >= idx_abs_high:
                break
        else:
            curr_idx += 2

    # ======================================================
    # 2) NEW parallel bearish scan:
    #    Start from absolute high area and scan forward.
    #    This helps detect Down → Up → Down → Up → Down.
    # ======================================================
    idx_bear_start = int(np.argmax(highs))
    idx_bear_low = int(np.argmin(lows))

    if idx_bear_low > idx_bear_start:
        idx_bear_end = idx_bear_low
    else:
        idx_bear_end = len(df_proc) - 1

    curr_idx = idx_bear_start

    while curr_idx < idx_bear_end - 5:
        p_type, waves = find_best_bearish_fit(
            wa,
            curr_idx,
            idx_bear_end,
            gens
        )

        if waves:
            add_wave_object(p_type, waves)

            next_idx = waves[-1].idx_end

            if next_idx <= curr_idx:
                curr_idx += 2
            else:
                curr_idx = next_idx

            if curr_idx >= idx_bear_end:
                break
        else:
            curr_idx += 2

    # Keep wave objects roughly chronological.
    try:
        final_wave_objects.sort(key=lambda x: x[1][0].idx_start if x[1] else 0)
    except Exception:
        pass

    df_raw_phase2 = df_proc.iloc[idx_abs_high:]
    json_waves = []

    for p_type, waves in final_wave_objects:
        labels = ["1", "2", "3", "4", "5"] if p_type in ["Impulse", "BearishImpulse"] else ["A", "B", "C"]

        for i, w in enumerate(waves):
            is_up = "Up" in w.__class__.__name__
            end_price = w.high if is_up else w.low
            end_idx = w.idx_end
            end_date = df_proc.iloc[end_idx]["Date"]
            lbl = labels[i] if i < len(labels) else "?"

            json_waves.append({
                "lbl": lbl,
                "pattern_type": p_type,
                "p": {
                    "price": float(end_price),
                    "date": str(end_date),
                    "type": "high" if is_up else "low"
                }
            })

    return json_waves, df_raw_phase2


# ==========================================================
# --- CORE BACKEND FUNCTION (CALL THIS FROM SERVER.PY) ---
# ==========================================================
def run_analysis_backend(symbol="BTCUSDT", interval="15m", action="analyze", market_type="crypto"):
    """
    Main entry point for API calls. Returns a DICT (not JSON string).
    """
    symbol = normalize_symbol(symbol)

    analysis_deadline_monotonic = (
        time.monotonic()
        + AI_ANALYSIS_DEADLINE_SECONDS
    )

    # 1. Initialize Keys
    if not load_api_key():
        if not hasattr(config, "API_KEY_POOL") and not hasattr(config, "PAID_API_KEY"):
            print("[WARN] No API keys found, AI might fail.")

    # --- ACTION: RECENT LOCKED SIGNALS ---
    if action == "recent_locked_signals":
        return get_recent_locked_signals(limit=5)

    # --- ACTION: SCAN ---
    if action == "scan":
        mode = symbol if symbol in ["scalp", "swing"] else "scalp"

        if market_type == "forex":
            try:
                from .ai_engine import scan_forex_market
                scan_result = scan_forex_market(mode=mode)
            except Exception:
                scan_result = {
                    "candidates": [],
                    "logs": ["Forex Scan Not implemented in modular refactor yet"]
                }
        else:
            scan_result = scan_market(mode=mode)

        final_output = {
            "candidates": scan_result.get("candidates", []),
            "logs": scan_result.get("logs", []),
            "ai_analysis": []
        }

        if final_output["candidates"]:
            try:
                ai_picks_str = analyze_scanner_candidates(final_output["candidates"])
                final_output["ai_analysis"] = json.loads(ai_picks_str)
            except Exception:
                final_output["ai_analysis"] = []

        return final_output

    # --- ACTION: HISTORY ---
    elif action == "history":
        df = get_data(symbol, interval=interval, period="200d", market_type=market_type)
        if not df.empty and "Date" in df.columns:
            df.set_index("Date", inplace=True)
        df = df.tail(1000)

        ohlc_data = []
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        for date, row in df.iterrows():
            ohlc_data.append({
                "time": int(date.timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"])
            })

        return {"ohlc_data": ohlc_data}

    # --- ACTION: STATUS ---
    elif action == "status":
        return refresh_existing_signal_guidance(
            symbol=symbol,
            interval=interval,
            market_type=market_type
        )  

    # --- ACTION: ANALYZE (DEFAULT) ---
    else:
        # --------------------------------------------------
        # EARLY GLOBAL LOCK CHECK
        # If this coin/timeframe already has a global active signal,
        # do NOT run full MTF/Elliott/backend analysis again.
        # Just refresh guidance for the existing signal.
        # --------------------------------------------------
        early_locked_scenario, early_lock_time = get_firestore_lock(symbol, interval)

        if early_locked_scenario and early_lock_time:
            refreshed_lock_result = refresh_existing_signal_guidance(
                symbol=symbol,
                interval=interval,
                market_type=market_type
            )

            try:
                status_msg = str(
                    refreshed_lock_result
                    .get("ai_prediction", {})
                    .get("trade_status_message", "")
                ).upper()
            except Exception:
                status_msg = ""

            # --------------------------------------------------
            # UX rule:
            # If old locked trade is LOST, do not return the old loss view.
            # The refresh_existing_signal_guidance() already deletes terminal LOST locks.
            # Continue this same request into fresh analysis and try to build a new setup.
            #
            # WON / MISSED / CANCELLED / EXPIRED can still be returned once,
            # because user may want to see that result.
            # --------------------------------------------------
            if "LOST" not in status_msg:
                return refreshed_lock_result

            print(
                f"[LOCK_REFRESH] Previous locked signal for {symbol} {interval} was LOST. "
                "Lock removed; continuing with fresh analysis."
            )

        elif early_locked_scenario and not early_lock_time:
            delete_firestore_lock(symbol, interval)

        # 1. LEAD AI v2: GET UNIVERSAL TIMEFRAMES
        tfs = get_dynamic_timeframes(interval)

        # 2. LOAD DYNAMIC DATA
        df_macro = get_data(symbol, interval=tfs["macro"]["tf"], period=tfs["macro"]["pd"], market_type=market_type)
        if not df_macro.empty and "Date" in df_macro.columns: df_macro.set_index("Date", inplace=True)

        df_swing = get_data(symbol, interval=tfs["swing"]["tf"], period=tfs["swing"]["pd"], market_type=market_type)
        if not df_swing.empty and "Date" in df_swing.columns: df_swing.set_index("Date", inplace=True)

        # Note: We keep the variable name 'df_confirm_1h' to avoid breaking old variables, but it represents 'Confirm TF' now.
        df_confirm_1h = get_data(symbol, interval=tfs["confirm"]["tf"], period=tfs["confirm"]["pd"], market_type=market_type)
        if not df_confirm_1h.empty and "Date" in df_confirm_1h.columns: df_confirm_1h.set_index("Date", inplace=True)

        df_micro = get_data(symbol, interval=tfs["micro"]["tf"], period=tfs["micro"]["pd"], market_type=market_type)
        if not df_micro.empty and "Date" in df_micro.columns: df_micro.set_index("Date", inplace=True)

        lower_exec_tf = get_lower_execution_timeframe(interval)
        df_lower_exec = pd.DataFrame()

        if lower_exec_tf:
            df_lower_exec = get_data(
                symbol,
                interval=lower_exec_tf["tf"],
                period=lower_exec_tf["pd"],
                market_type=market_type
            )
            if not df_lower_exec.empty and "Date" in df_lower_exec.columns:
                df_lower_exec.set_index("Date", inplace=True)

        # --------------------------------------------------
        # LIVE vs CLOSED-CANDLE ANALYSIS SPLIT
        # --------------------------------------------------
        analysis_candle_cutoff = pd.Timestamp.now(tz="UTC")
        crypto_candle_times = market_type == "crypto"

        df_macro_live, df_macro_analysis = split_live_and_analysis_df(
            df_macro,
            min_rows=100,
            interval=tfs["macro"]["tf"] if crypto_candle_times else None,
            now_utc=analysis_candle_cutoff,
        )
        _, df_swing_analysis = split_live_and_analysis_df(
            df_swing,
            min_rows=100,
            interval=tfs["swing"]["tf"] if crypto_candle_times else None,
            now_utc=analysis_candle_cutoff,
        )
        df_micro_live, df_micro_analysis = split_live_and_analysis_df(
            df_micro,
            min_rows=100,
            interval=tfs["micro"]["tf"] if crypto_candle_times else None,
            now_utc=analysis_candle_cutoff,
        )
        _, df_confirm_1h_analysis = split_live_and_analysis_df(
            df_confirm_1h,
            min_rows=100,
            interval=tfs["confirm"]["tf"] if crypto_candle_times else None,
            now_utc=analysis_candle_cutoff,
        )
        _, df_lower_exec_analysis = split_live_and_analysis_df(
            df_lower_exec,
            min_rows=80,
            interval=(
                lower_exec_tf["tf"]
                if crypto_candle_times and lower_exec_tf else None
            ),
            now_utc=analysis_candle_cutoff,
        )

        # --------------------------------------------------
        # HISTORICAL ANALOG ENGINE
        #
        # Shadow mode:
        # - Uses the already-fetched CLOSED candles.
        # - Makes no additional exchange/API requests.
        # - Does not change, approve or reject the current signal.
        # - Produces compact statistical evidence for later Gemini use.
        # --------------------------------------------------
        historical_analog_result = {
            "enabled": False,
            "status": "NOT_RUN",
            "engine_version": "1.0.0",
            "selected_timeframe": str(interval),
            "reason": "Historical analog analysis has not run.",
            "combined_forecast": {},
            "timeframes": {},
        }

        historical_analog_payload = {
            "available": False,
            "confidence": "UNAVAILABLE",
            "instruction": (
                "Historical analog evidence is unavailable. "
                "Do not invent historical probabilities."
            ),
        }

        try:
            historical_analog_result = analyze_historical_analogs(
                # Main user-selected timeframe:
                # 1h analysis -> 1h candles
                # 4h analysis -> 4h candles
                selected_df=df_micro_analysis,
                selected_timeframe=tfs["micro"]["tf"],

                # Lower execution / entry timing timeframe:
                # 1h analysis -> 15m
                # 4h analysis -> 1h
                lower_df=(
                    df_lower_exec_analysis
                    if df_lower_exec_analysis is not None
                    and not df_lower_exec_analysis.empty
                    else None
                ),
                lower_timeframe=(
                    lower_exec_tf.get("tf")
                    if isinstance(lower_exec_tf, dict)
                    else None
                ),

                # Confirmation timeframe:
                # 1h analysis -> 4h
                # 4h analysis -> 1d
                confirm_df=df_confirm_1h_analysis,
                confirm_timeframe=tfs["confirm"]["tf"],

                # Swing timeframe:
                # 1h analysis -> 1d
                # 4h analysis -> 1w
                swing_df=df_swing_analysis,
                swing_timeframe=tfs["swing"]["tf"],

                # Macro timeframe:
                # 1h analysis -> 1w
                # 4h analysis -> 1mo
                macro_df=df_macro_analysis,
                macro_timeframe=tfs["macro"]["tf"],

                # Keep internal analog paths in memory.
                # They will later be used to test the exact
                # AI-generated Entry / SL / TP1 / TP2 plan.
                include_internal=True,
            )

            historical_analog_payload = build_gemini_analog_payload(
                historical_analog_result
            )

            combined_analog = (
                historical_analog_payload.get("combined_forecast") or {}
            )

            print(
                f"[HISTORICAL_ANALOG] {symbol} {interval} | "
                f"direction={combined_analog.get('likely_direction', 'UNCLEAR')} | "
                f"LONG={float(combined_analog.get('long_probability') or 0.0):.2f} | "
                f"SHORT={float(combined_analog.get('short_probability') or 0.0):.2f} | "
                f"RANGE={float(combined_analog.get('range_probability') or 0.0):.2f} | "
                f"alignment={combined_analog.get('multi_timeframe_alignment', 'UNKNOWN')} | "
                f"confidence={combined_analog.get('confidence', 'UNAVAILABLE')}"
            )

        except Exception as e:
            print(f"[HISTORICAL_ANALOG] analysis skipped/error: {e}")

        macro_waves, _ = analyze_structure_phase1_only(df_macro_analysis)
        swing_waves, _ = analyze_structure_phase1_only(df_swing_analysis)
        confirm_1h_waves, _ = analyze_structure_phase1_only(df_confirm_1h_analysis)
        micro_waves, _ = analyze_structure_phase1_only(df_micro_analysis)

        # --------------------------------------------------
        # MTF Elliott confirmation context
        # Used by the AI market packet and final Elliott evidence.
        # --------------------------------------------------
        elliott_mtf_context = {
            "execution_tf": tfs["micro"]["tf"],
            "confirm_tf": tfs["confirm"]["tf"],
            "swing_tf": tfs["swing"]["tf"],
            "macro_tf": tfs["macro"]["tf"],

            "confirm_waves": confirm_1h_waves,
            "swing_waves": swing_waves,
            "macro_waves": macro_waves,
            "micro_waves": micro_waves,

            "symbol": symbol,
            "user_tf": interval,
        }

        mtf_elliott_context_text = (
            "\n\n--- MTF ELLIOTT CONFIRMATION INPUT ---\n"
            f"Execution TF: {elliott_mtf_context.get('execution_tf')}\n"
            f"Confirm TF: {elliott_mtf_context.get('confirm_tf')} | Waves: {len(confirm_1h_waves or [])}\n"
            f"Swing TF: {elliott_mtf_context.get('swing_tf')} | Waves: {len(swing_waves or [])}\n"
            f"Macro TF: {elliott_mtf_context.get('macro_tf')} | Waves: {len(macro_waves or [])}\n"
            "Usage: Multi-timeframe Elliott structure is quantitative evidence for the final AI decision."
        )

        # --------------------------------------------------
        # Elliott full phase context
        # Uses detected 1-2-3-4-5 / A-B-C wave labels
        # to classify current Elliott phase.
        # --------------------------------------------------
        elliott_phase_context = detect_elliott_phase_context(
            execution_waves=micro_waves,
            confirm_waves=confirm_1h_waves,
            swing_waves=swing_waves,
            macro_waves=macro_waves,
            execution_tf=tfs["micro"]["tf"],
            confirm_tf=tfs["confirm"]["tf"],
            swing_tf=tfs["swing"]["tf"],
            macro_tf=tfs["macro"]["tf"],
        )

        elliott_phase_context_text = format_elliott_phase_context_text(
            elliott_phase_context
        )

        # Momentum breakout is detected after backend_trend_regime.
        # It remains post-signal shadow/diagnostic evidence only and
        # must never be appended to Gemini's market context.
        momentum_breakout_context = {}

        market_context_text = build_market_context_text(
            df_micro_analysis,
            micro_waves,
            interval=interval
        )

        try:
            # --------------------------------------------------
            # RSI DIVERGENCE CONTEXT
            # Confirmation / warning only.
            #
            # Rule:
            # - 15m setups use 1H RSI divergence to avoid noisy 15m divergence.
            # - 1H setups use 1H RSI divergence.
            # - 4H setups use 4H RSI divergence.
            # - 1D setups use 1D RSI divergence.
            # --------------------------------------------------
            rsi_user_tf = str(interval or "").lower().strip()

            if rsi_user_tf in ["15m"]:
                rsi_divergence_df = df_confirm_1h_analysis
                rsi_divergence_tf = tfs["confirm"]["tf"]  # usually 1h
                rsi_min_price_diff_pct = 0.20
                rsi_min_rsi_diff = 4.0
            else:
                rsi_divergence_df = df_micro_analysis
                rsi_divergence_tf = tfs["micro"]["tf"]    # user selected timeframe
                rsi_min_price_diff_pct = 0.15
                rsi_min_rsi_diff = 3.0

            rsi_divergence_context = detect_rsi_divergence(
                df=rsi_divergence_df,
                market_context_text=market_context_text,
                rsi_period=14,
                pivot_left=2,
                pivot_right=2,
                lookback=120,
                min_price_diff_pct=rsi_min_price_diff_pct,
                min_rsi_diff=rsi_min_rsi_diff,
                max_bars_between_pivots=80,
            )

            rsi_divergence_context["timeframe"] = rsi_divergence_tf
            rsi_divergence_context["usage_note"] = (
                f"RSI divergence checked on {rsi_divergence_tf}. "
                "Used as confirmation/warning only, not as a standalone signal."
            )
            rsi_divergence_context_text = format_rsi_divergence_context_text(
                rsi_divergence_context
            )
        except Exception as e:
            rsi_divergence_context = {
                "has_divergence": False,
                "divergence_type": "NONE",
                "side": "NONE",
                "reason": f"RSI divergence detection error: {str(e)}",
            }
            rsi_divergence_context_text = (
                "--- RSI DIVERGENCE CONTEXT ---\n"
                f"RSI divergence detection error: {str(e)}\n"
                "Usage Rule: RSI divergence is confirmation/warning only."
            )

        market_context_text += mtf_elliott_context_text
        market_context_text += "\n\n" + elliott_phase_context_text
        market_context_text += "\n\n" + rsi_divergence_context_text
        
        recent_candle_context_text = build_recent_candle_rejection_context(
            df_micro_analysis,
            market_context_text=market_context_text,
            use_last_closed=(market_type != "crypto"),
        )
        market_context_text += "\n\n" + recent_candle_context_text

        execution_price_context_text = build_execution_price_context(df_micro_live, df_micro_analysis)
        market_context_text += "\n\n" + execution_price_context_text

        # --- DOMINANT TREND / DIRECTION CONTEXT ---
        try:
            dominant_trend_context_text = build_dominant_trend_context(
                df_macro=df_macro_analysis,
                df_swing=df_swing_analysis,
                df_confirm_1h=df_confirm_1h_analysis,
                df_micro=df_micro_analysis,
                macro_waves=macro_waves,
                swing_waves=swing_waves,
                confirm_1h_waves=confirm_1h_waves,
                micro_waves=micro_waves,
                user_tf=interval
            )
            market_context_text += "\n\n" + dominant_trend_context_text
        except Exception as e:
            dominant_trend_context_text = ""
            market_context_text += (
                "\n\n--- DOMINANT TREND CONTEXT ---\n"
                f"Dominant Trend Error: {str(e)}"
            )

        # --- BACKEND TREND REGIME + STRUCTURED MARKET EVIDENCE ---
        # The backend calculates market evidence only.
        # Fresh Entry / SL / TP prices are generated by Gemini
        # and accepted only after deterministic validation.
        backend_trend_regime = {}
        context_bundle = {}
        cross_tf_setups = []

        # Market evidence that Gemini may inspect.
        # This must never include a backend-generated trade plan.
        ai_market_context_text = market_context_text

        try:
            # --- TEMP STRUCTURED CONTEXT BUNDLE ---
            # Build once before trend_regime so trend_regime can see CHoCH,
            # dominant WAIT, rejection/chop, Fib status, and context conflicts.
            context_bundle = build_context_bundle(
                symbol=symbol,
                timeframe=interval,
                market_context_text=market_context_text,
                recent_candle_context_text=recent_candle_context_text,
                execution_price_context_text=execution_price_context_text,
                dominant_trend_context_text=dominant_trend_context_text,
                backend_trend_regime={},
                macro_waves=macro_waves,
                swing_waves=swing_waves,
                confirm_1h_waves=confirm_1h_waves,
                micro_waves=micro_waves,
            )

            backend_trend_regime = detect_backend_trend_regime(
                df_macro=df_macro_analysis,
                df_swing=df_swing_analysis,
                df_confirm=df_confirm_1h_analysis,
                df_micro=df_micro_analysis,
                user_tf=interval,
                context_bundle=context_bundle
            )

            backend_trend_regime_text = format_backend_trend_regime_context(
                backend_trend_regime
            )
            market_context_text += "\n\n" + backend_trend_regime_text

            # --------------------------------------------------
            # Lower timeframe momentum breakout context
            # Run AFTER backend_trend_regime so breakout side follows trend.
            #
            # 1h setup -> check 15m
            # 4h setup -> check 1h
            # 1d setup -> check 4h
            # 15m setup -> none for now
            # --------------------------------------------------
            try:
                preferred_side = str(
                    backend_trend_regime.get("preferred_side")
                    or backend_trend_regime.get("market_bias")
                    or ""
                ).upper().strip()

                if preferred_side not in ["LONG", "SHORT"]:
                    preferred_side = "ANY"

                momentum_breakout_context = detect_momentum_breakout(
                    df_lower=df_lower_exec_analysis if lower_exec_tf else None,
                    side_hint=preferred_side,
                    timeframe=(lower_exec_tf or {}).get("tf", "none"),
                )

            except Exception as e:
                momentum_breakout_context = {
                    "enabled": False,
                    "momentum_breakout": False,
                    "side": "UNKNOWN",
                    "score": 0,
                    "strength": "NONE",
                    "timeframe": (lower_exec_tf or {}).get("tf", "none"),
                    "reason": f"Momentum breakout detection failed after trend regime: {e}",
                }

            # Keep momentum breakout available for post-signal
            # shadow diagnostics only. Do not append it to
            # market_context_text or ai_market_context_text.

            # --- FINAL STRUCTURED CONTEXT BUNDLE ---
            # Rebuild with backend_trend_regime included so Gemini
            # receives the complete deterministic market evidence.
            context_bundle = build_context_bundle(
                symbol=symbol,
                timeframe=interval,
                market_context_text=market_context_text,
                recent_candle_context_text=recent_candle_context_text,
                execution_price_context_text=execution_price_context_text,
                dominant_trend_context_text=dominant_trend_context_text,
                backend_trend_regime=backend_trend_regime,
                macro_waves=macro_waves,
                swing_waves=swing_waves,
                confirm_1h_waves=confirm_1h_waves,
                micro_waves=micro_waves,
            )

            market_context_text += "\n\n" + summarize_context_bundle(
                context_bundle
            )

            # Freeze the AI-visible quantitative market context here.
            #
            # Everything above this point contains AI-visible market evidence:
            # Elliott, trend, S/R, Fib, liquidity, candle context,
            # divergence and regime information.
            # Momentum breakout remains excluded and shadow-only.
            #
            # No backend-selected side / Entry / SL / TP plan is
            # generated in the fresh AI-first flow.
            ai_market_context_text = market_context_text

            cross_tf_setups = get_cross_timeframe_setups(
                symbol,
                interval,
            )

        except Exception as e:
            # Preserve any quantitative evidence that was calculated
            # before the market-evidence pipeline failed.
            ai_market_context_text = (
                market_context_text
            )

            market_context_text += (
                "\n\n--- BACKEND MARKET EVIDENCE ---\n"
                f"Market-evidence calculation error: {str(e)}"
            )

        # print("\n===== MARKET CONTEXT SENT TO AI =====")
        # ==================================================
        # AI MARKET PACKET
        #
        # Builds the complete structured evidence packet that
        # will later be sent to Gemini 3.5 Flash.
        #
        # Important:
        # - Reuses already-fetched CLOSED candles.
        # - Makes no additional Binance/API request.
        # - Does not generate or modify a trade here.
        # - Does not send the backend-selected setup to Gemini.
        # ==================================================
        ai_market_packet = {}

        ai_market_packet_validation = {
            "valid": False,
            "errors": [
                "AI market packet has not been built."
            ],
            "warnings": [],
        }

        try:
            # ----------------------------------------------
            # Resolve live and last-closed prices safely.
            # ----------------------------------------------
            current_market_price = None
            last_closed_market_price = None

            try:
                if (
                    df_micro_live is not None
                    and not df_micro_live.empty
                    and "Close" in df_micro_live.columns
                ):
                    current_market_price = float(
                        df_micro_live["Close"].iloc[-1]
                    )
            except Exception:
                current_market_price = None

            try:
                if (
                    df_micro_analysis is not None
                    and not df_micro_analysis.empty
                    and "Close" in df_micro_analysis.columns
                ):
                    last_closed_market_price = float(
                        df_micro_analysis["Close"].iloc[-1]
                    )
            except Exception:
                last_closed_market_price = None

            if current_market_price is None:
                current_market_price = last_closed_market_price

            # ----------------------------------------------
            # Structured quantitative evidence.
            #
            # No backend-selected trade plan is created or included.
            # Gemini must independently analyse the market evidence.
            # ----------------------------------------------
            WAVE_POINT_FIELD_ORDER = (
                "label",
                "pattern_type",
                "price",
                "date",
                "point_type",
            )

            def _compact_wave_points_for_ai(waves, limit=12):
                """
                Keep only the latest compact Elliott points for Gemini.

                Full wave arrays remain available to local analysis,
                deterministic modules and the frontend response. This helper
                only reduces the duplicated copy inside the AI packet.
                """
                if not isinstance(waves, list):
                    return []

                compact_points = []

                for item in waves[-max(1, int(limit)):]:
                    if not isinstance(item, dict):
                        continue

                    point = item.get("p") or {}

                    compact_points.append([
                        item.get("lbl"),
                        item.get("pattern_type"),
                        point.get("price"),
                        point.get("date"),
                        point.get("type"),
                    ])

                return compact_points

            def _prune_empty_ai_values(value):
                """
                Remove only empty mapping fields from the AI-only context.

                False, zero and list positions remain unchanged. This is
                important for boolean evidence and field-ordered row data.
                """
                if isinstance(value, dict):
                    compact_mapping = {}

                    for key, item in value.items():
                        compact_item = _prune_empty_ai_values(item)

                        is_empty = (
                            compact_item is None
                            or (
                                isinstance(compact_item, str)
                                and not compact_item.strip()
                            )
                            or (
                                isinstance(
                                    compact_item,
                                    (
                                        dict,
                                        list,
                                        tuple,
                                    ),
                                )
                                and len(compact_item) == 0
                            )
                        )

                        if is_empty:
                            continue

                        compact_mapping[key] = compact_item

                    return compact_mapping

                if isinstance(value, list):
                    return [
                        _prune_empty_ai_values(item)
                        for item in value
                    ]

                if isinstance(value, tuple):
                    return [
                        _prune_empty_ai_values(item)
                        for item in value
                    ]

                return value

            compact_elliott_mtf_context_for_ai = {
                "execution_tf": elliott_mtf_context.get("execution_tf"),
                "confirm_tf": elliott_mtf_context.get("confirm_tf"),
                "swing_tf": elliott_mtf_context.get("swing_tf"),
                "macro_tf": elliott_mtf_context.get("macro_tf"),
                "symbol": symbol,
                "user_tf": interval,
                "execution_wave_count": len(micro_waves or []),
                "confirm_wave_count": len(confirm_1h_waves or []),
                "swing_wave_count": len(swing_waves or []),
                "macro_wave_count": len(macro_waves or []),
            }

            # Compact, side-neutral institutional evidence for Gemini.
            # Uses CLOSED candles already loaded for the main and confirm TFs.
            # This evidence is advisory-only and cannot select a side,
            # block a trade, or change Entry / SL / TP prices.
            side_neutral_smc_evidence_for_ai = {}

            try:
                side_neutral_smc_evidence_for_ai = {
                    "main": build_side_neutral_smc_evidence(
                        df_selected=df_micro_analysis,
                        interval=tfs["micro"]["tf"],
                    ),
                    "confirmation": build_side_neutral_smc_evidence(
                        df_selected=df_confirm_1h_analysis,
                        interval=tfs["confirm"]["tf"],
                    ),
                }
            except Exception as e:
                side_neutral_smc_evidence_for_ai = {
                    "available": False,
                    "mode": "SIDE_NEUTRAL_ADVISORY",
                    "decision_authority": "NONE",
                    "can_block_trade": False,
                    "can_change_prices": False,
                    "reason": str(e)[:300],
                }

            # Compact side-neutral liquidity evidence.
            # No raw candle arrays are added here.
            side_neutral_liquidity_for_ai = {}

            try:
                side_neutral_liquidity_for_ai = (
                    build_side_neutral_liquidity_context(
                        df_analysis=(
                            df_micro_analysis
                        ),
                        df_lower=(
                            df_lower_exec_analysis
                            if lower_exec_tf
                            else None
                        ),
                        current_price=(
                            latest_close_price(
                                df_micro_analysis
                            )
                        ),
                    )
                )

            except Exception as e:
                side_neutral_liquidity_for_ai = {
                    "available": False,
                    "closed_candles_only": True,
                    "decision_authority": (
                        "EVIDENCE_ONLY"
                    ),
                    "reason": str(e)[:300],
                }

            # Build an AI-only compact copy.
            # Important parsed evidence remains available, while duplicate
            # raw text, trend-regime and wave copies are removed.
            compact_context_bundle_for_ai = (
                deepcopy(context_bundle)
                if isinstance(context_bundle, dict)
                else {}
            )

            # Already sent separately as:
            # trend_and_structure.backend_trend_regime
            compact_context_bundle_for_ai.pop(
                "backend_trend_regime",
                None,
            )

            # Already sent in the richer:
            # elliott_wave_analysis section.
            compact_context_bundle_for_ai.pop(
                "waves",
                None,
            )

            # Parsed Fibonacci values remain. Remove only their
            # repeated original raw-text copies.
            fibonacci_for_ai = (
                compact_context_bundle_for_ai.get(
                    "fibonacci"
                )
            )

            if isinstance(fibonacci_for_ai, dict):
                for fib_name in (
                    "elliott",
                    "visible",
                ):
                    fib_block = (
                        fibonacci_for_ai.get(
                            fib_name
                        )
                    )

                    if isinstance(fib_block, dict):
                        fib_block.pop(
                            "raw",
                            None,
                        )

            # Status and individual confluence items remain.
            # Only the repeated raw paragraph is removed.
            automated_confluence_for_ai = (
                compact_context_bundle_for_ai.get(
                    "automated_confluence"
                )
            )

            if isinstance(
                automated_confluence_for_ai,
                dict,
            ):
                automated_confluence_for_ai.pop(
                    "raw",
                    None,
                )

            quantitative_context_for_ai = {
                "institutional_evidence": {
                    "smc_liquidity_fvg": (
                        side_neutral_smc_evidence_for_ai
                    ),
                    "local_liquidity": (
                        side_neutral_liquidity_for_ai
                    ),
                },
                "elliott_wave_analysis": {
                    "multi_timeframe_context": (
                        compact_elliott_mtf_context_for_ai
                    ),

                    "phase_context": (
                        elliott_phase_context
                        if isinstance(elliott_phase_context, dict)
                        else {}
                    ),

                    "wave_point_encoding": (
                        "FIELD_ORDERED_ROWS"
                    ),

                    "wave_point_fields": list(
                        WAVE_POINT_FIELD_ORDER
                    ),

                    "detected_waves": {
                        "macro": _compact_wave_points_for_ai(
                            macro_waves,
                            8,
                        ),
                        "swing": _compact_wave_points_for_ai(
                            swing_waves,
                            10,
                        ),
                        "confirmation": _compact_wave_points_for_ai(
                            confirm_1h_waves,
                            12,
                        ),
                        "execution": _compact_wave_points_for_ai(
                            micro_waves,
                            16,
                        ),
                    },

                    "wave_counts": {
                        "macro": len(macro_waves or []),
                        "swing": len(swing_waves or []),
                        "confirmation": len(confirm_1h_waves or []),
                        "execution": len(micro_waves or []),
                    },
                },

                "trend_and_structure": {
                    "backend_trend_regime": (
                        backend_trend_regime
                        if isinstance(
                            backend_trend_regime,
                            dict,
                        )
                        else {}
                    ),

                    "context_bundle": (
                        compact_context_bundle_for_ai
                    ),

                    "cross_timeframe_active_setups": (
                        cross_tf_setups or []
                    ),
                },

                "momentum_and_confirmation": {
                    # Momentum breakout remains backend shadow/diagnostic data.
                    # It is intentionally excluded from the Gemini pre-signal
                    # market packet because it must not influence fresh signals.
                    "rsi_divergence": (
                        rsi_divergence_context
                        if isinstance(
                            rsi_divergence_context,
                            dict,
                        )
                        else {}
                    ),
                },

                "timeframe_roles": {
                    "lower_execution": (
                        (lower_exec_tf or {}).get("tf")
                        if isinstance(lower_exec_tf, dict)
                        else None
                    ),

                    "main_execution": tfs["micro"]["tf"],
                    "confirmation": tfs["confirm"]["tf"],
                    "swing": tfs["swing"]["tf"],
                    "macro": tfs["macro"]["tf"],
                },
            }

            quantitative_context_for_ai = (
                _prune_empty_ai_values(
                    quantitative_context_for_ai
                )
            )

            # ----------------------------------------------
            # Build the complete packet.
            # ----------------------------------------------
            ai_market_packet = build_ai_market_packet(
                symbol=symbol,

                selected_timeframe=tfs["micro"]["tf"],

                market_type=market_type,

                current_price=current_market_price,

                last_closed_price=last_closed_market_price,

                analysis_timestamp=(
                    df_micro_analysis.index[-1]
                    if (
                        df_micro_analysis is not None
                        and not df_micro_analysis.empty
                    )
                    else datetime.now(timezone.utc)
                ),

                timeframe_data={
                    "lower": {
                        "timeframe": (
                            (lower_exec_tf or {}).get("tf")
                            if isinstance(lower_exec_tf, dict)
                            else ""
                        ),

                        "df": (
                            df_lower_exec_analysis
                            if (
                                df_lower_exec_analysis is not None
                                and not df_lower_exec_analysis.empty
                            )
                            else None
                        ),
                    },

                    "main": {
                        "timeframe": tfs["micro"]["tf"],
                        "df": df_micro_analysis,
                    },

                    "confirm": {
                        "timeframe": tfs["confirm"]["tf"],
                        "df": df_confirm_1h_analysis,
                    },

                    "swing": {
                        "timeframe": tfs["swing"]["tf"],
                        "df": df_swing_analysis,
                    },

                    "macro": {
                        "timeframe": tfs["macro"]["tf"],
                        "df": df_macro_analysis,
                    },
                },

                # Limit only the candle copies sent to Gemini.
                # Full local dataframes and analysis remain unchanged.
                candle_limits={
                    "lower": 24,
                    "main": 32,
                    "confirm": 24,
                    "swing": 12,
                    "macro": 8,
                },

                quantitative_context=(
                    quantitative_context_for_ai
                ),

                historical_analog=(
                    historical_analog_payload
                    if isinstance(
                        historical_analog_payload,
                        dict,
                    )
                    else {}
                ),

                previous_trade_context={},
            )

            ai_market_packet_validation = (
                validate_ai_market_packet(
                    ai_market_packet
                )
            )

            packet_size = (
                ai_market_packet.get(
                    "packet_size",
                    {},
                )
                if isinstance(
                    ai_market_packet,
                    dict,
                )
                else {}
            )

            print(
                f"[AI_MARKET_PACKET] {symbol} {interval} | "
                f"valid={ai_market_packet_validation.get('valid')} | "
                f"errors={len(ai_market_packet_validation.get('errors') or [])} | "
                f"warnings={len(ai_market_packet_validation.get('warnings') or [])} | "
                f"rough_tokens={packet_size.get('rough_token_estimate')}"
            )

        except Exception as e:
            ai_market_packet = {}

            ai_market_packet_validation = {
                "valid": False,
                "errors": [
                    f"AI market packet build failed: {str(e)}"
                ],
                "warnings": [],
            }

            print(
                f"[AI_MARKET_PACKET] build skipped/error: {e}"
            )

        locked_scenario, lock_time = get_firestore_lock(symbol, interval)
        status_message = "LIMIT ORDER | Fresh Setup"

        previous_trade_status = None
        previous_trade_advice = None
        previous_trade_scenario = None

        if locked_scenario and lock_time:
            if not isinstance(df_micro_live.index, pd.DatetimeIndex):
                df_micro_live.index = pd.to_datetime(df_micro_live.index)
            
            # --- TIMEZONE BUG FIX ---
            # Firebase UTC Time එක Chart UTC Time එකට හරියටම ගැලපීම
            try:
                if df_micro_live.index.tz is None and hasattr(lock_time, "tzinfo") and lock_time.tzinfo:
                    from datetime import timezone
                    lock_time = lock_time.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                pass

            df_since_lock = df_micro_live[df_micro_live.index >= lock_time]

            if df_since_lock.empty:
                df_since_lock = df_micro_live.tail(1)
            status, advice = check_trade_status_historical(
                locked_scenario,
                df_since_lock,
                interval=interval
            )

            # Use the same authoritative guidance engine
            # used by the normal status refresh flow.
            guidance_facts_for_lock = (
                build_trade_guidance_facts(
                    scenario=locked_scenario,

                    df_live=(
                        df_micro_live.tail(350)
                    ),

                    df_since_lock=(
                        df_since_lock
                    ),

                    interval=interval,

                    existing_status=status,

                    existing_advice=advice,

                    df_confirm=(
                        df_confirm_1h.tail(350)
                    ),
                )
            )

            management_strategy = {
                "phase": (
                    guidance_facts_for_lock.get(
                        "guidance_status",
                        status,
                    )
                ),

                "advice": (
                    guidance_facts_for_lock.get(
                        "user_guidance",
                        advice,
                    )
                ),

                "action": (
                    guidance_facts_for_lock.get(
                        "action",
                        "",
                    )
                ),

                "risk_level": (
                    guidance_facts_for_lock.get(
                        "risk_level",
                        "",
                    )
                ),

                "danger_score": (
                    guidance_facts_for_lock.get(
                        "danger_score",
                        0,
                    )
                ),

                "danger_flags": (
                    guidance_facts_for_lock.get(
                        "danger_flags",
                        [],
                    )
                ),

                "management_mode": (
                    guidance_facts_for_lock.get(
                        "management_mode",
                        "",
                    )
                ),

                "tp2_probability": (
                    guidance_facts_for_lock.get(
                        "tp2_probability",
                        "",
                    )
                ),

                "reversal_risk": (
                    guidance_facts_for_lock.get(
                        "reversal_risk",
                        "",
                    )
                ),

                "profit_action": (
                    guidance_facts_for_lock.get(
                        "profit_action",
                        "",
                    )
                ),

                "sl_management": (
                    guidance_facts_for_lock.get(
                        "sl_management",
                        "",
                    )
                ),

                "exit_trigger": (
                    guidance_facts_for_lock.get(
                        "exit_trigger",
                        "",
                    )
                ),

                "confirm_tf": (
                    tfs["confirm"]["tf"]
                ),
            }

            advice = (
                guidance_facts_for_lock.get(
                    "user_guidance"
                )
                or advice
            )
            
            # --- PNL DISPLAY FIX FOR LOCKED TRADES ---
            if status == "RUNNING":
                status_message = f"RUNNING | {advice}"
            else:
                status_message = status

            if status in ["WON", "LOST"]:
                last_price = None
                try:
                    last_price = float(df_since_lock.iloc[-1]["Close"])
                except Exception:
                    pass

                if (
                    FIREBASE_AVAILABLE
                    and db is not None
                ):
                    try:
                        ok, audit_msg = (
                            audit_close_trade(
                                db=db,
                                symbol=symbol,
                                timeframe=interval,
                                result=status,
                                reason=advice,
                                close_price=last_price,
                            )
                        )

                        if not ok:
                            print(
                                "[TRADE_AUDIT] close "
                                f"skipped/error: {audit_msg}"
                            )

                    except Exception as e:
                        print(
                            "[TRADE_AUDIT] close "
                            f"skipped/error: {e}"
                        )

                else:
                    print(
                        "[TRADE_AUDIT] close skipped: "
                        "Firebase is unavailable."
                    )

                previous_trade_status = f"TRADE CLOSED ({status})"
                previous_trade_advice = advice
                previous_trade_scenario = dict(locked_scenario or {})
                previous_trade_scenario["summary"] = (
                    f"Previous locked setup closed as {status}. {advice}"
                )

                delete_firestore_lock(symbol, interval)
                locked_scenario = None
                status_message = "LIMIT ORDER | Fresh Setup"


            elif status in ["MISSED", "CANCELLED", "EXPIRED"]:
                previous_trade_status = f"{status} | {advice}"
                previous_trade_advice = advice
                previous_trade_scenario = dict(locked_scenario or {})
                previous_trade_scenario["probability"] = "Low"
                previous_trade_scenario["generated_advice"] = advice
                previous_trade_scenario["summary"] = (
                    f"Previous locked setup ended as {status}. {advice} Do not chase this old setup."
                )

                delete_firestore_lock(symbol, interval)
                locked_scenario = None
                status_message = "LIMIT ORDER | Fresh Setup"


            else:
                locked_scenario["generated_advice"] = advice
                ohlc_data = []
                chart_df = df_macro_analysis if not df_macro_analysis.empty else df_macro_live

                if not isinstance(chart_df.index, pd.DatetimeIndex):
                    chart_df.index = pd.to_datetime(chart_df.index)

                for date, row in chart_df.iterrows():
                    ohlc_data.append({
                        "time": int(date.timestamp()),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"])
                    })

                locked_ai_data = {
                    "trade_status_message": (
                        status_message
                    ),

                    "action_advice": (
                        advice
                    ),

                    "management_strategy": (
                        management_strategy
                    ),

                    "macro_analysis": {
                        "trend": (
                            "LOCKED (Monitoring)"
                        ),
                    },

                    "micro_analysis": {
                        "wave_personality": (
                            "Monitoring Phase"
                        ),
                    },

                    "trade_scenarios": [
                        locked_scenario
                    ],

                    "selected_scenario": (
                        locked_scenario
                    ),

                    "backend_trend_regime": (
                        backend_trend_regime
                    ),

                    "context_bundle": (
                        context_bundle
                    ),

                    "historical_analog_shadow": (
                        historical_analog_payload
                        if isinstance(
                            historical_analog_payload,
                            dict,
                        )
                        else {}
                    ),
                }

                return {
                    "meta": {
                        "symbol": symbol,

                        "timeframe": (
                            f"{interval} "
                            f"({tfs['confirm']['tf']} Confirm)"
                        ),
                    },

                    "waves": {
                        "macro": macro_waves,
                        "swing": swing_waves,

                        "confirmation_1h": (
                            confirm_1h_waves
                        ),

                        "micro": micro_waves,
                    },

                    "ohlc_data": ohlc_data,

                    "ai_prediction": (
                        locked_ai_data
                    ),
                }
            
        elif locked_scenario and not lock_time:
            delete_firestore_lock(symbol, interval)
            locked_scenario = None



        # ==================================================
        # REFRESH AI PACKET AFTER LATE LOCK PROCESSING
        #
        # The market packet was originally built before the
        # late Firestore lock/status check.
        #
        # If an old trade closed during this same request,
        # add compact terminal-trade context before Gemini
        # creates the next fresh signal.
        #
        # Also replace the raw pre-recheck watch object with
        # the latest compact watch-recheck facts.
        # ==================================================
        try:
            compact_previous_trade = {}

            if (
                previous_trade_status
                or isinstance(
                    previous_trade_scenario,
                    dict,
                )
            ):
                previous_scenario = (
                    previous_trade_scenario
                    if isinstance(
                        previous_trade_scenario,
                        dict,
                    )
                    else {}
                )

                previous_targets = (
                    previous_scenario.get(
                        "targets"
                    )
                    or []
                )

                previous_tp1 = None
                previous_tp2 = None

                if isinstance(
                    previous_targets,
                    list,
                ):
                    if len(previous_targets) > 0:
                        previous_tp1 = extract_price(
                            previous_targets[0]
                        )

                    if len(previous_targets) > 1:
                        previous_tp2 = extract_price(
                            previous_targets[1]
                        )

                if previous_tp1 is None:
                    previous_tp1 = extract_price(
                        previous_scenario.get(
                            "tp1"
                        )
                    )

                if previous_tp2 is None:
                    previous_tp2 = (
                        extract_price(
                            previous_scenario.get(
                                "tp2"
                            )
                        )
                        or extract_price(
                            previous_scenario.get(
                                "target"
                            )
                        )
                    )

                compact_previous_trade = {
                    "exists": True,

                    "status": (
                        previous_trade_status
                        or ""
                    ),

                    "advice": (
                        previous_trade_advice
                        or ""
                    ),

                    "side": str(
                        previous_scenario.get(
                            "trade_type"
                        )
                        or previous_scenario.get(
                            "side"
                        )
                        or "NONE"
                    ).upper().strip(),

                    "entry": extract_price(
                        previous_scenario.get(
                            "entry"
                        )
                        or previous_scenario.get(
                            "entry_zone"
                        )
                    ),

                    "stop_loss": extract_price(
                        previous_scenario.get(
                            "stop_loss"
                        )
                        or previous_scenario.get(
                            "invalidation"
                        )
                    ),

                    "tp1": previous_tp1,
                    "tp2": previous_tp2,

                    "setup_type": str(
                        previous_scenario.get(
                            "setup_type"
                        )
                        or ""
                    ),

                    "instruction": (
                        "This trade is already closed. "
                        "Do not reuse its Entry, SL or TP prices. "
                        "Use it only as recent outcome and "
                        "overtrading-risk context."
                    ),
                }

            if isinstance(
                ai_market_packet,
                dict,
            ):
                previous_context = (
                    ai_market_packet.get(
                        "previous_trade_context"
                    )
                )

                if not isinstance(
                    previous_context,
                    dict,
                ):
                    previous_context = {}

                previous_context.update(
                    {
                        "just_closed_trade": (
                            compact_previous_trade
                        ),
                    }
                )

                ai_market_packet[
                    "previous_trade_context"
                ] = previous_context

                # Packet was modified after its original
                # sanitizer pass, so sanitize it again.
                ai_market_packet = (
                    sanitize_ai_market_packet(
                        ai_market_packet
                    )
                )

                ai_market_packet_validation = (
                    validate_ai_market_packet(
                        ai_market_packet
                    )
                )

                print(
                    "[AI_MARKET_PACKET] "
                    "post-lock context refreshed | "
                    f"previous_trade="
                    f"{bool(compact_previous_trade)} | "
                    f"valid="
                    f"{ai_market_packet_validation.get('valid')}"
                )

        except Exception as e:
            # Do not destroy an otherwise valid market packet
            # only because optional previous-context refresh failed.
            print(
                "[AI_MARKET_PACKET] "
                f"post-lock refresh skipped/error: {e}"
            )

        

        # =================================================================
        # AI-FIRST FRESH SIGNAL FLOW
        #
        # Market evidence packet
        #     -> Gemini independent signal draft
        #     -> deterministic backend validation
        #     -> one Gemini correction attempt
        #     -> final validated signal or safe NO_TRADE
        #
        # No backend setup planner generates fresh trade prices.
        # Gemini proposes the plan; the validator owns approval and caps.
        # =================================================================

        ai_first_draft_result = {}
        ai_first_validation = {}
        ai_first_correction_result = {}
        ai_first_correction_validation = {}

        final_ai_signal = None
        final_ai_validation = {}
        final_ai_provider = {}
        final_liquidity_execution = {}

        ai_correction_used = False

        ai_first_skip = False
        ai_first_skip_reason = ""

        # --------------------------------------------------
        # Deterministic pre-AI hard gates only.
        #
        # Gemini is skipped only by deterministic packet or
        # safety failures, never by a backend setup planner.
        # --------------------------------------------------
        if not ai_market_packet_validation.get("valid"):
            ai_first_skip = True

            packet_errors = (
                ai_market_packet_validation.get(
                    "errors"
                )
                or []
            )

            ai_first_skip_reason = (
                "AI market packet failed validation. "
                + (
                    " | ".join(
                        str(item)
                        for item in packet_errors[:5]
                    )
                    if packet_errors
                    else "Required market evidence is unavailable."
                )
            )

        elif bool(
            backend_trend_regime.get(
                "hard_block_active"
            )
        ):
            ai_first_skip = True

            ai_first_skip_reason = str(
                backend_trend_regime.get(
                    "context_conflict_reason"
                )
                or (
                    "Deterministic trend-regime safety gate "
                    "detected a hard market conflict."
                )
            )

        # --------------------------------------------------
        # Safe fallback builder.
        # --------------------------------------------------
        def _build_validated_safe_no_trade(
            reason,
            previous_signal=None,
        ):
            safe_signal = (
                build_safe_no_trade_signal(
                    reason=str(
                        reason
                        or (
                            "Signal could not pass "
                            "deterministic validation."
                        )
                    ),

                    previous_signal=(
                        previous_signal
                        if isinstance(
                            previous_signal,
                            dict,
                        )
                        else None
                    ),
                )
            )

            safe_validation = (
                validate_ai_signal(
                    signal=safe_signal,

                    market_packet=(
                        ai_market_packet
                        if isinstance(
                            ai_market_packet,
                            dict,
                        )
                        else {}
                    ),

                    historical_analog_result=(
                        historical_analog_result
                        if isinstance(
                            historical_analog_result,
                            dict,
                        )
                        else None
                    ),
                )
            )

            return (
                safe_validation.get(
                    "final_signal"
                )
                or safe_signal,
                safe_validation,
            )

        # --------------------------------------------------
        # Skip Gemini only for a real deterministic hard gate.
        # --------------------------------------------------
        if ai_first_skip:
            print(
                f"[AI_FIRST] Gemini skipped: "
                f"{ai_first_skip_reason}"
            )

            (
                final_ai_signal,
                final_ai_validation,
            ) = _build_validated_safe_no_trade(
                ai_first_skip_reason
            )

            final_ai_provider = {
                "model": None,
                "key_source": None,
                "usage": {},
            }

        else:
            # ==================================================
            # PASS 1 — INDEPENDENT GEMINI SIGNAL DRAFT
            # ==================================================
            try:
                ai_first_draft_result = (
                    generate_ai_signal_draft(
                        ai_market_packet,
                        model_name=(
                            config.MODEL_SMART
                        ),

                        deadline_monotonic=(
                            analysis_deadline_monotonic
                        ),
                    )
                )

            except Exception as e:
                ai_first_draft_result = {
                    "ok": False,
                    "signal": None,
                    "provider": {},
                    "error": (
                        "AI_FIRST_DRAFT_EXCEPTION: "
                        f"{str(e)}"
                    ),
                }

            if not ai_first_draft_result.get(
                "ok"
            ):
                draft_error = str(
                    ai_first_draft_result.get(
                        "error"
                    )
                    or (
                        "Gemini did not return a valid "
                        "structured signal draft."
                    )
                )

                print(
                    f"[AI_FIRST] Draft failed: "
                    f"{draft_error}"
                )

                (
                    final_ai_signal,
                    final_ai_validation,
                ) = _build_validated_safe_no_trade(
                    draft_error
                )

                final_ai_provider = (
                    ai_first_draft_result.get(
                        "provider"
                    )
                    or {}
                )

            else:
                draft_signal = (
                    ai_first_draft_result.get(
                        "signal"
                    )
                    or {}
                )

                final_ai_provider = (
                    ai_first_draft_result.get(
                        "provider"
                    )
                    or {}
                )

                # ==============================================
                # DETERMINISTIC VALIDATION — PASS 1
                # ==============================================
                try:
                    ai_first_validation = (
                        validate_ai_signal(
                            signal=draft_signal,

                            market_packet=(
                                ai_market_packet
                            ),

                            historical_analog_result=(
                                historical_analog_result
                                if isinstance(
                                    historical_analog_result,
                                    dict,
                                )
                                else None
                            ),
                        )
                    )

                except Exception as e:
                    ai_first_validation = {
                        "valid": False,

                        "needs_correction": True,

                        "trade_publishable": False,

                        "watch_publishable": False,

                        "final_signal": (
                            draft_signal
                        ),

                        "errors": [
                            (
                                "Deterministic validator "
                                "raised an exception: "
                                f"{str(e)}"
                            )
                        ],

                        "warnings": [],
                    }

                if ai_first_validation.get(
                    "valid"
                ):
                    final_ai_signal = (
                        ai_first_validation.get(
                            "final_signal"
                        )
                        or draft_signal
                    )

                    final_ai_validation = (
                        ai_first_validation
                    )

                else:
                    # ==========================================
                    # PASS 2 — ONE GEMINI CORRECTION ATTEMPT
                    # ==========================================
                    ai_correction_used = True

                    first_errors = (
                        ai_first_validation.get(
                            "errors"
                        )
                        or []
                    )

                    first_warnings = (
                        ai_first_validation.get(
                            "warnings"
                        )
                        or []
                    )

                    print(
                        f"[AI_FIRST] Draft validation failed | "
                        f"errors={len(first_errors)} | "
                        "requesting one correction pass"
                    )

                    try:
                        if (
                            time.monotonic()
                            + AI_CORRECTION_MIN_REMAINING_SECONDS
                            >= analysis_deadline_monotonic
                        ):
                            ai_first_correction_result = {
                                "ok": False,
                                "signal": None,
                                "provider": {},
                                "error": (
                                    "AI_FIRST_CORRECTION_SKIPPED_"
                                    "DEADLINE: Not enough request "
                                    "time remains for a safe "
                                    "Gemini correction pass."
                                ),
                            }

                        else:
                            ai_first_correction_result = (
                                regenerate_ai_signal(
                                    market_packet=(
                                        ai_market_packet
                                    ),

                                    previous_signal=(
                                        draft_signal
                                    ),

                                    validation_errors=(
                                        first_errors
                                    ),

                                    validation_warnings=(
                                        first_warnings
                                    ),

                                    model_name=(
                                        config.MODEL_SMART
                                    ),

                                    deadline_monotonic=(
                                        analysis_deadline_monotonic
                                    ),
                                )
                            )

                    except Exception as e:
                        ai_first_correction_result = {
                            "ok": False,

                            "signal": None,

                            "provider": {},

                            "error": (
                                "AI_FIRST_CORRECTION_EXCEPTION: "
                                f"{str(e)}"
                            ),
                        }

                    if ai_first_correction_result.get(
                        "ok"
                    ):
                        corrected_signal = (
                            ai_first_correction_result.get(
                                "signal"
                            )
                            or {}
                        )

                        correction_provider = (
                            ai_first_correction_result.get(
                                "provider"
                            )
                            or {}
                        )

                        if correction_provider:
                            final_ai_provider = (
                                correction_provider
                            )

                        # ======================================
                        # DETERMINISTIC VALIDATION — PASS 2
                        # ======================================
                        try:
                            ai_first_correction_validation = (
                                validate_ai_signal(
                                    signal=(
                                        corrected_signal
                                    ),

                                    market_packet=(
                                        ai_market_packet
                                    ),

                                    historical_analog_result=(
                                        historical_analog_result
                                        if isinstance(
                                            historical_analog_result,
                                            dict,
                                        )
                                        else None
                                    ),
                                )
                            )

                        except Exception as e:
                            ai_first_correction_validation = {
                                "valid": False,

                                "needs_correction": True,

                                "trade_publishable": False,

                                "watch_publishable": False,

                                "final_signal": (
                                    corrected_signal
                                ),

                                "errors": [
                                    (
                                        "Correction validation "
                                        "raised an exception: "
                                        f"{str(e)}"
                                    )
                                ],

                                "warnings": [],
                            }

                        if (
                            ai_first_correction_validation
                            .get("valid")
                        ):
                            final_ai_signal = (
                                ai_first_correction_validation
                                .get(
                                    "final_signal"
                                )
                                or corrected_signal
                            )

                            final_ai_validation = (
                                ai_first_correction_validation
                            )

                        else:
                            correction_errors = (
                                ai_first_correction_validation
                                .get(
                                    "errors"
                                )
                                or []
                            )

                            fallback_reason = (
                                "Corrected AI signal still "
                                "failed deterministic validation."
                            )

                            if correction_errors:
                                fallback_reason += (
                                    " "
                                    + " | ".join(
                                        str(item)
                                        for item
                                        in correction_errors[:6]
                                    )
                                )

                            (
                                final_ai_signal,
                                final_ai_validation,
                            ) = (
                                _build_validated_safe_no_trade(
                                    fallback_reason,

                                    previous_signal=(
                                        corrected_signal
                                    ),
                                )
                            )

                    else:
                        correction_error = str(
                            ai_first_correction_result.get(
                                "error"
                            )
                            or (
                                "Gemini correction request "
                                "did not return a valid result."
                            )
                        )

                        fallback_reason = (
                            "Initial AI signal failed "
                            "deterministic validation, and the "
                            "single correction attempt failed. "
                            f"{correction_error}"
                        )

                        (
                            final_ai_signal,
                            final_ai_validation,
                        ) = _build_validated_safe_no_trade(
                            fallback_reason,

                            previous_signal=(
                                draft_signal
                            ),
                        )

        # --------------------------------------------------
        # Final absolute fallback.
        # --------------------------------------------------
        if not isinstance(
            final_ai_signal,
            dict,
        ):
            (
                final_ai_signal,
                final_ai_validation,
            ) = _build_validated_safe_no_trade(
                (
                    "AI-first signal flow produced "
                    "no usable final result."
                )
            )

        # --------------------------------------------------
        # Balanced liquidity execution guard.
        #
        # Gemini draft/correction original validator එක
        # pass වුණාට පස්සේ පමණක් මෙය run වෙනවා.
        # --------------------------------------------------
        if (
            isinstance(
                final_ai_signal,
                dict,
            )
            and final_ai_validation.get(
                "valid"
            )
        ):
            original_validated_signal = (
                deepcopy(
                    final_ai_signal
                )
            )

            (
                guarded_final_signal,
                final_liquidity_execution,
            ) = (
                apply_balanced_liquidity_execution_guard(
                    signal=(
                        deepcopy(original_validated_signal)
                    ),
                    interval=interval,
                    df_analysis=(
                        df_micro_analysis
                    ),
                    df_lower=(
                        df_lower_exec_analysis
                    ),
                    backend_trend_regime=(
                        backend_trend_regime
                    ),
                    current_price=(
                        latest_close_price(
                            df_micro_live
                        )
                    ),
                )
            )

            if guarded_final_signal == original_validated_signal:
                # Advisory output is unchanged. Preserve the successful
                # validation and original requested/effective TP2 metadata.
                final_ai_signal = original_validated_signal
            else:
                (
                    final_ai_signal,
                    final_ai_validation,
                ) = _build_validated_safe_no_trade(
                    (
                        "Advisory liquidity layer unexpectedly changed "
                        "the validated signal."
                    ),
                    previous_signal=original_validated_signal,
                )


        # ==================================================
        # CONVERT VALIDATED AI OUTPUT TO CURRENT APP SCHEMA
        # ==================================================
        final_scenario = (
            build_ai_first_scenario(
                signal=final_ai_signal,

                symbol=symbol,

                interval=interval,

                market_packet=(
                    ai_market_packet
                ),

                validation_result=(
                    final_ai_validation
                ),

                provider=(
                    final_ai_provider
                ),

                correction_used=(
                    ai_correction_used
                ),

                backend_trend_regime=(
                    backend_trend_regime
                ),
            )
        )

        final_scenario[
            "liquidity_execution"
        ] = deepcopy(
            final_liquidity_execution
        )

        final_decision = str(
            final_ai_signal.get(
                "decision"
            )
            or "NO_TRADE"
        ).upper().strip()

        final_side = str(
            final_ai_signal.get(
                "side"
            )
            or "NONE"
        ).upper().strip()

        final_status_text = str(
            final_ai_signal.get(
                "status_text"
            )
            or final_scenario.get(
                "status_text"
            )
            or ""
        ).strip()

        final_advice = str(
            final_ai_signal.get(
                "advice"
            )
            or final_scenario.get(
                "generated_advice"
            )
            or final_scenario.get(
                "summary"
            )
            or ""
        ).strip()

        # --------------------------------------------------
        # Keep cross-timeframe warning as structured context.
        #
        # Do not append it to the two-sentence AI advice.
        # --------------------------------------------------
        try:
            if final_side in [
                "LONG",
                "SHORT",
            ]:
                xtf_warning = (
                    build_cross_timeframe_risk_warning(
                        symbol=symbol,

                        current_tf=interval,

                        current_setup=(
                            final_scenario
                        ),
                    )
                )

                if xtf_warning:
                    final_scenario[
                        "cross_timeframe_warning"
                    ] = xtf_warning

        except Exception as e:
            print(
                f"[AI_FIRST] Cross-TF warning "
                f"skipped/error: {e}"
            )

        if not final_status_text:
            if final_decision in [
                "LONG",
                "SHORT",
            ]:
                if str(
                    final_scenario.get(
                        "order_type"
                    )
                    or ""
                ).upper() == "MARKET":
                    final_status_text = (
                        f"{final_decision} | "
                        "Entry Confirmed"
                    )

                else:
                    final_status_text = (
                        "LIMIT ORDER | Fresh Setup"
                    )

            elif final_decision == "WAIT":
                final_status_text = (
                    "WAIT | Confirmation Required"
                )

            else:
                final_status_text = (
                    "NO TRADE | Weak Setup"
                )

        if not final_advice:
            if final_decision == "NO_TRADE":
                final_advice = (
                    "ACTION: STAY OUT. Current evidence "
                    "does not support a validated trade."
                )

            else:
                final_advice = (
                    "ACTION: WAIT. A valid immediate "
                    "entry instruction is unavailable."
                )

        final_scenario[
            "status_text"
        ] = final_status_text

        final_scenario[
            "generated_advice"
        ] = final_advice

        final_scenario[
            "summary"
        ] = final_advice

        # Keep a separate validated scenario before
        # advisory decorators mutate ai_data.
        authoritative_scenario_snapshot = deepcopy(
            final_scenario
        )

        status_message = final_status_text


        # ==================================================
        # SAVE FINAL RESULT
        # ==================================================

        # --------------------------------------------------
        # Valid LONG / SHORT:
        # save global lock first, then cancel watch and
        # open trade audit only after successful persistence.
        # --------------------------------------------------
        if (
            final_ai_validation.get(
                "trade_publishable"
            )
            and final_decision in [
                "LONG",
                "SHORT",
            ]
        ):
            lock_saved, lock_message = (
                save_firestore_lock(
                    symbol,
                    interval,
                    final_scenario,
                )
            )

            if not lock_saved:
                print(
                    "[FIREBASE_LOCK] Validated trade "
                    "was not persisted: "
                    f"{lock_message}"
                )

                # Fail closed: never expose an active LONG/SHORT
                # when its global lock could not be persisted.
                persistence_failure_reason = (
                    "Validated trade was not published because "
                    "secure trade tracking is temporarily unavailable."
                )

                (
                    final_ai_signal,
                    final_ai_validation,
                ) = _build_validated_safe_no_trade(
                    persistence_failure_reason,
                    previous_signal=final_ai_signal,
                )

                final_scenario = build_ai_first_scenario(
                    signal=final_ai_signal,
                    symbol=symbol,
                    interval=interval,
                    market_packet=ai_market_packet,
                    validation_result=final_ai_validation,
                    provider=final_ai_provider,
                    correction_used=ai_correction_used,
                    backend_trend_regime=(
                        backend_trend_regime
                    ),
                )

                final_scenario[
                    "liquidity_execution"
                ] = deepcopy(
                    final_liquidity_execution
                )

                final_decision = "NO_TRADE"
                final_side = "NONE"
                final_status_text = (
                    "NO TRADE | Persistence Unavailable"
                )
                final_advice = (
                    "ACTION: STAY OUT. The setup was not "
                    "published because trade tracking is "
                    "temporarily unavailable."
                )

                final_scenario["status_text"] = (
                    final_status_text
                )
                final_scenario["generated_advice"] = (
                    final_advice
                )
                final_scenario["summary"] = (
                    final_advice
                )

                authoritative_scenario_snapshot = deepcopy(
                    final_scenario
                )
                status_message = final_status_text

            else:

                if (
                    FIREBASE_AVAILABLE
                    and db is not None
                ):
                    try:
                        ok, audit_msg = (
                            audit_open_trade(
                                db=db,
                                symbol=symbol,
                                timeframe=interval,
                                scenario=final_scenario,
                                market_type=market_type,
                            )
                        )

                        if not ok:
                            print(
                                "[TRADE_AUDIT] open "
                                f"skipped/error: {audit_msg}"
                            )

                    except Exception as e:
                        print(
                            "[TRADE_AUDIT] open "
                            f"skipped/error: {e}"
                        )

        # --------------------------------------------------
        # WAIT / NO_TRADE:
        # no global trade lock and no trade audit.
        # --------------------------------------------------
        # ==================================================
        # RESPONSE OBJECT USED BY EXISTING FRONTEND
        # ==================================================
        ai_data = {
            "trade_status_message": (
                final_status_text
            ),

            "action_advice": (
                final_advice
            ),

            "macro_analysis": {
                "trend": (
                    (
                        final_ai_signal.get(
                            "timeframe_bias"
                        )
                        or {}
                    ).get(
                        "macro",
                        "UNCLEAR",
                    )
                ),

                "detailed_breakdown": (
                    final_ai_signal.get(
                        "decision_reason"
                    )
                    or ""
                ),

                "key_levels": (
                    "See validated Entry, SL, "
                    "TP1 and TP2 fields."
                ),
            },

            "micro_analysis": {
                "wave_count_status": (
                    final_ai_signal.get(
                        "setup_type"
                    )
                    or "NO_TRADE"
                ),

                "wave_personality": (
                    final_ai_signal.get(
                        "expected_path"
                    )
                    or "UNCLEAR"
                ),
            },

            "trade_scenarios": [
                final_scenario
            ],

            "selected_scenario": (
                final_scenario
            ),

            # Deterministic backend market evidence remains visible,
            # but no backend-generated trade plan is returned.
            "backend_trend_regime": (
                backend_trend_regime
            ),

            "context_bundle": (
                context_bundle
            ),

            "ai_first": {
                "enabled": True,

                "decision": (
                    final_decision
                ),

                "side": (
                    final_side
                ),

                "correction_used": bool(
                    ai_correction_used
                ),

                "model": (
                    final_ai_provider.get(
                        "model"
                    )
                ),

                "key_source": (
                    final_ai_provider.get(
                        "key_source"
                    )
                ),

                "usage": (
                    final_ai_provider.get(
                        "usage"
                    )
                    or {}
                ),

                "packet_validation": (
                    ai_market_packet_validation
                ),

                "draft_validation": {
                    "valid": bool(
                        ai_first_validation.get(
                            "valid"
                        )
                    ),

                    "errors": (
                        ai_first_validation.get(
                            "errors"
                        )
                        or []
                    ),

                    "warnings": (
                        ai_first_validation.get(
                            "warnings"
                        )
                        or []
                    ),
                },

                "correction_validation": {
                    "valid": bool(
                        ai_first_correction_validation
                        .get(
                            "valid"
                        )
                    ),

                    "errors": (
                        ai_first_correction_validation
                        .get(
                            "errors"
                        )
                        or []
                    ),

                    "warnings": (
                        ai_first_correction_validation
                        .get(
                            "warnings"
                        )
                        or []
                    ),
                },

                "final_validation": {
                    "valid": bool(
                        final_ai_validation.get(
                            "valid"
                        )
                    ),

                    "trade_publishable": bool(
                        final_ai_validation.get(
                            "trade_publishable"
                        )
                    ),

                    "watch_publishable": bool(
                        final_ai_validation.get(
                            "watch_publishable"
                        )
                    ),

                    "errors": (
                        final_ai_validation.get(
                            "errors"
                        )
                        or []
                    ),

                    "warnings": (
                        final_ai_validation.get(
                            "warnings"
                        )
                        or []
                    ),

                    "historical_validation": (
                        final_ai_validation.get(
                            "historical_validation"
                        )
                        or {}
                    ),

                    "validator_version": (
                        final_ai_validation.get(
                            "validator_version"
                        )
                    ),
                },
            },
        }

        # ------------------------------------------------
        # 🔴 අලුත් කොටස මෙතනට Paste කරන්න
        # ------------------------------------------------
        if previous_trade_status:
            ai_data["previous_trade_status"] = previous_trade_status
        if previous_trade_advice:
            ai_data["previous_trade_advice"] = previous_trade_advice
        if previous_trade_scenario:
            ai_data["previous_trade_scenario"] = previous_trade_scenario

        # ------------------------------------------------
        # Reliable Elliott YES/NO flag for frontend / signal.py
        # IMPORTANT:
        # Use final selected scenario only.
        # Do NOT trust AI advice text, because AI may see rejected Elliott candidates.
        # ------------------------------------------------
        try:
            final_scenarios = ai_data.get("trade_scenarios", []) or []
            final_setup = final_scenarios[0] if final_scenarios else {}

            elliott_yes = bool(final_setup.get("elliott_confirmed"))
            wave_template = final_setup.get("wave_template") or ""
            wave_target = final_setup.get("wave_target") or ""
            wave_confidence = final_setup.get("wave_confidence") or ""

            ai_data["elliott_wave"] = "YES" if elliott_yes else "NO"
            ai_data["elliott_wave_confirmed"] = elliott_yes
            ai_data["elliott_wave_template"] = wave_template
            ai_data["elliott_wave_target"] = wave_target
            ai_data["elliott_wave_confidence"] = wave_confidence

            if elliott_yes:
                mtf_alignment = final_setup.get("mtf_elliott_alignment", "UNKNOWN")
                mtf_score = final_setup.get("mtf_confirmation_score", 0)

                ai_data["mtf_elliott_alignment"] = mtf_alignment
                ai_data["mtf_confirmation_score"] = mtf_score
                ai_data["mtf_confirmation_reason"] = final_setup.get("mtf_confirmation_reason", "")

                ai_data["elliott_wave_label"] = (
                    f"Elliott Wave: YES | {wave_template} | Target: {wave_target} | "
                    f"Confidence: {wave_confidence} | MTF: {mtf_alignment} ({mtf_score})"
                )
            else:
                ai_data["elliott_wave_label"] = "Elliott Wave: NO"

                # If the final validated AI setup is NOT Elliott,
                # do not allow AI advice to claim Elliott Wave YES.
                try:
                    raw_advice = str(ai_data.get("action_advice") or "").strip()

                    if raw_advice.lower().startswith("elliott wave: yes"):
                        raw_advice = re.sub(
                            r"^\s*elliott\s+wave\s*:\s*yes\.?",
                            "Elliott Wave: NO.",
                            raw_advice,
                            count=1,
                            flags=re.IGNORECASE,
                        )

                    raw_advice = re.sub(
                        r"\s*Wave\s*3\s*Status\s*:\s*(ANTICIPATION|CONFIRMED)\.?",
                        "",
                        raw_advice,
                        flags=re.IGNORECASE,
                    )

                    raw_advice = raw_advice.replace(
                        "potential Wave 3 surge",
                        "potential breakout continuation"
                    )
                    raw_advice = raw_advice.replace(
                        "Wave 3 surge",
                        "breakout continuation"
                    )

                    ai_data["action_advice"] = raw_advice.strip()

                except Exception:
                    pass

        except Exception:
            ai_data["elliott_wave"] = "NO"
            ai_data["elliott_wave_confirmed"] = False
            ai_data["elliott_wave_label"] = "Elliott Wave: NO"
        # ------------------------------------------------

        # ------------------------------------------------
        # Shadow/advice-only entry confirmation.
        # This is response decoration only. It does not affect status,
        # setup selection, Firestore lock writes, or entry/SL/TP levels.
        # ------------------------------------------------
        try:
            ai_data = add_entry_confirmation_to_ai_data(
                ai_data,
                interval,
                df_lower_exec_analysis,
                current_price=latest_close_price(df_micro_live),
            )
        except Exception as e:
            print(f"[ENTRY_CONFIRMATION] final attach skipped/error: {e}")

        try:
            ai_data = add_direction_confidence_to_ai_data(
                ai_data=ai_data,
                interval=interval,
                backend_trend_regime=locals().get("backend_trend_regime") or ai_data.get("backend_trend_regime") or {},
                momentum_context=locals().get("momentum_breakout_context")
                or locals().get("momentum_context")
                or "",
                rsi_divergence_context=locals().get("rsi_divergence_context")
                or locals().get("rsi_divergence")
                or "",
            )
        except Exception as e:
            print(f"[DIRECTION_CONFIDENCE] fresh response attach skipped/error: {e}")

        try:
            ai_data = add_tp_path_quality_to_ai_data(
                ai_data=ai_data,
                interval=interval,
                market_context_text=locals().get("market_context_text")
                or locals().get("market_context")
                or ai_data.get("market_context")
                or "",
            )
        except Exception as e:
            print(f"[TP_PATH_QUALITY] fresh response attach skipped/error: {e}")

        try:
            ai_data = add_imbalance_to_ai_data(
                ai_data=ai_data,
                interval=interval,
                df_analysis=df_micro_analysis,
                df_lower=df_lower_exec_analysis if locals().get("lower_exec_tf") else None,
            )
        except Exception as e:
            print(f"[IMBALANCE_RISK] fresh response attach skipped/error: {e}")

        try:
            ai_data = add_liquidity_to_ai_data(
                ai_data=ai_data,
                interval=interval,
                df_analysis=df_micro_analysis,
                df_lower=df_lower_exec_analysis if locals().get("lower_exec_tf") else None,
            )
        except Exception as e:
            print(f"[LIQUIDITY_RISK] fresh response attach skipped/error: {e}")

        try:
            # --------------------------------------------------
            # Crash regime / momentum pressure shadow warning.
            # ZERO extra API version:
            # - selected TF uses df_micro_analysis
            # - fast TF uses already fetched df_lower_exec_analysis only
            # - no new get_data() calls here
            # --------------------------------------------------
            crash_fast_tf = get_crash_fast_timeframe_from_existing_lower(interval)

            df_crash_fast_analysis = pd.DataFrame()
            try:
                if (
                    crash_fast_tf
                    and locals().get("df_lower_exec_analysis") is not None
                    and not locals().get("df_lower_exec_analysis").empty
                ):
                    df_crash_fast_analysis = locals().get("df_lower_exec_analysis")
            except Exception:
                df_crash_fast_analysis = pd.DataFrame()

            ai_data = add_crash_regime_to_ai_data(
                ai_data=ai_data,
                interval=interval,
                df_selected=df_micro_analysis,
                df_fast=df_crash_fast_analysis if not df_crash_fast_analysis.empty else None,
                fast_timeframe=crash_fast_tf or "",
            )

        except Exception as e:
            print(f"[CRASH_REGIME] fresh response attach skipped/error: {e}")

        # AI-FIRST:
        # Crash continuation was already supplied to Gemini as quantitative
        # market evidence. Do not run the legacy post-signal setup fallback,
        # because it could replace the backend-validated AI scenario.

        try:
            ai_data = add_smc_context_to_ai_data(
                ai_data=ai_data,
                interval=interval,
                df_selected=df_micro_analysis,
                symbol=symbol,
            )
        except Exception as e:
            print(f"[SMC_CONTEXT] fresh response attach skipped/error: {e}")

            # AI-FIRST:
            # Range-sweep evidence was already included in the AI market packet.
            # Do not allow the legacy range-sweep fallback to generate or replace
            # the final validated AI scenario after deterministic validation.

        try:
            audit_scenario = (
                (
                    ai_data.get("trade_scenarios")
                    or [{}]
                )[0]
                or {}
            )

            audit_backend_status = str(
                audit_scenario.get("status")
                or ai_data.get("trade_status_message")
                or "NO_TRADE"
            ).upper().strip()

            ai_data = add_decision_audit_to_ai_data(
                ai_data=ai_data,
                backend_status=(
                    audit_backend_status
                ),
            )

        except Exception as e:
            print(
                "[DECISION_AUDIT] "
                f"fresh response attach skipped/error: {e}"
            )

        # ==================================================
        # AI-FIRST FINAL AUTHORITY GUARD
        #
        # Legacy shadow decorators may add useful advisory
        # information, but they must never change:
        # - decision / side
        # - Entry / SL / TP1 / TP2
        # - order type / execution state
        # - status text
        # - Gemini frontend advice
        # ==================================================
        try:
            decorated_scenario = (
                (
                    ai_data.get("trade_scenarios")
                    or [{}]
                )[0]
                or {}
            )

            authoritative_scenario = (
                build_ai_first_scenario(
                    signal=final_ai_signal,

                    symbol=symbol,

                    interval=interval,

                    market_packet=(
                        ai_market_packet
                    ),

                    validation_result=(
                        final_ai_validation
                    ),

                    provider=(
                        final_ai_provider
                    ),

                    correction_used=(
                        ai_correction_used
                    ),

                    backend_trend_regime=(
                        backend_trend_regime
                    ),
                )
            )

            authoritative_scenario[
                "liquidity_execution"
            ] = deepcopy(
                final_liquidity_execution
            )

            # Preserve only advisory/decorative results that were
            # calculated after the final AI signal was validated.
            advisory_fields = [
                "entry_confirmation",
                "direction_confidence",
                "tp_path_quality",
                "imbalance_risk",
                "liquidity_risk",
                "crash_regime",
                "smc_context",
                "decision_audit",
                "cross_timeframe_warning",
            ]

            for field_name in advisory_fields:
                if field_name in decorated_scenario:
                    authoritative_scenario[
                        field_name
                    ] = decorated_scenario[
                        field_name
                    ]

            authoritative_scenario[
                "status_text"
            ] = final_status_text

            authoritative_scenario[
                "generated_advice"
            ] = final_advice

            authoritative_scenario[
                "summary"
            ] = final_advice

            ai_data["trade_scenarios"] = [
                authoritative_scenario
            ]

            ai_data["selected_scenario"] = (
                authoritative_scenario
            )

            ai_data[
                "trade_status_message"
            ] = final_status_text

            ai_data[
                "action_advice"
            ] = final_advice

        except Exception as e:
            print(
                "[AI_FIRST_AUTHORITY] "
                f"final guard skipped/error: {e}"
            )

            # Restore the validated pre-decoration scenario.
            # Never return a partially mutated trade plan.
            fallback_scenario = deepcopy(
                authoritative_scenario_snapshot
            )

            fallback_scenario[
                "status_text"
            ] = final_status_text

            fallback_scenario[
                "generated_advice"
            ] = final_advice

            fallback_scenario[
                "summary"
            ] = final_advice

            ai_data["trade_scenarios"] = [
                fallback_scenario
            ]

            ai_data["selected_scenario"] = (
                fallback_scenario
            )

            ai_data[
                "trade_status_message"
            ] = final_status_text

            ai_data[
                "action_advice"
            ] = final_advice

        # Final Payload Return
        ohlc_data = []
        chart_df = df_macro_analysis if not df_macro_analysis.empty else df_macro_live
        if not isinstance(chart_df.index, pd.DatetimeIndex):
            chart_df.index = pd.to_datetime(chart_df.index)

        for date, row in chart_df.iterrows():
            ohlc_data.append({
                "time": int(date.timestamp()),
                "open": float(row["Open"]), "high": float(row["High"]),
                "low": float(row["Low"]), "close": float(row["Close"])
            })

        return {
            "meta": {"symbol": symbol, "timeframe": f"{interval} ({tfs['confirm']['tf']} Confirm)"},
            "waves": {
                "macro": macro_waves,
                "swing": swing_waves,
                "confirmation_1h": confirm_1h_waves,
                "micro": micro_waves
            },

            "mtf_elliott_context": {
                "execution_tf": elliott_mtf_context.get("execution_tf"),
                "confirm_tf": elliott_mtf_context.get("confirm_tf"),
                "swing_tf": elliott_mtf_context.get("swing_tf"),
                "macro_tf": elliott_mtf_context.get("macro_tf"),
                "confirm_wave_count": len(confirm_1h_waves or []),
                "swing_wave_count": len(swing_waves or []),
                "macro_wave_count": len(macro_waves or []),
            },
            "elliott_phase_context": {
                "phase_alignment": elliott_phase_context.get("elliott_phase_alignment"),
                "phase_score": elliott_phase_context.get("elliott_phase_score"),
                "phase_permission": elliott_phase_context.get("elliott_phase_permission"),
                "execution_phase": (elliott_phase_context.get("execution") or {}).get("elliott_phase"),
                "confirm_phase": (elliott_phase_context.get("confirm") or {}).get("elliott_phase"),
                "swing_phase": (elliott_phase_context.get("swing") or {}).get("elliott_phase"),
                "macro_phase": (elliott_phase_context.get("macro") or {}).get("elliott_phase"),
                "phase_reason": elliott_phase_context.get("elliott_phase_reason"),
            },
            "momentum_breakout_context": {
                "enabled": momentum_breakout_context.get("enabled"),
                "momentum_breakout": momentum_breakout_context.get("momentum_breakout"),
                "timeframe": momentum_breakout_context.get("timeframe"),
                "side": momentum_breakout_context.get("side"),
                "strength": momentum_breakout_context.get("strength"),
                "score": momentum_breakout_context.get("score"),
                "breakout_level": momentum_breakout_context.get("breakout_level"),
                "current_price": momentum_breakout_context.get("current_price"),
                "is_extended": momentum_breakout_context.get("is_extended"),
                "needs_retest": momentum_breakout_context.get("needs_retest"),
                "reason": momentum_breakout_context.get("reason"),
            },
            "ohlc_data": ohlc_data,

            # Shadow-mode historical forecast.
            # Frontend can ignore this field for now.
            # Later this same compact payload will be passed to Gemini.
            "historical_analog_shadow": historical_analog_payload,

            "ai_prediction": ai_data
        }


# --- CLI ENTRY POINT (For Testing) ---
if __name__ == "__main__":
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    arg2 = sys.argv[2] if len(sys.argv) > 2 else "15m"
    action = sys.argv[3] if len(sys.argv) > 3 else "analyze"
    market_type = sys.argv[4] if len(sys.argv) > 4 else "crypto"

    result = run_analysis_backend(arg1, arg2, action, market_type)
    print(json.dumps(result, indent=4))
