import sys
import json
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# --- IMPORT CONFIG ---
from . import config

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
from .ai_engine import generate_mtf_analysis, analyze_scanner_candidates, generate_status_update
from .scanner import scan_market
from .models.market_context import build_market_context_text
from .candle_context import build_recent_candle_rejection_context
from .models.trend_direction import build_dominant_trend_context

from .trade_audit import audit_open_trade, audit_close_trade

# --- FIREBASE SETUP ---
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_AVAILABLE = True
except Exception as e:
    FIREBASE_AVAILABLE = False
    print(f"[WARN] Firebase setup failed: {e}")

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
                print("Loaded Gemini free keys:", len(config.API_KEY_POOL))
                for i, k in enumerate(config.API_KEY_POOL):
                    print(f"Free key {i} suffix:", k[-6:] if k else "EMPTY")
                print("Paid key suffix:", config.PAID_API_KEY[-6:] if config.PAID_API_KEY else "EMPTY")
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
            return data.get("scenario"), data.get("updated_at")
    except Exception as e:
        print(f"[WARN] Failed to get Firebase lock: {e}")
    return None, None


def get_cross_timeframe_context(symbol, current_tf):
    if not FIREBASE_AVAILABLE:
        return ""
    try:
        clean_symbol = normalize_symbol(symbol)
        docs = db.collection(COLLECTION_NAME).where("symbol", "==", clean_symbol).stream()
        summary_list = []
        for doc in docs:
            data = doc.to_dict()
            tf = data.get("timeframe")
            if tf == current_tf:
                continue
            scenario = data.get("scenario", {})
            t_type = scenario.get("trade_type", "Unknown")
            if t_type in ["WAIT", "NEUTRAL", "Unknown"]:
                continue
            entry = scenario.get("entry_zone") or scenario.get("entry", "N/A")
            targets = scenario.get("targets", [])
            tp = targets[0] if targets else scenario.get("target", "N/A")
            summary = f"{tf}:{t_type}(E:{entry},TP:{tp})"
            summary_list.append(summary)
        return " | ".join(summary_list) if summary_list else ""
    except Exception as e:
        print(f"[WARN] Error fetching cross-tf context: {e}")
        return ""


def save_firestore_lock(symbol, tf, scenario):
    if not FIREBASE_AVAILABLE:
        return
    try:
        clean_symbol = normalize_symbol(symbol)
        doc_id = f"{clean_symbol}_{tf}"
        doc_ref = db.collection(COLLECTION_NAME).document(doc_id)
        doc_ref.set({
            "scenario": scenario,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "symbol": clean_symbol,
            "timeframe": tf
        })
    except Exception as e:
        print(f"[ERROR] Failed to save Firebase lock: {e}")


def delete_firestore_lock(symbol, tf):
    if not FIREBASE_AVAILABLE:
        return
    try:
        clean_symbol = normalize_symbol(symbol)
        doc_id = f"{clean_symbol}_{tf}"
        db.collection(COLLECTION_NAME).document(doc_id).delete()
    except Exception as e:
        print(f"Error deleting lock: {e}")


def get_market_bias():
    if not FIREBASE_AVAILABLE:
        return "Analyze the market neutrally."
    try:
        doc_ref = db.collection("app_config").document("trading_bias")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("instruction", "Analyze the market neutrally.")
    except Exception:
        pass
    return "Analyze the market neutrally."


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


def split_live_and_analysis_df(df, min_rows=50):
    """
    Keep full dataframe for live/status checks.
    Use closed-candle-only dataframe for analysis.

    df_live:
      includes latest available candle/current price

    df_analysis:
      removes latest candle to avoid using incomplete candle
      for Elliott/Fib/SR/trend/AI decision
    """
    try:
        if df is None or df.empty:
            return df, df

        df_live = df.copy()

        # If not enough rows, avoid breaking analysis.
        if len(df_live) <= min_rows:
            return df_live, df_live.copy()

        df_analysis = df_live.iloc[:-1].copy()

        if df_analysis.empty:
            df_analysis = df_live.copy()

        return df_live, df_analysis

    except Exception:
        return df, df


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


def extract_context_line(market_context_text, label):
    """
    Extract value after a market context label.
    Example:
      Major Support Zone: 0.17200000 - 0.17300000 (strength 2)
    """
    try:
        if not market_context_text:
            return ""

        pattern = rf"{re.escape(label)}:\s*(.+)"
        match = re.search(pattern, market_context_text)

        if match:
            return match.group(1).strip()
    except Exception:
        pass

    return ""


def extract_context_zone(market_context_text, label):
    """
    Extract readable price/zone from market context.
    """
    try:
        line = extract_context_line(market_context_text, label)
        if not line:
            return ""

        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if not nums:
            return ""

        first = extract_price(nums[0])
        second = extract_price(nums[1]) if len(nums) > 1 and "Zone" in label else None

        if first is None:
            return ""

        if second is not None and abs(first - second) > 0:
            return f"{format_signal_price(first)} - {format_signal_price(second)}"

        return format_signal_price(first)

    except Exception:
        return ""


def generate_dynamic_management_strategy(scenario, df_live):
    """
    පවතින ට්‍රේඩ් එකක සජීවී දත්ත මත පදනම්ව කළමනාකරණ උපදෙස් ජෙනරේට් කිරීම.
    """
    try:
        trade_type = str(scenario.get("trade_type", "")).upper()
        curr_price = float(df_live.iloc[-1]["Close"])
        # Locked/running trades only: use smoothed last-3-candle volume.
        # Fresh signal generation does NOT use volume as a filter.
        curr_vol = 0.0
        avg_vol = 0.0
        try:
            if "Volume" in df_live.columns:
                volume_series = pd.to_numeric(df_live["Volume"], errors="coerce").dropna()
                recent_3 = volume_series.tail(3)
                previous = volume_series.iloc[:-3].tail(30)

                if len(recent_3) > 0:
                    curr_vol = float(recent_3.mean())

                if len(previous) > 0:
                    avg_vol = float(previous.mean())
                else:
                    avg_vol = curr_vol
        except Exception:
            curr_vol = 0.0
            avg_vol = 0.0
        
        # 1. TP1 වලට සාපේක්ෂව මිල කොතනද තියෙන්නේ කියලා බැලීම
        targets = scenario.get("targets", [])
        tp1 = extract_price(targets[0]) if targets else None
        
        strategy = {
            "current_phase": "MONITORING",
            "advice": "Maintain current stop loss and wait for target reach.",
            "new_sl_level": None
        }

        if not tp1: return strategy

        # 2. TP1 HIT වෙලා නම් දෙන Dynamic Advice
        tp1_hit = (trade_type == "LONG" and curr_price >= tp1) or (trade_type == "SHORT" and curr_price <= tp1)
        
        if tp1_hit:
            strategy["current_phase"] = "TP1 TARGET ACHIEVED"
            # Volume එක වැඩිනම් Momentum එක වැඩියි
            if curr_vol > (avg_vol * 1.2):
                strategy["advice"] = "TP1 hit with high volume. Strong momentum detected. Hold for TP2. Trail SL to entry level."
            else:
                strategy["advice"] = "TP1 hit but volume is decreasing. Suggest taking 50% partial profit now and securing the rest."
            
            # Entry එකට වඩා පොඩි Buffer එකක් එක්ක SL එක යෝජනා කිරීම (Rule 21)
            entry = extract_price(scenario.get("entry_zone") or scenario.get("entry"))
            if entry:
                buffer = entry * 0.001 # 0.1% for fees
                strategy["new_sl_level"] = format_signal_price(entry + buffer if trade_type == "LONG" else entry - buffer)

        return strategy
    except Exception:
        return None


def build_wait_watch_advice(market_context_text):
    """
    Build useful WAIT advice when AI returns WAIT/None advice.
    """
    try:
        support_zone = (
            extract_context_zone(market_context_text, "Major Support Zone")
            or extract_context_zone(market_context_text, "Nearest Support")
            or extract_context_zone(market_context_text, "Long Lookback Major Support Zone")
        )

        resistance_zone = (
            extract_context_zone(market_context_text, "Major Resistance Zone")
            or extract_context_zone(market_context_text, "Nearest Resistance")
            or extract_context_zone(market_context_text, "Long Lookback Major Resistance Zone")
        )

        if support_zone and resistance_zone:
            return (
                f"WAIT. Watch {support_zone} support for bullish rejection, "
                f"or {resistance_zone} resistance for bearish rejection."
            )

        if support_zone:
            return f"WAIT. Watch {support_zone} support for bullish rejection."

        if resistance_zone:
            return f"WAIT. Watch {resistance_zone} resistance for bearish rejection."

    except Exception:
        pass

    return "Wait for clear setup."


def is_empty_advice(value):
    try:
        text = str(value or "").strip().lower()
        return text in ["", "none", "null", "nan"]
    except Exception:
        return True

def is_generic_wait_advice(value):
    try:
        text = str(value or "").strip().lower()
        return text in [
            "wait",
            "wait.",
            "wait for clear setup",
            "wait for clear setup.",
            "wait for a clear setup",
            "wait for a clear setup."
        ]
    except Exception:
        return False


def enforce_max_rr(scenario, max_r=5.0):
    try:
        t_type = scenario.get("trade_type", "").upper()
        if "LONG" not in t_type and "SHORT" not in t_type:
            return scenario

        entry = extract_price(scenario.get("entry_zone") or scenario.get("entry"))
        sl = extract_price(scenario.get("invalidation") or scenario.get("stop_loss"))
        if entry is None or sl is None:
            return scenario

        risk = abs(entry - sl)
        if risk == 0:
            return scenario

        limit_price = entry + (risk * max_r) if "LONG" in t_type else entry - (risk * max_r)

        raw_targets = scenario.get("targets", [])
        if not raw_targets and "target" in scenario:
            raw_targets = [scenario.get("target")]

        new_targets = []
        for t in raw_targets:
            val = extract_price(t)
            if val is None:
                new_targets.append(t)
                continue

            capped_val = val
            if "LONG" in t_type:
                if val > limit_price:
                    capped_val = limit_price
            else:
                if val < limit_price:
                    capped_val = limit_price

            new_targets.append(format_signal_price(capped_val))

        scenario["targets"] = new_targets
        if new_targets:
            scenario["target"] = new_targets[-1]
        return scenario
    except Exception:
        return scenario


TERMINAL_STATUSES = {"WON", "LOST", "MISSED", "CANCELLED", "EXPIRED"}


def is_terminal_status(status):
    """
    True if the status means the lock should be removed.
    """
    s = str(status or "").upper()
    return any(key in s for key in TERMINAL_STATUSES)


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


def calculate_running_pnl_percent(scenario, current_price):
    """
    Calculate unrealized PnL % for a running LONG/SHORT trade.
    This is price movement %, not leverage-adjusted %.
    """
    try:
        trade_type = str(scenario.get("trade_type", "")).upper()

        entry_raw = scenario.get("entry_zone") or scenario.get("entry")
        entry = float(str(entry_raw).replace(",", "").split("-")[0].strip())
        current_price = float(current_price)

        if entry <= 0:
            return None

        if trade_type == "LONG":
            return ((current_price - entry) / entry) * 100

        if trade_type == "SHORT":
            return ((entry - current_price) / entry) * 100

    except Exception:
        pass

    return None


def format_running_pnl_advice(pnl_percent):
    if pnl_percent is None:
        return "Trade running. Waiting for TP/SL."

    if pnl_percent > 0:
        return f"🟢 In Profit (+{pnl_percent:.2f}%). Secure positions..."

    if pnl_percent < 0:
        return f"🔴 In Loss ({pnl_percent:.2f}%). Hold if structure valid."

    return "⚪ At entry price (0.00%). Trade is running."


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

        is_active = False
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
                            halfway_to_tp1 = entry_price + ((tp1 - entry_price) * 0.50)

                            if curr_high >= halfway_to_tp1:
                                return (
                                    "MISSED",
                                    "No chase. Price moved more than 50% toward TP1 before entry."
                                )

                        elif "SHORT" in trade_type and tp1 < entry_price:
                            halfway_to_tp1 = entry_price - ((entry_price - tp1) * 0.50)

                            if curr_low <= halfway_to_tp1:
                                return (
                                    "MISSED",
                                    "No chase. Price moved more than 50% toward TP1 before entry."
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
                        return "LOST", "Stop Loss Hit"

                    for i, tp in enumerate(tp_levels):
                        if curr_high >= tp and i > highest_tp_hit_idx:
                            highest_tp_hit_idx = i

                elif "SHORT" in trade_type:
                    if curr_high >= stop_loss:
                        return "LOST", "Stop Loss Hit"

                    for i, tp in enumerate(tp_levels):
                        if curr_low <= tp and i > highest_tp_hit_idx:
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

    # 1. Initialize Keys
    if not load_api_key():
        if not hasattr(config, "API_KEY_POOL") and not hasattr(config, "PAID_API_KEY"):
            print("[WARN] No API keys found, AI might fail.")

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
        locked_scenario, lock_time = get_firestore_lock(symbol, interval)
        if not locked_scenario:
            return {
                "ai_prediction": {
                    "trade_status_message": "NO ACTIVE TRADE",
                    "macro_analysis": {"trend": "Unknown"},
                    "micro_analysis": {"wave_personality": "Unknown"},
                    "trade_scenarios": []
                }
            }

            

        df_micro = get_data(symbol, interval=interval, period="1mo", market_type=market_type)
        if not df_micro.empty and "Date" in df_micro.columns:
            df_micro.set_index("Date", inplace=True)

        if not isinstance(df_micro.index, pd.DatetimeIndex):
            df_micro.index = pd.to_datetime(df_micro.index)

        if df_micro.index.tz is None and hasattr(lock_time, "tzinfo") and lock_time.tzinfo:
            lock_time = lock_time.replace(tzinfo=None)

        df_since_lock = df_micro[df_micro.index >= lock_time]
        if df_since_lock.empty:
            df_since_lock = df_micro.tail(1)

        status, advice = check_trade_status_historical(
            locked_scenario,
            df_since_lock,
            interval=interval
        )

        if (
            not is_terminal_status(status)
            and not str(status).upper().startswith("RUNNING")
            and not str(status).upper().startswith("TP 1 HIT")
        ):
            advice = generate_status_update(locked_scenario, df_since_lock)

        if status in ["WON", "LOST"]:
            last_price = None
            try:
                last_price = float(df_since_lock.iloc[-1]["Close"])
            except Exception:
                pass

            ok, audit_msg = audit_close_trade(
                db=db,
                symbol=symbol,
                timeframe=interval,
                result=status,
                reason=advice,
                close_price=last_price
            )

            if not ok:
                print(f"[TRADE_AUDIT] close skipped/error: {audit_msg}")

            delete_firestore_lock(symbol, interval)
            status_msg = f"TRADE CLOSED ({status})"

        elif status in ["MISSED", "CANCELLED", "EXPIRED"]:
            delete_firestore_lock(symbol, interval)
            status_msg = f"{status} | {advice}"

        else:
            status_msg = status

        return {
            "ai_prediction": {
                "trade_status_message": status_msg,
                "action_advice": advice,
                "macro_analysis": {"trend": "LOCKED (Monitoring)"},
                "micro_analysis": {"wave_personality": "Monitoring Phase"},
                "trade_scenarios": [locked_scenario]
            }
        }

    # --- ACTION: ANALYZE (DEFAULT) ---
    else:
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

        # --------------------------------------------------
        # LIVE vs CLOSED-CANDLE ANALYSIS SPLIT
        # --------------------------------------------------
        df_macro_live, df_macro_analysis = split_live_and_analysis_df(df_macro, min_rows=100)
        df_swing_live, df_swing_analysis = split_live_and_analysis_df(df_swing, min_rows=100)
        df_micro_live, df_micro_analysis = split_live_and_analysis_df(df_micro, min_rows=100)
        df_confirm_1h_live, df_confirm_1h_analysis = split_live_and_analysis_df(df_confirm_1h, min_rows=100)

        macro_waves, _ = analyze_structure_phase1_only(df_macro_analysis)
        swing_waves, _ = analyze_structure_phase1_only(df_swing_analysis)
        confirm_1h_waves, _ = analyze_structure_phase1_only(df_confirm_1h_analysis)
        micro_waves, raw_micro_tail = analyze_structure_phase1_only(df_micro_analysis)

        market_context_text = build_market_context_text(df_micro_analysis, micro_waves, interval=interval)
        
        recent_candle_context_text = build_recent_candle_rejection_context(
            df_micro_analysis, market_context_text=market_context_text, use_last_closed=True
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
            market_context_text += f"\n\n--- DOMINANT TREND CONTEXT ---\nDominant Trend Error: {str(e)}"

        print("\n===== MARKET CONTEXT SENT TO AI =====")
        print(market_context_text[:12000])
        print("===== END MARKET CONTEXT =====\n")

        locked_scenario, lock_time = get_firestore_lock(symbol, interval)
        status_message = "LIMIT ORDER | Fresh Setup"
        generated_advice = None
        terminal_status_message = None

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
            status, advice = check_trade_status_historical(locked_scenario, df_since_lock, interval=interval)

            # --- NEW: ACTIVE MANAGEMENT STRATEGY INJECTION ---
            management_strategy = generate_dynamic_management_strategy(locked_scenario, df_micro_live)
            
            # --- PNL DISPLAY FIX FOR LOCKED TRADES ---
            if status == "RUNNING":
                status_message = f"RUNNING | {advice}"
            else:
                status_message = status
                
            generated_advice = advice

            if status in ["WON", "LOST"]:
                last_price = None
                try:
                    last_price = float(df_since_lock.iloc[-1]["Close"])
                except Exception:
                    pass

                ok, audit_msg = audit_close_trade(
                    db=db,
                    symbol=symbol,
                    timeframe=interval,
                    result=status,
                    reason=advice,
                    close_price=last_price
                )

                if not ok:
                    print(f"[TRADE_AUDIT] close skipped/error: {audit_msg}")

                previous_trade_status = f"TRADE CLOSED ({status})"
                previous_trade_advice = advice
                previous_trade_scenario = dict(locked_scenario or {})
                previous_trade_scenario["summary"] = (
                    f"Previous locked setup closed as {status}. {advice}"
                )

                delete_firestore_lock(symbol, interval)
                locked_scenario = None
                status_message = "LIMIT ORDER | Fresh Setup"
                terminal_status_message = None
                generated_advice = None

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
                terminal_status_message = None
                generated_advice = None

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

                return {
                    "meta": {"symbol": symbol, "timeframe": f"{interval} ({tfs['confirm']['tf']} Confirm)"},
                    "waves": {
                        "macro": macro_waves,
                        "swing": swing_waves,
                        "confirmation_1h": confirm_1h_waves,
                        "micro": micro_waves
                    },
                    "ohlc_data": ohlc_data,
                    "market_context": market_context_text,
                    "ai_prediction": {
                        "trade_status_message": status_message,
                        "action_advice": advice,
                        "management_strategy": management_strategy,
                        "macro_analysis": {"trend": "LOCKED (Monitoring)"},
                        "micro_analysis": {"wave_personality": "Monitoring Phase"},
                        "trade_scenarios": [locked_scenario]
                    }
                }
            
        elif locked_scenario and not lock_time:
            delete_firestore_lock(symbol, interval)
            locked_scenario = None

        # =================================================================
        # LEAD AI v2: PRE-AI COST SAVER FAST EXIT 
        # (Skip AI completely if Market is Broken or Conflicted)
        # =================================================================
        skip_ai = False
        fast_exit_reason = ""
        
        if not locked_scenario:
            # 1. Structure Broken (CHoCH) check - Absolute Red Flag
            choch_state = extract_context_line(
                dominant_trend_context_text,
                "Structure Broken (CHoCH)"
            ).upper()

            if "YES" in choch_state:
                skip_ai = True
                fast_exit_reason = "Structure is Broken (CHoCH). No safe trade available."

            # NOTE:
            # We do NOT skip if Preferred Trade Side is WAIT anymore.
            # We also do NOT hard-skip fresh trades using volume.
            # Volume is used only for locked/running trade management.

        if skip_ai:
            print(f"[COST SAVER] {fast_exit_reason} Bypassing Gemini API to save tokens.")
            ai_data = {
                "trade_status_message": "WAIT | No Fresh Setup",
                "action_advice": f"WAIT. {fast_exit_reason} Wait for a clear, safe setup.",
                "macro_analysis": {"trend": "Conflicted / Exhausted", "detailed_breakdown": "Macro structure does not align safely.", "key_levels": "N/A"},
                "micro_analysis": {"wave_count_status": "Awaiting clear continuation", "wave_personality": "Corrective / Choppy"},
                "trade_scenarios": [{
                    "name": "Sniper Setup",
                    "trade_type": "WAIT",
                    "probability": "Low",
                    "summary": f"Setup blocked by Universal Gatekeeper. {fast_exit_reason}",
                    "entry_zone": "",
                    "targets": [],
                    "invalidation": ""
                }]
            }
        else:
            # --------------------------------------------------
            # CALL AI ONLY IF SETUP IS SAFE
            # --------------------------------------------------
            candles_for_ai = raw_micro_tail if len(raw_micro_tail) > 5 else df_micro_analysis.tail(20)
            market_bias = get_market_bias()
            cross_tf_context = get_cross_timeframe_context(symbol, interval)

            ai_response_str = generate_mtf_analysis(
                symbol, macro_waves, swing_waves, micro_waves,
                candles_for_ai, df_macro_analysis, interval, "Pro",
                locked_scenario, market_bias, market_type, cross_tf_context,
                market_context_text, confirm_1h_waves=confirm_1h_waves,
                df_confirm_1h=df_confirm_1h_analysis, df_swing=df_swing_analysis
            )

            try:
                ai_data = json.loads(ai_response_str)

                if "action_advice" in ai_data and generated_advice == "CHECK_AI":
                    generated_advice = ai_data["action_advice"]
                elif generated_advice and "action_advice" in ai_data and not locked_scenario:
                    generated_advice = ai_data["action_advice"]

                if locked_scenario:
                    locked_scenario["generated_advice"] = generated_advice
                    ai_data["trade_scenarios"] = [locked_scenario]
                    if generated_advice: ai_data["action_advice"] = generated_advice
                else:
                    if "trade_scenarios" in ai_data and len(ai_data["trade_scenarios"]) > 0:
                        primary_setup = ai_data["trade_scenarios"][0]
                        if "trade_type" in primary_setup and primary_setup["trade_type"] not in ["WAIT", "NEUTRAL"]:
                            primary_setup = enforce_max_rr(primary_setup, max_r=8.0)
                            
                            # --- LEAD AI v2.1: INSTANT RUNNING CHECK (WITH PNL) ---
                            try:
                                entry_price = extract_price(primary_setup.get("entry_zone"))
                                curr_price = float(df_micro_live.iloc[-1]["Close"])
                                t_type = str(primary_setup.get("trade_type", "")).upper()
                                
                                if entry_price and curr_price:
                                    # අලුත් සිග්නල් එක ආපු ගමන් දැනටමත් Entry එක Hit වෙලා නම්
                                    if (t_type == "LONG" and curr_price <= entry_price) or (t_type == "SHORT" and curr_price >= entry_price):
                                        pnl_percent = calculate_running_pnl_percent(primary_setup, curr_price)
                                        running_adv = format_running_pnl_advice(pnl_percent)
                                        
                                        status_message = f"RUNNING | {running_adv}"
                                        generated_advice = running_adv
                                        primary_setup["summary"] += f" [Status: Entry triggered. {running_adv}]"
                            except Exception as e:
                                pass

                            ai_data["trade_scenarios"][0] = primary_setup
                            ai_data["selected_scenario"] = primary_setup

                            save_firestore_lock(symbol, interval, primary_setup)
                            ok, audit_msg = audit_open_trade(db=db, symbol=symbol, timeframe=interval, scenario=primary_setup, market_type=market_type)
                            if not ok: print(f"[TRADE_AUDIT] open skipped/error: {audit_msg}")

            except Exception as e:
                ai_data = {"error": "AI Parsing Error", "raw": str(ai_response_str)}
                print(f"[ERROR] AI processing failed: {e}")

            # --- FINAL STATUS & ADVICE FALLBACK (BUG FIX FOR 'NONE') ---
            try:
                final_scenarios = ai_data.get("trade_scenarios", [])
                final_setup = final_scenarios[0] if final_scenarios else {}
                final_trade_type = str(final_setup.get("trade_type", "WAIT")).upper().strip()

                if terminal_status_message is not None:
                    ai_data["trade_status_message"] = terminal_status_message
                    ai_data["action_advice"] = generated_advice or terminal_status_message
                elif str(status_message).startswith("TRADE CLOSED"): 
                    ai_data["trade_status_message"] = status_message
                elif final_trade_type in ["WAIT", "NEUTRAL"]: 
                    ai_data["trade_status_message"] = "WAIT | No Fresh Setup"
                elif locked_scenario: 
                    ai_data["trade_status_message"] = status_message
                elif "RUNNING" in status_message:
                    ai_data["trade_status_message"] = status_message
                else: 
                    ai_data["trade_status_message"] = "LIMIT ORDER | Fresh Setup"

                if is_empty_advice(ai_data.get("action_advice")):
                    ai_data["action_advice"] = "Monitor setup carefully."
            except Exception:
                ai_data["trade_status_message"] = status_message if status_message else "LIMIT ORDER | Fresh Setup"

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
                "macro": macro_waves, "swing": swing_waves,
                "confirmation_1h": confirm_1h_waves, "micro": micro_waves
            },
            "ohlc_data": ohlc_data,
            "market_context": market_context_text,
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