# app/trade_audit.py

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from firebase_admin import firestore


TRADE_AUDIT_COLLECTION = "test_trade_history"
TRADE_AUDIT_ACTIVE_INDEX = "test_trade_active_index"


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(symbol: str) -> str:
    """
    BTC/USDT -> BTCUSDT
    BTCUSDT  -> BTCUSDT
    """
    return str(symbol).replace("/", "").replace(":", "").replace(" ", "").upper()


def make_active_key(symbol: str, timeframe: str) -> str:
    """
    One active audit trade per symbol + timeframe.
    Example: BTCUSDT_15m
    """
    return f"{normalize_symbol(symbol)}_{timeframe}"


def make_trade_id(symbol: str, timeframe: str) -> str:
    """
    Unique history document id.
    Example: BTCUSDT_15m_20260426_101530_ab12cd
    """
    clean_symbol = normalize_symbol(symbol)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:6]
    return f"{clean_symbol}_{timeframe}_{ts}_{short_id}"


def extract_price(value: Any) -> Optional[float]:
    """
    Extract first valid number from values like:
    65000
    "65000"
    "65000 - 66000"
    "TP1: 67000"
    "Invalidation: 64000"
    """
    if isinstance(value, (int, float)):
        return float(value)

    if value is None:
        return None

    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


def extract_targets(scenario: Dict[str, Any]) -> list:
    """
    Supports:
    targets: [65000, 66000]
    targets: ["TP1: 65000", "TP2: 66000"]
    target: 65000
    """
    raw_targets = scenario.get("targets", [])

    if not raw_targets and scenario.get("target"):
        raw_targets = [scenario.get("target")]

    parsed = []

    if isinstance(raw_targets, (str, int, float)):
        raw_targets = [raw_targets]

    for item in raw_targets:
        price = extract_price(item)
        if price is not None:
            parsed.append(price)

    return parsed


def detect_side(scenario: Dict[str, Any]) -> Optional[str]:
    """
    Convert trade_type to LONG / SHORT.
    """
    trade_type = str(scenario.get("trade_type", "")).upper()

    if "LONG" in trade_type or "BUY" in trade_type:
        return "LONG"

    if "SHORT" in trade_type or "SELL" in trade_type:
        return "SHORT"

    return None


def is_wait_signal(scenario: Dict[str, Any]) -> bool:
    trade_type = str(scenario.get("trade_type", "")).upper()
    return "WAIT" in trade_type or "NEUTRAL" in trade_type or "NO TRADE" in trade_type


def json_safe(value: Any) -> Any:
    """
    Firestore usually accepts normal dict/list/string/float.
    This helper avoids issues with non-serializable values.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    try:
        # numpy / pandas scalar support
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass

    return str(value)


# --------------------------------------------------
# Audit open trade
# --------------------------------------------------

def audit_open_trade(
    db,
    symbol: str,
    timeframe: str,
    scenario: Dict[str, Any],
    market_type: str = "crypto",
    source: str = "ai_signal",
) -> Tuple[bool, Optional[str]]:
    """
    Copy generated AI signal into test_trade_history as OPEN trade.

    Important:
    - Main signal flow break වෙන්න දෙන්නේ නෑ.
    - Error එකක් ආවොත් False + error return කරනවා.
    - Same symbol/timeframe වල OPEN audit trade එකක් තියෙනවා නම් duplicate create කරන්නේ නෑ.
    """

    try:
        if db is None:
            return False, "Firestore db is None"

        if not scenario or not isinstance(scenario, dict):
            return False, "Invalid scenario"

        if is_wait_signal(scenario):
            return False, "WAIT/NEUTRAL signal skipped"

        side = detect_side(scenario)

        if side not in ["LONG", "SHORT"]:
            return False, "Could not detect LONG/SHORT side"

        clean_symbol = normalize_symbol(symbol)
        active_key = make_active_key(clean_symbol, timeframe)

        entry = extract_price(
            scenario.get("entry_zone")
            or scenario.get("entry")
            or scenario.get("entry_price")
        )

        stop_loss = extract_price(
            scenario.get("invalidation")
            or scenario.get("stop_loss")
            or scenario.get("sl")
        )

        targets = extract_targets(scenario)

        if entry is None:
            return False, "Entry price missing"

        if stop_loss is None:
            return False, "Stop loss price missing"

        if not targets:
            return False, "Targets missing"

        active_ref = db.collection(TRADE_AUDIT_ACTIVE_INDEX).document(active_key)
        active_doc = active_ref.get()

        # Already tracking this symbol/timeframe
        if active_doc.exists:
            active_data = active_doc.to_dict() or {}
            existing_trade_id = active_data.get("trade_id")

            if existing_trade_id:
                trade_ref = db.collection(TRADE_AUDIT_COLLECTION).document(existing_trade_id)
                trade_doc = trade_ref.get()

                if trade_doc.exists:
                    trade_data = trade_doc.to_dict() or {}

                    if trade_data.get("status") == "OPEN":
                        # Update latest scenario snapshot without creating duplicate
                        trade_ref.set({
                            "latest_scenario": json_safe(scenario),
                            "updated_at": firestore.SERVER_TIMESTAMP,
                            "updated_at_iso": utc_now_iso(),
                        }, merge=True)

                        return True, existing_trade_id

        # Create new audit trade
        trade_id = make_trade_id(clean_symbol, timeframe)

        trade_data = {
            "trade_id": trade_id,
            "active_key": active_key,

            "symbol": clean_symbol,
            "timeframe": timeframe,
            "market_type": market_type,
            "source": source,

            "status": "OPEN",
            "result": None,

            "side": side,
            "trade_type": scenario.get("trade_type"),

            "entry": float(entry),
            "stop_loss": float(stop_loss),
            "targets": [float(t) for t in targets],

            "scenario": json_safe(scenario),
            "latest_scenario": json_safe(scenario),

            "opened_at": firestore.SERVER_TIMESTAMP,
            "opened_at_iso": utc_now_iso(),

            "closed_at": None,
            "closed_at_iso": None,

            "close_reason": None,
            "close_price": None,

            "updated_at": firestore.SERVER_TIMESTAMP,
            "updated_at_iso": utc_now_iso(),
        }

        db.collection(TRADE_AUDIT_COLLECTION).document(trade_id).set(trade_data)

        active_ref.set({
            "active_key": active_key,
            "trade_id": trade_id,
            "symbol": clean_symbol,
            "timeframe": timeframe,
            "status": "OPEN",
            "created_at": firestore.SERVER_TIMESTAMP,
            "created_at_iso": utc_now_iso(),
        })

        return True, trade_id

    except Exception as e:
        print(f"[TRADE_AUDIT] audit_open_trade error: {e}")
        return False, str(e)


# --------------------------------------------------
# Audit close trade
# --------------------------------------------------

def audit_close_trade(
    db,
    symbol: str,
    timeframe: str,
    result: str,
    reason: str = "",
    close_price: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Close active audit trade as WON / LOST / CANCELLED.

    result:
    - WON
    - LOST
    - CANCELLED
    """

    try:
        if db is None:
            return False, "Firestore db is None"

        clean_symbol = normalize_symbol(symbol)
        active_key = make_active_key(clean_symbol, timeframe)

        result = str(result).upper()

        if result not in ["WON", "LOST", "CANCELLED"]:
            return False, f"Invalid result: {result}"

        active_ref = db.collection(TRADE_AUDIT_ACTIVE_INDEX).document(active_key)
        active_doc = active_ref.get()

        if not active_doc.exists:
            return False, "No active audit trade found"

        active_data = active_doc.to_dict() or {}
        trade_id = active_data.get("trade_id")

        if not trade_id:
            return False, "Active index has no trade_id"

        trade_ref = db.collection(TRADE_AUDIT_COLLECTION).document(trade_id)

        close_payload = {
            "status": "CLOSED",
            "result": result,
            "close_reason": reason,
            "close_price": float(close_price) if close_price is not None else None,

            "closed_at": firestore.SERVER_TIMESTAMP,
            "closed_at_iso": utc_now_iso(),

            "updated_at": firestore.SERVER_TIMESTAMP,
            "updated_at_iso": utc_now_iso(),
        }

        trade_ref.set(close_payload, merge=True)

        # Remove active index only. History remains.
        active_ref.delete()

        return True, trade_id

    except Exception as e:
        print(f"[TRADE_AUDIT] audit_close_trade error: {e}")
        return False, str(e)


# --------------------------------------------------
# Manual cancel active audit trade
# --------------------------------------------------

def audit_cancel_trade(
    db,
    symbol: str,
    timeframe: str,
    reason: str = "Manually cancelled",
    close_price: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    return audit_close_trade(
        db=db,
        symbol=symbol,
        timeframe=timeframe,
        result="CANCELLED",
        reason=reason,
        close_price=close_price,
    )


# --------------------------------------------------
# Stats
# --------------------------------------------------

def _timestamp_to_week_key(value: Any) -> str:
    """
    Returns example: 2026-W17
    """
    try:
        if value is None:
            return "unknown"

        # Firestore DatetimeWithNanoseconds supports isocalendar()
        if hasattr(value, "isocalendar"):
            iso = value.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"

        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            iso = dt.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"

    except Exception:
        pass

    return "unknown"


def get_trade_audit_stats(
    db,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """
    Calculate:
    - total closed trades
    - open trades
    - wins
    - losses
    - cancelled
    - win rate
    - weekly stats
    """

    try:
        if db is None:
            return {
                "error": "Firestore db is None",
                "closed_trades": 0,
                "open_trades": 0,
                "wins": 0,
                "losses": 0,
                "cancelled": 0,
                "win_rate": 0,
                "weekly": {},
            }

        query = db.collection(TRADE_AUDIT_COLLECTION)

        if symbol:
            query = query.where("symbol", "==", normalize_symbol(symbol))

        if timeframe:
            query = query.where("timeframe", "==", timeframe)

        docs = list(query.limit(limit).stream())

        closed_trades = 0
        open_trades = 0
        wins = 0
        tp1_hits = 0
        losses = 0
        cancelled = 0


        weekly = {}

        symbol_stats = {}
        timeframe_stats = {}

        for doc in docs:
            data = doc.to_dict() or {}

            status = str(data.get("status", "")).upper()
            result = str(data.get("result", "")).upper()

            trade_symbol = data.get("symbol", "UNKNOWN")
            trade_tf = data.get("timeframe", "UNKNOWN")

            if trade_symbol not in symbol_stats:
                symbol_stats[trade_symbol] = {
                    "closed_trades": 0,
                    "open_trades": 0,
                    "wins": 0,
                    "tp1_hits": 0,
                    "losses": 0,
                    "cancelled": 0,
                    "win_rate": 0,
                    "tp1_or_win_rate": 0,
                }

            if trade_tf not in timeframe_stats:
                timeframe_stats[trade_tf] = {
                    "closed_trades": 0,
                    "open_trades": 0,
                    "wins": 0,
                    "tp1_hits": 0,
                    "losses": 0,
                    "cancelled": 0,
                    "win_rate": 0,
                    "tp1_or_win_rate": 0,
                }

            if status == "OPEN":
                open_trades += 1
                symbol_stats[trade_symbol]["open_trades"] += 1
                timeframe_stats[trade_tf]["open_trades"] += 1
                continue

            if status != "CLOSED":
                continue

            closed_trades += 1
            symbol_stats[trade_symbol]["closed_trades"] += 1
            timeframe_stats[trade_tf]["closed_trades"] += 1

            if result == "WON":
                wins += 1
                symbol_stats[trade_symbol]["wins"] += 1
                timeframe_stats[trade_tf]["wins"] += 1

            elif result == "TP1_HIT":
                tp1_hits += 1
                symbol_stats[trade_symbol]["tp1_hits"] += 1
                timeframe_stats[trade_tf]["tp1_hits"] += 1

            elif result == "LOST":
                losses += 1
                symbol_stats[trade_symbol]["losses"] += 1
                timeframe_stats[trade_tf]["losses"] += 1

            elif result == "CANCELLED":
                cancelled += 1
                symbol_stats[trade_symbol]["cancelled"] += 1
                timeframe_stats[trade_tf]["cancelled"] += 1

            closed_at = data.get("closed_at") or data.get("closed_at_iso")
            week_key = _timestamp_to_week_key(closed_at)

            if week_key not in weekly:
                weekly[week_key] = {
                    "trades": 0,
                    "wins": 0,
                    "tp1_hits": 0,
                    "losses": 0,
                    "cancelled": 0,
                    "win_rate": 0,
                    "tp1_or_win_rate": 0,
                }

            weekly[week_key]["trades"] += 1

            if result == "WON":
                weekly[week_key]["wins"] += 1
            elif result == "TP1_HIT":
                weekly[week_key]["tp1_hits"] += 1
            elif result == "LOST":
                weekly[week_key]["losses"] += 1
            elif result == "CANCELLED":
                weekly[week_key]["cancelled"] += 1

        win_rate = round((wins / (wins + losses)) * 100, 2) if (wins + losses) > 0 else 0
        tp1_or_win_rate = round(((wins + tp1_hits) / (wins + tp1_hits + losses)) * 100, 2) if (wins + tp1_hits + losses) > 0 else 0

        # weekly win rates
        for week_key, item in weekly.items():
            w = item["wins"]
            l = item["losses"]
            item["win_rate"] = round((w / (w + l)) * 100, 2) if (w + l) > 0 else 0

        # symbol win rates
        for _, item in symbol_stats.items():
            w = item["wins"]
            l = item["losses"]
            item["win_rate"] = round((w / (w + l)) * 100, 2) if (w + l) > 0 else 0

        # timeframe win rates
        for _, item in timeframe_stats.items():
            w = item["wins"]
            l = item["losses"]
            item["win_rate"] = round((w / (w + l)) * 100, 2) if (w + l) > 0 else 0

        return {
            "filters": {
                "symbol": normalize_symbol(symbol) if symbol else "ALL",
                "timeframe": timeframe if timeframe else "ALL",
                "limit": limit,
            },
            "closed_trades": closed_trades,
            "open_trades": open_trades,
            "wins": wins,
            "tp1_hits": tp1_hits,
            "losses": losses,
            "cancelled": cancelled,
            "win_rate": win_rate,
            "tp1_or_win_rate": tp1_or_win_rate,
            "weekly": weekly,
            "symbol_stats": symbol_stats,
            "timeframe_stats": timeframe_stats,
        }

    except Exception as e:
        print(f"[TRADE_AUDIT] get_trade_audit_stats error: {e}")
        return {
            "error": str(e),
            "closed_trades": 0,
            "open_trades": 0,
            "wins": 0,
            "losses": 0,
            "cancelled": 0,
            "win_rate": 0,
            "weekly": {},
            "symbol_stats": {},
            "timeframe_stats": {},
        }


# --------------------------------------------------
# List audit trades
# --------------------------------------------------

def list_trade_audit_records(
    db,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    status: Optional[str] = None,
    result: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Optional helper for route:
    GET /trade-audit/trades
    """

    try:
        if db is None:
            return {
                "error": "Firestore db is None",
                "count": 0,
                "trades": [],
            }

        query = db.collection(TRADE_AUDIT_COLLECTION)

        if symbol:
            query = query.where("symbol", "==", normalize_symbol(symbol))

        if timeframe:
            query = query.where("timeframe", "==", timeframe)

        if status:
            query = query.where("status", "==", status.upper())

        if result:
            query = query.where("result", "==", result.upper())

        docs = list(query.limit(limit).stream())

        trades = []

        for doc in docs:
            data = doc.to_dict() or {}
            data["doc_id"] = doc.id
            trades.append(json_safe(data))

        return {
            "count": len(trades),
            "trades": trades,
        }

    except Exception as e:
        print(f"[TRADE_AUDIT] list_trade_audit_records error: {e}")
        return {
            "error": str(e),
            "count": 0,
            "trades": [],
        }

        # --------------------------------------------------
# Live / scheduled open trade checker
# --------------------------------------------------

def _symbol_for_data_loader(symbol: str) -> str:
    """
    BTCUSDT -> BTCUSDT
    BTC/USDT -> BTCUSDT

    Your app.data_loader.get_data() already receives normalized symbols
    in main.py, so we keep normalized format here.
    """
    return normalize_symbol(symbol)


def _check_audit_trade_result_from_candles(trade: Dict[str, Any], df) -> Dict[str, Any]:
    """
    Check one OPEN audit trade using candle High/Low.

    Conservative rule:
    - Entry must trigger first.
    - If SL and TP hit in same candle, SL is counted first.
    - If TP2 exists, TP2 = WON.
    - If only TP1 exists, TP1 = WON.
    """

    try:
        if df is None or df.empty:
            return {
                "status": "OPEN",
                "result": None,
                "reason": "No candle data",
                "close_price": None,
            }

        side = str(trade.get("side") or trade.get("trade_type", "")).upper()

        if "LONG" in side:
            side = "LONG"
        elif "SHORT" in side:
            side = "SHORT"
        else:
            return {
                "status": "OPEN",
                "result": None,
                "reason": "Unknown trade side",
                "close_price": None,
            }

        entry = float(trade["entry"])
        stop_loss = float(trade["stop_loss"])
        targets = [float(x) for x in trade.get("targets", [])]

        if not targets:
            return {
                "status": "OPEN",
                "result": None,
                "reason": "No targets",
                "close_price": None,
            }

        if side == "LONG":
            targets = sorted(targets)
        else:
            targets = sorted(targets, reverse=True)

        # TP2 = full win if available.
        # TP1 = partial win / TP1_HIT if TP2 is not hit yet.
        required_tp_index = 1 if len(targets) >= 2 else 0
        highest_tp_hit_index = -1
        entry_triggered = False

        opened_at_iso = trade.get("opened_at_iso")

        # Make sure index is datetime
        try:
            import pandas as pd

            if "Date" in df.columns:
                df = df.copy()
                df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
                df = df.set_index("Date")

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

            if opened_at_iso:
                opened_dt = pd.to_datetime(opened_at_iso, utc=True, errors="coerce")
                df = df[df.index >= opened_dt]
        except Exception:
            pass

        if df.empty:
            return {
                "status": "OPEN",
                "result": None,
                "reason": "No candles after signal open time",
                "close_price": None,
            }

        for row in df.itertuples():
            curr_low = float(row.Low)
            curr_high = float(row.High)

            if not entry_triggered:
                if side == "LONG" and curr_low <= entry:
                    entry_triggered = True
                elif side == "SHORT" and curr_high >= entry:
                    entry_triggered = True

            if entry_triggered:
                if side == "LONG":
                    # Conservative: SL first if same candle touches both
                    if curr_low <= stop_loss:
                        return {
                            "status": "CLOSED",
                            "result": "LOST",
                            "reason": "Stop Loss Hit",
                            "close_price": stop_loss,
                        }

                    for i, tp in enumerate(targets):
                        if curr_high >= tp and i > highest_tp_hit_index:
                            highest_tp_hit_index = i

                elif side == "SHORT":
                    # Conservative: SL first if same candle touches both
                    if curr_high >= stop_loss:
                        return {
                            "status": "CLOSED",
                            "result": "LOST",
                            "reason": "Stop Loss Hit",
                            "close_price": stop_loss,
                        }

                    for i, tp in enumerate(targets):
                        if curr_low <= tp and i > highest_tp_hit_index:
                            highest_tp_hit_index = i

                # TP2 hit = full WON.
                # TP1 hit only = close as TP1_HIT, not WON.
                if highest_tp_hit_index >= required_tp_index:
                    return {
                        "status": "CLOSED",
                        "result": "WON",
                        "reason": f"Target {highest_tp_hit_index + 1} Hit",
                        "close_price": targets[highest_tp_hit_index],
                    }

                if highest_tp_hit_index == 0 and len(targets) > 1:
                    return {
                        "status": "CLOSED",
                        "result": "TP1_HIT",
                        "reason": "Target 1 Hit",
                        "close_price": targets[0],
                    }

        last_close = None
        try:
            last_close = float(df.iloc[-1]["Close"])
        except Exception:
            pass

        if entry_triggered:
            return {
                "status": "OPEN",
                "result": None,
                "reason": "Trade running",
                "close_price": last_close,
            }

        return {
            "status": "OPEN",
            "result": None,
            "reason": "Waiting for entry",
            "close_price": last_close,
        }

    except Exception as e:
        return {
            "status": "OPEN",
            "result": None,
            "reason": f"Check error: {e}",
            "close_price": None,
        }


def check_open_audit_trades(
    db,
    market_type: str = "crypto",
    limit: int = 100,
    candle_period: str = "7d",
) -> Dict[str, Any]:
    """
    Check all OPEN audit trades using latest market candles.

    Use this from:
    GET/POST /trade-audit/check-open

    This can be called manually, by frontend, Windows Task Scheduler,
    cron-job.org, or Cloud Scheduler.
    """

    try:
        if db is None:
            return {
                "error": "Firestore db is None",
                "checked": 0,
                "closed": 0,
                "still_open": 0,
                "errors": [],
            }

        # Local import avoids circular import problems
        from .data_loader import get_data

        docs = list(
            db.collection(TRADE_AUDIT_COLLECTION)
            .where("status", "==", "OPEN")
            .limit(limit)
            .stream()
        )

        checked = 0
        closed = 0
        still_open = 0
        errors = []
        results = []

        for doc in docs:
            trade = doc.to_dict() or {}
            trade_id = trade.get("trade_id") or doc.id

            symbol = trade.get("symbol")
            timeframe = trade.get("timeframe", "15m")

            if not symbol:
                errors.append({
                    "trade_id": trade_id,
                    "error": "Missing symbol",
                })
                continue

            try:
                data_symbol = _symbol_for_data_loader(symbol)

                df = get_data(
                    data_symbol,
                    interval=timeframe,
                    period=candle_period,
                    market_type=market_type,
                )

                check_result = _check_audit_trade_result_from_candles(trade, df)
                checked += 1

                update_payload = {
                    "last_check_at": firestore.SERVER_TIMESTAMP,
                    "last_check_at_iso": utc_now_iso(),
                    "last_check_reason": check_result.get("reason"),
                    "last_check_price": check_result.get("close_price"),
                    "updated_at": firestore.SERVER_TIMESTAMP,
                    "updated_at_iso": utc_now_iso(),
                }

                if check_result.get("status") == "CLOSED":
                    update_payload.update({
                        "status": "CLOSED",
                        "result": check_result.get("result"),
                        "close_reason": check_result.get("reason"),
                        "close_price": check_result.get("close_price"),
                        "closed_at": firestore.SERVER_TIMESTAMP,
                        "closed_at_iso": utc_now_iso(),
                    })

                    db.collection(TRADE_AUDIT_COLLECTION).document(doc.id).set(
                        update_payload,
                        merge=True,
                    )

                    # Remove active index too
                    active_key = trade.get("active_key")
                    if active_key:
                        try:
                            db.collection(TRADE_AUDIT_ACTIVE_INDEX).document(active_key).delete()
                        except Exception:
                            pass

                    closed += 1

                else:
                    db.collection(TRADE_AUDIT_COLLECTION).document(doc.id).set(
                        update_payload,
                        merge=True,
                    )
                    still_open += 1

                results.append({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": check_result.get("status"),
                    "result": check_result.get("result"),
                    "reason": check_result.get("reason"),
                    "close_price": check_result.get("close_price"),
                })

            except Exception as e:
                errors.append({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "error": str(e),
                })

        return {
            "checked": checked,
            "closed": closed,
            "still_open": still_open,
            "errors": errors,
            "results": results,
        }

    except Exception as e:
        return {
            "error": str(e),
            "checked": 0,
            "closed": 0,
            "still_open": 0,
            "errors": [],
            "results": [],
        }