import requests
import pandas as pd

try:
    import pandas_ta as ta
except Exception:
    ta = None


# ==========================================================
# SETTINGS
# ==========================================================

IGNORE_SYMBOLS = {
    "USDCUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "USDTDAI",
    "BUSDUSDT",
    "EURUSDT",
    "DAIUSDT",
    "PAXGUSDT",
}


# ==========================================================
# TIMEFRAME / MODE MAPPING
# ==========================================================

def get_interval_for_mode(mode):
    """
    Supports both mode names and direct timeframe values.

    UI examples:
      Scalp (15m) -> mode="scalp" -> interval="15m"
      Mid (1h)    -> mode="mid"    -> interval="1h"
      Swing (4h)  -> mode="swing"  -> interval="4h"

    Direct examples:
      mode="15m" -> interval="15m"
      mode="1h"  -> interval="1h"
      mode="4h"  -> interval="4h"
    """
    mode = str(mode or "mid").lower().strip()

    mapping = {
        "scalp": "15m",
        "15m": "15m",

        "mid": "1h",
        "1h": "1h",
        "60m": "1h",

        "swing": "4h",
        "4h": "4h",
        "240m": "4h",
    }

    return mapping.get(mode, "1h")


# ==========================================================
# DATA FETCHING
# ==========================================================

def get_top_volume_coins(limit=50):
    """
    Get top USDT pairs by 24h quote volume from Binance spot.
    Scanner only shortlists coins. Final signal still comes from your main analyzer.
    """
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10).json()

        if not isinstance(response, list):
            return []

        filtered = []

        for item in response:
            symbol = item.get("symbol", "")

            if not symbol.endswith("USDT"):
                continue

            if symbol in IGNORE_SYMBOLS:
                continue

            quote_volume = float(item.get("quoteVolume", 0) or 0)
            last_price = float(item.get("lastPrice", 0) or 0)

            if quote_volume <= 0 or last_price <= 0:
                continue

            filtered.append(item)

        sorted_coins = sorted(
            filtered,
            key=lambda x: float(x.get("quoteVolume", 0) or 0),
            reverse=True
        )

        return [x["symbol"] for x in sorted_coins[:limit]]

    except Exception:
        return []


def get_candles(symbol, interval="15m", limit=500):
    """
    Fetch Binance spot klines.
    Returns DataFrame with lowercase OHLCV columns.
    """
    try:
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&limit={limit}"
        )

        data = requests.get(url, timeout=8).json()

        if isinstance(data, dict) and "code" in data:
            return pd.DataFrame()

        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "vol",
            "quote_asset_volume",
        ]

        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "vol"])

        return df

    except Exception:
        return pd.DataFrame()


# ==========================================================
# INDICATORS
# ==========================================================

def calculate_indicators(df):
    """
    Calculate scanner indicators.

    Uses:
      - EMA50 / EMA200 for trend regime
      - RSI for entry quality zone
      - ADX for trend strength
      - Volume ratio for activity
    """
    try:
        if ta is None:
            return None

        if df is None or df.empty or len(df) < 220:
            return None

        data = df.copy()

        data["rsi"] = ta.rsi(data["close"], length=14)
        data["ema_50"] = ta.ema(data["close"], length=50)
        data["ema_200"] = ta.ema(data["close"], length=200)

        adx_df = ta.adx(data["high"], data["low"], data["close"], length=14)

        if adx_df is not None and "ADX_14" in adx_df.columns:
            data["adx"] = adx_df["ADX_14"]
        else:
            return None

        data["avg_vol_20"] = data["vol"].rolling(window=20).mean()
        data["vol_ratio"] = data["vol"] / data["avg_vol_20"]

        data["range"] = data["high"] - data["low"]
        data["avg_range_14"] = data["range"].rolling(window=14).mean()

        last = data.iloc[-1]

        required = [
            "close",
            "rsi",
            "ema_50",
            "ema_200",
            "adx",
            "avg_vol_20",
            "vol_ratio",
            "avg_range_14",
        ]

        for col in required:
            if col not in last or pd.isna(last[col]):
                return None

        return last

    except Exception:
        return None


# ==========================================================
# SCORING ENGINE
# ==========================================================

def score_candidate(last, mode="scalp"):
    """
    Score coin instead of hard filtering.

    Reason:
      ADX and volume are useful, but making them mandatory can miss good pullback setups.
      This keeps good coins while ranking stronger ones higher.
    """
    try:
        price = float(last["close"])
        rsi = float(last["rsi"])
        ema_50 = float(last["ema_50"])
        ema_200 = float(last["ema_200"])
        adx = float(last["adx"])
        vol_ratio = float(last["vol_ratio"])

        long_score = 0
        short_score = 0
        long_reasons = []
        short_reasons = []

        # ==================================================
        # LONG SCORE
        # ==================================================

        if price > ema_200:
            long_score += 1
            long_reasons.append("price above EMA200")

        if ema_50 > ema_200:
            long_score += 2
            long_reasons.append("EMA50 above EMA200")

        if price > ema_50:
            long_score += 1
            long_reasons.append("price above EMA50")

        if 45 <= rsi <= 65:
            long_score += 2
            long_reasons.append("RSI in long sweet spot")
        elif 40 <= rsi < 45:
            long_score += 1
            long_reasons.append("RSI near long pullback zone")
        elif rsi > 72:
            long_score -= 2
            long_reasons.append("RSI overbought risk")

        if adx >= 25:
            long_score += 2
            long_reasons.append("strong ADX")
        elif adx >= 18:
            long_score += 1
            long_reasons.append("acceptable ADX")
        else:
            long_score -= 1
            long_reasons.append("weak ADX")

        if vol_ratio >= 1.30:
            long_score += 2
            long_reasons.append("volume expansion")
        elif vol_ratio >= 1.05:
            long_score += 1
            long_reasons.append("normal/above avg volume")
        elif vol_ratio < 0.65:
            long_score -= 1
            long_reasons.append("low volume")

        # ==================================================
        # SHORT SCORE
        # ==================================================

        if price < ema_200:
            short_score += 1
            short_reasons.append("price below EMA200")

        if ema_50 < ema_200:
            short_score += 2
            short_reasons.append("EMA50 below EMA200")

        if price < ema_50:
            short_score += 1
            short_reasons.append("price below EMA50")

        if 35 <= rsi <= 55:
            short_score += 2
            short_reasons.append("RSI in short sweet spot")
        elif 55 < rsi <= 60:
            short_score += 1
            short_reasons.append("RSI near short bounce zone")
        elif rsi < 28:
            short_score -= 2
            short_reasons.append("RSI oversold bounce risk")

        if adx >= 25:
            short_score += 2
            short_reasons.append("strong ADX")
        elif adx >= 18:
            short_score += 1
            short_reasons.append("acceptable ADX")
        else:
            short_score -= 1
            short_reasons.append("weak ADX")

        if vol_ratio >= 1.30:
            short_score += 2
            short_reasons.append("volume expansion")
        elif vol_ratio >= 1.05:
            short_score += 1
            short_reasons.append("normal/above avg volume")
        elif vol_ratio < 0.65:
            short_score -= 1
            short_reasons.append("low volume")

        # ==================================================
        # DECIDE SIDE
        # ==================================================

        if long_score > short_score:
            side = "LONG"
            score = long_score
            reasons = long_reasons
        elif short_score > long_score:
            side = "SHORT"
            score = short_score
            reasons = short_reasons
        else:
            side = "NEUTRAL"
            score = max(long_score, short_score)
            reasons = ["mixed conditions"]

        if score >= 7:
            quality = "STRONG"
        elif score >= 5:
            quality = "GOOD"
        elif score >= 3:
            quality = "WATCHLIST"
        else:
            quality = "WEAK"

        return {
            "side": side,
            "score": score,
            "quality": quality,
            "reasons": reasons,
            "price": price,
            "rsi": rsi,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "adx": adx,
            "vol_ratio": vol_ratio,
        }

    except Exception:
        return None


# ==========================================================
# MAIN SCANNER
# ==========================================================

def scan_market(mode="scalp", top_limit=50, min_score=None, max_candidates=15):
    """
    Main scanner.

    mode:
      scalp = 15m
      mid   = 1h
      swing = 4h

    Also supports direct:
      15m, 1h, 4h

    min_score:
      None = auto by mode
      4    = more candidates
      5    = balanced
      6    = stricter
      7    = very strict

    max_candidates:
      returns top ranked candidates only
    """
    if ta is None:
        return {
            "candidates": [],
            "logs": ["pandas_ta not installed. Scanner disabled."],
        }

    interval = get_interval_for_mode(mode)

    if min_score is None:
        if interval == "1h":
            min_score = 5
        elif interval == "4h":
            min_score = 6
        else:
            min_score = 5

    top_coins = get_top_volume_coins(top_limit)

    results = {
        "candidates": [],
        "logs": [],
        "interval": interval,
        "mode": mode,
    }

    if not top_coins:
        results["logs"].append("No top volume coins found.")
        return results

    for symbol in top_coins:
        df = get_candles(symbol, interval=interval, limit=500)

        if df.empty:
            results["logs"].append(f"❌ {symbol}: No data")
            continue

        last = calculate_indicators(df)

        if last is None:
            results["logs"].append(f"⚠️ {symbol}: Indicator/data error")
            continue

        scored = score_candidate(last, mode=mode)

        if scored is None:
            results["logs"].append(f"⚠️ {symbol}: Score error")
            continue

        side = scored["side"]
        score = scored["score"]
        quality = scored["quality"]
        price = scored["price"]
        rsi = scored["rsi"]
        adx = scored["adx"]
        vol_ratio = scored["vol_ratio"]

        if side != "NEUTRAL" and score >= min_score:
            signal = f"{str(mode).upper()}_{side}"

            results["candidates"].append({
                "symbol": symbol,
                "price": price,
                "signal": signal,
                "side": side,
                "score": score,
                "quality": quality,
                "rsi": rsi,
                "adx": adx,
                "vol_ratio": vol_ratio,
                "interval": interval,
                "reasons": scored["reasons"][:5],
            })

            results["logs"].append(
                f"✅ MATCH ({signal} | {quality} | score {score}) -> "
                f"{symbol} | TF: {interval} | Price: {price:.6f} | "
                f"RSI: {rsi:.1f} | ADX: {adx:.1f} | VolRatio: {vol_ratio:.2f}x"
            )
        else:
            results["logs"].append(
                f"❌ SKIP -> {symbol} | TF: {interval} | side={side} | score={score} | "
                f"Price: {price:.6f} | RSI: {rsi:.1f} | "
                f"ADX: {adx:.1f} | VolRatio: {vol_ratio:.2f}x"
            )

    results["candidates"] = sorted(
        results["candidates"],
        key=lambda x: x.get("score", 0),
        reverse=True
    )[:max_candidates]

    return results


# ==========================================================
# OPTIONAL QUICK TEST
# ==========================================================

if __name__ == "__main__":
    output = scan_market(
        mode="mid",          # scalp / mid / swing OR 15m / 1h / 4h
        top_limit=50,
        min_score=None,
        max_candidates=15,
    )

    print(f"\n=== SCANNER MODE: {output.get('mode')} | TF: {output.get('interval')} ===")

    print("\n=== CANDIDATES ===")
    for c in output["candidates"]:
        print(
            f"{c['symbol']} | {c['signal']} | TF={c['interval']} | "
            f"score={c['score']} | RSI={c['rsi']:.1f} | "
            f"ADX={c['adx']:.1f} | Vol={c['vol_ratio']:.2f}x"
        )

    print("\n=== LOGS ===")
    for log in output["logs"][:100]:
        print(log)