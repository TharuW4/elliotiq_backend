import requests
import pandas as pd
import pandas_ta as ta

def get_top_volume_coins(limit=30):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10).json()
        ignore = ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'USDTDAI', 'BUSDUSDT', 'EURUSDT']
        filtered = [x for x in response if x['symbol'].endswith('USDT') and x['symbol'] not in ignore]
        sorted_coins = sorted(filtered, key=lambda x: float(x['quoteVolume']), reverse=True)
        return [x['symbol'] for x in sorted_coins[:limit]]
    except:
        return []

def get_candles(symbol, interval="15m", limit=500):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=5).json()
        if isinstance(data, dict) and 'code' in data: return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'x', 'y', 'z', 'w', 'q', 'k'])
        cols = ['open', 'high', 'low', 'close', 'vol']
        df[cols] = df[cols].apply(pd.to_numeric, axis=1)
        return df
    except:
        return pd.DataFrame()

def calculate_indicators(df):
    try:
        if len(df) < 200: return None
        
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_200'] = ta.ema(df['close'], length=200)
        
        last_row = df.iloc[-1]
        # Check for NaN in critical indicators
        if pd.isna(last_row['ema_200']) or pd.isna(last_row['rsi']):
            return None
            
        return last_row
    except:
        return None

def scan_market(mode="scalp"):
    if ta is None:
        return {
        "candidates": [],
        "logs": ["pandas_ta not installed. Scanner disabled."]
        }
    interval = "15m" if mode == "scalp" else "4h"
    top_coins = get_top_volume_coins(40)
    
    results = {
        "candidates": [],
        "logs": []
    }
    
    # print(f">> Scanning {len(top_coins)} coins...")

    for symbol in top_coins:
        df = get_candles(symbol, interval, limit=500)
        
        if df.empty:
            results["logs"].append(f"❌ {symbol}: No Data")
            continue
        
        last = calculate_indicators(df)
        if last is None:
            results["logs"].append(f"⚠️ {symbol}: Data Error")
            continue
        
        price = last['close']
        rsi = last['rsi']
        ema_200 = last['ema_200']
        ema_50 = last['ema_50']
        
        signal = None
        status = "SKIP"
        
        if mode == "scalp":
            if (price > ema_200 and rsi < 60): 
                signal = "POTENTIAL_LONG"
            elif (price < ema_200 and rsi > 40):
                signal = "POTENTIAL_SHORT"
                
        elif mode == "swing":
            if (pd.notna(ema_50) and price > ema_50 and rsi < 70):
                signal = "SWING_LONG"
            elif (pd.notna(ema_50) and price < ema_50 and rsi > 30):
                signal = "SWING_SHORT"

        log_msg = f"{symbol} | Price: {price:.2f} | RSI: {rsi:.1f}"
        
        if signal:
            status = f"✅ MATCH ({signal})"
            results["candidates"].append({
                "symbol": symbol, "price": price, "signal": signal, "rsi": rsi, "interval": interval
            })
        else:
            status = "❌ SKIP"

        results["logs"].append(f"{status} -> {log_msg}")

    return results