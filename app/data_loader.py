import ccxt
import pandas as pd
import yfinance as yf
import time
import threading


_CRYPTO_EXCHANGE_LOCAL = threading.local()


def _get_binance_futures_exchange():
    exchange = getattr(
        _CRYPTO_EXCHANGE_LOCAL,
        "binance_futures",
        None,
    )

    if exchange is None:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {
                "defaultType": "future",
            },
        })

        _CRYPTO_EXCHANGE_LOCAL.binance_futures = exchange

    return exchange


def get_data(symbol, interval, limit=1000, period=None, market_type="crypto"):
    """
    Fetches OHLCV data.
    - If market_type='crypto': Uses Binance (CCXT)
    - If market_type='forex': Uses Yahoo Finance (yfinance)
    """
    
    # ==========================================
    # --- 1. FOREX LOGIC (YAHOO FINANCE) ---
    # ==========================================
    if market_type == "forex":
        
        # --- SPECIAL MAPPINGS FOR COMMODITIES & INDICES ---
        # Yahoo Finance tickers usually act different for metals/oil.
        # We map common trader terms (MT4 style) to Yahoo Finance Tickers.
        special_map = {
            "XAUUSD": "GC=F",   # Gold Futures
            "GOLD": "GC=F",
            "XAGUSD": "SI=F",   # Silver Futures
            "SILVER": "SI=F",
            "OIL": "CL=F",      # Crude Oil
            "WTI": "CL=F",
            "US30": "^DJI",     # Dow Jones Industrial Average
            "SPX500": "^GSPC",  # S&P 500
            "NAS100": "^IXIC",  # Nasdaq 100
            "NDX": "^IXIC",
            "DXY": "DX-Y.NYB"   # US Dollar Index
        }

        upper_sym = symbol.upper()
        
        if upper_sym in special_map:
            symbol = special_map[upper_sym]
            print(f"--- [INFO] Auto-mapped {upper_sym} to {symbol} for better data ---")
        
        # --- STANDARD FOREX PAIR HANDLING ---
        # If it's not a mapped symbol and doesn't have a special suffix yet
        # Standard pairs like EURUSD need '=X' (e.g., EURUSD=X)
        elif not symbol.endswith("=X") and not symbol.endswith("=F") and not symbol.startswith("^"):
             symbol = f"{symbol}=X"

        print(f"--- [DEBUG] Connecting to Yahoo Finance for {symbol} ({interval}) ---")
        try:
            # Map Intervals & Periods
            # yfinance supports: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            yf_interval = interval
            
            # Auto-adjust period if not provided
            if not period:
                if interval == "15m": period = "5d"      # Short term
                elif interval == "1h": period = "1mo"    # Medium term
                elif interval == "4h": period = "3mo"    # Swing term
                elif interval == "1d": period = "2y"     # Macro
                else: period = "1mo"

            # HANDLE 4H SPECIAL CASE (Resample from 1h)
            if interval == "4h":
                yf_interval = "1h"

            # Fetch Data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            if df.empty:
                print(f"--- [ERROR] Yahoo Finance returned NO data for {symbol}. ---")
                return pd.DataFrame()

            # Clean up Index
            df.reset_index(inplace=True)

            # Standardize Columns
            # YF often includes 'Dividends', 'Stock Splits'. We only need OHLCV.
            if 'Date' not in df.columns and 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            
            keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[c for c in keep_cols if c in df.columns]]

            # Remove Timezone info (make it naive) to match Binance format
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
                if df['Date'].dt.tz is not None:
                    df['Date'] = df['Date'].dt.tz_localize(None)

            # RESAMPLING LOGIC FOR 4H
            if interval == "4h":
                df.set_index('Date', inplace=True)
                df_resampled = df.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                # Drop rows with NaNs (incomplete candles) and reset index
                df = df_resampled.dropna().reset_index()

            print(f"--- [SUCCESS] Fetched {len(df)} candles from YF. ---")
            return df

        except Exception as e:
            print(f"--- [ERROR] YF Fetch Failed: {e} ---")
            return pd.DataFrame()

    # ==========================================
    # --- 2. CRYPTO LOGIC (BINANCE - CCXT) ---
    # ==========================================
    else:
        print(f"--- [DEBUG] Connecting to Binance USD-M Futures for {symbol} ({interval}) ---")
        try:
            exchange = _get_binance_futures_exchange()
            
            print(f"--- [DEBUG] CCXT defaultType: {exchange.options.get('defaultType')} ---")
            # exchange.options['defaultType'] = 'future' # Optional: Enable for Futures data

            # Binance/CCXT uses "1M" for monthly candles, while the UI/map may pass "1mo".
            binance_timeframe = str(interval).strip()
            binance_timeframe_map = {
                "1mo": "1M",
                "1month": "1M",
                "monthly": "1M",
            }
            binance_timeframe = binance_timeframe_map.get(binance_timeframe.lower(), binance_timeframe)

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=binance_timeframe, limit=limit)

            if not ohlcv:
                print(f"--- [ERROR] Binance returned NO data for {symbol}. ---")
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[cols] = df[cols].astype(float)

            print(f"--- [SUCCESS] Fetched {len(df)} candles from Binance. ---")
            return df

        except Exception as e:
            print("\n" + "="*40)
            print("!!! BINANCE DATA ERROR !!!")
            print(f"Error Message: {str(e)}")
            print("="*40 + "\n")
            return pd.DataFrame()

def get_mtf_data(symbol, base_tf, market_type="crypto"):
    """
    Fetches Multi-Timeframe (MTF) Data based on the user's requested timeframe.
    Returns a dictionary of DataFrames:
    - Target TF (What user asked for)
    - Confirm TF (One level higher)
    - Macro TF (Two levels higher)
    """
    print(f"\n--- [DEBUG] Fetching MTF Data for {symbol} ({base_tf} Base) ---")
    
    # 1. Define the MTF Hierarchy
    tf_hierarchy = {
        "15m": ["15m", "1h", "4h"],   # Day Trading
        "1h":  ["1h", "4h", "1d"],    # Short-term Swing
        "4h":  ["4h", "1d", "1w"]     # Macro Swing
    }
    
    # If the user asks for a TF not in our hierarchy, default to fetching just that one
    tf_list = tf_hierarchy.get(base_tf, [base_tf])
    
    # 2. Fetch data for all required timeframes
    mtf_data = {}
    for tf in tf_list:
        print(f"-> Fetching {tf} data...")
        df = get_data(symbol=symbol, interval=tf, market_type=market_type)
        if df.empty:
            print(f"-> [WARNING] Empty data returned for {tf}. MTF Context might be incomplete.")
        mtf_data[tf] = df
        time.sleep(1) # Small pause to avoid rate limits (especially for Binance)

    # 3. Organize and return the payload
    # We always return 'target', 'confirm', and 'macro' keys so AI engine knows what to look for.
    payload = {
        "target": mtf_data.get(tf_list[0], pd.DataFrame()),
        "confirm": mtf_data.get(tf_list[1], pd.DataFrame()) if len(tf_list) > 1 else pd.DataFrame(),
        "macro": mtf_data.get(tf_list[2], pd.DataFrame()) if len(tf_list) > 2 else pd.DataFrame(),
        "target_tf": tf_list[0],
        "confirm_tf": tf_list[1] if len(tf_list) > 1 else None,
        "macro_tf": tf_list[2] if len(tf_list) > 2 else None
    }
    
    print("--- [SUCCESS] MTF Data Fetch Complete ---\n")
    return payload