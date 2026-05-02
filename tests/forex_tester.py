import yfinance as yf
import pandas as pd

def test_connection(symbol):
    print(f"\n--- Testing Connection for: {symbol} ---")
    
    # 1. Ticker Object එක හදමු
    ticker = yf.Ticker(symbol)
    
    try:
        # 2. අන්තිම දවස් 5 ඩේටා ඉල්ලමු (1 Hour candles)
        print("Fetching data...")
        df = ticker.history(period="5d", interval="1h")
        
        # 3. Data ආවද බලමු
        if df.empty:
            print(f"❌ FAILED: No data found for '{symbol}'.")
            print("Tip: Forex pairs on Yahoo Finance often need '=X' at the end (e.g., EURUSD=X)")
        else:
            print(f"✅ SUCCESS: Received {len(df)} candles.")
            print("Sample Data:")
            print(df.tail(3)) # අන්තිම පේළි 3 පෙන්නන්න
            
    except Exception as e:
        print(f"❌ ERROR: Something went wrong -> {e}")

if __name__ == "__main__":
    print("=== FOREX DATA CONNECTION TEST ===")
    
    # Test 1: වැරදි විදිහ (සරලව)
    test_connection("EURUSD")
    
    # Test 2: හරි විදිහ (Yahoo Finance Format)
    test_connection("EURUSD=X")
    
    # Test 3: Gold (Optional)
    test_connection("GC=F")