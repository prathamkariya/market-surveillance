import os
import pandas as pd
import requests
import datetime
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from _common import compute_engineered_features

# Load .env from repo root regardless of cwd
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]
US_SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY", "AMZN", "META", "GOOGL"]
N_DAYS = 400  # Fetch 400 days to ensure >250 usable trading days

def fetch_crypto_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch daily OHLCV from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": days
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    records = []
    for row in data:
        # timestamp, open, high, low, close, volume, close_time, ...
        dt = pd.to_datetime(row[0], unit='ms').strftime('%Y-%m-%d')
        records.append({
            'date': dt,
            'close': float(row[4]),
            'volume': float(row[5])
        })
    df = pd.DataFrame(records).set_index('date')
    return df

def fetch_us_equity_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch daily OHLCV from Alpaca API"""
    api_key = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Alpaca API keys not found in .env")
        
    client = StockHistoricalDataClient(api_key, secret_key)
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=days)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        feed="iex"  # Free tier: IEX feed only (SIP requires paid subscription)
    )
    
    bars = client.get_stock_bars(req)
    if bars.df.empty:
        return pd.DataFrame()
        
    # Alpaca returns multi-index (symbol, timestamp)
    df = bars.df.loc[symbol].copy()
    df.index = df.index.tz_localize(None).strftime('%Y-%m-%d')
    return df[['close', 'volume']]

def process_market(market: str, symbols: list[str], fetch_func):
    out_dir = os.path.join("trained_models", market)
    os.makedirs(out_dir, exist_ok=True)
    
    all_features = []
    for sym in symbols:
        print(f"Fetching {sym}...")
        df_raw = fetch_func(sym, N_DAYS)
        if df_raw.empty:
            print(f"Warning: No data for {sym}")
            continue
            
        print(f"  Computing features for {sym} ({len(df_raw)} raw rows)...")
        # Rule 2.2: compute features PER SYMBOL
        df_feat = compute_engineered_features(df_raw)
        
        # Add symbol column for later evaluation
        df_feat['symbol'] = sym
        
        # Drop the warmup rows containing NaNs
        df_feat = df_feat.dropna(subset=['return', 'volume_ratio_20d', 'volatility_20d'])
        all_features.append(df_feat)
        
    pooled_df = pd.concat(all_features)
    print(f"\n{market} Pooled Data: {len(pooled_df)} total usable rows")
    
    out_file = os.path.join(out_dir, "real_if_input.csv")
    pooled_df.to_csv(out_file)
    print(f"Saved to {out_file}\n")

if __name__ == "__main__":
    print("=== Processing CRYPTO ===")
    process_market("CRYPTO", CRYPTO_SYMBOLS, fetch_crypto_data)
    
    print("=== Processing US_EQUITY ===")
    process_market("US_EQUITY", US_SYMBOLS, fetch_us_equity_data)
