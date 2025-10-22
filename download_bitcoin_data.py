
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_bitcoin_data_yahoo(days: int = 365) -> pd.DataFrame:
    
    try:
        import yfinance as yf
        logger.info("Using yfinance to download Bitcoin data...")
        
        btc = yf.Ticker("BTC-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = btc.history(start=start_date, end=end_date)
        
        df = pd.DataFrame({
            'timestamp': data.index,
            'symbol': 'BTC',
            'open_price': data['Open'],
            'high_price': data['High'],
            'low_price': data['Low'],
            'close_price': data['Close'],
            'volume': data['Volume']
        })
        
        df = df.reset_index(drop=True)
        logger.info(f"Downloaded {len(df)} days of Bitcoin data")
        return df
        
    except ImportError:
        logger.warning("yfinance not available, creating sample data instead")
        return create_realistic_sample_data(days)

def create_realistic_sample_data(days: int = 365) -> pd.DataFrame:
    
    logger.info(f"Creating {days} days of realistic Bitcoin sample data...")
    
    import numpy as np
    
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')[:days]
    
    initial_price = 45000.0
    prices = [initial_price]
    
    daily_drift = 0.0005  # Slight upward trend
    daily_volatility = 0.04  # 4% daily volatility
    
    for i in range(1, days):
        random_change = np.random.normal(daily_drift, daily_volatility)
        
        if np.random.random() < 0.05:  # 5% chance of large move
            random_change += np.random.normal(0, 0.1)  # Extra volatility
        
        new_price = prices[-1] * (1 + random_change)
        new_price = max(new_price, 1000)  # Minimum price
        prices.append(new_price)
    
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_range = abs(np.random.normal(0, 0.02))  # Daily range
        
        high = close * (1 + daily_range/2)
        low = close * (1 - daily_range/2)
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        
        base_volume = 50000000  # 50M base volume
        volatility_factor = abs(random_change) * 10  # Higher vol = higher volume
        volume = int(base_volume * (1 + volatility_factor))
        
        data.append({
            'timestamp': date,
            'symbol': 'BTC',
            'open_price': round(open_price, 2),
            'high_price': round(high, 2),
            'low_price': round(low, 2),
            'close_price': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created realistic Bitcoin data: {len(df)} records")
    logger.info(f"Price range: ${df['close_price'].min():.2f} - ${df['close_price'].max():.2f}")
    
    return df

def save_bitcoin_data(df: pd.DataFrame, filename: str = "data/bitcoin_historical.csv"):
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    logger.info(f"Bitcoin data saved to {filename}")
    return filename

def main():
    
    logger.info("=== Downloading Bitcoin Data ===")
    
    try:
        df = download_bitcoin_data_yahoo(days=500)  # Get more data for better training
    except Exception as e:
        logger.warning(f"Failed to download real data: {e}")
        df = create_realistic_sample_data(days=500)
    
    filename = save_bitcoin_data(df)
    
    logger.info("=== Data Summary ===")
    logger.info(f"Records: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Price range: ${df['close_price'].min():.2f} - ${df['close_price'].max():.2f}")
    logger.info(f"Average volume: {df['volume'].mean():,.0f}")
    
    return filename

if __name__ == "__main__":
    main()