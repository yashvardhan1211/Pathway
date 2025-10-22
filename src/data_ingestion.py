
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
import logging
from pathlib import Path

from data_models import StockPrice

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.symbol: Optional[str] = None
    
    def load_coinmarketcap_csv(self, file_path: str, symbol: str = "BTC") -> pd.DataFrame:
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            column_mapping = {
                'Date': 'timestamp',
                'Open': 'open_price',
                'High': 'high_price', 
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            df['symbol'] = symbol
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df = df.dropna(subset=['open_price', 'high_price', 'low_price', 'close_price'])
            
            logger.info(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            self.data = df
            self.symbol = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def create_sample_data(self, symbol: str = "BTC", days: int = 100) -> pd.DataFrame:
        
        logger.info(f"Creating {days} days of sample data for {symbol}")
        
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        np.random.seed(42)  # For reproducible results
        
        initial_price = 45000.0
        prices = [initial_price]
        
        for i in range(1, days):
            change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Minimum price of $1000
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily range
            
            high = close * (1 + volatility/2)
            low = close * (1 - volatility/2)
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
            
            base_volume = 1000000
            volume_multiplier = 1 + abs(np.random.normal(0, 0.5))
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open_price': round(open_price, 2),
                'high_price': round(high, 2),
                'low_price': round(low, 2),
                'close_price': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        self.data = df
        self.symbol = symbol
        
        logger.info(f"Created sample data: {len(df)} records")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Cleaning data...")
        
        original_len = len(df)
        
        df = df.drop_duplicates(subset=['timestamp', 'symbol'])
        
        df = df[
            (df['high_price'] >= df['low_price']) &
            (df['high_price'] >= df['open_price']) &
            (df['high_price'] >= df['close_price']) &
            (df['low_price'] <= df['open_price']) &
            (df['low_price'] <= df['close_price']) &
            (df['volume'] >= 0)
        ]
        
        df = df.fillna(method='ffill')
        
        df = df.dropna()
        
        logger.info(f"Data cleaning: {original_len} -> {len(df)} records")
        
        return df
    
    def get_data_summary(self) -> dict:
        
        if self.data is None:
            return {"error": "No data loaded"}
        
        df = self.data
        
        return {
            "symbol": self.symbol,
            "total_records": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "price_stats": {
                "min_close": df['close_price'].min(),
                "max_close": df['close_price'].max(),
                "mean_close": df['close_price'].mean(),
                "std_close": df['close_price'].std()
            },
            "volume_stats": {
                "min_volume": df['volume'].min(),
                "max_volume": df['volume'].max(),
                "mean_volume": df['volume'].mean()
            }
        }