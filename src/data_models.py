
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import pandas as pd

@dataclass
class StockPrice:
    
    timestamp: datetime
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    
    def to_dict(self) -> dict:
        
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume
        }

@dataclass
class Prediction:
    
    timestamp: datetime
    symbol: str
    predicted_direction: str  # "up" or "down"
    confidence: float
    actual_direction: Optional[str] = None  # filled after outcome known
    correct: Optional[bool] = None  # filled after outcome known
    
    def to_dict(self) -> dict:
        
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'predicted_direction': self.predicted_direction,
            'confidence': self.confidence,
            'actual_direction': self.actual_direction,
            'correct': self.correct
        }

@dataclass
class ModelConfig:
    
    lookback_window: int  # number of previous timestamps to consider
    prediction_horizon: int  # how far ahead to predict
    features: List[str]  # which technical indicators to use
    model_type: str  # "lstm", "linear_regression", etc.
    
    def to_dict(self) -> dict:
        
        return {
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'features': self.features,
            'model_type': self.model_type
        }