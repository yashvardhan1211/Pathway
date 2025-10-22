
from datetime import datetime
from typing import List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TemporalValidator:
    
    
    def __init__(self):
        self.current_time: Optional[datetime] = None
        self.validation_enabled = True
    
    def set_current_time(self, timestamp: datetime):
        
        self.current_time = timestamp
        logger.debug(f"Current simulation time set to: {timestamp}")
    
    def validate_data_access(self, data_timestamps: List[datetime], 
                           prediction_time: datetime) -> bool:
        
        if not self.validation_enabled:
            return True
            
        future_data = [ts for ts in data_timestamps if ts > prediction_time]
        
        if future_data:
            logger.error(f"LOOKAHEAD BIAS DETECTED!")
            logger.error(f"Prediction time: {prediction_time}")
            logger.error(f"Future data timestamps: {future_data[:5]}...")  # Show first 5
            return False
        
        logger.debug(f"Temporal validation passed for prediction at {prediction_time}")
        return True
    
    def validate_prediction_request(self, data: pd.DataFrame, 
                                  prediction_time: datetime) -> bool:
        
        if not self.validation_enabled:
            return True
            
        if 'timestamp' not in data.columns:
            logger.error("Data must have 'timestamp' column for validation")
            return False
        
        timestamps = pd.to_datetime(data['timestamp']).tolist()
        
        return self.validate_data_access(timestamps, prediction_time)
    
    def get_valid_data_window(self, data: pd.DataFrame, 
                            prediction_time: datetime) -> pd.DataFrame:
        
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")
        
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        valid_data = data[data['timestamp'] <= prediction_time]
        
        logger.debug(f"Filtered data from {len(data)} to {len(valid_data)} rows "
                    f"for prediction at {prediction_time}")
        
        return valid_data
    
    def disable_validation(self):
        
        logger.warning("Temporal validation DISABLED - use only for testing!")
        self.validation_enabled = False
    
    def enable_validation(self):
        
        logger.info("Temporal validation ENABLED")
        self.validation_enabled = True