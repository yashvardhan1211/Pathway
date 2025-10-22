
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time

from data_models import Prediction, StockPrice
from ai_predictor import AIPredictor
from temporal_validator import TemporalValidator

logger = logging.getLogger(__name__)

class MarketSimulator:
    
    
    def __init__(self, predictor: AIPredictor, data: pd.DataFrame):
        self.predictor = predictor
        self.data = data.copy().sort_values('timestamp').reset_index(drop=True)
        self.validator = TemporalValidator()
        
        self.predictions: List[Prediction] = []
        self.current_index = 0
        self.simulation_results = []
        
    def run_simulation(self, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None,
                      prediction_interval_days: int = 1) -> Dict:
        
        logger.info("Starting market simulation...")
        
        if start_date is None:
            start_idx = int(len(self.data) * 0.8)  # Start at 80% through data
            start_date = self.data.iloc[start_idx]['timestamp']
        
        if end_date is None:
            end_date = self.data.iloc[-2]['timestamp']  # Leave one day for validation
        
        logger.info(f"Simulation period: {start_date} to {end_date}")
        
        start_idx = self.data[self.data['timestamp'] >= start_date].index[0]
        end_idx = self.data[self.data['timestamp'] <= end_date].index[-1]
        
        predictions_made = 0
        correct_predictions = 0
        
        for idx in range(start_idx, end_idx, prediction_interval_days):
            current_time = self.data.iloc[idx]['timestamp']
            
            try:
                prediction = self.predictor.predict_direction(self.data, current_time)
                
                if idx + 1 < len(self.data):
                    current_price = self.data.iloc[idx]['close_price']
                    next_price = self.data.iloc[idx + 1]['close_price']
                    actual_direction = "up" if next_price > current_price else "down"
                    
                    prediction.actual_direction = actual_direction
                    prediction.correct = (prediction.predicted_direction == actual_direction)
                    
                    if prediction.correct:
                        correct_predictions += 1
                    
                    predictions_made += 1
                    
                    result = {
                        'timestamp': current_time,
                        'current_price': current_price,
                        'next_price': next_price,
                        'predicted_direction': prediction.predicted_direction,
                        'actual_direction': actual_direction,
                        'confidence': prediction.confidence,
                        'correct': prediction.correct
                    }
                    self.simulation_results.append(result)
                
                self.predictions.append(prediction)
                
                if predictions_made % 10 == 0:
                    logger.info(f"Made {predictions_made} predictions...")
                
            except Exception as e:
                logger.warning(f"Prediction failed at {current_time}: {str(e)}")
                continue
        
        accuracy = correct_predictions / predictions_made if predictions_made > 0 else 0
        
        simulation_summary = {
            'total_predictions': predictions_made,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'simulation_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'results': self.simulation_results
        }
        
        logger.info(f"Simulation completed: {correct_predictions}/{predictions_made} "
                   f"correct ({accuracy:.3f} accuracy)")
        
        return simulation_summary
    
    def get_performance_metrics(self) -> Dict:
        
        if not self.simulation_results:
            return {"error": "No simulation results available"}
        
        df = pd.DataFrame(self.simulation_results)
        
        total_predictions = len(df)
        correct_predictions = df['correct'].sum()
        accuracy = correct_predictions / total_predictions
        
        up_predictions = df[df['predicted_direction'] == 'up']
        down_predictions = df[df['predicted_direction'] == 'down']
        
        up_accuracy = up_predictions['correct'].mean() if len(up_predictions) > 0 else 0
        down_accuracy = down_predictions['correct'].mean() if len(down_predictions) > 0 else 0
        
        high_confidence = df[df['confidence'] > 0.8]
        high_conf_accuracy = high_confidence['correct'].mean() if len(high_confidence) > 0 else 0
        
        return {
            'overall_accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': int(correct_predictions),
            'up_predictions': {
                'count': len(up_predictions),
                'accuracy': up_accuracy
            },
            'down_predictions': {
                'count': len(down_predictions),
                'accuracy': down_accuracy
            },
            'high_confidence_accuracy': high_conf_accuracy,
            'average_confidence': df['confidence'].mean()
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        
        return pd.DataFrame(self.simulation_results)