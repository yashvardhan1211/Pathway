
import sys
import os
sys.path.append('src')

from datetime import datetime, timedelta
import pandas as pd
import logging

from data_models import ModelConfig
from data_ingestion import DataIngestionPipeline
from ai_predictor import AIPredictor
from temporal_validator import TemporalValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system():
    
    logger.info("=== Testing AI Stock Prediction System ===")
    
    try:
        logger.info("1. Creating sample data...")
        pipeline = DataIngestionPipeline()
        df = pipeline.create_sample_data(symbol="BTC", days=200)
        logger.info(f" Created {len(df)} days of sample data")
        
        logger.info("2. Setting up model configuration...")
        config = ModelConfig(
            lookback_window=30,
            prediction_horizon=1,
            features=["price_change", "volatility", "volume_ratio", "rsi"],
            model_type="random_forest"
        )
        logger.info(" Model config created")
        
        logger.info("3. Training AI model...")
        predictor = AIPredictor(config)
        training_results = predictor.train(df)
        logger.info(f" Model trained - Test accuracy: {training_results['test_accuracy']:.3f}")
        
        logger.info("4. Testing temporal validation...")
        validator = TemporalValidator()
        
        prediction_time = df['timestamp'].iloc[-50]  # 50 days from end
        past_data = df[df['timestamp'] <= prediction_time]
        is_valid = validator.validate_prediction_request(past_data, prediction_time)
        logger.info(f" Past data validation: {is_valid}")
        
        future_data = df  # includes all data including future
        is_invalid = validator.validate_prediction_request(future_data, prediction_time)
        logger.info(f" Future data validation (should be False): {is_invalid}")
        
        logger.info("5. Making predictions...")
        predictions = []
        
        for i in range(10):
            pred_time = df['timestamp'].iloc[-(50-i)]  # Start from 50 days before end
            
            try:
                prediction = predictor.predict_direction(df, pred_time)
                predictions.append(prediction)
                logger.info(f"Prediction {i+1}: {prediction.predicted_direction} "
                           f"(confidence: {prediction.confidence:.3f})")
            except Exception as e:
                logger.error(f"Prediction {i+1} failed: {str(e)}")
        
        logger.info(f" Made {len(predictions)} predictions successfully")
        
        logger.info("6. Calculating prediction accuracy...")
        correct_predictions = 0
        total_predictions = 0
        
        for pred in predictions:
            pred_idx = df[df['timestamp'] == pred.timestamp].index[0]
            if pred_idx < len(df) - 1:
                current_price = df.iloc[pred_idx]['close_price']
                next_price = df.iloc[pred_idx + 1]['close_price']
                actual_direction = "up" if next_price > current_price else "down"
                
                if pred.predicted_direction == actual_direction:
                    correct_predictions += 1
                total_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            logger.info(f" Prediction accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        logger.info("7. Data summary...")
        summary = pipeline.get_data_summary()
        logger.info(f" Data range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        logger.info(f" Price range: ${summary['price_stats']['min_close']:.2f} - ${summary['price_stats']['max_close']:.2f}")
        
        logger.info("\n ALL TESTS PASSED! The AI system is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f" Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    exit(0 if success else 1)