
import sys
import os
sys.path.append('src')

from datetime import datetime, timedelta
import pandas as pd
import logging

from data_models import ModelConfig
from data_ingestion import DataIngestionPipeline
from enhanced_ai_predictor import EnhancedAIPredictor
from market_simulator import MarketSimulator
from visualization import StockVisualization
from download_bitcoin_data import main as download_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_system():
    
    logger.info("=== ENHANCED AI SYSTEM TEST ===")
    
    try:
        logger.info("1. Getting Bitcoin data...")
        data_file = download_data()
        
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f" Loaded {len(df)} days of Bitcoin data")
        
        logger.info("2. Setting up enhanced AI model...")
        config = ModelConfig(
            lookback_window=50,  # Longer lookback
            prediction_horizon=1,
            features=["enhanced"],  # Will use all advanced features
            model_type="ensemble"
        )
        
        predictor = EnhancedAIPredictor(config)
        
        logger.info("3. Training enhanced ensemble...")
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        
        training_results = predictor.train_ensemble(train_data)
        logger.info(f" Enhanced model trained - Best accuracy: {training_results['best_test_accuracy']:.3f}")
        
        logger.info("4. Running enhanced market simulation...")
        
        class EnhancedMarketSimulator(MarketSimulator):
            def __init__(self, predictor, data):
                super().__init__(predictor, data)
            
            def run_simulation(self, start_date=None, end_date=None, prediction_interval_days=1):
                logger.info("Starting enhanced market simulation...")
                
                if start_date is None:
                    start_idx = int(len(self.data) * 0.8)
                    start_date = self.data.iloc[start_idx]['timestamp']
                
                if end_date is None:
                    end_date = self.data.iloc[-2]['timestamp']
                
                logger.info(f"Simulation period: {start_date} to {end_date}")
                
                start_idx = self.data[self.data['timestamp'] >= start_date].index[0]
                end_idx = self.data[self.data['timestamp'] <= end_date].index[-1]
                
                predictions_made = 0
                correct_predictions = 0
                
                for idx in range(start_idx, end_idx, prediction_interval_days):
                    current_time = self.data.iloc[idx]['timestamp']
                    
                    try:
                        prediction = self.predictor.predict_direction_ensemble(self.data, current_time)
                        
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
                            logger.info(f"Made {predictions_made} enhanced predictions...")
                        
                    except Exception as e:
                        logger.warning(f"Enhanced prediction failed at {current_time}: {str(e)}")
                        continue
                
                accuracy = correct_predictions / predictions_made if predictions_made > 0 else 0
                
                return {
                    'total_predictions': predictions_made,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy,
                    'simulation_period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'results': self.simulation_results
                }
        
        simulator = EnhancedMarketSimulator(predictor, df)
        simulation_start = df.iloc[train_size]['timestamp']
        simulation_results = simulator.run_simulation(start_date=simulation_start)
        
        logger.info(f" Enhanced simulation completed - Accuracy: {simulation_results['accuracy']:.3f}")
        
        logger.info("5. Performance comparison...")
        performance_metrics = simulator.get_performance_metrics()
        
        logger.info("=" * 60)
        logger.info(" ENHANCED RESULTS")
        logger.info("=" * 60)
        logger.info(f" Dataset: {len(df)} days of Bitcoin data")
        logger.info(f" Model: Enhanced Ensemble (RF + GB + LR)")
        logger.info(f" Predictions Made: {simulation_results['total_predictions']}")
        logger.info(f" Correct Predictions: {simulation_results['correct_predictions']}")
        logger.info(f" Enhanced Accuracy: {performance_metrics['overall_accuracy']:.1%}")
        logger.info(f" Random Baseline: 50.0%")
        logger.info(f" Improvement: {(performance_metrics['overall_accuracy'] - 0.5) * 100:.1f} percentage points")
        logger.info("=" * 60)
        
        logger.info("6. Creating enhanced visualizations...")
        os.makedirs('output', exist_ok=True)
        
        viz = StockVisualization(output_dir='output')
        results_df = simulator.get_results_dataframe()
        
        dashboard = viz.create_performance_dashboard(df, results_df, performance_metrics)
        logger.info(f" Enhanced dashboard: {dashboard}")
        
        if hasattr(predictor, 'feature_importance') and predictor.feature_importance:
            logger.info("7. Top predictive features:")
            top_features = sorted(predictor.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(top_features, 1):
                logger.info(f"   {i:2d}. {feature}: {importance:.4f}")
        
        final_accuracy = performance_metrics['overall_accuracy']
        if final_accuracy > 0.6:
            logger.info(" EXCELLENT! Enhanced model shows strong predictive power!")
        elif final_accuracy > 0.55:
            logger.info(" GOOD! Enhanced model shows meaningful improvement!")
        else:
            logger.info("  Enhanced model needs further tuning")
        
        return True
        
    except Exception as e:
        logger.error(f" Enhanced test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_system()
    exit(0 if success else 1)