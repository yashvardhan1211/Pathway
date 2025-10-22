
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import ta  # Technical Analysis library

from data_models import Prediction, ModelConfig
from temporal_validator import TemporalValidator

logger = logging.getLogger(__name__)

class EnhancedAIPredictor:
    
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.validator = TemporalValidator()
        self.is_trained = False
        self.feature_importance = None
        
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.debug("Creating advanced features...")
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['price_change'] = df['close_price'].pct_change()
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['open_close_ratio'] = df['open_price'] / df['close_price']
        df['body_size'] = abs(df['close_price'] - df['open_price']) / df['close_price']
        df['upper_shadow'] = (df['high_price'] - np.maximum(df['open_price'], df['close_price'])) / df['close_price']
        df['lower_shadow'] = (np.minimum(df['open_price'], df['close_price']) - df['low_price']) / df['close_price']
        
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close_price'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close_price'].ewm(span=window).mean()
            df[f'price_to_sma_{window}'] = df['close_price'] / df[f'sma_{window}']
            df[f'price_to_ema_{window}'] = df['close_price'] / df[f'ema_{window}']
        
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(50).mean()
        
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['price_volume'] = df['close_price'] * df['volume']
        df['vwap_5'] = (df['price_volume'].rolling(5).sum() / df['volume'].rolling(5).sum())
        df['price_to_vwap'] = df['close_price'] / df['vwap_5']
        
        try:
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close_price'], window=14).rsi()
            df['rsi_7'] = ta.momentum.RSIIndicator(df['close_price'], window=7).rsi()
            
            macd = ta.trend.MACD(df['close_price'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            bb = ta.volatility.BollingerBands(df['close_price'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            stoch = ta.momentum.StochasticOscillator(df['high_price'], df['low_price'], df['close_price'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high_price'], df['low_price'], df['close_price']).williams_r()
            
            df['atr'] = ta.volatility.AverageTrueRange(df['high_price'], df['low_price'], df['close_price']).average_true_range()
            df['atr_ratio'] = df['atr'] / df['close_price']
            
        except Exception as e:
            logger.warning(f"Error creating TA indicators: {e}")
        
        for period in [3, 7, 14]:
            df[f'momentum_{period}'] = df['close_price'] / df['close_price'].shift(period) - 1
            df[f'roc_{period}'] = df['close_price'].pct_change(periods=period)
        
        df['high_20'] = df['high_price'].rolling(20).max()
        df['low_20'] = df['low_price'].rolling(20).min()
        df['resistance_distance'] = (df['high_20'] - df['close_price']) / df['close_price']
        df['support_distance'] = (df['close_price'] - df['low_20']) / df['close_price']
        
        df['higher_high'] = (df['high_price'] > df['high_price'].shift(1)).astype(int)
        df['lower_low'] = (df['low_price'] < df['low_price'].shift(1)).astype(int)
        df['inside_bar'] = ((df['high_price'] <= df['high_price'].shift(1)) & 
                           (df['low_price'] >= df['low_price'].shift(1))).astype(int)
        
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        df['next_close'] = df['close_price'].shift(-1)
        df['target'] = (df['next_close'] > df['close_price']).astype(int)
        
        df['target_3d'] = (df['close_price'].shift(-3) > df['close_price']).astype(int)
        df['target_5d'] = (df['close_price'].shift(-5) > df['close_price']).astype(int)
        
        return df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str = 'target') -> List[str]:
        
        
        exclude_cols = ['timestamp', 'symbol', 'next_close', 'target', 'target_3d', 'target_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        feature_cols = [col for col in feature_cols if df[col].isna().sum() / len(df) < 0.3]
        
        logger.info(f"Selected {len(feature_cols)} features for training")
        return feature_cols
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        
        logger.info("Preparing enhanced training data...")
        
        df_features = self.create_advanced_features(df)
        
        feature_cols = self.select_best_features(df_features)
        
        df_clean = df_features[feature_cols + ['target']].dropna()
        
        if len(df_clean) < 100:
            raise ValueError(f"Not enough clean data for training (got {len(df_clean)}, need at least 100)")
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        logger.info(f"Enhanced training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def train_ensemble(self, df: pd.DataFrame) -> dict:
        
        logger.info("Training enhanced ensemble model...")
        
        X, y, feature_cols = self.prepare_training_data(df)
        self.feature_cols = feature_cols
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_scores = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            model_scores[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score
            }
            
            logger.info(f"{name}: train={train_score:.3f}, test={test_score:.3f}")
        
        if hasattr(self.models['rf'], 'feature_importances_'):
            self.feature_importance = dict(zip(feature_cols, self.models['rf'].feature_importances_))
            
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("Top 10 most important features:")
            for feat, importance in top_features:
                logger.info(f"  {feat}: {importance:.4f}")
        
        self.is_trained = True
        
        best_model = max(model_scores.items(), key=lambda x: x[1]['test_accuracy'])
        
        training_results = {
            "model_type": "ensemble",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "best_model": best_model[0],
            "best_test_accuracy": best_model[1]['test_accuracy'],
            "model_scores": model_scores,
            "features_used": len(feature_cols),
            "top_features": top_features[:5] if self.feature_importance else []
        }
        
        logger.info(f"Ensemble training completed. Best model: {best_model[0]} ({best_model[1]['test_accuracy']:.3f})")
        return training_results
    
    def predict_direction_ensemble(self, df: pd.DataFrame, prediction_time: datetime) -> Prediction:
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        valid_data = self.validator.get_valid_data_window(df, prediction_time)
        
        if len(valid_data) < self.config.lookback_window:
            raise ValueError(f"Not enough historical data for prediction")
        
        recent_data = valid_data.tail(self.config.lookback_window)
        
        df_features = self.create_advanced_features(recent_data)
        
        last_row = df_features.iloc[-1]
        
        X = last_row[self.feature_cols].values.reshape(1, -1)
        
        X = np.nan_to_num(X, nan=0.0)
        
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            
            predictions[name] = pred_class
            probabilities[name] = pred_proba[1]  # Probability of "up"
        
        up_votes = sum(1 for pred in predictions.values() if pred == 1)
        total_votes = len(predictions)
        
        avg_probability = np.mean(list(probabilities.values()))
        
        ensemble_direction = "up" if up_votes > total_votes / 2 else "down"
        ensemble_confidence = max(avg_probability, 1 - avg_probability)
        
        prediction = Prediction(
            timestamp=prediction_time,
            symbol=recent_data['symbol'].iloc[-1],
            predicted_direction=ensemble_direction,
            confidence=ensemble_confidence
        )
        
        logger.debug(f"Ensemble prediction at {prediction_time}: {ensemble_direction} "
                    f"(confidence: {ensemble_confidence:.3f}, votes: {up_votes}/{total_votes})")
        
        return prediction