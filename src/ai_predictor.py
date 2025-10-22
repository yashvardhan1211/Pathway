
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from data_models import Prediction, ModelConfig
from temporal_validator import TemporalValidator

logger = logging.getLogger(__name__)

class AIPredictor:
    
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.validator = TemporalValidator()
        self.is_trained = False
        
        if config.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif config.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.debug("Creating features...")
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['price_change'] = df['close_price'].pct_change()
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['open_close_ratio'] = df['open_price'] / df['close_price']
        
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['close_price'].rolling(window=window).mean()
            df[f'price_to_ma_{window}'] = df['close_price'] / df[f'ma_{window}']
        
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['next_close'] = df['close_price'].shift(-1)
        df['target'] = (df['next_close'] > df['close_price']).astype(int)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        logger.info("Preparing training data...")
        
        df_features = self.create_features(df)
        
        feature_cols = [col for col in df_features.columns 
                       if col not in ['timestamp', 'symbol', 'next_close', 'target'] 
                       and not col.startswith('ma_')]  # Exclude raw MA values
        
        df_clean = df_features[feature_cols + ['target']].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> dict:
        
        logger.info("Training AI model...")
        
        X, y = self.prepare_training_data(df)
        
        if len(X) < 50:
            raise ValueError("Not enough data for training (need at least 50 samples)")
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        training_results = {
            "model_type": self.config.model_type,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "features_used": X.shape[1]
        }
        
        logger.info(f"Training completed: {training_results}")
        return training_results
    
    def predict_direction(self, df: pd.DataFrame, prediction_time: datetime) -> Prediction:
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        valid_data = self.validator.get_valid_data_window(df, prediction_time)
        
        if len(valid_data) < self.config.lookback_window:
            raise ValueError(f"Not enough historical data for prediction "
                           f"(need {self.config.lookback_window}, got {len(valid_data)})")
        
        recent_data = valid_data.tail(self.config.lookback_window)
        
        df_features = self.create_features(recent_data)
        
        last_row = df_features.iloc[-1]
        
        feature_cols = [col for col in df_features.columns 
                       if col not in ['timestamp', 'symbol', 'next_close', 'target'] 
                       and not col.startswith('ma_')]
        
        X = last_row[feature_cols].values.reshape(1, -1)
        
        X = X.astype(float)
        if np.isnan(X).any():
            logger.warning("NaN values in features, using mean imputation")
            X = np.nan_to_num(X, nan=0.0)
        
        X_scaled = self.scaler.transform(X)
        
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        predicted_class = self.model.predict(X_scaled)[0]
        
        direction = "up" if predicted_class == 1 else "down"
        confidence = max(prediction_proba)  # Confidence is the max probability
        
        prediction = Prediction(
            timestamp=prediction_time,
            symbol=recent_data['symbol'].iloc[-1],
            predicted_direction=direction,
            confidence=confidence
        )
        
        logger.debug(f"Prediction at {prediction_time}: {direction} (confidence: {confidence:.3f})")
        
        return prediction
    
    def get_model_info(self) -> dict:
        
        return {
            "is_trained": self.is_trained,
            "model_type": self.config.model_type,
            "lookback_window": self.config.lookback_window,
            "prediction_horizon": self.config.prediction_horizon,
            "features": self.config.features
        }