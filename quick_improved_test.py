
import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_models import ModelConfig
from data_ingestion import DataIngestionPipeline
from download_bitcoin_data import main as download_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_better_features(df):
    
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    
    for period in [1, 3, 5, 10]:
        df[f'return_{period}d'] = df['close_price'].pct_change(periods=period)
        df[f'momentum_{period}d'] = df['close_price'] / df['close_price'].shift(period) - 1
    
    for window in [5, 10, 20]:
        df[f'ma_{window}'] = df['close_price'].rolling(window=window).mean()
        df[f'price_above_ma_{window}'] = (df['close_price'] > df[f'ma_{window}']).astype(int)
    
    df['volatility_5d'] = df['return_1d'].rolling(5).std()
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    
    df['higher_high'] = (df['high_price'] > df['high_price'].shift(1)).astype(int)
    df['lower_low'] = (df['low_price'] < df['low_price'].shift(1)).astype(int)
    
    df['uptrend_5d'] = (df['close_price'] > df['close_price'].shift(5)).astype(int)
    df['uptrend_10d'] = (df['close_price'] > df['close_price'].shift(10)).astype(int)
    
    df['target'] = (df['close_price'].shift(-1) > df['close_price']).astype(int)
    
    return df

def test_improved_model():
    
    logger.info("=== IMPROVED MODEL TEST ===")
    
    data_file = download_data()
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} days of data")
    
    df_features = create_better_features(df)
    
    feature_cols = [col for col in df_features.columns 
                   if col not in ['timestamp', 'symbol', 'target', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
                   and not col.startswith('ma_')]  # Exclude raw MA values
    
    df_clean = df_features[feature_cols + ['target', 'timestamp']].dropna()
    logger.info(f"Clean data: {len(df_clean)} rows, {len(feature_cols)} features")
    
    split_idx = int(len(df_clean) * 0.7)  # Use 70% for training
    
    X_train = df_clean[feature_cols].iloc[:split_idx].values
    y_train = df_clean['target'].iloc[:split_idx].values
    X_test = df_clean[feature_cols].iloc[split_idx:].values
    y_test = df_clean['target'].iloc[split_idx:].values
    
    logger.info(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    logger.info(f"Train target distribution: {np.bincount(y_train)}")
    logger.info(f"Test target distribution: {np.bincount(y_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=300,  # More trees
        max_depth=8,       # Prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    high_conf_mask = (y_pred_proba > 0.7) | (y_pred_proba < 0.3)
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
    else:
        high_conf_accuracy = 0
    
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    logger.info("=" * 60)
    logger.info(" IMPROVED MODEL RESULTS")
    logger.info("=" * 60)
    logger.info(f" Training Accuracy: {train_accuracy:.1%}")
    logger.info(f" Test Accuracy: {test_accuracy:.1%}")
    logger.info(f" High Confidence Accuracy: {high_conf_accuracy:.1%}")
    logger.info(f" Improvement over random: {(test_accuracy - 0.5) * 100:.1f} percentage points")
    logger.info("=" * 60)
    
    logger.info(" Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"   {i:2d}. {feature}: {importance:.4f}")
    
    if test_accuracy > 0.65:
        logger.info(" EXCELLENT! Model shows strong predictive power!")
    elif test_accuracy > 0.58:
        logger.info(" GOOD! Model shows meaningful improvement!")
    elif test_accuracy > 0.52:
        logger.info(" DECENT! Model beats random chance!")
    else:
        logger.info("  Model needs more work")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = test_improved_model()
    logger.info(f"\nFinal accuracy: {accuracy:.1%}")
    exit(0)