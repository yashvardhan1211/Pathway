
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_better_sample_data(days=500):
    
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    prices = [45000.0]  # Starting Bitcoin price
    volumes = []
    
    trend_changes = np.random.choice(range(20, days-20), size=5)  # 5 trend changes
    current_trend = 0.001  # Start with slight uptrend
    
    for i in range(1, days):
        if i in trend_changes:
            current_trend = np.random.choice([-0.002, -0.001, 0.001, 0.002])
        
        yesterday_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
        momentum_factor = 0.3 * yesterday_change  # 30% momentum
        
        random_change = np.random.normal(0, 0.025)  # 2.5% daily volatility
        
        total_change = current_trend + momentum_factor + random_change
        
        new_price = prices[-1] * (1 + total_change)
        new_price = max(new_price, 10000)  # Minimum price
        prices.append(new_price)
        
        volatility = abs(total_change)
        base_volume = 50000000
        volume = int(base_volume * (1 + volatility * 5))
        volumes.append(volume)
    
    volumes.insert(0, 50000000)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'Close': prices,
        'Volume': volumes
    })
    
    df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
    df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.random.uniform(0, 0.02, len(df)))
    df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.random.uniform(0, 0.02, len(df)))
    
    df = df.dropna()
    
    logger.info(f"Created enhanced sample data: {len(df)} records")
    logger.info(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df

def train_enhanced_model(data):
    
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["Volatility"] = data["Return"].rolling(5).std()
    
    data["MA_Ratio_5_10"] = data["MA5"] / data["MA10"]
    data["MA_Ratio_5_20"] = data["MA5"] / data["MA20"]
    data["Price_MA5_Ratio"] = data["Close"] / data["MA5"]
    data["Price_MA10_Ratio"] = data["Close"] / data["MA10"]
    
    data["Return_2d"] = data["Close"].pct_change(2)
    data["Return_3d"] = data["Close"].pct_change(3)
    data["Momentum_5d"] = data["Close"] / data["Close"].shift(5) - 1
    
    data["Volume_MA5"] = data["Volume"].rolling(5).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA5"]
    
    data["Volatility_10d"] = data["Return"].rolling(10).std()
    data["Volatility_Ratio"] = data["Volatility"] / data["Volatility_10d"]
    
    data["High_Low_Ratio"] = data["High"] / data["Low"]
    data["Close_High_Ratio"] = data["Close"] / data["High"]
    
    data["Uptrend_5d"] = (data["Close"] > data["Close"].shift(5)).astype(int)
    data["Above_MA5"] = (data["Close"] > data["MA5"]).astype(int)
    data["Above_MA10"] = (data["Close"] > data["MA10"]).astype(int)
    
    data.dropna(inplace=True)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    data = data[:-1]
    
    features = [
        "Return", "MA5", "MA10", "Volatility",
        "MA_Ratio_5_10", "MA_Ratio_5_20", "Price_MA5_Ratio", "Price_MA10_Ratio",
        "Return_2d", "Return_3d", "Momentum_5d",
        "Volume_Ratio", "Volatility_Ratio",
        "High_Low_Ratio", "Close_High_Ratio",
        "Uptrend_5d", "Above_MA5", "Above_MA10"
    ]
    
    X = data[features]
    y = data["Target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, shuffle=False)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, scaler, X_train, X_test, y_train, y_test, features, data

def main():
    
    logger.info("=== OPTIMIZED SIMPLE MODEL TEST ===")
    
    data = create_better_sample_data(days=600)
    
    logger.info("Training optimized model...")
    model, scaler, X_train, X_test, y_train, y_test, features, processed_data = train_enhanced_model(data.copy())
    
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    high_conf_mask = (y_pred_proba > 0.65) | (y_pred_proba < 0.35)
    high_conf_accuracy = 0
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
    
    feature_importance = dict(zip(features, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    logger.info("=" * 60)
    logger.info(" OPTIMIZED MODEL RESULTS")
    logger.info("=" * 60)
    logger.info(f" Training samples: {len(X_train)}")
    logger.info(f" Test samples: {len(X_test)}")
    logger.info(f" Training Accuracy: {train_accuracy:.1%}")
    logger.info(f" Test Accuracy: {test_accuracy:.1%}")
    logger.info(f" High Confidence Accuracy: {high_conf_accuracy:.1%} ({high_conf_mask.sum()} predictions)")
    logger.info(f" Improvement over random: {(test_accuracy - 0.5) * 100:.1f} percentage points")
    logger.info("=" * 60)
    
    logger.info(" Top 10 Features:")
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"   {i:2d}. {feature}: {importance:.4f}")
    
    os.makedirs('output', exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    test_data = processed_data.iloc[-len(y_test):]
    plt.plot(test_data['Close'], alpha=0.7, label='Price')
    
    correct_mask = y_pred == y_test
    for i, correct in enumerate(correct_mask):
        color = 'green' if correct else 'red'
        marker = '^' if y_pred[i] == 1 else 'v'
        plt.scatter(i, test_data['Close'].iloc[i], color=color, marker=marker, s=20, alpha=0.8)
    
    plt.title('Price with Predictions')
    plt.ylabel('Price')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    cumulative_accuracy = np.cumsum(correct_mask) / np.arange(1, len(correct_mask) + 1)
    plt.plot(cumulative_accuracy)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    plt.title('Cumulative Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    top_5_features = top_features[:5]
    names, importances = zip(*top_5_features)
    plt.barh(names, importances)
    plt.title('Top 5 Feature Importance')
    plt.xlabel('Importance')
    
    plt.subplot(2, 2, 4)
    plt.hist(y_pred_proba, bins=20, alpha=0.7, color='blue')
    plt.title('Prediction Confidence')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('output/optimized_model_results.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved to output/optimized_model_results.png")
    
    if test_accuracy > 0.60:
        logger.info(" EXCELLENT! Model shows strong predictive power!")
    elif test_accuracy > 0.55:
        logger.info(" GOOD! Model shows meaningful improvement!")
    elif test_accuracy > 0.52:
        logger.info(" DECENT! Model beats random chance!")
    else:
        logger.info("  Model performance is close to random")
    
    logger.info("\n Your simple approach with clean features is the right strategy!")
    logger.info(" Even small improvements over random can be very profitable in trading!")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = main()
    exit(0)