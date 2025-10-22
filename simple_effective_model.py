
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import sys
import os

sys.path.append('src')
from download_bitcoin_data import main as download_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data):
    
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["Volatility"] = data["Return"].rolling(5).std()
    
    data["MA_Ratio"] = data["MA5"] / data["MA10"]  # MA crossover signal
    data["Price_MA5_Ratio"] = data["Close"] / data["MA5"]  # Price relative to MA
    data["Volume_MA"] = data["Volume"].rolling(5).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]
    
    data.dropna(inplace=True)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    data = data[:-1]
    
    features = ["Return", "MA5", "MA10", "Volatility", "MA_Ratio", "Price_MA5_Ratio", "Volume_Ratio"]
    X = data[features]
    y = data["Target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler, X_train, X_test, y_train, y_test, features

def evaluate_model(model, X_test, y_test):
    
    accuracy = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    high_conf_mask = (y_pred_proba > 0.7) | (y_pred_proba < 0.3)
    high_conf_accuracy = 0
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
    
    up_mask = y_pred == 1
    down_mask = y_pred == 0
    
    up_accuracy = 0
    down_accuracy = 0
    if up_mask.sum() > 0:
        up_accuracy = (y_pred[up_mask] == y_test[up_mask]).mean()
    if down_mask.sum() > 0:
        down_accuracy = (y_pred[down_mask] == y_test[down_mask]).mean()
    
    return {
        'accuracy': accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'high_conf_count': high_conf_mask.sum()
    }

def create_visualization(data, y_test, y_pred, y_pred_proba):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    test_data = data.iloc[-len(y_test):]
    ax1.plot(test_data.index, test_data['Close'], label='Price', alpha=0.7)
    
    correct_mask = y_pred == y_test
    for i, (idx, correct) in enumerate(zip(test_data.index, correct_mask)):
        color = 'green' if correct else 'red'
        marker = '^' if y_pred[i] == 1 else 'v'
        ax1.scatter(idx, test_data['Close'].iloc[i], color=color, marker=marker, s=30, alpha=0.8)
    
    ax1.set_title('Price with Predictions (Green=Correct, Red=Wrong)')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    cumulative_accuracy = np.cumsum(correct_mask) / np.arange(1, len(correct_mask) + 1)
    ax2.plot(cumulative_accuracy)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax2.set_title('Cumulative Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    ax3.hist(y_pred_proba, bins=20, alpha=0.7, color='blue')
    ax3.set_title('Prediction Confidence Distribution')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    
    conf_bins = np.linspace(0, 1, 6)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(conf_bins) - 1):
        mask = (y_pred_proba >= conf_bins[i]) & (y_pred_proba < conf_bins[i + 1])
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_test[mask]).mean()
            bin_accuracies.append(acc)
            bin_centers.append((conf_bins[i] + conf_bins[i + 1]) / 2)
    
    ax4.bar(bin_centers, bin_accuracies, width=0.15, alpha=0.7, color='orange')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax4.set_title('Accuracy by Confidence Level')
    ax4.set_xlabel('Confidence Level')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('output/simple_model_results.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved to output/simple_model_results.png")
    
    return 'output/simple_model_results.png'

def main():
    
    logger.info("=== SIMPLE EFFECTIVE MODEL TEST ===")
    
    data_file = download_data()
    data = pd.read_csv(data_file)
    
    data = data.rename(columns={
        'close_price': 'Close',
        'volume': 'Volume'
    })
    
    logger.info(f"Loaded {len(data)} days of Bitcoin data")
    logger.info(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    logger.info("Training simple model...")
    model, scaler, X_train, X_test, y_train, y_test, features = train_model(data.copy())
    
    logger.info("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    feature_importance = dict(zip(features, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("=" * 60)
    logger.info(" SIMPLE MODEL RESULTS")
    logger.info("=" * 60)
    logger.info(f" Training samples: {len(X_train)}")
    logger.info(f" Test samples: {len(X_test)}")
    logger.info(f" Overall Accuracy: {results['accuracy']:.1%}")
    logger.info(f" High Confidence Accuracy: {results['high_conf_accuracy']:.1%} ({results['high_conf_count']} predictions)")
    logger.info(f" Up Predictions Accuracy: {results['up_accuracy']:.1%}")
    logger.info(f" Down Predictions Accuracy: {results['down_accuracy']:.1%}")
    logger.info(f" Improvement over random: {(results['accuracy'] - 0.5) * 100:.1f} percentage points")
    logger.info("=" * 60)
    
    logger.info(" Feature Importance:")
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"   {i}. {feature}: {importance:.4f}")
    
    os.makedirs('output', exist_ok=True)
    viz_path = create_visualization(data, y_test, results['predictions'], results['probabilities'])
    
    accuracy = results['accuracy']
    if accuracy > 0.60:
        logger.info(" EXCELLENT! Model shows strong predictive power!")
    elif accuracy > 0.55:
        logger.info(" GOOD! Model shows meaningful improvement!")
    elif accuracy > 0.52:
        logger.info(" DECENT! Model beats random chance!")
    else:
        logger.info("  Model performance is close to random")
    
    logger.info(f"\n Visualization saved: {viz_path}")
    logger.info(" Simple approach with clean features often works best!")
    
    return results['accuracy']

if __name__ == "__main__":
    accuracy = main()
    exit(0)