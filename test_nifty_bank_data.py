
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

def create_nifty_bank_data():
    
    data = [
        ["30-JAN-2025", 49206.2, 49426.2, 49031.8, 49311.95, 197906088, 8567.72],
        ["29-JAN-2025", 48997.2, 49199.65, 48849.8, 49165.95, 177169221, 5952.79],
        ["28-JAN-2025", 48642.5, 49247.15, 48449.05, 48866.85, 299944475, 12208.48],
        ["27-JAN-2025", 47881.65, 48319.2, 47844.15, 48064.65, 271808231, 8969.36],
        ["24-JAN-2025", 48546.05, 48858.65, 48203, 48367.8, 139948431, 7505.19],
        ["23-JAN-2025", 48770.15, 48892.7, 48493, 48589, 124573638, 8807.77],
        ["22-JAN-2025", 48689.55, 48781.75, 48074.05, 48724.4, 151683951, 8347.43],
        ["21-JAN-2025", 49532, 49543.15, 48430.95, 48570.9, 134686328, 9193.77],
        ["20-JAN-2025", 48834.15, 49650.6, 48683.6, 49350.8, 173115109, 10519.88],
        ["17-JAN-2025", 48959.5, 49047.2, 48309.5, 48540.6, 130598508, 8620.64],
        ["16-JAN-2025", 49082.9, 49459, 49038.45, 49278.7, 146732136, 7742.3],
        ["15-JAN-2025", 48832.75, 49083.65, 48522.4, 48751.7, 145112403, 5702.11],
        ["14-JAN-2025", 48266.9, 49007.35, 48235.2, 48729.15, 151923269, 7745.76],
        ["13-JAN-2025", 48264.25, 48606.35, 47898.35, 48041.25, 168780899, 7299.58],
        ["10-JAN-2025", 49426.5, 49483.15, 48631.2, 48734.15, 142386637, 6354.11],
        ["09-JAN-2025", 49712.55, 49798.1, 49230.15, 49503.5, 136379631, 8842.81],
        ["08-JAN-2025", 50201.75, 50246.9, 49389.75, 49835.05, 131178850, 6950.22],
        ["07-JAN-2025", 50661.2, 50447.6, 49969.3, 50202.15, 103825140, 5397.67],
        ["06-JAN-2025", 50990.65, 51026.1, 49751, 49922, 184590426, 7006.9],
        ["03-JAN-2025", 51567.15, 51671.6, 50904.35, 50988.8, 148238026, 8040.74],
        ["02-JAN-2025", 51084.95, 51672.75, 50992.8, 51605.55, 134402813, 6772.81],
        ["01-JAN-2025", 50841.9, 51321.95, 50485.05, 51060.6, 81129924, 3485.94],
        ["31-DEC-2024", 50648.2, 50945.55, 50599.8, 50860.2, 105274010, 5698.85],
        ["30-DEC-2024", 51255.35, 51979.75, 50718.35, 50952.75, 316292024, 11753.32],
        ["27-DEC-2024", 51268.2, 51628.45, 51240.1, 51311.3, 78171919, 4120.89],
        ["26-DEC-2024", 51395.8, 51740, 50951.8, 51170.7, 91270844, 4498.69],
        ["24-DEC-2024", 51314.95, 51382.1, 51137.5, 51233, 112046629, 5487.6],
        ["23-DEC-2024", 51044.4, 51417.35, 51030.4, 51317.6, 110847696, 4510.06],
        ["20-DEC-2024", 51401.35, 51629, 50609.35, 50759.2, 228020799, 16375.69],
        ["19-DEC-2024", 51428.45, 51789.85, 51263.75, 51575.7, 145941694, 8027.55],
        ["18-DEC-2024", 52696.95, 52827.6, 52010.65, 52139.55, 149536630, 6980.26],
        ["17-DEC-2024", 53394.1, 53515.7, 52709.4, 52834.8, 116219553, 6731.56],
        ["16-DEC-2024", 53502.5, 53738.9, 53335, 53581.35, 85755127, 4980.12],
        ["13-DEC-2024", 53109.8, 53654, 52264.55, 53583.8, 139912483, 7776.14],
        ["12-DEC-2024", 53201, 53537.45, 53174.4, 53216.45, 99473861, 6140.28],
        ["11-DEC-2024", 53493.9, 53648.05, 53302.15, 53391.35, 101567795, 5690.56],
        ["10-DEC-2024", 53450.05, 53624.05, 53302.65, 53577.7, 121836590, 5376.71],
        ["09-DEC-2024", 53380.75, 53775.1, 53326.4, 53407.75, 121821865, 6989.26],
        ["06-DEC-2024", 53634.2, 53868.5, 53160.65, 53509.5, 209369122, 8276.79],
        ["05-DEC-2024", 53354.45, 53888.3, 52850.35, 53603.55, 155605811, 11440.43],
        ["04-DEC-2024", 52775, 53387.1, 52685.15, 53266.9, 228584540, 11813.29],
        ["03-DEC-2024", 52357.95, 52780.9, 52216.85, 52695.75, 173561594, 10201.25],
        ["02-DEC-2024", 52087.65, 52197.25, 51693.95, 52109, 102500670, 5850.86],
        ["29-NOV-2024", 51984.15, 52170.9, 51759.45, 52055.6, 139625360, 7754.53],
        ["28-NOV-2024", 52389.95, 52760.2, 51782.9, 51906.85, 172034869, 11035.07],
        ["27-NOV-2024", 52154.3, 52444.35, 52019.65, 52301.8, 151590455, 9483.92],
        ["26-NOV-2024", 52554.9, 52555.5, 51999.75, 52191.5, 113118684, 6692.01],
        ["25-NOV-2024", 52046.35, 52331.1, 51774.05, 52207.5, 458462507, 48724.25],
        ["22-NOV-2024", 50512.8, 51271.5, 50508.25, 51135.4, 183377690, 9726.6],
        ["21-NOV-2024", 50625, 50652.15, 49787.1, 50372.9, 254623745, 11059.22],
        ["19-NOV-2024", 50580.55, 50983.5, 50440.85, 50626.5, 185755396, 11808.91],
        ["18-NOV-2024", 50312.45, 50445.8, 50074, 50363.8, 161525597, 7291.66],
        ["14-NOV-2024", 50053.45, 50561.8, 49939.35, 50179.55, 133887043, 6834.4],
        ["13-NOV-2024", 51030.95, 51353.5, 49904.4, 50088.35, 207050329, 9648.29],
        ["12-NOV-2024", 52053.75, 52169.05, 51006.85, 51157.8, 131415636, 8256.31],
        ["11-NOV-2024", 51562.7, 52177.7, 51294.2, 51876.75, 160694052, 7576.85],
        ["08-NOV-2024", 51869.15, 52007.15, 51494, 51561.2, 157226476, 8047.16],
        ["07-NOV-2024", 52558.95, 52377.25, 51752.25, 51916.5, 160476603, 7418.76],
        ["06-NOV-2024", 52440.4, 52493.95, 52185.4, 52317.4, 219097414, 10051.57],
        ["05-NOV-2024", 51052.6, 52299.55, 50865.45, 52207.25, 244958392, 10746.52],
        ["04-NOV-2024", 51764.5, 51784.5, 51066.8, 51215.25, 320472374, 10675.25],
        ["01-NOV-2024", 51550.15, 51825.5, 51459.4, 51673.9, 51350377, 1023.61],
        ["31-OCT-2024", 51649.45, 52005.6, 51318.1, 51475.35, 319910762, 10626.73],
        ["30-OCT-2024", 51988.7, 52220, 51733, 51807.5, 247658224, 11307.58],
        ["29-OCT-2024", 51404.1, 52354.85, 51278.9, 52320.7, 374629247, 14536.44],
        ["28-OCT-2024", 51061.35, 51589.15, 51012.55, 51259.3, 447571902, 12761.66],
    ]
    
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Shares_Traded', 'Turnover'])
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Shares_Traded', 'Turnover']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.rename(columns={
        'Date': 'timestamp',
        'Shares_Traded': 'Volume'
    })
    
    logger.info(f"Created NIFTY Bank dataset: {len(df)} records")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Price range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
    
    return df

def train_model(data):
    
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["Volatility"] = data["Return"].rolling(5).std()
    
    data["MA_Ratio"] = data["MA5"] / data["MA10"]  # MA crossover signal
    data["Price_MA5_Ratio"] = data["Close"] / data["MA5"]  # Price relative to MA
    data["Price_MA10_Ratio"] = data["Close"] / data["MA10"]
    data["Volume_MA"] = data["Volume"].rolling(5).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]
    
    data["Return_2d"] = data["Close"].pct_change(2)
    data["Return_3d"] = data["Close"].pct_change(3)
    
    data["High_Low_Ratio"] = data["High"] / data["Low"]
    data["Close_High_Ratio"] = data["Close"] / data["High"]
    
    data.dropna(inplace=True)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    data = data[:-1]
    
    features = [
        "Return", "MA5", "MA10", "Volatility", 
        "MA_Ratio", "Price_MA5_Ratio", "Price_MA10_Ratio",
        "Volume_Ratio", "Return_2d", "Return_3d",
        "High_Low_Ratio", "Close_High_Ratio"
    ]
    
    X = data[features]
    y = data["Target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler, X_train, X_test, y_train, y_test, features, data

def main():
    
    logger.info("=== NIFTY BANK DATA TEST ===")
    
    data = create_nifty_bank_data()
    
    logger.info("Training model on NIFTY Bank data...")
    model, scaler, X_train, X_test, y_train, y_test, features, processed_data = train_model(data.copy())
    
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
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
    
    feature_importance = dict(zip(features, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("=" * 60)
    logger.info(" NIFTY BANK RESULTS")
    logger.info("=" * 60)
    logger.info(f" Training samples: {len(X_train)}")
    logger.info(f" Test samples: {len(X_test)}")
    logger.info(f" Training Accuracy: {train_accuracy:.1%}")
    logger.info(f" Test Accuracy: {test_accuracy:.1%}")
    logger.info(f" High Confidence Accuracy: {high_conf_accuracy:.1%} ({high_conf_mask.sum()} predictions)")
    logger.info(f" Up Predictions Accuracy: {up_accuracy:.1%}")
    logger.info(f" Down Predictions Accuracy: {down_accuracy:.1%}")
    logger.info(f" Improvement over random: {(test_accuracy - 0.5) * 100:.1f} percentage points")
    logger.info("=" * 60)
    
    logger.info(" Feature Importance:")
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"   {i:2d}. {feature}: {importance:.4f}")
    
    os.makedirs('output', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    test_data = processed_data.iloc[-len(y_test):]
    plt.plot(test_data['Close'], alpha=0.7, label='NIFTY Bank Price')
    
    correct_mask = y_pred == y_test
    for i, correct in enumerate(correct_mask):
        color = 'green' if correct else 'red'
        marker = '^' if y_pred[i] == 1 else 'v'
        plt.scatter(i, test_data['Close'].iloc[i], color=color, marker=marker, s=30, alpha=0.8)
    
    plt.title('NIFTY Bank Price with Predictions')
    plt.ylabel('Price (₹)')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    cumulative_accuracy = np.cumsum(correct_mask) / np.arange(1, len(correct_mask) + 1)
    plt.plot(cumulative_accuracy, linewidth=2)
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
    plt.hist(y_pred_proba, bins=15, alpha=0.7, color='blue')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('output/nifty_bank_results.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved to output/nifty_bank_results.png")
    
    if test_accuracy > 0.65:
        logger.info(" EXCELLENT! Model shows strong predictive power on NIFTY Bank!")
    elif test_accuracy > 0.58:
        logger.info(" GOOD! Model shows meaningful improvement on NIFTY Bank!")
    elif test_accuracy > 0.52:
        logger.info(" DECENT! Model beats random chance on NIFTY Bank!")
    else:
        logger.info("  Model performance is close to random on NIFTY Bank")
    
    logger.info("\n Real NIFTY Bank data often shows better patterns than crypto!")
    logger.info(" Indian banking sector has more predictable trends!")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = main()
    exit(0)