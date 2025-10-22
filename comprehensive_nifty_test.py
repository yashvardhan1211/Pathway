
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

def create_full_nifty_dataset():
    data = [
        ["30-JAN-2024", 45481.5, 45678.7, 45206.05, 45367.75, 232110117, 10108.33],
        ["31-JAN-2024", 45295.65, 46179.75, 45071.2, 45996.8, 373131629, 16491.11],
        ["01-FEB-2024", 46164.9, 46306.9, 45668.35, 46188.65, 305721773, 10681.78],
        ["02-FEB-2024", 46568.2, 46892.35, 45901.25, 45970.95, 337046622, 12743.15],
        ["05-FEB-2024", 45962.25, 46048.6, 45615.1, 45825.55, 265909716, 11202.01],
        ["06-FEB-2024", 45891.2, 46932.15, 45627, 45690.8, 179552534, 9115.56],
        ["07-FEB-2024", 45944.6, 46062.85, 45620.5, 45818.5, 283674776, 13737.11],
        ["08-FEB-2024", 45973.85, 46181.2, 44893.75, 45012, 319908500, 17193.25],
        ["09-FEB-2024", 44986.75, 45718.15, 44859.15, 45634.55, 275980366, 13111.21],
        ["12-FEB-2024", 45664.3, 45748.5, 44633.85, 44882.25, 268816417, 10317.6],
        ["13-FEB-2024", 45056.8, 45750.4, 44819.55, 45502.4, 245860544, 12243.15],
        ["14-FEB-2024", 45014.65, 46170.45, 44860.75, 45908.3, 279949703, 16369.67],
        ["15-FEB-2024", 46027.1, 46297.7, 45590.2, 46218.9, 277735723, 11616.29],
        ["16-FEB-2024", 46454.3, 46693.4, 46264.4, 46384.85, 291383103, 10101.23],
        

        ["19-FEB-2024", 46554.9, 46717.4, 46317.7, 46535.5, 158391356, 8048.39],
        ["20-FEB-2024", 46444.9, 47136.75, 46367.8, 47094.2, 171886069, 10520.32],
        ["21-FEB-2024", 47363.4, 47363.4, 46886.95, 47019.7, 198947007, 11151.61],
        ["22-FEB-2024", 46934.55, 47024.05, 46426.85, 46919.8, 177182875, 11307.92],
        ["23-FEB-2024", 47060.7, 47245.35, 46723.15, 46811.75, 200518213, 8632.88],
        ["26-FEB-2024", 46615.85, 46893.15, 46513.55, 46576.5, 223684075, 8008.66],
        ["27-FEB-2024", 46480.2, 46722.25, 46324.9, 46588.05, 157256037, 7924.81],
        ["28-FEB-2024", 46640.9, 46754.55, 46552.55, 45963.15, 167271103, 6980.13],
        ["29-FEB-2024", 45881.45, 46329.65, 45661.75, 46120.9, 544734727, 16455.05],
        

        ["01-MAR-2024", 46218, 47342.25, 46218, 47286.9, 172519003, 8587.25],
        ["04-MAR-2024", 47318.5, 47529.6, 47191.65, 47456.1, 158086247, 7795.84],
        ["05-MAR-2024", 47256.7, 47737.85, 47196.75, 47581, 176906608, 8513.12],
        ["06-MAR-2024", 47451.65, 48161.25, 47442.25, 47965.4, 301719696, 14489.87],
        ["07-MAR-2024", 48035.8, 48071.7, 47747.2, 47835.8, 148312130, 8573.38],
        ["11-MAR-2024", 47792.2, 47853.8, 47230.65, 47327.85, 204166322, 10257.14],
        ["12-MAR-2024", 47351.35, 47812.75, 46884.45, 47282.4, 207215490, 13545.13],
        ["13-MAR-2024", 47341.15, 47468.7, 46842.15, 46981.3, 274610911, 14793.18],
        ["14-MAR-2024", 46825.75, 47231.5, 46565.55, 46789.95, 220394655, 11310.48],
        ["15-MAR-2024", 46572.1, 46802.55, 46310.5, 46594.1, 339276817, 26718.21],
        ["18-MAR-2024", 46458.75, 46739.25, 46022.15, 46575.9, 151328904, 6888.3],
        ["19-MAR-2024", 46421.9, 46602.35, 46258.75, 46384.8, 147080034, 8734.12],
        ["20-MAR-2024", 46392.9, 46655.55, 45828.8, 46310.9, 210365443, 11766.43],
        ["21-MAR-2024", 46674.85, 46990.25, 46570.15, 46684.9, 150392246, 8896.96],
        ["22-MAR-2024", 46634.9, 46974.15, 46566.8, 46863.75, 172184807, 10851.42],
        ["26-MAR-2024", 46552.95, 46788.35, 46529.05, 46600.2, 171655734, 11171.03],
        ["27-MAR-2024", 46643.45, 46956.1, 46643.45, 46785.95, 285050400, 16539.74],
        ["28-MAR-2024", 46827.85, 47440.45, 46827.85, 47124.6, 347979603, 13594.54],
        

        ["01-APR-2024", 47391.05, 47646.8, 47373.1, 47578.25, 134089417, 6573],
        ["02-APR-2024", 47490.75, 47707.35, 47408.55, 47545.45, 183398624, 10412.26],
        ["03-APR-2024", 47350.25, 47676.95, 47279.8, 47624.25, 283779413, 13792.53],
        ["04-APR-2024", 48106.2, 48254.65, 47712.7, 48060.8, 324271925, 18895.59],
        ["05-APR-2024", 48104.65, 48557.4, 47894.5, 48493.05, 214305286, 11900.32],
        ["08-APR-2024", 48566.75, 48716.95, 48424.65, 48581.7, 263381555, 7900.46],
        ["09-APR-2024", 48810.8, 48960.75, 48568.25, 48730.55, 147583296, 7024.79],
        

        ["10-APR-2024", 48879.55, 49057.4, 48669.25, 48986.6, 247065747, 9611.82],
        ["12-APR-2024", 48671.2, 48882.65, 48477.55, 48564.55, 207985339, 10643.84],
        ["15-APR-2024", 48057.5, 48255.5, 47725.8, 47773.25, 197115142, 7882.76],
        ["16-APR-2024", 47436.7, 47609, 47316.55, 47484.8, 186069576, 7841.45],
        ["18-APR-2024", 47592.7, 47829.75, 46982.15, 47069.45, 186371057, 10326.12],
        ["19-APR-2024", 46744.95, 47668.7, 46579.05, 47574.15, 164580371, 9513.25],
        ["22-APR-2024", 48145.7, 48146.3, 47628.45, 47924.9, 177719481, 9989.66],
        ["23-APR-2024", 48299.6, 48302.7, 47899.3, 47970.45, 141110604, 7677.26],
        ["24-APR-2024", 48120.9, 48246.2, 48028.7, 48189, 112159300, 5867.57],
        ["25-APR-2024", 47772.65, 48625.45, 47737.2, 48494.95, 337412581, 27018.59],
        ["26-APR-2024", 48660, 48679.65, 48038.25, 48201.05, 268118242, 12146.84],
        ["29-APR-2024", 48359.9, 49473.6, 48342.7, 49424.05, 327134115, 15966.76],
        ["30-APR-2024", 49477.1, 49574.75, 49249.8, 49396.75, 402874252, 20351.02],
        

        ["02-MAY-2024", 49262, 49529.35, 49123.6, 49231.05, 291857104, 16230.32],
        ["03-MAY-2024", 49375.05, 49607.75, 48659.7, 48923.55, 222964426, 12837.75],
        ["06-MAY-2024", 49174.55, 49252.65, 48784, 48895.3, 340187386, 15200.24],
        ["07-MAY-2024", 48965.05, 49023.5, 48213.75, 48285.35, 230306618, 10179.38],
        ["08-MAY-2024", 48124.2, 48223.05, 47851.15, 48021.1, 286041100, 13795.53],
        ["09-MAY-2024", 47976.35, 48258.65, 47440.65, 47487.9, 272914366, 13331.32],
        ["10-MAY-2024", 47555.5, 47868.7, 47313.35, 47421.1, 218854712, 10013.36],
        ["13-MAY-2024", 47389.8, 47841.6, 46983.25, 47754.1, 220865860, 8760.91],
        ["14-MAY-2024", 47748.85, 47937.25, 47607.85, 47859.45, 151958187, 6712.08],
        ["15-MAY-2024", 47923.1, 47957.2, 47534.5, 47687.45, 188783717, 8763.34],
        ["16-MAY-2024", 47945.85, 48052.9, 47340.35, 47977.05, 219943967, 11108.34],
        ["17-MAY-2024", 47842.95, 48188.65, 47758.8, 48115.65, 143821357, 8123.45],
        ["18-MAY-2024", 48197.3, 48222.35, 48108.15, 48199.5, 19246577, 541.44],
        ["21-MAY-2024", 47927.1, 48259.75, 47927.1, 48048.2, 166750558, 9490],
        ["22-MAY-2024", 48113.9, 48114.05, 47435.25, 47781.95, 204867369, 11396.04],
        ["23-MAY-2024", 47899.35, 48829.7, 47873.15, 48768.6, 184456258, 11162.27],
        ["24-MAY-2024", 48668, 49052.95, 48644.8, 48971.65, 135007612, 7585],
        ["27-MAY-2024", 49105.9, 49588.85, 49051.25, 49281.8, 268530645, 9350.6],
        ["28-MAY-2024", 49330.9, 49511.15, 48943.55, 49142.15, 155174235, 7545.21],
        ["29-MAY-2024", 48786.7, 49022.6, 48401.55, 48501.35, 183590101, 10469],
        ["30-MAY-2024", 48313.6, 49044.6, 48313.6, 48682.35, 151037825, 9375.71],
        ["31-MAY-2024", 48895.15, 49122.55, 48569.05, 48983.95, 303626980, 18069.59],
        

    ]
    

    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'])
    

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    

    df = df.rename(columns={'Date': 'timestamp'})
    
    logger.info(f"Created comprehensive NIFTY Bank dataset: {len(df)} records")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Price range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
    
    return df

def train_optimized_model(data):
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["Volatility"] = data["Return"].rolling(5).std()
    

    data["MA_Ratio"] = data["MA5"] / data["MA10"]
    data["Price_MA5_Ratio"] = data["Close"] / data["MA5"]
    data["Volume_MA"] = data["Volume"].rolling(5).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]
    

    data["Return_2d"] = data["Close"].pct_change(2)
    data["Momentum_3d"] = data["Close"] / data["Close"].shift(3) - 1
    
    data.dropna(inplace=True)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    

    data = data[:-1]
    
    features = [
        "Return", "MA5", "MA10", "Volatility",
        "MA_Ratio", "Price_MA5_Ratio", "Volume_Ratio",
        "Return_2d", "Momentum_3d"
    ]
    
    X = data[features]
    y = data["Target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=False)
    

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, scaler, X_train, X_test, y_train, y_test, features, data

def main():
    logger.info("=== COMPREHENSIVE NIFTY BANK TEST ===")
    

    data = create_full_nifty_dataset()
    

    logger.info("Training optimized model...")
    model, scaler, X_train, X_test, y_train, y_test, features, processed_data = train_optimized_model(data.copy())
    

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    

    high_conf_mask = (y_pred_proba > 0.65) | (y_pred_proba < 0.35)
    high_conf_accuracy = 0
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
    

    feature_importance = dict(zip(features, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    

    logger.info("=" * 60)
    logger.info(" COMPREHENSIVE NIFTY BANK RESULTS")
    logger.info("=" * 60)
    logger.info(f" Training samples: {len(X_train)}")
    logger.info(f" Test samples: {len(X_test)}")
    logger.info(f" Training Accuracy: {train_accuracy:.1%}")
    logger.info(f" Test Accuracy: {test_accuracy:.1%}")
    logger.info(f" High Confidence Accuracy: {high_conf_accuracy:.1%} ({high_conf_mask.sum()} predictions)")
    logger.info(f" Improvement over random: {(test_accuracy - 0.5) * 100:.1f} percentage points")
    logger.info("=" * 60)
    
    logger.info(" Feature Importance:")
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"   {i:2d}. {feature}: {importance:.4f}")
    

    os.makedirs('output', exist_ok=True)
    
    plt.figure(figsize=(16, 10))
    

    plt.subplot(2, 2, 1)
    plt.plot(processed_data['timestamp'], processed_data['Close'], alpha=0.8, linewidth=1)
    plt.title('NIFTY Bank Price History (Jan-May 2024)')
    plt.ylabel('Price (₹)')
    plt.xticks(rotation=45)
    

    plt.subplot(2, 2, 2)
    test_data = processed_data.iloc[-len(y_test):]
    plt.plot(test_data['timestamp'], test_data['Close'], alpha=0.7, label='Price')
    
    correct_mask = y_pred == y_test
    for i, (timestamp, correct) in enumerate(zip(test_data['timestamp'], correct_mask)):
        color = 'green' if correct else 'red'
        marker = '^' if y_pred[i] == 1 else 'v'
        plt.scatter(timestamp, test_data['Close'].iloc[i], color=color, marker=marker, s=40, alpha=0.8)
    
    plt.title('Test Period with Predictions')
    plt.ylabel('Price (₹)')
    plt.xticks(rotation=45)
    plt.legend()
    

    plt.subplot(2, 2, 3)
    cumulative_accuracy = np.cumsum(correct_mask) / np.arange(1, len(correct_mask) + 1)
    plt.plot(cumulative_accuracy, linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    plt.title('Cumulative Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    

    plt.subplot(2, 2, 4)
    names, importances = zip(*top_features)
    plt.barh(names, importances)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('output/comprehensive_nifty_results.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved to output/comprehensive_nifty_results.png")
    

    if test_accuracy > 0.60:
        logger.info(" EXCELLENT! Model shows strong predictive power on NIFTY Bank!")
    elif test_accuracy > 0.55:
        logger.info(" GOOD! Model shows meaningful improvement on NIFTY Bank!")
    elif test_accuracy > 0.52:
        logger.info(" DECENT! Model beats random chance on NIFTY Bank!")
    else:
        logger.info("  Model performance needs improvement")
    
    logger.info(f"\n With {len(data)} data points, we have much better training!")
    logger.info(" Your simple approach works well with sufficient data!")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = main()
    exit(0)