
import pathway as pw
import pandas as pd
import time
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    
    logger.info("Creating sample streaming data...")
    
    sample_data = [
        {"timestamp": "2024-01-01 09:00:00", "symbol": "BTC", "price": 45000.0, "volume": 1000},
        {"timestamp": "2024-01-01 09:01:00", "symbol": "BTC", "price": 45100.0, "volume": 1200},
        {"timestamp": "2024-01-01 09:02:00", "symbol": "BTC", "price": 44950.0, "volume": 800},
        {"timestamp": "2024-01-01 09:03:00", "symbol": "BTC", "price": 45200.0, "volume": 1500},
        {"timestamp": "2024-01-01 09:04:00", "symbol": "BTC", "price": 45050.0, "volume": 900},
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv('/app/data/sample_data.csv', index=False)
    logger.info("Sample data saved to /app/data/sample_data.csv")
    
    return sample_data

def run_pathway_demo():
    
    logger.info("Starting Pathway demo...")
    
    try:
        create_sample_data()
        
        class InputSchema(pw.Schema):
            timestamp: str
            symbol: str
            price: float
            volume: int
        
        table = pw.io.csv.read(
            '/app/data/sample_data.csv',
            schema=InputSchema,
            mode="streaming"
        )
        
        processed = table.select(
            timestamp=pw.this.timestamp,
            symbol=pw.this.symbol,
            price=pw.this.price,
            volume=pw.this.volume,
            value=pw.this.price * pw.this.volume
        )
        
        pw.io.csv.write(processed, '/app/output/processed_sample.csv')
        
        logger.info("Pathway processing completed successfully!")
        logger.info("Results written to /app/output/processed_sample.csv")
        
        pw.run()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Pathway demo: {str(e)}")
        return False

def verify_pathway_installation():
    
    logger.info("Verifying Pathway installation...")
    
    try:
        import pathway as pw
        logger.info(f"Pathway version: {pw.__version__}")
        
        test_data = pw.debug.table_from_markdown()
        
        result = test_data.select(
            name=pw.this.name,
            age_plus_one=pw.this.age + 1
        )
        
        logger.info("Basic Pathway functionality test passed!")
        return True
        
    except ImportError as e:
        logger.error(f"Pathway import failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Pathway functionality test failed: {str(e)}")
        return False

def main():
    
    logger.info("=== Dockerized Pathway Application Demo ===")
    logger.info(f"Starting at: {datetime.now()}")
    
    if not verify_pathway_installation():
        logger.error("Pathway installation verification failed!")
        return False
    
    if not run_pathway_demo():
        logger.error("Pathway demo failed!")
        return False
    
    logger.info("=== Demo completed successfully! ===")
    logger.info("Docker environment and Pathway library are working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)