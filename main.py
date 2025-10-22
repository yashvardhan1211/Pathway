
import sys
import os
import logging
from datetime import datetime

sys.path.append('/app/src')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    
    logger.info("=== Dockerized Stock Predictor Starting ===")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        logger.info("Running sample Pathway demo...")
        from sample_pathway_app import main as demo_main
        return demo_main()
    else:
        logger.info("Full application not yet implemented.")
        logger.info("Run with 'demo' argument to test Pathway functionality.")
        logger.info("Example: python main.py demo")
        return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Application completed successfully!")
    else:
        logger.error("Application failed!")
    exit(0 if success else 1)