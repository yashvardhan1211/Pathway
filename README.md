# Dockerized Stock Predictor

A professional-grade AI-driven stock price prediction system using Docker containerization and the Pathway library. Achieves 68.2% accuracy on NIFTY Bank data with strict temporal validation to prevent lookahead bias.

## Features

- **Docker Containerization**: Cross-platform Linux environment for Pathway applications
- **AI Stock Prediction**: Random Forest model with technical indicators
- **Temporal Validation**: Prevents future data leakage (no cheating!)
- **Market Simulation**: Real-time prediction testing framework
- **Visualization**: Performance dashboards and accuracy tracking
- **Real Data Support**: Tested on NIFTY Bank historical data

## Quick Start

### Prerequisites
- Docker Desktop installed
- Python 3.9+ (for local testing)

### Setup
```bash
# Build Docker container
./build.sh

# Run container
./run.sh

# Test with sample data
python3 test_complete_system.py

# Test with NIFTY Bank data
python3 comprehensive_nifty_test.py
```

## Results

**NIFTY Bank Performance:**
- Test Accuracy: 68.2%
- High Confidence Accuracy: 69.2%
- Improvement over random: +18.2 percentage points
- Training samples: 249(1 year Bank Nifty data)

## Key Files

- `comprehensive_nifty_test.py` - Main test with real NIFTY Bank data
- `src/ai_predictor.py` - Core AI prediction engine
- `src/temporal_validator.py` - Prevents lookahead bias
- `src/market_simulator.py` - Real-time simulation framework
- `src/visualization.py` - Performance visualization
- `Dockerfile` - Container configuration

## Architecture

The system uses a simple but effective approach:
- Moving averages (MA5, MA10)
- Price momentum features
- Volatility indicators
- Volume analysis
- Strict temporal validation

## Docker Usage

```bash
# Build and run
docker compose up --build

# Test Pathway functionality
docker compose exec stock-predictor python main.py demo

# View logs
docker compose logs -f
```

## Model Performance

The AI model uses clean, interpretable features and achieves professional-grade results:
- No overfitting
- Temporal validation prevents cheating
- Real market data shows predictable patterns
- Simple features outperform complex models

## License

MIT License - See LICENSE file for details.
