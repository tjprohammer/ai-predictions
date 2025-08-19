# MLB AI Predictions System

A comprehensive machine learning system for predicting MLB game outcomes with real-time data integration and enhanced bullpen analysis.

## 🎯 System Overview

This system combines historical MLB data, real-time weather conditions, betting markets, and advanced pitcher statistics to generate accurate game predictions. The enhanced bullpen predictor incorporates rolling pitcher statistics and sophisticated feature engineering.

## 🚀 Key Features

- **Enhanced Bullpen Predictor**: Advanced ML pipeline with pitcher rolling statistics
- **Real-time Data Integration**: Live odds, weather, and game data
- **PostgreSQL Database**: Structured data storage with optimized queries
- **FastAPI Backend**: RESTful API for predictions and data access
- **Comprehensive Feature Engineering**: 50+ features including weather, market, and pitcher stats

## 📊 Current System Status

✅ **Working Components:**
- Enhanced predictor generating 16 daily games successfully
- Database connections with proper table mapping
- Pitcher rolling stats integration (ERA, WHIP, K/9, BB/9)
- Weather and market data integration
- Fallback logic for robust prediction generation

⚠️ **In Development:**
- merge_asof optimization for pitcher stats (currently using fallback)
- Environment feature flattening investigation
- Duplicate game processing refinement

## 🏗️ System Architecture

```
├── enhanced_bullpen_predictor.py     # Core ML prediction engine
├── mlb-overs/                        # Main prediction system
│   ├── api/                         # FastAPI backend
│   ├── data_collection/             # Data ingestion pipelines
│   ├── feature_engineering/         # Feature processing
│   └── deployment/                  # Docker and deployment configs
├── comprehensive_training_builder.py # Model training pipeline
└── validate_enhanced_pipeline.py    # System validation
```

## 🛠️ Technology Stack

- **Python 3.13**: Core language
- **PostgreSQL**: Database with tables for games, pitcher stats, features
- **FastAPI + SQLAlchemy**: API and database ORM
- **pandas**: Data processing with merge_asof for time-series joins
- **scikit-learn**: Machine learning models and calibration
- **Docker**: Containerized deployment

## 📈 Prediction Pipeline

1. **Data Collection**: Ingest game schedules, weather, odds, pitcher stats
2. **Feature Engineering**: Generate 50+ features including rolling statistics
3. **Model Inference**: Enhanced bullpen predictor with calibrated outputs
4. **API Response**: JSON predictions with confidence intervals

## 🎮 Quick Start

```bash
# Clone the repository
git clone https://github.com/tjprohammer/ai-predictions.git
cd ai-predictions

# Set up Python environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Start the API server
cd mlb-overs
python -m uvicorn api.main:app --reload
```

## 📋 Recent Improvements

- **Database Connection Fixes**: Resolved API table mapping issues
- **Enhanced Predictor**: Successfully integrated pitcher rolling statistics
- **Robust Fallback Logic**: merge_asof failures handled gracefully
- **Comprehensive Debugging**: Added extensive logging and error handling
- **Version Control**: Full git history and GitHub integration

## 🔧 Configuration

Key configuration files:
- `docker-compose.yml`: Database and service orchestration
- `.env`: Environment variables (not tracked)
- `mlb-overs/api/config.py`: API configuration
- `mlb-overs/deployment/`: Production deployment configs

## 📊 Database Schema

Primary tables:
- `enhanced_games`: Game schedules and basic features
- `pitcher_daily_rolling`: Rolling pitcher statistics (ERA, WHIP, etc.)
- `legitimate_game_features`: Comprehensive feature set for predictions
- `weather_data`: Game-day weather conditions
- `market_data`: Betting odds and market information

## 🤝 Contributing

This is an active development project. Key areas for contribution:
- Optimize merge_asof operations for better performance
- Enhance feature engineering with additional data sources
- Improve model calibration and uncertainty quantification

## 📧 Contact

For questions or collaboration: prohammer1@gmail.com

---
*Last updated: December 2024*
*System Status: Production-ready with ongoing optimizations*
