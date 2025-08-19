#!/usr/bin/env python3
"""
Enhanced ML API Server for MLB Predictions
Provides REST endpoints for the enhanced ML model predictions
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

try:
    from enhanced_ml_predictions import EnhancedMLPredictor
    ENHANCED_MODEL_AVAILABLE = True
except ImportError:
    ENHANCED_MODEL_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the enhanced ML predictor"""
    global predictor
    
    if ENHANCED_MODEL_AVAILABLE and predictor is None:
        try:
            predictor = EnhancedMLPredictor()
            print("✅ Enhanced ML Predictor initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize enhanced predictor: {e}")
            return False
    return ENHANCED_MODEL_AVAILABLE

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'enhanced_model_available': ENHANCED_MODEL_AVAILABLE,
        'predictor_initialized': predictor is not None
    })

@app.route('/api/predictions/today', methods=['GET'])
def get_todays_predictions():
    """Get predictions for today's games"""
    
    if not initialize_predictor():
        return jsonify({
            'error': 'Enhanced ML model not available',
            'fallback': 'Use /api/predictions/basic for basic predictions'
        }), 503
    
    try:
        predictions = predictor.predict_games()
        
        if not predictions:
            return jsonify({
                'message': 'No games found for today',
                'date': str(date.today()),
                'predictions': []
            })
        
        return jsonify({
            'success': True,
            'date': str(date.today()),
            'total_games': len(predictions),
            'model_version': 'enhanced_v1.0',
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to generate predictions: {str(e)}'
        }), 500

@app.route('/api/predictions/date/<target_date>', methods=['GET'])
def get_predictions_for_date(target_date):
    """Get predictions for a specific date (YYYY-MM-DD)"""
    
    if not initialize_predictor():
        return jsonify({
            'error': 'Enhanced ML model not available'
        }), 503
    
    try:
        # Parse date
        prediction_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        predictions = predictor.predict_games(prediction_date)
        
        return jsonify({
            'success': True,
            'date': str(prediction_date),
            'total_games': len(predictions),
            'model_version': 'enhanced_v1.0',
            'predictions': predictions
        })
        
    except ValueError:
        return jsonify({
            'error': 'Invalid date format. Use YYYY-MM-DD'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Failed to generate predictions: {str(e)}'
        }), 500

@app.route('/api/recommendations/today', methods=['GET'])
def get_todays_recommendations():
    """Get betting recommendations for today"""
    
    if not initialize_predictor():
        return jsonify({
            'error': 'Enhanced ML model not available'
        }), 503
    
    try:
        recommendations = predictor.generate_recommendations_json()
        
        if not recommendations:
            return jsonify({
                'message': 'No recommendations available for today',
                'date': str(date.today())
            })
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to generate recommendations: {str(e)}'
        }), 500

@app.route('/api/recommendations/best', methods=['GET'])
def get_best_bets():
    """Get only the best betting recommendations"""
    
    if not initialize_predictor():
        return jsonify({
            'error': 'Enhanced ML model not available'
        }), 503
    
    try:
        recommendations = predictor.generate_recommendations_json()
        
        if not recommendations or not recommendations.get('best_bets'):
            return jsonify({
                'message': 'No strong recommendations available',
                'date': str(date.today()),
                'best_bets': []
            })
        
        # Filter for only high-confidence bets
        high_confidence_bets = [
            bet for bet in recommendations['best_bets']
            if bet.get('confidence', 0) >= 0.7
        ]
        
        return jsonify({
            'success': True,
            'date': recommendations['date'],
            'model_version': recommendations['model_version'],
            'total_opportunities': len(recommendations['best_bets']),
            'high_confidence_count': len(high_confidence_bets),
            'best_bets': high_confidence_bets,
            'summary': recommendations.get('summary', {})
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get best bets: {str(e)}'
        }), 500

@app.route('/api/predictions/basic', methods=['GET'])
def get_basic_predictions():
    """Fallback to basic predictions"""
    
    try:
        # Try to load from existing recommendations file
        recommendations_file = Path(__file__).parent.parent / "web_app" / "daily_recommendations.json"
        
        if recommendations_file.exists():
            with open(recommendations_file) as f:
                data = json.load(f)
                return jsonify(data)
        else:
            return jsonify({
                'error': 'No predictions available. Run the daily workflow first.'
            }), 404
            
    except Exception as e:
        return jsonify({
            'error': f'Failed to load basic predictions: {str(e)}'
        }), 500

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """Get detailed model status information"""
    
    status = {
        'enhanced_model_available': ENHANCED_MODEL_AVAILABLE,
        'predictor_initialized': predictor is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    if predictor:
        try:
            # Test prediction capability
            test_games = predictor.get_todays_games()
            status['todays_games_count'] = len(test_games)
            status['model_loaded'] = predictor.model is not None
            status['database_connected'] = True
            
        except Exception as e:
            status['error'] = str(e)
            status['database_connected'] = False
    
    return jsonify(status)

@app.route('/api/features/game/<game_id>', methods=['GET'])
def get_game_features(game_id):
    """Get detailed features for a specific game"""
    
    if not initialize_predictor():
        return jsonify({
            'error': 'Enhanced ML model not available'
        }), 503
    
    try:
        # Get today's games and find the specific game
        games_df = predictor.get_todays_games()
        game = games_df[games_df['game_id'] == game_id]
        
        if game.empty:
            return jsonify({
                'error': f'Game {game_id} not found'
            }), 404
        
        # Create features for this game
        features_df = predictor.create_enhanced_features(game)
        
        if features_df.empty:
            return jsonify({
                'error': f'Could not create features for game {game_id}'
            }), 500
        
        # Return detailed feature breakdown
        game_features = features_df.iloc[0].to_dict()
        
        return jsonify({
            'game_id': game_id,
            'matchup': f"{game.iloc[0]['away_team']} @ {game.iloc[0]['home_team']}",
            'features': game_features,
            'feature_count': len(game_features) - 1,  # Exclude game_id
            'model_version': 'enhanced_v1.0'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get game features: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced ML API Server...")
    print(f"📊 Enhanced Model Available: {ENHANCED_MODEL_AVAILABLE}")
    
    # Initialize predictor on startup
    if initialize_predictor():
        print("✅ Predictor initialized successfully")
    else:
        print("⚠️  Predictor initialization failed, some endpoints may not work")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
