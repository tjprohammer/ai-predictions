#!/usr/bin/env python3
"""
Enhanced ML Prediction Pipeline for Gameday
Integrates the enhanced ML model with real-time gameday data
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, date
from pathlib import Path
import joblib
from sqlalchemy import create_engine

# Add paths for imports
sys.path.append(str(Pa    def _get_betting_total(self, game_id):
        """Get the betting market total for this game"""
        try:
            # Try different possible table/column names
            queries = [
                "SELECT total FROM betting_lines WHERE game_id = %s ORDER BY updated_at DESC LIMIT 1",
                "SELECT total_line FROM markets_totals WHERE game_id = %s ORDER BY updated_at DESC LIMIT 1",
                "SELECT market_total FROM game_lines WHERE game_id = %s ORDER BY updated_at DESC LIMIT 1"
            ]
            
            for query in queries:
                try:
                    result = pd.read_sql(query, self.engine, params=(game_id,))
                    if not result.empty:
                        return float(result.iloc[0].iloc[0])
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️  Could not get betting total for game {game_id}: {e}")
            
        return 8.5  # Default total.parent))
sys.path.append(str(Path(__file__).parent.parent / "mlb-overs"))

try:
    from mlb_overs.features.build_features import build_features_for_date
    from mlb_overs.models.infer import predict_games_for_date
except ImportError:
    # Fallback for path issues
    pass

class EnhancedMLPredictor:
    """Enhanced ML Predictor using the trained model with enhanced features"""
    
    def __init__(self, model_path=None, db_url=None):
        """Initialize the predictor with model and database connection"""
        
        # Set up paths
        self.base_path = Path(__file__).parent.parent
        self.model_path = model_path or self.base_path / "mlb-overs" / "models" / "enhanced_mlb_predictor.joblib"
        
        # Database connection
        self.db_url = db_url or "postgresql://mlbuser:mlbpass@localhost:5432/mlb"
        self.engine = create_engine(self.db_url)
        
        # Load model
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained ML model"""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                
                # Handle different model formats
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    print(f"✅ Loaded enhanced ML model (dict format) from {self.model_path}")
                else:
                    self.model = model_data
                    print(f"✅ Loaded enhanced ML model (direct format) from {self.model_path}")
                    
                if self.model is None:
                    print(f"❌ Model object is None - check model file format")
                    return False
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("   Run the enhanced model training first!")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
        
    def get_todays_games(self, target_date=None):
        """Get today's games with weather data from MLB API"""
        if target_date is None:
            target_date = date.today()
            
        # Fetch from MLB API with weather data
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={date_str}&endDate={date_str}&sportId=1&hydrate=weather,venue,team,probablePitcher"
            
            response = requests.get(url)
            if response.status_code == 200:
                mlb_data = response.json()
                
                games_list = []
                for date_obj in mlb_data.get('dates', []):
                    for game in date_obj.get('games', []):
                        try:
                            # Extract game data
                            game_info = {
                                'game_id': game['gamePk'],
                                'date': target_date,
                                'home_team': game['teams']['home']['team']['abbreviation'],
                                'away_team': game['teams']['away']['team']['abbreviation'],
                                'venue_name': game['venue']['name'],
                                'park_id': game['venue']['id'],
                                'day_night': 'D' if '12:' in game['gameDate'] or '13:' in game['gameDate'] else 'N'
                            }
                            
                            # Extract pitcher info
                            try:
                                if game.get('teams', {}).get('home', {}).get('probablePitcher'):
                                    game_info['home_sp_id'] = game['teams']['home']['probablePitcher']['id']
                                else:
                                    game_info['home_sp_id'] = None
                                    
                                if game.get('teams', {}).get('away', {}).get('probablePitcher'):
                                    game_info['away_sp_id'] = game['teams']['away']['probablePitcher']['id']
                                else:
                                    game_info['away_sp_id'] = None
                            except:
                                game_info['home_sp_id'] = None
                                game_info['away_sp_id'] = None
                            
                            # Extract weather data
                            weather = game.get('weather', {})
                            if weather:
                                # Temperature
                                temp = 75  # Default
                                if weather.get('temp'):
                                    try:
                                        temp_str = str(weather['temp']).replace('°', '').replace('F', '').strip()
                                        temp = float(temp_str)
                                    except:
                                        pass
                                game_info['temperature'] = temp
                                
                                # Wind
                                game_info['wind_speed'] = weather.get('wind', '5 mph').replace(' mph', '').split()[0] if weather.get('wind') else '5'
                                try:
                                    game_info['wind_speed'] = float(game_info['wind_speed'])
                                except:
                                    game_info['wind_speed'] = 5.0
                                    
                                game_info['wind_direction'] = weather.get('wind', 'Calm').split()[-1] if weather.get('wind') else 'N'
                                game_info['weather_condition'] = weather.get('condition', 'Clear')
                            else:
                                # Default weather
                                game_info['temperature'] = 75
                                game_info['wind_speed'] = 5.0
                                game_info['wind_direction'] = 'N'
                                game_info['weather_condition'] = 'Clear'
                            
                            games_list.append(game_info)
                            
                        except Exception as e:
                            print(f"⚠️  Error processing game {game.get('gamePk', 'unknown')}: {e}")
                            continue
                
                return pd.DataFrame(games_list)
            else:
                print(f"❌ Failed to fetch games from MLB API: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error fetching games from MLB API: {e}")
        
        # Fallback to database if API fails
        try:
            query = """
            SELECT DISTINCT
                g.game_id,
                g.date,
                g.home_team,
                g.away_team,
                g.home_sp_id,
                g.away_sp_id,
                g.park_id,
                p.name as venue_name,
                CASE WHEN EXTRACT(hour FROM g.date) < 17 THEN 'D' ELSE 'N' END as day_night,
                75 as temperature,
                5.0 as wind_speed,
                'N' as wind_direction,
                'Clear' as weather_condition
            FROM games g
            LEFT JOIN parks p ON g.park_id = p.park_id
            WHERE g.date = %s
            ORDER BY g.game_id
            """
            
            df = pd.read_sql(query, self.engine, params=(target_date,))
            return df
            
        except Exception as e:
            print(f"❌ Database fallback also failed: {e}")
            return pd.DataFrame()
        
    def create_enhanced_features(self, games_df):
        """Create enhanced features for today's games using the same logic as training"""
        
        features_list = []
        
        for _, game in games_df.iterrows():
            try:
                # Get enhanced features for this game
                features = self._extract_game_features(game)
                features['game_id'] = game['game_id']
                features_list.append(features)
            except Exception as e:
                print(f"❌ Error creating features for game {game['game_id']}: {e}")
                continue
                
        if not features_list:
            return pd.DataFrame()
            
        features_df = pd.DataFrame(features_list)
        return features_df
        
    def _extract_game_features(self, game):
        """Extract enhanced features for a single game"""
        
        features = {}
        
        # === PITCHER FEATURES ===
        home_pitcher_stats = self._get_pitcher_stats(game['home_sp_id'], game['away_team'])
        away_pitcher_stats = self._get_pitcher_stats(game['away_sp_id'], game['home_team'])
        
        # Season ERA
        features['home_sp_era_season'] = home_pitcher_stats.get('era_season', 4.0)
        features['away_sp_era_season'] = away_pitcher_stats.get('era_season', 4.0)
        
        # Recent ERA (last 3, 5, 10 games)
        features['home_sp_era_l3'] = home_pitcher_stats.get('era_l3', 4.0)
        features['away_sp_era_l3'] = away_pitcher_stats.get('era_l3', 4.0)
        features['home_sp_era_l5'] = home_pitcher_stats.get('era_l5', 4.0)
        features['away_sp_era_l5'] = away_pitcher_stats.get('era_l5', 4.0)
        features['home_sp_era_l10'] = home_pitcher_stats.get('era_l10', 4.0)
        features['away_sp_era_l10'] = away_pitcher_stats.get('era_l10', 4.0)
        
        # VS opponent ERA
        features['home_sp_era_vs_opp'] = home_pitcher_stats.get('era_vs_opp', 4.0)
        features['away_sp_era_vs_opp'] = away_pitcher_stats.get('era_vs_opp', 4.0)
        
        # === TEAM OFFENSE FEATURES ===
        home_offense = self._get_team_offense_stats(game['home_team'])
        away_offense = self._get_team_offense_stats(game['away_team'])
        
        # Recent team stats (14 and 30 day windows)
        features['hxwoba14'] = home_offense.get('xwoba_14d', 0.320)
        features['axwoba14'] = away_offense.get('xwoba_14d', 0.320)
        features['hiso14'] = home_offense.get('iso_14d', 0.150)
        features['aiso14'] = away_offense.get('iso_14d', 0.150)
        features['hbb14'] = home_offense.get('bb_pct_14d', 0.080)
        features['abb14'] = away_offense.get('bb_pct_14d', 0.080)
        
        features['hxwoba30'] = home_offense.get('xwoba_30d', 0.320)
        features['axwoba30'] = away_offense.get('xwoba_30d', 0.320)
        features['hiso30'] = home_offense.get('iso_30d', 0.150)
        features['aiso30'] = away_offense.get('iso_30d', 0.150)
        features['hbb30'] = home_offense.get('bb_pct_30d', 0.080)
        features['abb30'] = away_offense.get('bb_pct_30d', 0.080)
        
        # === BULLPEN FEATURES ===
        home_bullpen = self._get_bullpen_stats(game['home_team'])
        away_bullpen = self._get_bullpen_stats(game['away_team'])
        
        features['hbp_fip_yday'] = home_bullpen.get('fip_yesterday', 4.0)
        features['abp_fip_yday'] = away_bullpen.get('fip_yesterday', 4.0)
        features['hbp_b2b'] = home_bullpen.get('back_to_back', 0)
        features['abp_b2b'] = away_bullpen.get('back_to_back', 0)
        
        # === WEATHER & PARK FEATURES ===
        # Market close (betting total)
        features['k_close'] = self._get_betting_total(game['game_id'])
        
        # Weather impact
        weather_factor = self._calculate_weather_impact(game)
        features['weather_impact'] = weather_factor
        
        # Park factor
        park_factor = self._get_park_factor(game.get('venue_name', ''))
        features['park_factor'] = park_factor
        
        # Day/Night factor
        features['day_night'] = 1 if game.get('day_night') == 'D' else 0
        
        return features
        
    def _get_pitcher_stats(self, pitcher_id, opponent_team):
        """Get comprehensive pitcher statistics from training data"""
        if pd.isna(pitcher_id) or not pitcher_id:
            return {
                'era_season': 4.0,
                'era_l3': 4.0,
                'era_l5': 4.0,
                'era_l10': 4.0,
                'era_vs_opp': 4.0,
                'whip': 1.25,
                'k_bb_ratio': 2.8
            }
            
        try:
            # Load training data to get pitcher stats
            training_data = pd.read_parquet(self.base_path / "mlb-overs" / "data" / "enhanced_historical_games_2025.parquet")
            
            # Find games where this pitcher started
            home_games = training_data[training_data['home_sp_id'] == int(pitcher_id)]
            away_games = training_data[training_data['away_sp_id'] == int(pitcher_id)]
            
            all_era = []
            all_whip = []
            all_k_bb = []
            
            # Process home games
            for _, game in home_games.iterrows():
                if pd.notna(game.get('home_sp_er')) and pd.notna(game.get('home_sp_ip')) and float(game['home_sp_ip']) > 0:
                    era = (float(game['home_sp_er']) * 9.0) / float(game['home_sp_ip'])
                    all_era.append(era)
                    
                    # WHIP calculation
                    if pd.notna(game.get('home_sp_h')) and pd.notna(game.get('home_sp_bb')):
                        whip = (float(game['home_sp_h']) + float(game['home_sp_bb'])) / float(game['home_sp_ip'])
                        all_whip.append(whip)
                    
                    # K/BB ratio
                    if pd.notna(game.get('home_sp_k')) and pd.notna(game.get('home_sp_bb')) and float(game['home_sp_bb']) > 0:
                        k_bb = float(game['home_sp_k']) / float(game['home_sp_bb'])
                        all_k_bb.append(k_bb)
            
            # Process away games
            for _, game in away_games.iterrows():
                if pd.notna(game.get('away_sp_er')) and pd.notna(game.get('away_sp_ip')) and float(game['away_sp_ip']) > 0:
                    era = (float(game['away_sp_er']) * 9.0) / float(game['away_sp_ip'])
                    all_era.append(era)
                    
                    # WHIP calculation
                    if pd.notna(game.get('away_sp_h')) and pd.notna(game.get('away_sp_bb')):
                        whip = (float(game['away_sp_h']) + float(game['away_sp_bb'])) / float(game['away_sp_ip'])
                        all_whip.append(whip)
                    
                    # K/BB ratio
                    if pd.notna(game.get('away_sp_k')) and pd.notna(game.get('away_sp_bb')) and float(game['away_sp_bb']) > 0:
                        k_bb = float(game['away_sp_k']) / float(game['away_sp_bb'])
                        all_k_bb.append(k_bb)
            
            # Calculate stats
            if all_era:
                era_season = np.mean(all_era)
                era_l10 = np.mean(all_era[-10:]) if len(all_era) >= 10 else np.mean(all_era)
                era_l5 = np.mean(all_era[-5:]) if len(all_era) >= 5 else np.mean(all_era)
                era_l3 = np.mean(all_era[-3:]) if len(all_era) >= 3 else np.mean(all_era)
            else:
                era_season = era_l10 = era_l5 = era_l3 = 4.0
            
            whip = np.mean(all_whip) if all_whip else 1.25
            k_bb_ratio = np.mean(all_k_bb) if all_k_bb else 2.8
            
            return {
                'era_season': era_season,
                'era_l3': era_l3,
                'era_l5': era_l5,
                'era_l10': era_l10,
                'era_vs_opp': era_season,  # Simplified for now
                'whip': whip,
                'k_bb_ratio': k_bb_ratio
            }
            
        except Exception as e:
            print(f"⚠️  Could not get pitcher stats for {pitcher_id}: {e}")
            return {
                'era_season': 4.0,
                'era_l3': 4.0,
                'era_l5': 4.0,
                'era_l10': 4.0,
                'era_vs_opp': 4.0,
                'whip': 1.25,
                'k_bb_ratio': 2.8
            }
        
    def _get_team_offense_stats(self, team):
        """Get team offensive statistics"""
        try:
            query = """
            SELECT 
                AVG(CASE WHEN date >= CURRENT_DATE - INTERVAL '14 days' THEN woba END) as xwoba_14d,
                AVG(CASE WHEN date >= CURRENT_DATE - INTERVAL '30 days' THEN woba END) as xwoba_30d,
                AVG(CASE WHEN date >= CURRENT_DATE - INTERVAL '14 days' THEN iso END) as iso_14d,
                AVG(CASE WHEN date >= CURRENT_DATE - INTERVAL '30 days' THEN iso END) as iso_30d,
                AVG(CASE WHEN date >= CURRENT_DATE - INTERVAL '14 days' THEN bb_pct END) as bb_pct_14d,
                AVG(CASE WHEN date >= CURRENT_DATE - INTERVAL '30 days' THEN bb_pct END) as bb_pct_30d
            FROM teams_offense_daily 
            WHERE team = %s
            """
            
            result = pd.read_sql(query, self.engine, params=(team,))
            if not result.empty:
                return result.iloc[0].to_dict()
                
        except Exception as e:
            print(f"⚠️  Could not get team stats for {team}: {e}")
            
        return {}
        
    def _get_bullpen_stats(self, team):
        """Get bullpen statistics"""
        # Placeholder - implement based on your bullpen data
        return {
            'fip_yesterday': 4.0,
            'back_to_back': 0
        }
        
    def _get_betting_total(self, game_id):
        """Get the betting market total for this game"""
        try:
            query = """
            SELECT total_line 
            FROM markets_totals 
            WHERE game_id = %s 
            ORDER BY updated_at DESC 
            LIMIT 1
            """
            
            result = pd.read_sql(query, self.engine, params=(game_id,))
            if not result.empty:
                return float(result.iloc[0]['total_line'])
                
        except Exception as e:
            print(f"⚠️  Could not get betting total for game {game_id}: {e}")
            
        return 8.5  # Default total
        
    def _calculate_weather_impact(self, game):
        """Calculate weather impact on scoring using same logic as working predictor"""
        try:
            temp = float(game.get('temperature', 75))
            wind_speed = float(game.get('wind_speed', 5))
            wind_dir = str(game.get('wind_direction', 'N'))
            
            # Temperature impact (warmer = more offense)
            temp_factor = (temp - 70) * 0.01
            
            # Wind impact (simplified - out to any direction = slight boost)
            wind_factor = 0
            if 'out' in wind_dir.lower():
                wind_factor = wind_speed * 0.02
            elif 'in' in wind_dir.lower():
                wind_factor = wind_speed * -0.02
            else:
                wind_factor = wind_speed * 0.01  # Side wind slight boost
                
            # Weather condition impact
            condition_factor = 0.05  # Default clear weather
            condition = str(game.get('weather_condition', 'Clear')).lower()
            if 'dome' in condition or 'indoor' in condition:
                condition_factor = 0.1  # Controlled environment
            elif 'rain' in condition or 'storm' in condition:
                condition_factor = -0.1  # Suppress offense
            elif 'clear' in condition or 'sunny' in condition:
                condition_factor = 0.05  # Slight boost
                
            total_impact = temp_factor + wind_factor + condition_factor
            return total_impact
            
        except Exception as e:
            print(f"⚠️  Weather impact calculation error: {e}")
            return 0.05  # Default clear weather impact
            
    def _get_park_factor(self, venue_name):
        """Get park factor for venue"""
        park_factors = {
            'Coors Field': 1.15,
            'Fenway Park': 1.05,
            'Yankee Stadium': 1.03,
            'Minute Maid Park': 1.02,
            'Progressive Field': 0.95,
            'Tropicana Field': 0.93,
            'Petco Park': 0.92,
            'Marlins Park': 0.90
        }
        
        return park_factors.get(venue_name, 1.0)
        
    def predict_games(self, target_date=None):
        """Generate predictions for today's games"""
        
        if not self.model:
            print("❌ No model loaded!")
            return []
            
        # Get today's games
        games_df = self.get_todays_games(target_date)
        if games_df.empty:
            print("ℹ️  No games found for today")
            return []
            
        print(f"🎯 Found {len(games_df)} games to predict")
        
        # Create features
        features_df = self.create_enhanced_features(games_df)
        if features_df.empty:
            print("❌ Could not create features for any games")
            return []
            
        predictions = []
        
        for _, game in games_df.iterrows():
            try:
                game_features = features_df[features_df['game_id'] == game['game_id']]
                if game_features.empty:
                    continue
                    
                # Prepare feature vector (match training features)
                feature_cols = [col for col in game_features.columns if col != 'game_id']
                X = game_features[feature_cols].values.reshape(1, -1)
                
                # Make prediction
                predicted_total = float(self.model.predict(X)[0])
                
                # Get market total for comparison
                market_total = game_features.iloc[0].get('k_close', 8.5)
                
                # Calculate edge and recommendation
                edge = predicted_total - market_total
                confidence = min(abs(edge) * 0.3, 0.95)  # Convert edge to confidence
                
                recommendation = "OVER" if edge > 0.3 else "UNDER" if edge < -0.3 else "PASS"
                
                prediction = {
                    'game_id': game['game_id'],
                    'date': str(game['date']),
                    'matchup': f"{game['away_team']} @ {game['home_team']}",
                    'predicted_total': round(predicted_total, 1),
                    'market_total': round(market_total, 1),
                    'edge': round(edge, 1),
                    'confidence': round(confidence, 2),
                    'recommendation': recommendation,
                    'factors': {
                        'home_pitcher_era': round(game_features.iloc[0].get('home_sp_era_season', 4.0), 2),
                        'away_pitcher_era': round(game_features.iloc[0].get('away_sp_era_season', 4.0), 2),
                        'weather_impact': round(game_features.iloc[0].get('weather_impact', 0.0), 2),
                        'park_factor': round(game_features.iloc[0].get('park_factor', 1.0), 2)
                    }
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                print(f"❌ Error predicting game {game['game_id']}: {e}")
                continue
                
        return predictions
        
    def generate_recommendations_json(self, output_path=None, target_date=None):
        """Generate recommendations in JSON format for the frontend"""
        
        predictions = self.predict_games(target_date)
        
        if not predictions:
            return None
            
        # Sort by edge (highest first)
        predictions.sort(key=lambda x: abs(x['edge']), reverse=True)
        
        # Filter for strong recommendations
        strong_bets = [p for p in predictions if abs(p['edge']) >= 0.3 and p['confidence'] >= 0.6]
        
        recommendations = {
            'generated_at': datetime.now().isoformat(),
            'date': str(target_date or date.today()),
            'total_games': len(predictions),
            'model_version': 'enhanced_v1.0',
            'games': predictions,
            'best_bets': strong_bets[:5],  # Top 5 recommendations
            'summary': {
                'over_count': len([p for p in predictions if p['recommendation'] == 'OVER']),
                'under_count': len([p for p in predictions if p['recommendation'] == 'UNDER']),
                'pass_count': len([p for p in predictions if p['recommendation'] == 'PASS']),
                'strong_bets_count': len(strong_bets),
                'avg_confidence': round(np.mean([p['confidence'] for p in predictions]), 2)
            }
        }
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(recommendations, f, indent=2)
            print(f"✅ Saved recommendations to {output_path}")
            
        return recommendations


def main():
    """Main execution for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Enhanced ML Predictions')
    parser.add_argument('--date', type=str, help='Date to predict (YYYY-MM-DD)', default=None)
    parser.add_argument('--output', type=str, help='Output JSON file path', 
                       default='../web_app/enhanced_daily_recommendations.json')
    
    args = parser.parse_args()
    
    # Parse date if provided
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    
    # Create predictor and generate recommendations
    predictor = EnhancedMLPredictor()
    recommendations = predictor.generate_recommendations_json(args.output, target_date)
    
    if recommendations:
        print(f"\n🎯 ENHANCED ML PREDICTIONS SUMMARY")
        print(f"📅 Date: {recommendations['date']}")
        print(f"🎮 Total Games: {recommendations['total_games']}")
        print(f"🔥 Strong Bets: {recommendations['summary']['strong_bets_count']}")
        print(f"📊 Average Confidence: {recommendations['summary']['avg_confidence']}")
        
        if recommendations['best_bets']:
            print(f"\n🏆 TOP RECOMMENDATION:")
            top_bet = recommendations['best_bets'][0]
            print(f"   {top_bet['matchup']}")
            print(f"   {top_bet['recommendation']} {top_bet['market_total']} (Edge: {top_bet['edge']:+.1f})")
            print(f"   Confidence: {top_bet['confidence']}")
    else:
        print("❌ No predictions generated")


if __name__ == "__main__":
    main()
