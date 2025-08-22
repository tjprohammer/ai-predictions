#!/usr/bin/env python3
"""
Model Performance Enhancement System
====================================
Implements automated bias corrections and model improvements based on 
real-time performance tracking insights.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

from enhanced_prediction_tracker import EnhancedPredictionTracker

class ModelPerformanceEnhancer:
    def __init__(self):
        # Database connection
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
        # Initialize tracker
        self.tracker = EnhancedPredictionTracker()
        
        # Model paths
        self.model_dir = Path("mlb-overs/models")
        self.corrections_file = "model_bias_corrections.json"
        
    def analyze_current_performance(self, days=14) -> Dict:
        """Analyze current model performance and identify correction needs"""
        print("[ANALYZING] CURRENT MODEL PERFORMANCE")
        print("=" * 45)
        
        analysis = self.tracker.get_comprehensive_performance_analysis(days)
        
        if not analysis:
            print("[ERROR] Unable to get performance analysis")
            return {}
        
        # Extract key performance indicators
        metrics = analysis['metrics']['overall']
        
        performance_status = {
            'mae': metrics['mean_absolute_error'],
            'bias': metrics['mean_bias'],
            'accuracy_1run': metrics['accuracy_within_1'],
            'r_squared': metrics['r_squared'],
            'needs_correction': False,
            'correction_priority': 'LOW'
        }
        
        # Determine if corrections are needed
        if abs(performance_status['bias']) > 0.5:
            performance_status['needs_correction'] = True
            performance_status['correction_priority'] = 'HIGH' if abs(performance_status['bias']) > 1.0 else 'MEDIUM'
        
        if performance_status['mae'] > 3.5:
            performance_status['needs_correction'] = True
            performance_status['correction_priority'] = 'HIGH'
        
        if performance_status['accuracy_1run'] < 0.4:
            performance_status['needs_correction'] = True
            if performance_status['correction_priority'] == 'LOW':
                performance_status['correction_priority'] = 'MEDIUM'
        
        print(f"[PERFORMANCE STATUS]:")
        print(f"  • MAE: {performance_status['mae']:.2f} runs")
        print(f"  • Bias: {performance_status['bias']:.2f} runs")
        print(f"  • 1-Run Accuracy: {performance_status['accuracy_1run']:.1%}")
        print(f"  • R²: {performance_status['r_squared']:.3f}")
        print(f"  • Needs Correction: {performance_status['needs_correction']}")
        print(f"  • Priority: {performance_status['correction_priority']}")
        
        return {
            'performance': performance_status,
            'full_analysis': analysis
        }
    
    def generate_bias_corrections(self, analysis: Dict) -> Dict:
        """Generate comprehensive bias correction parameters"""
        print("\n[GENERATING] BIAS CORRECTIONS")
        print("=" * 35)
        
        corrections = {
            'global_adjustment': 0.0,
            'scoring_range_adjustments': {},
            'confidence_adjustments': {},
            'temperature_adjustments': {},
            'venue_adjustments': {},
            'pitcher_quality_adjustments': {},  # NEW: ERA-based corrections
            'day_of_week_adjustments': {},      # NEW: Day pattern corrections
            'market_deviation_adjustments': {}, # NEW: Market disagreement corrections
            'high_scoring_adjustments': {},     # NEW: Specific high-scoring corrections
            'timestamp': datetime.now().isoformat(),
            'based_on_days': 14,
            'games_analyzed': analysis['games_analyzed']
        }
        
        metrics = analysis['metrics']
        
        # Global bias correction
        global_bias = metrics['overall']['mean_bias']
        if abs(global_bias) > 0.3:
            corrections['global_adjustment'] = -global_bias
            print(f"  [SUCCESS] Global adjustment: {-global_bias:.2f} runs")
        
        # Scoring range corrections
        if 'by_scoring_range' in metrics:
            for range_name, range_metrics in metrics['by_scoring_range'].items():
                bias = range_metrics['mean_bias']
                games = range_metrics['games']
                
                # Only apply correction if sufficient sample size and significant bias
                if games >= 5 and abs(bias) > 0.5:
                    corrections['scoring_range_adjustments'][range_name] = -bias
                    print(f"  [SUCCESS] {range_name}: {-bias:.2f} runs ({games} games)")
        
        # Confidence-based corrections
        if 'confidence_analysis' in metrics:
            for conf_type, conf_metrics in metrics['confidence_analysis'].items():
                bias = conf_metrics['mean_bias']
                games = conf_metrics['games']
                
                if games >= 3 and abs(bias) > 0.7:
                    corrections['confidence_adjustments'][conf_type] = -bias
                    print(f"  [SUCCESS] {conf_type}: {-bias:.2f} runs ({games} games)")
        
        # Weather-based corrections
        if 'weather_impact' in metrics:
            for temp_range, temp_metrics in metrics['weather_impact'].items():
                bias = temp_metrics['mean_bias']
                games = temp_metrics['games']
                
                if games >= 8 and abs(bias) > 0.6:
                    corrections['temperature_adjustments'][temp_range] = -bias
                    print(f"  [SUCCESS] Temperature {temp_range}: {-bias:.2f} runs ({games} games)")
        
        # NEW: High-scoring game corrections (biggest opportunity from analysis)
        if 'by_scoring_range' in metrics:
            very_high_games = metrics['by_scoring_range'].get('Very High (12+)', {})
            if very_high_games.get('games', 0) >= 5:
                bias = very_high_games['mean_bias']
                mae = very_high_games.get('mae', 0)
                if abs(bias) > 2.0:  # Significant bias for high-scoring games
                    corrections['high_scoring_adjustments']['very_high_scoring'] = -bias
                    print(f"  [SUCCESS] Very High Scoring (12+ runs): {-bias:.2f} runs ({very_high_games['games']} games, MAE: {mae:.2f})")
        
        # NEW: Day-of-week corrections (Friday/Wednesday issues from analysis)
        if 'day_patterns' in metrics:
            for day, day_metrics in metrics['day_patterns'].items():
                bias = day_metrics['mean_bias']
                games = day_metrics['games']
                mae = day_metrics.get('mae', 0)
                
                # Focus on days with significant bias or poor MAE
                if games >= 10 and (abs(bias) > 0.8 or mae > 2.5):
                    corrections['day_of_week_adjustments'][day] = -bias
                    print(f"  [SUCCESS] {day}: {-bias:.2f} runs ({games} games, MAE: {mae:.2f})")
        
        # NEW: Pitcher quality corrections (ERA matchup issues from analysis)
        if 'pitcher_quality' in metrics:
            for era_range, era_metrics in metrics['pitcher_quality'].items():
                bias = era_metrics['mean_bias']
                games = era_metrics['games']
                mae = era_metrics.get('mae', 0)
                
                # Focus on matchups with poor performance
                if games >= 8 and mae > 3.0:  # High MAE indicates poor prediction quality
                    corrections['pitcher_quality_adjustments'][era_range] = -bias
                    print(f"  [SUCCESS] Pitcher Quality {era_range}: {-bias:.2f} runs ({games} games, MAE: {mae:.2f})")
        
        # NEW: Market deviation handling (large deviations actually perform better)
        if 'market_analysis' in metrics:
            for dev_range, dev_metrics in metrics['market_analysis'].items():
                bias = dev_metrics['mean_bias']
                games = dev_metrics['games']
                mae = dev_metrics.get('mae', 0)
                
                # Large deviations from analysis show they're often correct
                if dev_range == 'Large (2.0+)' and games >= 20:
                    # Don't penalize large deviations, they're actually good
                    if abs(bias) > 0.5:
                        corrections['market_deviation_adjustments'][dev_range] = -bias * 0.5  # Partial correction
                        print(f"  [SUCCESS] Market Deviation {dev_range}: {-bias * 0.5:.2f} runs ({games} games, MAE: {mae:.2f})")
        
        return corrections
    
    def apply_real_time_corrections(self, prediction: float, game_data: Dict, corrections: Dict) -> float:
        """Apply real-time bias corrections to a prediction"""
        
        adjusted_prediction = prediction
        adjustments_applied = []
        
        # Apply global adjustment
        if corrections.get('global_adjustment', 0) != 0:
            global_adj = corrections['global_adjustment']
            adjusted_prediction += global_adj
            adjustments_applied.append(f"Global: {global_adj:+.2f}")
        
        # Apply scoring range adjustment
        if 'scoring_range_adjustments' in corrections:
            predicted_range = self._categorize_scoring_range(prediction)
            if predicted_range in corrections['scoring_range_adjustments']:
                range_adj = corrections['scoring_range_adjustments'][predicted_range]
                adjusted_prediction += range_adj
                adjustments_applied.append(f"Range: {range_adj:+.2f}")
        
        # Apply confidence adjustment
        if 'confidence_adjustments' in corrections and game_data.get('is_high_confidence'):
            if game_data.get('is_premium_pick') and 'premium_picks' in corrections['confidence_adjustments']:
                conf_adj = corrections['confidence_adjustments']['premium_picks']
                adjusted_prediction += conf_adj
                adjustments_applied.append(f"Premium: {conf_adj:+.2f}")
            elif 'high_confidence' in corrections['confidence_adjustments']:
                conf_adj = corrections['confidence_adjustments']['high_confidence']
                adjusted_prediction += conf_adj
                adjustments_applied.append(f"HighConf: {conf_adj:+.2f}")
        
        # Apply temperature adjustment
        if 'temperature_adjustments' in corrections and 'temperature' in game_data:
            temp_range = self._categorize_temperature(game_data['temperature'])
            if temp_range in corrections['temperature_adjustments']:
                temp_adj = corrections['temperature_adjustments'][temp_range]
                adjusted_prediction += temp_adj
                adjustments_applied.append(f"Temp: {temp_adj:+.2f}")
        
        # NEW: Apply high-scoring game adjustments
        if 'high_scoring_adjustments' in corrections:
            predicted_range = self._categorize_scoring_range(adjusted_prediction)
            if predicted_range == "Very High (12+)" and 'very_high_scoring' in corrections['high_scoring_adjustments']:
                high_adj = corrections['high_scoring_adjustments']['very_high_scoring']
                adjusted_prediction += high_adj
                adjustments_applied.append(f"HighScoring: {high_adj:+.2f}")
        
        # NEW: Apply day-of-week adjustments
        if 'day_of_week_adjustments' in corrections and 'day_of_week' in game_data:
            day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                        5: 'Friday', 6: 'Saturday', 0: 'Sunday'}
            day_name = day_names.get(game_data['day_of_week'])
            if day_name and day_name in corrections['day_of_week_adjustments']:
                day_adj = corrections['day_of_week_adjustments'][day_name]
                adjusted_prediction += day_adj
                adjustments_applied.append(f"Day: {day_adj:+.2f}")
        
        # NEW: Apply pitcher quality adjustments
        if 'pitcher_quality_adjustments' in corrections and 'combined_era' in game_data:
            era_range = self._categorize_era_quality(game_data['combined_era'])
            if era_range in corrections['pitcher_quality_adjustments']:
                era_adj = corrections['pitcher_quality_adjustments'][era_range]
                adjusted_prediction += era_adj
                adjustments_applied.append(f"ERA: {era_adj:+.2f}")
        
        # NEW: Apply market deviation adjustments
        if 'market_deviation_adjustments' in corrections and 'market_deviation' in game_data:
            dev_range = self._categorize_market_deviation(game_data['market_deviation'])
            if dev_range in corrections['market_deviation_adjustments']:
                market_adj = corrections['market_deviation_adjustments'][dev_range]
                adjusted_prediction += market_adj
                adjustments_applied.append(f"Market: {market_adj:+.2f}")
        
        # Log adjustments if any were applied
        if adjustments_applied:
            print(f"[ADJUSTMENTS] Applied: {', '.join(adjustments_applied)} (Original: {prediction:.1f} → Adjusted: {adjusted_prediction:.1f})")
        
        return adjusted_prediction
    
    def _categorize_scoring_range(self, total_runs: float) -> str:
        """Categorize total runs into scoring ranges"""
        if total_runs <= 7:
            return "Low (<=7)"
        elif total_runs <= 9:
            return "Medium (8-9)"
        elif total_runs <= 11:
            return "High (10-11)"
        else:
            return "Very High (12+)"
    
    def _categorize_temperature(self, temperature: float) -> str:
        """Categorize temperature into ranges"""
        if temperature <= 60:
            return "Cold"
        elif temperature <= 75:
            return "Cool"
        elif temperature <= 85:
            return "Warm"
        else:
            return "Hot"
    
    def _categorize_era_quality(self, combined_era: float) -> str:
        """Categorize combined ERA into quality ranges"""
        if combined_era < 3.5:
            return "Elite (<3.5)"
        elif combined_era < 4.5:
            return "Good (3.5-4.5)"
        elif combined_era < 5.5:
            return "Average (4.5-5.5)"
        else:
            return "Poor (5.5+)"
    
    def _categorize_market_deviation(self, deviation: float) -> str:
        """Categorize market deviation into ranges"""
        if deviation <= 0.5:
            return "Close (≤0.5)"
        elif deviation <= 1.0:
            return "Small (0.5-1.0)"
        elif deviation <= 2.0:
            return "Medium (1.0-2.0)"
        else:
            return "Large (2.0+)"
    
    def save_corrections(self, corrections: Dict):
        """Save bias corrections to file"""
        with open(self.corrections_file, 'w') as f:
            json.dump(corrections, f, indent=2)
        print(f"[SAVED] Corrections saved to: {self.corrections_file}")
    
    def load_corrections(self) -> Dict:
        """Load existing bias corrections"""
        if os.path.exists(self.corrections_file):
            with open(self.corrections_file, 'r') as f:
                return json.load(f)
        return {}
    
    def update_model_corrections(self, force_update=False):
        """Update model bias corrections based on recent performance"""
        print("[UPDATING] MODEL BIAS CORRECTIONS")
        print("=" * 40)
        
        # Check if update is needed
        existing_corrections = self.load_corrections()
        
        if not force_update and existing_corrections:
            last_update = datetime.fromisoformat(existing_corrections.get('timestamp', '2024-01-01'))
            hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
            
            if hours_since_update < 6:  # Update at most every 6 hours
                print(f"⏰ Corrections updated {hours_since_update:.1f} hours ago, skipping update")
                return existing_corrections
        
        # Analyze current performance
        performance_analysis = self.analyze_current_performance()
        
        if not performance_analysis:
            print("[ERROR] Unable to analyze performance")
            return existing_corrections
        
        # Generate new corrections if needed
        if performance_analysis['performance']['needs_correction']:
            corrections = self.generate_bias_corrections(performance_analysis['full_analysis'])
            self.save_corrections(corrections)
            
            print(f"\n[SUCCESS] Model corrections updated successfully!")
            print(f"   Priority: {performance_analysis['performance']['correction_priority']}")
            return corrections
        else:
            print("[SUCCESS] Model performance is acceptable, no corrections needed")
            return existing_corrections
    
    def enhanced_prediction_with_corrections(self, base_prediction: float, game_data: Dict) -> Dict:
        """Generate enhanced prediction with bias corrections and confidence metrics"""
        
        # Load current corrections
        corrections = self.load_corrections()
        
        if not corrections:
            print("⚠️ No bias corrections available, using base prediction")
            return {
                'base_prediction': base_prediction,
                'corrected_prediction': base_prediction,
                'corrections_applied': [],
                'confidence_boost': 0.0
            }
        
        # Apply corrections
        corrected_prediction = self.apply_real_time_corrections(base_prediction, game_data, corrections)
        
        # Calculate confidence boost from corrections
        correction_magnitude = abs(corrected_prediction - base_prediction)
        confidence_boost = min(correction_magnitude * 0.1, 0.15)  # Max 15% boost
        
        return {
            'base_prediction': base_prediction,
            'corrected_prediction': corrected_prediction,
            'correction_magnitude': correction_magnitude,
            'confidence_boost': confidence_boost,
            'corrections_timestamp': corrections.get('timestamp', 'Unknown')
        }

def main():
    """Run model performance enhancement system"""
    enhancer = ModelPerformanceEnhancer()
    
    print("[MODEL PERFORMANCE ENHANCEMENT SYSTEM]")
    print("=" * 50)
    
    # Update corrections based on recent performance
    corrections = enhancer.update_model_corrections(force_update=True)
    
    if corrections:
        print(f"\n[SUMMARY] ACTIVE CORRECTIONS SUMMARY:")
        print(f"  • Global Adjustment: {corrections.get('global_adjustment', 0):.2f} runs")
        print(f"  • Scoring Range Adjustments: {len(corrections.get('scoring_range_adjustments', {}))}")
        print(f"  • Confidence Adjustments: {len(corrections.get('confidence_adjustments', {}))}")
        print(f"  • Temperature Adjustments: {len(corrections.get('temperature_adjustments', {}))}")
        print(f"  • Last Updated: {corrections.get('timestamp', 'Unknown')}")

if __name__ == "__main__":
    main()
