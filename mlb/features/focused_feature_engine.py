#!/usr/bin/env python3
"""
Focused MLB Feature Engine
=========================

Based on domain expertise, this engine focuses on the most predictive features
for MLB run totals, properly weighted by baseball importance.

Key Categories:
1. Starting Pitcher Performance (40% weight)
2. Team Offensive Stats (25% weight) 
3. Bullpen Performance (20% weight)
4. Weather & Ballpark (10% weight)
5. Recent Form & Momentum (5% weight)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple, Optional
import logging

log = logging.getLogger(__name__)

class FocusedFeatureEngine:
    """Focused feature engine using domain expertise"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
        self.engine = create_engine(database_url)
        
        # Define feature importance weights based on baseball domain knowledge
        self.feature_weights = {
            # Starting Pitcher Performance (40% total weight)
            'pitcher_features': {
                'home_sp_season_era': 0.08,
                'away_sp_season_era': 0.08,
                'home_sp_whip': 0.06,
                'away_sp_whip': 0.06,
                'home_sp_season_k': 0.04,
                'away_sp_season_k': 0.04,
                'home_sp_season_bb': 0.02,
                'away_sp_season_bb': 0.02,
                'total_weight': 0.40
            },
            
            # Team Offensive Stats (25% total weight)
            'offensive_features': {
                'home_team_ops': 0.05,
                'away_team_ops': 0.05,
                'home_team_woba': 0.03,
                'away_team_woba': 0.03,
                'home_team_runs_pg': 0.03,
                'away_team_runs_pg': 0.03,
                'home_team_avg': 0.015,
                'away_team_avg': 0.015,
                'home_team_slg': 0.01,
                'away_team_slg': 0.01,
                'total_weight': 0.25
            },
            
            # Bullpen Performance (20% total weight)
            'bullpen_features': {
                'home_bullpen_era': 0.05,
                'away_bullpen_era': 0.05,
                'home_bullpen_era_l30': 0.03,
                'away_bullpen_era_l30': 0.03,
                'home_bullpen_whip_l30': 0.02,
                'away_bullpen_whip_l30': 0.02,
                'total_weight': 0.20
            },
            
            # Weather & Ballpark (10% total weight)
            'environment_features': {
                'temperature': 0.03,
                'wind_speed': 0.02,
                'ballpark_run_factor': 0.025,
                'ballpark_hr_factor': 0.025,
                'total_weight': 0.10
            },
            
            # Recent Form & Momentum (5% total weight)
            'form_features': {
                'home_team_runs_l7': 0.01,
                'away_team_runs_l7': 0.01,
                'home_team_ops_l14': 0.01,
                'away_team_ops_l14': 0.01,
                'home_sp_era_l3starts': 0.005,
                'away_sp_era_l3starts': 0.005,
                'total_weight': 0.05
            }
        }
        
    def get_available_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check which of our target features are available in the dataframe"""
        
        available_features = {}
        
        for category, features in self.feature_weights.items():
            if category.endswith('_features'):
                available = []
                for feature_name in features.keys():
                    if feature_name != 'total_weight' and feature_name in df.columns:
                        available.append(feature_name)
                available_features[category] = available
                
        return available_features
    
    def create_weighted_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features using domain-knowledge weights"""
        
        df_enhanced = df.copy()
        available_features = self.get_available_features(df)
        
        log.info("Creating weighted composite features...")
        
        # 1. Starting Pitcher Strength Score (40% weight)
        pitcher_components = []
        pitcher_weights = []
        
        pitcher_features = available_features.get('pitcher_features', [])
        for feature in pitcher_features:
            if feature in df.columns and feature != 'total_weight':
                # For ERA/WHIP/BB - lower is better, so invert
                if any(x in feature.lower() for x in ['era', 'whip', 'bb']):
                    # Convert to strength score (5.0 - value, capped at reasonable range)
                    if 'era' in feature.lower():
                        component = np.clip(5.0 - df[feature].fillna(4.5), 0, 5)
                    elif 'whip' in feature.lower():
                        component = np.clip(2.0 - df[feature].fillna(1.3), 0, 2)
                    else:  # bb
                        component = np.clip(5.0 - df[feature].fillna(3.0), 0, 5)
                else:  # K - higher is better
                    component = np.clip(df[feature].fillna(8.0) / 2.0, 0, 5)
                
                pitcher_components.append(component)
                pitcher_weights.append(self.feature_weights['pitcher_features'][feature])
        
        if pitcher_components:
            df_enhanced['pitcher_strength_composite'] = np.average(
                pitcher_components, axis=0, weights=pitcher_weights
            )
        else:
            df_enhanced['pitcher_strength_composite'] = 3.0  # Neutral
            
        # 2. Team Offensive Power Score (25% weight)
        offensive_components = []
        offensive_weights = []
        
        offensive_features = available_features.get('offensive_features', [])
        for feature in offensive_features:
            if feature in df.columns:
                # Normalize different offensive stats to 0-5 scale
                if 'ops' in feature.lower():
                    component = np.clip((df[feature].fillna(0.75) - 0.6) * 10, 0, 5)
                elif 'woba' in feature.lower():
                    component = np.clip((df[feature].fillna(0.32) - 0.28) * 25, 0, 5)
                elif 'avg' in feature.lower():
                    component = np.clip((df[feature].fillna(0.25) - 0.20) * 20, 0, 5)
                elif 'runs_pg' in feature.lower():
                    component = np.clip(df[feature].fillna(4.5) - 2.0, 0, 5)
                else:  # slg, etc.
                    component = np.clip((df[feature].fillna(0.42) - 0.35) * 10, 0, 5)
                
                offensive_components.append(component)
                offensive_weights.append(self.feature_weights['offensive_features'][feature])
        
        if offensive_components:
            df_enhanced['offensive_power_composite'] = np.average(
                offensive_components, axis=0, weights=offensive_weights
            )
        else:
            df_enhanced['offensive_power_composite'] = 3.0  # Neutral
            
        # 3. Bullpen Strength Score (20% weight)
        bullpen_components = []
        bullpen_weights = []
        
        bullpen_features = available_features.get('bullpen_features', [])
        for feature in bullpen_features:
            if feature in df.columns:
                # For bullpen ERA/WHIP - lower is better
                if 'era' in feature.lower():
                    component = np.clip(5.5 - df[feature].fillna(4.0), 0, 5)
                elif 'whip' in feature.lower():
                    component = np.clip(2.2 - df[feature].fillna(1.3), 0, 5)
                else:
                    component = np.clip(df[feature].fillna(3.0), 0, 5)
                
                bullpen_components.append(component)
                bullpen_weights.append(self.feature_weights['bullpen_features'][feature])
        
        if bullpen_components:
            df_enhanced['bullpen_strength_composite'] = np.average(
                bullpen_components, axis=0, weights=bullpen_weights
            )
        else:
            df_enhanced['bullpen_strength_composite'] = 3.0  # Neutral
            
        # 4. Environmental Impact Score (10% weight)
        env_score = 3.0  # Start neutral
        
        # Temperature effect (higher temps = more offense)
        if 'temperature' in df.columns:
            temp_effect = np.clip((df['temperature'].fillna(72) - 65) * 0.05, -1, 1)
            env_score += temp_effect
        
        # Wind effect (wind speed matters, direction is complex)
        if 'wind_speed' in df.columns:
            wind_effect = np.clip((df['wind_speed'].fillna(8) - 8) * 0.1, -0.5, 0.5)
            env_score += wind_effect
            
        # Ballpark effects
        if 'ballpark_run_factor' in df.columns:
            park_effect = (df['ballpark_run_factor'].fillna(1.0) - 1.0) * 2
            env_score += park_effect
            
        df_enhanced['environmental_impact_composite'] = np.clip(env_score, 1, 5)
        
        # 5. Recent Form Momentum Score (5% weight)
        form_components = []
        form_weights = []
        
        form_features = available_features.get('form_features', [])
        for feature in form_features:
            if feature in df.columns:
                if 'runs' in feature.lower():
                    component = np.clip(df[feature].fillna(4.5) - 2.0, 0, 5)
                elif 'ops' in feature.lower():
                    component = np.clip((df[feature].fillna(0.75) - 0.6) * 10, 0, 5)
                elif 'era' in feature.lower():
                    component = np.clip(6.0 - df[feature].fillna(4.5), 0, 5)
                else:
                    component = np.clip(df[feature].fillna(3.0), 0, 5)
                
                form_components.append(component)
                form_weights.append(self.feature_weights['form_features'][feature])
        
        if form_components:
            df_enhanced['recent_form_composite'] = np.average(
                form_components, axis=0, weights=form_weights
            )
        else:
            df_enhanced['recent_form_composite'] = 3.0  # Neutral
            
        # 6. Master Prediction Score (weighted combination)
        master_score = (
            df_enhanced['pitcher_strength_composite'] * 0.40 +
            df_enhanced['offensive_power_composite'] * 0.25 +
            df_enhanced['bullpen_strength_composite'] * 0.20 +
            df_enhanced['environmental_impact_composite'] * 0.10 +
            df_enhanced['recent_form_composite'] * 0.05
        )
        
        # Convert to expected runs (scale to reasonable MLB range)
        df_enhanced['expected_total_runs'] = np.clip(master_score * 2.2 + 1.0, 6.0, 16.0)
        
        # Add individual vs combined metrics
        if all(col in df.columns for col in ['home_team_ops', 'away_team_ops']):
            df_enhanced['combined_team_ops'] = (
                df['home_team_ops'].fillna(0.75) + df['away_team_ops'].fillna(0.75)
            ) / 2
            
        if all(col in df.columns for col in ['home_sp_season_era', 'away_sp_season_era']):
            df_enhanced['combined_sp_era'] = (
                df['home_sp_season_era'].fillna(4.5) + df['away_sp_season_era'].fillna(4.5)
            ) / 2
            
        # Interaction features (most important combinations)
        if all(col in df.columns for col in ['temperature', 'ballpark_hr_factor']):
            df_enhanced['temp_ballpark_interaction'] = (
                (df['temperature'].fillna(72) - 70) * df['ballpark_hr_factor'].fillna(1.0)
            )
            
        if all(col in df.columns for col in ['wind_speed', 'ballpark_run_factor']):
            df_enhanced['wind_park_interaction'] = (
                df['wind_speed'].fillna(8) * df['ballpark_run_factor'].fillna(1.0)
            )
        
        log.info(f"Created {len([c for c in df_enhanced.columns if c not in df.columns])} new composite features")
        
        return df_enhanced
    
    def get_feature_importance_ranking(self, df: pd.DataFrame) -> List[Tuple[str, float, str]]:
        """Return features ranked by domain expertise importance"""
        
        feature_rankings = []
        
        # Add all weighted features with their importance scores
        for category, features in self.feature_weights.items():
            if category.endswith('_features'):
                category_name = category.replace('_features', '').title()
                for feature_name, weight in features.items():
                    if feature_name != 'total_weight' and feature_name in df.columns:
                        feature_rankings.append((feature_name, weight, category_name))
        
        # Add composite features (highest importance)
        composite_features = [
            ('expected_total_runs', 1.0, 'Master Prediction'),
            ('pitcher_strength_composite', 0.40, 'Pitcher Composite'),
            ('offensive_power_composite', 0.25, 'Offensive Composite'),
            ('bullpen_strength_composite', 0.20, 'Bullpen Composite'),
            ('environmental_impact_composite', 0.10, 'Environmental Composite'),
            ('recent_form_composite', 0.05, 'Form Composite')
        ]
        
        for feature_name, weight, category in composite_features:
            if feature_name in df.columns:
                feature_rankings.append((feature_name, weight, category))
        
        # Sort by importance weight (descending)
        feature_rankings.sort(key=lambda x: x[1], reverse=True)
        
        return feature_rankings
    
    def process_focused_features(self, df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
        """Main processing function - create focused, weighted features"""
        
        log.info(f"Processing focused features for {len(df)} games...")
        
        # Get available features
        available_features = self.get_available_features(df)
        
        log.info("Available feature categories:")
        for category, features in available_features.items():
            total_weight = self.feature_weights[category].get('total_weight', 0)
            log.info(f"  {category}: {len(features)} features (weight: {total_weight:.0%})")
        
        # Create weighted composite features
        df_enhanced = self.create_weighted_composite_features(df)
        
        # Get final feature rankings
        feature_rankings = self.get_feature_importance_ranking(df_enhanced)
        
        log.info(f"Top 10 most important features:")
        for i, (feature, weight, category) in enumerate(feature_rankings[:10]):
            log.info(f"  {i+1:2d}. {feature:<30} ({weight:.3f}) [{category}]")
        
        log.info(f"Enhanced dataset: {len(df.columns)} → {len(df_enhanced.columns)} features (+{len(df_enhanced.columns) - len(df.columns)})")
        
        return df_enhanced

def main():
    """Test the focused feature engine"""
    
    import sys
    from datetime import datetime
    
    logging.basicConfig(level=logging.INFO)
    
    target_date = sys.argv[1] if len(sys.argv) > 1 else '2025-08-30'
    
    # Initialize engine
    engine = FocusedFeatureEngine()
    
    # Test with sample data
    with engine.engine.connect() as conn:
        query = text("""
            SELECT * FROM enhanced_games 
            WHERE date = :target_date 
            LIMIT 10
        """)
        
        df = pd.read_sql(query, conn, params={'target_date': target_date})
    
    if df.empty:
        print(f"No data found for {target_date}")
        return
    
    print(f"Testing focused feature engine with {len(df)} games from {target_date}")
    
    # Process features
    df_enhanced = engine.process_focused_features(df, target_date)
    
    # Show results
    print(f"\nFeature processing complete!")
    print(f"Original features: {len(df.columns)}")
    print(f"Enhanced features: {len(df_enhanced.columns)}")
    print(f"New features added: {len(df_enhanced.columns) - len(df.columns)}")
    
    # Show sample predictions
    if 'expected_total_runs' in df_enhanced.columns:
        print(f"\nSample predictions:")
        for i, row in df_enhanced.head(5).iterrows():
            home_team = row.get('home_team', 'Home')
            away_team = row.get('away_team', 'Away')
            pred = row['expected_total_runs']
            print(f"  {away_team} @ {home_team}: {pred:.1f} runs")

if __name__ == "__main__":
    main()
