#!/usr/bin/env python3
"""Test prediction with ERA=0.00 to identify NaN source."""

import pandas as pd
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

def main():
    print('🔍 Testing prediction with ERA=0.00...')
    
    # Create predictor
    predictor = EnhancedBullpenPredictor()
    
    # Test predict_today_games for the problematic date
    try:
        result = predictor.predict_today_games('2025-08-16')
        print(f'✅ Prediction succeeded')
        print(f'Number of predictions: {len(result)}')
        
        # Check for any NaN values in the features
        athletics_games = [game for game in result 
                          if 'Athletics' in game.get('home_team', '') or 'Athletics' in game.get('away_team', '')]
        
        if athletics_games:
            athletics_game = athletics_games[0]
            print(f'\n📊 Athletics game features:')
            print(f'   Home team: {athletics_game.get("home_team", "N/A")}')
            print(f'   Away team: {athletics_game.get("away_team", "N/A")}')
            
            # Check ERA-related features
            era_features = ['home_sp_era', 'away_sp_era', 'combined_sp_era', 'sp_era_differential']
            nan_count = 0
            for feature in era_features:
                value = athletics_game.get(feature, 'missing')
                if pd.isna(value):
                    print(f'   ❌ {feature}: NaN')
                    nan_count += 1
                else:
                    print(f'   ✅ {feature}: {value}')
            
            print(f'\n📋 ERA Features Summary: {len(era_features)-nan_count}/{len(era_features)} valid, {nan_count} NaN')
            
            # Check all features for NaN
            all_nan_features = []
            for key, value in athletics_game.items():
                if pd.isna(value):
                    all_nan_features.append(key)
            
            if all_nan_features:
                print(f'\n❌ All NaN features ({len(all_nan_features)}):')
                for feature in all_nan_features[:10]:  # Show first 10
                    print(f'   • {feature}')
                if len(all_nan_features) > 10:
                    print(f'   ... and {len(all_nan_features)-10} more')
            else:
                print(f'\n✅ No NaN features found!')
        else:
            print('No Athletics game found in predictions')
            
    except Exception as e:
        print(f'❌ Prediction failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
