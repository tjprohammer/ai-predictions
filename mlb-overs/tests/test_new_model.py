#!/usr/bin/env python3

import sys
sys.path.append('mlb-overs/deployment')
from enhanced_bullpen_predictor import EnhancedBullpenPredictor
import pandas as pd

def test_new_model():
    """Test the newly trained model to see if predictions are realistic"""
    
    print('🎯 TESTING NEWLY CALIBRATED MODEL')
    print('='*50)
    
    # Initialize predictor (it will load the new model)
    predictor = EnhancedBullpenPredictor()
    
    # Test with a sample game with realistic stats
    mock_game = pd.DataFrame([{
        'game_id': 1234,
        'id': 1234,
        'home_team': 'LAA',
        'away_team': 'CIN', 
        'home_sp_era': 3.11,
        'away_sp_era': 4.59,
        'temperature': 78,
        'wind_speed': 13,
        'venue_id': 1,
        'home_sp_id': 579328,
        'away_sp_id': 607259,
        'market_total': 8.5,
        'home_team_runs_pg': 4.2,
        'away_team_runs_pg': 4.8,
        'home_sp_whip': 1.15,
        'away_sp_whip': 1.25,
        'home_sp_k_per_9': 9.5,
        'away_sp_k_per_9': 8.2,
        'home_sp_bb_per_9': 2.8,
        'away_sp_bb_per_9': 3.1,
        'home_sp_starts': 25,
        'away_sp_starts': 28
    }])
    
    try:
        # Generate features first
        features = predictor.engineer_features(mock_game)
        print(f'📋 Generated {len(features.columns)} features')
        
        # Make prediction directly with model
        prediction_raw = predictor.model.predict(features)[0]
        
        # Apply bias corrections
        final_prediction = prediction_raw + predictor.global_adjustment
        
        print(f'🔧 Raw model output: {prediction_raw:.2f} runs')
        print(f'⚡ Bias adjustment: +{predictor.global_adjustment:.2f} runs') 
        print(f'✅ Final prediction: {final_prediction:.2f} runs')
        print(f'📊 Market total: {mock_game.iloc[0]["market_total"]} runs')
        print(f'🎯 Difference: {final_prediction - mock_game.iloc[0]["market_total"]:.2f} runs')
        
        if 6.0 <= final_prediction <= 12.0:
            print('✅ Prediction is in realistic MLB range!')
            
            if abs(final_prediction - mock_game.iloc[0]["market_total"]) < 2.0:
                print('🎉 EXCELLENT: Prediction within 2 runs of market!')
            else:
                print(f'⚠️  Still {abs(final_prediction - mock_game.iloc[0]["market_total"]):.1f} runs from market')
        else:
            print('❌ Prediction still outside realistic range')
            
        # Test a few more scenarios
        print('\n🔬 TESTING MULTIPLE SCENARIOS:')
        
        test_scenarios = [
            {'name': 'High Offense Game', 'home_team_runs_pg': 5.5, 'away_team_runs_pg': 5.8, 'market_total': 10.5},
            {'name': 'Pitcher Duel', 'home_sp_era': 2.1, 'away_sp_era': 2.3, 'market_total': 7.0},
            {'name': 'Coors Field High', 'venue_id': 19, 'temperature': 85, 'market_total': 11.5}
        ]
        
        for scenario in test_scenarios:
            test_game = mock_game.copy()
            for key, value in scenario.items():
                if key != 'name':
                    test_game.iloc[0, test_game.columns.get_loc(key)] = value
            
            try:
                test_features = predictor.engineer_features(test_game)
                pred_raw = predictor.model.predict(test_features)[0]
                pred_final = pred_raw + predictor.global_adjustment
                diff = pred_final - test_game.iloc[0]['market_total']
                print(f"   {scenario['name']}: {pred_final:.1f} runs (market: {test_game.iloc[0]['market_total']}, diff: {diff:+.1f})")
            except Exception as e:
                print(f"   {scenario['name']}: ERROR - {e}")
        
    except Exception as e:
        print(f'❌ Error making prediction: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_model()
