#!/usr/bin/env python3

import joblib
import pandas as pd
from pathlib import Path

def analyze_model_feature_importance():
    """Check if the model properly weights starting pitcher performance"""
    try:
        models_dir = Path("../models")
        model_path = models_dir / "legitimate_model_latest.joblib"
        
        # Load the model
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print('=== TOP 15 MOST IMPORTANT FEATURES ===')
            print('(Checking if starting pitcher ERA gets proper weight)\n')
            
            for i, row in importance_df.head(15).iterrows():
                print(f'{row["importance"]:.4f} - {row["feature"]}')
                
                # Highlight pitcher-related features
                if any(term in row["feature"].lower() for term in ['era', 'pitcher', 'whip']):
                    if 'bullpen' not in row["feature"].lower():
                        print('         ^^^^ STARTING PITCHER FEATURE')
                    else:
                        print('         ^^^^ BULLPEN FEATURE')
            
            print('\n=== STARTING PITCHER FEATURE ANALYSIS ===')
            pitcher_features = importance_df[
                importance_df['feature'].str.contains('pitcher|era|whip', case=False) &
                ~importance_df['feature'].str.contains('bullpen', case=False)
            ]
            
            total_pitcher_importance = pitcher_features['importance'].sum()
            print(f'Total starting pitcher feature importance: {total_pitcher_importance:.4f}')
            
            if total_pitcher_importance < 0.15:
                print('❌ PROBLEM: Starting pitcher features have very low importance!')
                print('   This explains why Skubal\'s dominance isn\'t reflected in predictions.')
            else:
                print('✅ Starting pitcher features have reasonable importance.')
            
            print('\nTop starting pitcher features:')
            for _, row in pitcher_features.head(5).iterrows():
                print(f'  {row["importance"]:.4f} - {row["feature"]}')
        
        else:
            print('Model does not have feature_importances_ attribute')
            
    except Exception as e:
        print(f"Error analyzing model: {e}")

if __name__ == "__main__":
    analyze_model_feature_importance()
