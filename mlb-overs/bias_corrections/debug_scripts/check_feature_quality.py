"""
Debug script to check if the 170 features actually have meaningful data
and aren't just NaN/null values that could cause bad predictions.
"""

import pandas as pd
import numpy as np
import psycopg2
import sys
sys.path.append('s:\\Projects\\AI_Predictions\\mlb-overs\\deployment')

from enhanced_bullpen_predictor import EnhancedBullpenPredictor

def check_feature_data_quality():
    """Check the actual feature data quality for today's predictions"""
    
    print("🔍 CHECKING FEATURE DATA QUALITY FOR 170 FEATURES")
    print("=" * 60)
    
    # Initialize the predictor to get the same feature pipeline
    predictor = EnhancedBullpenPredictor()
    
    # Get today's data just like the prediction system does
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )
    
    # Get the same raw data the predictor uses
    query = """
    SELECT *
    FROM enhanced_games 
    WHERE date = '2025-08-20'
    ORDER BY game_id
    LIMIT 1
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        print("❌ No games found for 2025-08-20")
        return
    
    print(f"📊 Analyzing features for: {df.iloc[0]['away_team']} @ {df.iloc[0]['home_team']}")
    print()
    
    # Try to replicate the feature engineering process
    try:
        # This should give us the same features the model sees
        featured_df = predictor.engineer_features(df)
        
        print(f"✅ Feature engineering completed: {featured_df.shape[1]} total features")
        print()
        
        # Analyze feature quality
        total_features = featured_df.shape[1]
        total_null = featured_df.isnull().sum().sum()
        total_inf = np.isinf(featured_df.select_dtypes(include=[np.number])).sum().sum()
        total_zero = (featured_df.select_dtypes(include=[np.number]) == 0).sum().sum()
        
        print(f"📈 FEATURE QUALITY SUMMARY:")
        print(f"   Total features: {total_features}")
        print(f"   Total null values: {total_null}")
        print(f"   Total infinite values: {total_inf}")
        print(f"   Total zero values: {total_zero}")
        print()
        
        # Check features with too many nulls
        null_counts = featured_df.isnull().sum()
        high_null_features = null_counts[null_counts > 0].sort_values(ascending=False)
        
        if len(high_null_features) > 0:
            print(f"⚠️  FEATURES WITH NULL VALUES:")
            for feature, null_count in high_null_features.head(10).items():
                print(f"   {feature}: {null_count} nulls")
            print()
        
        # Check features with zero variance (constant values)
        numeric_features = featured_df.select_dtypes(include=[np.number])
        zero_variance = numeric_features.std() == 0
        constant_features = zero_variance[zero_variance].index.tolist()
        
        if len(constant_features) > 0:
            print(f"⚠️  CONSTANT FEATURES (zero variance):")
            for feature in constant_features[:10]:
                value = numeric_features[feature].iloc[0]
                print(f"   {feature}: constant value = {value}")
            print()
        
        # Check features with very low variance
        low_variance = numeric_features.std() < 0.01
        low_var_features = low_variance[low_variance].index.tolist()
        low_var_features = [f for f in low_var_features if f not in constant_features]
        
        if len(low_var_features) > 0:
            print(f"⚠️  LOW VARIANCE FEATURES (std < 0.01):")
            for feature in low_var_features[:10]:
                std_val = numeric_features[feature].std()
                mean_val = numeric_features[feature].mean()
                print(f"   {feature}: std={std_val:.6f}, mean={mean_val:.3f}")
            print()
        
        # Check for reasonable value ranges on key features
        key_features = [
            'home_sp_era', 'away_sp_era', 'combined_era',
            'home_team_runs_pg', 'away_team_runs_pg', 'combined_team_offense',
            'market_total', 'expected_total', 'ballpark_run_factor'
        ]
        
        print(f"🎯 KEY FEATURE VALUES:")
        for feature in key_features:
            if feature in featured_df.columns:
                value = featured_df[feature].iloc[0]
                print(f"   {feature}: {value}")
        print()
        
        # Get the actual feature array that would go to the model
        # Drop non-numeric and ID columns like the real predictor does
        feature_cols = [col for col in featured_df.columns 
                       if col not in ['game_id', 'date', 'home_team', 'away_team', 'total_runs']]
        
        X = featured_df[feature_cols].select_dtypes(include=[np.number])
        
        # Replace inf with NaN and fill NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"🔧 MODEL INPUT FEATURES:")
        print(f"   Features going to model: {X.shape[1]}")
        print(f"   Sample feature values (first 10):")
        for i, col in enumerate(X.columns[:10]):
            value = X[col].iloc[0]
            print(f"     {col}: {value:.6f}")
        print()
        
        # Check if any critical features are missing or have bad values
        critical_missing = []
        if 'market_total' not in featured_df.columns or featured_df['market_total'].isnull().iloc[0]:
            critical_missing.append('market_total')
        if 'combined_era' not in featured_df.columns or featured_df['combined_era'].isnull().iloc[0]:
            critical_missing.append('combined_era')
        if 'combined_team_offense' not in featured_df.columns or featured_df['combined_team_offense'].isnull().iloc[0]:
            critical_missing.append('combined_team_offense')
            
        if critical_missing:
            print(f"❌ CRITICAL FEATURES MISSING:")
            for feature in critical_missing:
                print(f"   {feature}")
            print()
        else:
            print("✅ All critical features present")
            print()
            
        # Make a raw prediction to see what the model actually outputs
        try:
            raw_prediction = predictor.model.predict(X)[0]
            print(f"🎯 RAW MODEL OUTPUT: {raw_prediction:.3f} runs")
            print(f"   (This is BEFORE bias corrections)")
            print()
            
            # Apply bias corrections manually to see the math
            bias_corrections = predictor.bias_corrections
            global_adj = bias_corrections.get('global_adjustment', 0)
            final_prediction = raw_prediction + global_adj
            
            print(f"🔧 BIAS CORRECTION MATH:")
            print(f"   Raw prediction: {raw_prediction:.3f}")
            print(f"   Global adjustment: +{global_adj:.3f}")
            print(f"   Final prediction: {final_prediction:.3f}")
            print()
            
            # Compare to market
            market_total = featured_df['market_total'].iloc[0]
            difference = final_prediction - market_total
            print(f"📊 MARKET COMPARISON:")
            print(f"   Market total: {market_total:.1f}")
            print(f"   Our prediction: {final_prediction:.1f}")
            print(f"   Difference: {difference:+.1f} runs")
            
            if abs(difference) > 2:
                print(f"   ⚠️  Large gap! This suggests model calibration issues.")
            
        except Exception as e:
            print(f"❌ Error making raw prediction: {e}")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_feature_data_quality()
