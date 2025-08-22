#!/usr/bin/env python3
"""
Historical Dual Prediction Analysis
===================================
Analyze historical performance of both models to build trust in the learning model
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import json
from datetime import datetime, timedelta

def get_historical_performance(days_back=30):
    """Get historical performance comparison"""
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    query = text("""
    SELECT 
        date,
        game_id,
        home_team,
        away_team,
        market_total,
        predicted_total,
        predicted_total_original,
        predicted_total_learning,
        total_runs,
        
        -- Calculate errors
        CASE 
            WHEN predicted_total_original IS NOT NULL AND total_runs IS NOT NULL 
            THEN ABS(predicted_total_original - total_runs)
            ELSE NULL 
        END as original_error,
        
        CASE 
            WHEN predicted_total_learning IS NOT NULL AND total_runs IS NOT NULL 
            THEN ABS(predicted_total_learning - total_runs)
            ELSE NULL 
        END as learning_error,
        
        CASE 
            WHEN market_total IS NOT NULL AND total_runs IS NOT NULL 
            THEN ABS(market_total - total_runs)
            ELSE NULL 
        END as market_error,
        
        -- Winner determination
        CASE 
            WHEN predicted_total_original IS NOT NULL AND predicted_total_learning IS NOT NULL AND total_runs IS NOT NULL
            THEN (ABS(predicted_total_learning - total_runs) < ABS(predicted_total_original - total_runs))
            ELSE NULL
        END as learning_wins
        
    FROM enhanced_games
    WHERE date BETWEEN :start_date AND :end_date
    AND total_runs IS NOT NULL
    ORDER BY date DESC, game_id
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
    
    return df

def analyze_model_performance(df):
    """Analyze performance metrics"""
    
    print('📊 HISTORICAL DUAL MODEL PERFORMANCE ANALYSIS')
    print('='*70)
    
    if df.empty:
        print("❌ No historical data with completed games found")
        return None
    
    # Filter to games with both predictions and results
    both_models = df[(df['predicted_total_original'].notna()) & 
                     (df['predicted_total_learning'].notna()) & 
                     (df['total_runs'].notna())]
    
    if both_models.empty:
        print("❌ No games with both model predictions and results found")
        return None
    
    print(f"📈 Analysis Period: {df['date'].min()} to {df['date'].max()}")
    print(f"🎯 Games Analyzed: {len(both_models)} completed games with both predictions\n")
    
    # Overall accuracy metrics
    orig_mae = both_models['original_error'].mean()
    learn_mae = both_models['learning_error'].mean()
    market_mae = both_models['market_error'].mean() if both_models['market_error'].notna().any() else None
    
    print('🎯 MEAN ABSOLUTE ERROR (MAE) - Lower is Better')
    print('-' * 50)
    print(f"🔵 Original Model: {orig_mae:.3f} runs")
    print(f"🟢 Learning Model: {learn_mae:.3f} runs")
    if market_mae:
        print(f"📊 Market Lines:   {market_mae:.3f} runs")
    
    improvement = ((orig_mae - learn_mae) / orig_mae) * 100
    print(f"\n💡 Learning Model Improvement: {improvement:+.1f}%")
    
    # Head-to-head comparison
    learning_wins = both_models['learning_wins'].sum()
    original_wins = len(both_models) - learning_wins
    win_rate = (learning_wins / len(both_models)) * 100
    
    print(f"\n🥊 HEAD-TO-HEAD COMPARISON")
    print('-' * 30)
    print(f"🟢 Learning Model Wins: {learning_wins} ({win_rate:.1f}%)")
    print(f"🔵 Original Model Wins: {original_wins} ({100-win_rate:.1f}%)")
    
    # Performance by game total ranges
    print(f"\n📊 PERFORMANCE BY GAME TOTAL RANGES")
    print('-' * 40)
    
    ranges = [
        (0, 7, "Low Scoring (≤7)"),
        (7.5, 9, "Medium Scoring (7.5-9)"), 
        (9.5, 15, "High Scoring (≥9.5)")
    ]
    
    for min_val, max_val, label in ranges:
        range_df = both_models[(both_models['total_runs'] >= min_val) & 
                              (both_models['total_runs'] <= max_val)]
        
        if len(range_df) > 0:
            orig_mae_range = range_df['original_error'].mean()
            learn_mae_range = range_df['learning_error'].mean()
            learn_wins_range = range_df['learning_wins'].sum()
            
            print(f"{label}:")
            print(f"  Games: {len(range_df)}")
            print(f"  Original MAE: {orig_mae_range:.3f}")
            print(f"  Learning MAE: {learn_mae_range:.3f}")
            print(f"  Learning Win Rate: {(learn_wins_range/len(range_df)*100):.1f}%")
    
    # Recent performance trends
    print(f"\n📈 RECENT PERFORMANCE TREND (Last 7 Days)")
    print('-' * 45)
    
    recent_cutoff = datetime.now().date() - timedelta(days=7)
    recent_df = both_models[both_models['date'] >= recent_cutoff]
    
    if len(recent_df) > 0:
        recent_orig_mae = recent_df['original_error'].mean()
        recent_learn_mae = recent_df['learning_error'].mean()
        recent_wins = recent_df['learning_wins'].sum()
        
        print(f"Recent Games: {len(recent_df)}")
        print(f"Original MAE: {recent_orig_mae:.3f}")
        print(f"Learning MAE: {recent_learn_mae:.3f}")
        print(f"Learning Win Rate: {(recent_wins/len(recent_df)*100):.1f}%")
    else:
        print("No recent completed games found")
    
    # Biggest differences analysis
    print(f"\n🔥 BIGGEST PREDICTION DIFFERENCES")
    print('-' * 35)
    
    both_models['prediction_diff'] = abs(both_models['predicted_total_learning'] - 
                                        both_models['predicted_total_original'])
    
    big_diffs = both_models.nlargest(3, 'prediction_diff')
    
    for _, row in big_diffs.iterrows():
        diff = row['predicted_total_learning'] - row['predicted_total_original']
        actual = row['total_runs']
        orig_error = abs(row['predicted_total_original'] - actual)
        learn_error = abs(row['predicted_total_learning'] - actual)
        
        print(f"\n{row['home_team']} vs {row['away_team']} ({row['date']})")
        print(f"  Original: {row['predicted_total_original']:.2f} (Error: {orig_error:.2f})")
        print(f"  Learning: {row['predicted_total_learning']:.2f} (Error: {learn_error:.2f})")
        print(f"  Actual: {actual:.0f}")
        print(f"  Difference: {diff:+.2f} ({'Learning' if learn_error < orig_error else 'Original'} was better)")
    
    return {
        'total_games': len(both_models),
        'original_mae': orig_mae,
        'learning_mae': learn_mae,
        'improvement_pct': improvement,
        'learning_win_rate': win_rate,
        'learning_wins': learning_wins,
        'original_wins': original_wins
    }

def create_trust_metrics():
    """Create specific metrics to build trust in the learning model"""
    
    print(f"\n🛡️ LEARNING MODEL TRUST METRICS")
    print('='*45)
    
    df = get_historical_performance(30)
    
    if df.empty:
        print("❌ No data available for trust analysis")
        return
    
    # Filter completed games with both predictions
    both_models = df[(df['predicted_total_original'].notna()) & 
                     (df['predicted_total_learning'].notna()) & 
                     (df['total_runs'].notna())]
    
    if both_models.empty:
        print("❌ No completed games with both predictions")
        return
    
    # Trust Metric 1: Consistency 
    prediction_variance = both_models['predicted_total_learning'].std()
    print(f"📊 Prediction Consistency:")
    print(f"   Learning model std dev: {prediction_variance:.3f}")
    print(f"   {'✅ Stable' if prediction_variance < 1.5 else '⚠️ High variance'}")
    
    # Trust Metric 2: No extreme outliers
    q25 = both_models['predicted_total_learning'].quantile(0.25)
    q75 = both_models['predicted_total_learning'].quantile(0.75)
    iqr = q75 - q25
    outliers = both_models[
        (both_models['predicted_total_learning'] < q25 - 1.5*iqr) |
        (both_models['predicted_total_learning'] > q75 + 1.5*iqr)
    ]
    
    print(f"\n📈 Outlier Analysis:")
    print(f"   Outlier predictions: {len(outliers)}/{len(both_models)} ({len(outliers)/len(both_models)*100:.1f}%)")
    print(f"   {'✅ Reasonable' if len(outliers)/len(both_models) < 0.1 else '⚠️ Too many outliers'}")
    
    # Trust Metric 3: Performance stability over time
    daily_performance = both_models.groupby('date').agg({
        'learning_error': 'mean',
        'original_error': 'mean'
    }).reset_index()
    
    if len(daily_performance) > 1:
        learning_performance_std = daily_performance['learning_error'].std()
        print(f"\n📅 Daily Performance Stability:")
        print(f"   Daily MAE std dev: {learning_performance_std:.3f}")
        print(f"   {'✅ Stable performance' if learning_performance_std < 0.5 else '⚠️ Inconsistent performance'}")
    
    # Trust Metric 4: Reasonable prediction range
    min_pred = both_models['predicted_total_learning'].min()
    max_pred = both_models['predicted_total_learning'].max()
    
    print(f"\n🎯 Prediction Range:")
    print(f"   Range: {min_pred:.1f} - {max_pred:.1f} runs")
    reasonable = min_pred >= 4.0 and max_pred <= 15.0
    print(f"   {'✅ Reasonable range' if reasonable else '⚠️ Extreme predictions'}")
    
    # Overall trust score
    trust_factors = [
        prediction_variance < 1.5,
        len(outliers)/len(both_models) < 0.1,
        reasonable
    ]
    
    if len(daily_performance) > 1:
        trust_factors.append(learning_performance_std < 0.5)
    
    trust_score = sum(trust_factors) / len(trust_factors)
    
    print(f"\n🏆 OVERALL TRUST SCORE: {trust_score*100:.0f}%")
    if trust_score >= 0.8:
        print("   ✅ HIGH TRUST - Learning model appears reliable")
    elif trust_score >= 0.6:
        print("   ⚠️ MEDIUM TRUST - Learning model needs monitoring")
    else:
        print("   ❌ LOW TRUST - Learning model needs investigation")

def save_analysis_json():
    """Save analysis results to JSON for UI consumption"""
    
    df = get_historical_performance(30)
    analysis = analyze_model_performance(df)
    
    if analysis:
        # Add current predictions for context
        current_query = text("""
        SELECT 
            game_id, home_team, away_team, 
            predicted_total_original, predicted_total_learning,
            market_total, total_runs
        FROM enhanced_games 
        WHERE date = CURRENT_DATE
        AND (predicted_total_original IS NOT NULL OR predicted_total_learning IS NOT NULL)
        """)
        
        engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
        with engine.connect() as conn:
            current_df = pd.read_sql(current_query, conn)
        
        output = {
            'analysis_date': datetime.now().isoformat(),
            'historical_performance': analysis,
            'current_predictions': current_df.to_dict('records'),
            'summary': {
                'learning_model_better': analysis['learning_mae'] < analysis['original_mae'],
                'confidence_level': 'HIGH' if analysis['learning_win_rate'] > 60 else 'MEDIUM' if analysis['learning_win_rate'] > 40 else 'LOW',
                'recommendation': 'USE_LEARNING' if analysis['improvement_pct'] > 5 else 'USE_BOTH' if analysis['improvement_pct'] > -5 else 'USE_ORIGINAL'
            }
        }
        
        # Save to file
        output_file = 'historical_dual_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Analysis saved to: {output_file}")
        return output_file
    
    return None

if __name__ == "__main__":
    # Run complete historical analysis
    df = get_historical_performance(30)
    analyze_model_performance(df)
    create_trust_metrics()
    save_analysis_json()
