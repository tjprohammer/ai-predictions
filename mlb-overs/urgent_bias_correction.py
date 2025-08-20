#!/usr/bin/env python3
"""
URGENT BIAS CORRECTION SYSTEM
Applies immediate bias correction to address -3.01 run under-prediction bias
"""

import psycopg2
import numpy as np
from datetime import datetime, date, timedelta

def analyze_recent_bias(days=14):
    """Analyze recent prediction bias to calculate correction factor"""
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )
    
    cursor = conn.cursor()
    
    # Get recent completed games for bias analysis
    start_date = (date.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    cursor.execute('''
    SELECT predicted_total, total_runs
    FROM enhanced_games 
    WHERE total_runs IS NOT NULL 
      AND predicted_total IS NOT NULL
      AND date >= %s
    ORDER BY date DESC
    ''', (start_date,))
    
    results = cursor.fetchall()
    
    if not results:
        print(f"❌ No completed games found in last {days} days")
        return None
    
    predictions = [float(row[0]) for row in results]
    actuals = [float(row[1]) for row in results]
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    bias = np.mean(predictions - actuals)  # Negative = under-predicting
    mae = np.mean(np.abs(predictions - actuals))
    count = len(predictions)
    
    print(f"📊 BIAS ANALYSIS ({count} games, last {days} days):")
    print(f"   Mean Bias: {bias:+.2f} runs")
    print(f"   MAE: {mae:.2f} runs")
    print(f"   Recommended Correction: {-bias:+.2f} runs")
    
    conn.close()
    
    return {
        'bias': bias,
        'correction': -bias,
        'mae': mae,
        'count': count,
        'confidence': min(100, max(0, 100 - abs(bias) * 10))  # Lower confidence for high bias
    }

def apply_bias_correction_to_future_games():
    """Apply bias correction to all future game predictions"""
    
    # Calculate current bias
    bias_analysis = analyze_recent_bias(days=14)
    
    if not bias_analysis or abs(bias_analysis['bias']) < 0.3:
        print("✅ No significant bias detected, no correction needed")
        return
    
    correction = bias_analysis['correction']
    
    print(f"\n🔧 APPLYING BIAS CORRECTION: {correction:+.2f} runs")
    
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )
    
    cursor = conn.cursor()
    
    # Get all future games (including today) that need correction
    today = date.today().strftime('%Y-%m-%d')
    
    cursor.execute('''
    SELECT game_id, predicted_total, market_total, edge
    FROM enhanced_games 
    WHERE date >= %s 
      AND total_runs IS NULL
      AND predicted_total IS NOT NULL
    ''', (today,))
    
    future_games = cursor.fetchall()
    
    if not future_games:
        print("❌ No future games found to correct")
        conn.close()
        return
    
    print(f"🎯 Correcting {len(future_games)} future game predictions...")
    
    corrected_count = 0
    
    for game_id, old_pred, market_total, old_edge in future_games:
        if old_pred is not None:
            old_pred_float = float(old_pred)
            new_pred = old_pred_float + correction
            
            # Recalculate edge with corrected prediction
            if market_total is not None:
                market_float = float(market_total)
                new_edge = new_pred - market_float
            else:
                new_edge = old_edge
            
            # Update the database
            cursor.execute('''
            UPDATE enhanced_games 
            SET predicted_total = %s,
                edge = %s,
                recommendation = CASE 
                    WHEN %s < -1.5 THEN 'UNDER'
                    WHEN %s > 1.5 THEN 'OVER'
                    ELSE 'HOLD'
                END
            WHERE game_id = %s
            ''', (new_pred, new_edge, new_edge, new_edge, game_id))
            
            corrected_count += 1
            
            if corrected_count <= 3:  # Show first few examples
                print(f"   Game {game_id}: {old_pred_float:.1f} → {new_pred:.1f} runs (edge: {old_edge} → {new_edge:.1f})")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Successfully corrected {corrected_count} predictions")
    print(f"📈 All predictions increased by {correction:.2f} runs")
    
    return {
        'corrected_games': corrected_count,
        'correction_applied': correction,
        'bias_analysis': bias_analysis
    }

def main():
    """Main execution"""
    print("🚨 URGENT BIAS CORRECTION SYSTEM")
    print("=" * 50)
    
    try:
        result = apply_bias_correction_to_future_games()
        
        if result:
            print(f"\n📊 CORRECTION SUMMARY:")
            print(f"   Bias Detected: {result['bias_analysis']['bias']:+.2f} runs")
            print(f"   Correction Applied: {result['correction_applied']:+.2f} runs")
            print(f"   Games Corrected: {result['corrected_games']}")
            print(f"   Model Confidence: {result['bias_analysis']['confidence']:.0f}%")
            
            print(f"\n✅ URGENT CORRECTION COMPLETE!")
            print(f"🔥 All future predictions now adjusted for {result['correction_applied']:+.2f} run bias")
        
    except Exception as e:
        print(f"❌ Error applying bias correction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
