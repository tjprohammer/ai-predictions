"""
Debug bias correction application to see what's actually happening
"""

import psycopg2
import json

def debug_bias_corrections():
    """Debug the bias correction application"""
    
    # Load bias corrections
    with open('s:\\Projects\\AI_Predictions\\model_bias_corrections.json', 'r') as f:
        corrections = json.load(f)
    
    print("🔍 DEBUGGING BIAS CORRECTIONS")
    print("=" * 50)
    print(f"Global Adjustment: {corrections.get('global_adjustment', 0)}")
    print(f"Scoring Range Adjustments:")
    for range_name, adj in corrections.get('scoring_range_adjustments', {}).items():
        print(f"  {range_name}: {adj:+.2f}")
    
    # Get current predictions from database
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT home_team, away_team, predicted_total, market_total, 
               predicted_total - market_total as gap
        FROM enhanced_games 
        WHERE date = '2025-08-20'
        ORDER BY predicted_total
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    print(f"\n📊 CURRENT PREDICTION ANALYSIS:")
    print("-" * 70)
    total_gap = 0
    for home, away, pred, market, gap in results:
        print(f"{away[:15]:<15} @ {home[:15]:<15} | {pred:5.1f} vs {market:4.1f} | Gap: {gap:+5.1f}")
        total_gap += gap
    
    avg_gap = total_gap / len(results)
    print(f"\nAverage gap: {avg_gap:+.2f} runs")
    print(f"All predictions are below market by an average of {abs(avg_gap):.2f} runs")
    
    # Simulate what should happen with raw model predictions
    print(f"\n🧮 BIAS CORRECTION SIMULATION:")
    print("-" * 50)
    
    print("If raw model predictions were ~3 runs lower and +3.0 was applied:")
    for home, away, pred, market, gap in results:
        # Convert decimal to float for calculations
        pred_float = float(pred)
        market_float = float(market)
        
        # What the raw prediction might have been before +3.0 adjustment
        estimated_raw = pred_float - 3.0
        # What it should be after +3.0 adjustment (which should equal current pred)
        should_be_after_correction = estimated_raw + 3.0
        
        print(f"  {away[:12]:<12} @ {home[:12]:<12} | Est Raw: {estimated_raw:4.1f} → +3.0 → {should_be_after_correction:4.1f} | Current: {pred_float:4.1f} | Market: {market_float:4.1f}")
        
        # Check if current prediction matches what we'd expect after +3.0
        if abs(pred_float - should_be_after_correction) > 0.1:
            print(f"    ⚠️  MISMATCH: Expected {should_be_after_correction:4.1f} after +3.0, got {pred_float:4.1f}")
    
    print(f"\n⚠️  ANALYSIS:")
    print(f"If +3.0 global adjustment was properly applied, we should see predictions")
    print(f"that are much closer to market. The fact that we're still seeing")
    print(f"systematic under-prediction suggests:")
    print(f"1. The bias corrections aren't being applied correctly")
    print(f"2. There's an issue with the calculation logic")
    print(f"3. Market data might be wrong")

if __name__ == "__main__":
    debug_bias_corrections()
