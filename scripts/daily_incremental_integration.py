#!/usr/bin/env python3
"""
Daily Incremental Learning Integration
=====================================

This script demonstrates how the incremental_ultra_80_system integrates 
into the daily workflow for continuous learning and prediction.

Daily Cycle:
1. Learn from yesterday's completed games (incremental updates)
2. Predict tomorrow's upcoming games
3. Export predictions for UI consumption
4. Save updated model state

Usage:
    python daily_incremental_integration.py --date 2025-08-27
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add the pipeline directory to path
sys.path.append(str(Path(__file__).parent / 'mlb-overs' / 'pipelines'))

from incremental_ultra_80_system import IncrementalUltra80System


def run_daily_incremental_cycle(target_date: str = None):
    """
    Run the complete daily incremental learning cycle
    
    Args:
        target_date: Date to predict for (YYYY-MM-DD format)
    """
    if not target_date:
        # Default to tomorrow
        tomorrow = datetime.now() + timedelta(days=1)
        target_date = tomorrow.strftime('%Y-%m-%d')
    
    print(f"🚀 DAILY INCREMENTAL LEARNING CYCLE for {target_date}")
    print("="*60)
    
    # Initialize the system
    system = IncrementalUltra80System()
    
    # Load existing state (if any)
    state_loaded = system.load_state()
    if state_loaded:
        print('🔄 Loaded existing model state - continuing incremental learning...')
    else:
        print('🆕 No existing state found - will start fresh learning...')
    
    # Step 1: Learn from recent completed games (last 7 days)
    print("\n📚 STEP 1: Learning from recent completed games...")
    yesterday = datetime.now() - timedelta(days=1)
    week_ago = yesterday - timedelta(days=7)
    
    results = system.team_level_incremental_learn(
        start_date=week_ago.strftime('%Y-%m-%d'),
        end_date=yesterday.strftime('%Y-%m-%d')
    )
    
    if results:
        print(f"✅ Learned from {len(results.get('predictions', []))} completed games")
        print(f"📊 Final Coverage: {results['final_coverage']:.1%}")
        print(f"📊 Final MAE: {results['final_mae']:.2f}")
        print(f"📊 Final ROI: {results['final_roi']:+.2%}")
        
        # Save updated model state
        system.save_state()
        print("💾 Model state saved")
    else:
        print("⚠️  No recent completed games found for learning")
    
    # Step 2: Predict tomorrow's games
    print(f"\n🔮 STEP 2: Predicting games for {target_date}...")
    
    if not system.is_fitted:
        print("⚠️  Models not fitted - need historical training data first")
        return None
    
    predictions = system.predict_future_slate(target_date, outdir='outputs')
    
    if not predictions.empty:
        print(f"✅ Generated predictions for {len(predictions)} games")
        
        # Show summary of predictions
        print(f"\n📋 PREDICTION SUMMARY for {target_date}:")
        print("-" * 60)
        for _, pred in predictions.iterrows():
            print(f"{pred['away_team']} @ {pred['home_team']}")
            print(f"  Market: {pred['market_total']} | Predicted: {pred['pred_total']}")
            print(f"  80% Range: [{pred['lower_80']}, {pred['upper_80']}]")
            print(f"  Edge: {pred['diff']:+.2f} | EV: {pred['ev']:+.1%}")
            print(f"  Recommendation: {pred['best_side']} {pred['market_total']} ({pred['best_odds']:+d})")
            print()
        
        # Generate betting recommendations if any high-EV opportunities
        high_ev_games = predictions[
            (predictions['ev'] >= 0.05) & 
            (predictions['trust'] >= 0.6) & 
            (abs(predictions['diff']) >= 0.5)
        ]
        
        if not high_ev_games.empty:
            print(f"💎 HIGH-VALUE BETTING OPPORTUNITIES ({len(high_ev_games)} games):")
            print("-" * 60)
            for _, bet in high_ev_games.iterrows():
                print(f"🎯 {bet['away_team']} @ {bet['home_team']}")
                print(f"   Bet: {bet['best_side']} {bet['market_total']} ({bet['best_odds']:+d})")
                print(f"   EV: {bet['ev']:+.1%} | Trust: {bet['trust']:.2f} | Edge: {bet['diff']:+.1f}")
                print()
        else:
            print("🔒 No high-confidence betting opportunities found")
        
        return predictions
    else:
        print(f"⚠️  No upcoming games found for {target_date}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Daily Incremental Learning Integration')
    parser.add_argument('--date', type=str, help='Target date for predictions (YYYY-MM-DD)', default=None)
    parser.add_argument('--force-reset', action='store_true', help='Reset model state and start fresh')
    
    args = parser.parse_args()
    
    # Set environment variables for system configuration
    if args.force_reset:
        os.environ['FORCE_RESET'] = '1'
    
    # Run the daily cycle
    predictions = run_daily_incremental_cycle(args.date)
    
    print("\n🏁 Daily incremental learning cycle complete!")
    
    if predictions is not None:
        print(f"📁 Predictions exported to: outputs/slate_{args.date or 'tomorrow'}_predictions.csv")
        print("🎯 Ready for UI integration and betting analysis")
    

if __name__ == '__main__':
    main()
