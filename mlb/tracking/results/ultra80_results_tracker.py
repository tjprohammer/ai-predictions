#!/usr/bin/env python3
"""
Ultra 80 Results Tracker
========================

Tracks Ultra 80 Incremental System predictions vs actual results.
Separate from learning model tracking to compare both systems.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import psycopg2

def update_ultra80_results():
    """Update today's Ultra 80 predictions with actual results"""
    print("🚀 UPDATING ULTRA 80 SYSTEM RESULTS")
    print("=" * 45)
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"📅 Processing Ultra 80 results for: {today}")
    
    # Connect to database
    engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
    
    # Get today's Ultra 80 predictions
    with engine.connect() as conn:
        ultra_query = text("""
            SELECT u.game_id, u.home_team, u.away_team, u.market_total, 
                   u.pred_total, u.lower_80, u.upper_80, u.best_side, u.ev,
                   e.total_runs, e.home_score, e.away_score
            FROM ultra80_predictions u
            LEFT JOIN enhanced_games e ON u.game_id = e.game_id AND u.date = e.date
            WHERE u.date = :today
            ORDER BY u.home_team
        """)
        
        ultra_df = pd.read_sql(ultra_query, conn, params={'today': today})
    
    if ultra_df.empty:
        print(f"❌ No Ultra 80 predictions found for {today}")
        return 0, 0
    
    print(f"📊 Found {len(ultra_df)} Ultra 80 predictions for {today}")
    
    # Show current status
    completed_games = ultra_df[ultra_df['total_runs'].notna()]
    pending_games = ultra_df[ultra_df['total_runs'].isna()]
    
    print(f"✅ Completed games: {len(completed_games)}")
    print(f"⏳ Pending games: {len(pending_games)}")
    
    if len(completed_games) > 0:
        print("\n🏁 ULTRA 80 COMPLETED GAMES:")
        print("-" * 80)
        
        for _, game in completed_games.iterrows():
            actual = game['total_runs']
            pred = game['pred_total']
            market = game['market_total']
            lower = game['lower_80']
            upper = game['upper_80']
            side = game['best_side']
            ev = game['ev']
            
            # Check if prediction was accurate
            in_interval = lower <= actual <= upper if pd.notna(lower) and pd.notna(upper) else False
            interval_status = "✅ IN 80% INTERVAL" if in_interval else "❌ OUTSIDE INTERVAL"
            
            # Check betting recommendation
            bet_result = ""
            if side == "OVER" and actual > market:
                bet_result = "✅ OVER WIN"
            elif side == "UNDER" and actual < market:
                bet_result = "✅ UNDER WIN"
            elif side in ["OVER", "UNDER"]:
                bet_result = "❌ BET LOSS"
            else:
                bet_result = "⏸️ NO BET"
            
            error = abs(actual - pred) if pd.notna(pred) else 0
            
            print(f"  {game['away_team']} @ {game['home_team']}")
            print(f"    Actual: {actual} | Ultra Pred: {pred:.1f} | Market: {market}")
            print(f"    80% Interval: [{lower:.1f}, {upper:.1f}] | {interval_status}")
            print(f"    Recommendation: {side} | EV: {ev:.1f}% | {bet_result}")
            print(f"    Error: {error:.2f}")
            print()
    
    if len(pending_games) > 0:
        print(f"\n⏳ PENDING ULTRA 80 GAMES:")
        print("-" * 40)
        for _, game in pending_games.iterrows():
            pred = game['pred_total']
            market = game['market_total']
            side = game['best_side']
            ev = game['ev']
            
            print(f"  {game['away_team']} @ {game['home_team']}")
            print(f"    Ultra Pred: {pred:.1f} | Market: {market} | Rec: {side} | EV: {ev:.1f}%")
    
    # Calculate performance metrics for completed games
    if len(completed_games) > 0:
        print(f"\n📈 ULTRA 80 PERFORMANCE SUMMARY:")
        print("=" * 45)
        
        # Prediction accuracy
        ultra_errors = [abs(row['total_runs'] - row['pred_total']) for _, row in completed_games.iterrows() if pd.notna(row['pred_total'])]
        market_errors = [abs(row['total_runs'] - row['market_total']) for _, row in completed_games.iterrows() if pd.notna(row['market_total'])]
        
        if ultra_errors:
            ultra_mae = sum(ultra_errors) / len(ultra_errors)
            print(f"  Ultra 80 MAE: {ultra_mae:.2f}")
        
        if market_errors:
            market_mae = sum(market_errors) / len(market_errors)
            print(f"  Market MAE: {market_mae:.2f}")
        
        # Interval coverage
        interval_coverage = sum(1 for _, row in completed_games.iterrows() 
                               if pd.notna(row['lower_80']) and pd.notna(row['upper_80']) 
                               and row['lower_80'] <= row['total_runs'] <= row['upper_80'])
        total_with_intervals = sum(1 for _, row in completed_games.iterrows() 
                                  if pd.notna(row['lower_80']) and pd.notna(row['upper_80']))
        
        if total_with_intervals > 0:
            coverage_pct = (interval_coverage / total_with_intervals) * 100
            print(f"  80% Interval Coverage: {interval_coverage}/{total_with_intervals} ({coverage_pct:.1f}%)")
        
        # Betting performance
        bet_games = completed_games[completed_games['best_side'].isin(['OVER', 'UNDER'])]
        if len(bet_games) > 0:
            wins = 0
            for _, game in bet_games.iterrows():
                actual = game['total_runs']
                market = game['market_total']
                side = game['best_side']
                
                if (side == "OVER" and actual > market) or (side == "UNDER" and actual < market):
                    wins += 1
            
            win_rate = (wins / len(bet_games)) * 100
            print(f"  Betting Record: {wins}/{len(bet_games)} ({win_rate:.1f}%)")
    
    print(f"\n✅ Ultra 80 results update complete for {today}")
    return len(completed_games), len(pending_games)

if __name__ == "__main__":
    completed, pending = update_ultra80_results()
    print(f"\n📊 SUMMARY: {completed} completed, {pending} pending Ultra 80 games")
