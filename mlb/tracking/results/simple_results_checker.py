#!/usr/bin/env python3
"""
Simple Daily Results Checker
============================

Quick way to see: "Did my predictions win or lose today?"
Just enter final scores and get instant feedback on your betting performance.
"""

import psycopg2
from datetime import datetime

def connect_db():
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )

def show_todays_picks():
    """Show today's actionable picks"""
    conn = connect_db()
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute('''
    SELECT game_id, away_team, home_team, predicted_total, market_total, 
           recommendation, confidence, edge, total_runs
    FROM enhanced_games 
    WHERE date = %s 
    AND recommendation IN ('OVER', 'UNDER')
    ORDER BY confidence DESC
    ''', (today,))
    
    picks = cursor.fetchall()
    
    print(f"🎯 YOUR ACTIONABLE PICKS FOR {today}:")
    print("=" * 60)
    
    if not picks:
        print("No actionable picks found for today")
        return []
    
    for i, pick in enumerate(picks, 1):
        game_id, away, home, pred, market, rec, conf, edge, actual = pick
        status = f"Final: {actual} runs" if actual else "⏳ In Progress"
        
        print(f"{i}. {away} @ {home}")
        print(f"   📊 Bet: {rec} {market} (Market Total)")
        print(f"   🎯 Confidence: {conf}% | Edge: {edge:+.1f}")
        print(f"   📈 Our Prediction: {pred} runs")
        print(f"   🏁 Status: {status}")
        
        if actual:
            # Determine if bet won
            if rec == 'OVER' and actual > market:
                print(f"   ✅ BET WON! ({actual} > {market})")
            elif rec == 'UNDER' and actual < market:
                print(f"   ✅ BET WON! ({actual} < {market})")
            else:
                print(f"   ❌ BET LOST")
        print()
    
    conn.close()
    return picks

def quick_score_update(game_id: str, total_runs: int):
    """Quickly update just the total runs for a game"""
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        UPDATE enhanced_games 
        SET total_runs = %s, game_state = 'Final'
        WHERE game_id = %s
        ''', (total_runs, game_id))
        
        conn.commit()
        print(f"✅ Updated game {game_id} with {total_runs} total runs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    
    conn.close()

def daily_summary():
    """Show win/loss summary for today - both betting and prediction accuracy"""
    conn = connect_db()
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Betting results (only games where we made OVER/UNDER bets)
    cursor.execute('''
    SELECT 
        COUNT(*) as total_picks,
        COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as completed,
        COUNT(CASE WHEN recommendation = 'OVER' AND total_runs > market_total THEN 1 END) +
        COUNT(CASE WHEN recommendation = 'UNDER' AND total_runs < market_total THEN 1 END) as wins,
        AVG(confidence) as avg_confidence,
        AVG(ABS(predicted_total - total_runs)) as avg_error
    FROM enhanced_games 
    WHERE date = %s 
    AND recommendation IN ('OVER', 'UNDER')
    AND total_runs IS NOT NULL
    ''', (today,))
    
    betting_result = cursor.fetchone()
    bet_total_picks, bet_completed, bet_wins, bet_avg_conf, bet_avg_error = betting_result
    
    # Prediction accuracy (ALL games including HOLD - did we predict direction correctly vs market?)
    cursor.execute('''
    SELECT 
        COUNT(*) as total_predictions,
        COUNT(CASE 
            WHEN (predicted_total > market_total AND total_runs > market_total) OR
                 (predicted_total < market_total AND total_runs < market_total)
            THEN 1 END) as correct_predictions,
        AVG(ABS(predicted_total - total_runs)) as pred_avg_error
    FROM enhanced_games 
    WHERE date = %s 
    AND total_runs IS NOT NULL
    AND predicted_total != market_total  -- Exclude neutral predictions
    ''', (today,))
    
    prediction_result = cursor.fetchone()
    pred_total, pred_correct, pred_avg_error = prediction_result
    
    print(f"📊 DAILY SUMMARY - {today}:")
    
    # Betting Performance
    if bet_completed > 0:
        bet_win_rate = (bet_wins / bet_completed * 100)
        
        # Calculate profit/loss (assuming $100 bets at -110 odds)
        profit = (bet_wins * 90.91) - ((bet_completed - bet_wins) * 100)
        profit_display = f"+${profit:.0f}" if profit > 0 else f"-${abs(profit):.0f}"
        
        print(f"   🎯 Actionable Picks: {bet_total_picks}")
        print(f"   ✅ Completed: {bet_completed}")
        print(f"   🏆 Wins: {bet_wins}/{bet_completed} ({bet_win_rate:.1f}%)")
        print(f"   � Profit/Loss: {profit_display} (@ $100/bet)")
        print(f"   �📈 Avg Confidence: {bet_avg_conf:.1f}%")
        print(f"   📊 Avg Error: {bet_avg_error:.1f} runs")
        
        if bet_win_rate >= 70:
            print(f"   🔥 ON FIRE! Elite win rate!")
        elif bet_win_rate >= 60:
            print(f"   🎉 GREAT DAY! Win rate above 60%")
        elif bet_win_rate >= 50:
            print(f"   👍 DECENT DAY! Win rate above 50%")
        else:
            print(f"   📉 TOUGH DAY! Win rate below 50%")
    else:
        print(f"   ⏳ No completed bets yet today")
    
    # Prediction Accuracy
    if pred_total > 0:
        pred_accuracy = (pred_correct / pred_total * 100)
        print(f"\n   🔮 PREDICTION ACCURACY (All Games):")
        print(f"   📊 Directional Calls: {pred_correct}/{pred_total} ({pred_accuracy:.1f}%)")
        print(f"   📈 Avg Prediction Error: {pred_avg_error:.1f} runs")
        
        if pred_accuracy >= 60:
            print(f"   ✨ EXCELLENT predictions! Above 60% accuracy")
        elif pred_accuracy >= 50:
            print(f"   📈 SOLID predictions! Above 50% accuracy")
        else:
            print(f"   🔄 LEARNING mode - predictions need work")
    
    conn.close()

def main():
    """Simple interface for checking daily results"""
    print("🎯 DAILY BETTING RESULTS CHECKER")
    print("=" * 40)
    
    # Show today's picks
    picks = show_todays_picks()
    
    if not picks:
        return
    
    # Show summary if any games are complete
    print()
    daily_summary()
    
    # Quick update option
    print(f"\n🔄 QUICK UPDATES:")
    print(f"To update a game result, just run:")
    print(f"python -c \"from simple_results_checker import quick_score_update; quick_score_update('GAME_ID', TOTAL_RUNS)\"")
    print(f"\nExample: quick_score_update('776652', 9)  # Game had 9 total runs")
    
    # Performance insights
    if picks:
        print(f"\n🧠 QUICK INSIGHTS:")
        
        # Best pick of the day
        completed_picks = [p for p in picks if p[8] is not None]  # p[8] is total_runs
        if completed_picks:
            best_pick = None
            best_margin = 0
            
            for pick in completed_picks:
                game_id, away, home, pred, market, rec, conf, edge, actual = pick
                
                if rec == 'OVER' and actual > market:
                    margin = actual - market
                    if margin > best_margin:
                        best_margin = margin
                        best_pick = (f"{away} @ {home}", f"OVER {market}", actual, margin)
                elif rec == 'UNDER' and actual < market:
                    margin = market - actual
                    if margin > best_margin:
                        best_margin = margin
                        best_pick = (f"{away} @ {home}", f"UNDER {market}", actual, margin)
            
            if best_pick:
                game, bet_type, actual, margin = best_pick
                print(f"   🏆 Best Win: {game} - {bet_type} (won by {margin:.1f} runs)")
        
        # Weekly tracker reminder
        print(f"   📊 Run 'python weekly_performance_tracker.py' for weekly analysis")

if __name__ == "__main__":
    main()
