#!/usr/bin/env python3
"""
Manual Game Result Updater
=========================

Quick script to manually enter game results for testing validation system
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

def show_todays_games():
    """Show today's games that need results"""
    conn = connect_db()
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute('''
    SELECT game_id, away_team, home_team, predicted_total, market_total, 
           recommendation, confidence, total_runs
    FROM enhanced_games 
    WHERE date = %s 
    ORDER BY game_id
    ''', (today,))
    
    games = cursor.fetchall()
    
    print(f"🎯 GAMES FOR {today}:")
    print("=" * 60)
    
    for i, game in enumerate(games, 1):
        game_id, away, home, pred, market, rec, conf, total = game
        status = "✅ COMPLETE" if total else "⏳ PENDING"
        
        print(f"{i:2d}. {away} @ {home} (ID: {game_id})")
        print(f"    Prediction: {pred} | Market: {market} | Rec: {rec} | Conf: {conf}%")
        print(f"    Status: {status}")
        if total:
            print(f"    Final Score: {total} total runs")
        print()
    
    conn.close()
    return games

def update_game_result(game_id: str, home_score: int, away_score: int):
    """Update a game's final score"""
    conn = connect_db()
    cursor = conn.cursor()
    
    total_runs = home_score + away_score
    
    try:
        cursor.execute('''
        UPDATE enhanced_games 
        SET home_score = %s, away_score = %s, total_runs = %s, 
            game_state = 'Final', result_updated_at = NOW()
        WHERE game_id = %s
        ''', (home_score, away_score, total_runs, game_id))
        
        conn.commit()
        print(f"✅ Updated game {game_id}: {away_score}-{home_score} = {total_runs} runs")
        
    except Exception as e:
        print(f"❌ Error updating game: {e}")
        conn.rollback()
    
    conn.close()

def quick_validation():
    """Quick validation of today's results"""
    conn = connect_db()
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as completed,
        COUNT(CASE WHEN recommendation = 'OVER' AND total_runs > market_total THEN 1 END) as over_wins,
        COUNT(CASE WHEN recommendation = 'UNDER' AND total_runs < market_total THEN 1 END) as under_wins,
        COUNT(CASE WHEN recommendation IN ('OVER', 'UNDER') THEN 1 END) as actionable,
        AVG(ABS(predicted_total - total_runs)) as avg_error
    FROM enhanced_games 
    WHERE date = %s 
    AND total_runs IS NOT NULL
    ''', (today,))
    
    result = cursor.fetchone()
    total, completed, over_wins, under_wins, actionable, avg_error = result
    
    if completed > 0:
        wins = over_wins + under_wins
        win_rate = (wins / actionable * 100) if actionable > 0 else 0
        
        print(f"📊 QUICK VALIDATION - {today}:")
        print(f"   Completed: {completed}/{total} games")
        print(f"   Wins: {wins}/{actionable} actionable picks ({win_rate:.1f}%)")
        print(f"   Average Error: {avg_error:.2f} runs")
    else:
        print("⏳ No completed games yet")
    
    conn.close()

def main():
    """Interactive result entry"""
    print("🎯 MANUAL GAME RESULT UPDATER")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Show today's games")
        print("2. Update game result")  
        print("3. Quick validation")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            show_todays_games()
        
        elif choice == '2':
            game_id = input("Enter game ID: ").strip()
            try:
                away_score = int(input("Enter away team score: ").strip())
                home_score = int(input("Enter home team score: ").strip())
                update_game_result(game_id, home_score, away_score)
                print("🔄 Running quick validation...")
                quick_validation()
            except ValueError:
                print("❌ Invalid scores entered")
        
        elif choice == '3':
            quick_validation()
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
