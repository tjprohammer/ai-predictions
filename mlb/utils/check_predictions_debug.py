#!/usr/bin/env python3

import sqlite3

def check_predictions():
    conn = sqlite3.connect('mlb.db')
    cursor = conn.cursor()
    
    print('=== CHECKING AVAILABLE TABLES ===')
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print('Available tables:')
    for table in tables:
        print(f'  - {table[0]}')
    
    print('\n=== CHECKING FOR PREDICTIONS IN GAMES TABLE ===')
    cursor.execute("SELECT COUNT(*) FROM games WHERE game_date = '2025-08-29'")
    count = cursor.fetchone()[0]
    print(f'Found {count} games for 2025-08-29 in games table')
    
    if count > 0:
        cursor.execute("""
            SELECT game_date, away_team, home_team, predicted_total, confidence
            FROM games 
            WHERE game_date = '2025-08-29' 
            ORDER BY game_time
        """)
        results = cursor.fetchall()
        
        print(f'\nAll {len(results)} predictions for today:')
        for i, row in enumerate(results, 1):
            print(f'{i:2d}. {row[1]} @ {row[2]}: Predicted Total = {row[3]}, Confidence = {row[4]}')
        
        print('\n=== CHECKING FOR DUPLICATE PREDICTIONS ===')
        cursor.execute("""
            SELECT predicted_total, confidence, COUNT(*) as count
            FROM games 
            WHERE game_date = '2025-08-29' 
            GROUP BY predicted_total, confidence
            HAVING COUNT(*) > 1
        """)
        duplicates = cursor.fetchall()
        
        if duplicates:
            print('FOUND DUPLICATE PREDICTIONS:')
            for dup in duplicates:
                print(f'Predicted Total: {dup[0]}, Confidence: {dup[1]} appears {dup[2]} times')
        else:
            print('No duplicate predictions found')
    
    conn.close()

if __name__ == "__main__":
    check_predictions()
