#!/usr/bin/env python3

import psycopg2
import pandas as pd

def check_yesterday_games():
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='mlb_overs',
        user='postgres',
        password='password'
    )

    # Check games from August 23rd
    query = '''
    SELECT date, home_team, away_team, total_runs, market_total,
           enhanced_prediction, learning_prediction, home_score, away_score
    FROM enhanced_games 
    WHERE date = '2025-08-23'
    ORDER BY date, home_team;
    '''

    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) > 0:
        print(f'📅 AUGUST 23rd GAMES: {len(df)} games')
        print('=' * 60)
        
        total_enhanced_error = 0
        total_learning_error = 0
        enhanced_count = 0
        learning_count = 0
        
        for _, game in df.iterrows():
            actual = game['total_runs'] if pd.notna(game['total_runs']) else None
            enhanced = game['enhanced_prediction'] if pd.notna(game['enhanced_prediction']) else None
            learning = game['learning_prediction'] if pd.notna(game['learning_prediction']) else None
            market = game['market_total'] if pd.notna(game['market_total']) else None
            
            away_team = game['away_team']
            home_team = game['home_team']
            print(f'{away_team} @ {home_team}')
            print(f'  Market: {market} | Enhanced: {enhanced} | Learning: {learning} | Actual: {actual}')
            
            # Calculate errors if we have predictions and actual results
            if actual is not None:
                if enhanced is not None:
                    error = abs(enhanced - actual)
                    total_enhanced_error += error
                    enhanced_count += 1
                    print(f'  Enhanced Error: {error:.2f}')
                    
                if learning is not None:
                    error = abs(learning - actual)
                    total_learning_error += error
                    learning_count += 1
                    print(f'  Learning Error: {error:.2f}')
            
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                away_score = int(game['away_score'])
                home_score = int(game['home_score'])
                print(f'  Final Score: {away_team} {away_score} - {home_score} {home_team}')
            print()
        
        # Summary
        if enhanced_count > 0:
            avg_enhanced_error = total_enhanced_error / enhanced_count
            print(f'📊 Enhanced Model Average Error: {avg_enhanced_error:.3f} ({enhanced_count} games)')
            
        if learning_count > 0:
            avg_learning_error = total_learning_error / learning_count
            print(f'📊 Learning Model Average Error: {avg_learning_error:.3f} ({learning_count} games)')
            
    else:
        print('❌ No games found for August 23rd')

if __name__ == "__main__":
    check_yesterday_games()
