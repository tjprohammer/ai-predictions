#!/usr/bin/env python3

import psycopg2

def analyze_tigers_features():
    """Analyze why Skubal's dominance isn't reflected in predictions"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        cur = conn.cursor()
        
        # Check the Tigers game specifically
        cur.execute("""
        SELECT 
            home_team, away_team,
            home_sp_season_era, away_sp_season_era,
            home_team_ops, away_team_ops,
            ballpark_run_factor, temperature,
            combined_era, starter_pitching_quality,
            predicted_total
        FROM enhanced_games 
        WHERE (home_team LIKE '%Tigers%' OR away_team LIKE '%Tigers%')
        AND date = '2025-08-14';
        """)
        
        row = cur.fetchone()
        if row:
            print('=== TIGERS GAME DETAILED ANALYSIS ===')
            print(f'Game: {row[1]} @ {row[0]}')
            print(f'Predicted Total: {row[10]} runs')
            print()
            print('PITCHING:')
            print(f'  Away Pitcher ERA: {row[2]:.2f} (Skubal - ACE LEVEL)')
            print(f'  Home Pitcher ERA: {row[3]:.2f}')
            print(f'  Combined ERA: {row[8]:.2f}')
            print(f'  Starter Pitching Quality: {row[9]:.2f}')
            print()
            print('OFFENSE:')
            print(f'  Away Team OPS: {row[4]:.3f}')
            print(f'  Home Team OPS: {row[5]:.3f}')
            print()
            print('ENVIRONMENT:')
            print(f'  Ballpark Factor: {row[6]:.2f}x')
            print(f'  Temperature: {row[7]}°F')
            print()
            print('🤔 PROBLEM ANALYSIS:')
            
            # Expected total with ace pitcher should be much lower
            expected_with_ace = 7.5  # Realistic expectation with Skubal
            actual_prediction = row[10]
            
            if actual_prediction > expected_with_ace + 1.5:
                print(f'❌ MAJOR ISSUE: Prediction {actual_prediction} runs is {actual_prediction - expected_with_ace:.1f} runs too high!')
                print(f'   With Skubal (2.35 ERA), total should be closer to {expected_with_ace} runs')
                print(f'   Model is not properly weighting elite starting pitching!')
            
        else:
            print('❌ Tigers game not found in database!')
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_tigers_features()
