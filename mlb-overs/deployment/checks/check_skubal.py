#!/usr/bin/env python3

import psycopg2

def check_skubal_data():
    """Check if Skubal's ERA data is correct"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        cur = conn.cursor()
        
        cur.execute("""
        SELECT 
            home_team, 
            away_team,
            ROUND(home_sp_season_era::numeric, 2) as home_era,
            ROUND(away_sp_season_era::numeric, 2) as away_era,
            predicted_total
        FROM enhanced_games 
        WHERE date = '2025-08-14' 
        ORDER BY predicted_total;
        """)
        
        print('=== TODAY\'S GAMES WITH STARTING PITCHER ERAs ===')
        print('(Looking for Skubal - should have very low ERA)\n')
        
        tigers_game_found = False
        for row in cur.fetchall():
            home_team, away_team, home_era, away_era, predicted_total = row
            print(f'{away_team} @ {home_team}: {predicted_total} runs')
            print(f'  Starting Pitchers: Away ERA {away_era} vs Home ERA {home_era}')
            
            # Check if this is the Tigers game
            if 'Tigers' in away_team or 'Tigers' in home_team:
                tigers_game_found = True
                print(f'  🏆 TIGERS GAME FOUND!')
                if away_era < 3.0 or home_era < 3.0:
                    print(f'  🌟 ACE PITCHER DETECTED: This should be a low-scoring game!')
                else:
                    print(f'  ❌ PROBLEM: Neither pitcher has ace-level ERA. Where is Skubal?')
            
            # Check for other ace pitchers
            if away_era < 3.0:
                print(f'  🌟 AWAY ACE: {away_era} ERA')
            if home_era < 3.0:
                print(f'  🌟 HOME ACE: {home_era} ERA')
            print()
        
        if not tigers_game_found:
            print('❌ NO TIGERS GAME FOUND IN TODAY\'S DATA!')
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_skubal_data()
