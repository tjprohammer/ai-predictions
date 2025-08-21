#!/usr/bin/env python3
"""
Data Source Explorer - Check what additional data we can leverage
"""
import psycopg2

def explore_data_sources():
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )
    cursor = conn.cursor()

    print('🔍 EXPLORING AVAILABLE DATA SOURCES')
    print('=' * 45)

    # Check bullpens_daily columns
    print('🎯 BULLPENS_DAILY TABLE:')
    cursor.execute('''
    SELECT column_name
    FROM information_schema.columns 
    WHERE table_name = 'bullpens_daily'
    ORDER BY column_name
    ''')
    bullpen_cols = [col[0] for col in cursor.fetchall()]
    print(f'   Columns: {bullpen_cols}')

    # Sample bullpen data
    cursor.execute('SELECT * FROM bullpens_daily LIMIT 1')
    sample_bullpen = cursor.fetchall()
    if sample_bullpen:
        print(f'   Sample row has {len(sample_bullpen[0])} values')

    # Check parks table
    print(f'\n🏟️ PARKS TABLE:')
    cursor.execute('''
    SELECT column_name
    FROM information_schema.columns 
    WHERE table_name = 'parks'
    ORDER BY column_name
    ''')
    park_cols = [col[0] for col in cursor.fetchall()]
    print(f'   Columns: {park_cols}')

    # Check weather_game
    print(f'\n🌤️ WEATHER_GAME TABLE:')
    cursor.execute('''
    SELECT column_name
    FROM information_schema.columns 
    WHERE table_name = 'weather_game'
    ORDER BY column_name
    ''')
    weather_cols = [col[0] for col in cursor.fetchall()]
    print(f'   Columns: {weather_cols}')

    # Check pitchers_starts columns  
    print(f'\n⚾ PITCHERS_STARTS TABLE:')
    cursor.execute('''
    SELECT column_name
    FROM information_schema.columns 
    WHERE table_name = 'pitchers_starts'
    ORDER BY column_name
    LIMIT 15
    ''')
    pitcher_cols = [col[0] for col in cursor.fetchall()]
    print(f'   First 15 columns: {pitcher_cols}')

    # Check for injury data
    print(f'\n🏥 INJURIES TABLE:')
    cursor.execute('SELECT COUNT(*) FROM injuries')
    injury_count = cursor.fetchone()[0]
    print(f'   Records: {injury_count}')

    # Check for lineup data
    print(f'\n👥 LINEUPS TABLE:')
    cursor.execute('SELECT COUNT(*) FROM lineups')
    lineup_count = cursor.fetchone()[0]
    print(f'   Records: {lineup_count}')

    conn.close()

if __name__ == "__main__":
    explore_data_sources()
