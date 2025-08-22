#!/usr/bin/env python3
"""
Collect Umpire Assignments for All 1,987 Games

This script runs the existing Sportradar umpire collector across all games
in the enhanced_games table to build a complete umpire assignment database.
"""

import psycopg2
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the data_collection directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mlb-overs', 'data_collection'))

from sportradar_umpire_collector_simple import SportradarUmpireCollector

load_dotenv()

def get_all_game_dates():
    """Get all unique dates with games in the database"""
    db_url = os.getenv('DATABASE_URL')
    if 'postgresql+psycopg2://' in db_url:
        db_url = db_url.replace('postgresql+psycopg2://', 'postgresql://')
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT DISTINCT date, COUNT(*) as games_count
        FROM enhanced_games 
        WHERE date >= '2025-03-20' 
        ORDER BY date
    """)
    
    dates_with_games = cur.fetchall()
    cur.close()
    conn.close()
    
    return dates_with_games

def main():
    """Collect umpire assignments for all games"""
    print("🔍 Collecting umpire assignments for all 1,987 games...")
    
    # Get all game dates
    dates_with_games = get_all_game_dates()
    total_dates = len(dates_with_games)
    
    print(f"📅 Found {total_dates} unique dates with games")
    
    # Initialize collector
    try:
        collector = SportradarUmpireCollector()
    except Exception as e:
        print(f"❌ Failed to initialize Sportradar collector: {e}")
        return
    
    # Collect umpire data for each date
    total_games_processed = 0
    successful_assignments = 0
    
    for i, (game_date, games_count) in enumerate(dates_with_games, 1):
        print(f"\n📊 Processing {game_date} ({i}/{total_dates}) - {games_count} games")
        
        try:
            # Convert string date to datetime.date
            if isinstance(game_date, str):
                date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
            else:
                date_obj = game_date
            
            # Collect umpires for this date
            assignments = collector.collect_date_umpires(date_obj)
            
            if assignments:
                print(f"✅ Collected {len(assignments)} umpire assignments for {game_date}")
                successful_assignments += len(assignments)
            else:
                print(f"⚠️ No umpire data found for {game_date}")
            
            total_games_processed += games_count
            
        except Exception as e:
            print(f"❌ Error processing {game_date}: {e}")
            continue
    
    print(f"\n📈 COLLECTION SUMMARY:")
    print(f"Total games processed: {total_games_processed}")
    print(f"Successful umpire assignments: {successful_assignments}")
    print(f"Coverage: {(successful_assignments/total_games_processed)*100:.1f}%")
    
    if successful_assignments > 0:
        print(f"\n✅ Phase 4 umpire data collection completed!")
        print(f"Next step: Run umpire stats generation and ingestion")
    else:
        print(f"\n⚠️ No umpire assignments collected - API may be limited")
        print(f"Recommendation: Proceed with simulated umpire data for Phase 4")

if __name__ == "__main__":
    main()
