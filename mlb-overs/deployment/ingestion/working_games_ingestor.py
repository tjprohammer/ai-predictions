#!/usr/bin/env python3
"""
WORKING Games Ingestor
====================

Collects today's games and inserts them into enhanced_games table.
This replaces the broken mlb-overs/ingestors/games.py
"""

import argparse
import datetime as dt
import pandas as pd
import statsapi
import os
from sqlalchemy import create_engine, text
import psycopg2

def get_engine():
    """Get PostgreSQL database engine"""
    # PostgreSQL connection for Docker container
    DATABASE_URL = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
    return create_engine(DATABASE_URL)

def create_enhanced_games_table():
    """Create enhanced_games table if it doesn't exist"""
    engine = get_engine()
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS enhanced_games (
        id SERIAL PRIMARY KEY,
        game_id TEXT UNIQUE,
        date TEXT,
        home_team TEXT,
        away_team TEXT,
        home_team_id INTEGER,
        away_team_id INTEGER,
        venue_name TEXT,
        venue_id INTEGER,
        game_time TEXT,
        day_night TEXT,
        home_sp_id INTEGER,
        away_sp_id INTEGER,
        temperature REAL,
        wind_speed REAL,
        weather_condition TEXT,
        home_score INTEGER,
        away_score INTEGER,
        total_runs INTEGER,
        game_status TEXT,
        game_type TEXT DEFAULT 'R',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
        print("[OK] Enhanced games table ready")
    except Exception as e:
        print(f"Error creating table: {e}")

def fetch_todays_games(target_date=None):
    """Fetch MLB games for specified date or today"""
    if target_date:
        # Parse target date string
        if isinstance(target_date, str):
            game_date = dt.datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            game_date = target_date
    else:
        game_date = dt.date.today()
    
    try:
        # Get schedule for the specified date
        schedule = statsapi.schedule(start_date=game_date.strftime('%m/%d/%Y'), 
                                   end_date=game_date.strftime('%m/%d/%Y'))
        
        if not schedule:
            print(f"No games found for {game_date}")
            return []
        
        games = []
        for game in schedule:
            # Try different game ID fields (statsapi uses 'game_id' not 'game_pk')
            game_id = game.get('game_id') or game.get('game_pk') or game.get('gamePk')
            if not game_id:
                print(f"⚠️ Skipping game - no ID found: {game.get('away_name')} @ {game.get('home_name')}")
                continue
                
            # Get basic game info
            game_data = {
                'game_id': str(game_id),
                'date': game_date.strftime('%Y-%m-%d'),
                'home_team': game.get('home_name', ''),
                'away_team': game.get('away_name', ''),
                'home_team_id': game.get('home_id'),
                'away_team_id': game.get('away_id'),
                'venue_name': game.get('venue_name', ''),
                'venue_id': game.get('venue_id'),
                'game_type': 'R',  # Regular season
                'day_night': 'D' if 'day' in game.get('game_date', '').lower() else 'N'
            }
            
            # Try to get scores if game is final
            status = game.get('status', '').lower()
            if 'final' in status or 'completed' in status:
                home_score = game.get('home_score')
                away_score = game.get('away_score')
                if home_score is not None and away_score is not None:
                    game_data['home_score'] = int(home_score)
                    game_data['away_score'] = int(away_score)
                    game_data['total_runs'] = game_data['home_score'] + game_data['away_score']
            
            games.append(game_data)
        
        print(f"Found {len(games)} games for {game_date}")
        return games
        
    except Exception as e:
        print(f"Error fetching games: {e}")
        return []

def upsert_games(games):
    """Insert/update games in enhanced_games table"""
    if not games:
        return 0
        
    engine = get_engine()
    
    try:
        with engine.begin() as conn:
            # Insert games that don't exist
            for game in games:
                # Check if game already exists
                existing = conn.execute(
                    text("SELECT id FROM enhanced_games WHERE game_id = :game_id AND date = :date"),
                    {"game_id": game['game_id'], "date": game['date']}
                ).fetchone()
                
                if not existing:
                    # Insert new game
                    insert_sql = text("""
                        INSERT INTO enhanced_games (
                            game_id, date, home_team, away_team, home_team_id, away_team_id,
                            venue_name, venue_id, game_type, day_night, created_at
                        ) VALUES (
                            :game_id, :date, :home_team, :away_team, :home_team_id, :away_team_id,
                            :venue_name, :venue_id, :game_type, :day_night, NOW()
                        )
                    """)
                    
                    conn.execute(insert_sql, {
                        'game_id': game['game_id'],
                        'date': game['date'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'home_team_id': game.get('home_team_id'),
                        'away_team_id': game.get('away_team_id'),
                        'venue_name': game['venue_name'],
                        'venue_id': game.get('venue_id'),
                        'game_type': game.get('game_type', 'R'),
                        'day_night': game.get('day_night', 'N')
                    })
                    print(f"[OK] Inserted: {game['away_team']} @ {game['home_team']}")
                else:
                    # Always update existing games with latest data
                    update_sql = text("""
                        UPDATE enhanced_games 
                        SET home_team = :home_team,
                            away_team = :away_team,
                            home_team_id = :home_team_id,
                            away_team_id = :away_team_id,
                            venue_name = :venue_name,
                            venue_id = :venue_id,
                            game_type = :game_type,
                            day_night = :day_night,
                            home_score = :home_score, 
                            away_score = :away_score, 
                            total_runs = :total_runs
                        WHERE game_id = :game_id AND date = :date
                    """)
                    
                    conn.execute(update_sql, {
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'home_team_id': game.get('home_team_id'),
                        'away_team_id': game.get('away_team_id'),
                        'venue_name': game['venue_name'],
                        'venue_id': game.get('venue_id'),
                        'game_type': game.get('game_type', 'R'),
                        'day_night': game.get('day_night', 'N'),
                        'home_score': game.get('home_score'),
                        'away_score': game.get('away_score'),
                        'total_runs': game.get('total_runs'),
                        'game_id': game['game_id'],
                        'date': game['date']
                    })
                    print(f"🔄 Updated: {game['away_team']} @ {game['home_team']}")
        
        return len(games)
        
    except Exception as e:
        print(f"Error inserting games: {e}")
        return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect MLB games data')
    parser.add_argument('--target-date', type=str, help='Target date in YYYY-MM-DD format')
    args = parser.parse_args()
    
    target_date = args.target_date
    date_str = target_date if target_date else "today"
    
    print(f"[GAMES] Collecting MLB Games for {date_str}")
    print("=" * 40)
    
    # Create table if needed
    create_enhanced_games_table()
    
    # Fetch games
    games = fetch_todays_games(target_date)
    
    if games:
        inserted = upsert_games(games)
        print(f"[OK] Processed {inserted} games for {date_str}")
    else:
        print(f"❌ No games to process for {date_str}")

if __name__ == "__main__":
    main()
