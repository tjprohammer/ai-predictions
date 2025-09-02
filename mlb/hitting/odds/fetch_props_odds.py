"""
MLB Player Props Odds Fetcher

This module fetches hitting prop odds from DraftKings and other sportsbooks
for "1+ Hit" markets and stores them for analysis.
"""

import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Set
import time
import json

log = logging.getLogger(__name__)

class PropsOddsFetcher:
    """Fetches and stores MLB hitting props odds"""
    
    def __init__(self, engine):
        self.engine = engine
        # The Odds API - requires API key
        self.odds_api_key = None  # Set from environment
        self.odds_base_url = "https://api.the-odds-api.com/v4"
        
        # DraftKings API (public endpoints)
        self.dk_base_url = "https://sportsbook-nash.draftkings.com/sites/US-SB/api/v5"
        
    def get_todays_mlb_games(self) -> List[Dict]:
        """Get today's MLB games to know which players might have props"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        query = text("""
            SELECT DISTINCT game_id, home_team, away_team, home_sp_name, away_sp_name
            FROM enhanced_games
            WHERE date = :today
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'today': today})
            games = [dict(row) for row in result]
        
        log.info(f"Found {len(games)} games for {today}")
        return games
    
    def get_recent_players(self, days_back: int = 7) -> Set[int]:
        """Get player IDs who have played recently (for prop markets)"""
        
        query = text("""
            SELECT DISTINCT player_id
            FROM player_game_logs
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
              AND at_bats > 0
        """ % days_back)
        
        with self.engine.connect() as conn:
            result = conn.execute(query)
            player_ids = {row[0] for row in result}
        
        log.info(f"Found {len(player_ids)} active players in last {days_back} days")
        return player_ids
    
    def fetch_dk_props_odds(self, target_date: str = None) -> List[Dict]:
        """
        Fetch DraftKings hitting props odds
        
        Note: This is a simplified example. Real DraftKings API access requires
        authentication and proper endpoints. You may need to:
        1. Use a sports data provider like The Odds API
        2. Scrape with proper rate limiting and compliance
        3. Use official sportsbook partnerships
        """
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        log.info(f"Fetching DraftKings props for {target_date}")
        
        # This is a placeholder - you'll need real API endpoints
        props_odds = []
        
        # Example of what the data structure might look like
        # In reality, you'd make API calls to get this data
        sample_props = [
            {
                'player_id': 592450,  # Mookie Betts
                'player_name': 'Mookie Betts',
                'team': 'LAD',
                'prop_type': '1+ Hit',
                'line': 0.5,
                'over_odds': -150,
                'under_odds': +120,
                'sportsbook': 'DraftKings',
                'timestamp': datetime.now()
            },
            {
                'player_id': 545361,  # Mike Trout
                'player_name': 'Mike Trout', 
                'team': 'LAA',
                'prop_type': '1+ Hit',
                'line': 0.5,
                'over_odds': -140,
                'under_odds': +115,
                'sportsbook': 'DraftKings',
                'timestamp': datetime.now()
            }
        ]
        
        # TODO: Replace with real API calls
        # For now, return empty list until real integration
        log.warning("DraftKings props fetching not implemented - requires API access")
        return []
    
    def fetch_odds_api_props(self, target_date: str = None) -> List[Dict]:
        """
        Fetch props from The Odds API (requires API key)
        """
        
        if not self.odds_api_key:
            log.warning("No Odds API key configured")
            return []
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        props_odds = []
        
        try:
            # Get MLB games
            url = f"{self.odds_base_url}/sports/baseball_mlb/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'player_hits',  # or whatever the market is called
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse response (structure depends on API)
            for game in data:
                for bookmaker in game.get('bookmakers', []):
                    if bookmaker['title'] != 'DraftKings':
                        continue
                        
                    for market in bookmaker.get('markets', []):
                        if market['key'] != 'player_hits':
                            continue
                            
                        for outcome in market.get('outcomes', []):
                            props_odds.append({
                                'player_name': outcome.get('description', ''),
                                'prop_type': '1+ Hit',
                                'line': 0.5,
                                'over_odds': outcome.get('price'),
                                'under_odds': None,  # Would need both sides
                                'sportsbook': 'DraftKings',
                                'timestamp': datetime.now()
                            })
            
        except requests.RequestException as e:
            log.error(f"Error fetching from Odds API: {e}")
        
        log.info(f"Fetched {len(props_odds)} props from Odds API")
        return props_odds
    
    def generate_mock_props(self, target_date: str = None) -> List[Dict]:
        """
        Generate mock props data for testing purposes
        """
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        log.info(f"Generating mock props for {target_date}")
        
        # Get recent active players
        recent_players = self.get_recent_players(days_back=3)
        
        if not recent_players:
            log.warning("No recent players found")
            return []
        
        # Get player details from database
        player_ids_list = list(recent_players)[:50]  # Limit for testing
        
        query = text("""
            SELECT DISTINCT player_id, player_name, team
            FROM player_game_logs
            WHERE player_id = ANY(:player_ids)
              AND date >= CURRENT_DATE - INTERVAL '3 days'
            ORDER BY player_name
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'player_ids': player_ids_list})
            players = [dict(row) for row in result]
        
        # Generate mock odds for these players
        mock_props = []
        np.random.seed(42)  # For consistent results
        
        for player in players:
            # Generate realistic odds around fair value
            base_prob = np.random.uniform(0.55, 0.75)  # Most players have 55-75% chance of 1+ hit
            vig = np.random.uniform(0.05, 0.08)  # 5-8% vig
            
            # Calculate odds
            over_prob = base_prob + (vig / 2)
            under_prob = (1 - base_prob) + (vig / 2)
            
            # Convert to American odds
            over_odds = int(-100 * over_prob / (1 - over_prob)) if over_prob > 0.5 else int(100 * (1 - over_prob) / over_prob)
            under_odds = int(-100 * under_prob / (1 - under_prob)) if under_prob > 0.5 else int(100 * (1 - under_prob) / under_prob)
            
            mock_props.append({
                'player_id': player['player_id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'prop_type': '1+ Hit',
                'line': 0.5,
                'over_odds': over_odds,
                'under_odds': under_odds,
                'sportsbook': 'DraftKings',
                'market': 'hits',
                'timestamp': datetime.now()
            })
        
        log.info(f"Generated {len(mock_props)} mock props")
        return mock_props
    
    def save_props_odds(self, props_odds: List[Dict], target_date: str) -> int:
        """Save props odds to database"""
        
        if not props_odds:
            return 0
        
        df = pd.DataFrame(props_odds)
        df['date'] = target_date
        
        # Ensure required columns exist
        required_cols = ['date', 'player_id', 'player_name', 'team', 'prop_type', 
                        'line', 'over_odds', 'under_odds', 'sportsbook', 'timestamp']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Save to database
        with self.engine.begin() as conn:
            # Create temp table
            df[required_cols].to_sql('temp_props_odds', conn, if_exists='replace', index=False)
            
            # Upsert into main table
            upsert_query = text("""
                INSERT INTO player_props_odds (
                    date, player_id, player_name, team, prop_type, line,
                    over_odds, under_odds, sportsbook, market, timestamp
                )
                SELECT 
                    date, player_id, player_name, team, prop_type, line,
                    over_odds, under_odds, sportsbook, 'hits', timestamp
                FROM temp_props_odds
                ON CONFLICT (date, player_id, prop_type, sportsbook)
                DO UPDATE SET
                    player_name = EXCLUDED.player_name,
                    team = EXCLUDED.team,
                    line = EXCLUDED.line,
                    over_odds = EXCLUDED.over_odds,
                    under_odds = EXCLUDED.under_odds,
                    timestamp = EXCLUDED.timestamp
            """)
            
            conn.execute(upsert_query)
            
            # Drop temp table
            conn.execute(text("DROP TABLE temp_props_odds"))
        
        log.info(f"Saved {len(df)} props odds")
        return len(df)
    
    def fetch_all_props(self, target_date: str = None, use_mock: bool = False) -> int:
        """Fetch all available props for a target date"""
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        log.info(f"Fetching all props for {target_date}")
        
        all_props = []
        
        if use_mock:
            # Use mock data for testing
            mock_props = self.generate_mock_props(target_date)
            all_props.extend(mock_props)
        else:
            # Try real sources
            dk_props = self.fetch_dk_props_odds(target_date)
            all_props.extend(dk_props)
            
            odds_api_props = self.fetch_odds_api_props(target_date)
            all_props.extend(odds_api_props)
        
        # Save all props
        total_saved = self.save_props_odds(all_props, target_date)
        
        log.info(f"Completed fetching props for {target_date}: {total_saved} total props")
        return total_saved
    
    def cleanup_old_odds(self, days_to_keep: int = 7):
        """Remove odds older than specified days"""
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        
        with self.engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM player_props_odds WHERE date < :cutoff_date"),
                {'cutoff_date': cutoff_date}
            )
            
            deleted_count = result.rowcount
        
        log.info(f"Cleaned up {deleted_count} old odds records (older than {cutoff_date})")
        return deleted_count

def main():
    """CLI interface for fetching props odds"""
    import argparse
    import sys
    from pathlib import Path
    from sqlalchemy import create_engine
    import os
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description='Fetch MLB hitting props odds')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old odds data')
    parser.add_argument('--odds-api-key', type=str, help='The Odds API key')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load environment
    load_dotenv(Path(__file__).parent.parent / '.env')
    
    # Connect to database
    engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
    
    # Initialize fetcher
    fetcher = PropsOddsFetcher(engine)
    
    # Set API key if provided
    if args.odds_api_key:
        fetcher.odds_api_key = args.odds_api_key
    else:
        fetcher.odds_api_key = os.getenv('ODDS_API_KEY')
    
    # Get target date
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch props
    if args.mock:
        print(f"Generating mock props for {target_date}")
    else:
        print(f"Fetching real props for {target_date}")
    
    count = fetcher.fetch_all_props(target_date, use_mock=args.mock)
    print(f"Fetched {count} props odds")
    
    # Cleanup if requested
    if args.cleanup:
        deleted = fetcher.cleanup_old_odds()
        print(f"Cleaned up {deleted} old records")

if __name__ == "__main__":
    main()
