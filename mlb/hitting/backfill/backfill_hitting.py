"""
MLB Hitting Backfill Script

Backfills player game logs for a date range and refreshes materialized views.
This should be run once to populate historical data before daily predictions.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text
import time

# Add ingestion path for player logs fetcher
sys.path.append(str(Path(__file__).parent.parent / 'ingestion'))
from fetch_player_logs import MLBPlayerLogsFetcher

log = logging.getLogger(__name__)

def populate_pitcher_handedness(engine):
    """Populate pitcher handedness from player_game_logs data"""
    
    log.info("Populating pitcher handedness reference...")
    
    with engine.begin() as conn:
        # Extract unique pitcher handedness from game logs
        insert_query = text("""
            INSERT INTO pitchers (pitcher_id, pitcher_name, throws_hand)
            SELECT DISTINCT 
                starting_pitcher_id,
                starting_pitcher_name,
                starting_pitcher_hand
            FROM player_game_logs 
            WHERE starting_pitcher_id IS NOT NULL 
              AND starting_pitcher_hand IS NOT NULL
              AND starting_pitcher_name IS NOT NULL
            ON CONFLICT (pitcher_id) 
            DO UPDATE SET 
                pitcher_name = EXCLUDED.pitcher_name,
                throws_hand = EXCLUDED.throws_hand
        """)
        
        result = conn.execute(insert_query)
        count = result.rowcount
        
    log.info(f"Populated {count} pitcher records")
    return count

def refresh_views(engine):
    """Refresh all materialized views"""
    
    log.info("Refreshing materialized views...")
    
    views = ['mv_bvp_agg', 'mv_hitter_form', 'mv_pa_distribution']
    
    with engine.begin() as conn:
        for view in views:
            log.info(f"Refreshing {view}...")
            start_time = time.time()
            
            # Use CONCURRENTLY if the view already exists and has data
            try:
                conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
            except Exception as e:
                # Fall back to regular refresh if concurrent fails
                log.warning(f"Concurrent refresh failed for {view}, using regular: {e}")
                conn.execute(text(f"REFRESH MATERIALIZED VIEW {view}"))
            
            elapsed = time.time() - start_time
            log.info(f"Refreshed {view} in {elapsed:.1f}s")
    
    log.info("All materialized views refreshed")

def generate_date_range(start_date: str, end_date: str):
    """Generate list of dates between start and end (inclusive)"""
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    current = start
    dates = []
    
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def backfill_player_logs(engine, start_date: str, end_date: str, 
                        skip_existing: bool = True):
    """Backfill player logs for date range"""
    
    log.info(f"Backfilling player logs from {start_date} to {end_date}")
    
    # Check what dates we already have if skipping existing
    existing_dates = set()
    if skip_existing:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT date::text 
                FROM player_game_logs 
                WHERE date BETWEEN :start_date AND :end_date
            """), {'start_date': start_date, 'end_date': end_date})
            
            existing_dates = {row[0] for row in result}
        
        log.info(f"Found existing data for {len(existing_dates)} dates")
    
    # Generate date range
    all_dates = generate_date_range(start_date, end_date)
    
    # Filter out existing dates if skipping
    if skip_existing:
        dates_to_fetch = [d for d in all_dates if d not in existing_dates]
        log.info(f"Will fetch {len(dates_to_fetch)} new dates (skipping {len(existing_dates)} existing)")
    else:
        dates_to_fetch = all_dates
        log.info(f"Will fetch all {len(dates_to_fetch)} dates (overwriting any existing)")
    
    if not dates_to_fetch:
        log.info("No dates to fetch")
        return 0
    
    # Initialize fetcher
    fetcher = MLBPlayerLogsFetcher(engine)
    
    total_logs = 0
    errors = []
    
    # Process each date
    for i, date in enumerate(dates_to_fetch):
        log.info(f"Processing {date} ({i+1}/{len(dates_to_fetch)})")
        
        try:
            logs_count = fetcher.fetch_date(date)
            total_logs += logs_count
            
            if logs_count == 0:
                log.warning(f"No logs found for {date}")
            
            # Be nice to the API
            time.sleep(1)
            
        except Exception as e:
            log.error(f"Error fetching {date}: {e}")
            errors.append((date, str(e)))
            continue
    
    log.info(f"Backfill complete: {total_logs} total logs from {len(dates_to_fetch)} dates")
    
    if errors:
        log.warning(f"Errors encountered for {len(errors)} dates:")
        for date, error in errors[:5]:  # Show first 5 errors
            log.warning(f"  {date}: {error}")
        if len(errors) > 5:
            log.warning(f"  ... and {len(errors) - 5} more")
    
    return total_logs

def main():
    """Main backfill orchestrator"""
    
    parser = argparse.ArgumentParser(description='Backfill MLB hitting data')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip dates that already have data (default: True)')
    parser.add_argument('--force', action='store_true', 
                       help='Overwrite existing data (opposite of --skip-existing)')
    parser.add_argument('--skip-refresh', action='store_true',
                       help='Skip materialized view refresh')
    parser.add_argument('--refresh-only', action='store_true',
                       help='Only refresh views, skip data fetching')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / '.env')
    
    # Connect to database
    try:
        engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            
    except Exception as e:
        log.error(f"Database connection failed: {e}")
        sys.exit(1)
    
    skip_existing = args.skip_existing and not args.force
    
    start_time = time.time()
    
    try:
        if not args.refresh_only:
            # Step 1: Backfill player logs
            total_logs = backfill_player_logs(
                engine, 
                args.start, 
                args.end, 
                skip_existing=skip_existing
            )
            
            # Step 2: Populate pitcher handedness
            pitcher_count = populate_pitcher_handedness(engine)
        
        # Step 3: Refresh materialized views
        if not args.skip_refresh:
            refresh_views(engine)
        
        elapsed = time.time() - start_time
        
        if args.refresh_only:
            log.info(f"✅ View refresh completed in {elapsed:.1f}s")
        else:
            log.info(f"✅ Backfill completed in {elapsed:.1f}s")
            log.info(f"   Player logs: {total_logs}")
            log.info(f"   Pitchers: {pitcher_count}")
            if not args.skip_refresh:
                log.info(f"   Views: refreshed")
        
    except Exception as e:
        log.error(f"Backfill failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
