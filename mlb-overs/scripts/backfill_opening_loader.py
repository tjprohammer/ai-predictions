#!/usr/bin/env python3
"""
Backfill opening lines loader - loads historical line data into book_odds_events table
"""
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_csv_odds_history(csv_path: str, engine, book_key: str = "pinnacle"):
    """Load CSV with columns: game_id,captured_at,line_total,over_price,under_price"""
    log.info(f"Loading odds history from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Standardize columns
    required_cols = ['game_id', 'captured_at', 'line_total', 'over_price', 'under_price']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add metadata
    df['book_key'] = book_key
    df['market'] = 'totals'
    
    # Parse timestamps
    df['captured_at'] = pd.to_datetime(df['captured_at'])
    
    # Clean data
    df = df.dropna(subset=['game_id', 'line_total'])
    df['line_total'] = pd.to_numeric(df['line_total'], errors='coerce')
    df['over_price'] = pd.to_numeric(df['over_price'], errors='coerce').fillna(-110).astype(int)
    df['under_price'] = pd.to_numeric(df['under_price'], errors='coerce').fillna(-110).astype(int)
    
    log.info(f"Processed {len(df)} odds records")
    
    # Insert with conflict handling
    df.to_sql('book_odds_events', engine, if_exists='append', index=False, method='multi')
    log.info("Inserted odds events")
    
    return len(df)

def create_proxy_openings(engine, use_proxy: bool = True):
    """Create proxy openings from current market_total if no real data exists"""
    if not use_proxy:
        return
        
    log.info("Creating proxy openings from current market_total...")
    
    with engine.begin() as conn:
        result = conn.execute(text("""
            UPDATE enhanced_games
            SET opening_total = market_total,
                opening_captured_at = COALESCE(opening_captured_at, created_at, date),
                opening_is_proxy = TRUE
            WHERE opening_total IS NULL 
              AND market_total IS NOT NULL 
              AND date >= '2024-04-01'
        """))
        
        count = result.rowcount
        log.info(f"Created {count} proxy openings")
        
        # Check coverage
        coverage_result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_games,
                COUNT(opening_total) as have_opening,
                ROUND(100.0 * COUNT(opening_total) / COUNT(*), 1) as coverage_pct
            FROM enhanced_games 
            WHERE date BETWEEN '2024-04-01' AND '2025-08-25'
        """)).fetchone()
        
        log.info(f"Opening total coverage: {coverage_result.have_opening}/{coverage_result.total_games} ({coverage_result.coverage_pct}%)")

def refresh_materialized_views(engine):
    """Refresh the opening totals materialized views"""
    log.info("Refreshing materialized views...")
    
    with engine.begin() as conn:
        try:
            conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY opening_totals_v1"))
            log.info("Refreshed opening_totals_v1")
        except Exception as e:
            log.warning(f"Concurrent refresh failed, trying normal refresh: {e}")
            conn.execute(text("REFRESH MATERIALIZED VIEW opening_totals_v1"))
            
        try:
            conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY opening_totals_consensus_v1"))
            log.info("Refreshed opening_totals_consensus_v1")
        except Exception as e:
            log.warning(f"Concurrent refresh failed, trying normal refresh: {e}")
            conn.execute(text("REFRESH MATERIALIZED VIEW opening_totals_consensus_v1"))

def update_enhanced_games_from_openings(engine, book_key: str = "pinnacle", use_consensus: bool = False):
    """Update enhanced_games with opening totals from materialized views"""
    log.info(f"Updating enhanced_games with openings from {'consensus' if use_consensus else book_key}...")
    
    with engine.begin() as conn:
        if use_consensus:
            result = conn.execute(text("""
                UPDATE enhanced_games eg
                SET opening_total = c.opening_total_consensus,
                    opening_captured_at = c.first_open_seen_at,
                    opening_is_proxy = FALSE
                FROM opening_totals_consensus_v1 c
                WHERE c.game_id::text = eg.game_id::text
                  AND (eg.opening_total IS DISTINCT FROM c.opening_total_consensus
                       OR eg.opening_captured_at IS DISTINCT FROM c.first_open_seen_at)
            """))
        else:
            result = conn.execute(text("""
                UPDATE enhanced_games eg
                SET opening_total = ot.opening_total,
                    opening_over_odds = ot.opening_over_odds,
                    opening_under_odds = ot.opening_under_odds,
                    opening_captured_at = ot.opening_captured_at,
                    opening_is_proxy = FALSE
                FROM opening_totals_v1 ot
                WHERE ot.book_key = :book_key
                  AND ot.game_id::text = eg.game_id::text
                  AND (eg.opening_total IS DISTINCT FROM ot.opening_total
                       OR eg.opening_captured_at IS DISTINCT FROM ot.opening_captured_at)
            """), book_key=book_key)
        
        count = result.rowcount
        log.info(f"Updated {count} games with opening totals")

def analyze_opening_closing_moves(engine):
    """Analyze opening to closing line movements"""
    log.info("Analyzing opening → closing line movements...")
    
    with engine.begin() as conn:
        moves = conn.execute(text("""
            SELECT
                date,
                COUNT(*) as games,
                ROUND(AVG(market_total - opening_total), 3) as avg_move,
                ROUND(STDDEV(market_total - opening_total), 3) as move_std,
                ROUND(AVG(ABS(market_total - opening_total)), 3) as avg_abs_move,
                COUNT(*) FILTER (WHERE opening_is_proxy) as proxy_count
            FROM enhanced_games
            WHERE opening_total IS NOT NULL 
              AND market_total IS NOT NULL
              AND date >= '2024-04-01'
            GROUP BY date
            ORDER BY date DESC
            LIMIT 10
        """)).fetchall()
        
        log.info("Recent opening → closing moves:")
        log.info("Date       | Games | Avg Move | Std  | Avg |Move| | Proxy")
        log.info("-" * 60)
        for row in moves:
            log.info(f"{row.date} | {row.games:5d} | {row.avg_move:8.3f} | {row.move_std:4.3f} | {row.avg_abs_move:8.3f} | {row.proxy_count:5d}")

def main():
    parser = argparse.ArgumentParser(description="Backfill opening lines data")
    parser.add_argument("--db", required=True, help="Database connection string")
    parser.add_argument("--csv", help="CSV file with historical odds data")
    parser.add_argument("--book-key", default="pinnacle", help="Book identifier")
    parser.add_argument("--use-consensus", action="store_true", help="Use consensus opening instead of single book")
    parser.add_argument("--create-proxy", action="store_true", help="Create proxy openings from market_total")
    parser.add_argument("--analyze", action="store_true", help="Analyze opening/closing moves")
    
    args = parser.parse_args()
    
    engine = create_engine(args.db)
    
    try:
        # Load CSV if provided
        if args.csv:
            if not Path(args.csv).exists():
                raise FileNotFoundError(f"CSV file not found: {args.csv}")
            load_csv_odds_history(args.csv, engine, args.book_key)
            refresh_materialized_views(engine)
            update_enhanced_games_from_openings(engine, args.book_key, args.use_consensus)
        
        # Create proxy openings if requested
        if args.create_proxy:
            create_proxy_openings(engine, use_proxy=True)
        
        # Analyze movements if requested
        if args.analyze:
            analyze_opening_closing_moves(engine)
            
        # Always show final coverage
        with engine.begin() as conn:
            coverage = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_games,
                    COUNT(opening_total) as have_opening,
                    COUNT(*) FILTER (WHERE opening_is_proxy) as proxy_count,
                    ROUND(100.0 * COUNT(opening_total) / COUNT(*), 1) as coverage_pct
                FROM enhanced_games 
                WHERE date BETWEEN '2024-04-01' AND '2025-08-25'
            """)).fetchone()
            
            log.info(f"Final coverage: {coverage.have_opening}/{coverage.total_games} ({coverage.coverage_pct}%) - {coverage.proxy_count} proxy")
            
    except Exception as e:
        log.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
