#!/usr/bin/env python3
"""
coverage_check.py
==================
Check training data coverage for a date range before retraining.

Examples:
  python coverage_check.py --start 2025-07-15 --end 2025-08-14
  python coverage_check.py --start 2025-07-15 --end 2025-08-14 --require-market
"""
import os
import argparse
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def check_coverage(start_date, end_date, require_market=False):
    """Check training data coverage for a date range."""
    
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    market_filter = "AND eg.market_total IS NOT NULL" if require_market else ""
    
    query = text(f"""
        SELECT
          COUNT(*)                             AS total_games,
          COUNT(*) FILTER (WHERE lgf.total_runs IS NOT NULL
                         OR (eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL)) AS with_truth,
          COUNT(*) FILTER (WHERE eg.market_total IS NOT NULL)                            AS with_market,
          COUNT(*) FILTER (WHERE (lgf.total_runs IS NOT NULL
                              OR (eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL))
                             AND eg.market_total IS NOT NULL)                            AS fully_trainable,
          COUNT(*) FILTER (WHERE eg.home_sp_whip IS NOT NULL 
                             AND eg.away_sp_whip IS NOT NULL)                           AS with_pitcher_stats,
          COUNT(*) FILTER (WHERE eg.home_team_avg IS NOT NULL 
                             AND eg.away_team_avg IS NOT NULL)                          AS with_team_stats,
          MIN(lgf."date") AS min_date,
          MAX(lgf."date") AS max_date
        FROM legitimate_game_features lgf
        JOIN enhanced_games eg
          ON eg.game_id = lgf.game_id AND eg."date" = lgf."date"
        WHERE lgf."date" BETWEEN :start_d AND :end_d
          {market_filter}
    """)
    
    try:
        result = pd.read_sql(query, engine, params={
            "start_d": start_date, 
            "end_d": end_date
        })
        
        if result.empty or result.iloc[0]['total_games'] == 0:
            print("❌ No games found in the specified date range.")
            return False
            
        row = result.iloc[0]
        total = row['total_games']
        with_truth = row['with_truth']
        with_market = row['with_market']
        fully_trainable = row['fully_trainable']
        with_pitcher_stats = row['with_pitcher_stats']
        with_team_stats = row['with_team_stats']
        
        print(f"📊 Training Data Coverage: {start_date} → {end_date}")
        print("=" * 60)
        print(f"Total games:           {total:4d}")
        print(f"With final scores:     {with_truth:4d} ({with_truth/total*100:5.1f}%)")
        print(f"With market totals:    {with_market:4d} ({with_market/total*100:5.1f}%)")
        print(f"Fully trainable:       {fully_trainable:4d} ({fully_trainable/total*100:5.1f}%)")
        print(f"With pitcher stats:    {with_pitcher_stats:4d} ({with_pitcher_stats/total*100:5.1f}%)")
        print(f"With team stats:       {with_team_stats:4d} ({with_team_stats/total*100:5.1f}%)")
        print(f"Date range:            {row['min_date']} → {row['max_date']}")
        
        # Quality assessments
        if fully_trainable < 100:
            print(f"\n⚠️  Warning: Only {fully_trainable} fully trainable games found.")
            print("   Consider running backfill or extending date range.")
        elif fully_trainable < 200:
            print(f"\n⚠️  Warning: Only {fully_trainable} fully trainable games.")
            print("   Model may benefit from larger training set.")
        else:
            print(f"\n✅ Good: {fully_trainable} fully trainable games available.")
        
        if with_truth / total < 0.9:
            print(f"⚠️  Warning: Only {with_truth/total*100:.1f}% of games have final scores.")
        
        if with_market / total < 0.8:
            print(f"⚠️  Warning: Only {with_market/total*100:.1f}% of games have market totals.")
        
        if with_pitcher_stats / total < 0.8:
            print(f"⚠️  Warning: Only {with_pitcher_stats/total*100:.1f}% of games have pitcher stats.")
            
        if with_team_stats / total < 0.8:
            print(f"⚠️  Warning: Only {with_team_stats/total*100:.1f}% of games have team stats.")
        
        # Training recommendation
        print(f"\n🚀 Recommended retraining command:")
        window_days = (datetime.strptime(end_date, "%Y-%m-%d") - 
                      datetime.strptime(start_date, "%Y-%m-%d")).days + 1
        holdout_days = min(7, max(3, window_days // 5))  # 20% holdout, min 3, max 7 days
        
        cmd = f"python retrain_model.py --end {end_date} --window-days {window_days} --holdout-days {holdout_days}"
        if require_market:
            cmd += " --require-market"
        cmd += " --audit --deploy"
        print(cmd)
        
        return fully_trainable >= 100
        
    except Exception as e:
        print(f"❌ Error checking coverage: {e}")
        return False
    finally:
        engine.dispose()

def main():
    parser = argparse.ArgumentParser(description="Check training data coverage")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--require-market", action="store_true", 
                       help="Only count games with market data")
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        return False
    
    if end_date < start_date:
        print("❌ End date must be >= start date")
        return False
    
    return check_coverage(start_date, end_date, args.require_market)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
