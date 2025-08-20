#!/usr/bin/env python3
"""
Historical Backfill Analysis Script
=================================

Quick analysis of the historical backfill results with comprehensive queries
for checking model performance, accuracy, and data coverage.

Usage:
  python analyze_backfill.py --start 2025-05-01 --end 2025-08-15
  python analyze_backfill.py --date 2025-08-14  # single day analysis
"""

import os, argparse, sys
import pandas as pd
from sqlalchemy import create_engine, text

# force UTF-8 on Windows consoles to avoid �
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def american_to_decimal(odds: int) -> float:
    # -110 -> 1.909..., +150 -> 2.5
    return (100/abs(odds) + 1) if odds < 0 else (odds/100 + 1)

def create_results_view():
    """Create the api_results_history view for easy analysis"""
    engine = get_engine()
    
    view_sql = """
    CREATE OR REPLACE VIEW api_results_history AS
    SELECT eg.game_id, eg."date", eg.home_team, eg.away_team,
           eg.market_total, eg.predicted_total, eg.total_runs,
           ROUND(eg.predicted_total - eg.market_total,2) AS edge,
           CASE
             WHEN eg.predicted_total IS NULL OR eg.market_total IS NULL THEN NULL
             WHEN (eg.predicted_total - eg.market_total) >=  1.0 THEN 'OVER'
             WHEN (eg.predicted_total - eg.market_total) <= -1.0 THEN 'UNDER'
             ELSE 'NO BET'
           END AS pick,
           CASE
             WHEN eg.total_runs IS NULL OR eg.market_total IS NULL THEN NULL
             WHEN eg.total_runs > eg.market_total THEN 'OVER'
             WHEN eg.total_runs < eg.market_total THEN 'UNDER'
             ELSE 'PUSH'
           END AS result,
           CASE
             WHEN ABS(eg.predicted_total - eg.market_total) < 1.0 OR eg.total_runs = eg.market_total THEN NULL
             WHEN (eg.predicted_total - eg.market_total >= 1.0 AND eg.total_runs > eg.market_total)
               OR (eg.predicted_total - eg.market_total <= -1.0 AND eg.total_runs < eg.market_total)
               THEN TRUE ELSE FALSE END AS correct
    FROM enhanced_games eg
    """
    
    try:
        with engine.begin() as conn:
            conn.execute(text(view_sql))
        print("✅ Created api_results_history view")
        return True
    except Exception as e:
        print(f"❌ Failed to create view: {e}")
        return False

def coverage_analysis(start_date, end_date):
    """Analyze data coverage for the specified period"""
    engine = get_engine()
    
    query = """
    SELECT "date",
           COUNT(*) AS games,
           COUNT(market_total) AS markets,
           COUNT(predicted_total) AS preds,
           COUNT(total_runs) AS finals
    FROM enhanced_games
    WHERE "date" BETWEEN :start AND :end
    GROUP BY 1 ORDER BY 1
    """
    
    print(f"\n{'='*80}")
    print("DATA COVERAGE ANALYSIS")
    print(f"{'='*80}")
    
    with engine.begin() as conn:
        result = conn.execute(text(query), {"start": start_date, "end": end_date}).fetchall()
    
    print(f"{'Date':<12} {'Games':<6} {'Markets':<8} {'Preds':<6} {'Finals':<7}")
    print("-" * 80)
    
    total_games = total_markets = total_preds = total_finals = 0
    missing_preds_dates = []
    
    for row in result:
        date, games, markets, preds, finals = row
        total_games += games
        total_markets += markets
        total_preds += preds
        total_finals += finals
        
        if preds == 0 and games > 0:
            missing_preds_dates.append(str(date))
        
        print(f"{date:<12} {games:<6} {markets:<8} {preds:<6} {finals:<7}")
    
    print("-" * 80)
    print(f"{'TOTAL':<12} {total_games:<6} {total_markets:<8} {total_preds:<6} {total_finals:<7}")
    
    market_coverage = (total_markets / total_games * 100) if total_games > 0 else 0
    pred_coverage = (total_preds / total_games * 100) if total_games > 0 else 0
    finals_coverage = (total_finals / total_games * 100) if total_games > 0 else 0
    
    print(f"\nCOVERAGE PERCENTAGES:")
    print(f"Markets: {market_coverage:.1f}%")
    print(f"Predictions: {pred_coverage:.1f}%")
    print(f"Finals: {finals_coverage:.1f}%")
    
    if missing_preds_dates:
        print(f"\n⚠️  DATES MISSING PREDICTIONS: {', '.join(missing_preds_dates)}")
        print("Run: python predict_from_range.py --start YYYY-MM-DD --end YYYY-MM-DD --thr 1.0")

def metrics_over_range(start_date, end_date, thr=1.0, odds=-110):
    """Calculate comprehensive metrics over a date range"""
    engine = get_engine()
    q = text("""
      WITH rows AS (
        SELECT market_total, predicted_total, total_runs,
               (predicted_total - market_total) AS edge,
               CASE WHEN total_runs IS NULL OR market_total IS NULL THEN NULL
                    WHEN total_runs > market_total THEN 'OVER'
                    WHEN total_runs < market_total THEN 'UNDER'
                    ELSE 'PUSH' END AS result
        FROM enhanced_games
        WHERE "date" BETWEEN :s AND :e
          AND predicted_total IS NOT NULL
          AND market_total IS NOT NULL
          AND total_runs IS NOT NULL
      ),
      bets AS (
        SELECT *,
               CASE WHEN edge >= :thr THEN 'OVER'
                    WHEN edge <= -:thr THEN 'UNDER'
                    ELSE 'NO BET' END AS pick
        FROM rows
      )
      SELECT
        COUNT(*)                                           AS n_rows,
        AVG(ABS(predicted_total - total_runs))             AS mae_model,
        AVG(ABS(market_total - total_runs))                AS mae_market,
        AVG(predicted_total - total_runs)                  AS bias_pred_minus_actual,
        COUNT(*) FILTER (WHERE pick IN ('OVER','UNDER'))   AS n_bets,
        COUNT(*) FILTER (WHERE pick=result)                AS wins,
        COUNT(*) FILTER (WHERE pick IN ('OVER','UNDER') AND result IN ('OVER','UNDER') AND pick<>result) AS losses,
        COUNT(*) FILTER (WHERE result='PUSH' AND pick IN ('OVER','UNDER')) AS pushes
      FROM bets
    """)
    with engine.begin() as conn:
        r = conn.execute(q, {"s": start_date, "e": end_date, "thr": thr}).mappings().one()

    n_bets = r["n_bets"] or 0
    wins   = r["wins"] or 0
    losses = r["losses"] or 0
    pushes = r["pushes"] or 0

    # ROI per bet in units using American odds
    dec = american_to_decimal(odds)  # e.g., -110 -> 1.909...
    win_profit = dec - 1.0           # profit per 1u stake
    roi = (wins * win_profit - losses * 1.0) / (n_bets if n_bets else 1)

    print(f"\n{'='*80}")
    print("RANGE METRICS")
    print(f"{'='*80}")
    print(f"Rows with truth+prediction: {r['n_rows'] or 0}")
    print(f"Model MAE vs truth:   {float(r['mae_model'] or 0):6.3f}")
    print(f"Market MAE vs truth:  {float(r['mae_market'] or 0):6.3f}")
    print(f"Model bias (pred-actual): {float(r['bias_pred_minus_actual'] or 0):+6.3f}")
    print(f"\nBetting rule: |edge|>={thr} @ odds {odds}")
    if n_bets:
        winrate = wins / max(1, (n_bets - pushes)) * 100
    else:
        winrate = 0.0
    print(f"Placed bets: {n_bets}  |  Wins: {wins}  Push: {pushes}")
    print(f"Win rate: {winrate:.1f}% (pushes excluded)")
    print(f"Avg ROI per bet: {roi:+.3f}u")

def _fmt(x):
    """Format value for display, showing '-' for None"""
    return "-" if x is None else x
    """Analyze model accuracy by date"""
    engine = get_engine()
    
    query = """
    WITH labeled AS (
      SELECT "date",
             CASE
               WHEN predicted_total IS NULL OR market_total IS NULL THEN NULL
               WHEN (predicted_total - market_total) >=  1.0 THEN 'OVER'
               WHEN (predicted_total - market_total) <= -1.0 THEN 'UNDER'
               ELSE 'NO BET'
             END AS pick,
             CASE
               WHEN total_runs IS NULL OR market_total IS NULL THEN NULL
               WHEN total_runs > market_total THEN 'OVER'
               WHEN total_runs < market_total THEN 'UNDER'
               ELSE 'PUSH'
             END AS result
      FROM enhanced_games
      WHERE "date" BETWEEN :start AND :end
    )
    SELECT "date",
           COUNT(*)                              AS games,
           COUNT(*) FILTER (WHERE pick IN ('OVER','UNDER')) AS bets,
           COUNT(*) FILTER (WHERE pick IN ('OVER','UNDER') AND result IN ('OVER','UNDER') AND pick=result) AS wins,
           COUNT(*) FILTER (WHERE result='PUSH') AS pushes
    FROM labeled
    GROUP BY 1
    ORDER BY 1
    """
    
    print(f"\n{'='*80}")
    print("DAILY ACCURACY ANALYSIS")
    print(f"{'='*80}")
    
    with engine.begin() as conn:
        result = conn.execute(text(query), {"start": start_date, "end": end_date}).fetchall()
    
    print(f"{'Date':<12} {'Games':<6} {'Bets':<5} {'Wins':<5} {'Win%':<6} {'Pushes':<7}")
    print("-" * 80)
    
    total_games = total_bets = total_wins = total_pushes = 0
    
    for row in result:
        date, games, bets, wins, pushes = row
        total_games += games
        total_bets += bets
        total_wins += wins
        total_pushes += pushes
        
        win_pct = (wins / bets * 100) if bets > 0 else 0
        print(f"{date:<12} {games:<6} {bets:<5} {wins:<5} {win_pct:<6.1f} {pushes:<7}")
    
    print("-" * 80)
    overall_win_pct = (total_wins / total_bets * 100) if total_bets > 0 else 0
    print(f"{'TOTAL':<12} {total_games:<6} {total_bets:<5} {total_wins:<5} {overall_win_pct:<6.1f} {total_pushes:<7}")
    
    # Break-even analysis
    breakeven_pct = 52.38  # Need ~52.38% to break even at -110 odds
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"Overall Win Rate: {overall_win_pct:.1f}%")
    print(f"Break-even Rate: {breakeven_pct:.1f}%")
    if overall_win_pct > breakeven_pct:
        print(f"✅ PROFITABLE (+{overall_win_pct - breakeven_pct:.1f}% above break-even)")
    else:
        print(f"❌ UNPROFITABLE ({breakeven_pct - overall_win_pct:.1f}% below break-even)")

def single_date_analysis(date):
    """Detailed analysis for a single date"""
    engine = get_engine()
    
    query = """
    WITH base AS (
      SELECT game_id, "date", home_team, away_team,
             market_total, predicted_total, total_runs,
             ROUND(predicted_total - market_total, 2) AS edge
      FROM enhanced_games
      WHERE "date" = :date
    ),
    labeled AS (
      SELECT *,
        CASE
          WHEN predicted_total IS NULL OR market_total IS NULL THEN NULL
          WHEN edge >=  1.0 THEN 'OVER'
          WHEN edge <= -1.0 THEN 'UNDER'
          ELSE 'NO BET'
        END AS pick,
        CASE
          WHEN total_runs IS NULL OR market_total IS NULL THEN NULL
          WHEN total_runs > market_total THEN 'OVER'
          WHEN total_runs < market_total THEN 'UNDER'
          ELSE 'PUSH'
        END AS result
      FROM base
    )
    SELECT game_id, date, away_team || ' @ ' || home_team AS matchup,
           market_total, predicted_total, total_runs, edge, pick, result,
           CASE WHEN pick IN ('OVER','UNDER') AND result IN ('OVER','UNDER')
                     THEN (pick = result) ELSE NULL END AS correct
    FROM labeled
    ORDER BY ABS(edge) DESC, game_id
    """
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS FOR {date}")
    print(f"{'='*80}")
    
    with engine.begin() as conn:
        df = pd.read_sql(text(query), conn, params={"date": date})
    
    if df.empty:
        print(f"No games found for {date}")
        return
    
    print(f"{'Game ID':<8} {'Matchup':<25} {'Market':<7} {'Pred':<6} {'Final':<6} {'Edge':<6} {'Pick':<6} {'Result':<6} {'Correct':<7}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        correct_str = "✅" if row['correct'] == True else "❌" if row['correct'] == False else "-"
        print(f"{row['game_id']:<8} {row['matchup']:<25} {row['market_total'] or '-':<7} "
              f"{row['predicted_total'] or '-':<6} {row['total_runs'] or '-':<6} "
              f"{row['edge'] or '-':<6} {row['pick'] or '-':<6} {row['result'] or '-':<6} {correct_str:<7}")
    
    # Summary stats for the date
    bets = df[df['pick'].isin(['OVER', 'UNDER'])]
    if not bets.empty:
        wins = bets[bets['correct'] == True]
        win_rate = len(wins) / len(bets) * 100
        print(f"\nDATE SUMMARY: {len(bets)} bets, {len(wins)} wins ({win_rate:.1f}%)")

def check_reports_directory():
    """Check if evaluation reports exist"""
    reports_dir = "reports"
    if os.path.exists(reports_dir):
        print(f"\n📊 EVALUATION REPORTS AVAILABLE:")
        for file in os.listdir(reports_dir):
            if file.endswith(('.csv', '.png')):
                print(f"   {file}")
    else:
        print(f"\n⚠️  No reports directory found. Run evaluation step to generate charts.")

def accuracy_analysis(start_date, end_date):
    """Legacy analysis function"""
    print(f"\nACCURACY ANALYSIS: {start_date} to {end_date}")
    metrics_over_range(start_date, end_date, thr=1.0, odds=-110)

def main():
    parser = argparse.ArgumentParser(description="Analyze historical backfill results")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--date", help="Single date analysis (YYYY-MM-DD)")
    parser.add_argument("--create-view", action="store_true", help="Create api_results_history view")
    parser.add_argument("--threshold", "-t", type=float, default=1.0, help="Edge threshold for metrics")
    parser.add_argument("--odds", "-o", type=int, default=-110, help="American odds for ROI calculation")
    parser.add_argument("--export-csv", "-c", action="store_true", help="Export CSV results (future)")
    
    args = parser.parse_args()
    
    if args.create_view:
        create_results_view()
        return
    
    if args.date:
        create_results_view()
        single_date_analysis(args.date)
    elif args.start and args.end:
        create_results_view()
        coverage_analysis(args.start, args.end)
        print(f"\nCalculating metrics with threshold={args.threshold}, odds={args.odds}")
        metrics_over_range(args.start, args.end, thr=args.threshold, odds=args.odds)
        check_reports_directory()
    else:
        print("ERROR: Specify either --date for single day or --start/--end for range analysis")
        return 1
    
    if args.export_csv:
        print("\n📋 CSV export functionality to be implemented in future update")
    
    print(f"\n💡 TIP: Use these queries for further analysis:")
    print("   SELECT * FROM api_results_history WHERE date='2025-08-14' ORDER BY ABS(edge) DESC;")
    print("   SELECT * FROM api_results_history WHERE pick='OVER' AND correct=true;")

if __name__ == "__main__":
    main()
