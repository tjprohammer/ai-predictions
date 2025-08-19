#!/usr/bin/env python3
"""
Production System Summary

Shows the current status of the MLB betting system with key metrics.
"""

import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas as pd

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def main():
    print("[STATUS] MLB BETTING SYSTEM - Production Status")
    print("=" * 50)
    print(f"📅 Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    with engine.begin() as conn:
        # Latest run info
        latest = conn.execute(text("""
            SELECT run_id, game_date, sigma, temp_s, bias, calibration_samples, created_at,
                   (SELECT COUNT(*) FROM probability_predictions pp WHERE pp.run_id = cm.run_id) as bet_count
            FROM calibration_meta cm
            ORDER BY created_at DESC LIMIT 1
        """)).fetchone()
        
        if latest:
            print(f"\n🆔 Latest Run: {str(latest.run_id)[:8]}...")
            print(f"📅 Date: {latest.game_date}")
            print(f"🕒 Created: {latest.created_at}")
            print(f"🔬 Calibration: σ={latest.sigma:.3f}, temp_s={latest.temp_s:.3f}, bias={latest.bias:+.3f}")
            print(f"📊 Samples: {latest.calibration_samples}")
            print(f"🎲 Bets: {latest.bet_count}")
            
            # Get betting details for latest run using the new view
            bets = pd.read_sql(text("""
                SELECT lpp.game_id, lpp.recommendation, lpp.p_side, 
                       GREATEST(lpp.ev_over, lpp.ev_under) as best_ev,
                       GREATEST(lpp.kelly_over, lpp.kelly_under) as best_kelly,
                       lpp.stake, lpp.priced_total, lpp.priced_book
                FROM latest_probability_predictions lpp
                WHERE lpp.run_id = :rid AND lpp.recommendation != 'PASS'
                ORDER BY GREATEST(lpp.kelly_over, lpp.kelly_under) DESC
            """), conn, params={"rid": str(latest.run_id)})
            
            if not bets.empty:
                print(f"\n📋 Betting Recommendations ({len(bets)} games):")
                for _, bet in bets.iterrows():
                    print(f"   {bet.game_id} {bet.recommendation:>5} | P={bet.p_side:.2f} | EV={bet.best_ev:+.1%} | K={bet.best_kelly:.1%} | ${bet.stake:.0f}")
                
                total_stake = bets.stake.sum()
                avg_ev = bets.best_ev.mean()
                print(f"\n💰 Portfolio: ${total_stake:.0f} total, {avg_ev:+.1%} avg EV")
        else:
            print("\n❌ No runs found")
        
        # Recent performance (if any outcomes logged)
        outcomes = conn.execute(text("""
            SELECT COUNT(*) as total_bets,
                   SUM(CASE WHEN won = TRUE THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN won = FALSE THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN won IS NULL THEN 1 ELSE 0 END) as pushes,
                   SUM(stake) as total_stake,
                   SUM(pnl) as total_pnl,
                   AVG(clv) as avg_clv
            FROM bet_outcomes
            WHERE game_date >= :start
        """), {"start": datetime.now().date() - timedelta(days=7)}).fetchone()
        
        if outcomes and outcomes.total_bets > 0:
            win_rate = outcomes.wins / (outcomes.wins + outcomes.losses) if (outcomes.wins + outcomes.losses) > 0 else 0
            roi = outcomes.total_pnl / outcomes.total_stake if outcomes.total_stake > 0 else 0
            
            print(f"\n📊 Last 7 Days Performance:")
            print(f"   Record: {outcomes.wins}-{outcomes.losses}-{outcomes.pushes} ({win_rate:.1%})")
            print(f"   Stake: ${outcomes.total_stake:.0f}")
            print(f"   P&L: ${outcomes.total_pnl:+.0f}")
            print(f"   ROI: {roi:+.1%}")
            print(f"   CLV: {outcomes.avg_clv:+.0f} cents" if outcomes.avg_clv else "   CLV: N/A")
        
        # System health checks
        print(f"\n🔧 System Health:")
        
        # Check recent data availability
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        games_today = conn.execute(text("""
            SELECT COUNT(*) FROM enhanced_games WHERE "date" = :d
        """), {"d": today}).scalar()
        
        games_yesterday = conn.execute(text("""
            SELECT COUNT(*) FROM enhanced_games WHERE "date" = :d
        """), {"d": yesterday}).scalar()
        
        print(f"   Games today ({today}): {games_today}")
        print(f"   Games yesterday ({yesterday}): {games_yesterday}")
        
        # Check for recent predictions
        recent_preds = conn.execute(text("""
            SELECT COUNT(*) FROM probability_predictions 
            WHERE game_date >= :d
        """), {"d": today}).scalar()
        
        print(f"   Recent predictions: {recent_preds}")
        
        if games_today == 0:
            print("   ⚠️  No games found for today - check data pipeline")
        if recent_preds == 0:
            print("   ⚠️  No predictions generated today")
        
        print(f"\n✅ System operational and ready for production")
        print(f"💡 Next: Run 'python daily_runbook.py --date {today} --mode predictions'")

if __name__ == "__main__":
    main()
