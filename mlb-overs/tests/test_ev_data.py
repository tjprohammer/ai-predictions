#!/usr/bin/env python3
"""
Quick test of the EV prediction data with diagnostic information
"""
import sys
sys.path.append('../')
from sqlalchemy import create_engine, text
import pandas as pd

def main():
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    # Check today's data with edge analysis
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT 
                pp.game_id,
                eg.market_total,
                eg.predicted_total,
                (eg.predicted_total - eg.market_total) as edge,
                pp.adj_edge,
                ROUND(pp.p_over::numeric, 3) as p_over,
                ROUND(pp.p_under::numeric, 3) as p_under,
                pp.over_odds,
                pp.under_odds,
                ROUND(pp.ev_over::numeric, 4) as ev_over,
                ROUND(pp.ev_under::numeric, 4) as ev_under,
                ROUND(pp.kelly_over::numeric, 4) as kelly_over,
                ROUND(pp.kelly_under::numeric, 4) as kelly_under
            FROM probability_predictions pp
            JOIN enhanced_games eg USING (game_id)
            WHERE pp.game_date = '2025-08-14'
            ORDER BY ABS(COALESCE(pp.adj_edge, eg.predicted_total - eg.market_total)) DESC
        """), conn)
    
    # Use adjusted edge if available, otherwise fall back to raw edge
    edge_col = "adj_edge" if "adj_edge" in df.columns and not df["adj_edge"].isna().all() else "edge"
    
    print("🎯 EV-Based Betting Recommendations for 2025-08-14")
    print("=" * 80)
    print("📊 Edge vs Probability Analysis:")
    print("Game ID    | Market | Predicted | Adj Edge | P(Over) | Status")
    print("-----------|--------|-----------|----------|---------|--------")
    
    for _, row in df.iterrows():
        edge = row[edge_col]
        p_over = row['p_over']
        if edge > 0 and p_over < 0.5:
            status = "⚠️  POS_EDGE_UNDER"
        elif edge < 0 and p_over > 0.5:
            status = "⚠️  NEG_EDGE_OVER"
        else:
            status = "✅ ALIGNED"
        
        print(f"{row['game_id']:<10} | {row['market_total']:<6} | {row['predicted_total']:<9.2f} | {edge:+8.2f} | {p_over:<7} | {status}")
    
    print("\n" + "=" * 80)
    print("🎲 EV Recommendations:")
    
    for _, row in df.iterrows():
        best_bet = "OVER" if row['ev_over'] > row['ev_under'] else "UNDER"
        best_ev = row['ev_over'] if best_bet == "OVER" else row['ev_under']
        best_kelly = row['kelly_over'] if best_bet == "OVER" else row['kelly_under']
        best_odds = row['over_odds'] if best_bet == "OVER" else row['under_odds']
        edge = row[edge_col]
        
        if best_ev > 0:
            print(f"🟢 Game {row['game_id']}: {best_bet} {row['market_total']} (odds: {best_odds:+d})")
            print(f"   EV: {best_ev:+.1%} | Kelly: {best_kelly:.1%} | Adj Edge: {edge:+.2f}")
        else:
            print(f"🔴 Game {row['game_id']}: No positive EV (adj edge: {edge:+.2f})")
    
    pos_ev_count = len(df[df[['ev_over', 'ev_under']].max(axis=1) > 0])
    edge_prob_misalign = len(df[(df[edge_col] > 0) & (df['p_over'] < 0.5)])
    
    print(f"\n📊 Summary:")
    print(f"   Games with positive EV: {pos_ev_count}")
    print(f"   Edge/Probability misalignments: {edge_prob_misalign}")
    
    if edge_prob_misalign > 0:
        print(f"\n⚠️  DIAGNOSTIC: {edge_prob_misalign} games have positive adj edges but favor UNDER")
        print("    This suggests systematic model overestimation that isotonic calibration corrects")

if __name__ == "__main__":
    main()
