"""
MLB Hitting Props Predictor

Predicts probabilities for "≥1 hit" (HITS 0.5) and "≥2 hits" (HITS 1.5) 
using empirical Bayes blending of form, vs-hand splits, and BvP data.
"""

import math
import pandas as pd
from sqlalchemy import text, create_engine
from dataclasses import dataclass
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class EB:
    """Empirical Bayes parameters"""
    alpha: float = 4.0  # Prior hits
    beta: float = 16.0  # Prior non-hits

def eb_rate(h, ab, eb: EB):
    """Calculate EB-adjusted hit rate"""
    h = 0 if h is None else h
    ab = 0 if ab is None else ab
    return (h + eb.alpha) / (ab + eb.alpha + eb.beta)

def p_ge1_hit(p_pa, exp_pa):
    """Probability of ≥1 hit using binomial"""
    p_pa = max(0.01, min(0.9, p_pa))
    exp_pa = max(2.5, min(5.8, exp_pa))
    return 1 - (1 - p_pa) ** exp_pa

def p_ge2_hits(p_pa, exp_ab):
    """Probability of ≥2 hits using binomial"""
    p_pa = max(0.01, min(0.9, p_pa))
    n = int(round(max(2.0, min(6.0, exp_ab))))
    # 1 - [P(0) + P(1)]
    p0 = (1 - p_pa) ** n
    p1 = n * p_pa * ((1 - p_pa) ** (n - 1))
    return max(0.0, min(1.0, 1 - (p0 + p1)))

def american_to_prob(odds):
    """Convert American odds to implied probability"""
    if odds is None:
        return None
    return 100/(odds+100) if odds > 0 else (-odds)/((-odds)+100)

def kelly(p, odds):
    """Calculate Kelly criterion bet size"""
    if odds is None:
        return None
    b = (odds/100) if odds > 0 else (100/(-odds))
    return max(0.0, (p*(b+1) - 1)/b)

def build_features(engine, target_date: str) -> pd.DataFrame:
    """Build hitting features for target date"""
    
    eb = EB()
    
    # Get today's hitters tied to today's games
    hitters_query = text("""
        SELECT DISTINCT 
            p.player_id, p.player_name, p.team, 
            CASE 
                WHEN p.team = eg.home_team THEN eg.away_team
                WHEN p.team = eg.away_team THEN eg.home_team
                ELSE NULL
            END as opponent,
            eg.game_id, eg.date, p.lineup_spot, 
            CASE 
                WHEN p.team = eg.home_team THEN 'H'
                WHEN p.team = eg.away_team THEN 'A'
                ELSE NULL
            END as home_away,
            CASE 
                WHEN p.team = eg.home_team THEN eg.away_sp_id
                WHEN p.team = eg.away_team THEN eg.home_sp_id
                ELSE NULL
            END as sp_id
        FROM enhanced_games eg
        JOIN (
            SELECT DISTINCT player_id, player_name, team, lineup_spot,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date DESC) as rn
            FROM player_game_logs 
            WHERE date >= :d::date - INTERVAL '7 days'
              AND at_bats > 0
        ) p ON p.rn = 1 AND (p.team = eg.home_team OR p.team = eg.away_team)
        WHERE eg.date = :d
    """)
    
    hitters = pd.read_sql(hitters_query, engine, params={"d": target_date})
    
    if hitters.empty:
        log.warning(f"No hitters found for {target_date}")
        return hitters
    
    log.info(f"Found {len(hitters)} hitters for {target_date}")
    
    # Get form windows
    form_query = text("""
        SELECT player_id, date, hits_l5, ab_l5, hits_l10, ab_l10, hits_l15, ab_l15
        FROM mv_hitter_form 
        WHERE date <= :d 
          AND player_id = ANY(:pids)
          AND date = (
              SELECT MAX(date) 
              FROM mv_hitter_form mhf2 
              WHERE mhf2.player_id = mv_hitter_form.player_id 
                AND mhf2.date <= :d
          )
    """)
    
    player_ids = list(hitters.player_id.unique())
    form = pd.read_sql(form_query, engine, params={"d": target_date, "pids": player_ids})
    hitters = hitters.merge(form, on=["player_id"], how="left")
    
    # Get season vs-hand splits
    splits_query = text("""
        SELECT player_id,
               SUM(CASE WHEN starting_pitcher_hand='R' THEN hits ELSE 0 END) AS h_vs_r,
               SUM(CASE WHEN starting_pitcher_hand='R' THEN at_bats ELSE 0 END) AS ab_vs_r,
               SUM(CASE WHEN starting_pitcher_hand='L' THEN hits ELSE 0 END) AS h_vs_l,
               SUM(CASE WHEN starting_pitcher_hand='L' THEN at_bats ELSE 0 END) AS ab_vs_l
        FROM player_game_logs
        WHERE player_id = ANY(:pids) 
          AND date <= :d
          AND starting_pitcher_hand IS NOT NULL
        GROUP BY player_id
    """)
    
    splits = pd.read_sql(splits_query, engine, params={"pids": player_ids, "d": target_date})
    hitters = hitters.merge(splits, on="player_id", how="left")
    
    # Get starting pitcher handedness
    sp_ids = list(hitters.dropna(subset=['sp_id']).sp_id.astype(int).unique())
    if sp_ids:
        sp_query = text("""
            SELECT pitcher_id, throws_hand AS sp_hand 
            FROM pitchers
            WHERE pitcher_id = ANY(:ids)
        """)
        
        sp = pd.read_sql(sp_query, engine, params={"ids": sp_ids})
        hitters = hitters.merge(sp, left_on="sp_id", right_on="pitcher_id", how="left")
    else:
        hitters['sp_hand'] = 'R'  # Default
    
    # Get BvP data
    if sp_ids:
        bvp_query = text("""
            SELECT player_id, pitcher_id, ab, h
            FROM mv_bvp_agg
            WHERE player_id = ANY(:pids) 
              AND pitcher_id = ANY(:spids)
        """)
        
        bvp = pd.read_sql(bvp_query, engine, params={"pids": player_ids, "spids": sp_ids})
        hitters = hitters.merge(bvp, left_on=["player_id","sp_id"], right_on=["player_id","pitcher_id"], how="left")
    else:
        hitters['ab'] = None
        hitters['h'] = None
    
    # Get expected PA/AB by lineup spot and home_away
    pa_dist_query = text("SELECT lineup_spot, home_away, exp_pa, exp_ab FROM mv_pa_distribution")
    pa_dist = pd.read_sql(pa_dist_query, engine)
    hitters = hitters.merge(pa_dist, on=["lineup_spot","home_away"], how="left")
    
    # Fill expected PA/AB with defaults
    def get_exp_pa(lineup_spot):
        if pd.isna(lineup_spot):
            return 4.0
        return 4.2 if 1 <= lineup_spot <= 5 else 3.7
    
    hitters["exp_pa"] = hitters["exp_pa"].fillna(hitters["lineup_spot"].apply(get_exp_pa))
    hitters["exp_ab"] = hitters["exp_ab"].fillna(hitters["exp_pa"] * 0.85)
    
    # Compute per-PA hit probability (EB blend)
    def compute_p_pa(r):
        # Recent form (prefer 10, fallback to 5 or 15)
        if pd.notna(r.get("hits_l10")) and pd.notna(r.get("ab_l10")) and r.get("ab_l10", 0) >= 20:
            form_hits, form_ab = r["hits_l10"], r["ab_l10"]
        elif pd.notna(r.get("hits_l5")) and pd.notna(r.get("ab_l5")) and r.get("ab_l5", 0) >= 10:
            form_hits, form_ab = r["hits_l5"], r["ab_l5"]
        elif pd.notna(r.get("hits_l15")) and pd.notna(r.get("ab_l15")):
            form_hits, form_ab = r["hits_l15"], r["ab_l15"]
        else:
            form_hits, form_ab = 0, 0
            
        form_rate = eb_rate(form_hits, form_ab, eb)
        
        # Vs-hand splits
        vs_r = eb_rate(r.get("h_vs_r", 0), r.get("ab_vs_r", 0), eb)
        vs_l = eb_rate(r.get("h_vs_l", 0), r.get("ab_vs_l", 0), eb)
        vs_hand = vs_r if r.get("sp_hand", "R") == "R" else vs_l
        
        # BvP component with tiny weight unless AB is big
        ab_bvp = r.get("ab") or 0
        h_bvp = r.get("h") or 0
        bvp_rate = eb_rate(h_bvp, ab_bvp, eb)
        
        if ab_bvp < 5:
            w_bvp = 0.0
        elif ab_bvp < 15:
            w_bvp = 0.02
        else:
            w_bvp = 0.10
        
        # Blend: 50/50 form vs vs-hand, small BvP component
        base = 0.5 * form_rate + 0.5 * vs_hand
        return (1 - w_bvp) * base + w_bvp * bvp_rate
    
    hitters["p_hit_pa"] = hitters.apply(compute_p_pa, axis=1)
    hitters["p_ge1"] = hitters.apply(lambda r: p_ge1_hit(r["p_hit_pa"], r["exp_pa"]), axis=1)
    hitters["p_ge2"] = hitters.apply(lambda r: p_ge2_hits(r["p_hit_pa"], r["exp_ab"]), axis=1)
    
    # Sanity checks
    median_p_ge1 = hitters["p_ge1"].median()
    high_prob_count = (hitters["p_ge1"] > 0.85).sum()
    
    if not (0.45 <= median_p_ge1 <= 0.75):
        log.warning(f"Median p_ge1 outside expected range: {median_p_ge1:.3f}")
    
    if high_prob_count > len(hitters) * 0.1:
        log.warning(f"Too many high probabilities: {high_prob_count}/{len(hitters)} > 0.85")
    
    log.info(f"Features built - median p_ge1: {median_p_ge1:.3f}, max: {hitters['p_ge1'].max():.3f}")
    
    return hitters

def predict_and_upsert(engine, target_date: str) -> pd.DataFrame:
    """Generate predictions and save to database"""
    
    log.info(f"Generating hitting props predictions for {target_date}")
    
    # Build features
    feats = build_features(engine, target_date)
    if feats.empty:
        log.warning("No features built")
        return feats
    
    # Get odds
    odds_query = text("""
        SELECT date, game_id, player_id, market, line, over_odds, under_odds, book
        FROM player_props_odds
        WHERE date = :d AND market IN ('HITS_0.5','HITS_1.5')
    """)
    
    odds = pd.read_sql(odds_query, engine, params={"d": target_date})
    
    if odds.empty:
        log.warning("No odds found")
        return pd.DataFrame()
    
    log.info(f"Found {len(odds)} odds records")
    
    # Generate predictions
    rows = []
    for _, o in odds.iterrows():
        # Match player and game
        player_feats = feats[
            (feats.game_id == o.game_id) & 
            (feats.player_id == o.player_id)
        ]
        
        if player_feats.empty:
            continue
            
        r = player_feats.iloc[0]
        
        # Get model probability
        p_over = float(r["p_ge1"] if o.market == "HITS_0.5" else r["p_ge2"])
        p_under = 1 - p_over
        
        # Calculate implied odds and EV
        io = american_to_prob(o.over_odds)
        iu = american_to_prob(o.under_odds)
        
        ev_over = None if io is None else (p_over - io) / io
        ev_under = None if iu is None else (p_under - iu) / iu
        
        # Kelly criterion
        kelly_over = None if o.over_odds is None else kelly(p_over, int(o.over_odds))
        kelly_under = None if o.under_odds is None else kelly(p_under, int(o.under_odds))
        
        rows.append({
            'date': o.date,
            'game_id': o.game_id,
            'player_id': o.player_id,
            'market': o.market,
            'p_over': round(p_over, 4),
            'p_under': round(p_under, 4),
            'ev_over': None if ev_over is None else round(ev_over, 4),
            'ev_under': None if ev_under is None else round(ev_under, 4),
            'kelly_over': None if kelly_over is None else round(kelly_over, 4),
            'kelly_under': None if kelly_under is None else round(kelly_under, 4),
            'features': '{}'  # Empty JSON for now
        })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        log.warning("No predictions generated")
        return df
    
    # Save to database
    with engine.begin() as conn:
        # Create temp table for upsert
        df.to_sql('temp_predictions', conn, if_exists='replace', index=False)
        
        # Upsert
        upsert_sql = text("""
            INSERT INTO hitter_prop_predictions
                (date, game_id, player_id, market, p_over, p_under, 
                 ev_over, ev_under, kelly_over, kelly_under, features)
            SELECT date, game_id, player_id, market, p_over, p_under,
                   ev_over, ev_under, kelly_over, kelly_under, features::jsonb
            FROM temp_predictions
            ON CONFLICT (date, game_id, player_id, market)
            DO UPDATE SET
                p_over = EXCLUDED.p_over,
                p_under = EXCLUDED.p_under,
                ev_over = EXCLUDED.ev_over,
                ev_under = EXCLUDED.ev_under,
                kelly_over = EXCLUDED.kelly_over,
                kelly_under = EXCLUDED.kelly_under,
                features = EXCLUDED.features
        """)
        
        conn.execute(upsert_sql)
        conn.execute(text("DROP TABLE temp_predictions"))
    
    log.info(f"Saved {len(df)} predictions to database")
    return df

def get_recommendations(engine, target_date: str, min_ev: float = 0.03, 
                       max_kelly: float = 0.05) -> dict:
    """Get betting recommendations with filters"""
    
    query = text("""
        SELECT hp.*, po.over_odds, po.under_odds, po.book,
               pgl.player_name, pgl.team,
               CASE 
                   WHEN hp.market = 'HITS_0.5' THEN 'player_name || " 1+ Hit"'
                   WHEN hp.market = 'HITS_1.5' THEN 'player_name || " 2+ Hits"'
                   ELSE market
               END as bet_description
        FROM hitter_prop_predictions hp
        JOIN player_props_odds po USING (date, game_id, player_id, market)
        JOIN (
            SELECT DISTINCT player_id, player_name, team
            FROM player_game_logs 
            WHERE date >= :d::date - INTERVAL '3 days'
        ) pgl USING (player_id)
        WHERE hp.date = :d
          AND (
              (hp.ev_over >= :min_ev AND hp.kelly_over <= :max_kelly) OR
              (hp.ev_under >= :min_ev AND hp.kelly_under <= :max_kelly)
          )
        ORDER BY GREATEST(COALESCE(hp.ev_over, 0), COALESCE(hp.ev_under, 0)) DESC
    """)
    
    recs = pd.read_sql(query, engine, params={
        'd': target_date,
        'min_ev': min_ev,
        'max_kelly': max_kelly
    })
    
    over_bets = []
    under_bets = []
    
    for _, r in recs.iterrows():
        base_info = {
            'player_name': r['player_name'],
            'team': r['team'],
            'market': r['market'],
            'game_id': r['game_id']
        }
        
        # Check OVER bet
        if (r.get('ev_over', 0) >= min_ev and 
            r.get('kelly_over', 1) <= max_kelly):
            over_bets.append({
                **base_info,
                'side': 'OVER',
                'probability': r['p_over'],
                'odds': r['over_odds'],
                'ev': r['ev_over'],
                'kelly': r['kelly_over']
            })
        
        # Check UNDER bet
        if (r.get('ev_under', 0) >= min_ev and 
            r.get('kelly_under', 1) <= max_kelly):
            under_bets.append({
                **base_info,
                'side': 'UNDER',
                'probability': r['p_under'],
                'odds': r['under_odds'],
                'ev': r['ev_under'],
                'kelly': r['kelly_under']
            })
    
    return {
        'date': target_date,
        'over_bets': over_bets[:10],  # Top 10
        'under_bets': under_bets[:5],  # Top 5
        'total_opportunities': len(over_bets) + len(under_bets),
        'summary': {
            'total_players': len(recs['player_id'].unique()),
            'over_count': len(over_bets),
            'under_count': len(under_bets),
            'avg_ev': recs[['ev_over', 'ev_under']].max(axis=1).mean()
        }
    }

def main():
    """CLI interface"""
    import argparse
    from dotenv import load_dotenv
    import os
    
    parser = argparse.ArgumentParser(description='Generate hitting props predictions')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--min-ev', type=float, default=0.03, help='Minimum EV threshold')
    parser.add_argument('--max-kelly', type=float, default=0.05, help='Maximum Kelly size')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load environment
    load_dotenv()
    
    # Connect to database
    engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
    
    # Get target date
    target_date = args.date or datetime.now().strftime('%Y-%m-%d')
    
    print(f"Generating predictions for {target_date}")
    
    # Generate predictions
    predictions = predict_and_upsert(engine, target_date)
    
    if predictions.empty:
        print("No predictions generated")
        return
    
    # Get recommendations
    recs = get_recommendations(engine, target_date, args.min_ev, args.max_kelly)
    
    print(f"\n=== HITTING PROPS RECOMMENDATIONS ===")
    print(f"Date: {target_date}")
    print(f"Players analyzed: {recs['summary']['total_players']}")
    print(f"OVER opportunities: {recs['summary']['over_count']}")
    print(f"UNDER opportunities: {recs['summary']['under_count']}")
    
    if recs['over_bets']:
        print(f"\nTOP OVER BETS:")
        for i, bet in enumerate(recs['over_bets'][:5], 1):
            print(f"{i}. {bet['player_name']} {bet['market']} OVER")
            print(f"   Prob: {bet['probability']:.1%} | EV: {bet['ev']:+.1%} | Kelly: {bet['kelly']:.1%}")
    
    if recs['under_bets']:
        print(f"\nTOP UNDER BETS:")
        for i, bet in enumerate(recs['under_bets'][:3], 1):
            print(f"{i}. {bet['player_name']} {bet['market']} UNDER")
            print(f"   Prob: {bet['probability']:.1%} | EV: {bet['ev']:+.1%} | Kelly: {bet['kelly']:.1%}")

if __name__ == "__main__":
    main()
