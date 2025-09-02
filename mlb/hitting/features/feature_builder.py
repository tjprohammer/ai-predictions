"""
MLB Hitting Features Builder

This module builds player hitting features for prop betting predictions including:
- Rolling form (5/10/15 game windows)
- Batter vs Pitcher matchups 
- Handedness splits
- Expected plate appearances
- Empirical Bayes blended hit rates
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sqlalchemy import text
from typing import Optional
import logging

log = logging.getLogger(__name__)

@dataclass
class EBParams:
    """Empirical Bayes parameters for shrinkage towards league average"""
    alpha: float = 4.0   # prior "hits" 
    beta: float = 16.0   # prior "non-hits" (tunes shrinkage)

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability"""
    if pd.isna(odds) or odds == 0:
        return 0.5
    return (100 / (odds + 100)) if odds > 0 else ((-odds) / ((-odds) + 100))

def kelly(p: float, odds: int) -> float:
    """Calculate Kelly criterion bet size"""
    if pd.isna(odds) or odds == 0 or p <= 0:
        return 0.0
    b = (odds / 100) if odds > 0 else (100 / (-odds))
    kelly_size = max(0.0, (p*(b+1) - 1) / b)
    return min(kelly_size, 0.25)  # Cap at 25% of bankroll

def eb_rate(success: float, trials: float, prior: EBParams) -> float:
    """Calculate Empirical Bayes adjusted rate"""
    if pd.isna(success) or pd.isna(trials) or trials <= 0:
        return prior.alpha / (prior.alpha + prior.beta)
    return (success + prior.alpha) / (trials + prior.alpha + prior.beta)

def estimate_pas(lineup_spot: Optional[float]) -> float:
    """Estimate expected plate appearances based on lineup position"""
    if pd.isna(lineup_spot) or lineup_spot is None:
        return 4.0  # Default for unknown
    return 4.2 if 1 <= lineup_spot <= 5 else 3.7

def get_pitcher_handedness(engine, sp_id: int, default: str = 'R') -> str:
    """Get pitcher handedness from database"""
    try:
        query = text("""
            SELECT DISTINCT starting_pitcher_hand 
            FROM player_game_logs 
            WHERE starting_pitcher_id = :sp_id 
              AND starting_pitcher_hand IS NOT NULL
            LIMIT 1
        """)
        result = pd.read_sql(query, engine, params={'sp_id': sp_id})
        return result.iloc[0]['starting_pitcher_hand'] if not result.empty else default
    except:
        return default

def build_today_player_features(engine, target_date: str) -> pd.DataFrame:
    """
    Build comprehensive hitting features for all players expected to play today
    
    Returns DataFrame with columns:
    - Basic info: game_id, date, player_id, player_name, team, opponent
    - Form: hits_l5, ab_l5, hits_l10, ab_l10, hits_l15, ab_l15
    - Splits: h_vs_r, ab_vs_r, h_vs_l, ab_vs_l  
    - BvP: bvp_ab, bvp_h (if available)
    - Predictions: exp_pa, p_hit_per_pa, p_ge1_hit
    """
    
    log.info(f"Building hitting features for {target_date}")
    
    # 1) Get today's slate of hitters from enhanced_games and recent player logs
    hitters_query = text("""
        WITH todays_games AS (
            SELECT DISTINCT eg.game_id, eg.date, eg.home_team, eg.away_team,
                   eg.home_sp_id, eg.away_sp_id,
                   eg.home_sp_name, eg.away_sp_name
            FROM enhanced_games eg
            WHERE eg.date = :target_date
        ),
        recent_players AS (
            SELECT DISTINCT pgl.player_id, pgl.player_name, pgl.team,
                   ROW_NUMBER() OVER (PARTITION BY pgl.player_id ORDER BY pgl.date DESC) as rn
            FROM player_game_logs pgl
            WHERE pgl.date >= :target_date::date - INTERVAL '7 days'
              AND pgl.at_bats > 0
        )
        SELECT DISTINCT 
            tg.game_id, 
            tg.date,
            rp.player_id,
            rp.player_name,
            rp.team,
            CASE 
                WHEN rp.team = tg.home_team THEN tg.away_team
                WHEN rp.team = tg.away_team THEN tg.home_team
                ELSE NULL
            END as opponent,
            CASE 
                WHEN rp.team = tg.home_team THEN 'H'
                WHEN rp.team = tg.away_team THEN 'A' 
                ELSE NULL
            END as home_away,
            CASE 
                WHEN rp.team = tg.home_team THEN tg.away_sp_id
                WHEN rp.team = tg.away_team THEN tg.home_sp_id
                ELSE NULL  
            END as opposing_sp_id,
            CASE 
                WHEN rp.team = tg.home_team THEN tg.away_sp_name
                WHEN rp.team = tg.away_team THEN tg.home_sp_name
                ELSE NULL
            END as opposing_sp_name
        FROM todays_games tg
        CROSS JOIN recent_players rp
        WHERE rp.rn = 1  -- Most recent appearance only
          AND (rp.team = tg.home_team OR rp.team = tg.away_team)
        ORDER BY tg.game_id, rp.player_name
    """)
    
    hitters = pd.read_sql(hitters_query, engine, params={'target_date': target_date})
    
    if hitters.empty:
        log.warning(f"No hitters found for {target_date}")
        return hitters
        
    log.info(f"Found {len(hitters)} potential hitters for {target_date}")

    # 2) Get latest known lineup positions  
    lineup_query = text("""
        SELECT DISTINCT ON (player_id) 
            player_id, 
            lineup_spot, 
            date
        FROM player_game_logs
        WHERE player_id = ANY(:player_ids)
          AND lineup_spot IS NOT NULL
        ORDER BY player_id, date DESC
    """)
    
    player_ids = list(hitters.player_id.unique())
    latest_lineup = pd.read_sql(lineup_query, engine, params={'player_ids': player_ids})
    hitters = hitters.merge(latest_lineup[['player_id', 'lineup_spot']], on='player_id', how='left')
    hitters['exp_pa'] = hitters['lineup_spot'].apply(estimate_pas)

    # 3) Get rolling form from materialized view
    form_query = text("""
        SELECT player_id, hits_l5, ab_l5, hits_l10, ab_l10, hits_l15, ab_l15,
               hits_season, ab_season, pa_l5, pa_l10, pa_l15, pa_season
        FROM mv_hitter_form
        WHERE date <= :target_date
          AND player_id = ANY(:player_ids)
          AND date = (
              SELECT MAX(date) 
              FROM mv_hitter_form mhf2 
              WHERE mhf2.player_id = mv_hitter_form.player_id 
                AND mhf2.date <= :target_date
          )
    """)
    
    form = pd.read_sql(form_query, engine, params={'target_date': target_date, 'player_ids': player_ids})
    hitters = hitters.merge(form, on='player_id', how='left')

    # 4) Get handedness splits (season-to-date)
    splits_query = text("""
        SELECT player_id,
               SUM(CASE WHEN starting_pitcher_hand='R' THEN hits ELSE 0 END) AS h_vs_r,
               SUM(CASE WHEN starting_pitcher_hand='R' THEN at_bats ELSE 0 END) AS ab_vs_r,
               SUM(CASE WHEN starting_pitcher_hand='L' THEN hits ELSE 0 END) AS h_vs_l,
               SUM(CASE WHEN starting_pitcher_hand='L' THEN at_bats ELSE 0 END) AS ab_vs_l
        FROM player_game_logs
        WHERE player_id = ANY(:player_ids) 
          AND date <= :target_date
          AND starting_pitcher_hand IS NOT NULL
        GROUP BY player_id
    """)
    
    splits = pd.read_sql(splits_query, engine, params={'player_ids': player_ids, 'target_date': target_date})
    hitters = hitters.merge(splits, on='player_id', how='left')

    # 5) Get Batter vs Pitcher data
    opposing_pitcher_ids = list(hitters.dropna(subset=['opposing_sp_id']).opposing_sp_id.astype(int).unique())
    
    if opposing_pitcher_ids:
        bvp_query = text("""
            SELECT player_id, pitcher_id, ab, h, g
            FROM mv_bvp_agg
            WHERE player_id = ANY(:player_ids) 
              AND pitcher_id = ANY(:pitcher_ids)
        """)
        
        bvp = pd.read_sql(bvp_query, engine, params={
            'player_ids': player_ids,
            'pitcher_ids': opposing_pitcher_ids
        })
        
        hitters = hitters.merge(
            bvp, 
            left_on=['player_id', 'opposing_sp_id'], 
            right_on=['player_id', 'pitcher_id'], 
            how='left'
        )
    else:
        hitters['ab'] = np.nan
        hitters['h'] = np.nan
        hitters['g'] = np.nan

    # 6) Build EB-blended per-PA hit rate
    prior = EBParams()
    
    def calculate_hit_rate(row):
        # Get best available recent form (prefer 10 games, fallback to 5 or 15)
        if pd.notna(row.get('hits_l10')) and pd.notna(row.get('ab_l10')) and row.get('ab_l10', 0) >= 20:
            form_hits, form_abs = row['hits_l10'], row['ab_l10']
        elif pd.notna(row.get('hits_l5')) and pd.notna(row.get('ab_l5')) and row.get('ab_l5', 0) >= 10:
            form_hits, form_abs = row['hits_l5'], row['ab_l5']
        elif pd.notna(row.get('hits_l15')) and pd.notna(row.get('ab_l15')):
            form_hits, form_abs = row['hits_l15'], row['ab_l15']
        else:
            form_hits, form_abs = 0, 0
            
        form_rate = eb_rate(form_hits, form_abs, prior)
        
        # Handedness split (get pitcher handedness if available)
        pitcher_hand = 'R'  # Default assumption
        if pd.notna(row.get('opposing_sp_id')):
            pitcher_hand = get_pitcher_handedness(engine, int(row['opposing_sp_id']), 'R')
            
        vs_r_hits = row.get('h_vs_r', 0) or 0
        vs_r_abs = row.get('ab_vs_r', 0) or 0
        vs_l_hits = row.get('h_vs_l', 0) or 0  
        vs_l_abs = row.get('ab_vs_l', 0) or 0
        
        vs_r_rate = eb_rate(vs_r_hits, vs_r_abs, prior)
        vs_l_rate = eb_rate(vs_l_hits, vs_l_abs, prior)
        vs_hand_rate = vs_r_rate if pitcher_hand == 'R' else vs_l_rate
        
        # BvP component (light weight if decent sample)
        bvp_abs = row.get('ab', 0) or 0
        bvp_hits = row.get('h', 0) or 0
        bvp_rate = eb_rate(bvp_hits, bvp_abs, prior)
        
        # Weight BvP based on sample size
        if bvp_abs >= 15:
            w_bvp = 0.1
        elif bvp_abs >= 5:
            w_bvp = 0.05
        else:
            w_bvp = 0.0
            
        # Blend: mostly form + vs-hand, small BvP component
        base_rate = 0.6 * form_rate + 0.4 * vs_hand_rate
        final_rate = (1 - w_bvp) * base_rate + w_bvp * bvp_rate
        
        return max(0.05, min(0.50, final_rate))  # Reasonable bounds
    
    hitters['p_hit_per_pa'] = hitters.apply(calculate_hit_rate, axis=1)
    
    # 7) Convert per-PA hit rate to "≥ 1 hit" probability using binomial
    def prob_at_least_one_hit(row):
        p_hit = row['p_hit_per_pa']
        exp_pa = row['exp_pa']
        
        # Probability of 0 hits in exp_pa plate appearances
        p_no_hits = (1 - p_hit) ** exp_pa
        
        # Probability of at least 1 hit
        p_ge1 = 1 - p_no_hits
        
        return max(0.01, min(0.99, p_ge1))
    
    hitters['p_ge1_hit'] = hitters.apply(prob_at_least_one_hit, axis=1)
    
    # 8) Add some derived metrics for analysis
    hitters['hit_pct_l5'] = np.where(
        hitters['ab_l5'] > 0, 
        (hitters['hits_l5'] / hitters['ab_l5']) * 100, 
        np.nan
    )
    hitters['hit_pct_l10'] = np.where(
        hitters['ab_l10'] > 0, 
        (hitters['hits_l10'] / hitters['ab_l10']) * 100, 
        np.nan
    )
    
    # Fill NaN values with reasonable defaults
    numeric_cols = ['hits_l5', 'ab_l5', 'hits_l10', 'ab_l10', 'hits_l15', 'ab_l15',
                   'h_vs_r', 'ab_vs_r', 'h_vs_l', 'ab_vs_l', 'ab', 'h', 'g']
    for col in numeric_cols:
        if col in hitters.columns:
            hitters[col] = hitters[col].fillna(0)
    
    # Select final columns
    final_cols = [
        'game_id', 'date', 'player_id', 'player_name', 'team', 'opponent', 'home_away',
        'opposing_sp_id', 'opposing_sp_name', 'lineup_spot', 'exp_pa',
        'p_hit_per_pa', 'p_ge1_hit',
        'hits_l5', 'ab_l5', 'hits_l10', 'ab_l10', 'hits_l15', 'ab_l15',
        'h_vs_r', 'ab_vs_r', 'h_vs_l', 'ab_vs_l', 
        'ab', 'h', 'g',  # BvP stats
        'hit_pct_l5', 'hit_pct_l10'
    ]
    
    result = hitters[[col for col in final_cols if col in hitters.columns]].copy()
    
    log.info(f"Built features for {len(result)} players")
    log.info(f"Average p(≥1 hit): {result['p_ge1_hit'].mean():.3f}")
    
    return result

if __name__ == "__main__":
    # Test the feature builder
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from sqlalchemy import create_engine
    import os
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv(Path(__file__).parent.parent / '.env')
    
    # Connect to database
    engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
    
    # Test with today's date
    from datetime import datetime
    target_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Testing feature builder for {target_date}")
    df = build_today_player_features(engine, target_date)
    
    if not df.empty:
        print(f"\nBuilt features for {len(df)} players")
        print("\nSample features:")
        print(df[['player_name', 'team', 'p_ge1_hit', 'hit_pct_l10', 'exp_pa']].head())
    else:
        print("No features built - check if games and player data exist for today")
