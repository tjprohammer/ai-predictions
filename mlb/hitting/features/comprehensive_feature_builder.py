"""
Comprehensive MLB Hitting Props Feature Builder
Enhanced version with full statistical features for hitting props predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging

log = logging.getLogger(__name__)

class ComprehensiveHittingFeatureBuilder:
    
    def __init__(self, database_url: str = None):
        """Initialize feature builder with database connection"""
        if database_url is None:
            database_url = "postgresql://mlbuser:mlbpass@localhost/mlb"
        self.engine = create_engine(database_url)
        
    def build_features_for_date(self, target_date: str) -> pd.DataFrame:
        """Build comprehensive hitting features for all players on target date"""
        
        log.info(f"Building comprehensive hitting features for {target_date}")
        
        # Get today's projected lineups/games
        todays_players = self._get_todays_players(target_date)
        
        if todays_players.empty:
            log.warning(f"No players found for {target_date}")
            return pd.DataFrame()
        
        # Build each feature component
        form_features = self._build_form_features(todays_players, target_date)
        bvp_features = self._build_bvp_features(todays_players, target_date)
        hand_features = self._build_vs_hand_features(todays_players, target_date)
        pa_features = self._build_pa_features(todays_players, target_date)
        momentum_features = self._build_momentum_features(todays_players, target_date)
        
        # Merge all features
        features = todays_players.copy()
        
        for feat_df in [form_features, bvp_features, hand_features, pa_features, momentum_features]:
            if not feat_df.empty:
                features = features.merge(feat_df, on='player_id', how='left')
        
        # Fill missing values and add derived features
        features = self._add_derived_features(features)
        
        log.info(f"Built features for {len(features)} players with {len(features.columns)} features")
        return features
    
    def _get_todays_players(self, target_date: str) -> pd.DataFrame:
        """Get all players expected to play on target date"""
        
        query = text("""
            -- Get players from recent lineups or game logs
            SELECT DISTINCT
                pgl.player_id,
                pgl.player_name,
                pgl.team,
                pgl.lineup_spot,
                sp.pitcher_id as starting_pitcher_id,
                sp.throws as pitcher_hand
            FROM player_game_logs pgl
            LEFT JOIN (
                -- Get likely starting pitchers from recent games
                SELECT DISTINCT ON (team)
                    team,
                    starting_pitcher_id as pitcher_id,
                    p.throws
                FROM player_game_logs
                JOIN pitchers p ON p.pitcher_id = starting_pitcher_id
                WHERE date >= :recent_date
                ORDER BY team, date DESC
            ) sp ON sp.team != pgl.team  -- Opposing pitcher
            WHERE pgl.date >= :recent_date
              AND pgl.lineup_spot BETWEEN 1 AND 9
            ORDER BY pgl.player_id
        """)
        
        recent_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={
                'recent_date': recent_date
            })
    
    def _build_form_features(self, players: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Build recent form features from materialized view"""
        
        if players.empty:
            return pd.DataFrame()
        
        player_ids = tuple(players['player_id'].unique())
        
        query = text("""
            SELECT 
                player_id,
                -- L5 form
                hits_l5, ab_l5, hr_l5, rbi_l5, bb_l5, k_l5, tb_l5, xbh_l5,
                CASE WHEN ab_l5 > 0 THEN hits_l5::FLOAT / ab_l5 ELSE 0 END as avg_l5,
                CASE WHEN ab_l5 > 0 THEN tb_l5::FLOAT / ab_l5 ELSE 0 END as slg_l5,
                CASE WHEN (ab_l5 + bb_l5) > 0 THEN (hits_l5 + bb_l5)::FLOAT / (ab_l5 + bb_l5) ELSE 0 END as obp_l5,
                
                -- L10 form  
                hits_l10, ab_l10, hr_l10, rbi_l10, bb_l10, k_l10, tb_l10, xbh_l10,
                CASE WHEN ab_l10 > 0 THEN hits_l10::FLOAT / ab_l10 ELSE 0 END as avg_l10,
                CASE WHEN ab_l10 > 0 THEN tb_l10::FLOAT / ab_l10 ELSE 0 END as slg_l10,
                CASE WHEN (ab_l10 + bb_l10) > 0 THEN (hits_l10 + bb_l10)::FLOAT / (ab_l10 + bb_l10) ELSE 0 END as obp_l10,
                
                -- L15 form
                hits_l15, ab_l15, hr_l15, rbi_l15, bb_l15, k_l15, tb_l15, xbh_l15,
                CASE WHEN ab_l15 > 0 THEN hits_l15::FLOAT / ab_l15 ELSE 0 END as avg_l15,
                CASE WHEN ab_l15 > 0 THEN tb_l15::FLOAT / ab_l15 ELSE 0 END as slg_l15,
                CASE WHEN (ab_l15 + bb_l15) > 0 THEN (hits_l15 + bb_l15)::FLOAT / (ab_l15 + bb_l15) ELSE 0 END as obp_l15
                
            FROM mv_hitter_form
            WHERE player_id = ANY(:player_ids)
              AND date = (
                  SELECT MAX(date) 
                  FROM mv_hitter_form 
                  WHERE date <= :target_date
                    AND player_id = mv_hitter_form.player_id
              )
        """)
        
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={
                'player_ids': list(player_ids),
                'target_date': target_date
            })
    
    def _build_bvp_features(self, players: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Build Batter vs Pitcher historical features"""
        
        if players.empty:
            return pd.DataFrame()
        
        # Create player-pitcher pairs
        player_pitcher_pairs = []
        for _, player in players.iterrows():
            if pd.notna(player.get('starting_pitcher_id')):
                player_pitcher_pairs.append({
                    'player_id': player['player_id'],
                    'pitcher_id': player['starting_pitcher_id']
                })
        
        if not player_pitcher_pairs:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=[
                'player_id', 'bvp_games', 'bvp_pa', 'bvp_ab', 'bvp_h', 'bvp_hr', 'bvp_rbi',
                'bvp_avg', 'bvp_slg', 'bvp_tb', 'has_bvp_history'
            ])
        
        pairs_df = pd.DataFrame(player_pitcher_pairs)
        
        query = text("""
            SELECT 
                b.player_id,
                COALESCE(bvp.g, 0) as bvp_games,
                COALESCE(bvp.pa, 0) as bvp_pa,
                COALESCE(bvp.ab, 0) as bvp_ab,
                COALESCE(bvp.h, 0) as bvp_h,
                COALESCE(bvp.hr, 0) as bvp_hr,
                COALESCE(bvp.rbi, 0) as bvp_rbi,
                COALESCE(bvp.avg_vs_pitcher, 0) as bvp_avg,
                COALESCE(bvp.slg_vs_pitcher, 0) as bvp_slg,
                COALESCE(bvp.tb, 0) as bvp_tb,
                CASE WHEN bvp.g >= 3 THEN 1 ELSE 0 END as has_bvp_history
            FROM (VALUES {pairs_values}) AS b(player_id, pitcher_id)
            LEFT JOIN mv_bvp_agg bvp 
              ON bvp.player_id = b.player_id 
              AND bvp.pitcher_id = b.pitcher_id
        """)
        
        # Format the VALUES clause
        pairs_values = ', '.join([
            f"({row['player_id']}, {row['pitcher_id']})" 
            for _, row in pairs_df.iterrows()
        ])
        
        final_query = query.text.format(pairs_values=pairs_values)
        
        with self.engine.connect() as conn:
            return pd.read_sql(text(final_query), conn)
    
    def _build_vs_hand_features(self, players: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Build vs-Hand (L/R pitcher) performance features"""
        
        if players.empty:
            return pd.DataFrame()
        
        player_ids = tuple(players['player_id'].unique())
        
        query = text("""
            WITH hand_stats AS (
                SELECT 
                    pgl.player_id,
                    p.throws as pitcher_hand,
                    COUNT(*) as games,
                    SUM(pgl.plate_appearances) as pa,
                    SUM(pgl.at_bats) as ab,
                    SUM(pgl.hits) as h,
                    SUM(pgl.home_runs) as hr,
                    SUM(pgl.runs_batted_in) as rbi,
                    SUM(pgl.walks) as bb,
                    SUM(pgl.strikeouts) as k,
                    SUM(pgl.total_bases) as tb,
                    CASE WHEN SUM(pgl.at_bats) > 0 
                         THEN SUM(pgl.hits)::FLOAT / SUM(pgl.at_bats) 
                         ELSE 0 END as avg,
                    CASE WHEN SUM(pgl.at_bats) > 0 
                         THEN SUM(pgl.total_bases)::FLOAT / SUM(pgl.at_bats) 
                         ELSE 0 END as slg
                FROM player_game_logs pgl
                JOIN pitchers p ON p.pitcher_id = pgl.starting_pitcher_id
                WHERE pgl.player_id = ANY(:player_ids)
                  AND pgl.date >= :lookback_date
                  AND pgl.date < :target_date
                GROUP BY pgl.player_id, p.throws
            )
            SELECT 
                player_id,
                -- vs LHP
                MAX(CASE WHEN pitcher_hand = 'L' THEN games ELSE 0 END) as vs_lhp_games,
                MAX(CASE WHEN pitcher_hand = 'L' THEN ab ELSE 0 END) as vs_lhp_ab,
                MAX(CASE WHEN pitcher_hand = 'L' THEN h ELSE 0 END) as vs_lhp_h,
                MAX(CASE WHEN pitcher_hand = 'L' THEN hr ELSE 0 END) as vs_lhp_hr,
                MAX(CASE WHEN pitcher_hand = 'L' THEN avg ELSE 0 END) as vs_lhp_avg,
                MAX(CASE WHEN pitcher_hand = 'L' THEN slg ELSE 0 END) as vs_lhp_slg,
                
                -- vs RHP
                MAX(CASE WHEN pitcher_hand = 'R' THEN games ELSE 0 END) as vs_rhp_games,
                MAX(CASE WHEN pitcher_hand = 'R' THEN ab ELSE 0 END) as vs_rhp_ab,
                MAX(CASE WHEN pitcher_hand = 'R' THEN h ELSE 0 END) as vs_rhp_h,
                MAX(CASE WHEN pitcher_hand = 'R' THEN hr ELSE 0 END) as vs_rhp_hr,
                MAX(CASE WHEN pitcher_hand = 'R' THEN avg ELSE 0 END) as vs_rhp_avg,
                MAX(CASE WHEN pitcher_hand = 'R' THEN slg ELSE 0 END) as vs_rhp_slg
            FROM hand_stats
            GROUP BY player_id
        """)
        
        # Look back 2 years for vs-hand data
        lookback_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')
        
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={
                'player_ids': list(player_ids),
                'target_date': target_date,
                'lookback_date': lookback_date
            })
    
    def _build_pa_features(self, players: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Build expected plate appearances features"""
        
        if players.empty:
            return pd.DataFrame()
        
        query = text("""
            SELECT 
                p.player_id,
                p.lineup_spot,
                COALESCE(pa_dist.avg_pa, 3.5) as expected_pa,
                COALESCE(pa_dist.std_pa, 1.0) as pa_variance,
                
                -- Lineup spot tiers
                CASE 
                    WHEN p.lineup_spot IN (1,2,3) THEN 'top'
                    WHEN p.lineup_spot IN (4,5,6) THEN 'middle' 
                    WHEN p.lineup_spot IN (7,8,9) THEN 'bottom'
                    ELSE 'unknown'
                END as lineup_tier,
                
                -- PA probability estimates  
                CASE 
                    WHEN p.lineup_spot <= 3 THEN 0.95
                    WHEN p.lineup_spot <= 6 THEN 0.85
                    ELSE 0.75
                END as prob_4plus_pa
                
            FROM (VALUES {player_values}) AS p(player_id, lineup_spot)
            LEFT JOIN mv_pa_distribution pa_dist ON pa_dist.lineup_spot = p.lineup_spot
        """)
        
        # Format player values
        player_values = ', '.join([
            f"({row['player_id']}, {row.get('lineup_spot', 5)})" 
            for _, row in players.iterrows()
        ])
        
        final_query = query.text.format(player_values=player_values)
        
        with self.engine.connect() as conn:
            return pd.read_sql(text(final_query), conn)
    
    def _build_momentum_features(self, players: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Build momentum/hotness indicators"""
        
        if players.empty:
            return pd.DataFrame()
        
        player_ids = tuple(players['player_id'].unique())
        
        query = text("""
            WITH recent_games AS (
                SELECT 
                    player_id,
                    date,
                    hits,
                    at_bats,
                    ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date DESC) as game_rank
                FROM player_game_logs
                WHERE player_id = ANY(:player_ids)
                  AND date < :target_date
                  AND at_bats > 0
            ),
            momentum_calcs AS (
                SELECT 
                    player_id,
                    -- Multi-hit games in last 5
                    SUM(CASE WHEN hits >= 2 AND game_rank <= 5 THEN 1 ELSE 0 END) as multi_hit_l5,
                    
                    -- Hitless games in last 5
                    SUM(CASE WHEN hits = 0 AND game_rank <= 5 THEN 1 ELSE 0 END) as hitless_l5,
                    
                    -- Hit streak (consecutive games with hit)
                    MIN(CASE WHEN hits = 0 THEN game_rank ELSE 999 END) - 1 as hit_streak,
                    
                    -- Recent trend (hits in last 3 vs previous 3)
                    AVG(CASE WHEN game_rank BETWEEN 1 AND 3 THEN hits::FLOAT / NULLIF(at_bats, 0) ELSE NULL END) as avg_last3,
                    AVG(CASE WHEN game_rank BETWEEN 4 AND 6 THEN hits::FLOAT / NULLIF(at_bats, 0) ELSE NULL END) as avg_prev3
                    
                FROM recent_games
                WHERE game_rank <= 10
                GROUP BY player_id
            )
            SELECT 
                player_id,
                COALESCE(multi_hit_l5, 0) as multi_hit_games_l5,
                COALESCE(hitless_l5, 0) as hitless_games_l5,
                CASE WHEN hit_streak = 998 THEN 5 ELSE COALESCE(hit_streak, 0) END as current_hit_streak,
                COALESCE(avg_last3, 0) as trend_recent_avg,
                COALESCE(avg_prev3, 0) as trend_baseline_avg,
                COALESCE(avg_last3 - avg_prev3, 0) as trend_delta,
                
                -- Hotness indicators
                CASE 
                    WHEN multi_hit_l5 >= 3 THEN 'hot'
                    WHEN hitless_l5 >= 3 THEN 'cold'
                    ELSE 'neutral'
                END as hotness_indicator
                
            FROM momentum_calcs
        """)
        
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={
                'player_ids': list(player_ids),
                'target_date': target_date
            })
    
    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add derived features and handle missing values"""
        
        # Fill missing values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        categorical_cols = features.select_dtypes(include=['object']).columns
        features[categorical_cols] = features[categorical_cols].fillna('unknown')
        
        # Derived features
        if 'avg_l5' in features.columns and 'avg_l15' in features.columns:
            features['form_consistency'] = np.abs(features['avg_l5'] - features['avg_l15'])
        
        if 'vs_lhp_avg' in features.columns and 'vs_rhp_avg' in features.columns:
            features['platoon_advantage'] = features['vs_lhp_avg'] - features['vs_rhp_avg']
        
        if 'bvp_avg' in features.columns and 'avg_l15' in features.columns:
            features['bvp_vs_season'] = features['bvp_avg'] - features['avg_l15']
        
        # Interaction features
        if 'expected_pa' in features.columns and 'avg_l10' in features.columns:
            features['expected_hits'] = features['expected_pa'] * features['avg_l10']
        
        return features


def main():
    """Test the comprehensive feature builder"""
    
    import sys
    from datetime import datetime
    
    logging.basicConfig(level=logging.INFO)
    
    # Use command line date or default to today
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    
    builder = ComprehensiveHittingFeatureBuilder()
    features = builder.build_features_for_date(target_date)
    
    if not features.empty:
        print(f"\nBuilt comprehensive features for {len(features)} players:")
        print(f"Feature columns: {len(features.columns)}")
        print(f"Columns: {list(features.columns)}")
        print("\nSample features:")
        print(features.head())
        
        # Show feature summary
        print(f"\nFeature summary:")
        print(features.describe())
    else:
        print(f"No features built for {target_date}")


if __name__ == "__main__":
    main()
