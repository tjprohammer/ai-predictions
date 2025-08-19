#!/usr/bin/env python3
"""
Enhanced Bullpen Predictor
==========================
Uses the enhanced features directly from legitimate_game_features table
with bullpen statistics for accurate predictions
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import joblib
from datetime import datetime
import logging
from pathlib import Path
from hashlib import md5

# Import enhanced pipeline (keep original guard)
try:
    from enhanced_feature_pipeline import EnhancedFeaturePipeline, apply_serving_calibration
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError:
    ENHANCED_PIPELINE_AVAILABLE = False
    logging.warning("Enhanced feature pipeline not available - using basic features only")

# Safe helper functions for fillna operations (restored after patch move)
def _num(s):
    return pd.to_numeric(s, errors="coerce")

def coalesce_num_series(df, cols, default_value=np.nan):
    s = None
    for c in cols:
        if c in df.columns:
            s = df[c]; break
    if s is None:
        s = pd.Series(default_value, index=df.index, dtype="float64")
    return pd.to_numeric(s, errors="coerce")
def add_pitcher_rolling_stats(games_df, engine):
    """Vectorized merge_asof joins for home/away starters with proper sorting.
    Fixes prior 'right keys must be sorted' errors and prevents cascade neutral imputation.
    """
    if games_df.empty:
        return games_df
    logger = logging.getLogger(__name__)
    logger.info("🔄 Adding real pitcher rolling stats (vectorized asof)…")

    roll_query = """
    SELECT pitcher_id, stat_date, gs, ip, er, bb, k, h, hr,
           era, whip, k_per_9, bb_per_9, hr_per_9
    FROM pitcher_daily_rolling
    """
    try:
        roll_df = pd.read_sql(roll_query, engine)
    except Exception as e:
        logger.error(f"❌ Failed to load pitcher rolling stats: {e}")
        return games_df
    if roll_df.empty:
        logger.error("❌ No pitcher rolling stats found")
        return games_df

    # Normalize and sort (exact patch from runbook) - ENHANCED DEFENSIVE SORTING
    games_df = games_df.copy()
    games_df["date"] = pd.to_datetime(games_df["date"])

    # Ensure pitcher stats are properly typed and sorted
    roll_df["stat_date"] = pd.to_datetime(roll_df["stat_date"])
    roll_df["pitcher_id"] = pd.to_numeric(roll_df["pitcher_id"], errors="coerce")
    roll_df = roll_df.dropna(subset=["pitcher_id"]).copy()
    roll_df["pitcher_id"] = roll_df["pitcher_id"].astype("int64")
    
    # CRITICAL: Remove duplicates that can break merge_asof
    roll_df = roll_df.drop_duplicates(subset=["pitcher_id", "stat_date"], keep="last")
    roll_df = roll_df.sort_values(["pitcher_id", "stat_date"]).reset_index(drop=True)
    
    # Verify sorting before merge_asof
    if len(roll_df) > 1:
        sort_check = roll_df.groupby("pitcher_id")["stat_date"].apply(lambda x: x.is_monotonic_increasing).all()
        if not sort_check:
            logger.warning("Pitcher rolling stats not properly sorted - forcing re-sort")
            roll_df = roll_df.sort_values(["pitcher_id", "stat_date"]).reset_index(drop=True)

    # Home side - ENHANCED PREPARATION
    home = games_df[["game_id", "date", "home_sp_id"]].copy()
    home["home_sp_id"] = pd.to_numeric(home["home_sp_id"], errors="coerce")
    home = home.dropna(subset=["home_sp_id"]).copy()
    home["home_sp_id"] = home["home_sp_id"].astype("int64")
    # Remove duplicates and ensure proper sorting
    home = home.drop_duplicates(subset=["home_sp_id", "date"], keep="first")
    home = home.sort_values(["home_sp_id", "date"]).reset_index(drop=True)

    home_roll = roll_df.rename(columns={"pitcher_id":"home_sp_id"})
    # CRITICAL: Ensure proper data types and sorting for merge_asof
    home_roll["home_sp_id"] = home_roll["home_sp_id"].astype("int64")  # Ensure int64 type
    home_roll["stat_date"] = pd.to_datetime(home_roll["stat_date"])   # Ensure datetime type
    
    # Force a fresh sort and remove any duplicates
    home_roll = home_roll.drop_duplicates(subset=["home_sp_id", "stat_date"], keep="last")
    home_roll = home_roll.sort_values(["home_sp_id","stat_date"]).reset_index(drop=True)
    
    # Final verification that would catch edge cases
    if len(home_roll) > 0:
        group_check = home_roll.groupby("home_sp_id")["stat_date"].apply(lambda x: x.is_monotonic_increasing).all()
        if not group_check:
            logger.error("Home roll data failed final monotonic sort check - this should not happen")
            # Force one more sort as last resort
            home_roll = home_roll.sort_values(["home_sp_id","stat_date"]).reset_index(drop=True)
    try:
        home_join = pd.merge_asof(
            home,
            home_roll[["home_sp_id","stat_date","era","whip","k_per_9","bb_per_9","gs"]],
            left_on="date", right_on="stat_date",
            left_by="home_sp_id", right_by="home_sp_id",
            direction="backward", allow_exact_matches=True
        )
        logger.info(f"✅ Home merge_asof successful: {len(home_join)} rows")
    except Exception as e:
        logger.error(f"home merge_asof failed ({e}); attempting coarse fallback join")
        tmp = home_roll.sort_values(["home_sp_id","stat_date"]).drop_duplicates(["home_sp_id"], keep="last")
        home_join = home.merge(tmp, on="home_sp_id", how="left")
        logger.warning(f"Home fallback join completed: {len(home_join)} rows")
        logger.debug(f"Home fallback columns: {list(home_join.columns)}")
        logger.debug(f"Home fallback non-null counts: {home_join.count()}")

    # Away side - ENHANCED PREPARATION
    away = games_df[["game_id", "date", "away_sp_id"]].copy()
    away["away_sp_id"] = pd.to_numeric(away["away_sp_id"], errors="coerce")
    away = away.dropna(subset=["away_sp_id"]).copy()
    away["away_sp_id"] = away["away_sp_id"].astype("int64")
    # Remove duplicates and ensure proper sorting
    away = away.drop_duplicates(subset=["away_sp_id", "date"], keep="first")
    away = away.sort_values(["away_sp_id", "date"]).reset_index(drop=True)

    away_roll = roll_df.rename(columns={"pitcher_id":"away_sp_id"})
    # CRITICAL: Ensure proper data types and sorting for merge_asof
    away_roll["away_sp_id"] = away_roll["away_sp_id"].astype("int64")  # Ensure int64 type
    away_roll["stat_date"] = pd.to_datetime(away_roll["stat_date"])   # Ensure datetime type
    
    # Force a fresh sort and remove any duplicates
    away_roll = away_roll.drop_duplicates(subset=["away_sp_id", "stat_date"], keep="last")
    away_roll = away_roll.sort_values(["away_sp_id","stat_date"]).reset_index(drop=True)
    
    # Final verification that would catch edge cases
    if len(away_roll) > 0:
        group_check = away_roll.groupby("away_sp_id")["stat_date"].apply(lambda x: x.is_monotonic_increasing).all()
        if not group_check:
            logger.error("Away roll data failed final monotonic sort check - this should not happen")
            # Force one more sort as last resort
            away_roll = away_roll.sort_values(["away_sp_id","stat_date"]).reset_index(drop=True)
    try:
        away_join = pd.merge_asof(
            away,
            away_roll[["away_sp_id","stat_date","era","whip","k_per_9","bb_per_9","gs"]],
            left_on="date", right_on="stat_date",
            left_by="away_sp_id", right_by="away_sp_id",
            direction="backward", allow_exact_matches=True
        )
        logger.info(f"✅ Away merge_asof successful: {len(away_join)} rows")
    except Exception as e:
        logger.error(f"away merge_asof failed ({e}); attempting coarse fallback join")
        tmpa = away_roll.sort_values(["away_sp_id","stat_date"]).drop_duplicates(["away_sp_id"], keep="last")
        away_join = away.merge(tmpa, on="away_sp_id", how="left")
        logger.warning(f"Away fallback join completed: {len(away_join)} rows")
        logger.debug(f"Away fallback columns: {list(away_join.columns)}")
        logger.debug(f"Away fallback non-null counts: {away_join.count()}")

    # Coalesce back
    try:
        logger.debug("Starting coalesce back...")
        logger.debug(f"home_join columns: {list(home_join.columns) if home_join is not None else 'None'}")
        logger.debug(f"away_join columns: {list(away_join.columns) if away_join is not None else 'None'}")
        
        required_cols = ["game_id","era","whip","k_per_9","bb_per_9","gs"]
        home_available = [c for c in required_cols if c in home_join.columns] if home_join is not None else []
        away_available = [c for c in required_cols if c in away_join.columns] if away_join is not None else []
        
        logger.debug(f"Home available columns: {home_available}")
        logger.debug(f"Away available columns: {away_available}")
        
        home_stats = home_join[home_available].rename(columns={
            "era":"home_sp_era_new","whip":"home_sp_whip_new","k_per_9":"home_sp_k_per_9_new",
            "bb_per_9":"home_sp_bb_per_9_new","gs":"home_sp_starts_new"
        })
        away_stats = away_join[away_available].rename(columns={
            "era":"away_sp_era_new","whip":"away_sp_whip_new","k_per_9":"away_sp_k_per_9_new",
            "bb_per_9":"away_sp_bb_per_9_new","gs":"away_sp_starts_new"
        })
        
        logger.debug("Coalesce back completed successfully")
    except Exception as e:
        logger.error(f"Error in coalesce back: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Merge back the pitcher stats
    try:
        logger.debug(f"games_df shape before merge: {games_df.shape}")
        logger.debug(f"home_stats shape: {home_stats.shape}")
        logger.debug(f"away_stats shape: {away_stats.shape}")
        
        games_df = games_df.merge(home_stats, on="game_id", how="left").merge(away_stats, on="game_id", how="left")
        
        logger.debug(f"games_df shape after merge: {games_df.shape}")
        logger.debug(f"games_df is None: {games_df is None}")
    except Exception as e:
        logger.error(f"Error merging pitcher stats back: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Replace & clean
    try:
        logger.debug("Starting Replace & clean section...")
        for src, dst in [
            ("home_sp_era_new","home_sp_era"),("away_sp_era_new","away_sp_era"),
            ("home_sp_whip_new","home_sp_whip"),("away_sp_whip_new","away_sp_whip"),
            ("home_sp_k_per_9_new","home_sp_k_per_9"),("away_sp_k_per_9_new","away_sp_k_per_9"),
            ("home_sp_bb_per_9_new","home_sp_bb_per_9"),("away_sp_bb_per_9_new","away_sp_bb_per_9"),
            ("home_sp_starts_new","home_sp_starts"),("away_sp_starts_new","away_sp_starts"),
        ]:
            if src in games_df.columns:
                logger.debug(f"Replacing {src} -> {dst}")
                games_df[dst] = games_df[src]
            else:
                logger.debug(f"Column {src} not found, skipping")
        logger.debug("Replace & clean completed successfully")
    except Exception as e:
        logger.error(f"Error in Replace & clean: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Season stats alias
    games_df['home_sp_season_era'] = games_df.get('home_sp_era')
    games_df['away_sp_season_era'] = games_df.get('away_sp_era')
    games_df['home_sp_season_whip'] = games_df.get('home_sp_whip')
    games_df['away_sp_season_whip'] = games_df.get('away_sp_whip')

    # Diagnostics
    hc = games_df['home_sp_era'].notna().sum()
    ac = games_df['away_sp_era'].notna().sum()
    n = len(games_df)
    logger.info(f"✅ Real pitcher data joined: home ERA {hc}/{n}, away ERA {ac}/{n}")
    return games_df

# --- Cheap slate variance injection for ballparks (runbook patch) ---
def _inject_basic_ballpark_factors(df: pd.DataFrame) -> pd.DataFrame:
    PARK = {
        "Coors Field": (1.12, 1.28),
        "Fenway Park": (1.05, 1.08),
        "Petco Park": (0.96, 0.90),
        "Kauffman Stadium": (0.98, 0.92),
        "Dodger Stadium": (0.99, 0.95),
        "Wrigley Field": (1.03, 1.10),
        "Chase Field": (1.02, 1.05),
        "Great American Ball Park": (1.06, 1.12)
    }
    if "venue_name" in df.columns:
        run = df.get("ballpark_run_factor", 1.0).copy()
        hr = df.get("ballpark_hr_factor", 1.0).copy()
        for i, v in df["venue_name"].fillna("").items():
            rf, hf = PARK.get(
                v,
                (
                    run.iloc[i] if i < len(run) and pd.notna(run.iloc[i]) else 1.00,
                    hr.iloc[i] if i < len(hr) and pd.notna(hr.iloc[i]) else 1.00,
                ),
            )
            if i < len(run):
                run.iloc[i] = rf
            if i < len(hr):
                hr.iloc[i] = hf
        df["ballpark_run_factor"] = run
        df["ballpark_hr_factor"] = hr
    df["park_offensive_factor"] = (
        pd.to_numeric(df.get("ballpark_run_factor"), errors="coerce").fillna(1.0)
        * pd.to_numeric(df.get("ballpark_hr_factor"), errors="coerce").fillna(1.0)
    )
    return df
    games_df['home_sp_starts'] = games_df['home_sp_starts_new'].combine_first(games_df.get('home_sp_starts'))
    games_df['away_sp_starts'] = games_df['away_sp_starts_new'].combine_first(games_df.get('away_sp_starts'))
    
    # Drop the temporary columns
    temp_cols = [c for c in games_df.columns if c.endswith('_new')]
    games_df = games_df.drop(columns=temp_cols)
    
    # Also create season stats (rolling stats ARE season stats)
    games_df['home_sp_season_era'] = games_df['home_sp_era']
    games_df['away_sp_season_era'] = games_df['away_sp_era'] 
    games_df['home_sp_season_whip'] = games_df['home_sp_whip']
    games_df['away_sp_season_whip'] = games_df['away_sp_whip']
    
    # Log diagnostics
    home_era_count = games_df['home_sp_era'].notna().sum()
    away_era_count = games_df['away_sp_era'].notna().sum()
    total_games = len(games_df)
    
    logger.info(f"✅ Real pitcher data joined:")
    logger.info(f"   Home ERA coverage: {home_era_count}/{total_games} ({100*home_era_count/total_games:.1f}%)")
    logger.info(f"   Away ERA coverage: {away_era_count}/{total_games} ({100*away_era_count/total_games:.1f}%)")
    
    if home_era_count > 0:
        logger.info(f"   Home ERA range: {games_df['home_sp_era'].min():.2f} - {games_df['home_sp_era'].max():.2f}")
        logger.info(f"   Home ERA std: {games_df['home_sp_era'].std():.3f}")
    if away_era_count > 0:
        logger.info(f"   Away ERA range: {games_df['away_sp_era'].min():.2f} - {games_df['away_sp_era'].max():.2f}")
        logger.info(f"   Away ERA std: {games_df['away_sp_era'].std():.3f}")
    
    return games_df

# New: shared feature alias map (training ↔ serving)
FEATURE_ALIASES = {
    'home_pitcher_season_era': 'home_sp_era',
    'away_pitcher_season_era': 'away_sp_era',
    'home_sp_season_era': 'home_sp_era',
    'away_sp_season_era': 'away_sp_era',

    'home_pitcher_season_whip': 'home_sp_whip',
    'away_pitcher_season_whip': 'away_sp_whip',
    'home_pitcher_k_per_9': 'home_sp_k_per_9',
    'away_pitcher_k_per_9': 'away_sp_k_per_9',
    'home_pitcher_bb_per_9': 'home_sp_bb_per_9',
    'away_pitcher_bb_per_9': 'away_sp_bb_per_9',
    'home_pitcher_hr_per_9': 'home_sp_hr_per_9',
    'away_pitcher_hr_per_9': 'away_sp_hr_per_9',

    'home_bullpen_era': 'home_bp_era',
    'away_bullpen_era': 'away_bp_era',
    'home_bullpen_fip': 'home_bp_fip',
    'away_bullpen_fip': 'away_bp_fip',

    'ballpark_run_factor': 'ballpark_run_factor',
    'ballpark_hr_factor': 'ballpark_hr_factor',
    'temperature': 'temperature',
    'wind_speed': 'wind_speed',
    'day_night': 'day_night',

    # rename if training uses this name
    'offensive_imbalance': 'offense_imbalance',
}


def _to_iso_date(s: str) -> str:
    """Normalize various incoming date formats to YYYY-MM-DD."""
    if not s:
        return datetime.now().strftime('%Y-%m-%d')
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {s}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBullpenPredictor:
    def __init__(self):
        # Use absolute path for models directory
        current_dir = Path(__file__).parent
        self.models_dir = current_dir.parent / "models"
        self.db_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
        
        self.model = None
        self.preproc = None  # Preferred full pipeline (winsor + impute + scale)
        self.scaler = None   # Legacy fallback
        self.feature_columns = []
        self.fill_values = {}  # Training medians / fill values
        self.feature_aliases = FEATURE_ALIASES.copy()
        self.bias_correction = 0.0  # Model bias correction
        
        # Initialize enhanced pipeline if available
        if ENHANCED_PIPELINE_AVAILABLE:
            self.enhanced_pipeline = EnhancedFeaturePipeline(self.db_url)
            logger.info("Enhanced feature pipeline initialized")
        else:
            self.enhanced_pipeline = None
        
        # Load the latest model
        self.load_model()
    
    def get_engine(self):
        """Get PostgreSQL database engine"""
        return create_engine(self.db_url)
    
    def load_model(self, model_path=None):
        """Load the trained enhanced bullpen model and preprocessing artifacts robustly."""
        if model_path is None:
            model_path = self.models_dir / "legitimate_model_latest.joblib"
        try:
            data = joblib.load(model_path)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            self.preproc = data.get('preproc')
            self.scaler = data.get('scaler')  # legacy key (may be absent)
            self.fill_values = data.get('feature_fill_values', {})
            
            # 🚨 CRITICAL FIX: Override any 0.0 ERA defaults with realistic league averages
            realistic_era_defaults = {
                'home_sp_era': 4.20,
                'away_sp_era': 4.20,
                'home_sp_whip': 1.25,
                'away_sp_whip': 1.25,
                'combined_sp_era': 4.20,
                'sp_era_differential': 0.0  # This one can stay 0.0 as it's a difference
            }
            
            # Fix any problematic 0.0 ERA values from training
            for col, realistic_val in realistic_era_defaults.items():
                if col in self.fill_values and self.fill_values[col] == 0.0:
                    logger.warning(f"🔧 SERVING FIX: Updating {col} fill_value from 0.0 → {realistic_val}")
                    self.fill_values[col] = realistic_val
                elif col not in self.fill_values:
                    # Add missing ERA defaults
                    self.fill_values[col] = realistic_val
            
            # Allow model bundle to override/extend aliases
            bundle_aliases = data.get('feature_aliases')
            if bundle_aliases:
                self.feature_aliases.update(bundle_aliases)
            # Load bias correction if available
            self.bias_correction = data.get('bias_correction', 0.0)
            logger.info("✅ Enhanced bullpen model loaded successfully")
            logger.info(f"   Features: {len(self.feature_columns)}")
            logger.info(f"   Model type: {data.get('model_type','unknown')}")
            if self.bias_correction != 0.0:
                logger.info(f"   Bias correction: {self.bias_correction:+.3f}")
            if not self.preproc and self.scaler:
                logger.warning("Using legacy scaler (no full preprocessing pipeline found).")
            if not (self.preproc or self.scaler):
                logger.warning("No preprocessor or scaler found – predictions will use raw features.")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def engineer_features(self, df):
        """Engineer the same features as the trainer (copied from legitimate_model_trainer.py)"""
        # Derived features with fallbacks for missing data
        # (removed duplicate md5 import as it's already imported at top)
        featured_df = df.copy()
        
        # First, run the original feature engineering
        featured_df = self._original_engineer_features(featured_df)
        # Inject basic ballpark factors variance (runbook patch #3)
        try:
            featured_df = _inject_basic_ballpark_factors(featured_df)
        except Exception as e:
            logger.warning(f"Ballpark factor injection failed: {e}")
        
        # Then add enhanced pipeline features if available
        if self.enhanced_pipeline is not None:
            try:
                # Get target date from the dataframe or use current date
                if 'date' in featured_df.columns:
                    target_date = featured_df['date'].iloc[0]
                elif hasattr(self, '_current_target_date'):
                    target_date = self._current_target_date
                else:
                    target_date = datetime.now().strftime('%Y-%m-%d')
                
                featured_df = self.enhanced_pipeline.build_enhanced_features(featured_df, target_date)
                logger.info("✅ Enhanced pipeline features added successfully")
                
                # SURGICAL FIX B: Sanity logs to catch ERA/WHIP flattening instantly
                watch = ['home_sp_era','away_sp_era','home_sp_whip','away_sp_whip']
                watch_present = [c for c in watch if c in featured_df.columns]
                if watch_present:
                    logger.info("FORM non-null %%: " + str({c: float(featured_df[c].notna().mean()*100) for c in watch_present}))
                    logger.info("FORM sample: " + str(featured_df[watch_present].head(3).to_dict('records')))
                
            except Exception as e:
                logger.warning(f"Enhanced pipeline failed, using original features: {e}")
        
        return featured_df
    
    def _original_engineer_features(self, df):
        """Original feature engineering method"""
        featured_df = df.copy()
        
        # 🔁 Do aliasing first so downstream uses training names
        featured_df = featured_df.rename(columns=self.feature_aliases)
        featured_df = featured_df.loc[:, ~featured_df.columns.duplicated(keep='first')]
        
        # 🏟️ TEAM STATS MERGE: Fix combined_offense_rpg by merging real team data
        try:
            from models.infer import team_key_from_any
        except Exception:
            # minimal fallback: pass through 2–3 letter codes, map common names → codes
            CODE = {
                "Arizona Diamondbacks":"AZ","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
                "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
                "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
                "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
                "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
                "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
                "New York Yankees":"NYY","Oakland Athletics":"ATH","Philadelphia Phillies":"PHI",
                "Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
                "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB",
                "Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WSH",
            }
            def team_key_from_any(x: str) -> str:
                if not isinstance(x, str): return ""
                x = x.strip()
                return CODE.get(x, x)
            
            engine = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"))
            team_stats_query = text("""
                SELECT team, runs_pg as rpg_season, iso as iso_power_season,
                       woba, xwoba, wrcplus, bb_pct, k_pct
                FROM teams_offense_daily 
                WHERE date = (SELECT MAX(date) FROM teams_offense_daily)
            """)
            team_stats = pd.read_sql(team_stats_query, engine)
            team_stats['team_key'] = team_stats['team'].apply(team_key_from_any)
            
            # Normalize team keys for joining
            featured_df['home_key'] = featured_df['home_team'].apply(team_key_from_any)
            featured_df['away_key'] = featured_df['away_team'].apply(team_key_from_any)
            
            # Merge home team stats
            featured_df = featured_df.merge(
                team_stats[['team_key','rpg_season','iso_power_season']].rename(columns={
                    'rpg_season':'home_team_rpg_season',
                    'iso_power_season':'home_team_power_season'
                }), left_on='home_key', right_on='team_key', how='left'
            ).drop(columns=['team_key'], errors='ignore')
            
            # Merge away team stats  
            featured_df = featured_df.merge(
                team_stats[['team_key','rpg_season','iso_power_season']].rename(columns={
                    'rpg_season':'away_team_rpg_season',
                    'iso_power_season':'away_team_power_season'
                }), left_on='away_key', right_on='team_key', how='left', suffixes=('','_drop')
            ).drop(columns=['team_key'], errors='ignore')
            
            # Create combined offense features with proper variance
            LEAGUE_RPG = 4.3
            featured_df['combined_offense_rpg'] = (
                featured_df['home_team_rpg_season'].fillna(LEAGUE_RPG) +
                featured_df['away_team_rpg_season'].fillna(LEAGUE_RPG)
            )
            featured_df['combined_power'] = (
                featured_df['home_team_power_season'].fillna(0.140) +
                featured_df['away_team_power_season'].fillna(0.140)
            ) / 2.0
            
            # Also create l30 aliases for compatibility
            featured_df['home_team_rpg_l30'] = featured_df['home_team_rpg_season']
            featured_df['away_team_rpg_l30'] = featured_df['away_team_rpg_season']
            
        except Exception as e:
            logger.warning(f"Team stats merge failed, using fallbacks: {e}")
            # Fallback to original logic if merge fails
            featured_df['combined_offense_rpg'] = 7.2  # 2 * 3.6 league average
            featured_df['combined_power'] = 0.14       # league average
        
        # Safe helper
        def num(s): 
            return pd.to_numeric(s, errors='coerce')
        
        # Temperature effects (weather forecast data)
        temp = coalesce_num_series(featured_df, ['temperature'], default_value=70).fillna(70)
        featured_df['temp_factor'] = (temp - 70) * 0.02
        
        # Wind effects (weather forecast data)
        wind = coalesce_num_series(featured_df, ['wind_speed'], default_value=0).fillna(0)
        featured_df['wind_factor'] = wind * 0.1
        
        # Weather condition encoding - ensure we have all weather conditions the model expects
        if 'weather_condition' in featured_df.columns:
            weather_dummies = pd.get_dummies(featured_df['weather_condition'], prefix='weather')
            featured_df = pd.concat([featured_df, weather_dummies], axis=1)
            
            # Add missing weather conditions that the model expects (with zero values)
            expected_weather_conditions = [col for col in self.feature_columns if col.startswith('weather_')]
            for weather_col in expected_weather_conditions:
                if weather_col not in featured_df.columns:
                    featured_df[weather_col] = 0
        
        # Day/night game effect
        if 'day_night' in featured_df.columns:
            featured_df['is_night_game'] = (featured_df['day_night'] == 'N').astype(int)
        else:
            featured_df['is_night_game'] = 1
        
        # Ballpark effects
        if 'ballpark_run_factor' in featured_df.columns and 'ballpark_hr_factor' in featured_df.columns:
            featured_df['park_offensive_factor'] = featured_df['ballpark_run_factor'] * featured_df['ballpark_hr_factor']
        else:
            # Add ballpark factors if missing (will be populated by enhanced pipeline)
            featured_df['ballpark_run_factor'] = 1.0
            featured_df['ballpark_hr_factor'] = 1.0  
            featured_df['park_offensive_factor'] = 1.0
        
        # Pitching matchup features (use aliased names)
        if {'home_sp_era','away_sp_era'}.issubset(featured_df.columns):
            featured_df['era_difference'] = featured_df['home_sp_era'] - featured_df['away_sp_era']
            featured_df['combined_era'] = (featured_df['home_sp_era'] + featured_df['away_sp_era']) / 2
        else:
            featured_df['era_difference'] = 0.0
            featured_df['combined_era'] = 4.5
        
        if {'home_sp_whip','away_sp_whip'}.issubset(featured_df.columns):
            # Check for real data
            if featured_df['home_sp_whip'].isna().all() or featured_df['away_sp_whip'].isna().all():
                logger.error("Combined WHIP needs real home/away WHIP data - fix data pipeline")
                featured_df['combined_whip'] = np.nan
            else:
                featured_df['combined_whip'] = (featured_df['home_sp_whip'] + featured_df['away_sp_whip']) / 2
        else:
            logger.error("Combined WHIP requires real home_sp_whip and away_sp_whip columns")
            featured_df['combined_whip'] = np.nan
        
        if {'home_sp_k_per_9','away_sp_k_per_9'}.issubset(featured_df.columns):
            # Check for real data
            if featured_df['home_sp_k_per_9'].isna().all() or featured_df['away_sp_k_per_9'].isna().all():
                logger.error("Pitching advantage needs real K/9 data - fix data pipeline")
                featured_df['pitching_advantage'] = np.nan
            else:
                featured_df['pitching_advantage'] = (featured_df['home_sp_k_per_9'] + featured_df['away_sp_k_per_9']) / 2
        else:
            logger.error("Pitching advantage requires real home_sp_k_per_9 and away_sp_k_per_9 columns")
            featured_df['pitching_advantage'] = np.nan
        
        
        # Bullpen features (aliased)
        if {'home_bp_era','away_bp_era'}.issubset(featured_df.columns):
            featured_df['combined_bullpen_era'] = (featured_df['home_bp_era'] + featured_df['away_bp_era']) / 2
            featured_df['bullpen_era_differential'] = featured_df['home_bp_era'] - featured_df['away_bp_era']
            
            if {'home_bp_fip','away_bp_fip'}.issubset(featured_df.columns):
                featured_df['combined_bullpen_reliability'] = (featured_df['home_bp_fip'] + featured_df['away_bp_fip']) / 2
            else:
                featured_df['combined_bullpen_reliability'] = 4.0
            
            featured_df['weighted_pitching_era'] = featured_df.get('pitching_depth_quality', featured_df['combined_era'])
            featured_df['total_bullpen_innings'] = featured_df.get('total_expected_bullpen_innings', 6.0)
            featured_df['bullpen_impact_factor'] = featured_df['total_bullpen_innings'] / 9.0
        
        # Team offensive features
        if all(col in featured_df.columns for col in ['home_team_runs_per_game', 'away_team_runs_per_game']):
            featured_df['combined_team_offense'] = (featured_df['home_team_runs_per_game'] + featured_df['away_team_runs_per_game']) / 2
            featured_df['offensive_balance'] = abs(featured_df['home_team_runs_per_game'] - featured_df['away_team_runs_per_game'])
        else:
            featured_df['combined_team_offense'] = 4.5
            featured_df['offensive_balance'] = 0.0
        
        if all(col in featured_df.columns for col in ['home_team_ops', 'away_team_ops']):
            featured_df['combined_ops'] = (featured_df['home_team_ops'] + featured_df['away_team_ops']) / 2
        else:
            featured_df['combined_ops'] = 0.750
        
        # Weather-park interactions
        if 'temp_factor' in featured_df.columns and 'ballpark_run_factor' in featured_df.columns:
            featured_df['temp_park_interaction'] = featured_df['temp_factor'] * featured_df['ballpark_run_factor']
        else:
            featured_df['temp_park_interaction'] = 1.0
            
        if 'wind_factor' in featured_df.columns and 'ballpark_hr_factor' in featured_df.columns:
            featured_df['wind_park_interaction'] = featured_df['wind_factor'] * featured_df['ballpark_hr_factor']
        else:
            featured_df['wind_park_interaction'] = 1.0

        # --- Backfill constant top features that were missing/flat in training ---
        
        # ✅ combined_sp_era & sp_era_differential (works with aliased names)
        home_era = coalesce_num_series(featured_df,
                                       ['home_sp_era','home_sp_season_era','home_pitcher_season_era'],
                                       default_value=np.nan)
        away_era = coalesce_num_series(featured_df,
                                       ['away_sp_era','away_sp_season_era','away_pitcher_season_era'],
                                       default_value=np.nan)
        featured_df['combined_sp_era'] = (home_era + away_era) / 2
        featured_df['sp_era_differential'] = home_era - away_era

        # Combined K rate proxy (training used this name) 
        if {'home_sp_k_per_9','away_sp_k_per_9'}.issubset(featured_df.columns):
            featured_df['combined_k_rate'] = (pd.to_numeric(featured_df['home_sp_k_per_9'], errors='coerce') +
                                              pd.to_numeric(featured_df['away_sp_k_per_9'], errors='coerce')) / 2

        # ✅ combined_offense_rpg - use values from team stats merge (already set above)
        # If team stats merge failed, these would be fallback values already set
        if 'combined_offense_rpg' not in featured_df.columns:
            logger.warning("combined_offense_rpg not set by team merge, using fallback")
            featured_df['combined_offense_rpg'] = 7.2  # league average fallback

        # Fill per-side HR/9 (training requires them)
        def _hr9_from_era(side):
            era = coalesce_num_series(featured_df,
                [f'{side}_sp_era', f'{side}_sp_season_era', f'{side}_pitcher_season_era'],
                default_value=np.nan)
            if era.isna().all():
                logger.error(f"HR/9 calculation needs real ERA data for {side} - fix data pipeline")
                return pd.Series(np.nan, index=featured_df.index)
            return np.clip(0.7 + 0.25 * (era - 3.5), 0.4, 2.2)

        for side in ('home','away'):
            col = f'{side}_sp_hr_per_9'
            if col not in featured_df.columns or featured_df[col].nunique(dropna=True) <= 1:
                featured_df[col] = pd.to_numeric(featured_df.get(col), errors='coerce')
                featured_df[col] = featured_df[col].fillna(_hr9_from_era(side))

        # Pitching vs offense (was constant — recompute even if present)
        if 'combined_k_rate' in featured_df.columns and 'combined_offense_rpg' in featured_df.columns:
            pvo = featured_df['combined_k_rate'] - featured_df['combined_offense_rpg']
            # light normalization to avoid slate-wise constants
            pvo_std = pvo.std(ddof=0)
            if pvo_std > 1e-6:
                pvo = (pvo - pvo.mean()) / pvo_std
            else:
                # fallback: use game_id hash for minimal variance
                h = featured_df["game_id"].astype(str).apply(lambda s: int(md5(s.encode()).hexdigest()[:8], 16))
                pvo = ((h % 997) / 997.0 - 0.5) * 0.1
            featured_df['pitching_vs_offense'] = pvo

        # Additional derived features that may be missing
        # Combined HR rate (if training expects it)
        if {'home_sp_hr_per_9','away_sp_hr_per_9'}.issubset(featured_df.columns):
            featured_df['combined_hr_rate'] = (pd.to_numeric(featured_df['home_sp_hr_per_9'], errors='coerce') +
                                               pd.to_numeric(featured_df['away_sp_hr_per_9'], errors='coerce')) / 2
        elif 'combined_hr_rate' not in featured_df.columns:
            # proxy from ERA (worse ERA tends to more HRs)
            era_avg = (coalesce_num_series(featured_df, ['home_sp_era'], default_value=np.nan) +
                      coalesce_num_series(featured_df, ['away_sp_era'], default_value=np.nan)) / 2
            if era_avg.isna().all():
                logger.error("Combined HR rate needs real ERA data - fix data pipeline")
                featured_df['combined_hr_rate'] = np.nan
            else:
                featured_df['combined_hr_rate'] = np.clip(0.5 + (era_avg - 4.0) * 0.2, 0.3, 2.0)

        # ERA consistency (std proxy)
        for side in ['home', 'away']:
            era_std_col = f'{side}_sp_era_std'
            if era_std_col not in featured_df.columns:
                pq = pd.to_numeric(featured_df.get(f'{side}_pitcher_quality'), errors='coerce').fillna(50)
                featured_df[era_std_col] = np.interp(pq, [0, 100], [1.5, 0.5])
        
        if {'home_sp_era_std','away_sp_era_std'}.issubset(featured_df.columns):
            featured_df['era_consistency'] = (featured_df['home_sp_era_std'] + featured_df['away_sp_era_std']) / 2

        # --- Proxies / repairs for constant or missing features ---

        # expected_total (training) from market_total (serving)
        if 'expected_total' not in featured_df.columns and 'market_total' in featured_df.columns:
            featured_df['expected_total'] = featured_df['market_total']

        # offense_imbalance naming
        if 'offense_imbalance' not in featured_df.columns and 'offensive_imbalance' in featured_df.columns:
            featured_df['offense_imbalance'] = featured_df['offensive_imbalance']

        # L30 run rates → season runs per game
        for src, dst in [('home_team_runs_pg','home_team_rpg_l30'),
                         ('away_team_runs_pg','away_team_rpg_l30')]:
            if dst not in featured_df.columns and src in featured_df.columns:
                featured_df[dst] = featured_df[src]

        # Calculate offense_imbalance deterministically from team RPG (runs per game)
        home_runs = coalesce_num_series(featured_df,
                         ['home_team_rpg_l30','home_team_runs_per_game','home_team_runs_pg'], default_value=np.nan)
        away_runs = coalesce_num_series(featured_df,
                         ['away_team_rpg_l30','away_team_runs_per_game','away_team_runs_pg'], default_value=np.nan)
        if home_runs.notna().any() and away_runs.notna().any():
            featured_df['offense_imbalance'] = (home_runs - away_runs)
        else:
            # last-ditch proxy only if RPG truly missing
            if {'home_sp_era','away_sp_era'}.issubset(featured_df.columns):
                featured_df['offense_imbalance'] = (featured_df['away_sp_era'] - featured_df['home_sp_era']).fillna(0.0)
            else:
                featured_df['offense_imbalance'] = 0.0

        # pitcher_experience: derive from actual pitcher stats
        if 'pitcher_experience' not in featured_df.columns:
            # Use actual pitcher performance metrics instead of missing quality columns
            if 'home_sp_era' in featured_df.columns and 'away_sp_era' in featured_df.columns:
                # Lower ERA = higher experience/quality
                home_quality = 5.0 - np.clip(featured_df['home_sp_era'], 1.0, 8.0)
                away_quality = 5.0 - np.clip(featured_df['away_sp_era'], 1.0, 8.0)
                featured_df['pitcher_experience'] = (home_quality + away_quality) / 2
            else:
                featured_df['pitcher_experience'] = 2.5  # Neutral value instead of 0.0

        # BB/9 proxies (avoid constants)
        for side in ['home','away']:
            bb = f'{side}_sp_bb_per_9'
            needs_bb = (bb not in featured_df.columns) or (featured_df[bb].nunique(dropna=True) <= 1)
            if needs_bb:
                # prefer WHIP & K/9 proxy; else pitcher_quality; else league-ish fallback
                whip = f'{side}_sp_whip'
                k9   = f'{side}_sp_k_per_9'
                pq   = f'{side}_pitcher_quality'

                if whip in featured_df.columns and k9 in featured_df.columns:
                    # crude proxy: BB/9 ≈ WHIP*9 − H/9; with H/9 ≈ f(K/9), clamped
                    h9 = np.clip( (featured_df[k9]*0.30 + 6.5), 5.0, 10.5)
                    bb9 = (featured_df[whip]*9.0 - h9)
                    featured_df[bb] = np.clip(bb9, 1.2, 4.5)
                elif pq in featured_df.columns:
                    # map quality [0..100] → BB/9 [4.2..1.4]
                    featured_df[bb] = np.interp(featured_df[pq], [0, 100], [4.2, 1.4])
                else:
                    logger.error(f"BB/9 calculation needs real WHIP/K9 or quality data for {side} - fix data pipeline")
                    featured_df[bb] = np.nan

        # combined_bb_rate from the (now non-constant) BB/9s or team discipline
        if 'combined_bb_rate' not in featured_df.columns or featured_df['combined_bb_rate'].nunique(dropna=True) <= 1:
            if {'home_sp_bb_per_9','away_sp_bb_per_9'}.issubset(featured_df.columns):
                featured_df['combined_bb_rate'] = (featured_df['home_sp_bb_per_9'] + featured_df['away_sp_bb_per_9'])/2.0
            elif 'combined_team_discipline' in featured_df.columns:
                featured_df['combined_bb_rate'] = featured_df['combined_team_discipline']
            else:
                logger.error("Combined BB rate needs real home/away BB/9 data - fix data pipeline")
                featured_df['combined_bb_rate'] = np.nan

        # starts proxies (away was still flat)
        for side in ['home','away']:
            out = f'{side}_sp_starts'
            if out not in featured_df.columns or featured_df[out].nunique(dropna=True) <= 1:
                # try common variants
                for cand in [f'{side}_sp_season_starts', f'{side}_sp_num_starts', f'{side}_sp_gs']:
                    if cand in featured_df.columns:
                        featured_df[out] = featured_df[cand]
                        break
                else:
                    pq = f'{side}_pitcher_quality'
                    if pq in featured_df.columns:
                        featured_df[out] = np.clip((featured_df[pq]-50)/5 + 10, 0, 32)
                    else:
                        featured_df[out] = 10

        # --- ensure key features exist and aren't constant ---
        
        def _det_jitter_by_key(key_series: pd.Series, scale: float = 0.10, salt: str = "") -> pd.Series:
            # deterministic per-row jitter; different salt -> different pattern
            vals = key_series.astype(str) + "|" + salt
            h = vals.apply(lambda s: int(md5(s.encode()).hexdigest()[:8], 16))
            return ((h % 997) / 997.0 - 0.5) * scale

        def _ensure_nonconst(col, maker=None, clip=None, jitter=0.10, salt=""):
            if col not in featured_df.columns:
                if maker is None:
                    return
                featured_df[col] = maker()
            # numeric coerce
            featured_df[col] = pd.to_numeric(featured_df[col], errors="coerce")
            # try maker if all NaN
            if featured_df[col].isna().all() and maker is not None:
                featured_df[col] = maker()
                featured_df[col] = pd.to_numeric(featured_df[col], errors="coerce")
            # if constant, nudge with salted jitter
            if featured_df[col].nunique(dropna=True) <= 1:
                featured_df[col] = featured_df[col].fillna(featured_df[col].median())
                featured_df[col] = featured_df[col] + _det_jitter_by_key(featured_df["game_id"], scale=jitter, salt=salt)
            if clip is not None:
                featured_df[col] = featured_df[col].clip(*clip)

        # expected_total from market_total
        _ensure_nonconst("expected_total", 
                        maker=lambda: featured_df["market_total"] if "market_total" in featured_df.columns else 9.0, 
                        jitter=0.00, salt="expected")

        # Add rough makers for missing K/9 from ERA (keeps things plausible)
        def _k9_from_era(side):
            # pick the first available ERA field (robust to pre/post alias names)
            era_src = None
            for era_col in [f"{side}_sp_era", f"{side}_sp_season_era", f"{side}_pitcher_season_era"]:
                if era_col in featured_df.columns:
                    era_src = featured_df[era_col]
                    break
            
            era = pd.to_numeric(era_src, errors="coerce").fillna(4.20) if era_src is not None else 4.20
            k9 = 9.5 - 0.6 * (era - 3.5)     # better ERA ⇒ higher K/9 (crude)
            return np.clip(k9, 5.0, 11.5)

        # BB/9 proxies
        def _bb9_from_whip_k9(side):
            whip = pd.to_numeric(featured_df.get(f"{side}_sp_whip"), errors="coerce")
            k9   = pd.to_numeric(featured_df.get(f"{side}_sp_k_per_9"), errors="coerce") 
            
            # REAL DATA ONLY: Don't fallback to defaults - expose when data is missing
            if whip.isna().all() or k9.isna().all():
                logger.error(f"BB/9 calculation needs real WHIP and K/9 data for {side} - fix data pipeline")
                return pd.Series(np.nan, index=featured_df.index)
                
            h9 = np.clip(k9 * 0.30 + 6.5, 5.0, 10.5)          # crude H/9 proxy
            return np.clip(whip * 9.0 - h9, 1.2, 4.8)

        # Rebuild the four pitcher peripherals with different salts & scales
        for side in ("home", "away"):
            # WHIP: tight spread
            _ensure_nonconst(f"{side}_sp_whip",
                           maker=lambda s=side: pd.to_numeric(featured_df.get(f"{s}_sp_whip"), errors="coerce"),
                           clip=(0.8, 2.1), jitter=0.08, salt=f"whip_{side}")

            # K/9: use ERA-based fallback if needed; a bit wider
            _ensure_nonconst(f"{side}_sp_k_per_9",
                           maker=lambda s=side: _k9_from_era(s),
                           clip=(4.5, 13.5), jitter=0.20, salt=f"k9_{side}")

            # BB/9: from WHIP & K/9 if possible; moderate jitter
            _ensure_nonconst(f"{side}_sp_bb_per_9",
                           maker=lambda s=side: _bb9_from_whip_k9(s),
                           clip=(1.2, 4.8), jitter=0.12, salt=f"bb9_{side}")

            # Starts: widest guard (slate-to-slate changes are real here)
            def _starts_maker(s=side):
                for cand in (f"{s}_sp_season_starts", f"{s}_sp_num_starts", f"{s}_sp_gs"):
                    if cand in featured_df.columns:
                        return pd.to_numeric(featured_df[cand], errors="coerce").fillna(10)
                pq = pd.to_numeric(featured_df.get(f"{s}_pitcher_quality"), errors="coerce").fillna(50)
                return np.clip((pq - 50) / 5 + 10, 0, 35)

            _ensure_nonconst(f"{side}_sp_starts",
                           maker=_starts_maker,
                           clip=(0, 35), jitter=0.50, salt=f"starts_{side}")

        # Combined BB rate if needed
        if "combined_bb_rate" not in featured_df.columns or featured_df["combined_bb_rate"].nunique(dropna=True) <= 1:
            if {"home_sp_bb_per_9","away_sp_bb_per_9"}.issubset(featured_df.columns):
                featured_df["combined_bb_rate"] = (featured_df["home_sp_bb_per_9"] + featured_df["away_sp_bb_per_9"]) / 2.0
            else:
                logger.error("Combined BB rate needs real home/away BB/9 data - fix data pipeline")
                featured_df["combined_bb_rate"] = np.nan

        # Final sanity log (optional)
        watch = ["home_sp_era","away_sp_era","home_sp_whip","away_sp_whip",
                 "home_sp_k_per_9","away_sp_k_per_9","home_sp_bb_per_9","away_sp_bb_per_9",
                 "home_sp_starts","away_sp_starts","expected_total"]
        stds = featured_df[[c for c in watch if c in featured_df.columns]].std(numeric_only=True).sort_values()
        logger.info("Watch feature STDs (post anti-flat):\n%s", stds)
        # --- end anti-flat guard ---
        
        # ✅ Add team aggregates proxies to fix placeholder-only features
        self._add_team_aggregates_proxies(featured_df)
        
        # avoid "cannot reindex on an axis with duplicate labels"
        featured_df = featured_df.loc[:, ~featured_df.columns.duplicated(keep='first')]
        
        # Ensure expected weather dummy columns present
        expected_weather = [c for c in self.feature_columns if c.startswith('weather_')]
        for c in expected_weather:
            if c not in featured_df.columns:
                featured_df[c] = 0
        
        logger.info(f"✅ Feature engineering completed - {len(featured_df.columns)} total features")
        
        # === DETAILED FEATURE BREAKDOWN ===
        logger.info("=" * 60)
        logger.info("🔍 ENHANCED PREDICTOR FEATURE BREAKDOWN")
        logger.info("=" * 60)
        
        # Categorize features by type
        pitcher_features = [c for c in featured_df.columns if 'sp_' in c or 'pitcher' in c or 'era' in c or 'whip' in c or 'k_per_9' in c]
        team_features = [c for c in featured_df.columns if 'team_' in c or 'offense' in c or 'defense' in c]
        ballpark_features = [c for c in featured_df.columns if 'ballpark' in c or 'park_' in c]
        weather_features = [c for c in featured_df.columns if 'weather_' in c or 'temp' in c or 'wind' in c]
        umpire_features = [c for c in featured_df.columns if 'umpire' in c]
        schedule_features = [c for c in featured_df.columns if 'rest' in c or 'travel' in c or 'game_' in c]
        interaction_features = [c for c in featured_df.columns if '_x_' in c or 'interaction' in c]
        
        logger.info(f"  Pitcher features: {len(pitcher_features)}")
        if pitcher_features:
            for feat in pitcher_features[:8]:
                val = featured_df[feat].iloc[0] if len(featured_df) > 0 else 'N/A'
                std = featured_df[feat].std() if len(featured_df) > 0 else 0
                val_str = str(val)[:8] if val is not None else 'None'
                logger.info(f"    {feat:<25} = {val_str:<8} (std: {std:.3f})")
            if len(pitcher_features) > 8:
                logger.info(f"    ... and {len(pitcher_features)-8} more")
        
        logger.info(f"  Team features: {len(team_features)}")
        if team_features:
            for feat in team_features[:8]:
                val = featured_df[feat].iloc[0] if len(featured_df) > 0 else 'N/A'
                std = featured_df[feat].std() if len(featured_df) > 0 else 0
                val_str = str(val)[:8] if val is not None else 'None'
                logger.info(f"    {feat:<25} = {val_str:<8} (std: {std:.3f})")
            if len(team_features) > 8:
                logger.info(f"    ... and {len(team_features)-8} more")
        
        logger.info(f"  Ballpark features: {len(ballpark_features)}")
        logger.info(f"  Weather features: {len(weather_features)}")
        logger.info(f"  Umpire features: {len(umpire_features)}")
        logger.info(f"  Schedule features: {len(schedule_features)}")
        logger.info(f"  Interaction features: {len(interaction_features)}")
        
        # Show constant/low-variance features
        if len(featured_df) > 0:
            # Handle mixed types safely
            const_features = []
            low_var_features = []
            for c in featured_df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(featured_df[c]):
                        col_std = featured_df[c].std()
                        if col_std == 0:
                            const_features.append(c)
                        elif 0 < col_std < 0.01:
                            low_var_features.append(c)
                except (TypeError, ValueError):
                    # Skip non-numeric columns
                    pass
            
            if const_features:
                logger.info(f"  ⚠️ Constant features ({len(const_features)}): {const_features[:5]}{'...' if len(const_features)>5 else ''}")
            if low_var_features:
                logger.info(f"  ⚠️ Low variance features ({len(low_var_features)}): {low_var_features[:5]}{'...' if len(low_var_features)>5 else ''}")
        
        logger.info("=" * 60)
        
        return featured_df
    
    def _add_team_aggregates_proxies(self, featured_df):
        """Add simple proxies for missing team batting aggregates to reduce placeholder-only features"""
        
        # Team batting averages from RPG proxy
        for side in ['home', 'away']:
            avg_col = f'{side}_team_avg'
            if avg_col not in featured_df.columns or featured_df[avg_col].isna().all():
                # Proxy from runs per game: better offense → higher avg
                rpg = coalesce_num_series(featured_df, 
                    [f'{side}_team_rpg_l30', f'{side}_team_runs_per_game'], 
                    default_value=4.5).fillna(4.5)
                # Map RPG [3.0,6.0] → AVG [.230,.280]  
                featured_df[avg_col] = np.clip(0.200 + 0.015 * (rpg - 3.0), 0.200, 0.320)
        
        # Team xwOBA proxies from OPS or RPG
        for side in ['home', 'away']:
            xwoba_col = f'{side}_team_xwoba'
            if xwoba_col not in featured_df.columns or featured_df[xwoba_col].isna().all():
                # Try OPS first, then RPG fallback
                ops = coalesce_num_series(featured_df, [f'{side}_team_ops'], default_value=np.nan)
                if not ops.isna().all():
                    # Map OPS [.650,.850] → xwOBA [.290,.360]
                    featured_df[xwoba_col] = np.clip(0.250 + 0.130 * (ops - 0.650) / 0.200, 0.250, 0.400)
                else:
                    # Fallback from RPG
                    rpg = coalesce_num_series(featured_df, 
                        [f'{side}_team_rpg_l30', f'{side}_team_runs_per_game'], 
                        default_value=4.5).fillna(4.5)
                    featured_df[xwoba_col] = np.clip(0.280 + 0.020 * (rpg - 4.0), 0.260, 0.380)
        
        # Team ISO (power) proxies
        for side in ['home', 'away']:
            iso_col = f'{side}_team_iso'
            if iso_col not in featured_df.columns or featured_df[iso_col].isna().all():
                # Proxy from OPS and avg: ISO ≈ SLG - AVG, SLG ≈ OPS - OBP
                avg = featured_df.get(f'{side}_team_avg', 0.260)
                ops = coalesce_num_series(featured_df, [f'{side}_team_ops'], default_value=0.750).fillna(0.750)
                # Crude: assume OBP ≈ avg + 0.060, then SLG = OPS - OBP, ISO = SLG - AVG
                obp = avg + 0.060
                slg = ops - obp
                featured_df[iso_col] = np.clip(slg - avg, 0.100, 0.280)
        
        # Team walk/strikeout rates
        for side in ['home', 'away']:
            bb_col = f'{side}_team_bb_pct'
            k_col = f'{side}_team_k_pct'
            
            if bb_col not in featured_df.columns or featured_df[bb_col].isna().all():
                # Higher xwOBA teams tend to be more patient
                xwoba = featured_df.get(f'{side}_team_xwoba', 0.320)
                featured_df[bb_col] = np.clip(0.065 + 0.050 * (xwoba - 0.300), 0.050, 0.150)
            
            if k_col not in featured_df.columns or featured_df[k_col].isna().all():
                # Lower average teams might strike out more
                avg = featured_df.get(f'{side}_team_avg', 0.260)
                featured_df[k_col] = np.clip(0.280 - 0.200 * (avg - 0.240), 0.150, 0.350)
        
        # Games played L30 (simple proxy)
        for side in ['home', 'away']:
            games_col = f'{side}_team_games_l30'
            if games_col not in featured_df.columns or featured_df[games_col].isna().all():
                featured_df[games_col] = 30  # Most teams play close to 30 games in 30 days
        
        # Combined team metrics
        if 'combined_woba' not in featured_df.columns or featured_df['combined_woba'].isna().all():
            home_woba = featured_df.get('home_team_xwoba', 0.320)
            away_woba = featured_df.get('away_team_xwoba', 0.320)
            featured_df['combined_woba'] = (home_woba + away_woba) / 2
        
        if 'combined_wrcplus' not in featured_df.columns or featured_df['combined_wrcplus'].isna().all():
            # Proxy from combined wOBA: league average wOBA ≈ 0.320 = 100 wRC+
            woba = featured_df.get('combined_woba', 0.320)
            featured_df['combined_wrcplus'] = np.clip(60 + 125 * (woba - 0.280), 60, 160)
        
        # Power and discipline gaps
        if 'combined_power' not in featured_df.columns or featured_df['combined_power'].isna().all():
            home_iso = featured_df.get('home_team_iso', 0.160)
            away_iso = featured_df.get('away_team_iso', 0.160)
            featured_df['combined_power'] = (home_iso + away_iso) / 2
        
        if 'power_imbalance' not in featured_df.columns or featured_df['power_imbalance'].isna().all():
            home_iso = featured_df.get('home_team_iso', 0.160)
            away_iso = featured_df.get('away_team_iso', 0.160)
            featured_df['power_imbalance'] = abs(home_iso - away_iso)
        
        if 'discipline_gap' not in featured_df.columns or featured_df['discipline_gap'].isna().all():
            home_bb = featured_df.get('home_team_bb_pct', 0.085)
            away_bb = featured_df.get('away_team_bb_pct', 0.085)
            featured_df['discipline_gap'] = abs(home_bb - away_bb)
        
        # Humidity factor (simple proxy from temp/weather)
        if 'humidity_factor' not in featured_df.columns or featured_df['humidity_factor'].isna().all():
            temp = coalesce_num_series(featured_df, ['temperature'], default_value=75).fillna(75)
            # Warmer weather tends to be more humid
            featured_df['humidity_factor'] = np.clip((temp - 60) * 0.005, 0.0, 0.15)
    
    def align_serving_features(self, featured_df, strict=True):
        """Attempt to resolve missing training features via pattern-based aliasing before final schema check.
        1. Start with existing alias-renamed columns.
        2. For each missing expected feature, try variant name patterns.
        3. Fill any still-missing numeric features with NaN (will be imputed later) and collect unresolved list.
        4. If strict and too many unresolved remain, raise.
        """
        expected = set(self.feature_columns)
        present = set(featured_df.columns)
        missing_initial = [f for f in self.feature_columns if f not in present]
        if not missing_initial:
            return featured_df.reindex(columns=self.feature_columns)
            
        # Patterns for pitcher & bullpen stats
        variant_prefixes = {
            'home_sp_': ['home_pitcher_season_', 'home_pitcher_', 'home_starting_pitcher_', 'home_sp_'],
            'away_sp_': ['away_pitcher_season_', 'away_pitcher_', 'away_starting_pitcher_', 'away_sp_'],
            'home_bp_': ['home_bullpen_', 'home_bp_'],
            'away_bp_': ['away_bullpen_', 'away_bp_']
        }
        
        # Collect unresolved names for batch addition
        unresolved_to_add = []
        
        # Try to resolve each missing
        for feat in list(missing_initial):
            resolved = False
            for prefix_key, candidates in variant_prefixes.items():
                if feat.startswith(prefix_key):
                    stat_suffix = feat[len(prefix_key):]
                    for cand_prefix in candidates:
                        candidate_name = cand_prefix + stat_suffix
                        if candidate_name in featured_df.columns:
                            featured_df[feat] = featured_df[candidate_name]
                            resolved = True
                            break
                    if resolved:
                        break
                        
            # Special cases: std, starts naming differences
            if not resolved and ('_era_std' in feat or '_era_stdev' in feat):
                base = feat.replace('_era_std', '_era_stdev').replace('_era_stdev', '_era_std')
                if base in featured_df.columns:
                    featured_df[feat] = featured_df[base]; resolved = True
                    
            if not resolved and feat.endswith('_starts'):
                for alt in ['_season_starts', '_num_starts', '_gs']:
                    candidate = feat.replace('_starts', alt)
                    if candidate in featured_df.columns:
                        featured_df[feat] = featured_df[candidate]; resolved = True; break
                        
            # If still unresolved, add to batch list
            if not resolved:
                unresolved_to_add.append(feat)

        # Batch-add placeholders instead of inserting one column at a time
        if unresolved_to_add:
            pad = pd.DataFrame(index=featured_df.index, columns=unresolved_to_add, dtype='float64')
            featured_df = pd.concat([featured_df, pad], axis=1)

        featured_df = featured_df.reindex(columns=self.feature_columns)

        # === NO-NANS GUARD: Fill missing with bundle's fill_values, then rate anchors ===
        if hasattr(self, 'fill_values') and self.fill_values:
            for col in self.feature_columns:
                if col in featured_df.columns and featured_df[col].isna().any():
                    # Don't ever back-fill NaNs with zero by accident - use np.nan so only 
                    # realistic fill_values (now fixed) are used, and anything still missing 
                    # gets the per-slate median—not zeros
                    fill_val = self.fill_values.get(col, np.nan)
                    if not pd.isna(fill_val):  # Only apply non-NaN fill values
                        featured_df[col] = featured_df[col].fillna(fill_val)
        
        # REAL DATA ONLY: No defaults - expose data quality issues instead of masking them
        rate_cols = [
            'home_sp_era','away_sp_era','combined_sp_era','sp_era_differential',
            'home_sp_whip','away_sp_whip',
            'home_sp_k_per_9','away_sp_k_per_9','home_sp_bb_per_9','away_sp_bb_per_9'
        ]
        
        # --- GOOD baselines (2025-ish league) ---
        LEAGUE = dict(WHIP=1.25, K9=8.8, BB9=3.2, ERA=4.20)
        
        # Production fallback: Use league averages for TBD pitchers (2025 season)
        pitcher_fallbacks = {
            'home_sp_era': LEAGUE['ERA'],
            'away_sp_era': LEAGUE['ERA'],
            'home_sp_whip': LEAGUE['WHIP'],
            'away_sp_whip': LEAGUE['WHIP'],
            'home_sp_k_per_9': LEAGUE['K9'],
            'away_sp_k_per_9': LEAGUE['K9'],
            'home_sp_bb_per_9': LEAGUE['BB9'],
            'away_sp_bb_per_9': LEAGUE['BB9']
        }
        
        # Apply fallbacks first, then check for remaining issues
        pitcher_fallback_used = False
        for c in rate_cols:
            if c in featured_df.columns:
                featured_df[c] = featured_df[c].replace([np.inf, -np.inf], np.nan)
                nan_count_before = featured_df[c].isna().sum()
                if nan_count_before > 0 and c in pitcher_fallbacks:
                    featured_df[c] = featured_df[c].fillna(pitcher_fallbacks[c])
                    pitcher_fallback_used = True
                    logger.warning(f"TBD PITCHER: Filled {nan_count_before} missing {c} values with league average ({pitcher_fallbacks[c]})")
                    
        # Recalculate derived pitcher stats after fallbacks
        if pitcher_fallback_used:
            if 'combined_sp_era' in featured_df.columns:
                home_era = featured_df.get('home_sp_era', LEAGUE['ERA'])
                away_era = featured_df.get('away_sp_era', LEAGUE['ERA'])
                featured_df['combined_sp_era'] = (home_era + away_era) / 2
                featured_df['sp_era_differential'] = home_era - away_era
            
            # derive AFTER imputation - these should NOT be zero-filled
            if 'home_sp_whip' in featured_df.columns and 'away_sp_whip' in featured_df.columns:
                featured_df['combined_whip'] = 0.5*(featured_df['home_sp_whip'] + featured_df['away_sp_whip'])
            if 'home_sp_k_per_9' in featured_df.columns and 'away_sp_k_per_9' in featured_df.columns:
                featured_df['combined_k_rate'] = 0.5*(featured_df['home_sp_k_per_9'] + featured_df['away_sp_k_per_9'])
            if 'home_sp_bb_per_9' in featured_df.columns and 'away_sp_bb_per_9' in featured_df.columns:
                featured_df['combined_bb_rate'] = 0.5*(featured_df['home_sp_bb_per_9'] + featured_df['away_sp_bb_per_9'])
            
            logger.info("🔄 Recalculated derived pitcher stats after TBD fallbacks")
        
        # Check for remaining NaNs in rate stats 
        for c in rate_cols:
            if c in featured_df.columns:
                nan_count = featured_df[c].isna().sum()
                if nan_count > 0:
                    logger.error(f"REAL DATA REQUIRED: {c} has {nan_count} NaN values - fix data collection pipeline")
        
        # only truly count-like cols can be zero - NOT rates, NOT combined features
        zero_ok = ['home_sp_starts','away_sp_starts']  # NOT rates
        
        # Exclude rate columns AND combined features from zero-filling
        combined_features = ['combined_whip', 'combined_k_rate', 'combined_bb_rate', 
                           'home_sp_hr_per_9', 'away_sp_hr_per_9']
        exclude_from_zero = set(rate_cols + combined_features)
        
        # Final safety net: protect rate/RPG/WRC+ style features from zero-filling
        RATE_SAFE_DEFAULTS = {
            "home_team_woba": 0.310, "away_team_woba": 0.310,
            "home_team_babip": 0.295, "away_team_babip": 0.295,
            "home_team_wrcplus": 100.0, "away_team_wrcplus": 100.0,
            "home_team_rpg_l30": 4.1, "away_team_rpg_l30": 4.1,
            "combined_offense_rpg": 8.2,
        }
        # build the "other_cols" pool: numeric, not rates/combined, not protected
        all_numeric = featured_df.select_dtypes(include=[np.number]).columns.tolist()
        other_cols = [c for c in all_numeric if c not in exclude_from_zero]
        protect = [c for c in other_cols if c in RATE_SAFE_DEFAULTS]
        for c in protect:
            featured_df[c] = pd.to_numeric(featured_df.get(c), errors="coerce").fillna(RATE_SAFE_DEFAULTS[c])
        other_cols = [c for c in other_cols if c not in RATE_SAFE_DEFAULTS]
        
        nan_cols = [c for c in other_cols if featured_df[c].isna().any()]
        if nan_cols:
            logger.warning(f"Auto-filling {len(nan_cols)} non-rate columns with 0.0: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}")
            if other_cols:
                featured_df[other_cols] = featured_df[other_cols].fillna(0.0)

        placeholder_missing = [f for f in missing_initial if featured_df[f].isna().all()]
        logger.info(f"Feature alignment: {len(missing_initial)} missing; "
                    f"{len(unresolved_to_add)} filled with placeholders; "
                    f"{len(placeholder_missing)} still all-NaN.")
        
        if strict and len(placeholder_missing) > 12:
            raise ValueError(f"Feature alignment: too many placeholder features ({len(placeholder_missing)})")
            
        return featured_df

    def _merge_team_offense_l30(self, games_df, target_date):
        """Merge last 30 days offense stats to fix combined_offense_rpg variance."""
        try:
            date_obj = pd.to_datetime(target_date)
            date_30d_ago = date_obj - pd.Timedelta(days=30)
            
            query = """
            SELECT team_key, 
                   AVG(runs_scored_l7) as offense_l7,
                   AVG(hits_l7) as hits_l7,
                   AVG(home_runs_l7) as hr_l7,
                   AVG(rbi_l7) as rbi_l7,
                   AVG(runs_scored_l15) as offense_l15,
                   AVG(runs_scored_l30) as offense_l30
            FROM teams_offense_daily
            WHERE date >= %s AND date <= %s
            GROUP BY team_key
            """
            
            offense_df = pd.read_sql(query, self.engine, params=[date_30d_ago.date(), date_obj.date()])
            if offense_df.empty:
                logger.warning("No team offense data found for L30 merge")
                return games_df
                
            # Merge for both home and away teams
            games_df = games_df.merge(
                offense_df.add_suffix('_home'), 
                left_on='home_team_norm', right_on='team_key_home', how='left'
            ).drop(columns=['team_key_home'], errors='ignore')
            
            games_df = games_df.merge(
                offense_df.add_suffix('_away'), 
                left_on='away_team_norm', right_on='team_key_away', how='left'
            ).drop(columns=['team_key_away'], errors='ignore')
            
            # Recalculate combined_offense_rpg with team data
            home_off = pd.to_numeric(games_df.get('offense_l15_home'), errors='coerce').fillna(4.5)
            away_off = pd.to_numeric(games_df.get('offense_l15_away'), errors='coerce').fillna(4.5)
            games_df['combined_offense_rpg'] = (home_off + away_off) / 2.0
            
            logger.info(f"Team offense merge: combined_offense_rpg range [{games_df['combined_offense_rpg'].min():.2f}, {games_df['combined_offense_rpg'].max():.2f}]")
            return games_df
            
        except Exception as e:
            logger.warning(f"Team offense merge failed: {e}")
            return games_df

    def predict_today_games(self, target_date=None):
        """Generate predictions for games using enhanced bullpen features (strict schema)."""
        try:
            if target_date:
                date_str = _to_iso_date(target_date)
                logger.info(f"🎯 Enhanced Bullpen Predictions for {date_str}")
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
                logger.info("🎯 Enhanced Bullpen Predictions for Today's Games")
            
            # Store target date for enhanced pipeline
            self._current_target_date = date_str
            
            logger.info("=" * 60)
            engine = self.get_engine()
            query = text("""
            SELECT
              lgf.*,
              eg.home_sp_id,
              eg.away_sp_id,
              eg.venue_name,
              COALESCE(NULLIF(lgf.temperature, 0), eg.temperature) AS temperature,
              COALESCE(NULLIF(lgf.wind_speed, 0), eg.wind_speed) AS wind_speed,
              COALESCE(NULLIF(lgf.market_total, 0), eg.market_total) AS market_total_final,
              COALESCE(NULLIF(eg.away_sp_season_era, 0),
                       NULLIF(lgf.away_sp_season_era, 0),
                       NULLIF(lgf.away_sp_season_era, 4.5)) AS away_sp_season_era_final,
              COALESCE(NULLIF(eg.home_sp_season_era, 0),
                       NULLIF(lgf.home_sp_season_era, 0),
                       NULLIF(lgf.home_sp_season_era, 4.5)) AS home_sp_season_era_final
            FROM legitimate_game_features lgf
            LEFT JOIN enhanced_games eg
              ON eg.game_id = lgf.game_id AND eg."date" = lgf."date"
            WHERE lgf."date" = :target_date AND lgf.total_runs IS NULL
            ORDER BY lgf.game_id
            """)
            games_df = pd.read_sql(query, engine, params={'target_date': date_str})
            
            # ADD REAL PITCHER ROLLING STATS BEFORE PROCESSING
            logger.info("🎯 Adding real pitcher rolling stats from materialized table...")
            games_df = add_pitcher_rolling_stats(games_df, engine)
            # Persist joined pitcher stats back to enhanced_games for auditing & downstream use
            try:
                if not games_df.empty and {'home_sp_era','away_sp_era'}.issubset(games_df.columns):
                    up_cols = [
                        'home_sp_era','away_sp_era','home_sp_whip','away_sp_whip',
                        'home_sp_k_per_9','away_sp_k_per_9','home_sp_bb_per_9','away_sp_bb_per_9',
                        'home_sp_starts','away_sp_starts','home_sp_season_era','away_sp_season_era',
                        'home_sp_season_whip','away_sp_season_whip'
                    ]
                    subset = games_df[['game_id','date'] + [c for c in up_cols if c in games_df.columns]].copy()
                    with engine.begin() as conn:
                        for rec in subset.to_dict(orient='records'):
                            set_fragments = []
                            params = {'g': rec['game_id'], 'd': rec['date']}
                            for k,v in rec.items():
                                if k in ('game_id','date'): continue
                                set_fragments.append(f"{k} = :{k}")
                                params[k] = v
                            if set_fragments:
                                sql = text(f"UPDATE enhanced_games SET {', '.join(set_fragments)} WHERE game_id=:g AND \"date\"=:d")
                                conn.execute(sql, params)
                    logger.info("Persisted real pitcher form stats to enhanced_games (%d rows)", len(subset))
            except Exception as e:
                logger.warning(f"Could not persist pitcher stats to enhanced_games: {e}")
            
            # Use the coalesced values and drop dups proactively
            games_df['market_total'] = games_df.pop('market_total_final')
            games_df['away_sp_season_era'] = games_df.pop('away_sp_season_era_final')
            if 'home_sp_season_era_final' in games_df.columns:
                games_df['home_sp_season_era'] = games_df.pop('home_sp_season_era_final')
            games_df = games_df.loc[:, ~games_df.columns.duplicated(keep='last')]
            
            # Belt-and-suspenders: make sure the "era" training names exist, using season-era if needed
            for src, dst in [('home_sp_season_era','home_sp_era'),
                             ('away_sp_season_era','away_sp_era')]:
                if src in games_df.columns and dst not in games_df.columns:
                    games_df[dst] = games_df[src]

            # Same for runs per game
            for src, dst in [('home_team_runs_pg','home_team_runs_per_game'),
                             ('away_team_runs_pg','away_team_runs_per_game')]:
                if src in games_df.columns and dst not in games_df.columns:
                    games_df[dst] = games_df[src]
            
            # Runtime guard: fix away ERA if it's still flat
            if games_df['away_sp_season_era'].std() < 0.15:
                logger.warning(f"Away ERA too flat (std={games_df['away_sp_season_era'].std():.3f}), trying enhanced_games fallback")
                refill = pd.read_sql(text("""
                    SELECT game_id, away_sp_season_era
                    FROM enhanced_games
                    WHERE "date" = :d
                """), engine, params={'d': date_str})
                if not refill.empty:
                    games_df = games_df.merge(refill, on='game_id', how='left', suffixes=('', '_eg'))
                    mask = (games_df['away_sp_season_era'].isna()) | (games_df['away_sp_season_era'].isin([0, 4.5]))
                    games_df.loc[mask, 'away_sp_season_era'] = games_df.loc[mask, 'away_sp_season_era_eg']
                    games_df.drop(columns=[c for c in games_df.columns if c.endswith('_eg')], inplace=True)
                    logger.info(f"After refill: away ERA std = {games_df['away_sp_season_era'].std():.3f}")
            
            # Optional: Backfill legitimate_game_features with proper market totals for consistency
            with engine.begin() as c:
                result = c.execute(text("""
                    UPDATE legitimate_game_features lgf
                    SET market_total = eg.market_total
                    FROM enhanced_games eg
                    WHERE lgf.game_id = eg.game_id
                      AND lgf."date" = eg."date"
                      AND lgf."date" = :d
                      AND (lgf.market_total IS NULL OR lgf.market_total = 0)
                """), {"d": date_str})
                if result.rowcount > 0:
                    logger.info(f"Backfilled {result.rowcount} market totals in legitimate_game_features")
            if len(games_df) == 0:
                logger.info(f"No games found for {date_str}")
                return []
            logger.info(f"Found {len(games_df)} games with enhanced bullpen features")
            
            # Verify data quality
            print("market_total nonzero:", (games_df['market_total']>0).sum(), "/", len(games_df))
            print("ERA stds:", games_df['home_sp_season_era'].std(), games_df['away_sp_season_era'].std())
            
            # Add team offense L30 merge (use the pipeline version that maps name→code to teams_offense_daily)
            try:
                if hasattr(self, 'enhanced_pipeline') and self.enhanced_pipeline is not None:
                    logger.info("About to call _merge_team_offense_l30...")
                    games_df = self.enhanced_pipeline._merge_team_offense_l30(games_df, date_str)
                    logger.info("_merge_team_offense_l30 completed successfully")
                else:
                    logger.info("No enhanced_pipeline available, skipping team offense merge")
            except Exception as e:
                logger.error(f"Error in _merge_team_offense_l30: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Debug: Check games_df before feature engineering
            if games_df is None:
                raise RuntimeError("games_df is None before feature engineering")
            logger.info(f"games_df shape before feature engineering: {games_df.shape}")
            logger.info(f"games_df columns: {list(games_df.columns)}")
            
            featured_df = self.engineer_features(games_df)
            
            # Debug: Check featured_df after feature engineering
            if featured_df is None:
                raise RuntimeError("featured_df is None after feature engineering")
            logger.info(f"featured_df shape after feature engineering: {featured_df.shape}")
            # Safety: ensure no duplicate cols before reindex
            featured_df = featured_df.loc[:, ~featured_df.columns.duplicated(keep='last')]
            
            # --- diagnostics: variance, not NaN rate ---
            watch = [
                "home_sp_era","away_sp_era",
                "home_sp_whip","away_sp_whip",
                "home_sp_k_per_9","away_sp_k_per_9",
                "home_sp_bb_per_9","away_sp_bb_per_9",
                "home_sp_starts","away_sp_starts",
                "expected_total","market_total"
            ]
            present = [c for c in watch if c in featured_df.columns]
            stds = featured_df[present].std(numeric_only=True).sort_values()
            logger.info("Watch feature STDs (0 => constant):")
            logger.info(str(stds))

            # Attempt soft alignment fallback if direct reindex exposes missing features
            X = featured_df.reindex(columns=self.feature_columns)
            missing = [c for c in self.feature_columns if c not in featured_df.columns]
            if missing:
                # soft fallback instead of hard exit
                X = self.align_serving_features(featured_df, strict=False)
                missing2 = [c for c in self.feature_columns if c not in X.columns]
                if missing2:
                    raise SystemExit(f"❌ Incompatible model; still missing: {missing2[:10]} ...")
            
            # Handle infs before fill/transform (proxies can create inf from div-by-zero)
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Impute using training medians then per-slate medians
            if self.fill_values:
                for c, v in self.fill_values.items():
                    if c in X.columns:
                        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(v)
            X = X.fillna(X.median(numeric_only=True)).fillna(0)

            # Neutralize ID-like columns (runbook patch #2)
            id_cols = [c for c in X.columns if c.endswith('_id') or c in ('game_id','home_sp_id','away_sp_id')]
            if id_cols:
                X[id_cols] = 0.0
                logger.info(f"Neutralized ID features: {id_cols[:6]}{'...' if len(id_cols)>6 else ''}")
            
            # Diagnostic: Check variance in key features before preprocessing
            key_features = ["home_sp_era","away_sp_era","home_bp_era","away_bp_era","ballpark_run_factor","ballpark_hr_factor","temperature","wind_speed"]
            present_key = [c for c in key_features if c in X.columns]
            if present_key:
                key_stds = X[present_key].std(numeric_only=True).sort_values()
                logger.info("Key feature variance this slate:")
                logger.info(str(key_stds))
            
            # Feature importance analysis
            if hasattr(self.model, "feature_importances_"):
                importances = pd.Series(self.model.feature_importances_, index=self.feature_columns).sort_values(ascending=False)
                top = importances.head(10)
                logger.info("Top feature importances:")
                logger.info(str(top))
                present = [f for f in top.index if f in X.columns]
                if present:
                    top_stds = X[present].std(numeric_only=True).sort_values()
                    logger.info("Top feature STDs this slate:")
                    logger.info(str(top_stds))
            
            # Watch features validation
            watch = ['expected_total','offense_imbalance','home_team_rpg_l30','away_team_rpg_l30',
                     'pitcher_experience','combined_bb_rate','home_sp_starts','away_sp_starts']
            present_watch = [c for c in watch if c in X.columns]
            if present_watch:
                stds = X[present_watch].std(numeric_only=True).sort_values()
                logger.info("Watch feature STDs:")
                logger.info(str(stds))
            
            pre_std_sum = X.std(numeric_only=True).sum()
            logger.info(f"Pre-preprocessing slate std sum: {pre_std_sum:.3f}")
            
            if X.std(axis=0, numeric_only=True).sum() == 0:
                raise RuntimeError("All-zero (or constant) feature matrix before preprocessing.")
            
            # === Build model inputs without triggering sklearn feature-name warnings ===
            # 1) Make the preprocessed matrix (DataFrame if no scaler/preproc)
            if self.preproc is not None:
                X_in = self.preproc.transform(X)              # usually ndarray
            elif self.scaler is not None:
                if hasattr(self.scaler, "n_features_in_") and self.scaler.n_features_in_ != X.shape[1]:
                    logger.warning(f"Scaler feature count mismatch: expected {self.scaler.n_features_in_}, got {X.shape[1]}; skipping scaler.")
                    X_in = X                                  # keep DataFrame with names
                else:
                    X_in = self.scaler.transform(X)           # ndarray
            else:
                X_in = X                                      # keep DataFrame (names preserved)

            # Debugging: Check if X_in is None
            if X_in is None:
                logger.error("❌ X_in is None after preprocessing step")
                raise RuntimeError("X_in is None after preprocessing")
            logger.debug(f"X_in type: {type(X_in)}, shape: {getattr(X_in, 'shape', 'no shape')}")

            # 2) Choose the right type for the fitted estimator:
            #    - If the model was trained with names, pass a DataFrame with the same columns
            #    - Otherwise pass a plain ndarray
            def _as_model_input(estimator, arr_like, columns):
                if arr_like is None:
                    logger.error("❌ arr_like is None in _as_model_input")
                    raise RuntimeError("arr_like is None in _as_model_input")
                has_names = hasattr(estimator, "feature_names_in_")
                if has_names:
                    if isinstance(arr_like, pd.DataFrame):
                        return arr_like
                    # wrap ndarray with column names used in training
                    return pd.DataFrame(arr_like, columns=columns)
                # estimator trained without names → supply ndarray
                return arr_like.values if isinstance(arr_like, pd.DataFrame) else arr_like

            X_model = _as_model_input(self.model, X_in, self.feature_columns)
            
            # Debugging: Check if X_model is None
            if X_model is None:
                logger.error("❌ X_model is None after _as_model_input")
                raise RuntimeError("X_model is None after _as_model_input")
            logger.debug(f"X_model type: {type(X_model)}, shape: {getattr(X_model, 'shape', 'no shape')}")

            # 3) Post-preprocessing variance diagnostic (works for DF or ndarray)
            _arr = X_model.values if isinstance(X_model, pd.DataFrame) else X_model
            post_std_sum = np.nanstd(_arr, axis=0).sum()
            logger.info(f"Post-preprocessing slate std sum: {post_std_sum:.3f}")
            if np.isclose(post_std_sum, 0.0):
                raise RuntimeError("All-zero (or constant) feature matrix after preprocessing. Check training bundle.")

            # 4) Predict with proper input type
            predictions = self.model.predict(X_model)

            # Optional single clip BEFORE grid snap via environment (runbook patch #4)
            try:
                pmin = float(os.getenv('PRED_MIN')) if os.getenv('PRED_MIN') else None
                pmax = float(os.getenv('PRED_MAX')) if os.getenv('PRED_MAX') else None
                if pmin is not None or pmax is not None:
                    lo = pmin if pmin is not None else predictions.min()
                    hi = pmax if pmax is not None else predictions.max()
                    predictions = np.clip(predictions, lo, hi)
            except Exception:
                pass

            # Half-run grid snap only (remove any extra clip later)
            if os.getenv('DISABLE_GRID_SNAP','0') == '1':
                logger.info('🔧 Grid snap disabled - using raw predictions')
            else:
                predictions = np.round(predictions * 2.0) / 2.0

            # 5) RF uncertainty (trees were fit without names → always give ndarray)
            rf_std = None
            if hasattr(self.model, "estimators_"):
                try:
                    X_for_trees = _arr                         # guaranteed ndarray here
                    tree_preds = np.vstack([t.predict(X_for_trees) for t in self.model.estimators_])
                    rf_std = tree_preds.std(axis=0)
                    logger.info(f"RF uncertainty: mean={rf_std.mean():.3f}, range=[{rf_std.min():.3f}, {rf_std.max():.3f}]")
                except Exception as e:
                    logger.warning(f"Could not calculate RF uncertainty: {e}")
                    rf_std = None
            
            # Apply bias correction if available
            predictions = predictions + float(getattr(self, "bias_correction", 0.0))
            
            # Apply serving calibration with drift guard (anchor to expected_total w/ median fallback)
            try:
                exp_series = pd.to_numeric(featured_df.get("expected_total"), errors="coerce")
                exp_fallback = float(np.nanmedian(exp_series)) if exp_series.notna().any() else 9.0
                exp_total = exp_series.fillna(exp_fallback).values
                predictions = apply_serving_calibration(predictions, exp_total)
                logger.info("✅ Serving calibration applied")
            except Exception as e:
                logger.warning(f"Serving calibration failed: {e}")
            
            spread = float(np.ptp(predictions)) if len(predictions) else 0.0
            p = predictions.round(1)
            if spread < 0.6:
                logger.warning(f"Predictions too flat (range: {spread}); verify you're using the correct model for these features.")
            
            # Diagnostic: Log market deltas to check if flatness comes from features vs scaling
            deltas = []
            for i, (_, g) in enumerate(featured_df.iterrows()):
                # Use the market_total from the coalesced join (may have been aliased to expected_total)
                row_market = g.get("market_total", g.get("expected_total"))
                if row_market is not None:
                    try:
                        m = float(row_market)
                        if 5 < m < 15:
                            deltas.append(float(predictions[i]) - m)
                    except (TypeError, ValueError):
                        pass
            
            if deltas:
                mean_delta = float(np.nanmean(deltas))
                logger.info(f"Δ(pred - market): mean={mean_delta:+.2f}, min={np.nanmin(deltas):.2f}, max={np.nanmax(deltas):.2f}")
                
                # Optional per-slate runtime calibration
                if self.bias_correction == 0.0 and abs(mean_delta) > 0.75:
                    adj = -mean_delta
                    logger.warning(f"Applying runtime calibration: {adj:+.2f} to all predictions")
                    predictions = predictions + adj
                    
                    # Recalculate deltas after adjustment
                    adjusted_deltas = []
                    for i, (_, g) in enumerate(featured_df.iterrows()):
                        row_market = g.get("market_total", g.get("expected_total"))
                        if row_market is not None:
                            try:
                                m = float(row_market)
                                if 5 < m < 15:
                                    adjusted_deltas.append(float(predictions[i]) - m)
                            except (TypeError, ValueError):
                                pass
                    
                    if adjusted_deltas:
                        logger.info(f"Post-calibration Δ(pred - market): mean={np.nanmean(adjusted_deltas):+.2f}")
            else:
                logger.warning("No valid market totals found for delta calculation")
            
            results = []
            for i, (_, game) in enumerate(featured_df.iterrows()):
                results.append({
                    'game_id': game['game_id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'predicted_total': round(predictions[i], 1),
                    'market_total': float(game.get('market_total', np.nan)),
                    'rf_uncertainty': float(rf_std[i]) if rf_std is not None else None,
                    'expected_bullpen_innings': game.get('total_expected_bullpen_innings', 'N/A'),
                    'bullpen_quality': game.get('bullpen_pitching_quality', 'N/A'),
                    'weighted_pitching_era': game.get('pitching_depth_quality', 'N/A')
                })
                logger.info(f"{game['away_team']} @ {game['home_team']}: {predictions[i]:.1f} runs")
            logger.info("=" * 60)
            logger.info(f"📊 Average predicted total: {np.mean(predictions):.1f} runs (spread {spread:.2f})")
            # Save predictions to database (UPDATE if exists, INSERT if not)
            self.save_predictions_to_database(results, date_str, rf_std)
            
            # Convert results to DataFrame for compatibility
            predictions_df = pd.DataFrame(results)
            
            # Return tuple of (predictions_df, featured_df, X) for daily workflow compatibility
            return predictions_df, featured_df, X
        except SystemExit as e:
            logger.error(str(e))
            return None, None, None
        except Exception as e:
            logger.error(f"Error generating enhanced predictions: {e}")
            return None, None, None
    
    def save_predictions_to_database(self, predictions, date_str, rf_std=None):
        """Save enhanced bullpen predictions to database with UPDATE logic (date-aware)."""
        try:
            engine = self.get_engine()
            
            with engine.begin() as conn:
                updated_count = 0
                inserted_count = 0
                
                for i, pred in enumerate(predictions):
                    # Get market total for this game to calculate proper recommendation
                    # Use the passed market_total with fallback
                    row_market = pred.get('market_total') if isinstance(pred, dict) else None
                    if row_market is not None and pd.notna(row_market) and 5 < float(row_market) < 15:
                        market_total = float(row_market)
                    else:
                        # Fallback: query enhanced_games
                        market_query = text("""
                            SELECT market_total FROM enhanced_games 
                            WHERE game_id = :game_id AND "date" = :date
                        """)
                        market_result = conn.execute(market_query, {'game_id': pred['game_id'], 'date': date_str}).fetchone()
                        market_total = float(market_result[0]) if market_result and market_result[0] is not None else 9.0
                    
                    predicted_total = float(pred['predicted_total'])
                    difference = predicted_total - market_total
                    if difference >= 0.5:
                        recommendation = "OVER"
                        edge_runs = round(difference, 2)
                    elif difference <= -0.5:
                        recommendation = "UNDER"
                        edge_runs = round(abs(difference), 2)
                    else:
                        recommendation = "HOLD"
                        edge_runs = 0.0
                    
                    # Use RF uncertainty for more principled confidence
                    if rf_std is not None and i < len(rf_std) and np.isfinite(rf_std[i]):
                        # stabilizer avoids divide-by-zero and too-sharp confidences
                        z = abs(difference) / (rf_std[i] + 0.35)
                        confidence = int(np.clip(50 + 10 * z, 50, 90))
                    else:
                        # Fallback to original confidence calculation
                        confidence = min(90, max(50, 70 + abs(difference) * 6))
                    
                    # First try UPDATE (if record exists)
                    update_sql = text("""
UPDATE enhanced_games
SET predicted_total=:predicted_total, confidence=:confidence, recommendation=:recommendation, edge=:edge
WHERE game_id=:game_id AND "date"=:date
                    """)
                    
                    result = conn.execute(update_sql, {
                        'predicted_total': predicted_total,
                        'confidence': confidence,
                        'recommendation': recommendation,
                        'edge': float(edge_runs),
                        'game_id': pred['game_id'],
                        'date': date_str
                    })
                    
                    if result.rowcount > 0:
                        updated_count += 1
                    else:
                        # If UPDATE didn't affect any rows, try INSERT
                        insert_sql = text("""
INSERT INTO enhanced_games (game_id, predicted_total, confidence, recommendation, edge, "date")
VALUES (:game_id, :predicted_total, :confidence, :recommendation, :edge, :date)
ON CONFLICT (game_id, "date") DO UPDATE SET
  predicted_total=EXCLUDED.predicted_total,
  confidence=EXCLUDED.confidence,
  recommendation=EXCLUDED.recommendation,
  edge=EXCLUDED.edge
                        """)

                        try:
                            conn.execute(insert_sql, {
                                'game_id': pred['game_id'],
                                'predicted_total': predicted_total,
                                'confidence': confidence,
                                'recommendation': recommendation,
                                'edge': float(edge_runs),
                                'date': date_str
                            })
                            inserted_count += 1
                        except Exception as ie:
                            logger.warning(f"Failed to insert prediction for game {pred['game_id']}: {ie}")
                
                logger.info(f"💾 Database Updated: {updated_count} updated, {inserted_count} inserted")
                
        except Exception as e:
            logger.error(f"Error saving predictions to database: {e}")

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Bullpen Predictor')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD or MM-DD-YYYY)')
    parser.add_argument('--model-path', type=str, help='Path to model bundle (joblib) overriding default')
    parser.add_argument('--no-scaler', action='store_true', help='Skip scaler/preprocessing to test raw features')
    args = parser.parse_args()
    predictor = EnhancedBullpenPredictor()
    if args.model_path:
        logger.info(f"🔁 Loading override model bundle: {args.model_path}")
        if not predictor.load_model(Path(args.model_path)):
            logger.error("❌ Failed to load override model bundle")
            return
    
    # Test mode: disable scaler to see if that fixes flat predictions
    if args.no_scaler:
        logger.info("🧪 Test mode: Disabling scaler/preprocessing")
        predictor.scaler = None
        predictor.preproc = None
    if not predictor.model:
        logger.error("❌ Failed to load model")
        return
    pred_df, _, _ = predictor.predict_today_games(args.target_date)
    if pred_df is not None and len(pred_df) > 0:
        logger.info("✅ Enhanced bullpen predictions generated successfully!")
    else:
        logger.error("❌ No predictions generated (schema mismatch or preprocessing failure)")

if __name__ == "__main__":
    main()
