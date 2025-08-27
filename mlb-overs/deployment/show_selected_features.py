#!/usr/bin/env python3
"""
🔍 FEATURE SELECTION ANALYZER
=============================
🎯 Show which specific features are selected in top 120
🧠 Analyze ultra feature selection process
=============================
"""

import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

warnings.filterwarnings('ignore')

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

def get_engine():
    """Get database engine for YOUR PostgreSQL database"""
    url = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def create_feature_map():
    """Create a mapping of feature indices to feature names"""
    
    feature_map = {}
    idx = 0
    
    # 1. BASIC GAME FEATURES (20)
    basic_features = [
        'market_total', 'temperature', 'wind_speed', 'is_day_game', 'park_run_factor',
        'park_hr_factor', 'market_total_squared', 'market_total_cubed', 'log_market_total',
        'inv_market_total', 'high_total_flag', 'low_total_flag', 'market_temp_interaction',
        'market_park_run', 'market_park_hr', 'temp_wind', 'temp_squared_norm',
        'sin_market_total', 'cos_market_total', 'market_deviation'
    ]
    
    for name in basic_features:
        feature_map[idx] = f"BASIC_{idx:03d}_{name}"
        idx += 1
    
    # 2. TEMPORAL FEATURES (25)
    temporal_features = [
        'day_of_year', 'weekday', 'month', 'day', 'sin_day_of_year', 'cos_day_of_year',
        'sin_weekday', 'cos_weekday', 'weekend_flag', 'summer_flag', 'spring_fall_flag',
        'month_norm', 'day_norm', 'days_from_season_start', 'august_flag', 'monday_flag',
        'sunday_flag', 'weekday_month', 'weekday_deviation', 'day_of_year_squared',
        'log_day_of_year', 'early_month_flag', 'month_market_interaction', 
        'day_of_year_market_interaction', 'weekday_park_interaction'
    ]
    
    for name in temporal_features:
        feature_map[idx] = f"TEMPORAL_{idx:03d}_{name}"
        idx += 1
    
    # 3. ENHANCED PITCHER FEATURES (80)
    
    # Basic pitcher features (20)
    basic_pitcher = [
        'home_sp_era', 'away_sp_era', 'home_sp_whip', 'away_sp_whip',
        'home_sp_er', 'away_sp_er', 'home_sp_ip', 'away_sp_ip',
        'home_sp_k', 'away_sp_k', 'home_sp_bb', 'away_sp_bb',
        'home_sp_rest', 'away_sp_rest', 'home_bp_era', 'away_bp_era',
        'combined_era', 'era_difference', 'combined_whip', 'whip_difference'
    ]
    
    for name in basic_pitcher:
        feature_map[idx] = f"PITCHER_BASIC_{idx:03d}_{name}"
        idx += 1
    
    # Advanced pitcher analytics (30)
    advanced_pitcher = [
        'home_sp_k9', 'away_sp_k9', 'home_sp_bb9', 'away_sp_bb9',
        'home_sp_kbb', 'away_sp_kbb', 'combined_k9', 'combined_bb9',
        'combined_kbb', 'k9_difference', 'bb9_difference', 'kbb_difference',
        'home_era_whip_product', 'away_era_whip_product', 'average_era',
        'geometric_mean_era', 'poor_home_pitcher_flag', 'poor_away_pitcher_flag',
        'elite_home_pitcher_flag', 'elite_away_pitcher_flag', 'home_era_vs_market',
        'away_era_vs_market', 'home_rest_norm', 'away_rest_norm',
        'home_well_rested_flag', 'away_well_rested_flag', 'home_short_rest_flag',
        'away_short_rest_flag', 'home_expected_innings', 'away_expected_innings'
    ]
    
    for name in advanced_pitcher:
        feature_map[idx] = f"PITCHER_ADV_{idx:03d}_{name}"
        idx += 1
    
    # Pitcher-bullpen integration (15)
    bullpen_integration = [
        'total_home_pitching', 'total_away_pitching', 'overall_pitching_avg',
        'bullpen_interaction', 'weak_home_bullpen_flag', 'weak_away_bullpen_flag',
        'home_starter_vs_bullpen', 'away_starter_vs_bullpen', 'cross_pitching',
        'exponential_quality', 'inverse_era_sum', 'home_whip_bullpen_cross',
        'away_whip_bullpen_cross', 'worst_pitcher', 'best_pitcher'
    ]
    
    for name in bullpen_integration:
        feature_map[idx] = f"PITCHER_BP_{idx:03d}_{name}"
        idx += 1
    
    # Weather-pitcher interactions (15)
    weather_pitcher = [
        'home_era_temp', 'away_era_temp', 'home_era_park', 'away_era_park',
        'home_era_wind', 'away_era_wind', 'home_whip_temp', 'away_whip_temp',
        'home_k9_daynight', 'away_k9_daynight', 'home_bb9_park', 'away_bb9_park',
        'combined_era_temp', 'combined_whip_wind', 'combined_kbb_park'
    ]
    
    for name in weather_pitcher:
        feature_map[idx] = f"PITCHER_WEATHER_{idx:03d}_{name}"
        idx += 1
    
    # 4. TEAM OFFENSE FEATURES (80 per team = 160 total... but let's map what we can)
    
    # Home team offense features (41)
    home_offense_basic = [
        'home_wrcplus', 'home_woba', 'home_iso', 'home_bb_pct', 'home_k_pct',
        'home_runs_l5', 'home_runs_l10', 'home_ops', 'home_volatility',
        'home_max_runs', 'home_min_runs', 'home_avg_runs_pg', 'home_avg_ba',
        'home_avg_obp', 'home_avg_slg', 'home_avg_sb', 'home_avg_babip',
        'home_runs_ratio'
    ]
    
    home_offense_advanced = [
        'home_wrcplus_norm', 'home_woba_scaled', 'home_iso_scaled', 'home_bb_k_ratio',
        'home_ops_squared', 'home_runs_wrc', 'home_volatility_ratio', 'home_runs_range',
        'home_runs_sb_product', 'home_obp_slg_sum', 'home_obp_slg_product',
        'home_ba_scaled', 'home_woba_iso_sqrt', 'home_woba_obp_ratio',
        'home_iso_slg_ratio', 'home_runs_squared', 'home_log_runs',
        'home_elite_offense_flag', 'home_poor_offense_flag', 'home_volatile_flag',
        'home_runs_vs_market', 'home_ops_park', 'home_wrcplus_temp'
    ]
    
    for name in home_offense_basic + home_offense_advanced:
        feature_map[idx] = f"HOME_OFFENSE_{idx:03d}_{name}"
        idx += 1
    
    # Away team offense features (41)
    away_offense_basic = [
        'away_wrcplus', 'away_woba', 'away_iso', 'away_bb_pct', 'away_k_pct',
        'away_runs_l5', 'away_runs_l10', 'away_ops', 'away_volatility',
        'away_max_runs', 'away_min_runs', 'away_avg_runs_pg', 'away_avg_ba',
        'away_avg_obp', 'away_avg_slg', 'away_avg_sb', 'away_avg_babip',
        'away_runs_ratio'
    ]
    
    away_offense_advanced = [
        'away_wrcplus_norm', 'away_woba_scaled', 'away_iso_scaled', 'away_bb_k_ratio',
        'away_ops_squared', 'away_runs_wrc', 'away_volatility_ratio', 'away_runs_range',
        'away_runs_sb_product', 'away_obp_slg_sum', 'away_obp_slg_product',
        'away_ba_scaled', 'away_woba_iso_sqrt', 'away_woba_obp_ratio',
        'away_iso_slg_ratio', 'away_runs_squared', 'away_log_runs',
        'away_elite_offense_flag', 'away_poor_offense_flag', 'away_volatile_flag',
        'away_runs_vs_market', 'away_ops_park', 'away_wrcplus_temp'
    ]
    
    for name in away_offense_basic + away_offense_advanced:
        feature_map[idx] = f"AWAY_OFFENSE_{idx:03d}_{name}"
        idx += 1
    
    # 5. TEAM INTERACTION FEATURES (30)
    interaction_features = [
        'combined_wrcplus', 'wrcplus_difference', 'wrcplus_product', 'combined_runs_l5',
        'runs_l5_difference', 'runs_l5_product', 'runs_l5_geometric_mean',
        'total_runs_vs_market', 'max_wrcplus', 'min_wrcplus', 'combined_ops',
        'ops_difference', 'ops_product', 'elite_offense_both_flag',
        'poor_offense_both_flag', 'unbalanced_offense_flag', 'combined_runs_squared',
        'home_away_wrcplus_ratio', 'away_home_wrcplus_ratio', 'log_combined_runs',
        'combined_runs_market_product', 'runs_over_market_flag', 'runs_under_market_flag',
        'home_ops_away_runs', 'away_ops_home_runs', 'average_wrcplus',
        'ops_market_product', 'home_away_runs_ratio', 'away_home_runs_ratio',
        'weighted_offense_cross'
    ]
    
    for name in interaction_features:
        feature_map[idx] = f"INTERACTION_{idx:03d}_{name}"
        idx += 1
    
    # 6. HISTORICAL MOMENTUM FEATURES (40)
    home_momentum = [
        'home_avg_totals', 'home_std_totals', 'home_avg_overs', 'home_avg_totals_l5',
        'home_avg_overs_l5', 'home_trend_slope', 'home_max_totals', 'home_min_totals',
        'home_avg_markets', 'home_avg_differential', 'home_over_rate',
        'home_avg_totals_l3', 'home_hot_streak_flag', 'home_over_market_flag',
        'home_games_count', 'home_totals_vs_market', 'home_totals_correlation',
        'home_recent_over_rate', 'home_totals_squared', 'home_log_totals'
    ]
    
    away_momentum = [
        'away_avg_totals', 'away_std_totals', 'away_avg_overs', 'away_avg_totals_l5',
        'away_avg_overs_l5', 'away_trend_slope', 'away_max_totals', 'away_min_totals',
        'away_avg_markets', 'away_avg_differential', 'away_over_rate',
        'away_avg_totals_l3', 'away_hot_streak_flag', 'away_over_market_flag',
        'away_games_count', 'away_totals_vs_market', 'away_totals_correlation',
        'away_recent_over_rate', 'away_totals_squared', 'away_log_totals'
    ]
    
    for name in home_momentum + away_momentum:
        feature_map[idx] = f"MOMENTUM_{idx:03d}_{name}"
        idx += 1
    
    # 7. ADVANCED DATABASE FEATURES (60+)
    
    # Home team extended stats (18)
    home_extended = [
        'home_runs_l7', 'home_runs_l20', 'home_runs_l30', 'home_ra_l7', 'home_ra_l20',
        'home_ra_l30', 'home_avg_l7', 'home_avg_l20', 'home_avg_l30', 'home_obp_l7',
        'home_obp_l20', 'home_obp_l30', 'home_slg_l7', 'home_slg_l20', 'home_slg_l30',
        'home_ops_l7', 'home_ops_l20', 'home_ops_l30'
    ]
    
    # Away team extended stats (18)
    away_extended = [
        'away_runs_l7', 'away_runs_l20', 'away_runs_l30', 'away_ra_l7', 'away_ra_l20',
        'away_ra_l30', 'away_avg_l7', 'away_avg_l20', 'away_avg_l30', 'away_obp_l7',
        'away_obp_l20', 'away_obp_l30', 'away_slg_l7', 'away_slg_l20', 'away_slg_l30',
        'away_ops_l7', 'away_ops_l20', 'away_ops_l30'
    ]
    
    # Multi-timeframe derivatives (24)
    timeframe_derivatives = [
        'combined_runs_l7', 'combined_runs_l20', 'combined_runs_l30',
        'runs_diff_l7', 'runs_diff_l20', 'runs_diff_l30',
        'combined_ops_l7', 'combined_ops_l20', 'combined_ops_l30',
        'home_recent_trend', 'away_recent_trend', 'l7_vs_market', 'l20_vs_market',
        'l30_vs_market', 'ops_interaction_l7', 'ops_interaction_l20', 'ops_interaction_l30',
        'home_avg_timeframes', 'away_avg_timeframes', 'home_volatility_timeframes',
        'away_volatility_timeframes', 'home_range_timeframes', 'away_range_timeframes',
        'park_adjusted_total'
    ]
    
    for name in home_extended + away_extended + timeframe_derivatives:
        feature_map[idx] = f"DATABASE_{idx:03d}_{name}"
        idx += 1
    
    # 8. ULTRA CROSS-DOMAIN INTERACTIONS (30)
    ultra_interactions = [
        'home_era_runs_temp', 'away_era_runs_temp', 'home_whip_ops_park',
        'away_whip_ops_park', 'combined_era_runs_market', 'home_k9_avg_wind',
        'away_k9_avg_wind', 'home_bp_ops_time', 'away_bp_ops_time',
        'market_rest_interaction', 'temp_slg_l7', 'wind_obp_l7', 'park_era_runs',
        'geometric_pitcher_quality', 'cube_root_ops', 'seasonal_strikeouts',
        'temp_adjusted_walks', 'runs_product_market', 'ops_era_interaction',
        'quarterly_seasonality', 'temp_deviation_impact', 'wind_pitching_impact',
        'park_market_ops', 'daynight_avg_interaction', 'home_control_offense_seasonal',
        'away_control_offense_seasonal', 'gaussian_pitcher_quality',
        'log_combined_runs_market', 'power_scaling_runs', 'unbalanced_quality_market'
    ]
    
    for name in ultra_interactions:
        feature_map[idx] = f"ULTRA_{idx:03d}_{name}"
        idx += 1
    
    # 9. FINAL DERIVED FEATURES (15)
    final_features = [
        'feature_count', 'avg_feature_value', 'feature_volatility', 'complexity_factor',
        'positive_feature_avg', 'high_value_count', 'probability_count',
        'cube_root_market', 'environmental_avg', 'triple_interaction',
        'decay_function', 'seasonal_market', 'pitcher_quality_ratio',
        'log_complexity', 'ultra_feature_indicator'
    ]
    
    for name in final_features:
        feature_map[idx] = f"FINAL_{idx:03d}_{name}"
        idx += 1
    
    return feature_map

def analyze_feature_selection():
    """Analyze which features are selected by the ultra system"""
    
    print("🔍 FEATURE SELECTION ANALYZER")
    print("=============================")
    
    # Import the ultra system
    import sys
    import importlib.util
    
    # Load the ultra system module directly
    ultra_path = 's:/Projects/AI_Predictions/mlb-overs/training/ultra_80_percent_system.py'
    
    try:
        spec = importlib.util.spec_from_file_location("ultra_80_percent_system", ultra_path)
        ultra_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ultra_module)
        
        get_historical_season_data = ultra_module.get_historical_season_data
        create_ultra_features = ultra_module.create_ultra_features
        Ultra80PercentSystem = ultra_module.Ultra80PercentSystem
        
    except Exception as e:
        print(f"❌ Could not import ultra system: {e}")
        return
    
    # Load data
    print("📊 Loading data from YOUR database...")
    df, dates = get_historical_season_data()
    
    if len(dates) < 30:
        print("❌ Not enough data for analysis")
        return
    
    # Prepare sample data for feature selection analysis
    print("🎯 Creating sample features for analysis...")
    
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    all_features = []
    all_targets = []
    
    for _, game in sample_df.iterrows():
        try:
            features, feature_count = create_ultra_features(
                df, game['date'], game, window_size=30
            )
            all_features.append(features)
            all_targets.append(game['total_runs'])
        except Exception as e:
            print(f"   ⚠️ Error creating features for game: {e}")
            continue
    
    if len(all_features) < 10:
        print("❌ Not enough valid features created")
        return
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_targets)
    
    print(f"📈 Analysis dataset: {X.shape[0]} games, {X.shape[1]} features")
    
    # Create feature mapping
    feature_map = create_feature_map()
    
    # Initialize ultra system and run feature selection
    ultra_system = Ultra80PercentSystem()
    ultra_system.create_ultra_ensemble()
    
    # Run ultra feature selection
    if X.shape[1] > 120:
        print(f"\n🧠 Running ultra feature selection: {X.shape[1]} → 120 features")
        
        try:
            # Multi-method ultra selection (from ultra_80_percent_system.py)
            
            # Method 1: F-statistic (40% weight)
            f_selector = SelectKBest(score_func=f_regression, k=min(120, X.shape[1]))
            f_selector.fit(X, y)
            f_scores = f_selector.scores_
            
            # Method 2: Mutual information (40% weight)
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Method 3: Variance-based (10% weight)
            var_scores = np.var(X, axis=0)
            
            # Method 4: Correlation with target (10% weight)
            corr_scores = np.abs([np.corrcoef(X[:, i], y)[0, 1] if len(np.unique(X[:, i])) > 1 else 0 
                                 for i in range(X.shape[1])])
            
            # Normalize all scores
            f_scores_norm = f_scores / (np.max(f_scores) + 1e-8)
            mi_scores_norm = mi_scores / (np.max(mi_scores) + 1e-8)
            var_scores_norm = var_scores / (np.max(var_scores) + 1e-8)
            corr_scores_norm = corr_scores / (np.max(corr_scores) + 1e-8)
            
            # Ultra-weighted combination
            ultra_scores = (
                0.4 * f_scores_norm +
                0.4 * mi_scores_norm +
                0.1 * var_scores_norm +
                0.1 * corr_scores_norm
            )
            
            # Select top features
            selected_indices = np.argsort(ultra_scores)[-120:]
            selected_indices = sorted(selected_indices)  # Sort for better readability
            
            print(f"\n🏆 TOP 120 SELECTED FEATURES:")
            print("=" * 80)
            
            # Group features by category
            categories = {
                'BASIC': [],
                'TEMPORAL': [],
                'PITCHER_BASIC': [],
                'PITCHER_ADV': [],
                'PITCHER_BP': [],
                'PITCHER_WEATHER': [],
                'HOME_OFFENSE': [],
                'AWAY_OFFENSE': [],
                'INTERACTION': [],
                'MOMENTUM': [],
                'DATABASE': [],
                'ULTRA': [],
                'FINAL': []
            }
            
            for idx in selected_indices:
                feature_name = feature_map.get(idx, f"UNKNOWN_{idx:03d}")
                category = feature_name.split('_')[0]
                
                if category in categories:
                    categories[category].append((idx, feature_name, ultra_scores[idx]))
                else:
                    categories['UNKNOWN'] = categories.get('UNKNOWN', [])
                    categories['UNKNOWN'].append((idx, feature_name, ultra_scores[idx]))
            
            # Display results by category
            total_selected = 0
            for category, features in categories.items():
                if features:
                    print(f"\n📊 {category} FEATURES ({len(features)} selected):")
                    print("-" * 60)
                    
                    # Sort by score within category
                    features.sort(key=lambda x: x[2], reverse=True)
                    
                    for idx, name, score in features:
                        print(f"  {idx:3d}: {name:<50} (Score: {score:.4f})")
                    
                    total_selected += len(features)
            
            print(f"\n📈 SELECTION SUMMARY:")
            print("=" * 50)
            print(f"Total features selected: {total_selected}/120")
            
            # Show category distribution
            category_counts = {cat: len(feats) for cat, feats in categories.items() if feats}
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_selected * 100
                print(f"  {category:<20}: {count:3d} features ({pct:5.1f}%)")
            
            # Show top 20 features overall
            print(f"\n🔥 TOP 20 HIGHEST SCORING FEATURES:")
            print("=" * 80)
            
            all_selected = []
            for category, features in categories.items():
                all_selected.extend(features)
            
            all_selected.sort(key=lambda x: x[2], reverse=True)
            
            for i, (idx, name, score) in enumerate(all_selected[:20]):
                print(f"  {i+1:2d}. [{idx:3d}] {name:<50} (Score: {score:.4f})")
            
        except Exception as e:
            print(f"❌ Feature selection analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"ℹ️ All {X.shape[1]} features would be selected (less than 120)")

if __name__ == "__main__":
    analyze_feature_selection()
