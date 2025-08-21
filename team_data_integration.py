#!/usr/bin/env python3
"""
Team Data Integration Strategy
=============================

IMMEDIATE ACTIONS to leverage team data for better predictions:

1. Fix team name mapping
2. Add recent team performance to enhanced_analysis.py  
3. Use hot/cold team trends in predictions
"""

def create_team_mapping():
    """Create team name standardization mapping"""
    
    # Based on the data we just analyzed
    team_mapping = {
        # Abbreviations to Full Names (matching enhanced_games format)
        'ATL': 'Atlanta Braves',
        'AZ': 'Arizona Diamondbacks', 
        'BAL': 'Baltimore Orioles',
        'BOS': 'Boston Red Sox',
        'CHC': 'Chicago Cubs',
        'CWS': 'Chicago White Sox',
        'CIN': 'Cincinnati Reds',
        'CLE': 'Cleveland Guardians',
        'COL': 'Colorado Rockies',
        'DET': 'Detroit Tigers',
        'HOU': 'Houston Astros',
        'KC': 'Kansas City Royals',
        'LAA': 'Los Angeles Angels',
        'LAD': 'Los Angeles Dodgers',
        'MIA': 'Miami Marlins',
        'MIL': 'Milwaukee Brewers',
        'MIN': 'Minnesota Twins',
        'NYM': 'New York Mets',
        'NYY': 'New York Yankees',
        'ATH': 'Oakland Athletics',
        'PHI': 'Philadelphia Phillies',
        'PIT': 'Pittsburgh Pirates',
        'SD': 'San Diego Padres',
        'SEA': 'Seattle Mariners',
        'SF': 'San Francisco Giants',
        'STL': 'St. Louis Cardinals',
        'TB': 'Tampa Bay Rays',
        'TEX': 'Texas Rangers',
        'TOR': 'Toronto Blue Jays',
        'WSH': 'Washington Nationals'
    }
    
    return team_mapping

def get_team_recent_performance(team_name, days=5):
    """Get recent team offensive performance"""
    
    # This would be implemented in enhanced_analysis.py
    query_template = f"""
    SELECT 
        AVG(runs_pg) as recent_runs_pg,
        AVG(woba) as recent_woba,
        AVG(wrcplus) as recent_wrcplus,
        COUNT(*) as games_played
    FROM teams_offense_daily 
    WHERE team = %s 
    AND date >= CURRENT_DATE - INTERVAL '{days} days'
    AND runs_pg IS NOT NULL
    """
    
    return query_template

def enhanced_prediction_logic():
    """Enhanced prediction logic using team data"""
    
    print("🚀 ENHANCED PREDICTION STRATEGY WITH TEAM DATA")
    print("=" * 50)
    
    prediction_factors = {
        'base_model': 0.35,          # 35% - Original ML model
        'recent_team_form': 0.30,    # 30% - Last 5-10 games (HUGE!)
        'season_averages': 0.15,     # 15% - Season-long team stats
        'bullpen_quality': 0.10,     # 10% - Relief pitching
        'ballpark_factors': 0.05,    # 5% - Venue effects
        'weather_impact': 0.05       # 5% - Weather conditions
    }
    
    # Hot team adjustments
    hot_team_thresholds = {
        'very_hot': 6.5,    # 6.5+ R/G recently = +0.5 runs adjustment
        'hot': 5.5,         # 5.5+ R/G recently = +0.3 runs adjustment
        'cold': 3.5,        # 3.5- R/G recently = -0.3 runs adjustment
        'very_cold': 2.5    # 2.5- R/G recently = -0.5 runs adjustment
    }
    
    print("📊 NEW PREDICTION WEIGHTING:")
    for factor, weight in prediction_factors.items():
        print(f"   {factor.replace('_', ' ').title()}: {weight*100}%")
    
    print(f"\n🔥 HOT TEAM ADJUSTMENTS:")
    for category, threshold in hot_team_thresholds.items():
        adjustment = '+0.5' if 'very_hot' in category else '+0.3' if 'hot' in category else '-0.3' if 'cold' in category else '-0.5'
        print(f"   {category.replace('_', ' ').title()}: {threshold} R/G = {adjustment} runs")
    
    print(f"\n💡 EXAMPLE IMPACT:")
    print(f"   Milwaukee Brewers (8.80 R/G recently) = +0.5 runs to total")
    print(f"   Combined with opponent adjustment = potentially +1.0 run edge!")
    print(f"   This explains why we're seeing too many HOLD recommendations")

if __name__ == "__main__":
    enhanced_prediction_logic()
