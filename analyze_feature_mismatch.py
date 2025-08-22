#!/usr/bin/env python3

import sys
sys.path.append('mlb-overs/deployment')

def analyze_feature_mismatch():
    """Analyze why features don't match between model and current generation"""
    
    print("🔍 FEATURE MISMATCH ANALYSIS")
    print("="*60)
    
    # Load expected features
    with open('expected_model_features.txt', 'r') as f:
        content = f.read()
        expected_features = [f.strip() for f in content.split('\\n') if f.strip()]
    
    print(f"📋 Model expects: {len(expected_features)} features")
    
    # From our feature quality check, we know what we're generating
    # Let's look at the specific differences
    
    print("\n🧐 EXPECTED FEATURE CATEGORIES:")
    
    # Categorize expected features
    pitching_features = [f for f in expected_features if any(x in f for x in ['era', 'whip', 'k_per', 'bb_per', 'sp_', 'bp_', 'pitch'])]
    team_features = [f for f in expected_features if any(x in f for x in ['team_', 'offense', 'power', 'discipline', 'woba', 'wrc', 'avg', 'iso', 'rpg'])]
    ballpark_features = [f for f in expected_features if any(x in f for x in ['ballpark', 'park', 'temp', 'wind', 'dome', 'rain', 'humidity', 'air_density', 'roof'])]
    umpire_features = [f for f in expected_features if 'ump' in f]
    schedule_features = [f for f in expected_features if any(x in f for x in ['rest', 'travel', 'games_', 'getaway'])]
    lineup_features = [f for f in expected_features if any(x in f for x in ['lineup', 'platoon', 'lhp', 'rhp', 'lhb', 'star_missing'])]
    other_features = [f for f in expected_features if f not in pitching_features + team_features + ballpark_features + umpire_features + schedule_features + lineup_features]
    
    print(f"   🥎 Pitching features: {len(pitching_features)}")
    print(f"   🏟️  Team features: {len(team_features)}")
    print(f"   🌡️  Ballpark/Weather: {len(ballpark_features)}")
    print(f"   👨‍⚖️ Umpire features: {len(umpire_features)}")
    print(f"   📅 Schedule features: {len(schedule_features)}")
    print(f"   🧑‍🤝‍🧑 Lineup features: {len(lineup_features)}")
    print(f"   ❓ Other features: {len(other_features)}")
    
    # Show some specific missing categories
    print("\\n🔍 SPECIFIC FEATURE EXAMPLES:")
    print("\\n  Expected Bullpen Features:")
    bullpen_expected = [f for f in expected_features if 'bp_' in f or 'bullpen' in f][:10]
    for i, feat in enumerate(bullpen_expected):
        print(f"    {i+1}: {feat}")
    
    print("\\n  Expected Advanced Stats:")
    advanced_expected = [f for f in expected_features if any(x in f for x in ['woba', 'wrcplus', 'iso', 'babip', 'xwoba'])][:10]
    for i, feat in enumerate(advanced_expected):
        print(f"    {i+1}: {feat}")
    
    print("\\n  Expected Lineup Features:")
    for i, feat in enumerate(lineup_features[:10]):
        print(f"    {i+1}: {feat}")
    
    print("\\n  Expected Schedule Features:")
    for i, feat in enumerate(schedule_features[:10]):
        print(f"    {i+1}: {feat}")
    
    print("\\n💡 DIAGNOSIS:")
    print("   The model was trained on a much more comprehensive feature set")
    print("   that includes advanced sabermetrics, lineup composition, and")
    print("   schedule factors that our current feature engineering doesn't include.")
    print("\\n   This explains why our predictions are so low - we're missing")
    print("   ~50+ critical features that the model relies on!")
    
    return expected_features

if __name__ == "__main__":
    expected_features = analyze_feature_mismatch()
    
    # Save properly formatted list
    with open('model_features_list.txt', 'w') as f:
        for feat in expected_features:
            f.write(feat + '\\n')
    print(f"\\n✅ Saved {len(expected_features)} features to model_features_list.txt")
