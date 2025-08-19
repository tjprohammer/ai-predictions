#!/usr/bin/env python3
"""
Quick check of bullpen features
"""
import pandas as pd

# Load the data
df = pd.read_csv('mlb-overs/data/legitimate_features_2025-08-14.csv')

print("=== Bullpen Features Analysis ===")
print(f"Total games: {len(df)}")
print("\nBullpen-related columns:")
bullpen_cols = [col for col in df.columns if 'bullpen' in col.lower() or 'innings' in col.lower()]
for col in bullpen_cols:
    print(f"  {col}")

print("\n=== Sample Game Analysis ===")
game = df.iloc[0]
print(f"Game: {game['away_team']} @ {game['home_team']}")
print(f"Venue: {game['venue_name']}")
print(f"\n--- Starting Pitchers ---")
print(f"Away starter ERA: {game['away_pitcher_season_era']:.2f}")
print(f"Home starter ERA: {game['home_pitcher_season_era']:.2f}")
print(f"Away starter avg innings: {game['away_pitcher_avg_innings']:.1f}")
print(f"Home starter avg innings: {game['home_pitcher_avg_innings']:.1f}")

print(f"\n--- Bullpen Stats ---")
print(f"Away bullpen ERA: {game['away_bullpen_era']:.2f}")
print(f"Home bullpen ERA: {game['home_bullpen_era']:.2f}")
print(f"Away bullpen reliability: {game['away_bullpen_reliability']:.2f}")
print(f"Home bullpen reliability: {game['home_bullpen_reliability']:.2f}")

print(f"\n--- Expected Bullpen Usage ---")
print(f"Away expected bullpen innings: {game['away_expected_bullpen_innings']:.1f}")
print(f"Home expected bullpen innings: {game['home_expected_bullpen_innings']:.1f}")
print(f"Total expected bullpen innings: {game['total_expected_bullpen_innings']:.1f}")

print(f"\n--- Composite Metrics ---")
print(f"Starter pitching quality: {game['starter_pitching_quality']:.2f}")
print(f"Bullpen pitching quality: {game['bullpen_pitching_quality']:.2f}")
print(f"Weighted pitching depth quality: {game['pitching_depth_quality']:.2f}")
print(f"Bullpen ERA differential: {game['bullpen_era_differential']:.2f}")

print("\n=== Key Insight ===")
print("The model now accounts for:")
print("1. How long starting pitchers typically go")
print("2. Team bullpen ERA and reliability")
print("3. Expected bullpen workload per game")
print("4. Weighted overall pitching quality (starters + bullpen)")
print("5. Bullpen advantages between teams")
