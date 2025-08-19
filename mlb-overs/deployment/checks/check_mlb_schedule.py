#!/usr/bin/env python3
import statsapi

# Check actual MLB schedule
games = statsapi.schedule(date='08/13/2025')
print(f"Actual MLB games today: {len(games)}")

for i, game in enumerate(games, 1):
    print(f"  {i:2d}. {game['away_name']} @ {game['home_name']}")
