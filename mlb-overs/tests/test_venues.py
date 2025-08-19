#!/usr/bin/env python3
from enhanced_feature_pipeline import BALLPARK_FACTORS, _resolve_park_meta

# Test the venue name matching
venues_from_db = [
    "Busch Stadium", "Citi Field", "Coors Field", "Daikin Park", 
    "Dodger Stadium", "Fenway Park", "Great American Ball Park", 
    "Kauffman Stadium", "Nationals Park", "Oracle Park", 
    "Progressive Field", "Rogers Centre", "Sutter Health Park", 
    "Target Field", "Wrigley Field"
]

print("Venue Matching Test:")
print("=" * 50)

for venue in venues_from_db:
    meta = _resolve_park_meta(venue)
    print(f"{venue:<25} -> Run: {meta['run_factor']:.2f}, HR: {meta['hr_factor']:.2f}")

print("\nMissing venues:")
missing = [v for v in venues_from_db if v not in BALLPARK_FACTORS and not any(BALLPARK_FACTORS.get(k, {}).get('alias') == v for k in BALLPARK_FACTORS)]
for v in missing:
    print(f"  {v}")
