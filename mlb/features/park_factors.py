"""Static park factor estimates (runs) relative to league average = 1.0.

These are coarse, single-number aggregates (mix of multi-year + recent season context).
Replace with dynamic, handedness / component split park factors later.
If a team key is missing, caller should default to 1.0.
"""

PARK_FACTORS_RUNS = {
    # High run environments
    "Colorado Rockies": 1.28,
    "Boston Red Sox": 1.06,
    "Cincinnati Reds": 1.10,
    "Texas Rangers": 1.07,
    "Philadelphia Phillies": 1.05,
    # Above average
    "Chicago Cubs": 1.04,
    "Atlanta Braves": 1.04,
    "Los Angeles Dodgers": 1.03,
    "New York Yankees": 1.03,
    "Houston Astros": 1.02,
    "Toronto Blue Jays": 1.02,
    "Baltimore Orioles": 1.02,
    # Near neutral
    "Arizona Diamondbacks": 1.01,
    "Milwaukee Brewers": 1.00,
    "Kansas City Royals": 1.00,
    "Detroit Tigers": 0.99,
    "Seattle Mariners": 0.98,
    "Miami Marlins": 0.98,
    "Chicago White Sox": 1.00,
    "Cleveland Guardians": 0.99,
    "Minnesota Twins": 1.01,
    "Pittsburgh Pirates": 0.99,
    "St. Louis Cardinals": 1.01,
    # Suppressive parks
    "San Diego Padres": 0.94,
    "San Francisco Giants": 0.90,
    "Tampa Bay Rays": 0.95,
    "Oakland Athletics": 0.91,
    "Los Angeles Angels": 0.97,
    "Washington Nationals": 0.99,
    "New York Mets": 0.97,
    "Colorado Rockies (Road)": 1.00,  # Placeholder / no-op
}

def get_park_factor(team: str) -> float:
    return PARK_FACTORS_RUNS.get(team, 1.0)
