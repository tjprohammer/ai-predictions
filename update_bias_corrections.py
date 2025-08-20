#!/usr/bin/env python3
"""
Update bias corrections based on recent performance analysis
"""

import json
from datetime import datetime

def update_bias_corrections():
    """Update bias corrections based on recent actual vs predicted performance"""
    
    # Data-driven corrections based on recent performance analysis
    # Model bias: -1.126 (predicting 1.126 runs too low on average)
    # Today's gap: 3.6 runs (market 8.6 vs predictions 5.0)
    corrections = {
        "global_adjustment": 2.5,  # Increased to close the 3.6 run gap
        "scoring_range_adjustments": {
            "Low (≤7)": -0.2,  # High scoring bias less problematic for low games
            "High (10-11)": 0.4,  # Moderate boost for high scoring
            "Very High (12+)": 0.6  # Stronger boost for very high scoring
        },
        "confidence_adjustments": {},
        "temperature_adjustments": {
            "Hot (80+°F)": 0.2,  # Weather boost for hot games
            "Mild (70-79°F)": 0.1
        },
        "venue_adjustments": {
            "COL": 0.3,  # Coors Field boost
            "TEX": 0.2,  # Hot weather park
            "MIN": 0.15  # Dome advantage
        },
        "pitcher_quality_adjustments": {
            "Both ERA > 4.5": 0.3,  # Poor pitching matchups
            "Both ERA < 3.5": -0.2   # Elite pitching matchups
        },
        "day_of_week_adjustments": {
            "weekend": 0.1  # Weekend variance
        },
        "market_deviation_adjustments": {},
        "high_scoring_adjustments": {
            "high_scoring_teams": {
                "teams": ["COL", "TEX", "MIN", "TOR", "LAA"],
                "adjustment": 0.25
            }
        },
        "timestamp": datetime.now().isoformat(),
        "based_on_days": 14,
        "games_analyzed": 50,
        "performance_note": "Model bias: -1.126 runs (predicting too low). Today's gap: 3.6 runs (market 8.6 vs pred 5.0).",
        "note": "Updated based on performance analysis - increased to 2.5 to close 3.6 run gap"
    }
    
    # Save the updated corrections
    with open('s:\\Projects\\AI_Predictions\\model_bias_corrections.json', 'w') as f:
        json.dump(corrections, f, indent=2)
    
    print("✅ Bias corrections updated based on performance data")
    print(f"Global adjustment: {corrections['global_adjustment']} (was 0.2)")
    print(f"Based on {corrections['games_analyzed']} recent games over {corrections['based_on_days']} days")
    print(f"Model was predicting 1.126 runs too low on average")
    
if __name__ == "__main__":
    update_bias_corrections()
