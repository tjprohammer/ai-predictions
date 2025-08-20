#!/usr/bin/env python3
"""
Reset bias corrections to prevent over-correction
"""

import json
from datetime import datetime

def reset_bias_corrections():
    """Reset bias corrections to more moderate levels"""
    
    # Start with minimal corrections
    corrections = {
        "global_adjustment": 0.2,  # Much smaller global adjustment
        "scoring_range_adjustments": {
            "Low (≤7)": -0.3,  # Smaller adjustment for low scoring
            "High (10-11)": 0.5,  # Smaller adjustment for high scoring
            "Very High (12+)": 1.0  # Much smaller adjustment for very high scoring
        },
        "confidence_adjustments": {},
        "temperature_adjustments": {},
        "venue_adjustments": {},
        "pitcher_quality_adjustments": {},
        "day_of_week_adjustments": {},
        "market_deviation_adjustments": {},
        "high_scoring_adjustments": {},
        "timestamp": datetime.now().isoformat(),
        "based_on_days": 0,
        "games_analyzed": 0,
        "note": "Reset to prevent over-correction - conservative adjustments only"
    }
    
    # Save the reset corrections
    with open('s:\\Projects\\AI_Predictions\\model_bias_corrections.json', 'w') as f:
        json.dump(corrections, f, indent=2)
    
    print("✅ Bias corrections reset to conservative levels")
    print(f"Global adjustment: {corrections['global_adjustment']}")
    print(f"Scoring range adjustments: {corrections['scoring_range_adjustments']}")
    
if __name__ == "__main__":
    reset_bias_corrections()
