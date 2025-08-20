"""
Create even stronger bias corrections to close the ~3 run gap with market expectations.
"""

import json

def create_stronger_corrections():
    """Create stronger bias corrections based on current market gaps"""
    
    print("🔧 Creating Stronger Bias Corrections...")
    print("Current gap with market: ~3.0 runs under-prediction")
    print("Current global adjustment: +0.696 runs")
    print("Needed additional adjustment: ~2.3 runs")
    
    # Create much stronger bias corrections
    bias_corrections = {
        "global_adjustment": 3.0,  # Significantly stronger correction
        "scoring_range_adjustments": {
            "Low (≤7)": 0.0,      # Neutral for low scoring
            "Mid (8-10)": 0.2,    # Slight boost for mid-range  
            "High (11+)": 0.5,    # Stronger boost for high scoring
            "Very High (14+)": 0.8  # Very strong boost for highest scoring
        },
        "confidence_adjustments": {
            "High Confidence (>80%)": 0.2,
            "Medium Confidence (60-80%)": 0.0,
            "Low Confidence (<60%)": -0.1
        },
        "temperature_adjustments": {
            "Hot (80+°F)": 0.3,
            "Warm (70-79°F)": 0.2,
            "Cool (60-69°F)": 0.0,
            "Cold (<60°F)": -0.1
        },
        "venue_adjustments": {
            "COL": 0.5,  # Coors Field - strongest boost
            "TEX": 0.3,  # Globe Life Field
            "LAD": 0.2,  # Dodger Stadium
            "BOS": 0.2,  # Fenway Park
            "CIN": 0.2,  # Great American Ball Park
            "MIL": -0.1, # American Family Field  
            "OAK": -0.2, # Oakland Coliseum
            "SEA": -0.1  # T-Mobile Park
        },
        "pitcher_quality_adjustments": {
            "Elite SP (ERA < 3.0)": -0.1,
            "Good SP (ERA 3.0-4.0)": 0.1,
            "Average SP (ERA 4.0-5.0)": 0.3,
            "Poor SP (ERA > 5.0)": 0.5
        },
        "metadata": {
            "last_updated": "2025-08-20T10:30:00",
            "purpose": "Address systematic 3-run under-prediction vs market",
            "target_improvement": 3.0,
            "version": "stronger_market_alignment_v1"
        }
    }
    
    return bias_corrections

def save_stronger_corrections(corrections):
    """Save the stronger bias corrections"""
    
    main_file = 's:\\Projects\\AI_Predictions\\model_bias_corrections.json'
    deployment_file = 's:\\Projects\\AI_Predictions\\mlb-overs\\deployment\\model_bias_corrections.json'
    
    # Save to main file
    with open(main_file, 'w') as f:
        json.dump(corrections, f, indent=2)
    print(f"✅ Saved stronger corrections to: {main_file}")
    
    # Save to deployment file  
    with open(deployment_file, 'w') as f:
        json.dump(corrections, f, indent=2)
    print(f"✅ Saved stronger corrections to: {deployment_file}")
    
    print(f"\n🎯 NEW STRONGER CORRECTIONS SUMMARY:")
    print(f"   Global Adjustment: +{corrections['global_adjustment']} runs")
    print(f"   Target: Close 3-run gap with market expectations")
    print(f"   Strategy: Aggressive upward adjustment")

def main():
    corrections = create_stronger_corrections()
    save_stronger_corrections(corrections)
    print("\n✅ Stronger bias corrections created!")
    print("Next step: Re-run predictions to test effectiveness")

if __name__ == "__main__":
    main()
