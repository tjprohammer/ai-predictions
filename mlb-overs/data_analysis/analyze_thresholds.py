#!/usr/bin/env python3
import requests

def analyze_current_thresholds():
    """Analyze current recommendation thresholds and suggest improvements"""
    data = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20').json()
    games = data.get('games', [])

    print('DETAILED GAME ANALYSIS:')
    print('=' * 50)

    for game in games[:8]:
        print(f'{game["away_team"]} @ {game["home_team"]}')
        print(f'  Predicted: {game["predicted_total"]:.2f}, Market: {game["market_total"]:.2f}')
        print(f'  Edge: {game["edge"]:+.2f}, Confidence: {game["confidence"]:.3f} ({game["confidence"]*100:.1f}%)')
        print(f'  Recommendation: {game["recommendation"]}')
        
        cal_pred = game.get('calibrated_predictions', {})
        if cal_pred:
            print(f'  Calibrated: {cal_pred.get("predicted_total", "N/A")}, Conf: {cal_pred.get("confidence", "N/A")}')
        print()

    print('\nTHRESHOLD ANALYSIS:')
    print('=' * 30)
    
    # Current thresholds based on the code
    current_edge_threshold = 0.3  # From enhanced_analysis.py
    
    print(f'Current edge threshold: {current_edge_threshold} runs')
    print(f'Games that would be actionable with lower thresholds:')
    
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        actionable = sum(1 for g in games if abs(float(g.get('edge', 0))) >= threshold)
        print(f'  Edge >= {threshold}: {actionable}/{len(games)} games ({actionable/len(games)*100:.1f}%)')

if __name__ == "__main__":
    analyze_current_thresholds()
