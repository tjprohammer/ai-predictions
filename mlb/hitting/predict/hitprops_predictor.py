"""
MLB Hitting Props Predictor

This module combines player hitting features with DraftKings odds to make
"1+ Hit" prop betting recommendations using Expected Value and Kelly Criterion.
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from typing import Dict, List, Optional
import logging
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'features'))

from feature_builder import build_today_player_features, american_to_prob, kelly

log = logging.getLogger(__name__)

class HitPropsPredictor:
    """Generates hitting props predictions and betting recommendations"""
    
    def __init__(self, engine, min_edge: float = 0.05, min_kelly: float = 0.01):
        """
        Initialize predictor
        
        Args:
            engine: SQLAlchemy database engine
            min_edge: Minimum edge required to recommend a bet
            min_kelly: Minimum Kelly criterion size to recommend
        """
        self.engine = engine
        self.min_edge = min_edge
        self.min_kelly = min_kelly
    
    def get_latest_odds(self, target_date: str) -> pd.DataFrame:
        """Fetch latest DraftKings 1+ Hit odds for target date"""
        
        odds_query = text("""
            SELECT DISTINCT ON (player_id) 
                player_id, player_name, team, prop_type, line,
                over_odds, under_odds, timestamp, sportsbook
            FROM player_props_odds
            WHERE date = :target_date
              AND prop_type = '1+ Hit'
              AND sportsbook = 'DraftKings'
              AND over_odds IS NOT NULL
              AND under_odds IS NOT NULL
            ORDER BY player_id, timestamp DESC
        """)
        
        odds = pd.read_sql(odds_query, self.engine, params={'target_date': target_date})
        
        if odds.empty:
            log.warning(f"No DraftKings 1+ Hit odds found for {target_date}")
            return odds
        
        # Calculate implied probabilities
        odds['market_prob_over'] = odds['over_odds'].apply(american_to_prob)
        odds['market_prob_under'] = odds['under_odds'].apply(american_to_prob)
        odds['total_implied'] = odds['market_prob_over'] + odds['market_prob_under']
        odds['vig'] = (odds['total_implied'] - 1) * 100  # Vig percentage
        
        # Fair odds (remove vig proportionally)
        odds['fair_prob_over'] = odds['market_prob_over'] / odds['total_implied']
        odds['fair_prob_under'] = odds['market_prob_under'] / odds['total_implied']
        
        log.info(f"Retrieved odds for {len(odds)} players")
        log.info(f"Average vig: {odds['vig'].mean():.1f}%")
        
        return odds
    
    def generate_predictions(self, target_date: str) -> pd.DataFrame:
        """
        Generate complete hitting props predictions with betting recommendations
        
        Returns DataFrame with:
        - Player info and features
        - Market odds and fair probabilities  
        - Model predictions and edge calculations
        - Kelly criterion bet sizes
        - Betting recommendations
        """
        
        log.info(f"Generating hitting props predictions for {target_date}")
        
        # 1) Build player features
        features = build_today_player_features(self.engine, target_date)
        
        if features.empty:
            log.warning("No player features found")
            return pd.DataFrame()
        
        # 2) Get latest odds
        odds = self.get_latest_odds(target_date)
        
        if odds.empty:
            log.warning("No odds found - returning features only")
            return features
        
        # 3) Merge features with odds
        predictions = features.merge(
            odds[['player_id', 'line', 'over_odds', 'under_odds', 
                  'market_prob_over', 'fair_prob_over', 'vig', 'timestamp']],
            on='player_id',
            how='left'
        )
        
        # Filter to only players with odds available
        predictions = predictions.dropna(subset=['over_odds']).copy()
        
        if predictions.empty:
            log.warning("No players match between features and odds")
            return pd.DataFrame()
        
        # 4) Calculate edges and Kelly sizes
        predictions['model_prob'] = predictions['p_ge1_hit']
        predictions['edge_vs_market'] = predictions['model_prob'] - predictions['market_prob_over']
        predictions['edge_vs_fair'] = predictions['model_prob'] - predictions['fair_prob_over']
        
        # Kelly criterion for OVER bets (betting player gets 1+ hit)
        predictions['kelly_over'] = predictions.apply(
            lambda row: kelly(row['model_prob'], row['over_odds']), axis=1
        )
        
        # Kelly criterion for UNDER bets (betting player gets 0 hits)
        predictions['model_prob_under'] = 1 - predictions['model_prob']
        predictions['kelly_under'] = predictions.apply(
            lambda row: kelly(row['model_prob_under'], row['under_odds']), axis=1
        )
        
        # 5) Betting recommendations
        predictions['rec_bet_over'] = (
            (predictions['edge_vs_fair'] >= self.min_edge) & 
            (predictions['kelly_over'] >= self.min_kelly)
        )
        
        predictions['rec_bet_under'] = (
            (predictions['model_prob_under'] - (1 - predictions['fair_prob_over']) >= self.min_edge) &
            (predictions['kelly_under'] >= self.min_kelly)
        )
        
        # 6) Expected value calculations (using fair odds to remove vig)
        predictions['ev_over'] = np.where(
            predictions['rec_bet_over'],
            (predictions['model_prob'] * (predictions['over_odds'] / 100 if predictions['over_odds'] > 0 else 100 / (-predictions['over_odds']))) - 
            ((1 - predictions['model_prob']) * 1),
            0
        )
        
        predictions['ev_under'] = np.where(
            predictions['rec_bet_under'],
            (predictions['model_prob_under'] * (predictions['under_odds'] / 100 if predictions['under_odds'] > 0 else 100 / (-predictions['under_odds']))) -
            ((1 - predictions['model_prob_under']) * 1),
            0
        )
        
        # 7) Confidence scoring (multiple factors)
        def confidence_score(row):
            score = 0.5  # Base confidence
            
            # Recent form sample size
            if row.get('ab_l10', 0) >= 25:
                score += 0.1
            elif row.get('ab_l5', 0) >= 10:
                score += 0.05
                
            # BvP history
            if row.get('ab', 0) >= 10:  # Good BvP sample
                score += 0.15
            elif row.get('ab', 0) >= 3:  # Some BvP data
                score += 0.05
                
            # Lineup position (more predictable PAs)
            if pd.notna(row.get('lineup_spot')):
                if 1 <= row['lineup_spot'] <= 3:  # Top of order
                    score += 0.1
                elif 4 <= row['lineup_spot'] <= 6:  # Middle
                    score += 0.05
                    
            # Edge magnitude
            edge = abs(row.get('edge_vs_fair', 0))
            if edge >= 0.15:
                score += 0.15
            elif edge >= 0.10:
                score += 0.10
            elif edge >= 0.05:
                score += 0.05
                
            return min(1.0, score)
        
        predictions['confidence'] = predictions.apply(confidence_score, axis=1)
        
        # 8) Final ranking
        predictions['rec_strength'] = predictions.apply(lambda row: max(row['kelly_over'], row['kelly_under']), axis=1)
        predictions = predictions.sort_values(['rec_strength'], ascending=False)
        
        log.info(f"Generated predictions for {len(predictions)} players")
        log.info(f"OVER recommendations: {predictions['rec_bet_over'].sum()}")
        log.info(f"UNDER recommendations: {predictions['rec_bet_under'].sum()}")
        
        return predictions
    
    def get_top_recommendations(self, target_date: str, limit: int = 10) -> Dict:
        """Get top betting recommendations in structured format"""
        
        predictions = self.generate_predictions(target_date)
        
        if predictions.empty:
            return {'over_bets': [], 'under_bets': [], 'summary': {}}
        
        # Top OVER bets
        over_bets = predictions[predictions['rec_bet_over']].head(limit)
        over_recs = []
        
        for _, row in over_bets.iterrows():
            over_recs.append({
                'player_id': int(row['player_id']),
                'player_name': row['player_name'],
                'team': row['team'],
                'opponent': row['opponent'],
                'model_prob': round(row['model_prob'], 3),
                'market_prob': round(row['market_prob_over'], 3),
                'fair_prob': round(row['fair_prob_over'], 3),
                'edge': round(row['edge_vs_fair'], 3),
                'kelly_size': round(row['kelly_over'], 3),
                'ev': round(row['ev_over'], 3),
                'confidence': round(row['confidence'], 3),
                'odds': int(row['over_odds']),
                'recent_form': f"{int(row.get('hits_l10', 0))}/{int(row.get('ab_l10', 0))}",
                'vs_pitcher': f"{int(row.get('h', 0))}/{int(row.get('ab', 0))}" if row.get('ab', 0) > 0 else "No history"
            })
        
        # Top UNDER bets  
        under_bets = predictions[predictions['rec_bet_under']].head(limit)
        under_recs = []
        
        for _, row in under_bets.iterrows():
            under_recs.append({
                'player_id': int(row['player_id']),
                'player_name': row['player_name'],
                'team': row['team'],
                'opponent': row['opponent'],
                'model_prob_under': round(row['model_prob_under'], 3),
                'market_prob_under': round(1 - row['market_prob_over'], 3),
                'edge': round(row['model_prob_under'] - (1 - row['fair_prob_over']), 3),
                'kelly_size': round(row['kelly_under'], 3),
                'ev': round(row['ev_under'], 3),
                'confidence': round(row['confidence'], 3),
                'odds': int(row['under_odds']),
                'recent_form': f"{int(row.get('hits_l10', 0))}/{int(row.get('ab_l10', 0))}",
                'vs_pitcher': f"{int(row.get('h', 0))}/{int(row.get('ab', 0))}" if row.get('ab', 0) > 0 else "No history"
            })
        
        # Summary stats
        summary = {
            'total_players': len(predictions),
            'over_recs': len(over_recs),
            'under_recs': len(under_recs),
            'avg_confidence': round(predictions['confidence'].mean(), 3),
            'max_kelly': round(predictions['rec_strength'].max(), 3),
            'avg_vig': round(predictions['vig'].mean(), 1)
        }
        
        return {
            'over_bets': over_recs,
            'under_bets': under_recs,
            'summary': summary,
            'generated_at': datetime.now().isoformat()
        }
    
    def save_predictions(self, target_date: str) -> int:
        """Save predictions to database and return count"""
        
        predictions = self.generate_predictions(target_date)
        
        if predictions.empty:
            log.warning("No predictions to save")
            return 0
        
        # Prepare data for database
        save_data = predictions[[
            'game_id', 'player_id', 'player_name', 'team', 'opponent',
            'model_prob', 'market_prob_over', 'fair_prob_over', 
            'edge_vs_fair', 'kelly_over', 'kelly_under',
            'rec_bet_over', 'rec_bet_under', 'confidence',
            'over_odds', 'under_odds', 'vig'
        ]].copy()
        
        save_data['date'] = target_date
        save_data['prop_type'] = '1+ Hit'
        save_data['sportsbook'] = 'DraftKings'
        save_data['created_at'] = datetime.now()
        
        # Save to database
        save_data.to_sql(
            'hitter_prop_predictions', 
            self.engine, 
            if_exists='append', 
            index=False
        )
        
        log.info(f"Saved {len(save_data)} predictions to database")
        return len(save_data)

def main():
    """CLI interface for generating hitting props predictions"""
    import argparse
    import sys
    from pathlib import Path
    from sqlalchemy import create_engine
    import os
    from dotenv import load_dotenv
    import json
    
    parser = argparse.ArgumentParser(description='Generate MLB hitting props predictions')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--save', action='store_true', help='Save predictions to database')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--min-edge', type=float, default=0.05, help='Minimum edge threshold')
    parser.add_argument('--min-kelly', type=float, default=0.01, help='Minimum Kelly size threshold')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load environment
    load_dotenv(Path(__file__).parent.parent / '.env')
    
    # Connect to database
    engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
    
    # Get target date
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize predictor
    predictor = HitPropsPredictor(engine, min_edge=args.min_edge, min_kelly=args.min_kelly)
    
    # Generate recommendations
    print(f"Generating hitting props predictions for {target_date}")
    recommendations = predictor.get_top_recommendations(target_date)
    
    # Print summary
    summary = recommendations['summary']
    print(f"\n=== PREDICTION SUMMARY ===")
    print(f"Players analyzed: {summary['total_players']}")
    print(f"OVER recommendations: {summary['over_recs']}")  
    print(f"UNDER recommendations: {summary['under_recs']}")
    print(f"Average confidence: {summary['avg_confidence']}")
    print(f"Max Kelly size: {summary['max_kelly']}")
    print(f"Average vig: {summary['avg_vig']}%")
    
    # Print top recommendations
    if recommendations['over_bets']:
        print(f"\n=== TOP OVER BETS (1+ Hit) ===")
        for i, bet in enumerate(recommendations['over_bets'][:5], 1):
            print(f"{i}. {bet['player_name']} ({bet['team']}) vs {bet['opponent']}")
            print(f"   Model: {bet['model_prob']:.1%} | Market: {bet['market_prob']:.1%} | Edge: {bet['edge']:+.1%}")
            print(f"   Kelly: {bet['kelly_size']:.1%} | EV: {bet['ev']:+.3f} | Odds: {bet['odds']:+d}")
            print(f"   Form: {bet['recent_form']} | vs P: {bet['vs_pitcher']}")
            print()
    
    if recommendations['under_bets']:
        print(f"\n=== TOP UNDER BETS (0 Hits) ===")
        for i, bet in enumerate(recommendations['under_bets'][:3], 1):
            print(f"{i}. {bet['player_name']} ({bet['team']}) vs {bet['opponent']}")
            print(f"   Model: {bet['model_prob_under']:.1%} | Edge: {bet['edge']:+.1%}")
            print(f"   Kelly: {bet['kelly_size']:.1%} | EV: {bet['ev']:+.3f} | Odds: {bet['odds']:+d}")
            print(f"   Form: {bet['recent_form']} | vs P: {bet['vs_pitcher']}")
            print()
    
    # Save to database if requested
    if args.save:
        count = predictor.save_predictions(target_date)
        print(f"Saved {count} predictions to database")
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"Saved recommendations to {args.output}")

if __name__ == "__main__":
    main()
