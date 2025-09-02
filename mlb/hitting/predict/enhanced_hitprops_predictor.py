"""
Enhanced MLB Hitting Props Predictor
Comprehensive predictor for multiple hitting props markets using Empirical Bayes methodology

Supports:
- HITS_0.5, HITS_1.5, HITS_2.5 (≥1, ≥2, ≥3 hits)
- HR_0.5 (≥1 home run)
- RBI_0.5, RBI_1.5 (≥1, ≥2 RBIs)
- TB_1.5, TB_2.5, TB_3.5 (≥2, ≥3, ≥4 total bases)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple, Optional
import logging

log = logging.getLogger(__name__)

class EnhancedHitPropsPredictor:
    
    def __init__(self, database_url: str = None):
        """Initialize predictor with database connection"""
        if database_url is None:
            database_url = "postgresql://mlbuser:mlbpass@localhost/mlb"
        self.engine = create_engine(database_url)
        
        # Empirical Bayes priors for different markets
        self.priors = {
            'hits': {'alpha': 4.0, 'beta': 16.0},    # ~20% hit rate prior
            'multi_hits': {'alpha': 2.0, 'beta': 18.0},  # ~10% multi-hit rate
            'home_runs': {'alpha': 1.0, 'beta': 29.0},   # ~3% HR rate  
            'rbis': {'alpha': 3.0, 'beta': 17.0},        # ~15% RBI rate
            'total_bases': {'alpha': 6.0, 'beta': 14.0}  # ~30% multi-TB rate
        }
        
        # League baseline rates for each market
        self.league_rates = {
            'hits_1plus': 0.70,   # P(≥1 hit)
            'hits_2plus': 0.25,   # P(≥2 hits)
            'hits_3plus': 0.05,   # P(≥3 hits)
            'hr_1plus': 0.12,     # P(≥1 HR)
            'rbi_1plus': 0.35,    # P(≥1 RBI)
            'rbi_2plus': 0.15,    # P(≥2 RBI)
            'tb_2plus': 0.60,     # P(≥2 TB)
            'tb_3plus': 0.40,     # P(≥3 TB)
            'tb_4plus': 0.25      # P(≥4 TB)
        }
    
    def predict_all_props(self, target_date: str) -> pd.DataFrame:
        """Generate predictions for all supported hitting props markets"""
        
        log.info(f"Generating hitting props predictions for {target_date}")
        
        # Get player features
        features = self._get_player_features(target_date)
        
        if features.empty:
            log.warning(f"No player features found for {target_date}")
            return pd.DataFrame()
        
        # Generate predictions for each market
        all_predictions = []
        
        markets = [
            ('HITS_0.5', self._predict_hits_market),
            ('HITS_1.5', self._predict_multi_hits_market),
            ('HR_0.5', self._predict_hr_market),
            ('RBI_0.5', self._predict_rbi_market),
            ('TB_1.5', self._predict_tb_market)
        ]
        
        for market_name, predict_func in markets:
            market_preds = predict_func(features, market_name)
            if not market_preds.empty:
                all_predictions.append(market_preds)
        
        if not all_predictions:
            return pd.DataFrame()
        
        # Combine all predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Add odds and EV calculations
        predictions_df = self._add_ev_calculations(predictions_df, target_date)
        
        log.info(f"Generated {len(predictions_df)} prop predictions across {len(markets)} markets")
        return predictions_df
    
    def _get_player_features(self, target_date: str) -> pd.DataFrame:
        """Get comprehensive player features for target date"""
        
        # Import here to avoid circular dependency
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
        from comprehensive_feature_builder import ComprehensiveHittingFeatureBuilder
        
        builder = ComprehensiveHittingFeatureBuilder(str(self.engine.url))
        return builder.build_features_for_date(target_date)
    
    def _predict_hits_market(self, features: pd.DataFrame, market: str) -> pd.DataFrame:
        """Predict ≥1 hit probability using Empirical Bayes blending"""
        
        predictions = []
        
        for _, player in features.iterrows():
            # Component probabilities
            form_prob = self._eb_hit_prob(player.get('hits_l10', 0), player.get('ab_l10', 0))
            bvp_prob = self._eb_hit_prob(player.get('bvp_h', 0), player.get('bvp_ab', 0))
            hand_prob = self._get_vs_hand_hit_prob(player)
            
            # Weighted blend based on sample sizes and reliability
            weights = self._calculate_weights(player)
            
            blended_prob = (
                weights['form'] * form_prob + 
                weights['bvp'] * bvp_prob + 
                weights['hand'] * hand_prob +
                weights['league'] * self.league_rates['hits_1plus']
            )
            
            # Adjust for plate appearances expectation
            expected_pa = player.get('expected_pa', 4.0)
            pa_adjusted_prob = self._adjust_for_pa(blended_prob, expected_pa, 'hits')
            
            # Apply momentum adjustments
            final_prob = self._apply_momentum_adjustment(pa_adjusted_prob, player)
            
            predictions.append({
                'player_id': player['player_id'],
                'player_name': player.get('player_name', ''),
                'team': player.get('team', ''),
                'opponent': player.get('opponent', ''),
                'market': market,
                'line': 0.5,
                'prob_over': max(0.01, min(0.99, final_prob)),
                'prob_under': max(0.01, min(0.99, 1 - final_prob)),
                'model_version': 'enhanced_v1',
                'confidence_score': self._calculate_confidence(weights)
            })
        
        return pd.DataFrame(predictions)
    
    def _predict_multi_hits_market(self, features: pd.DataFrame, market: str) -> pd.DataFrame:
        """Predict ≥2 hits probability"""
        
        predictions = []
        
        for _, player in features.iterrows():
            # Multi-hit rates from recent form
            multi_hit_l10 = player.get('multi_hit_games_l5', 0) / max(1, player.get('ab_l10', 1) / 4)
            
            # Base multi-hit probability using binomial approximation
            single_hit_prob = self._eb_hit_prob(player.get('hits_l10', 0), player.get('ab_l10', 0))
            expected_pa = player.get('expected_pa', 4.0)
            
            # P(≥2 hits) ≈ 1 - P(0 hits) - P(1 hit) for binomial(n=PA, p=hit_rate)
            p0 = (1 - single_hit_prob) ** expected_pa
            p1 = expected_pa * single_hit_prob * ((1 - single_hit_prob) ** (expected_pa - 1))
            multi_hit_prob = max(0.01, min(0.99, 1 - p0 - p1))
            
            # Blend with empirical multi-hit rate
            weights = self._calculate_weights(player)
            final_prob = 0.7 * multi_hit_prob + 0.3 * min(multi_hit_l10, 0.5)
            
            predictions.append({
                'player_id': player['player_id'],
                'player_name': player.get('player_name', ''),
                'team': player.get('team', ''),
                'opponent': player.get('opponent', ''),
                'market': market,
                'line': 1.5,
                'prob_over': max(0.01, min(0.99, final_prob)),
                'prob_under': max(0.01, min(0.99, 1 - final_prob)),
                'model_version': 'enhanced_v1',
                'confidence_score': self._calculate_confidence(weights) * 0.8  # Lower confidence for multi-hit
            })
        
        return pd.DataFrame(predictions)
    
    def _predict_hr_market(self, features: pd.DataFrame, market: str) -> pd.DataFrame:
        """Predict ≥1 home run probability"""
        
        predictions = []
        
        for _, player in features.iterrows():
            # HR rates from different windows
            hr_rate_l10 = player.get('hr_l10', 0) / max(1, player.get('ab_l10', 1))
            hr_rate_l15 = player.get('hr_l15', 0) / max(1, player.get('ab_l15', 1))
            
            # BvP HR rate
            bvp_hr_rate = player.get('bvp_hr', 0) / max(1, player.get('bvp_ab', 1))
            
            # Blend with league average
            weights = self._calculate_weights(player)
            blended_hr_rate = (
                weights['form'] * (hr_rate_l10 * 0.7 + hr_rate_l15 * 0.3) +
                weights['bvp'] * bvp_hr_rate +
                weights['league'] * (self.league_rates['hr_1plus'] / 4.0)  # per AB
            )
            
            # Convert to per-game probability
            expected_pa = player.get('expected_pa', 4.0)
            hr_prob = 1 - (1 - blended_hr_rate) ** expected_pa
            
            predictions.append({
                'player_id': player['player_id'],
                'player_name': player.get('player_name', ''),
                'team': player.get('team', ''),
                'opponent': player.get('opponent', ''),
                'market': market,
                'line': 0.5,
                'prob_over': max(0.01, min(0.99, hr_prob)),
                'prob_under': max(0.01, min(0.99, 1 - hr_prob)),
                'model_version': 'enhanced_v1',
                'confidence_score': self._calculate_confidence(weights) * 0.7  # Lower confidence for HR
            })
        
        return pd.DataFrame(predictions)
    
    def _predict_rbi_market(self, features: pd.DataFrame, market: str) -> pd.DataFrame:
        """Predict ≥1 RBI probability"""
        
        predictions = []
        
        for _, player in features.iterrows():
            # RBI rates from recent form
            rbi_rate_l10 = player.get('rbi_l10', 0) / max(1, player.get('ab_l10', 1))
            
            # Adjust for lineup position (middle order more RBI opportunities)
            lineup_multiplier = self._get_lineup_rbi_multiplier(player.get('lineup_spot', 5))
            
            # Base RBI probability
            expected_pa = player.get('expected_pa', 4.0)
            base_rbi_prob = 1 - (1 - rbi_rate_l10 * lineup_multiplier) ** expected_pa
            
            # Blend with league average
            weights = self._calculate_weights(player)
            final_prob = (
                weights['form'] * base_rbi_prob +
                weights['league'] * self.league_rates['rbi_1plus']
            )
            
            predictions.append({
                'player_id': player['player_id'],
                'player_name': player.get('player_name', ''),
                'team': player.get('team', ''),
                'opponent': player.get('opponent', ''),
                'market': market,
                'line': 0.5,
                'prob_over': max(0.01, min(0.99, final_prob)),
                'prob_under': max(0.01, min(0.99, 1 - final_prob)),
                'model_version': 'enhanced_v1',
                'confidence_score': self._calculate_confidence(weights) * 0.8
            })
        
        return pd.DataFrame(predictions)
    
    def _predict_tb_market(self, features: pd.DataFrame, market: str) -> pd.DataFrame:
        """Predict ≥2 total bases probability"""
        
        predictions = []
        
        for _, player in features.iterrows():
            # Total bases rate from recent form
            tb_rate_l10 = player.get('tb_l10', 0) / max(1, player.get('ab_l10', 1))
            
            # Expected total bases per game
            expected_pa = player.get('expected_pa', 4.0)
            expected_tb = tb_rate_l10 * expected_pa
            
            # P(≥2 TB) using empirical distribution
            if expected_tb <= 1.0:
                tb_prob = expected_tb * 0.3  # Low chance if expecting < 1 TB
            elif expected_tb <= 2.0:
                tb_prob = 0.3 + (expected_tb - 1.0) * 0.4  # Ramp up
            else:
                tb_prob = 0.7 + min(0.25, (expected_tb - 2.0) * 0.1)  # Cap at ~95%
            
            # Blend with league average
            weights = self._calculate_weights(player)
            final_prob = (
                weights['form'] * tb_prob +
                weights['league'] * self.league_rates['tb_2plus']
            )
            
            predictions.append({
                'player_id': player['player_id'],
                'player_name': player.get('player_name', ''),
                'team': player.get('team', ''),
                'opponent': player.get('opponent', ''),
                'market': market,
                'line': 1.5,
                'prob_over': max(0.01, min(0.99, final_prob)),
                'prob_under': max(0.01, min(0.99, 1 - final_prob)),
                'model_version': 'enhanced_v1',
                'confidence_score': self._calculate_confidence(weights)
            })
        
        return pd.DataFrame(predictions)
    
    def _eb_hit_prob(self, hits: float, at_bats: float) -> float:
        """Calculate Empirical Bayes hit probability"""
        if at_bats <= 0:
            return self.priors['hits']['alpha'] / (self.priors['hits']['alpha'] + self.priors['hits']['beta'])
        
        return (hits + self.priors['hits']['alpha']) / (at_bats + self.priors['hits']['alpha'] + self.priors['hits']['beta'])
    
    def _get_vs_hand_hit_prob(self, player: Dict) -> float:
        """Get vs-hand adjusted hit probability"""
        pitcher_hand = player.get('pitcher_hand', 'R')
        
        if pitcher_hand == 'L':
            return self._eb_hit_prob(player.get('vs_lhp_h', 0), player.get('vs_lhp_ab', 0))
        else:
            return self._eb_hit_prob(player.get('vs_rhp_h', 0), player.get('vs_rhp_ab', 0))
    
    def _calculate_weights(self, player: Dict) -> Dict[str, float]:
        """Calculate blending weights based on sample sizes"""
        
        # Sample sizes for each component
        form_ab = player.get('ab_l10', 0)
        bvp_ab = player.get('bvp_ab', 0)
        hand_ab = player.get('vs_rhp_ab', 0) + player.get('vs_lhp_ab', 0)
        
        # Weight based on sample size reliability
        form_weight = min(0.5, form_ab / 40.0)  # Max weight at 40 AB
        bvp_weight = min(0.3, bvp_ab / 20.0)    # Max weight at 20 AB vs this pitcher
        hand_weight = min(0.3, hand_ab / 100.0) # Max weight at 100 AB vs hand
        
        # Normalize weights
        total_weight = form_weight + bvp_weight + hand_weight
        league_weight = max(0.1, 1.0 - total_weight)  # Minimum 10% league average
        
        # Renormalize
        total_all = form_weight + bvp_weight + hand_weight + league_weight
        
        return {
            'form': form_weight / total_all,
            'bvp': bvp_weight / total_all,
            'hand': hand_weight / total_all,
            'league': league_weight / total_all
        }
    
    def _adjust_for_pa(self, base_prob: float, expected_pa: float, market_type: str) -> float:
        """Adjust probability based on expected plate appearances"""
        
        if market_type == 'hits':
            # More PAs = higher chance of ≥1 hit
            if expected_pa >= 4.5:
                return min(0.95, base_prob * 1.1)
            elif expected_pa <= 3.0:
                return max(0.05, base_prob * 0.9)
        
        return base_prob
    
    def _apply_momentum_adjustment(self, base_prob: float, player: Dict) -> float:
        """Apply momentum/hotness adjustments"""
        
        hotness = player.get('hotness_indicator', 'neutral')
        
        if hotness == 'hot':
            return min(0.95, base_prob * 1.05)
        elif hotness == 'cold':
            return max(0.05, base_prob * 0.95)
        
        return base_prob
    
    def _get_lineup_rbi_multiplier(self, lineup_spot: int) -> float:
        """Get RBI opportunity multiplier based on lineup position"""
        
        multipliers = {
            1: 0.8,   # Leadoff - fewer RBI opportunities
            2: 0.9,   # #2 hitter
            3: 1.1,   # #3 hitter - good RBI spot
            4: 1.2,   # Cleanup - best RBI spot
            5: 1.1,   # #5 hitter
            6: 1.0,   # #6 hitter
            7: 0.9,   # Lower order
            8: 0.8,
            9: 0.7    # Pitcher spot (NL) or worst hitter
        }
        
        return multipliers.get(lineup_spot, 1.0)
    
    def _calculate_confidence(self, weights: Dict[str, float]) -> float:
        """Calculate prediction confidence score"""
        
        # Higher confidence when we have more non-league data
        non_league_weight = 1.0 - weights.get('league', 0.5)
        
        # Confidence based on sample balance
        weight_variance = np.var(list(weights.values()))
        balance_score = max(0.3, 1.0 - weight_variance * 2)
        
        return min(0.95, non_league_weight * balance_score)
    
    def _add_ev_calculations(self, predictions: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Add expected value and Kelly calculations"""
        
        # Get odds from database if available
        odds_query = text("""
            SELECT 
                player_id, 
                market, 
                over_price, 
                under_price,
                line
            FROM player_props_odds 
            WHERE date = :target_date
        """)
        
        with self.engine.connect() as conn:
            odds_df = pd.read_sql(odds_query, conn, params={'target_date': target_date})
        
        # Merge odds with predictions
        predictions = predictions.merge(
            odds_df, 
            on=['player_id', 'market'], 
            how='left'
        )
        
        # Calculate EV and Kelly for rows with odds
        predictions['over_price'] = predictions['over_price'].fillna(0)
        predictions['under_price'] = predictions['under_price'].fillna(0)
        
        # EV calculations
        predictions['ev_over'] = np.where(
            predictions['over_price'] > 0,
            self._calculate_ev(predictions['prob_over'], predictions['over_price']),
            0
        )
        
        predictions['ev_under'] = np.where(
            predictions['under_price'] > 0,
            self._calculate_ev(predictions['prob_under'], predictions['under_price']),
            0
        )
        
        # Kelly calculations  
        predictions['kelly_over'] = np.where(
            predictions['over_price'] > 0,
            self._calculate_kelly(predictions['prob_over'], predictions['over_price']),
            0
        )
        
        predictions['kelly_under'] = np.where(
            predictions['under_price'] > 0,
            self._calculate_kelly(predictions['prob_under'], predictions['under_price']),
            0
        )
        
        # Add metadata
        predictions['date'] = target_date
        predictions['game_id'] = ''  # TODO: Get from games table
        predictions['created_at'] = datetime.now()
        
        return predictions
    
    def _calculate_ev(self, prob: float, american_odds: float) -> float:
        """Calculate expected value"""
        if american_odds == 0 or pd.isna(american_odds):
            return 0
        
        if american_odds > 0:
            payout = american_odds / 100
        else:
            payout = 100 / abs(american_odds)
        
        return prob * payout - (1 - prob)
    
    def _calculate_kelly(self, prob: float, american_odds: float) -> float:
        """Calculate Kelly criterion bet size"""
        if american_odds == 0 or pd.isna(american_odds) or prob <= 0:
            return 0
        
        if american_odds > 0:
            b = american_odds / 100
        else:
            b = 100 / abs(american_odds)
        
        kelly_size = max(0, (prob * (b + 1) - 1) / b)
        return min(kelly_size, 0.25)  # Cap at 25%
    
    def save_predictions(self, predictions: pd.DataFrame) -> int:
        """Save predictions to database"""
        
        if predictions.empty:
            return 0
        
        # Prepare for database insert
        predictions_clean = predictions[[
            'date', 'game_id', 'player_id', 'player_name', 'team', 'opponent',
            'market', 'line', 'prob_over', 'prob_under', 'over_price', 'under_price',
            'ev_over', 'ev_under', 'kelly_over', 'kelly_under', 
            'model_version', 'confidence_score'
        ]].copy()
        
        # Upsert to database
        with self.engine.begin() as conn:
            # Create temp table
            predictions_clean.to_sql('temp_hit_predictions', conn, if_exists='replace', index=False)
            
            # Upsert
            upsert_query = text("""
                INSERT INTO hitter_prop_predictions 
                SELECT * FROM temp_hit_predictions
                ON CONFLICT (date, game_id, player_id, market)
                DO UPDATE SET
                    prob_over = EXCLUDED.prob_over,
                    prob_under = EXCLUDED.prob_under,
                    ev_over = EXCLUDED.ev_over,
                    ev_under = EXCLUDED.ev_under,
                    kelly_over = EXCLUDED.kelly_over,
                    kelly_under = EXCLUDED.kelly_under,
                    confidence_score = EXCLUDED.confidence_score,
                    created_at = NOW()
            """)
            
            conn.execute(upsert_query)
            conn.execute(text("DROP TABLE temp_hit_predictions"))
        
        log.info(f"Saved {len(predictions_clean)} hitting prop predictions")
        return len(predictions_clean)


def main():
    """Test the enhanced hitting props predictor"""
    
    import sys
    from datetime import datetime
    
    logging.basicConfig(level=logging.INFO)
    
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    
    predictor = EnhancedHitPropsPredictor()
    predictions = predictor.predict_all_props(target_date)
    
    if not predictions.empty:
        print(f"\nGenerated {len(predictions)} predictions:")
        print(predictions.groupby('market').size())
        
        print(f"\nSample predictions:")
        print(predictions[['player_name', 'market', 'prob_over', 'confidence_score']].head(10))
        
        # Save to database
        saved_count = predictor.save_predictions(predictions)
        print(f"\nSaved {saved_count} predictions to database")
    else:
        print(f"No predictions generated for {target_date}")


if __name__ == "__main__":
    main()
