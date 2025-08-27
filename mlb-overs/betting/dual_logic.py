"""
Enhanced dual model betting logic with hold minimization
"""
import pandas as pd
import numpy as np
import logging
from sqlalchemy import text

log = logging.getLogger(__name__)

def enhance_betting_decisions(engine, game_data: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance betting decisions using dual model outputs and hold minimization
    
    Args:
        engine: Database connection
        game_data: DataFrame with dual model predictions
        
    Returns:
        Enhanced DataFrame with improved betting recommendations
    """
    
    log.info(f"🎯 Enhancing betting decisions for {len(game_data)} games")
    
    # Load dual model predictions if not already present
    if 'predicted_total_original' not in game_data.columns:
        log.info("Loading dual model predictions from database...")
        
        game_ids = game_data['game_id'].tolist()
        query = text("""
            SELECT game_id, predicted_total_original, predicted_total_learning, market_total
            FROM enhanced_games 
            WHERE game_id = ANY(:game_ids)
            AND predicted_total_original IS NOT NULL 
            AND predicted_total_learning IS NOT NULL
        """)
        
        dual_preds = pd.read_sql(query, engine, params={'game_ids': game_ids})
        game_data = game_data.merge(dual_preds, on='game_id', how='left', suffixes=('', '_db'))
    
    enhanced_data = game_data.copy()
    
    # Calculate dual model metrics
    enhanced_data['model_agreement'] = abs(
        enhanced_data['predicted_total_original'] - enhanced_data['predicted_total_learning']
    )
    
    enhanced_data['models_avg'] = (
        enhanced_data['predicted_total_original'] + enhanced_data['predicted_total_learning']
    ) / 2
    
    # Market comparison for each model
    enhanced_data['original_vs_market'] = enhanced_data['predicted_total_original'] - enhanced_data['market_total']
    enhanced_data['learning_vs_market'] = enhanced_data['predicted_total_learning'] - enhanced_data['market_total']
    
    # Determine if models agree on direction vs market
    enhanced_data['original_direction'] = np.where(enhanced_data['original_vs_market'] > 0.1, 'OVER',
                                                 np.where(enhanced_data['original_vs_market'] < -0.1, 'UNDER', 'NEUTRAL'))
    enhanced_data['learning_direction'] = np.where(enhanced_data['learning_vs_market'] > 0.1, 'OVER',
                                                 np.where(enhanced_data['learning_vs_market'] < -0.1, 'UNDER', 'NEUTRAL'))
    
    enhanced_data['models_agree'] = enhanced_data['original_direction'] == enhanced_data['learning_direction']
    enhanced_data['models_disagree'] = (enhanced_data['original_direction'] != enhanced_data['learning_direction']) & \
                                     (enhanced_data['original_direction'] != 'NEUTRAL') & \
                                     (enhanced_data['learning_direction'] != 'NEUTRAL')
    
    # Enhanced recommendation logic
    recommendations = []
    confidence_scores = []
    hold_reasons = []
    
    for idx, row in enhanced_data.iterrows():
        market_total = row['market_total']
        original_pred = row['predicted_total_original']
        learning_pred = row['predicted_total_learning']
        agreement = row['model_agreement']
        models_agree = row['models_agree']
        models_disagree = row['models_disagree']
        
        # Default values
        recommendation = 'HOLD'
        confidence = 0.0
        hold_reason = ''
        
        # Rule 1: If models disagree on direction, HOLD (unless very strong signal)
        if models_disagree:
            # Only override if one model is very confident and the other is borderline
            original_strength = abs(original_pred - market_total)
            learning_strength = abs(learning_pred - market_total)
            
            if original_strength > 1.5 and learning_strength < 0.5:
                # Original model is very confident, learning is neutral
                recommendation = 'OVER' if original_pred > market_total else 'UNDER'
                confidence = min(0.6, original_strength / 3.0)
                hold_reason = f'Strong original signal ({original_strength:.2f}) overrides weak learning'
            elif learning_strength > 1.5 and original_strength < 0.5:
                # Learning model is very confident, original is neutral
                recommendation = 'OVER' if learning_pred > market_total else 'UNDER'
                confidence = min(0.6, learning_strength / 3.0)
                hold_reason = f'Strong learning signal ({learning_strength:.2f}) overrides weak original'
            else:
                # True disagreement - HOLD
                recommendation = 'HOLD'
                confidence = 0.0
                hold_reason = f'Model disagreement: orig={original_pred:.1f} vs learn={learning_pred:.1f} (market={market_total:.1f})'
        
        # Rule 2: If models agree, use consensus with confidence based on agreement strength
        elif models_agree and row['original_direction'] != 'NEUTRAL':
            avg_pred = (original_pred + learning_pred) / 2
            edge_strength = abs(avg_pred - market_total)
            
            # Only bet if edge is meaningful (>0.3 runs)
            if edge_strength >= 0.3:
                recommendation = 'OVER' if avg_pred > market_total else 'UNDER'
                
                # Confidence based on edge strength and model agreement
                agreement_bonus = max(0, 1.0 - agreement / 2.0)  # Bonus for tight agreement
                edge_confidence = min(1.0, edge_strength / 2.0)   # Confidence from edge size
                confidence = min(0.9, (edge_confidence + agreement_bonus) / 2)
                
                hold_reason = f'Consensus: edge={edge_strength:.2f}, agreement={agreement:.2f}'
            else:
                recommendation = 'HOLD'
                confidence = 0.0
                hold_reason = f'Weak edge: {edge_strength:.2f} < 0.3 threshold'
        
        # Rule 3: Dead even with market (within 0.1 runs) - minimize holds
        elif abs((original_pred + learning_pred) / 2 - market_total) <= 0.1:
            # Even when dead even, look for slight edges
            avg_pred = (original_pred + learning_pred) / 2
            tiny_edge = avg_pred - market_total
            
            if abs(tiny_edge) >= 0.05:  # 0.05 run minimum edge
                recommendation = 'OVER' if tiny_edge > 0 else 'UNDER'
                confidence = 0.3  # Low confidence for tiny edges
                hold_reason = f'Micro edge: {tiny_edge:+.2f} runs'
            else:
                recommendation = 'HOLD'
                confidence = 0.0
                hold_reason = f'True dead even: avg={avg_pred:.2f} vs market={market_total:.2f}'
        
        # Rule 4: Both models neutral/weak - HOLD
        else:
            recommendation = 'HOLD'
            confidence = 0.0
            hold_reason = 'Both models neutral or weak signal'
        
        recommendations.append(recommendation)
        confidence_scores.append(confidence)
        hold_reasons.append(hold_reason)
    
    # Add enhanced columns
    enhanced_data['enhanced_recommendation'] = recommendations
    enhanced_data['enhanced_confidence'] = confidence_scores
    enhanced_data['hold_reason'] = hold_reasons
    
    # Statistics
    hold_count = sum(1 for r in recommendations if r == 'HOLD')
    over_count = sum(1 for r in recommendations if r == 'OVER')
    under_count = sum(1 for r in recommendations if r == 'UNDER')
    disagree_count = enhanced_data['models_disagree'].sum()
    
    log.info(f"📊 Enhanced betting recommendations:")
    log.info(f"   OVER: {over_count} games")
    log.info(f"   UNDER: {under_count} games")  
    log.info(f"   HOLD: {hold_count} games ({hold_count/len(enhanced_data)*100:.1f}%)")
    log.info(f"   Model disagreements: {disagree_count} games")
    log.info(f"   Average confidence: {np.mean(confidence_scores):.3f}")
    
    # Show hold reasons
    if hold_count > 0:
        log.info(f"📋 Hold reasons breakdown:")
        hold_reason_counts = enhanced_data[enhanced_data['enhanced_recommendation'] == 'HOLD']['hold_reason'].value_counts()
        for reason, count in hold_reason_counts.head(5).items():
            log.info(f"   '{reason}': {count} games")
    
    # Update database with enhanced recommendations
    try:
        log.info(f"💾 Updating database with enhanced betting logic...")
        
        with engine.begin() as conn:
            for idx, row in enhanced_data.iterrows():
                update_query = text("""
                    UPDATE enhanced_games 
                    SET enhanced_recommendation = :recommendation,
                        enhanced_confidence = :confidence,
                        models_agreement = :agreement,
                        models_disagree = :disagree,
                        hold_reason = :hold_reason,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE game_id = :game_id AND date = :date
                """)
                
                conn.execute(update_query, {
                    'recommendation': row['enhanced_recommendation'],
                    'confidence': float(row['enhanced_confidence']),
                    'agreement': float(row['model_agreement']),
                    'disagree': bool(row['models_disagree']),
                    'hold_reason': row['hold_reason'],
                    'game_id': row['game_id'],
                    'date': row['date']
                })
        
        log.info(f"✅ Updated database with enhanced betting recommendations")
        
    except Exception as e:
        log.warning(f"⚠️ Failed to update database with enhanced recommendations: {e}")
    
    return enhanced_data

def apply_dual_model_corrections(prob_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dual model logic to probability calculations and recommendations
    
    Args:
        prob_data: DataFrame from probabilities_and_ev.py with standard recommendations
        
    Returns:
        DataFrame with dual-model enhanced recommendations
    """
    
    log.info(f"🔧 Applying dual model corrections to {len(prob_data)} probability calculations")
    
    corrected_data = prob_data.copy()
    
    # Load dual model data if available
    try:
        from sqlalchemy import create_engine
        engine = create_engine('postgresql://user:password@localhost/mlb_predictions')
        
        game_ids = corrected_data['game_id'].tolist()
        query = text("""
            SELECT game_id, predicted_total_original, predicted_total_learning, enhanced_recommendation, enhanced_confidence
            FROM enhanced_games 
            WHERE game_id = ANY(:game_ids)
            AND predicted_total_original IS NOT NULL 
            AND predicted_total_learning IS NOT NULL
        """)
        
        dual_data = pd.read_sql(query, engine, params={'game_ids': game_ids})
        corrected_data = corrected_data.merge(dual_data, on='game_id', how='left')
        
        # Override standard recommendations with enhanced ones where available
        has_enhanced = ~corrected_data['enhanced_recommendation'].isna()
        corrected_data.loc[has_enhanced, 'recommendation'] = corrected_data.loc[has_enhanced, 'enhanced_recommendation']
        
        override_count = has_enhanced.sum()
        log.info(f"✅ Applied enhanced recommendations to {override_count} games")
        
    except Exception as e:
        log.warning(f"⚠️ Could not apply dual model corrections: {e}")
    
    return corrected_data

if __name__ == "__main__":
    # Test the enhanced betting logic
    from sqlalchemy import create_engine
    import os
    
    # Database connection
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/mlb_predictions')
    engine = create_engine(db_url)
    
    # Load today's games
    with engine.begin() as conn:
        today_games = pd.read_sql(text("""
            SELECT * FROM enhanced_games 
            WHERE date = '2025-08-24'
            AND predicted_total_original IS NOT NULL 
            AND predicted_total_learning IS NOT NULL
            AND market_total IS NOT NULL
        """), conn)
    
    if len(today_games) > 0:
        enhanced_games = enhance_betting_decisions(engine, today_games)
        print(f"\n📊 Enhanced betting analysis complete for {len(enhanced_games)} games")
        
        # Show sample results
        sample_cols = ['game_id', 'home_team', 'away_team', 'market_total', 
                      'predicted_total_original', 'predicted_total_learning',
                      'enhanced_recommendation', 'enhanced_confidence', 'hold_reason']
        
        print(enhanced_games[sample_cols].head(10).to_string(index=False))
    else:
        print("No games found with dual model predictions for 2025-08-24")
