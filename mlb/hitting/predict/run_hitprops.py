#!/usr/bin/env python3
"""
Run MLB Hitting Props Predictions
Orchestrates the complete hitting props prediction workflow

Usage:
    python run_hitprops.py [DATE]
    
Example:
    python run_hitprops.py 2025-08-29
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced_hitprops_predictor import EnhancedHitPropsPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def refresh_materialized_views(database_url: str):
    """Refresh hitting materialized views"""
    
    log.info("Refreshing hitting materialized views...")
    
    engine = create_engine(database_url)
    
    with engine.begin() as conn:
        # Refresh views in correct order (dependencies)
        conn.execute(text("REFRESH MATERIALIZED VIEW mv_bvp_agg;"))
        log.info("✓ Refreshed mv_bvp_agg")
        
        conn.execute(text("REFRESH MATERIALIZED VIEW mv_hitter_form;"))
        log.info("✓ Refreshed mv_hitter_form")
        
        conn.execute(text("REFRESH MATERIALIZED VIEW mv_pa_distribution;"))
        log.info("✓ Refreshed mv_pa_distribution")
    
    log.info("All materialized views refreshed successfully")

def check_data_quality(database_url: str, target_date: str) -> bool:
    """Check if we have sufficient data for predictions"""
    
    log.info(f"Checking data quality for {target_date}")
    
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        # Check for recent player game logs
        recent_logs_query = text("""
            SELECT COUNT(*) as log_count
            FROM player_game_logs 
            WHERE date >= :recent_date
        """)
        
        recent_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        result = conn.execute(recent_logs_query, {'recent_date': recent_date}).fetchone()
        
        log_count = result[0] if result else 0
        
        if log_count < 100:  # Arbitrary threshold
            log.warning(f"Only {log_count} player logs in past 7 days - may not have enough data")
            return False
        
        log.info(f"✓ Found {log_count} recent player logs - data quality OK")
        return True

def run_hitting_props_workflow(target_date: str = None):
    """Run the complete hitting props prediction workflow"""
    
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    log.info(f"Starting hitting props workflow for {target_date}")
    
    database_url = "postgresql://mlbuser:mlbpass@localhost/mlb"
    
    try:
        # Step 1: Check data quality
        if not check_data_quality(database_url, target_date):
            log.error("Data quality check failed - aborting workflow")
            return False
        
        # Step 2: Refresh materialized views
        refresh_materialized_views(database_url)
        
        # Step 3: Generate predictions
        log.info("Generating hitting props predictions...")
        predictor = EnhancedHitPropsPredictor(database_url)
        predictions = predictor.predict_all_props(target_date)
        
        if predictions.empty:
            log.warning(f"No predictions generated for {target_date}")
            return False
        
        # Step 4: Save predictions
        saved_count = predictor.save_predictions(predictions)
        
        # Step 5: Summary report
        log.info(f"=== HITTING PROPS WORKFLOW COMPLETE ===")
        log.info(f"Date: {target_date}")
        log.info(f"Total predictions: {len(predictions)}")
        log.info(f"Predictions saved: {saved_count}")
        
        # Market breakdown
        market_summary = predictions.groupby('market').agg({
            'player_id': 'count',
            'confidence_score': 'mean',
            'prob_over': 'mean'
        }).round(3)
        
        log.info(f"Market breakdown:")
        for market, stats in market_summary.iterrows():
            log.info(f"  {market}: {stats['player_id']} players, "
                    f"avg confidence: {stats['confidence_score']}, "
                    f"avg prob: {stats['prob_over']}")
        
        # Best picks (if we have odds)
        best_picks = predictions[
            (predictions['ev_over'] > 0.05) | (predictions['ev_under'] > 0.05)
        ].nlargest(10, ['ev_over', 'ev_under'])
        
        if not best_picks.empty:
            log.info(f"Top EV picks:")
            for _, pick in best_picks.iterrows():
                side = 'OVER' if pick['ev_over'] > pick['ev_under'] else 'UNDER'
                ev = max(pick['ev_over'], pick['ev_under'])
                log.info(f"  {pick['player_name']} {pick['market']} {side} (EV: {ev:.3f})")
        
        return True
        
    except Exception as e:
        log.error(f"Workflow failed: {e}", exc_info=True)
        return False

def main():
    """Main entry point"""
    
    # Parse command line arguments
    target_date = None
    if len(sys.argv) > 1:
        try:
            # Validate date format
            datetime.strptime(sys.argv[1], '%Y-%m-%d')
            target_date = sys.argv[1]
        except ValueError:
            log.error(f"Invalid date format: {sys.argv[1]}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Run workflow
    success = run_hitting_props_workflow(target_date)
    
    if success:
        log.info("Hitting props workflow completed successfully")
        sys.exit(0)
    else:
        log.error("Hitting props workflow failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
