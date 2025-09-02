"""
Daily Hitting Props Workflow

This script integrates hitting props predictions into the daily MLB workflow.
It fetches player logs, props odds, builds features, generates predictions,
and provides betting recommendations.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import json

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'ingestion'))
sys.path.append(str(Path(__file__).parent / 'odds'))
sys.path.append(str(Path(__file__).parent / 'features'))
sys.path.append(str(Path(__file__).parent / 'predict'))

from fetch_player_logs import MLBPlayerLogsFetcher
from fetch_props_odds import PropsOddsFetcher
from hitprops_predictor import HitPropsPredictor

log = logging.getLogger(__name__)

class DailyHittingWorkflow:
    """Orchestrates the daily hitting props workflow"""
    
    def __init__(self, engine):
        self.engine = engine
        self.player_fetcher = MLBPlayerLogsFetcher(engine)
        self.odds_fetcher = PropsOddsFetcher(engine)
        self.predictor = HitPropsPredictor(engine)
        
    def run_yesterday_backfill(self) -> dict:
        """Fetch yesterday's completed games and player logs"""
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        log.info(f"=== BACKFILL: Yesterday's Player Logs ({yesterday}) ===")
        
        try:
            logs_count = self.player_fetcher.fetch_date(yesterday)
            self.player_fetcher.refresh_materialized_views()
            
            return {
                'success': True,
                'date': yesterday,
                'player_logs': logs_count,
                'message': f"Successfully backfilled {logs_count} player logs"
            }
            
        except Exception as e:
            log.error(f"Error in yesterday backfill: {e}")
            return {
                'success': False,
                'date': yesterday,
                'error': str(e)
            }
    
    def fetch_today_props(self, use_mock: bool = False) -> dict:
        """Fetch today's hitting props odds"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        log.info(f"=== FETCH: Today's Props Odds ({today}) ===")
        
        try:
            props_count = self.odds_fetcher.fetch_all_props(today, use_mock=use_mock)
            
            return {
                'success': True,
                'date': today,
                'props_count': props_count,
                'mock': use_mock,
                'message': f"Successfully fetched {props_count} props odds"
            }
            
        except Exception as e:
            log.error(f"Error fetching props: {e}")
            return {
                'success': False,
                'date': today,
                'error': str(e)
            }
    
    def generate_predictions(self, output_file: str = None) -> dict:
        """Generate today's hitting props predictions"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        log.info(f"=== PREDICT: Today's Hitting Props ({today}) ===")
        
        try:
            # Generate predictions
            recommendations = self.predictor.get_top_recommendations(today)
            
            # Save to database
            saved_count = self.predictor.save_predictions(today)
            
            # Save to file if requested
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(recommendations, f, indent=2)
                
                log.info(f"Saved recommendations to {output_path}")
            
            summary = recommendations['summary']
            
            return {
                'success': True,
                'date': today,
                'saved_count': saved_count,
                'summary': summary,
                'output_file': output_file,
                'recommendations': recommendations
            }
            
        except Exception as e:
            log.error(f"Error generating predictions: {e}")
            return {
                'success': False,
                'date': today,
                'error': str(e)
            }
    
    def print_recommendations(self, recommendations: dict):
        """Print formatted recommendations to console"""
        
        summary = recommendations.get('summary', {})
        over_bets = recommendations.get('over_bets', [])
        under_bets = recommendations.get('under_bets', [])
        
        print("\n" + "="*80)
        print("🏟️  MLB HITTING PROPS RECOMMENDATIONS")
        print("="*80)
        
        print(f"📊 SUMMARY:")
        print(f"   Total Players Analyzed: {summary.get('total_players', 0)}")
        print(f"   OVER Recommendations: {summary.get('over_recs', 0)}")
        print(f"   UNDER Recommendations: {summary.get('under_recs', 0)}")
        print(f"   Average Confidence: {summary.get('avg_confidence', 0):.1%}")
        print(f"   Max Kelly Size: {summary.get('max_kelly', 0):.1%}")
        print(f"   Average Vig: {summary.get('avg_vig', 0):.1f}%")
        
        if over_bets:
            print(f"\n🎯 TOP OVER BETS (1+ Hit):")
            print("-" * 80)
            for i, bet in enumerate(over_bets[:8], 1):
                print(f"{i:2d}. {bet['player_name']} ({bet['team']}) vs {bet['opponent']}")
                print(f"    📈 Model: {bet['model_prob']:.1%} | Market: {bet['market_prob']:.1%} | Edge: {bet['edge']:+.1%}")
                print(f"    💰 Kelly: {bet['kelly_size']:.1%} | EV: {bet['ev']:+.3f} | Odds: {bet['odds']:+d}")
                print(f"    📋 L10: {bet['recent_form']} | vs P: {bet['vs_pitcher']}")
                print()
        
        if under_bets:
            print(f"\n❌ TOP UNDER BETS (0 Hits):")
            print("-" * 80)
            for i, bet in enumerate(under_bets[:5], 1):
                print(f"{i}. {bet['player_name']} ({bet['team']}) vs {bet['opponent']}")
                print(f"   📉 Model: {bet['model_prob_under']:.1%} | Edge: {bet['edge']:+.1%}")
                print(f"   💰 Kelly: {bet['kelly_size']:.1%} | EV: {bet['ev']:+.3f} | Odds: {bet['odds']:+d}")
                print(f"   📋 L10: {bet['recent_form']} | vs P: {bet['vs_pitcher']}")
                print()
        
        if not over_bets and not under_bets:
            print("\n⚠️  No betting recommendations found.")
            print("   This could mean:")
            print("   - No props odds available")
            print("   - No edges detected above threshold")
            print("   - Insufficient player data")
        
        print("="*80)
    
    def run_full_workflow(self, backfill_yesterday: bool = True, use_mock_props: bool = False, 
                         output_file: str = None) -> dict:
        """Run the complete daily hitting workflow"""
        
        log.info("🚀 Starting Daily Hitting Props Workflow")
        results = {
            'workflow_start': datetime.now().isoformat(),
            'steps': {}
        }
        
        # Step 1: Backfill yesterday's player logs
        if backfill_yesterday:
            backfill_result = self.run_yesterday_backfill()
            results['steps']['backfill'] = backfill_result
            
            if not backfill_result['success']:
                log.warning("Backfill failed, continuing with workflow...")
        
        # Step 2: Fetch today's props odds
        props_result = self.fetch_today_props(use_mock=use_mock_props)
        results['steps']['props'] = props_result
        
        if not props_result['success']:
            log.error("Props fetching failed, cannot continue")
            results['workflow_success'] = False
            return results
        
        # Step 3: Generate predictions
        predictions_result = self.generate_predictions(output_file)
        results['steps']['predictions'] = predictions_result
        
        if predictions_result['success']:
            # Print recommendations to console
            self.print_recommendations(predictions_result['recommendations'])
            results['workflow_success'] = True
        else:
            log.error("Predictions generation failed")
            results['workflow_success'] = False
        
        results['workflow_end'] = datetime.now().isoformat()
        log.info("✅ Daily Hitting Props Workflow Complete")
        
        return results

def main():
    """CLI interface for daily hitting workflow"""
    import argparse
    import os
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description='Run daily MLB hitting props workflow')
    parser.add_argument('--skip-backfill', action='store_true', help='Skip yesterday\'s player logs backfill')
    parser.add_argument('--mock-props', action='store_true', help='Use mock props data for testing')
    parser.add_argument('--output', type=str, help='Output JSON file for recommendations')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load environment
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    # Connect to database
    try:
        engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
        
        # Test connection
        with engine.connect() as conn:
            pass
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)
    
    # Initialize workflow
    workflow = DailyHittingWorkflow(engine)
    
    # Set output file with default
    output_file = args.output
    if not output_file and not args.quiet:
        output_file = f"exports/hitting_props_{datetime.now().strftime('%Y%m%d')}.json"
    
    # Run workflow
    try:
        results = workflow.run_full_workflow(
            backfill_yesterday=not args.skip_backfill,
            use_mock_props=args.mock_props,
            output_file=output_file
        )
        
        if results['workflow_success']:
            print(f"\n✅ Workflow completed successfully!")
            if output_file:
                print(f"📄 Results saved to: {output_file}")
        else:
            print(f"\n❌ Workflow failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        log.error(f"Workflow error: {e}")
        print(f"\n💥 Workflow crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
