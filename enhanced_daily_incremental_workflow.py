#!/usr/bin/env python3
"""
Enhanced Daily Workflow with Incremental Learning
================================================

This script enhances the existing daily_api_workflow.py with incremental learning
from the Ultra 80 system. It provides a complete daily cycle:

1. Collect yesterday's scores and update models
2. Run standard daily workflow (markets, features, predictions)
3. Learn incrementally from completed games
4. Generate predictions using multiple systems
5. Export results for UI and betting analysis

Usage:
    python enhanced_daily_incremental_workflow.py --target-date 2025-08-27
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Import existing daily workflow
try:
    from daily_api_workflow import (
        collect_markets_and_weather_and_pitcher_data, 
        engineer_and_align, 
        run_odds_collection,
        run_health_check,
        run_prob_export,
        export_to_ui,
        run_audit_analysis
    )
    WORKFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  daily_api_workflow.py not found - running in incremental-only mode")
    WORKFLOW_AVAILABLE = False

# Add pipeline path
pipeline_path = Path(__file__).parent / 'mlb-overs' / 'pipelines'
sys.path.append(str(pipeline_path))

try:
    from incremental_ultra_80_system import IncrementalUltra80System
    INCREMENTAL_AVAILABLE = True
except ImportError:
    print("⚠️  incremental_ultra_80_system.py not found")
    INCREMENTAL_AVAILABLE = False


class EnhancedDailyWorkflow:
    """Enhanced daily workflow with incremental learning integration"""
    
    def __init__(self):
        self.incremental_system = None
        if INCREMENTAL_AVAILABLE:
            self.incremental_system = IncrementalUltra80System()
    
    def run_complete_daily_cycle(self, target_date: str):
        """Run the complete enhanced daily workflow"""
        print(f"🚀 ENHANCED DAILY WORKFLOW with Incremental Learning")
        print(f"🎯 Target Date: {target_date}")
        print("="*80)
        
        results = {
            'target_date': target_date,
            'workflow_predictions': None,
            'incremental_predictions': None,
            'learning_results': None,
            'combined_insights': None
        }
        
        # Step 1: Run standard daily workflow if available
        if WORKFLOW_AVAILABLE:
            print("\n📊 STEP 1: Running standard daily workflow...")
            results['workflow_predictions'] = self._run_standard_workflow(target_date)
        else:
            print("\n⚠️  STEP 1: Standard workflow not available - skipping")
        
        # Step 2: Run incremental learning cycle
        if INCREMENTAL_AVAILABLE:
            print(f"\n🧠 STEP 2: Running incremental learning cycle...")
            results['incremental_predictions'], results['learning_results'] = self._run_incremental_cycle(target_date)
        else:
            print("\n⚠️  STEP 2: Incremental system not available - skipping")
        
        # Step 3: Combine insights from multiple systems
        print(f"\n🔍 STEP 3: Combining insights from multiple prediction systems...")
        results['combined_insights'] = self._combine_prediction_insights(results)
        
        # Step 4: Export comprehensive results
        print(f"\n📁 STEP 4: Exporting comprehensive results...")
        self._export_comprehensive_results(results)
        
        return results
    
    def _run_standard_workflow(self, target_date: str):
        """Run the standard daily API workflow"""
        try:
            print("   📡 Collecting markets, weather, and pitcher data...")
            collect_markets_and_weather_and_pitcher_data()
            
            print("   🔧 Engineering features and generating predictions...")
            predictions = engineer_and_align()
            
            print("   💰 Collecting odds data...")
            run_odds_collection()
            
            print("   🏥 Running health check...")
            run_health_check()
            
            print("   📊 Exporting probability data...")
            run_prob_export()
            
            print("   🎨 Exporting to UI...")
            export_to_ui()
            
            print("   📋 Running audit analysis...")
            run_audit_analysis()
            
            print(f"   ✅ Standard workflow complete")
            return predictions
            
        except Exception as e:
            print(f"   ❌ Error in standard workflow: {e}")
            return None
    
    def _run_incremental_cycle(self, target_date: str):
        """Run the incremental learning cycle"""
        try:
            # Load existing state
            state_loaded = self.incremental_system.load_state()
            if state_loaded:
                print('   🔄 Loaded existing incremental model state...')
            else:
                print('   🆕 Starting fresh incremental learning...')
            
            # Learn from recent completed games
            print("   📚 Learning from recent completed games...")
            yesterday = datetime.now() - timedelta(days=1)
            week_ago = yesterday - timedelta(days=7)
            
            learning_results = self.incremental_system.team_level_incremental_learn(
                start_date=week_ago.strftime('%Y-%m-%d'),
                end_date=yesterday.strftime('%Y-%m-%d')
            )
            
            if learning_results:
                print(f"   ✅ Learned from {len(learning_results.get('predictions', []))} games")
                print(f"   📊 Coverage: {learning_results['final_coverage']:.1%} | MAE: {learning_results['final_mae']:.2f} | ROI: {learning_results['final_roi']:+.2%}")
                
                # Save updated state
                self.incremental_system.save_state()
                print("   💾 Updated model state saved")
            
            # Generate predictions for target date
            if self.incremental_system.is_fitted:
                print(f"   🔮 Predicting games for {target_date}...")
                predictions = self.incremental_system.predict_future_slate(target_date, outdir='outputs')
                
                if not predictions.empty:
                    print(f"   ✅ Generated {len(predictions)} incremental predictions")
                    return predictions, learning_results
                else:
                    print(f"   ⚠️  No games found for {target_date}")
                    return None, learning_results
            else:
                print("   ⚠️  Incremental models not fitted - need more training data")
                return None, learning_results
                
        except Exception as e:
            print(f"   ❌ Error in incremental cycle: {e}")
            return None, None
    
    def _combine_prediction_insights(self, results):
        """Combine insights from multiple prediction systems"""
        insights = {
            'consensus_predictions': [],
            'system_agreements': [],
            'high_confidence_bets': [],
            'system_comparison': {}
        }
        
        workflow_preds = results.get('workflow_predictions')
        incremental_preds = results.get('incremental_predictions')
        
        if workflow_preds is not None and incremental_preds is not None:
            print("   🔍 Comparing predictions from multiple systems...")
            
            # Find common games between systems
            if hasattr(workflow_preds, 'iterrows') and hasattr(incremental_preds, 'iterrows'):
                for _, inc_pred in incremental_preds.iterrows():
                    game_key = f"{inc_pred['away_team']}_{inc_pred['home_team']}"
                    
                    # Look for matching game in workflow predictions
                    # (This would need adaptation based on workflow prediction format)
                    
                    insights['consensus_predictions'].append({
                        'game': f"{inc_pred['away_team']} @ {inc_pred['home_team']}",
                        'incremental_pred': inc_pred['pred_total'],
                        'incremental_confidence': inc_pred['trust'],
                        'incremental_edge': inc_pred['diff'],
                        'incremental_ev': inc_pred['ev']
                    })
            
            insights['system_comparison'] = {
                'workflow_games': len(workflow_preds) if hasattr(workflow_preds, '__len__') else 0,
                'incremental_games': len(incremental_preds) if hasattr(incremental_preds, '__len__') else 0,
                'systems_available': ['workflow', 'incremental']
            }
            
        elif incremental_preds is not None:
            print("   🧠 Using incremental predictions only...")
            for _, pred in incremental_preds.iterrows():
                insights['consensus_predictions'].append({
                    'game': f"{pred['away_team']} @ {pred['home_team']}",
                    'incremental_pred': pred['pred_total'],
                    'incremental_confidence': pred['trust'],
                    'incremental_edge': pred['diff'],
                    'incremental_ev': pred['ev']
                })
            
            # Identify high-confidence bets
            high_ev_mask = (incremental_preds['ev'] >= 0.05) & (incremental_preds['trust'] >= 0.6)
            if high_ev_mask.any():
                insights['high_confidence_bets'] = incremental_preds[high_ev_mask].to_dict('records')
                print(f"   💎 Found {len(insights['high_confidence_bets'])} high-confidence betting opportunities")
            
            insights['system_comparison'] = {
                'workflow_games': 0,
                'incremental_games': len(incremental_preds),
                'systems_available': ['incremental']
            }
        
        return insights
    
    def _export_comprehensive_results(self, results):
        """Export comprehensive results to multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export to JSON for programmatic access
        json_path = f"outputs/enhanced_daily_results_{timestamp}.json"
        
        # Prepare JSON-serializable results
        json_results = {
            'target_date': results['target_date'],
            'timestamp': timestamp,
            'systems_used': [],
            'summary': {}
        }
        
        if results['workflow_predictions'] is not None:
            json_results['systems_used'].append('workflow')
        
        if results['incremental_predictions'] is not None:
            json_results['systems_used'].append('incremental')
            json_results['incremental_summary'] = {
                'games_predicted': len(results['incremental_predictions']),
                'avg_confidence': float(results['incremental_predictions']['trust'].mean()),
                'high_ev_opportunities': len(results['incremental_predictions'][results['incremental_predictions']['ev'] >= 0.05])
            }
        
        if results['learning_results'] is not None:
            json_results['learning_summary'] = {
                'games_learned_from': len(results['learning_results'].get('predictions', [])),
                'final_coverage': results['learning_results']['final_coverage'],
                'final_mae': results['learning_results']['final_mae'],
                'final_roi': results['learning_results']['final_roi']
            }
        
        json_results['combined_insights'] = results['combined_insights']
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"   📄 Comprehensive results exported to: {json_path}")
        
        # Export betting recommendations if any
        if results['combined_insights'] and results['combined_insights']['high_confidence_bets']:
            bet_path = f"outputs/high_confidence_bets_{timestamp}.json"
            with open(bet_path, 'w') as f:
                json.dump(results['combined_insights']['high_confidence_bets'], f, indent=2, default=str)
            print(f"   💎 High-confidence bets exported to: {bet_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Daily Workflow with Incremental Learning')
    parser.add_argument('--target-date', type=str, help='Target date for predictions (YYYY-MM-DD)', 
                        default=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
    parser.add_argument('--force-reset', action='store_true', help='Reset incremental model state')
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.force_reset:
        os.environ['FORCE_RESET'] = '1'
    
    # Run enhanced workflow
    workflow = EnhancedDailyWorkflow()
    results = workflow.run_complete_daily_cycle(args.target_date)
    
    print(f"\n🏁 Enhanced daily workflow complete!")
    print(f"🎯 Target date: {args.target_date}")
    print(f"🤖 Systems used: {', '.join(results['combined_insights']['system_comparison']['systems_available'])}")
    
    if results['combined_insights']['high_confidence_bets']:
        print(f"💎 High-confidence betting opportunities: {len(results['combined_insights']['high_confidence_bets'])}")
    else:
        print("🔒 No high-confidence betting opportunities identified")


if __name__ == '__main__':
    main()
