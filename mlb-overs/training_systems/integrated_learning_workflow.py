#!/usr/bin/env python3
"""
Integrated Learning Workflow
Combines existing daily workflow with continuous learning stages
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from daily_learning_pipeline import DailyLearningPipeline
from live_learning_predictor import LiveLearningPredictor

class IntegratedLearningWorkflow:
    def __init__(self):
        self.learning_pipeline = DailyLearningPipeline()
        self.live_predictor = LiveLearningPredictor()
        
    def run_standard_workflow(self, date):
        """Run the existing daily API workflow"""
        print(f"🔄 RUNNING STANDARD WORKFLOW FOR {date}")
        print("=" * 50)
        
        try:
            # Run the existing daily workflow
            cmd = [
                'python', 'daily_api_workflow.py',
                '--date', date,
                '--stages', 'markets,features,predict,odds,health,prob,export,audit'
            ]
            
            result = subprocess.run(
                cmd, 
                cwd='mlb-overs/deployment',
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Standard workflow completed successfully")
                return True
            else:
                print(f"❌ Standard workflow failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running standard workflow: {e}")
            return False
    
    def run_learning_update(self, date):
        """Run learning update for previous day's results"""
        print(f"\\n🧠 RUNNING LEARNING UPDATE FOR {date}")
        print("=" * 50)
        
        try:
            # Update models with previous day's results
            success = self.learning_pipeline.update_daily_model(date)
            
            if success:
                print("✅ Learning update completed successfully")
                
                # Show updated model status
                status = self.learning_pipeline.get_model_status()
                print(f"🤖 Active Model: {status.get('model_type', 'Unknown')}")
                print(f"📊 Performance: MAE={status.get('performance', {}).get('mae', 'N/A'):.2f}")
                return True
            else:
                print("❌ Learning update failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during learning update: {e}")
            return False
    
    def generate_enhanced_predictions(self, date):
        """Generate enhanced predictions using learning models"""
        print(f"\\n🎯 GENERATING ENHANCED PREDICTIONS FOR {date}")
        print("=" * 50)
        
        try:
            # Get learning predictions
            predictions = self.live_predictor.get_learning_predictions_for_today()
            
            if predictions:
                # Save enhanced predictions
                output_file = f"enhanced_predictions_{date}.json"
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2, default=str)
                
                print(f"✅ Enhanced predictions saved to: {output_file}")
                
                # Generate betting summary
                self.generate_betting_summary(predictions, date)
                return True
            else:
                print("⚠️  No enhanced predictions generated")
                return False
                
        except Exception as e:
            print(f"❌ Error generating enhanced predictions: {e}")
            return False
    
    def generate_betting_summary(self, predictions, date):
        """Generate a concise betting summary"""
        
        learning_bets = [p for p in predictions if p['learning_recommendation'] in ['OVER', 'UNDER']]
        current_bets = [p for p in predictions if p['current_recommendation'] in ['OVER', 'UNDER']]
        
        summary = {
            'date': date,
            'total_games': len(predictions),
            'learning_bets': len(learning_bets),
            'current_bets': len(current_bets),
            'high_confidence_learning': [p for p in learning_bets if p['learning_edge'] >= 1.0],
            'consensus_bets': [],
            'learning_only_bets': [],
            'model_performance': {}
        }
        
        # Find consensus and learning-only bets
        for pred in predictions:
            learning_rec = pred['learning_recommendation']
            current_rec = pred['current_recommendation']
            
            if learning_rec in ['OVER', 'UNDER'] and current_rec in ['OVER', 'UNDER']:
                if learning_rec == current_rec:
                    summary['consensus_bets'].append(pred)
            elif learning_rec in ['OVER', 'UNDER'] and current_rec == 'HOLD':
                summary['learning_only_bets'].append(pred)
        
        # Add performance metrics if available
        completed_games = [p for p in predictions if p['is_completed']]
        if completed_games:
            learning_errors = [abs(p['learning_prediction'] - p['actual_total']) for p in completed_games]
            current_errors = [abs(p['current_prediction'] - p['actual_total']) for p in completed_games if p['current_prediction']]
            
            summary['model_performance'] = {
                'learning_mae': sum(learning_errors) / len(learning_errors) if learning_errors else None,
                'current_mae': sum(current_errors) / len(current_errors) if current_errors else None,
                'completed_games': len(completed_games)
            }
        
        # Save summary
        summary_file = f"betting_summary_{date}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print key insights
        print(f"\\n📊 BETTING SUMMARY FOR {date}:")
        print(f"   🎯 Learning Model Bets: {len(learning_bets)}")
        print(f"   🔧 Current System Bets: {len(current_bets)}")
        print(f"   🤝 Consensus Bets: {len(summary['consensus_bets'])}")
        print(f"   🧠 Learning-Only Bets: {len(summary['learning_only_bets'])}")
        print(f"   🔥 High Confidence: {len(summary['high_confidence_learning'])}")
        
        if summary['high_confidence_learning']:
            print("\\n🔥 HIGH CONFIDENCE LEARNING PICKS:")
            for bet in summary['high_confidence_learning']:
                print(f"   • {bet['game']}: {bet['learning_recommendation']} {bet['learning_prediction']:.1f} (edge: {bet['learning_edge']:.1f})")
        
        if summary['consensus_bets']:
            print("\\n🤝 CONSENSUS PICKS (Both models agree):")
            for bet in summary['consensus_bets']:
                print(f"   • {bet['game']}: {bet['learning_recommendation']} (L:{bet['learning_prediction']:.1f}, C:{bet['current_prediction']:.1f})")
        
        print(f"\\n💾 Summary saved to: {summary_file}")
    
    def run_morning_workflow(self, target_date=None):
        """Complete morning workflow: learn from yesterday, predict today"""
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        yesterday = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print("🌅 INTEGRATED MORNING WORKFLOW")
        print("=" * 60)
        print(f"Target Date: {target_date}")
        print(f"Learning from: {yesterday}")
        print()
        
        workflow_success = True
        
        # Stage 1: Run standard workflow for today
        if not self.run_standard_workflow(target_date):
            print("⚠️  Standard workflow failed, continuing with learning...")
            workflow_success = False
        
        # Stage 2: Learn from yesterday's results
        if not self.run_learning_update(yesterday):
            print("⚠️  Learning update failed, continuing...")
            workflow_success = False
        
        # Stage 3: Generate enhanced predictions for today
        if not self.generate_enhanced_predictions(target_date):
            print("⚠️  Enhanced predictions failed...")
            workflow_success = False
        
        # Summary
        print(f"\\n🏁 MORNING WORKFLOW COMPLETE")
        print("=" * 40)
        if workflow_success:
            print("✅ All stages completed successfully")
        else:
            print("⚠️  Some stages had issues, check logs above")
        
        return workflow_success
    
    def run_evening_workflow(self, target_date=None):
        """Evening workflow: collect results, update models"""
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print("🌙 INTEGRATED EVENING WORKFLOW")
        print("=" * 60)
        print(f"Processing results for: {target_date}")
        print()
        
        # Check if games are completed
        try:
            # Test if we can run learning update (requires completed games)
            success = self.run_learning_update(target_date)
            
            if success:
                print(f"✅ Evening workflow complete - models updated with {target_date} results")
                
                # Show performance on today's games
                print(f"\\n📊 TODAY'S PERFORMANCE:")
                self.learning_pipeline.test_production_model_on_today()
                
            else:
                print(f"⚠️  Games may not be completed yet for {target_date}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error in evening workflow: {e}")
            return False

def main():
    """Main execution with command line options"""
    
    workflow = IntegratedLearningWorkflow()
    
    if len(sys.argv) < 2:
        print("🤖 INTEGRATED LEARNING WORKFLOW")
        print("=" * 50)
        print("Usage:")
        print("  python integrated_learning_workflow.py morning [date]")
        print("  python integrated_learning_workflow.py evening [date]")
        print("  python integrated_learning_workflow.py full [date]")
        print()
        print("Examples:")
        print("  python integrated_learning_workflow.py morning")
        print("  python integrated_learning_workflow.py evening 2025-08-20")
        print("  python integrated_learning_workflow.py full 2025-08-21")
        return
    
    workflow_type = sys.argv[1].lower()
    target_date = sys.argv[2] if len(sys.argv) > 2 else None
    
    if workflow_type == 'morning':
        workflow.run_morning_workflow(target_date)
    elif workflow_type == 'evening':
        workflow.run_evening_workflow(target_date)
    elif workflow_type == 'full':
        # Run both morning and evening
        if workflow.run_morning_workflow(target_date):
            print("\\n" + "="*60)
            workflow.run_evening_workflow(target_date)
    else:
        print(f"❌ Unknown workflow type: {workflow_type}")
        print("Use: morning, evening, or full")

if __name__ == "__main__":
    main()
