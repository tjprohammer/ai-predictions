#!/usr/bin/env python3
"""
ENHANCED DAILY WORKFLOW ORCHESTRATOR
Integrates with your run_daily_workflow.bat to ensure:
1. Data injection happens first
2. Both models make predictions 
3. Completed games are processed for learning
4. Continuous improvement cycle runs

This replaces/enhances your existing daily workflow files.
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import psycopg2
import pandas as pd

class EnhancedDailyOrchestrator:
    def __init__(self):
        self.project_root = Path("S:/Projects/AI_Predictions")
        self.mlb_overs_dir = self.project_root / "mlb-overs"
        self.deployment_dir = self.mlb_overs_dir / "deployment"
        
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def log_step(self, step_name, status="RUNNING"):
        """Log workflow steps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_icon = "🔄" if status == "RUNNING" else "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} [{timestamp}] {step_name}")
        
        if status == "RUNNING":
            print("=" * (len(step_name) + 15))
    
    def run_command(self, command, cwd=None, description=""):
        """Run a command and return success status"""
        
        self.log_step(f"Running: {description or command}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.log_step(f"SUCCESS: {description or command}", "SUCCESS")
                return True, result.stdout
            else:
                self.log_step(f"FAILED: {description or command}", "ERROR")
                print(f"Error output: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            self.log_step(f"TIMEOUT: {description or command}", "ERROR")
            return False, "Command timed out"
        except Exception as e:
            self.log_step(f"EXCEPTION: {description or command}", "ERROR")
            print(f"Exception: {str(e)}")
            return False, str(e)
    
    def check_database_connection(self):
        """Verify database is accessible"""
        
        self.log_step("Checking database connection")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM enhanced_games WHERE date >= '2025-08-01';")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            self.log_step(f"Database connected - {count} recent games found", "SUCCESS")
            return True
        except Exception as e:
            self.log_step(f"Database connection failed: {str(e)}", "ERROR")
            return False
    
    def step_1_data_injection(self):
        """Step 1: Run data injection pipeline"""
        
        self.log_step("STEP 1: Data Injection Pipeline", "RUNNING")
        
        # Your existing data injection commands
        commands = [
            {
                'cmd': 'python team_data_backfill.py',
                'desc': 'Backfill team data',
                'cwd': self.project_root
            },
            {
                'cmd': 'python daily_learning_pipeline.py',
                'desc': 'Daily learning data pipeline',
                'cwd': self.project_root
            }
        ]
        
        for cmd_info in commands:
            success, output = self.run_command(
                cmd_info['cmd'], 
                cmd_info['cwd'], 
                cmd_info['desc']
            )
            if not success:
                print(f"❌ Data injection failed at: {cmd_info['desc']}")
                return False
        
        self.log_step("STEP 1: Data injection complete", "SUCCESS")
        return True
    
    def step_2_feature_building(self):
        """Step 2: Build features for prediction"""
        
        self.log_step("STEP 2: Feature Building Pipeline", "RUNNING")
        
        # Run feature building components
        commands = [
            {
                'cmd': 'python daily_api_workflow.py --stages markets,features',
                'desc': 'Market data and feature building',
                'cwd': self.deployment_dir
            }
        ]
        
        for cmd_info in commands:
            success, output = self.run_command(
                cmd_info['cmd'], 
                cmd_info['cwd'], 
                cmd_info['desc']
            )
            if not success:
                print(f"❌ Feature building failed at: {cmd_info['desc']}")
                return False
        
        self.log_step("STEP 2: Feature building complete", "SUCCESS")
        return True
    
    def step_3_dual_predictions(self, target_date):
        """Step 3: Generate dual model predictions"""
        
        self.log_step("STEP 3: Dual Model Predictions", "RUNNING")
        
        # Run continuous learning workflow for predictions
        success, output = self.run_command(
            f'python continuous_learning_workflow.py {target_date}',
            self.project_root,
            'Generate dual predictions'
        )
        
        if not success:
            print(f"❌ Dual predictions failed")
            return False
        
        self.log_step("STEP 3: Dual predictions complete", "SUCCESS")
        return True
    
    def step_4_odds_and_probabilities(self):
        """Step 4: Load odds and calculate probabilities"""
        
        self.log_step("STEP 4: Odds & Probabilities", "RUNNING")
        
        # Run odds and probability calculations
        commands = [
            {
                'cmd': 'python daily_api_workflow.py --stages odds,prob',
                'desc': 'Odds loading and probability calculations',
                'cwd': self.deployment_dir
            }
        ]
        
        for cmd_info in commands:
            success, output = self.run_command(
                cmd_info['cmd'], 
                cmd_info['cwd'], 
                cmd_info['desc']
            )
            if not success:
                print(f"❌ Odds/probabilities failed at: {cmd_info['desc']}")
                return False
        
        self.log_step("STEP 4: Odds & probabilities complete", "SUCCESS")
        return True
    
    def step_5_health_and_export(self):
        """Step 5: Health checks and data export"""
        
        self.log_step("STEP 5: Health Checks & Export", "RUNNING")
        
        # Run health checks and export
        commands = [
            {
                'cmd': 'python daily_api_workflow.py --stages health,export,audit',
                'desc': 'Health checks, export, and audit',
                'cwd': self.deployment_dir
            }
        ]
        
        for cmd_info in commands:
            success, output = self.run_command(
                cmd_info['cmd'], 
                cmd_info['cwd'], 
                cmd_info['desc']
            )
            if not success:
                print(f"❌ Health/export failed at: {cmd_info['desc']}")
                return False
        
        self.log_step("STEP 5: Health checks & export complete", "SUCCESS")
        return True
    
    def step_6_continuous_learning(self):
        """Step 6: Process completed games for learning"""
        
        self.log_step("STEP 6: Continuous Learning Update", "RUNNING")
        
        # Run continuous learning to process completed games
        success, output = self.run_command(
            'python continuous_learning_workflow.py',
            self.project_root,
            'Process completed games for learning'
        )
        
        if not success:
            print(f"❌ Continuous learning failed")
            return False
        
        self.log_step("STEP 6: Continuous learning complete", "SUCCESS")
        return True
    
    def generate_daily_summary(self, target_date):
        """Generate summary of today's workflow"""
        
        self.log_step("Generating Daily Summary", "RUNNING")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # Get today's predictions
            predictions_query = """
            SELECT 
                COUNT(*) as total_games,
                AVG(predicted_total) as avg_original_pred,
                AVG(learning_prediction) as avg_learning_pred,
                AVG(prediction_difference) as avg_difference
            FROM enhanced_games
            WHERE date = %s
              AND predicted_total IS NOT NULL
              AND learning_prediction IS NOT NULL;
            """
            
            df = pd.read_sql(predictions_query, conn, params=[target_date])
            
            # Get recent performance
            performance_query = """
            SELECT 
                COUNT(*) as completed_games,
                AVG(CASE WHEN ABS(predicted_total - total_runs) <= 0.5 THEN 1.0 ELSE 0.0 END) as original_accuracy,
                AVG(CASE WHEN ABS(learning_prediction - total_runs) <= 0.5 THEN 1.0 ELSE 0.0 END) as learning_accuracy
            FROM enhanced_games
            WHERE date >= %s
              AND date < %s
              AND total_runs IS NOT NULL
              AND predicted_total IS NOT NULL
              AND learning_prediction IS NOT NULL;
            """
            
            # Last 7 days performance
            week_ago = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
            perf_df = pd.read_sql(performance_query, conn, params=[week_ago, target_date])
            
            conn.close()
            
            summary = {
                'date': target_date,
                'predictions': {
                    'total_games': int(df.iloc[0]['total_games']),
                    'avg_original_prediction': float(df.iloc[0]['avg_original_pred'] or 0),
                    'avg_learning_prediction': float(df.iloc[0]['avg_learning_pred'] or 0),
                    'avg_difference': float(df.iloc[0]['avg_difference'] or 0)
                },
                'recent_performance': {
                    'completed_games': int(perf_df.iloc[0]['completed_games']),
                    'original_accuracy': float(perf_df.iloc[0]['original_accuracy'] or 0),
                    'learning_accuracy': float(perf_df.iloc[0]['learning_accuracy'] or 0)
                }
            }
            
            # Save summary
            summary_file = self.project_root / f"daily_summary_{target_date}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📊 DAILY SUMMARY ({target_date}):")
            print(f"   Predictions made: {summary['predictions']['total_games']}")
            print(f"   Avg original pred: {summary['predictions']['avg_original_prediction']:.1f}")
            print(f"   Avg learning pred: {summary['predictions']['avg_learning_prediction']:.1f}")
            print(f"   Learning accuracy (7d): {summary['recent_performance']['learning_accuracy']:.1%}")
            
            self.log_step("Daily summary generated", "SUCCESS")
            return summary
            
        except Exception as e:
            print(f"❌ Summary generation failed: {str(e)}")
            return None
    
    def run_full_workflow(self, target_date=None):
        """Run the complete enhanced daily workflow"""
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🚀 ENHANCED DAILY WORKFLOW")
        print(f"   Target Date: {target_date}")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Pre-check: Database connection
        if not self.check_database_connection():
            print("❌ Workflow aborted - database connection failed")
            return False
        
        # Step 1: Data Injection
        if not self.step_1_data_injection():
            print("❌ Workflow aborted - data injection failed")
            return False
        
        # Step 2: Feature Building
        if not self.step_2_feature_building():
            print("❌ Workflow aborted - feature building failed")
            return False
        
        # Step 3: Dual Predictions
        if not self.step_3_dual_predictions(target_date):
            print("❌ Workflow aborted - dual predictions failed")
            return False
        
        # Step 4: Odds & Probabilities
        if not self.step_4_odds_and_probabilities():
            print("❌ Workflow aborted - odds/probabilities failed")
            return False
        
        # Step 5: Health & Export
        if not self.step_5_health_and_export():
            print("❌ Workflow aborted - health/export failed")
            return False
        
        # Step 6: Continuous Learning
        if not self.step_6_continuous_learning():
            print("❌ Workflow aborted - continuous learning failed")
            return False
        
        # Generate summary
        summary = self.generate_daily_summary(target_date)
        
        print(f"\n🏆 ENHANCED DAILY WORKFLOW COMPLETE!")
        print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   All 6 steps successful for {target_date}")
        
        if summary:
            print(f"   Generated predictions for {summary['predictions']['total_games']} games")
        
        return True

def main():
    print("🚀 ENHANCED DAILY WORKFLOW ORCHESTRATOR")
    print("   Integrating data injection, dual predictions, and continuous learning")
    print("=" * 80)
    
    orchestrator = EnhancedDailyOrchestrator()
    
    # Get target date from command line or use today
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run complete workflow
    success = orchestrator.run_full_workflow(target_date)
    
    if success:
        print("\n✅ Ready for betting predictions!")
        sys.exit(0)
    else:
        print("\n❌ Workflow failed - check logs above")
        sys.exit(1)

if __name__ == "__main__":
    main()
