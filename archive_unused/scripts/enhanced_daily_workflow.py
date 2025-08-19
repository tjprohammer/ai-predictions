#!/usr/bin/env python3
"""
Enhanced Daily Workflow for MLB Predictions
Integrates the enhanced ML model into the gameday pipeline
"""

import sys
import os
import subprocess
from datetime import datetime, date
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

def run_script(script_path, description, critical=True):
    """Run a script and handle errors"""
    print(f"\n🔄 {description}")
    print(f"   Running: {script_path}")
    
    try:
        # Change to scripts directory for relative imports
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"   ✅ {description} completed successfully")
            if result.stdout:
                print(f"   📊 Output: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ {description} failed")
            print(f"   Error: {result.stderr}")
            if critical:
                print(f"   ⚠️  Critical step failed, stopping pipeline")
                return False
            else:
                print(f"   ⚠️  Non-critical step failed, continuing...")
                return True
                
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {description} timed out after 5 minutes")
        return not critical
        
    except Exception as e:
        print(f"   ❌ Error running {description}: {e}")
        return not critical

def main():
    """Enhanced Daily Workflow Pipeline"""
    
    print("🚀 ENHANCED MLB PREDICTIONS DAILY WORKFLOW")
    print("=" * 50)
    print(f"📅 Date: {date.today()}")
    print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
    
    # Pipeline steps with enhanced ML integration
    pipeline_steps = [
        {
            "script": "01_fetch_daily_games.py",
            "description": "Step 1: Fetch Today's Games",
            "critical": True
        },
        {
            "script": "02_update_pitcher_stats.py", 
            "description": "Step 2: Update Pitcher Statistics",
            "critical": True
        },
        {
            "script": "03_update_team_stats.py",
            "description": "Step 3: Update Team Statistics", 
            "critical": True
        },
        {
            "script": "03_update_weather_data.py",
            "description": "Step 4: Update Weather Data",
            "critical": False  # Weather not critical
        },
        {
            "script": "fetch_betting_lines.py",
            "description": "Step 5: Fetch Betting Lines",
            "critical": False  # Betting lines not critical
        },
        {
            "script": "enhanced_ml_predictions.py",
            "description": "Step 6: Generate Enhanced ML Predictions",
            "critical": True
        },
        {
            "script": "generate_app_recommendations.py",
            "description": "Step 7: Generate App Recommendations",
            "critical": True
        }
    ]
    
    # Execute pipeline
    success_count = 0
    total_steps = len(pipeline_steps)
    
    for i, step in enumerate(pipeline_steps, 1):
        print(f"\n{'='*20} {i}/{total_steps} {'='*20}")
        
        success = run_script(
            step["script"], 
            step["description"], 
            step["critical"]
        )
        
        if success:
            success_count += 1
        elif step["critical"]:
            print(f"\n❌ PIPELINE FAILED at critical step: {step['description']}")
            print(f"   Completed {success_count}/{total_steps} steps")
            return False
    
    # Pipeline completion summary
    print(f"\n{'='*50}")
    if success_count == total_steps:
        print("🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"✅ All {total_steps} steps completed")
        
        # Quick validation
        print(f"\n🔍 VALIDATION:")
        
        # Check if recommendations were generated
        recommendations_file = Path("../web_app/daily_recommendations.json")
        if recommendations_file.exists():
            print("   ✅ Daily recommendations file created")
            
            # Try to read and validate
            try:
                import json
                with open(recommendations_file) as f:
                    data = json.load(f)
                    
                game_count = len(data.get('games', []))
                bet_count = len(data.get('best_bets', []))
                model_version = data.get('model_version', 'unknown')
                
                print(f"   📊 {game_count} games analyzed")
                print(f"   🎯 {bet_count} betting opportunities found")
                print(f"   🤖 Model version: {model_version}")
                
                if 'enhanced' in model_version:
                    print("   🚀 Enhanced ML model successfully used!")
                
            except Exception as e:
                print(f"   ⚠️  Could not validate recommendations: {e}")
        else:
            print("   ❌ Daily recommendations file not found")
        
        # Check database status
        try:
            import psycopg2
            conn = psycopg2.connect("postgresql://mlb:mlbpass@localhost:5432/mlb")
            cursor = conn.cursor()
            
            # Check today's games count
            cursor.execute("SELECT COUNT(*) FROM games WHERE date = CURRENT_DATE")
            games_today = cursor.fetchone()[0]
            print(f"   🎮 {games_today} games in database for today")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"   ⚠️  Could not check database: {e}")
            
    else:
        print(f"⚠️  PIPELINE COMPLETED WITH WARNINGS")
        print(f"✅ {success_count}/{total_steps} steps completed successfully")
        
    print(f"\n🕐 Finished: {datetime.now().strftime('%H:%M:%S')}")
    return success_count == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
