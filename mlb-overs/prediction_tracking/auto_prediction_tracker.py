#!/usr/bin/env python3
"""
Automated Prediction Recording Script
Automatically records predictions for tracking when they're generated
"""

import requests
import json
import sys
from datetime import datetime
import os

def record_predictions_for_date(date_str: str):
    """Record predictions for the given date"""
    
    print(f"📊 Recording predictions for {date_str}...")
    
    # Call the API to record predictions
    try:
        response = requests.post(f"http://localhost:8000/api/prediction-tracking/record/{date_str}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['message']}")
            print(f"   - Recorded: {data['recorded']} predictions")
            print(f"   - Learning predictions: {data['learning_predictions']}")
            return True
        else:
            print(f"❌ Error recording predictions: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def update_results_for_date(date_str: str):
    """Update actual results for completed games"""
    
    print(f"🔄 Updating results for {date_str}...")
    
    try:
        response = requests.post(f"http://localhost:8000/api/prediction-tracking/update-results/{date_str}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['message']}")
            return True
        else:
            print(f"❌ Error updating results: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def check_tracking_status():
    """Check the current tracking status"""
    
    print("📈 Checking tracking status...")
    
    try:
        # Get recent tracking data
        response = requests.get("http://localhost:8000/api/prediction-tracking/recent?days=3")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Recent Tracking Summary:")
            print(f"   - Total games: {data['total_games']}")
            print(f"   - Completed: {data['summary']['total_completed']}")
            print(f"   - Current model correct: {data['summary']['current_correct']}")
            print(f"   - Learning model correct: {data['summary']['learning_correct']}")
            
            if data['summary']['current_avg_error']:
                print(f"   - Current model avg error: {data['summary']['current_avg_error']:.2f}")
            if data['summary']['learning_avg_error']:
                print(f"   - Learning model avg error: {data['summary']['learning_avg_error']:.2f}")
            
            return True
        else:
            print(f"❌ Error checking status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python auto_prediction_tracker.py record [date]")
        print("  python auto_prediction_tracker.py update [date]") 
        print("  python auto_prediction_tracker.py status")
        print("")
        print("Examples:")
        print("  python auto_prediction_tracker.py record 2025-08-21")
        print("  python auto_prediction_tracker.py update 2025-08-20")
        print("  python auto_prediction_tracker.py status")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "record":
        date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
        success = record_predictions_for_date(date)
        sys.exit(0 if success else 1)
    
    elif command == "update":
        date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
        success = update_results_for_date(date)
        sys.exit(0 if success else 1)
    
    elif command == "status":
        success = check_tracking_status()
        sys.exit(0 if success else 1)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
