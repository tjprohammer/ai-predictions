#!/usr/bin/env python3
"""
🔧 Pipeline Learning Integration
Integrates 20-session learnings into your existing prediction pipeline.
"""

import sys
import os
sys.path.append('mlb-overs')

from enhanced_learning_integration import EnhancedLearningIntegrator
from learning_impact_monitor import LearningImpactMonitor
import pandas as pd
import joblib
import json
from datetime import datetime

def integrate_learnings_into_pipeline():
    """Main integration function to apply learnings to your pipeline"""
    
    print("🎯 INTEGRATING 20-SESSION LEARNINGS INTO PIPELINE")
    print("="*60)
    
    # Step 1: Initialize systems
    integrator = EnhancedLearningIntegrator()
    monitor = LearningImpactMonitor()
    
    # Step 2: Train enhanced model with learnings
    print("\n📊 Step 1: Training Enhanced Model with Learnings")
    df = integrator.load_enhanced_data()
    
    if df.empty:
        print("❌ No data available for training")
        return False
    
    # Apply feature engineering from learnings
    df = integrator.engineer_enhanced_features(df)
    df = integrator.select_optimal_features(df)
    
    # Train model
    model, metadata = integrator.train_enhanced_model(df)
    
    if model is None:
        print("❌ Model training failed")
        return False
    
    # Step 3: Evaluate improvement
    print(f"\n📈 Step 2: Evaluating Improvement")
    improvement = integrator.evaluate_improvement(
        metadata['avg_mae'], 
        metadata['avg_r2']
    )
    
    # Step 4: Save enhanced model if improved
    if improvement['is_improving']:
        model_file = integrator.save_enhanced_model(model, metadata)
        print(f"✅ Enhanced model saved: {model_file}")
        
        # Update pipeline configuration
        update_pipeline_config(model_file, metadata)
        
    else:
        print("⚠️  Model performance not improved enough for deployment")
    
    # Step 5: Generate monitoring report
    print(f"\n📊 Step 3: Generating Performance Report")
    report = monitor.generate_performance_report(days_back=14)
    
    # Step 6: Create improvement recommendations
    generate_improvement_recommendations(improvement, report)
    
    return True

def update_pipeline_config(model_file, metadata):
    """Update pipeline configuration with enhanced model"""
    
    config = {
        'enhanced_model_path': model_file,
        'model_performance': {
            'mae': metadata['avg_mae'],
            'r2': metadata['avg_r2']
        },
        'feature_count': len(metadata['feature_importance']),
        'top_features': metadata['feature_importance'].head(10).to_dict('records'),
        'updated': datetime.now().isoformat()
    }
    
    # Save config for pipeline to use
    import json
    with open('enhanced_pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️  Pipeline configuration updated")

def generate_improvement_recommendations(improvement, report):
    """Generate specific recommendations for continued improvement"""
    
    print(f"\n🎯 IMPROVEMENT RECOMMENDATIONS")
    print("-" * 40)
    
    if improvement['is_improving']:
        print("✅ Current Status: Learning integration is working!")
        
        if improvement['overall_improvement'] > 10:
            print("🎉 Excellent improvement! Consider:")
            print("   • Deploy enhanced model to production")
            print("   • Expand learning to more sessions")
            print("   • Apply similar learning to other prediction targets")
        else:
            print("📈 Good progress! Next steps:")
            print("   • Monitor performance for 1-2 weeks")
            print("   • Fine-tune feature selection")
            print("   • Add more recent training data")
    
    else:
        print("⚠️  Current Status: Need refinement")
        print("🔧 Recommended actions:")
        print("   • Review feature engineering strategy")
        print("   • Check for data quality issues")
        print("   • Consider different model architectures")
        print("   • Analyze high-error predictions")
    
    # Specific feature recommendations
    print(f"\n📊 Feature Strategy:")
    print("   • Core Baseball features should dominate (45%+ importance)")
    print("   • Pitching features second priority (25%+ importance)")
    print("   • Focus on RBI, Runs, Hits as top predictors")
    print("   • Maintain ~182 optimal features")
    
    # Monitoring recommendations  
    if report and 'trend' in report:
        trend_direction = report['trend']['trend_direction']
        print(f"\n📈 Monitoring Strategy:")
        print(f"   • Current trend: {trend_direction}")
        print("   • Check performance weekly")
        print("   • Alert if MAE > 1.2 runs consistently")
        print("   • Target: MAE < 0.9 runs (best session level)")

def run_daily_learning_check():
    """Daily check to ensure learnings are still effective"""
    
    print("🔍 DAILY LEARNING EFFECTIVENESS CHECK")
    print("="*45)
    
    monitor = LearningImpactMonitor()
    
    # Quick 7-day performance check
    report = monitor.generate_performance_report(days_back=7)
    
    if report:
        performance = report['performance']
        
        # Alert conditions
        if performance['overall_mae'] > 1.2:
            print("🚨 ALERT: Performance degraded! MAE > 1.2 runs")
            print("   Action needed: Review recent predictions")
            
        elif performance['improvement_from_baseline'] < 0:
            print("⚠️  WARNING: Below baseline performance")
            print("   Action: Monitor closely, may need retraining")
            
        else:
            print("✅ Performance within acceptable range")
        
        # Trend check
        if 'trend' in report:
            trend = report['trend']['trend_direction']
            if "DECLINING" in trend:
                print("📉 TREND ALERT: Performance declining")
                print("   Recommendation: Investigate and retrain if needed")
    
    return report

def show_integration_status():
    """Show current status of learning integration"""
    
    print("📊 LEARNING INTEGRATION STATUS")
    print("="*35)
    
    # Check if enhanced model exists
    try:
        with open('enhanced_pipeline_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"✅ Enhanced Model: Active")
        print(f"   📁 Model File: {config.get('enhanced_model_path', 'Unknown')}")
        print(f"   📊 Performance: MAE={config['model_performance']['mae']:.3f}, R²={config['model_performance']['r2']:.3f}")
        print(f"   🔧 Features: {config.get('feature_count', 'Unknown')}")
        print(f"   📅 Updated: {config.get('updated', 'Unknown')}")
        
        # Show top features
        print(f"\n🔍 Top Enhanced Features:")
        for i, feature in enumerate(config.get('top_features', [])[:5], 1):
            print(f"   {i}. {feature['feature']}: {feature['importance']:.4f}")
            
    except FileNotFoundError:
        print("❌ No enhanced model configuration found")
        print("   Run integrate_learnings_into_pipeline() first")
    
    # Check recent performance
    try:
        monitor = LearningImpactMonitor()
        df = monitor.load_recent_predictions(7)
        
        if not df.empty:
            current_mae = df['absolute_error'].mean()
            print(f"\n📈 Recent Performance (7 days):")
            print(f"   Games: {len(df)}")
            print(f"   Current MAE: {current_mae:.3f} runs")
            
            if current_mae < 1.0:
                print("   Status: 🎉 Excellent")
            elif current_mae < 1.2:
                print("   Status: ✅ Good")
            else:
                print("   Status: ⚠️  Needs attention")
        
    except Exception as e:
        print(f"⚠️  Could not check recent performance: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "integrate":
            integrate_learnings_into_pipeline()
        elif command == "check":
            run_daily_learning_check()
        elif command == "status":
            show_integration_status()
        else:
            print("Usage: python pipeline_integration.py [integrate|check|status]")
    else:
        # Default: full integration
        integrate_learnings_into_pipeline()
