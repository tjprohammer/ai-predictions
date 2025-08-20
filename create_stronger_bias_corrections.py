"""
Create stronger bias corrections based on comprehensive 30-day analysis.
This addresses the systematic under-prediction issue.
"""

import json
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_systematic_bias():
    """Analyze systematic bias across recent predictions"""
    
    # Database connection
    conn = psycopg2.connect(
        host="localhost",
        database="mlb",
        user="mlbuser",
        password="mlbpass"
    )
    
    # Get last 30 days of completed games with predictions
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    query = """
    SELECT 
        date,
        home_team,
        away_team,
        total_runs as actual_total,
        predicted_total,
        market_total,
        predicted_total - total_runs as prediction_error,
        market_total - total_runs as market_error,
        predicted_total - market_total as pred_vs_market
    FROM enhanced_games 
    WHERE date >= %s 
    AND date <= %s
    AND total_runs IS NOT NULL 
    AND predicted_total IS NOT NULL
    AND market_total IS NOT NULL
    ORDER BY date DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    
    logger.info(f"🔍 COMPREHENSIVE BIAS ANALYSIS - Last 30 Days")
    logger.info(f"📅 Date Range: {start_date} to {end_date}")
    logger.info(f"📊 Total Games Analyzed: {len(df)}")
    
    if len(df) == 0:
        logger.error("❌ No completed games found with predictions!")
        return None
    
    # Calculate comprehensive bias metrics
    mean_predicted = df['predicted_total'].mean()
    mean_actual = df['actual_total'].mean()
    mean_market = df['market_total'].mean()
    
    prediction_bias = df['prediction_error'].mean()  # negative = under-prediction
    market_bias = df['market_error'].mean()
    pred_vs_market_bias = df['pred_vs_market'].mean()
    
    mae = df['prediction_error'].abs().mean()
    rmse = (df['prediction_error'] ** 2).mean() ** 0.5
    
    logger.info(f"\n🎯 BIAS ANALYSIS RESULTS:")
    logger.info(f"   Mean Predicted: {mean_predicted:.2f} runs")
    logger.info(f"   Mean Actual:    {mean_actual:.2f} runs")  
    logger.info(f"   Mean Market:    {mean_market:.2f} runs")
    logger.info(f"\n📈 ERROR METRICS:")
    logger.info(f"   Prediction Bias: {prediction_bias:.3f} runs ({'UNDER' if prediction_bias < 0 else 'OVER'}-predicting)")
    logger.info(f"   Market Bias:     {market_bias:.3f} runs")
    logger.info(f"   Pred vs Market:  {pred_vs_market_bias:.3f} runs")
    logger.info(f"   MAE:            {mae:.3f} runs")
    logger.info(f"   RMSE:           {rmse:.3f} runs")
    
    # Analyze by scoring ranges
    df['scoring_range'] = pd.cut(df['actual_total'], 
                                bins=[0, 7, 10, float('inf')], 
                                labels=['Low (≤7)', 'Mid (8-10)', 'High (11+)'],
                                include_lowest=True)
    
    range_analysis = df.groupby('scoring_range').agg({
        'prediction_error': ['count', 'mean'],
        'actual_total': 'mean',
        'predicted_total': 'mean'
    }).round(3)
    
    logger.info(f"\n📊 BIAS BY SCORING RANGE:")
    for range_name in ['Low (≤7)', 'Mid (8-10)', 'High (11+)']:
        if range_name in range_analysis.index:
            count = range_analysis.loc[range_name, ('prediction_error', 'count')]
            bias = range_analysis.loc[range_name, ('prediction_error', 'mean')]
            logger.info(f"   {range_name:<12}: {count:3.0f} games | Bias: {bias:+.2f} runs")
    
    # Calculate recommended global adjustment
    # Use the systematic bias with a confidence factor
    recommended_adjustment = abs(prediction_bias) * 1.1  # 10% buffer for robustness
    
    logger.info(f"\n💡 RECOMMENDED GLOBAL ADJUSTMENT:")
    logger.info(f"   Current systematic bias: {prediction_bias:.3f} runs")
    logger.info(f"   Recommended adjustment: +{recommended_adjustment:.3f} runs")
    
    return {
        'games_analyzed': len(df),
        'prediction_bias': prediction_bias,
        'recommended_adjustment': recommended_adjustment,
        'mae': mae,
        'rmse': rmse,
        'range_analysis': range_analysis,
        'pred_vs_market_bias': pred_vs_market_bias
    }

def create_enhanced_bias_corrections(analysis_results):
    """Create enhanced bias corrections based on analysis"""
    
    if not analysis_results:
        logger.error("❌ No analysis results provided!")
        return
    
    recommended_adj = analysis_results['recommended_adjustment']
    
    # Create comprehensive bias corrections
    bias_corrections = {
        "global_adjustment": round(recommended_adj, 3),
        "scoring_range_adjustments": {
            "Low (≤7)": round(-0.2, 2),    # Slightly reduce low scoring predictions
            "Mid (8-10)": round(0.0, 2),   # Neutral for mid-range
            "High (11+)": round(0.4, 2),   # Boost high scoring predictions more
            "Very High (14+)": round(0.6, 2)  # Extra boost for very high scoring
        },
        "confidence_adjustments": {
            "High Confidence (>80%)": round(0.1, 2),
            "Medium Confidence (60-80%)": round(0.0, 2), 
            "Low Confidence (<60%)": round(-0.1, 2)
        },
        "temperature_adjustments": {
            "Hot (80+°F)": 0.2,
            "Warm (70-79°F)": 0.1,
            "Cool (60-69°F)": 0.0,
            "Cold (<60°F)": -0.1
        },
        "venue_adjustments": {
            "COL": 0.3,  # Coors Field
            "TEX": 0.2,  # Globe Life Field
            "LAD": 0.1,  # Dodger Stadium
            "BOS": 0.1,  # Fenway Park
            "CIN": 0.1,  # Great American Ball Park
            "MIL": -0.1, # American Family Field  
            "OAK": -0.2, # Oakland Coliseum
            "SEA": -0.1  # T-Mobile Park
        },
        "pitcher_quality_adjustments": {
            "Elite SP (ERA < 3.0)": round(-0.2, 2),
            "Good SP (ERA 3.0-4.0)": round(0.0, 2),
            "Average SP (ERA 4.0-5.0)": round(0.1, 2),
            "Poor SP (ERA > 5.0)": round(0.3, 2)
        },
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "games_analyzed": analysis_results['games_analyzed'],
            "systematic_bias": round(analysis_results['prediction_bias'], 3),
            "mae": round(analysis_results['mae'], 3),
            "pred_vs_market_bias": round(analysis_results['pred_vs_market_bias'], 3),
            "version": "enhanced_30day_v2"
        }
    }
    
    return bias_corrections

def save_bias_corrections(corrections):
    """Save bias corrections to both main and deployment files"""
    
    main_file = 's:\\Projects\\AI_Predictions\\model_bias_corrections.json'
    deployment_file = 's:\\Projects\\AI_Predictions\\mlb-overs\\deployment\\model_bias_corrections.json'
    
    # Save to main file
    with open(main_file, 'w') as f:
        json.dump(corrections, f, indent=2)
    logger.info(f"✅ Saved bias corrections to: {main_file}")
    
    # Save to deployment file  
    with open(deployment_file, 'w') as f:
        json.dump(corrections, f, indent=2)
    logger.info(f"✅ Saved bias corrections to: {deployment_file}")
    
    logger.info(f"\n🎯 NEW BIAS CORRECTIONS SUMMARY:")
    logger.info(f"   Global Adjustment: +{corrections['global_adjustment']} runs")
    logger.info(f"   Games Analyzed: {corrections['metadata']['games_analyzed']}")
    logger.info(f"   Systematic Bias: {corrections['metadata']['systematic_bias']} runs")
    logger.info(f"   MAE: {corrections['metadata']['mae']} runs")

def main():
    """Main execution function"""
    logger.info("🚀 Creating Enhanced Bias Corrections...")
    
    # Analyze systematic bias
    analysis = analyze_systematic_bias()
    if not analysis:
        return
    
    # Create enhanced corrections
    corrections = create_enhanced_bias_corrections(analysis)
    
    # Save corrections
    save_bias_corrections(corrections)
    
    logger.info("✅ Enhanced bias corrections created and saved!")

if __name__ == "__main__":
    main()
