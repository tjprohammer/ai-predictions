"""
Enhanced AI Analysis and Calibrated Predictions for MLB Games
Provides detailed analysis and calibrated predictions with confidence levels
"""

import numpy as np
import psycopg2
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Team name mapping for standardization
TEAM_NAME_MAPPING = {
    'ATL': 'Atlanta Braves', 'AZ': 'Arizona Diamondbacks', 'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'NYY': 'New York Yankees', 'ATH': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SEA': 'Seattle Mariners',
    'SF': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals'
}

def get_db_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser', 
            password='mlbpass'
        )
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_team_recent_performance(team_name: str, days: int = 5) -> Dict[str, float]:
    """Get recent team offensive performance"""
    conn = get_db_connection()
    if not conn:
        return {'recent_runs_pg': 4.5, 'recent_woba': 0.320, 'games': 0}
    
    try:
        cursor = conn.cursor()
        
        # Try both full name and abbreviation mapping
        search_names = [team_name]
        abbrev = None
        for abbr, full_name in TEAM_NAME_MAPPING.items():
            if full_name == team_name:
                search_names.append(abbr)
                abbrev = abbr
                break
        
        # Search for team data with both names - get most recent games regardless of date
        placeholders = ','.join(['%s'] * len(search_names))
        query = f"""
        SELECT 
            AVG(runs_pg) as recent_runs_pg,
            AVG(woba) as recent_woba,
            AVG(wrcplus) as recent_wrcplus,
            COUNT(*) as games
        FROM (
            SELECT runs_pg, woba, wrcplus 
            FROM teams_offense_daily 
            WHERE team IN ({placeholders})
            AND runs_pg IS NOT NULL
            ORDER BY date DESC 
            LIMIT %s
        ) recent_games
        """
        
        cursor.execute(query, search_names + [days])
        
        result = cursor.fetchone()
        if result and result[0]:
            return {
                'recent_runs_pg': float(result[0]),
                'recent_woba': float(result[1]) if result[1] else 0.320,
                'recent_wrcplus': int(result[2]) if result[2] else 100,
                'games': int(result[3])
            }
        else:
            # Fallback to league average
            return {'recent_runs_pg': 4.5, 'recent_woba': 0.320, 'recent_wrcplus': 100, 'games': 0}
            
    except Exception as e:
        print(f"Error getting team performance for {team_name}: {e}")
        return {'recent_runs_pg': 4.5, 'recent_woba': 0.320, 'recent_wrcplus': 100, 'games': 0}
    finally:
        conn.close()

def get_team_form_adjustment(runs_pg: float) -> Dict[str, Any]:
    """Calculate team form adjustment based on recent performance"""
    if runs_pg >= 6.5:
        return {'adjustment': 0.5, 'category': 'Very Hot', 'description': f'{runs_pg:.1f} R/G - Elite offense'}
    elif runs_pg >= 5.5:
        return {'adjustment': 0.3, 'category': 'Hot', 'description': f'{runs_pg:.1f} R/G - Above average offense'}
    elif runs_pg <= 2.5:
        return {'adjustment': -0.5, 'category': 'Very Cold', 'description': f'{runs_pg:.1f} R/G - Struggling offense'}
    elif runs_pg <= 3.5:
        return {'adjustment': -0.3, 'category': 'Cold', 'description': f'{runs_pg:.1f} R/G - Below average offense'}
    else:
        return {'adjustment': 0.0, 'category': 'Average', 'description': f'{runs_pg:.1f} R/G - League average'}

def generate_enhanced_ai_analysis(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive AI analysis with detailed insights"""
    try:
        # Safe float conversion with proper null handling
        base_pred_total = float(game_data.get('predicted_total') or 0)
        market_total = float(game_data.get('market_total') or 0)
        edge = float(game_data.get('edge') or 0)
        confidence = float(game_data.get('confidence') or 0)
        recommendation = game_data.get('recommendation', 'HOLD')
        
        home_team = game_data.get('home_team', 'Home')
        away_team = game_data.get('away_team', 'Away')
        
        # Weather factors
        temp = float(game_data.get('temperature') or 75)
        wind_speed = float(game_data.get('wind_speed') or 10)
        wind_direction = game_data.get('wind_direction', 'N')
        
        # Pitching factors
        home_sp_era = float(game_data.get('home_sp_season_era') or 4.00)
        away_sp_era = float(game_data.get('away_sp_season_era') or 4.00)
        home_sp_name = game_data.get('home_sp_name', 'TBD')
        away_sp_name = game_data.get('away_sp_name', 'TBD')
        
        # Offense factors
        home_avg = float(game_data.get('home_team_avg') or 0.250)
        away_avg = float(game_data.get('away_team_avg') or 0.250)
        
        # Confidence is already in percentage format (85.0 = 85%)
        # No conversion needed - confidence values are correct as-is
        
        # Calculate enhanced confidence level
        confidence_level = "HIGH" if confidence >= 70 else "MEDIUM" if confidence >= 50 else "LOW"
        
        # Primary factors analysis
        primary_factors = []
        supporting_factors = []
        key_insights = []
        risk_factors = []
        
        # Pitching analysis
        avg_era = (home_sp_era + away_sp_era) / 2
        if avg_era < 3.50:
            primary_factors.append(f"Strong pitching matchup - both starters under 3.50 ERA ({home_sp_name}: {home_sp_era:.2f}, {away_sp_name}: {away_sp_era:.2f})")
        elif avg_era > 5.00:
            primary_factors.append(f"Weak pitching matchup - high ERAs favor offensive production ({home_sp_name}: {home_sp_era:.2f}, {away_sp_name}: {away_sp_era:.2f})")
        else:
            supporting_factors.append(f"Average pitching matchup ({home_sp_name}: {home_sp_era:.2f} vs {away_sp_name}: {away_sp_era:.2f})")
        
        # Pitching advantage
        era_diff = abs(home_sp_era - away_sp_era)
        if era_diff > 1.0:
            better_pitcher = home_sp_name if home_sp_era < away_sp_era else away_sp_name
            better_era = min(home_sp_era, away_sp_era)
            primary_factors.append(f"Significant pitching advantage to {better_pitcher} ({better_era:.2f} ERA) could suppress opponent's offense")
        
        # Weather analysis
        if temp > 85:
            supporting_factors.append(f"Hot weather ({temp}°F) typically increases offensive production")
        elif temp < 60:
            supporting_factors.append(f"Cold weather ({temp}°F) may suppress offensive numbers")
        
        if wind_speed > 15:
            if 'out' in wind_direction.lower():
                primary_factors.append(f"Strong outbound wind ({wind_speed}mph) strongly favors home runs and higher scoring")
            elif 'in' in wind_direction.lower():
                primary_factors.append(f"Strong inbound wind ({wind_speed}mph) will significantly suppress offensive production")
            else:
                supporting_factors.append(f"Strong crosswind ({wind_speed}mph) creates unpredictable conditions")
        elif wind_speed > 10:
            if 'out' in wind_direction.lower():
                supporting_factors.append(f"Moderate outbound wind ({wind_speed}mph) slightly favors offense")
            elif 'in' in wind_direction.lower():
                supporting_factors.append(f"Moderate inbound wind ({wind_speed}mph) slightly suppresses offense")
        
        # Offensive analysis
        
        # NEW: Get recent team performance data
        home_performance = get_team_recent_performance(home_team, days=5)
        away_performance = get_team_recent_performance(away_team, days=5)
        
        home_form = get_team_form_adjustment(home_performance['recent_runs_pg'])
        away_form = get_team_form_adjustment(away_performance['recent_runs_pg'])
        
        total_team_adjustment = home_form['adjustment'] + away_form['adjustment']
        
        # Team form analysis - this is HUGE for predictions!
        if home_form['category'] in ['Very Hot', 'Hot']:
            primary_factors.append(f"{home_team} offensive form: {home_form['description']} (last {home_performance['games']} games)")
        elif home_form['category'] in ['Very Cold', 'Cold']:
            primary_factors.append(f"{home_team} offensive struggles: {home_form['description']} (last {home_performance['games']} games)")
        else:
            supporting_factors.append(f"{home_team} recent form: {home_form['description']}")
            
        if away_form['category'] in ['Very Hot', 'Hot']:
            primary_factors.append(f"{away_team} offensive form: {away_form['description']} (last {away_performance['games']} games)")
        elif away_form['category'] in ['Very Cold', 'Cold']:
            primary_factors.append(f"{away_team} offensive struggles: {away_form['description']} (last {away_performance['games']} games)")
        else:
            supporting_factors.append(f"{away_team} recent form: {away_form['description']}")
        
        # Combined team form insight
        if total_team_adjustment >= 0.6:
            key_insights.append(f"Both teams hot offensively - expecting {total_team_adjustment:+.1f} runs above model prediction")
        elif total_team_adjustment <= -0.6:
            key_insights.append(f"Both teams struggling offensively - expecting {total_team_adjustment:.1f} runs below model prediction")
        elif abs(total_team_adjustment) >= 0.4:
            key_insights.append(f"Significant team form difference - net {total_team_adjustment:+.1f} run impact expected")
        
        # Traditional offensive metrics (keeping existing logic)
        combined_avg = (home_avg + away_avg) / 2
        if combined_avg > 0.270:
            supporting_factors.append(f"Strong offensive teams - combined .{combined_avg:.3f} batting average")
        elif combined_avg < 0.240:
            supporting_factors.append(f"Weak offensive teams - combined .{combined_avg:.3f} batting average")
        
        # Edge analysis - More aggressive thresholds for actionable picks
        if abs(edge) < 0.15:
            risk_factors.append("Very minimal edge vs market - proceed with caution")
            recommendation_reasoning = "Hold recommended - prediction too close to market"
        elif abs(edge) < 0.35:
            if confidence >= 60:
                key_insights.append(f"Small edge ({edge:+.2f}) but solid confidence creates potential value")
                recommendation_reasoning = f"Light {recommendation} with {abs(edge):.2f} run edge and {confidence:.0f}% confidence"
            else:
                risk_factors.append(f"Small edge ({edge:+.2f}) with lower confidence - high risk")
                recommendation_reasoning = "Hold recommended due to low confidence on small edge"
        elif abs(edge) < 0.75:
            if confidence >= 50:
                primary_factors.append(f"Solid {abs(edge):.2f} run edge vs market with decent confidence")
                key_insights.append("Good betting value opportunity identified")
                recommendation_reasoning = f"{recommendation} recommended with {abs(edge):.2f} run edge"
            else:
                risk_factors.append(f"Moderate edge ({edge:+.2f}) but low confidence suggests caution")
                recommendation_reasoning = f"Lean {recommendation} but proceed carefully"
        else:
            if confidence >= 45:
                primary_factors.append(f"Strong {abs(edge):.1f} run edge vs market - excellent value")
                key_insights.append("High-value betting opportunity - strong edge detected")
                recommendation_reasoning = f"Strong {recommendation} with {abs(edge):.1f} run edge"
            else:
                risk_factors.append(f"Large edge ({edge:+.1f}) but very low confidence - model uncertainty")
                recommendation_reasoning = f"Cautious {recommendation} - verify with additional analysis"
        
        # Confidence-based insights
        if confidence >= 80:
            key_insights.append("Model shows very high confidence in prediction accuracy")
        elif confidence >= 65:
            key_insights.append("Model shows good confidence in prediction")
        elif confidence < 40:
            risk_factors.append("Lower confidence prediction - consider smaller bet sizing or avoid")
        
        # Market analysis
        if market_total > 10.5:
            key_insights.append("High-total game - weather and pitching factors especially important")
        elif market_total < 7.5:
            key_insights.append("Low-total game - even small offensive improvements could create value")
        
        # Special venue considerations
        venue_name = game_data.get('venue_name', '').lower()
        if 'coors' in venue_name:
            key_insights.append("Coors Field high altitude significantly increases offensive production")
        elif 'fenway' in venue_name:
            key_insights.append("Fenway Park's Green Monster can impact scoring unpredictably")
        elif 'yankee' in venue_name:
            key_insights.append("Yankee Stadium's short right field porch favors left-handed power")
        
        # Add default factors if none found
        if not primary_factors and not supporting_factors:
            supporting_factors.append(f"Standard MLB matchup with typical scoring expectations")
        
        if not risk_factors and confidence < 65:
            risk_factors.append("Standard model uncertainty - monitor line movement")
        
        return {
            "prediction_summary": f"AI predicts {base_pred_total:.1f} total runs vs market {market_total:.1f} ({edge:+.1f} edge)",
            "confidence_level": confidence_level,
            "primary_factors": primary_factors,
            "supporting_factors": supporting_factors,
            "risk_factors": risk_factors,
            "recommendation_reasoning": recommendation_reasoning,
            "key_insights": key_insights
        }
        
    except Exception as e:
        # Fallback analysis if anything fails
        return {
            "prediction_summary": f"AI analysis error: {str(e)}",
            "confidence_level": "LOW",
            "primary_factors": [],
            "supporting_factors": [],
            "risk_factors": ["Analysis generation failed"],
            "recommendation_reasoning": "Hold due to analysis error",
            "key_insights": []
        }

def generate_calibrated_predictions(game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate calibrated predictions with enhanced confidence metrics and team data"""
    try:
        base_pred = float(game_data.get('predicted_total') or 0)
        market_total = float(game_data.get('market_total') or 0)
        confidence = float(game_data.get('confidence') or 0)
        edge = float(game_data.get('edge') or 0)
        
        # Get team performance adjustments
        home_team = game_data.get('home_team', 'Home')
        away_team = game_data.get('away_team', 'Away')
        
        home_performance = get_team_recent_performance(home_team, days=5)
        away_performance = get_team_recent_performance(away_team, days=5)
        
        home_form = get_team_form_adjustment(home_performance['recent_runs_pg'])
        away_form = get_team_form_adjustment(away_performance['recent_runs_pg'])
        
        total_team_adjustment = home_form['adjustment'] + away_form['adjustment']
        
        # Apply team form adjustments (30% weight to recent form)
        team_adjusted_pred = base_pred + (total_team_adjustment * 0.3)
        team_adjusted_edge = team_adjusted_pred - market_total
        
        # Apply calibration based on confidence and edge - MORE AGGRESSIVE
        calibration_factor = 1.0
        
        # Less conservative calibration
        if confidence < 40:
            calibration_factor = 0.8  # Move 20% closer to market (was 30%)
        elif confidence < 60:
            calibration_factor = 0.9  # Move 10% closer to market (was 15%)
        
        # Calibrated prediction with team adjustments
        calibrated_total = market_total + (team_adjusted_edge * calibration_factor)
        calibrated_edge = calibrated_total - market_total
        
        # Boost confidence if team data supports prediction
        confidence_boost = 0
        if abs(total_team_adjustment) > 0.4:  # Strong team form signal
            confidence_boost = 10
        elif abs(total_team_adjustment) > 0.2:  # Moderate team form signal
            confidence_boost = 5
            
        calibrated_confidence = min(confidence + confidence_boost, 95)
        
        # Generate recommendation based on LOWERED thresholds for more action
        if abs(calibrated_edge) < 0.2:  # Was 0.3 - more aggressive
            calibrated_recommendation = "HOLD"
        elif calibrated_edge > 0.2:
            calibrated_recommendation = "OVER"
        else:
            calibrated_recommendation = "UNDER"
        
        # Add team context to calibration reason
        calibration_reason = "Standard calibration"
        if total_team_adjustment != 0:
            team_impact = "positive" if total_team_adjustment > 0 else "negative"
            calibration_reason = f"Team form adjustment ({team_impact} {abs(total_team_adjustment):.1f} runs)"
        if calibration_factor < 1.0:
            calibration_reason += " + confidence adjustment"
        
        return {
            "predicted_total": round(calibrated_total, 1),
            "confidence": round(calibrated_confidence, 1),
            "recommendation": calibrated_recommendation,
            "edge": round(calibrated_edge, 2),
            "calibration_applied": calibration_factor < 1.0 or total_team_adjustment != 0,
            "calibration_reason": calibration_reason,
            "team_adjustment": round(total_team_adjustment, 2)
        }
    except Exception as e:
        return None

def calculate_enhanced_confidence_metrics(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate enhanced confidence metrics for UI display"""
    try:
        confidence = float(game_data.get('confidence') or 0)
        edge = abs(float(game_data.get('edge') or 0))
        
        # Determine confidence level with LOWERED thresholds for more action
        if confidence >= 65 and edge >= 0.4:  # Was 75/0.5 - more aggressive
            confidence_level = "HIGH"
            is_high_confidence = True
            is_strong_pick = edge >= 0.8  # Was 1.0 - more aggressive
            is_premium_pick = confidence >= 75 and edge >= 0.8  # Was 80/1.0
        elif confidence >= 50 and edge >= 0.2:  # Was 60/0.3 - more aggressive
            confidence_level = "MEDIUM"
            is_high_confidence = False
            is_strong_pick = False
            is_premium_pick = False
        else:
            confidence_level = "LOW"
            is_high_confidence = False
            is_strong_pick = False
            is_premium_pick = False
        
        return {
            "confidence_level": confidence_level,
            "is_high_confidence": is_high_confidence,
            "is_strong_pick": is_strong_pick,
            "is_premium_pick": is_premium_pick
        }
    except Exception as e:
        return {
            "confidence_level": "LOW",
            "is_high_confidence": False,
            "is_strong_pick": False,
            "is_premium_pick": False
        }
