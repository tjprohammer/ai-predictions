#!/usr/bin/env python3
"""
Comprehensive Data Validation Script
Compares database source data with API response to ensure accuracy.
"""

import os, sys, json, requests
from datetime import date
from sqlalchemy import create_engine, text

# ---------- Config ----------
API_URL = os.environ.get("API_URL", "http://localhost:8000/comprehensive-games")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb",
)

# ---------- Helpers ----------
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def approx_equal(a, b, tol=1e-3):
    """Safe float-ish comparison with None handling."""
    try:
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

def as_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def as_int(x):
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

# ---------- Validators ----------
def validate_game_data(target_date=None):
    """Comprehensive validation of all game data."""
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    print("🔍 COMPREHENSIVE DATA VALIDATION")
    print(f"📅 Target Date: {target_date}")
    print("=" * 50)

    # Fetch API data
    try:
        resp = requests.get(API_URL, timeout=10)
        if resp.status_code != 200:
            print(f"❌ API Error: {resp.status_code}")
            return
        api_games = resp.json()
        if isinstance(api_games, dict) and "games" in api_games:
            api_games = api_games["games"]
        elif isinstance(api_games, dict):
            api_games = [api_games]
        if not isinstance(api_games, list):
            print("❌ Unexpected API payload shape.")
            return
        print(f"✅ API Response: {len(api_games)} games")
    except Exception as e:
        print(f"❌ API Connection Error: {e}")
        return

    engine = get_engine()

    print(f"\n📊 DETAILED VALIDATION FOR ALL {len(api_games)} GAMES:")
    print("=" * 50)

    summary = {
        "total_games": len(api_games),
        "prediction_errors": 0,
        "betting_errors": 0,
        "weather_errors": 0,
        "pitcher_errors": 0,
        "team_stats_errors": 0,
        "venue_errors": 0,
    }

    for i, api_game in enumerate(api_games):
        game_id = api_game.get("game_id")
        print(f"\n🎯 GAME {i+1}/{len(api_games)}: {api_game.get('away_team')} @ {api_game.get('home_team')} (ID: {game_id})")
        print("-" * 60)
        errs = []

        with engine.begin() as conn:
            # Join to latest predictions view by BOTH game_id and date
            db = conn.execute(text("""
                SELECT 
                    eg.*,
                    lpp.predicted_total AS pred_total,
                    lpp.p_over, lpp.p_under,
                    lpp.recommendation, lpp.adj_edge,
                    lpp.priced_total,
                    lpp.ev_over, lpp.ev_under,
                    lpp.kelly_over, lpp.kelly_under,
                    lpp.over_odds, lpp.under_odds
                FROM enhanced_games eg
                LEFT JOIN latest_probability_predictions lpp
                  ON eg.game_id = lpp.game_id
                 AND eg.date = lpp.game_date
                WHERE eg.game_id = :gid AND eg.date = :dt
            """), {"gid": game_id, "dt": target_date}).fetchone()

        if not db:
            print("❌ No database data found")
            continue

        # ---- 1) Prediction ----
        print("🎯 PREDICTION DATA:")
        api_pred = as_float(api_game.get("predicted_total"))
        db_pred  = as_float(db.pred_total)
        pred_ok  = approx_equal(api_pred, db_pred, tol=1e-3)
        print(f"  Predicted Total: API={api_pred}, DB={db_pred} {'✅' if pred_ok else '❌'}")
        if not pred_ok:
            errs.append("Predicted Total mismatch")
            summary["prediction_errors"] += 1

        api_rec = api_game.get("recommendation")
        db_rec  = db.recommendation
        rec_ok  = (api_rec == db_rec)
        print(f"  Recommendation: API={api_rec}, DB={db_rec} {'✅' if rec_ok else '❌'}")
        if not rec_ok:
            errs.append("Recommendation mismatch")
            summary["prediction_errors"] += 1

        # Confidence derived from p_over/p_under
        api_conf = as_float(api_game.get("confidence"))
        if db_rec == "OVER":
            exp_conf = as_float(db.p_over) * 100 if db.p_over is not None else None
        elif db_rec == "UNDER":
            exp_conf = as_float(db.p_under) * 100 if db.p_under is not None else None
        else:
            mx = max(as_float(db.p_over) or 0, as_float(db.p_under) or 0)
            exp_conf = mx * 100 if mx else None
        # Round to 1 decimal to match API formatting
        exp_conf = round(exp_conf, 1) if exp_conf is not None else None
        conf_ok  = approx_equal(api_conf, exp_conf, tol=0.05)
        print(f"  Confidence: API={api_conf}%, Expected={exp_conf}% {'✅' if conf_ok else '❌'}")
        if not conf_ok:
            errs.append("Confidence mismatch")
            summary["prediction_errors"] += 1

        api_edge = as_float(api_game.get("edge"))
        db_edge  = as_float(db.adj_edge)
        edge_ok  = approx_equal(api_edge, db_edge, tol=1e-3)
        print(f"  Edge: API={api_edge}, DB={db_edge} {'✅' if edge_ok else '❌'}")
        if not edge_ok:
            errs.append("Edge mismatch")
            summary["prediction_errors"] += 1

        # ---- 2) Betting ----
        print("\n💰 BETTING DATA:")
        api_bet = api_game.get("betting_info", {}) or {}
        market_total_ok = approx_equal(api_bet.get("market_total"), db.priced_total, tol=1e-3)
        over_odds_ok    = as_int(api_game.get("over_odds"))  == as_int(db.over_odds)
        under_odds_ok   = as_int(api_game.get("under_odds")) == as_int(db.under_odds)
        print(f"  Market Total: API={api_bet.get('market_total')}, DB={db.priced_total} {'✅' if market_total_ok else '❌'}")
        print(f"  Over Odds:   API={api_game.get('over_odds')}, DB={db.over_odds} {'✅' if over_odds_ok else '❌'}")
        print(f"  Under Odds:  API={api_game.get('under_odds')}, DB={db.under_odds} {'✅' if under_odds_ok else '❌'}")
        if not market_total_ok:
            errs.append("Market total mismatch")
            summary["betting_errors"] += 1
        if not over_odds_ok:
            errs.append("Over odds mismatch")
            summary["betting_errors"] += 1
        if not under_odds_ok:
            errs.append("Under odds mismatch")
            summary["betting_errors"] += 1

        api_ev_over  = as_float(api_bet.get("expected_value_over"))
        api_ev_under = as_float(api_bet.get("expected_value_under"))
        ev_over_ok   = approx_equal(api_ev_over,  db.ev_over,  tol=1e-3)
        ev_under_ok  = approx_equal(api_ev_under, db.ev_under, tol=1e-3)
        print(f"  EV Over:  API={api_ev_over},  DB={db.ev_over}  {'✅' if ev_over_ok else '❌'}")
        print(f"  EV Under: API={api_ev_under}, DB={db.ev_under} {'✅' if ev_under_ok else '❌'}")
        if not ev_over_ok:
            errs.append("EV Over mismatch")
            summary["betting_errors"] += 1
        if not ev_under_ok:
            errs.append("EV Under mismatch")
            summary["betting_errors"] += 1

        # ---- 3) Weather ----
        print("\n🌤️ WEATHER DATA:")
        api_wx = api_game.get("weather", {}) or {}
        t_ok  = approx_equal(api_wx.get("temperature"), db.temperature, tol=0.5)
        ws_ok = approx_equal(api_wx.get("wind_speed"),  db.wind_speed,  tol=0.5)
        # Strings may differ in case; normalize
        wd_ok = (str(api_wx.get("wind_direction") or "").upper()
                 == str(db.wind_direction or "").upper())
        cond_ok = (str(api_wx.get("condition") or "").lower()
                   == str(db.weather_condition or "").lower())
        print(f"  Temperature:   API={api_wx.get('temperature')}, DB={db.temperature} {'✅' if t_ok else '❌'}")
        print(f"  Wind Speed:    API={api_wx.get('wind_speed')},  DB={db.wind_speed}  {'✅' if ws_ok else '❌'}")
        print(f"  Wind Direction:API={api_wx.get('wind_direction')}, DB={db.wind_direction} {'✅' if wd_ok else '❌'}")
        print(f"  Condition:     API={api_wx.get('condition')}, DB={db.weather_condition} {'✅' if cond_ok else '❌'}")
        if not t_ok:     errs.append("Temperature mismatch");       summary["weather_errors"] += 1
        if not ws_ok:    errs.append("Wind speed mismatch");        summary["weather_errors"] += 1
        if not wd_ok:    errs.append("Wind direction mismatch");    summary["weather_errors"] += 1
        if not cond_ok:  errs.append("Weather condition mismatch"); summary["weather_errors"] += 1

        # ---- 4) Pitchers ----
        print("\n⚾ PITCHER DATA:")
        ap_p = api_game.get("pitchers", {}) or {}
        h_p  = ap_p.get("home", {}) or {}
        a_p  = ap_p.get("away", {}) or {}
        home_name_ok = (str(h_p.get("name") or "") == str(db.home_sp_name or ""))
        away_name_ok = (str(a_p.get("name") or "") == str(db.away_sp_name or ""))
        home_era_ok  = approx_equal(h_p.get("era"), db.home_sp_season_era, tol=0.01)
        away_era_ok  = approx_equal(a_p.get("era"), db.away_sp_season_era, tol=0.01)
        print(f"  Home Pitcher: API={h_p.get('name')}, DB={db.home_sp_name} {'✅' if home_name_ok else '❌'}")
        print(f"  Away Pitcher: API={a_p.get('name')}, DB={db.away_sp_name} {'✅' if away_name_ok else '❌'}")
        print(f"  Home ERA:     API={h_p.get('era')},  DB={db.home_sp_season_era} {'✅' if home_era_ok else '❌'}")
        print(f"  Away ERA:     API={a_p.get('era')},  DB={db.away_sp_season_era} {'✅' if away_era_ok else '❌'}")
        if not home_name_ok: errs.append("Home pitcher name mismatch"); summary["pitcher_errors"] += 1
        if not away_name_ok: errs.append("Away pitcher name mismatch"); summary["pitcher_errors"] += 1
        if not home_era_ok:  errs.append("Home pitcher ERA mismatch");  summary["pitcher_errors"] += 1
        if not away_era_ok:  errs.append("Away pitcher ERA mismatch");  summary["pitcher_errors"] += 1

        # ---- 5) Team stats ----
        print("\n📈 TEAM STATS:")
        ap_ts = api_game.get("team_stats", {}) or {}
        h_s   = ap_ts.get("home", {}) or {}
        a_s   = ap_ts.get("away", {}) or {}
        home_avg_ok = approx_equal(h_s.get("batting_avg"), db.home_team_avg, tol=1e-3)
        away_avg_ok = approx_equal(a_s.get("batting_avg"), db.away_team_avg, tol=1e-3)
        print(f"  Home BA: API={h_s.get('batting_avg')}, DB={db.home_team_avg} {'✅' if home_avg_ok else '❌'}")
        print(f"  Away BA: API={a_s.get('batting_avg')}, DB={db.away_team_avg} {'✅' if away_avg_ok else '❌'}")
        if not home_avg_ok: errs.append("Home batting average mismatch"); summary["team_stats_errors"] += 1
        if not away_avg_ok: errs.append("Away batting average mismatch"); summary["team_stats_errors"] += 1

        # ---- 6) Venue ----
        print("\n🏟️ VENUE DATA:")
        ap_v  = api_game.get("venue_details", {}) or {}
        venue_name_ok = (str(api_game.get("venue") or "") == str(db.venue_name or ""))
        venue_id_ok   = (as_int(ap_v.get("id")) == as_int(db.venue_id))
        print(f"  Venue Name: API={api_game.get('venue')}, DB={db.venue_name} {'✅' if venue_name_ok else '❌'}")
        print(f"  Venue ID:   API={ap_v.get('id')}, DB={db.venue_id} {'✅' if venue_id_ok else '❌'}")
        if not venue_name_ok: errs.append("Venue name mismatch"); summary["venue_errors"] += 1
        if not venue_id_ok:   errs.append("Venue ID mismatch");   summary["venue_errors"] += 1

        # Game summary
        if errs:
            print(f"\n❌ GAME {i+1} ERRORS: {', '.join(errs)}")
        else:
            print(f"\n✅ GAME {i+1}: ALL DATA VALIDATED SUCCESSFULLY")

    # ---- Overall summary ----
    print(f"\n\n📋 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Games Validated: {summary['total_games']}")
    print(f"Prediction Errors: {summary['prediction_errors']}")
    print(f"Betting Data Errors: {summary['betting_errors']}")
    print(f"Weather Data Errors: {summary['weather_errors']}")
    print(f"Pitcher Data Errors: {summary['pitcher_errors']}")
    print(f"Team Stats Errors: {summary['team_stats_errors']}")
    print(f"Venue Data Errors: {summary['venue_errors']}")

    total_errs = sum([
        summary["prediction_errors"],
        summary["betting_errors"],
        summary["weather_errors"],
        summary["pitcher_errors"],
        summary["team_stats_errors"],
        summary["venue_errors"],
    ])
    if total_errs == 0:
        print(f"\n🎉 PERFECT! All {summary['total_games']} games have accurate data!")
    else:
        print(f"\n⚠️ FOUND {total_errs} TOTAL ERRORS across {summary['total_games']} games")
        # 6 categories per game
        denom = max(summary["total_games"] * 6, 1)
        print(f"📊 Error Rate: {100.0 * total_errs / denom:.1f}%")

def check_data_completeness(target_date):
    """Check for missing or null data for the given date."""
    print("\n\n🔍 DATA COMPLETENESS CHECK")
    print("=" * 50)
    engine = get_engine()
    with engine.begin() as conn:
        missing_weather = conn.execute(text("""
            SELECT COUNT(*) AS c
            FROM enhanced_games
            WHERE date = :d
              AND (temperature IS NULL OR wind_speed IS NULL OR weather_condition IS NULL)
        """), {"d": target_date}).fetchone()
        print(f"🌤️ Games missing weather data: {missing_weather.c}")

        missing_pitchers = conn.execute(text("""
            SELECT COUNT(*) AS c
            FROM enhanced_games
            WHERE date = :d
              AND (home_sp_name IS NULL OR away_sp_name IS NULL)
        """), {"d": target_date}).fetchone()
        print(f"⚾ Games missing pitcher data: {missing_pitchers.c}")

        # include date in the join to avoid cross-day mismatches
        missing_predictions = conn.execute(text("""
            SELECT COUNT(*) AS c
            FROM enhanced_games eg
            LEFT JOIN latest_probability_predictions lpp
              ON eg.game_id = lpp.game_id
             AND eg.date = lpp.game_date
            WHERE eg.date = :d AND lpp.game_id IS NULL
        """), {"d": target_date}).fetchone()
        print(f"🎯 Games missing predictions: {missing_predictions.c}")

        missing_market = conn.execute(text("""
            SELECT COUNT(*) AS c
            FROM enhanced_games
            WHERE date = :d
              AND (market_total IS NULL OR over_odds IS NULL OR under_odds IS NULL)
        """), {"d": target_date}).fetchone()
        print(f"💰 Games missing market data: {missing_market.c}")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2025-08-18"
    validate_game_data(target_date)
    check_data_completeness(target_date)
    print("\n" + "=" * 50)
    print("✅ VALIDATION COMPLETE")
