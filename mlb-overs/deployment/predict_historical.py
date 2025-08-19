#!/usr/bin/env python3
import os, argparse, logging
from datetime import datetime, timedelta, date as _date
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("predict_historical")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def get_engine(): return create_engine(DATABASE_URL, pool_pre_ping=True)

def seed_enhanced_from_lgf(conn, ds):
    conn.execute(text("""
        INSERT INTO enhanced_games (game_id, "date", home_team, away_team)
        SELECT game_id, "date", home_team, away_team
        FROM legitimate_game_features
        WHERE "date" = :d
        ON CONFLICT (game_id) DO NOTHING
    """), {"d": ds})

def load_lgf(engine, ds):
    q = text("""SELECT * FROM legitimate_game_features WHERE "date" = :d ORDER BY game_id""")
    return pd.read_sql(q, engine, params={"d": ds})

def fetch_markets(engine, ds):
    q = text("""
        SELECT eg.game_id, eg."date", eg.market_total
        FROM enhanced_games eg
        JOIN legitimate_game_features lgf ON lgf.game_id = eg.game_id AND lgf."date" = eg."date"
        WHERE eg."date" = :d
    """)
    mk = pd.read_sql(q, engine, params={"d": ds})
    return mk.dropna(subset=["market_total"]).drop_duplicates("game_id")

def engineer_and_align(df):
    from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    p = EnhancedBullpenPredictor()
    if "market_total_final" in df.columns:
        df = df.copy(); df["market_total"] = df.pop("market_total_final")
    feats = p.engineer_features(df)
    try:
        X = p.align_serving_features(feats, strict=False)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=p.feature_columns)
    except Exception:
        X = feats.reindex(columns=p.feature_columns).fillna(0)
    return feats, X, p

def predict_and_update(engine, ids_df, X, predictor, ds, update_signals=True, thresh=1.0):
    M = X
    if getattr(predictor, "preproc", None) is not None:
        M = predictor.preproc.transform(M)
    elif getattr(predictor, "scaler", None) is not None:
        sc = predictor.scaler
        if not hasattr(sc, "n_features_in_") or sc.n_features_in_ == M.shape[1]:
            M = sc.transform(M)
    y = predictor.model.predict(M)
    out = ids_df[["game_id","date"]].copy()
    out["predicted_total"] = y.astype(float)

    with engine.begin() as conn:
        sql = text("""UPDATE enhanced_games SET predicted_total=:predicted_total WHERE game_id=:game_id AND "date"=:date""")
        n=0
        for r in out.to_dict(orient="records"):
            n += conn.execute(sql, r).rowcount
        log.info("Updated predicted_total for %d games on %s", n, ds)

        if update_signals:
            conn.execute(text(f"""
                UPDATE enhanced_games
                   SET edge = CASE
                                 WHEN market_total IS NULL OR predicted_total IS NULL THEN NULL
                                 ELSE predicted_total - market_total
                              END,
                       recommendation = CASE
                                 WHEN market_total IS NULL OR predicted_total IS NULL THEN NULL
                                 WHEN (predicted_total - market_total) >= {thresh}  THEN 'OVER'
                                 WHEN (predicted_total - market_total) <= {-thresh} THEN 'UNDER'
                                 ELSE 'NO BET'
                              END
                 WHERE "date" = :d
            """), {"d": ds})

def daterange(s,e):
    d=s
    while d<=e:
        yield d
        d += timedelta(days=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (single day)")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--threshold", type=float, default=1.0, help="Edge threshold for recommendation")
    ap.add_argument("--no-signals", action="store_true", help="Do not update edge/recommendation")
    args = ap.parse_args()

    if args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end   = datetime.strptime(args.end,   "%Y-%m-%d").date()
        dates = list(daterange(start,end))

    engine = get_engine()
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        log.info("=== Predicting (historical) for %s ===", ds)
        with engine.begin() as conn:
            seed_enhanced_from_lgf(conn, ds)
        df = load_lgf(engine, ds)
        if df.empty:
            log.info("No LGF rows for %s; skipping.", ds); continue
        mk = fetch_markets(engine, ds)
        if not mk.empty:
            df = df.merge(mk[["game_id","market_total"]], on="game_id", how="left")
        ids = df[["game_id","date"]].copy()
        feats, X, predictor = engineer_and_align(df)
        predict_and_update(engine, ids, X, predictor, ds, update_signals=not args.no_signals, thresh=args.threshold)
    engine.dispose()
    log.info("Done.")

if __name__ == "__main__":
    main()
