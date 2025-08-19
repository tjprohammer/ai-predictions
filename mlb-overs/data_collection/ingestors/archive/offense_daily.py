## 11) Create `ingestors/offense_daily.py` (exact daily team offense)
from __future__ import annotations
import argparse, pandas as pd, numpy as np
from pybaseball import statcast, cache
from sqlalchemy import create_engine, text
import os
import datetime as dt
from pybaseball import statcast_batter
import datetime as dt
import statsapi



DB_URL = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
eng = create_engine(DB_URL)
cache.enable()

EVT_HIT = {'single','double','triple','home_run'}
EVT_BB = {'walk','intent_walk'}
EVT_K = {'strikeout','strikeout_double_play'}

import datetime as dt
import statsapi

def _team_abbr_from_id(team_id: int) -> str:
    try:
        recs = statsapi.lookup_team(team_id) or []
        if not recs: return ""
        return (recs[0].get("abbreviation") or "").upper()
    except Exception:
        return ""

def build_team_runs(start: str, end: str) -> pd.DataFrame:
    """Return rows (team, date, runs_pg) for each team per day."""
    s = dt.datetime.strptime(start, "%Y-%m-%d").date()
    e = dt.datetime.strptime(end, "%Y-%m-%d").date()
    rows = []
    cur = s
    while cur <= e:
        sched = statsapi.schedule(start_date=cur.strftime("%m/%d/%Y"),
                                  end_date=cur.strftime("%m/%d/%Y"))
        for g in sched:
            # skip postponed/unknown scores
            if g.get("status") not in ("Final","Game Over","Completed Early"):
                continue
            d = dt.datetime.strptime(g["game_date"], "%Y-%m-%d").date()
            hid, aid = g.get("home_id"), g.get("away_id")
            hr, ar   = g.get("home_score"), g.get("away_score")
            # translate to abbreviations; your offense table uses abbr like DET/KC
            habbr = _team_abbr_from_id(hid) if hid else ""
            aabbr = _team_abbr_from_id(aid) if aid else ""
            if habbr and hr is not None:
                rows.append({"team": habbr, "date": d, "runs_pg": float(hr)})
            if aabbr and ar is not None:
                rows.append({"team": aabbr, "date": d, "runs_pg": float(ar)})
        cur += dt.timedelta(days=1)
    return pd.DataFrame(rows)

def upsert_team_runs(df: pd.DataFrame) -> int:
    if df.empty: return 0
    with eng.begin() as cx:
        df.to_sql("tmp_team_runs", cx, index=False, if_exists="replace")
        cx.execute(text("""
            INSERT INTO teams_offense_daily (team, date, runs_pg)
            SELECT team, date, runs_pg FROM tmp_team_runs
            ON CONFLICT (team, date) DO UPDATE
            SET runs_pg = EXCLUDED.runs_pg
        """))
        cx.execute(text("DROP TABLE tmp_team_runs"))
    return len(df)


def infer_batting_team(df: pd.DataFrame) -> pd.Series:
    top = df['inning_topbot'].astype(str).str.upper().eq('TOP')
    return np.where(top, df['away_team'], df['home_team'])


def fetch_statcast_range(start: str, end: str) -> pd.DataFrame:
    sc = statcast(start_dt=start, end_dt=end)
    if sc is None or sc.empty:
        return pd.DataFrame()
    sc['game_date'] = pd.to_datetime(sc['game_date']).dt.date
    sc['batter_team'] = infer_batting_team(sc)
    sc['event'] = sc['events'].fillna('').str.lower()
    sc['is_bb'] = sc['event'].isin(EVT_BB).astype(int)
    sc['is_k'] = sc['event'].isin(EVT_K).astype(int)
    sc['is_1b'] = sc['event'].eq('single').astype(int)
    sc['is_2b'] = sc['event'].eq('double').astype(int)
    sc['is_3b'] = sc['event'].eq('triple').astype(int)
    sc['is_hr'] = sc['event'].eq('home_run').astype(int)
    sc['woba_value'] = pd.to_numeric(sc.get('woba_value', 0.0), errors='coerce').fillna(0.0)
    sc['woba_denom'] = pd.to_numeric(sc.get('woba_denom', 0.0), errors='coerce').fillna(0.0)
    return sc


def build_daily_offense(start: str, end: str) -> pd.DataFrame:
    sc = fetch_statcast_range(start, end)
    if sc.empty:
        return pd.DataFrame()

    g = sc.groupby(['batter_team','game_date'], as_index=False).agg(
        woba_value=('woba_value','sum'),
        woba_denom=('woba_denom','sum'),
        bb=('is_bb','sum'),
        k=('is_k','sum'),
        _1b=('is_1b','sum'),
        _2b=('is_2b','sum'),
        _3b=('is_3b','sum'),
        hr=('is_hr','sum'),
    )

    # core rates (guard infinities)
    g['xwoba']    = (g['woba_value'] / g['woba_denom']).replace([np.inf, -np.inf], np.nan)
    g['pa']       = g['woba_denom']
    g['ab_est']   = g['pa'] - g['bb'] - g['k']                         # plate-appearance based AB approximation
    g['hits_est'] = g['_1b'] + g['_2b'] + g['_3b'] + g['hr']
    g['ba']       = (g['hits_est'] / g['ab_est']).replace([np.inf, -np.inf], np.nan)  # NEW
    g['tb_extra'] = g['_2b'] + 2*g['_3b'] + 3*g['hr']
    g['iso']      = (g['tb_extra'] / g['ab_est']).replace([np.inf, -np.inf], np.nan)
    g['bb_pct']   = (g['bb'] / g['pa']).replace([np.inf, -np.inf], np.nan)
    g['k_pct']    = (g['k'] / g['pa']).replace([np.inf, -np.inf], np.nan)

    out = g.rename(columns={'batter_team':'team','game_date':'date'})[
        ['team','date','xwoba','iso','bb_pct','k_pct','ba']             # include ba
    ]

    # placeholders to match schema
    for c in ['wrcplus','woba','babip','vs_rhp_xwoba','vs_lhp_xwoba','home_xwoba','away_xwoba','runs_pg']:
        if c not in out.columns:
            out[c] = pd.NA

    # tidy types
    num_cols = ['wrcplus','woba','xwoba','iso','bb_pct','k_pct','babip','vs_rhp_xwoba','vs_lhp_xwoba','home_xwoba','away_xwoba','ba','runs_pg']
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out['date'] = pd.to_datetime(out['date']).dt.date
    out['team'] = out['team'].astype(str)

    return out[['team','date','wrcplus','woba','xwoba','iso','bb_pct','k_pct',
                'babip','vs_rhp_xwoba','vs_lhp_xwoba','home_xwoba','away_xwoba','ba','runs_pg']]




def upsert_offense(df: pd.DataFrame):
    if df.empty:
        return 0

    # discover existing columns in teams_offense_daily
    existing = pd.read_sql(
        text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'teams_offense_daily'
        """), eng
    )["column_name"].tolist()

    # keep only intersection (and in df order)
    cols = [c for c in df.columns if c in existing]
    if not cols:
        print("[offense] no matching columns in teams_offense_daily")
        return 0

    # build dynamic SET clause for ON CONFLICT
    set_clause = ", ".join([f'{c} = EXCLUDED.{c}' for c in cols if c not in ("team","date")])

    with eng.begin() as cx:
        df[cols].to_sql('tmp_offense', cx, index=False, if_exists='replace')
        cx.execute(text(f"""
            INSERT INTO teams_offense_daily ({",".join(cols)})
            SELECT {",".join(cols)} FROM tmp_offense
            ON CONFLICT (team, date) DO UPDATE SET
            {set_clause}
        """))
        cx.execute(text("DROP TABLE tmp_offense"))
    return len(df)


def compute_team_splits(engine, date: dt.date, lookback_days: int = 30):
    start = date - dt.timedelta(days=lookback_days)
    # get all events in [start, date-1]
    df = statcast(
        start_dt=start.strftime('%Y-%m-%d'),
        end_dt=(date - dt.timedelta(days=1)).strftime('%Y-%m-%d')
    )
    if df is None or df.empty:
        print("[offense_splits] statcast returned empty in window")
        return

    # normalize columns
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date
    df['batting_team'] = infer_batting_team(df)

    # pitcher hand column varies by version; normalize to 'throws'
    if 'p_throws' in df.columns:
        df['throws'] = df['p_throws']
    elif 'pitcher_throws' in df.columns:
        df['throws'] = df['pitcher_throws']
    else:
        print("[offense_splits] no pitcher hand column; skipping")
        return

    # numeric pieces for xwOBA
    df['woba_value'] = pd.to_numeric(df.get('woba_value'), errors='coerce').fillna(0.0)
    df['woba_denom'] = pd.to_numeric(df.get('woba_denom'), errors='coerce').fillna(0.0)

    # aggregate by team x pitcher hand
    grp = df.groupby(['batting_team','throws'], as_index=False).agg(
        wv=('woba_value','sum'),
        wd=('woba_denom','sum'),
        xwoba_bb=('estimated_woba_using_speedangle','mean')  # fallback if wd==0
    )
    # compute xwOBA robustly
    grp['xwoba'] = np.where(grp['wd'] > 0, grp['wv'] / grp['wd'], grp['xwoba_bb'])

    # pivot to wide
    wide = grp.pivot(index='batting_team', columns='throws', values='xwoba').reset_index()
    wide.columns.name = None
    if 'R' not in wide.columns and 'L' not in wide.columns:
        print("[offense_splits] no R/L splits present; skipping")
        return

    wide.rename(columns={'batting_team':'team','R':'vs_rhp_xwoba','L':'vs_lhp_xwoba'}, inplace=True)
    wide['date'] = date

    # ensure both split cols exist
    for c in ('vs_rhp_xwoba','vs_lhp_xwoba'):
        if c not in wide.columns:
            wide[c] = None

    cols = ['date','team','vs_rhp_xwoba','vs_lhp_xwoba']
    with engine.begin() as con:
        for r in wide[cols].itertuples(index=False):
            con.execute(text("""
                INSERT INTO teams_offense_daily (date, team, vs_rhp_xwoba, vs_lhp_xwoba)
                VALUES (:date, :team, :vsr, :vsl)
                ON CONFLICT (team, date) DO UPDATE SET
                    vs_rhp_xwoba = COALESCE(EXCLUDED.vs_rhp_xwoba, teams_offense_daily.vs_rhp_xwoba),
                    vs_lhp_xwoba = COALESCE(EXCLUDED.vs_lhp_xwoba, teams_offense_daily.vs_lhp_xwoba)
            """), {"date": r.date, "team": r.team, "vsr": r.vs_rhp_xwoba, "vsl": r.vs_lhp_xwoba})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    df = build_daily_offense(args.start, args.end)
    n = upsert_offense(df)
    print('upsert teams_offense_daily:', n)

    # Populate real runs per game for the same window
    runs_df = build_team_runs(args.start, args.end)
    m = upsert_team_runs(runs_df)
    print("upsert team runs:", m)


    # ← add these 2 lines
    end_date = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    compute_team_splits(eng, end_date, lookback_days=30)
