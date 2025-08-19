from __future__ import annotations
"""
Pitcher Statcast enrichment
- For pitcher starts already stored in `pitchers_starts`, compute Statcast-based per-start metrics:
  * csw_pct (Called + Swinging Strikes / Total Pitches)
  * xwoba_allowed (mean estimated_woba_using_speedangle over batted balls)
  * xslg_allowed (mean estimated_slg_using_speedangle over batted balls)
  * avg_ev_allowed (mean launch_speed over batted balls)
  * velo_fb (mean release_speed for FF; fallback overall)
  * velo_delta_3g (current velo_fb minus mean of previous 3 starts)

Usage:
  python -m ingestors.pitchers_statcast_enrich --start 2025-04-01 --end 2025-04-07

Requirements:
  pip install pybaseball pandas sqlalchemy
  DATABASE_URL set
"""
import argparse
import datetime as dt
import pandas as pd
import numpy as np
from sqlalchemy import text
from pybaseball import statcast_pitcher, cache

from .util import get_engine

cache.enable()

CSW_DESCRIPTIONS = {
    'called_strike', 'swinging_strike', 'swinging_strike_blocked',
    'foul_tip',  # often not included; include only called+swinging for strict CSW; here we keep classic CSW
}


def per_start_statcast(pid: int, game_date: dt.date) -> pd.DataFrame:
    # Fetch that day's Statcast for the pitcher
    sd = ed = game_date.strftime('%Y-%m-%d')
    try:
        sc = statcast_pitcher(start_dt=sd, end_dt=ed, player_id=pid)
    except Exception:
        return pd.DataFrame()
    if sc is None or sc.empty:
        return pd.DataFrame()
    sc['description'] = sc['description'].astype(str).str.lower()
    sc['is_csw'] = sc['description'].isin({'called_strike','swinging_strike','swinging_strike_blocked'}).astype(int)
    sc['is_bip'] = sc['type'].astype(str).str.upper().eq('X').astype(int)
    # Pitches
    total_pitches = len(sc)
    csw = sc['is_csw'].sum()
    csw_pct = csw / total_pitches if total_pitches > 0 else np.nan
    # Batted balls only
    bip = sc.loc[sc['is_bip'] == 1]
    xwoba_allowed = bip['estimated_woba_using_speedangle'].astype(float).mean() if not bip.empty else np.nan
    xslg_allowed = bip['estimated_slg_using_speedangle'].astype(float).mean() if not bip.empty else np.nan
    avg_ev_allowed = bip['launch_speed'].astype(float).mean() if not bip.empty else np.nan
    # Velocity (FF only, else all)
    ff = sc.loc[sc['pitch_type'] == 'FF']
    velo_fb = ff['release_speed'].astype(float).mean() if not ff.empty else sc['release_speed'].astype(float).mean()

    return pd.DataFrame([{
        'pitcher_id': pid,
        'date': game_date,
        'csw_pct': csw_pct,
        'xwoba_allowed': xwoba_allowed,
        'xslg_allowed': xslg_allowed,
        'avg_ev_allowed': avg_ev_allowed,
        'velo_fb': velo_fb,
    }])


def enrich_range(start: str, end: str) -> pd.DataFrame:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    eng = get_engine()
    base = pd.read_sql(text("""
        SELECT pitcher_id, date, game_id
        FROM pitchers_starts
        WHERE date BETWEEN :s AND :e
        ORDER BY pitcher_id, date
    """), eng, params={'s': s, 'e': e})
    if base.empty:
        return pd.DataFrame()
    rows = []
    for (pid, gdate), grp in base.groupby(['pitcher_id','date']):
        rows.append(per_start_statcast(int(pid), pd.to_datetime(gdate).date()))
    out = pd.concat([r for r in rows if r is not None and not r.empty], ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    # compute velo_delta_3g using prior enriched rows (pull prior 3 starts from table if present)
    prev = pd.read_sql(text("""
        SELECT pitcher_id, date, velo_fb
        FROM pitchers_starts
        WHERE pitcher_id = ANY(:pids)
    """), eng, params={'pids': list(out['pitcher_id'].unique())})
    prev['date'] = pd.to_datetime(prev['date']).dt.date
    out['velo_delta_3g'] = np.nan
    for pid, cur in out.groupby('pitcher_id'):
        prior = prev.loc[prev['pitcher_id']==pid, ['date','velo_fb']].sort_values('date').copy()
        cur = cur.sort_values('date').copy()
        combined = pd.concat([prior, cur[['date','velo_fb']]], ignore_index=True).sort_values('date')
        # compute previous 3-start mean for each date in combined
        combined['prev3_mean'] = combined['velo_fb'].rolling(window=3, min_periods=3).mean().shift(1)
        # assign deltas for current rows
        for idx, row in cur.iterrows():
            prev3 = combined.loc[combined['date'] < row['date'], 'velo_fb'].tail(3).mean()
            out.loc[idx, 'velo_delta_3g'] = row['velo_fb'] - prev3 if pd.notna(prev3) else np.nan
    return out


def upsert_enrichment(df: pd.DataFrame):
    """Update pitchers_starts rows by (pitcher_id, date).
    If your PK is (pitcher_id, game_id), we update via a correlated subquery on date.
    """
    if df is None or df.empty:
        return 0
    eng = get_engine()
    with eng.begin() as cx:
        df.to_sql('tmp_ps_enrich', cx, index=False, if_exists='replace')
        cx.execute(text("""
        UPDATE pitchers_starts p
        SET csw_pct = t.csw_pct,
            xwoba_allowed = t.xwoba_allowed,
            xslg_allowed = t.xslg_allowed,
            avg_ev_allowed = t.avg_ev_allowed,
            velo_fb = t.velo_fb,
            velo_delta_3g = t.velo_delta_3g
        FROM tmp_ps_enrich t
        WHERE p.pitcher_id = t.pitcher_id AND p.date = t.date
        """))
        cx.execute(text("DROP TABLE tmp_ps_enrich"))
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    df = enrich_range(args.start, args.end)
    if df.empty:
        print('no statcast enrichment rows built')
        return
    n = upsert_enrichment(df)
    print(f'updated pitchers_starts rows: {n}')


if __name__ == '__main__':
    main()
