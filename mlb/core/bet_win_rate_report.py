#!/usr/bin/env python3
"""
Bet Win-Rate Report
- Grades served totals predictions (published: predicted_total -> fallback predicted_total_whitelist)
  against market_total for completed games.
- Computes win/loss/push and summary win rate, with optional edge threshold to only count
  predictions that deviated from the market by a minimum amount (e.g., 0.5 runs).

Usage:
  python mlb/core/bet_win_rate_report.py --days 10 --min-edge 0.5 --outdir outputs

Outputs:
  - outputs/ledger/bet_results_<ts>.csv (per-game grading with side/edge and W/L/P)
  - Prints summary: n_bets, win%, loss%, push%, and breakdown by source (primary vs whitelist)
"""
import os
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def compute_bet_outcomes(
    days: int,
    min_edge: float,
    outdir: str,
    consensus: bool = False,
    hook_only: bool = False,
    same_day_only: bool = False,
    ev_filter: float = 0.0,
    assumed_rmse: float = 4.8,
    p_threshold: float = 0.0,
):
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    q = text('''
        SELECT 
            game_id,
            "date",
            home_team,
            away_team,
            market_total,
            over_odds,
            under_odds,
            predicted_total,
            predicted_total_whitelist,
            predicted_total_learning,
            prediction_timestamp,
            total_runs
        FROM enhanced_games
        WHERE "date" BETWEEN :s AND :e
          AND (predicted_total IS NOT NULL OR predicted_total_whitelist IS NOT NULL)
        ORDER BY "date", game_id
    ''')

    df = pd.read_sql(q, engine, params={'s': start_date, 'e': end_date})
    if df.empty:
        print('No served predictions found in window')
        return None

    # Published prediction selection
    pub = pd.to_numeric(df['predicted_total'], errors='coerce')
    wl = pd.to_numeric(df['predicted_total_whitelist'], errors='coerce')
    learn = pd.to_numeric(df.get('predicted_total_learning'), errors='coerce') if 'predicted_total_learning' in df.columns else pd.Series(index=df.index, dtype=float)
    pred = pub.where(pub.notna(), wl)
    source = np.where(pub.notna(), 'primary', np.where(wl.notna(), 'whitelist', 'unknown'))

    mkt = pd.to_numeric(df['market_total'], errors='coerce')
    y = pd.to_numeric(df['total_runs'], errors='coerce')

    out = pd.DataFrame({
        'game_id': df['game_id'],
        'date': df['date'],
        'home_team': df['home_team'],
        'away_team': df['away_team'],
        'prediction_timestamp': df['prediction_timestamp'],
        'market_total': mkt,
        'predicted_total': pred,
    'source': source,
        'total_runs': y,
        'over_odds': pd.to_numeric(df.get('over_odds'), errors='coerce'),
        'under_odds': pd.to_numeric(df.get('under_odds'), errors='coerce'),
    'predicted_total_primary': pub,
    'predicted_total_whitelist': wl,
    'predicted_total_learning': learn,
    })

    # Only grade completed games
    out = out[out['total_runs'].notna()].copy()
    if out.empty:
        print('No completed games to grade in window')
        return None

    out['edge'] = (out['predicted_total'] - out['market_total']).astype(float)
    out['bet_side'] = np.where(out['edge'] > 0, 'OVER', np.where(out['edge'] < 0, 'UNDER', 'NONE'))

    # Optional hook filter: prefer .5 lines (reduce pushes)
    if hook_only:
        frac = np.mod(out['market_total'].astype(float), 1.0)
        out = out[(frac != 0.0)].copy()

    # Optional same-day freshness (only predictions stamped on the same calendar date as the game)
    if same_day_only and 'prediction_timestamp' in out.columns:
        ts_dates = pd.to_datetime(out['prediction_timestamp']).dt.date
        out = out[ts_dates == pd.to_datetime(out['date']).dt.date].copy()

    # Apply edge filter
    qualifies = out['bet_side'].ne('NONE') & (out['edge'].abs() >= float(min_edge))

    # Optional consensus: require primary and whitelist (or learning) agree on side and both meet edge
    if consensus:
        p = out['predicted_total_primary']
        w = out['predicted_total_whitelist']
        l = out['predicted_total_learning']
        side_p = np.sign((p - out['market_total']).astype(float))
        side_w = np.sign((w - out['market_total']).astype(float))
        side_l = np.sign((l - out['market_total']).astype(float))
        agree_pw = (p.notna() & w.notna() & (side_p == side_w) & (np.abs(p - out['market_total']) >= min_edge) & (np.abs(w - out['market_total']) >= min_edge))
        agree_pl = (p.notna() & l.notna() & (side_p == side_l) & (np.abs(p - out['market_total']) >= min_edge) & (np.abs(l - out['market_total']) >= min_edge))
        agrees = agree_pw | agree_pl
        qualifies = qualifies & agrees
    graded = out[qualifies].copy()
    # Reset index so downstream numpy arrays (p_win, unit_return) can be safely indexed
    # using graded.index after applying further filters (EV, p-threshold).
    graded.reset_index(drop=True, inplace=True)

    if graded.empty:
        print(f'No qualifying bets with min-edge {min_edge}')
        return None

    # Grade outcome vs market_total
    # Over wins if total_runs > market_total; Under wins if total_runs < market_total; equal -> Push
    cmp = np.sign(graded['total_runs'] - graded['market_total'])
    pred_sign = np.sign(graded['edge'])  # OVER=+1, UNDER=-1

    result = np.where(cmp == 0, 'PUSH', np.where(pred_sign == cmp, 'WIN', 'LOSS'))
    graded['result'] = result

    # Select odds for the bet side and compute unit profit
    # Default odds when missing: -110
    def american_to_unit_return(odds):
        try:
            o = float(odds)
        except Exception:
            o = -110.0
        if np.isnan(o):
            o = -110.0
        return (o/100.0) if o > 0 else (100.0/abs(o))

    chosen_odds = np.where(graded['bet_side'] == 'OVER', graded['over_odds'], np.where(graded['bet_side'] == 'UNDER', graded['under_odds'], np.nan))
    unit_return = np.vectorize(american_to_unit_return)(chosen_odds)

    # Profit per bet (stake=1 unit): WIN -> +unit_return, LOSS -> -1, PUSH -> 0
    graded['unit_profit'] = np.where(graded['result'] == 'WIN', unit_return, np.where(graded['result'] == 'LOSS', -1.0, 0.0))

    # Normal CDF via math.erf (no SciPy dependency)
    import math
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    sigma = float(assumed_rmse) if assumed_rmse and assumed_rmse > 0 else 4.8
    diff = (graded['predicted_total'].astype(float) - graded['market_total'].astype(float))
    # P(total > line) under Normal(mean=pred, sd=sigma)
    p_over = 1.0 - graded.apply(lambda r: norm_cdf((r['market_total'] - r['predicted_total']) / sigma), axis=1)
    p_win = np.where(graded['bet_side'] == 'OVER', p_over, 1.0 - p_over)

    # EV and probability filters
    if ev_filter and ev_filter > 0:
        ev = p_win * unit_return - (1 - p_win)
        graded = graded[ev >= float(ev_filter)].copy()
        p_win = p_win[graded.index]
        unit_return = unit_return[graded.index]
        if graded.empty:
            print(f'No qualifying bets after EV >= {ev_filter:.3f} filter')
            return None

    if p_threshold and p_threshold > 0:
        graded = graded[p_win >= float(p_threshold)].copy()
        p_win = p_win[graded.index]
        unit_return = unit_return[graded.index]
        if graded.empty:
            print(f'No qualifying bets after P(win) >= {p_threshold:.2f} filter')
            return None

    # Summary
    n = len(graded)
    wins = int((graded['result'] == 'WIN').sum())
    losses = int((graded['result'] == 'LOSS').sum())
    pushes = int((graded['result'] == 'PUSH').sum())

    win_rate = wins / n if n else 0.0
    loss_rate = losses / n if n else 0.0
    push_rate = pushes / n if n else 0.0
    total_profit = float(graded['unit_profit'].sum()) if n else 0.0
    roi = total_profit / n if n else 0.0

    # By source
    by_src = (
        graded.groupby('source')['result']
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=['WIN','LOSS','PUSH'], fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )

    # Write CSV
    outdir_path = Path(outdir) / 'ledger'
    outdir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_path = outdir_path / f'bet_results_{ts}.csv'
    graded.to_csv(out_path, index=False)

    # Print summary
    avg_p = float(np.mean(p_win)) if isinstance(p_win, np.ndarray) and p_win.size else float('nan')
    print(f"Window: {start_date} → {end_date} | min-edge: {min_edge} | consensus: {consensus} | hook-only: {hook_only} | same-day: {same_day_only} | EV≥{ev_filter} | P(win)≥{p_threshold}")
    print(f"Graded bets: {n} | WIN: {wins} | LOSS: {losses} | PUSH: {pushes}")
    print(f"Win%: {win_rate:.1%} | Loss%: {loss_rate:.1%} | Push%: {push_rate:.1%} | Avg P(win): {avg_p:.3f}")
    print(f"Profit (units): {total_profit:.2f} | ROI per bet: {roi:.3f}")

    if not by_src.empty:
        for src, row in by_src.iterrows():
            total = int(row.get('total', 0))
            if total > 0:
                w = int(row.get('WIN', 0)); l = int(row.get('LOSS', 0)); p = int(row.get('PUSH', 0))
                wr = w / total
                print(f"  - {src}: {w}/{total} wins ({wr:.1%}), {l} losses, {p} pushes")

    print('Wrote:', out_path)
    return {
        'n_bets': n,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'push_rate': push_rate,
        'profit_units': total_profit,
        'roi_per_bet': roi,
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compute bet win-rate for totals predictions over the last N days')
    ap.add_argument('--days', type=int, default=10)
    ap.add_argument('--min-edge', type=float, default=0.5, help='Minimum difference vs market to count as a bet (runs)')
    ap.add_argument('--outdir', default='outputs')
    ap.add_argument('--consensus', action='store_true', help='Require primary + whitelist/learning to agree and both meet edge')
    ap.add_argument('--hook-only', action='store_true', help='Only bet .5 lines to reduce pushes')
    ap.add_argument('--same-day-only', action='store_true', help='Only grade predictions stamped same calendar day as game')
    ap.add_argument('--ev-filter', type=float, default=0.0, help='Minimum EV per bet using Normal approx and odds (e.g., 0.02 = +2%)')
    ap.add_argument('--assumed-rmse', type=float, default=4.8, help='Assumed RMSE (runs) for probability calculation')
    ap.add_argument('--p-threshold', type=float, default=0.0, help='Minimum model-estimated probability of winning the bet (0-1)')
    args = ap.parse_args()
    compute_bet_outcomes(args.days, args.min_edge, args.outdir, args.consensus, args.hook_only, args.same_day_only, args.ev_filter, args.assumed_rmse, args.p_threshold)
