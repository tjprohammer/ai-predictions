"""Grade all prediction lanes for a given date."""
import sys
import pandas as pd
from src.utils.db import query_df


def _ou_record(pred, actual, mkt):
    """Compute O/U record where market line is available."""
    has = mkt.notna()
    if has.sum() == 0:
        return "no market lines"
    sides = ["O" if p >= m else "U" for p, m in zip(pred[has], mkt[has])]
    actual_s = ["O" if a > m else ("U" if a < m else "P") for a, m in zip(actual[has], mkt[has])]
    w = sum(1 for s, a in zip(sides, actual_s) if s == a)
    l = sum(1 for s, a in zip(sides, actual_s) if s != a and a != "P")
    p = sum(1 for a in actual_s if a == "P")
    return f"{w}W-{l}L-{p}P  ({has.sum()} w/ lines)"


def grade(game_date: str):
    # --- FULL-GAME TOTALS ---
    df = query_df(
        """
        SELECT p.game_id, p.predicted_total_runs as pred, p.market_total as mkt,
               p.confidence_level, p.suppress_reason, p.lane_status,
               g.total_runs as actual, g.away_team, g.home_team
        FROM predictions_totals p
        JOIN games g ON g.game_id = p.game_id
        WHERE p.game_date = :d AND LOWER(g.status) = 'final'
        """,
        {"d": game_date},
    )
    if len(df) > 0:
        pred = df["pred"].astype(float)
        actual = df["actual"].astype(float)
        mkt = pd.to_numeric(df["mkt"], errors="coerce")
        mae = (pred - actual).abs().mean()
        print(f"=== FULL-GAME TOTALS ({game_date}) - {len(df)} games ===")
        print(f"  Lane: {df.lane_status.iloc[0]}, Collapse: {df.suppress_reason.iloc[0]}")
        print(f"  Pred range: {pred.min():.1f}-{pred.max():.1f} (std={pred.std():.2f})")
        print(f"  Avg pred: {pred.mean():.2f}  Avg actual: {actual.mean():.2f}  MAE: {mae:.2f}")
        print(f"  Would-be O/U (suppressed): {_ou_record(pred, actual, mkt)}")
    else:
        print(f"=== FULL-GAME TOTALS ({game_date}) - No graded games ===")

    # --- FIRST-5 TOTALS ---
    df5 = query_df(
        """
        SELECT p.game_id, p.predicted_total_runs as pred, p.market_total as mkt,
               p.confidence_level, p.suppress_reason,
               g.total_runs_first5 as actual, g.away_team, g.home_team
        FROM predictions_first5_totals p
        JOIN games g ON g.game_id = p.game_id
        WHERE p.game_date = :d AND LOWER(g.status) = 'final'
        """,
        {"d": game_date},
    )
    if len(df5) > 0:
        pred5 = df5["pred"].astype(float)
        actual5 = df5["actual"].astype(float)
        mkt5 = pd.to_numeric(df5["mkt"], errors="coerce")
        mae5 = (pred5 - actual5).abs().mean()
        print(f"\n=== FIRST-5 TOTALS ({game_date}) - {len(df5)} games ===")
        print(f"  Pred range: {pred5.min():.1f}-{pred5.max():.1f} (std={pred5.std():.2f})")
        print(f"  Avg pred: {pred5.mean():.2f}  Avg actual: {actual5.mean():.2f}  MAE: {mae5:.2f}")
        print(f"  Confidence: {dict(df5['confidence_level'].value_counts())}")
        suppress = df5['suppress_reason'].value_counts()
        if len(suppress) > 0:
            print(f"  Suppress: {dict(suppress)}")
        print(f"  O/U: {_ou_record(pred5, actual5, mkt5)}")
    else:
        print(f"\n=== FIRST-5 TOTALS ({game_date}) - No graded games ===")

    # --- STRIKEOUTS ---
    dfk = query_df(
        """
        SELECT p.game_id, p.pitcher_id, p.team,
               p.predicted_strikeouts as pred, p.market_line as mkt,
               pgp.strikeouts as actual
        FROM predictions_pitcher_strikeouts p
        JOIN player_game_pitching pgp ON pgp.game_id = p.game_id AND pgp.player_id = p.pitcher_id
        JOIN games g ON g.game_id = p.game_id
        WHERE p.game_date = :d AND LOWER(g.status) = 'final'
        """,
        {"d": game_date},
    )
    if len(dfk) > 0:
        predk = dfk["pred"].astype(float)
        actualk = dfk["actual"].astype(float)
        mktk = pd.to_numeric(dfk["mkt"], errors="coerce")
        maek = (predk - actualk).abs().mean()
        print(f"\n=== STRIKEOUTS ({game_date}) - {len(dfk)} pitchers ===")
        print(f"  Pred range: {predk.min():.1f}-{predk.max():.1f}")
        print(f"  Avg pred: {predk.mean():.2f}  Avg actual: {actualk.mean():.2f}  MAE: {maek:.2f}")
        print(f"  O/U: {_ou_record(predk, actualk, mktk)}")
    else:
        cnt = query_df(
            "SELECT COUNT(*) as n FROM predictions_pitcher_strikeouts WHERE game_date = :d",
            {"d": game_date},
        )
        print(f"\n=== STRIKEOUTS ({game_date}) - {cnt['n'].iloc[0]} predictions (no actuals/boxscores yet) ===")

    # --- HITS ---
    dfh = query_df(
        """
        SELECT p.game_id, p.player_id, p.team,
               p.predicted_hit_probability as pred_prob, p.fair_price, p.market_price as mkt,
               pgb.hits as actual_hits, pgb.at_bats
        FROM predictions_player_hits p
        JOIN player_game_batting pgb ON pgb.game_id = p.game_id AND pgb.player_id = p.player_id
        JOIN games g ON g.game_id = p.game_id
        WHERE p.game_date = :d AND LOWER(g.status) = 'final'
        """,
        {"d": game_date},
    )
    if len(dfh) > 0:
        pred_prob = dfh["pred_prob"].astype(float)
        actual_h = dfh["actual_hits"].astype(float)
        got_hit = (actual_h >= 1).astype(int)
        from sklearn.metrics import log_loss, brier_score_loss
        try:
            ll = log_loss(got_hit, pred_prob)
            bs = brier_score_loss(got_hit, pred_prob)
        except Exception:
            ll = bs = None
        hit_rate = got_hit.mean()
        print(f"\n=== HITS ({game_date}) - {len(dfh)} batters ===")
        print(f"  Avg pred prob: {pred_prob.mean():.3f}  Actual hit rate: {hit_rate:.3f}")
        if ll is not None:
            print(f"  Log-loss: {ll:.4f}  Brier: {bs:.4f}")
        mkt_h = pd.to_numeric(dfh["mkt"], errors="coerce")
        fair_h = pd.to_numeric(dfh["fair_price"], errors="coerce")
        has_edge = mkt_h.notna() & fair_h.notna()
        if has_edge.sum() > 0:
            edge = (fair_h[has_edge] - mkt_h[has_edge])
            print(f"  Avg edge (fair-market): {edge.mean():.1f}  ({has_edge.sum()} w/ prices)")
            # For those with edges, did the hit happen?
            edge_plays = dfh[has_edge & (fair_h > mkt_h)]
            if len(edge_plays) > 0:
                ep_hits = (edge_plays["actual_hits"].astype(float) >= 1).sum()
                print(f"  Playable edges: {len(edge_plays)}, hit: {ep_hits}/{len(edge_plays)} ({ep_hits/len(edge_plays)*100:.0f}%)")
    else:
        cnt = query_df(
            "SELECT COUNT(*) as n FROM predictions_player_hits WHERE game_date = :d",
            {"d": game_date},
        )
        print(f"\n=== HITS ({game_date}) - {cnt['n'].iloc[0]} predictions (no actuals/boxscores yet) ===")


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-04-06"
    grade(date)
