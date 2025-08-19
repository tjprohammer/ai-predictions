# ingestors/espn_totals.py
from __future__ import annotations
import argparse, datetime as dt, re
import requests, pandas as pd
from sqlalchemy import text
from ingestors.util import get_engine

# Try these endpoints + two date formats
ESPN_ENDPOINTS = [
    "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "https://site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
]
# map abbreviations -> nickname keys used in your games table normalization
ABBR2NICK = {
    "ARI":"diamondbacks","ATL":"braves","BAL":"orioles","BOS":"red sox","CHC":"cubs","CWS":"white sox",
    "CIN":"reds","CLE":"guardians","COL":"rockies","DET":"tigers","HOU":"astros","KC":"royals",
    "LAA":"angels","LAD":"dodgers","MIA":"marlins","MIL":"brewers","MIN":"twins","NYM":"mets",
    "NYY":"yankees","OAK":"athletics","PHI":"phillies","PIT":"pirates","SD":"padres","SEA":"mariners",
    "SF":"giants","STL":"cardinals","TB":"rays","TEX":"rangers","TOR":"blue jays","WSH":"nationals",
    # alt abbr variants we sometimes see
    "WSN":"nationals","SDP":"padres","SFG":"giants","TBR":"rays","CWS":"white sox","LAD":"dodgers",
}

def team_key_from_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    s = name.strip().lower()
    # multi-word exceptions
    if "white sox" in s: return "white sox"
    if "red sox"   in s: return "red sox"
    if "blue jays" in s: return "blue jays"
    # otherwise last token is the nickname
    toks = re.split(r"\s+", s)
    return toks[-1] if toks else s

def team_key_from_abbr_or_name(x: str) -> str:
    if not isinstance(x, str) or not x:
        return ""
    if x.upper() in ABBR2NICK:
        return ABBR2NICK[x.upper()]
    return team_key_from_name(x)

def parse_over_under(odds_obj) -> float | None:
    """ESPN odds may expose 'overUnder' or embed in 'details' like 'O/U 8.5' or 'Total: 9'."""
    if not odds_obj:
        return None
    ou = odds_obj.get("overUnder")
    if ou is not None:
        try:
            return float(ou)
        except:
            pass
    det = (odds_obj.get("details") or "").strip()
    # examples: 'O/U 8.5', 'Total: 9', 'over/under 9.5'
    m = re.search(r'(?i)(?:o/?u|over/?under|total)\s*[: ]*\s*(\d+(?:\.\d+)?)', det)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def _fmt_dates(d: str) -> tuple[str, str]:
    """Return (YYYY-MM-DD, YYYYMMDD)."""
    d_iso = pd.to_datetime(d).date().isoformat()
    return d_iso, d_iso.replace("-", "")

def _get_json(url: str, params: dict) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def fetch_espn_totals(date_str: str) -> pd.DataFrame:
    d_iso, d_comp = _fmt_dates(date_str)
    js = None
    # try YYYYMMDD then YYYY-MM-DD on both endpoints
    for base in ESPN_ENDPOINTS:
        for dates in (d_comp, d_iso):
            js = _get_json(base, {"dates": dates, "limit": 300})
            if js and js.get("events"):
                break
        if js and js.get("events"):
            break

    if not js or not js.get("events"):
        return pd.DataFrame()

    events = js.get("events") or []
    rows = []
    snap_ts = pd.Timestamp.utcnow()

    for ev in events:
        competitions = ev.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        if len(competitors) != 2:
            continue

        home_obj = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_obj = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_obj or not away_obj:
            continue

        home_team = (home_obj.get("team") or {}).get("displayName") or (home_obj.get("team") or {}).get("name")
        away_team = (away_obj.get("team") or {}).get("displayName") or (away_obj.get("team") or {}).get("name")
        home_key = team_key_from_abbr_or_name((home_obj.get("team") or {}).get("abbreviation") or home_team)
        away_key = team_key_from_abbr_or_name((away_obj.get("team") or {}).get("abbreviation") or away_team)

        status = (comp.get("status") or {}).get("type") or {}
        state = (status.get("state") or "").lower()  # 'pre' | 'in' | 'post'

        k_total = None
        book = None

        # look for a total in the odds blocks
        # 1) try odds array on the competition
        for bk in (comp.get("odds") or []):
            k = parse_over_under(bk)
            if k is not None:
                k_total = k
                book = book or (bk.get("provider", {}) or {}).get("name") or "espn"
                break  # we found a total; stop scanning odds blocks

        # 2) fallback: event summary endpoint
        if k_total is None:
            ev_id = ev.get("id")
            if ev_id:
                k_try = fetch_event_total_fallback(ev_id)
                if k_try is not None:
                    k_total = k_try
                    book = book or "espn"

        # if still none, skip this event
        if k_total is None:
            continue


        rows.append({
            "event_id": ev.get("id"),
            "date": pd.to_datetime(d_iso).date(),
            "home_team": home_team, "away_team": away_team,
            "home_key": home_key,   "away_key": away_key,
            "book": book or "espn",
            "k_total": float(k_total),
            "state": state,
            "snapshot_ts": snap_ts,
        })

    return pd.DataFrame(rows)

def fetch_event_total_fallback(event_id: str) -> float | None:
    url = "https://site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"
    try:
        r = requests.get(url, params={"event": event_id}, timeout=20)
        r.raise_for_status()
        js = r.json() or {}

        # odds can be in 'pickcenter' or 'odds'
        blocks = (js.get("pickcenter") or []) + (js.get("odds") or [])
        best = None
        for o in blocks:
            ou = o.get("overUnder")
            if ou is None:
                det = (o.get("details") or "")
                m = re.search(r'(?i)(?:o/?u|over/?under|total)\s*[: ]*\s*(\d+(?:\.\d+)?)', det)
                if m: ou = m.group(1)
            if ou is None:
                continue
            try:
                best = float(ou)
            except:
                pass
        return best
    except Exception:
        return None


def attach_game_ids(df: pd.DataFrame, eng, date_str: str) -> pd.DataFrame:
    if df.empty:
        return df
    games = pd.read_sql(
        text("SELECT game_id, date, home_team, away_team FROM games WHERE date = :d"),
        eng, params={"d": pd.to_datetime(date_str).date()}
    )
    if games.empty:
        return pd.DataFrame()
    games["home_key"] = games["home_team"].astype(str).apply(team_key_from_abbr_or_name)
    games["away_key"] = games["away_team"].astype(str).apply(team_key_from_abbr_or_name)
    out = df.merge(games[["game_id","home_key","away_key","date"]], on=["home_key","away_key","date"], how="inner")
    return out

def upsert_markets(df: pd.DataFrame, eng) -> int:
    if df.empty: return 0
    df = df.copy()
    df["market_type"] = "snapshot"
    with eng.begin() as cx:
        df.to_sql("tmp_markets_totals", cx, if_exists="replace", index=False)
        cx.execute(text("""
            INSERT INTO markets_totals (game_id, date, book, market_type, k_total, snapshot_ts)
            SELECT game_id, date, book, market_type, k_total, snapshot_ts
            FROM tmp_markets_totals
            -- use the column list, not a constraint name
            ON CONFLICT (game_id, date, book, market_type, snapshot_ts) DO NOTHING
        """))
        cx.execute(text("DROP TABLE tmp_markets_totals"))
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    eng = get_engine()
    date_str = args.date

    raw = fetch_espn_totals(date_str)
    if raw.empty:
        print(f"[ingestors.espn_totals] no odds rows for {date_str}")
        return

    matched = attach_game_ids(raw, eng, date_str)
    if matched.empty:
        print(f"[ingestors.espn_totals] no matches to games for {date_str} (check team keys)")
        return

    n = upsert_markets(matched, eng)
    print(f"[ingestors.espn_totals] upserted {n} rows (snapshots/close) for {date_str}")

if __name__ == "__main__":
    main()
