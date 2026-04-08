"""Ingest matchup split data from StatMuse.

Fetches batter-vs-pitcher (BvP) career stats and pitcher-vs-team career
stats for every lineup/starter combination on a given slate.  Results are
cached in the ``matchup_splits`` table so we only query StatMuse once per
unique pair per season.

Usage
-----
    python -m src.ingestors.matchup_splits --target-date 2026-04-08
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata
from datetime import datetime

import requests

from src.ingestors.common import record_ingest_event, record_source_health
from src.utils.cli import add_date_range_args, date_range, resolve_date_range
from src.utils.db import get_engine, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

STATMUSE_BASE = "https://www.statmuse.com/mlb/ask"
REQUEST_DELAY = 1.0  # seconds between requests
REQUEST_TIMEOUT = 15  # seconds
MAX_CONSECUTIVE_FAILURES = 5
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
CACHE_SEASON = 0  # career rows use season=0

# How many days a cached career row is considered fresh.
CACHE_TTL_DAYS = 30

TABLE_NAME = "matchup_splits"
CONFLICT_COLUMNS = ["player_id", "opponent_id", "split_type", "season"]

# ── team abbreviation → nickname for StatMuse URL construction ───────────────

TEAM_NICKNAMES: dict[str, str] = {
    "ARI": "diamondbacks", "ATL": "braves", "BAL": "orioles",
    "BOS": "red-sox", "CHC": "cubs", "CWS": "white-sox",
    "CIN": "reds", "CLE": "guardians", "COL": "rockies",
    "DET": "tigers", "HOU": "astros", "KCR": "royals",
    "LAA": "angels", "LAD": "dodgers", "MIA": "marlins",
    "MIL": "brewers", "MIN": "twins", "NYM": "mets",
    "NYY": "yankees", "ATH": "athletics", "OAK": "athletics",
    "PHI": "phillies", "PIT": "pirates", "SDP": "padres",
    "SFG": "giants", "SEA": "mariners", "STL": "cardinals",
    "TBR": "rays", "TEX": "rangers", "TOR": "blue-jays",
    "WSH": "nationals",
}


# ── name / URL helpers ───────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a player name to a StatMuse-friendly URL slug.

    "Shohei Ohtani" → "shohei-ohtani"
    "José Ramírez"  → "jose-ramirez"
    "Vladimir Guerrero Jr." → "vladimir-guerrero-jr"
    """
    # Decompose accented characters to base + combining mark, then drop marks.
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    slug = ascii_name.lower().strip()
    slug = slug.replace(".", "").replace("'", "").replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _bvp_url(batter_name: str, pitcher_name: str) -> str:
    return f"{STATMUSE_BASE}/{_slugify(batter_name)}-vs-{_slugify(pitcher_name)}-career"


def _pitcher_vs_team_url(pitcher_name: str, team_abbr: str) -> str:
    nickname = TEAM_NICKNAMES.get(team_abbr, team_abbr.lower())
    return f"{STATMUSE_BASE}/{_slugify(pitcher_name)}-vs-{nickname}"


def _platoon_url(player_name: str, hand: str, year: int) -> str:
    """Build a StatMuse URL for a batter's platoon split.

    ``hand`` should be ``"left"`` or ``"right"``.
    """
    return f"{STATMUSE_BASE}/{_slugify(player_name)}-vs-{hand}-handed-pitchers-{year}"


# ── HTTP fetching ────────────────────────────────────────────────────────────

def _fetch_page(url: str) -> str | None:
    """GET a StatMuse page and return the HTML body, or ``None`` on failure."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 429:
            log.warning("Rate limited by StatMuse, backing off")
            return None
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        log.warning("StatMuse fetch failed for %s: %s", url, exc)
        return None


# ── response parsing ─────────────────────────────────────────────────────────

# Pattern 1: small-sample BvP — "Player is H-for-AB in PA plate appearances"
_BVP_SHORT_RE = re.compile(
    r"is\s+(\d+)-for-(\d+)\s+in\s+(\d+)\s+plate\s+appearance",
    re.IGNORECASE,
)

# Pattern 2: larger-sample — "has a batting average of .XXX with H hits,
#   HR home runs, RBI RBIs and R runs scored in G games"
_BVP_FULL_RE = re.compile(
    r"batting\s+average\s+of\s+([.\d]+)\s+with\s+"
    r"(\d+)\s+hits?,\s+"
    r"(\d+)\s+home\s+runs?,\s+"
    r"(\d+)\s+RBIs?\s+and\s+"
    r"(\d+)\s+runs?\s+scored\s+in\s+"
    r"(\d+)\s+games?",
    re.IGNORECASE,
)

# Pattern 3: pitcher vs team — "has a W-L record with a X.XX ERA …
#   and YY strikeouts in G games/starts"
_PITCHER_VS_TEAM_RE = re.compile(
    r"(\d+)-(\d+)\s+record\s+with\s+a\s+([.\d]+)\s+ERA.*?"
    r"(\d+)\s+strikeouts?\s+in\s+(\d+)\s+(?:games?|starts?)",
    re.IGNORECASE,
)

# Pattern 3b: alternate pitcher summary — "X.XX ERA with YY strikeouts in G starts"
_PITCHER_VS_TEAM_ALT_RE = re.compile(
    r"([.\d]+)\s+ERA.*?(\d+)\s+strikeouts?\s+in\s+(\d+)\s+(?:games?|starts?)",
    re.IGNORECASE,
)

# Pattern 4: platoon split sentence — "batting average of .XXX with HR home
#   runs and RBI RBIs in PA plate appearances against {hand}-handed pitchers"
_PLATOON_SENTENCE_RE = re.compile(
    r"batting\s+average\s+of\s+([.\d]+)\s+with\s+"
    r"(\d+)\s+home\s+runs?\s+and\s+"
    r"(\d+)\s+RBIs?\s+in\s+"
    r"(\d+)\s+plate\s+appearances?\s+against\s+"
    r"(?:left|right)-handed\s+pitchers?",
    re.IGNORECASE,
)

# Pattern 4b: platoon single-row stat-table (non-"Total" row).
# Columns: # | Name | Abbrev | Year | Team | G | AB | H | 2B | 3B | HR |
#           RBI | BB | HBP | SO | PA | <variable int cols> | AVG | OBP | SLG | OPS
_PLATOON_TABLE_RE = re.compile(
    r"\|\s*\d+\s*\|"               # row number
    r"[^|]+\|"                      # full name
    r"[^|]+\|"                      # abbreviated name
    r"\s*\d{4}\s*\|"               # year
    r"[^|]+\|"                      # team
    r"\s*(\d+)\s*\|"               # G
    r"\s*(\d+)\s*\|"               # AB
    r"\s*(\d+)\s*\|"               # H
    r"\s*(\d+)\s*\|"               # 2B
    r"\s*(\d+)\s*\|"               # 3B
    r"\s*(\d+)\s*\|"               # HR
    r"\s*(\d+)\s*\|"               # RBI
    r"\s*(\d+)\s*\|"               # BB
    r"\s*(\d+)\s*\|"               # HBP
    r"\s*(\d+)\s*\|"               # SO
    r"\s*(\d+)\s*\|"               # PA
    r"(?:\s*\d+\s*\|)*"            # skip variable int cols (R, SF, SH, IBB, GDP…)
    r"\s*(\.\d+)\s*\|"             # AVG  (starts with .)
    r"\s*(\.\d+)\s*\|"             # OBP  (starts with .)
    r"\s*(\.\d+)\s*\|"             # SLG  (starts with .)
    r"\s*(\d*\.\d+)\s*\|",         # OPS  (≥1.000 possible, so optional digit before .)
    re.IGNORECASE,
)

# Stat-table totals row.  We look for the row after "Total" in the table.
_TABLE_TOTALS_RE = re.compile(
    r"Total.*?\|"
    r"\s*(\d+)\s*\|"   # AB
    r"\s*(\d+)\s*\|"   # R
    r"\s*(\d+)\s*\|"   # H
    r"\s*(\d+)\s*\|"   # 2B
    r"\s*(\d+)\s*\|"   # 3B
    r"\s*(\d+)\s*\|"   # HR
    r"\s*(\d+)\s*\|"   # RBI
    r"\s*(\d+)\s*\|"   # BB
    r"\s*(\d+)\s*\|"   # HBP
    r"\s*(\d+)\s*\|"   # SO
    r"(?:\s*\d+\s*\|){2,5}"  # skip SF/SH/etc variable columns
    r"\s*(\d+)\s*\|",  # PA (usually ~4-5 cols after SO)
    re.IGNORECASE | re.DOTALL,
)

# Pattern 5: Pitcher-vs-team totals row.
# Total row columns: G | ERA | K | W | L | SV | IP | H | R | ER | HR | BB | ...
_PVT_TOTALS_RE = re.compile(
    r"Total(?:\s*\|\s*)*"      # skip empty cells before numeric columns
    r"(\d+)\s*\|"              # G
    r"\s*([.\d]+)\s*\|"        # ERA
    r"\s*(\d+)\s*\|"           # K
    r"\s*(\d+)\s*\|"           # W
    r"\s*(\d+)\s*\|"           # L
    r"\s*(\d+)\s*\|"           # SV
    r"\s*([.\d]+)\s*\|"        # IP  (e.g. 28.0)
    r"\s*(\d+)\s*\|"           # H
    r"\s*(\d+)\s*\|"           # R
    r"\s*(\d+)\s*\|"           # ER
    r"\s*(\d+)\s*\|"           # HR
    r"\s*(\d+)\s*\|",          # BB
    re.IGNORECASE | re.DOTALL,
)


def _parse_bvp(html: str) -> dict | None:
    """Extract batter-vs-pitcher career stats from the StatMuse response."""
    # Try the full-stat sentence first (larger samples).
    m = _BVP_FULL_RE.search(html)
    if m:
        avg = float(m.group(1))
        hits = int(m.group(2))
        hr = int(m.group(3))
        rbi = int(m.group(4))
        runs = int(m.group(5))
        games = int(m.group(6))
        ab = round(hits / avg) if avg > 0 else None
        return {
            "games": games,
            "hits": hits,
            "home_runs": hr,
            "rbi": rbi,
            "runs": runs,
            "batting_avg": round(avg, 4),
            "at_bats": ab,
        }

    # Fallback: short sentence — "X-for-Y in Z plate appearances"
    m = _BVP_SHORT_RE.search(html)
    if m:
        hits = int(m.group(1))
        ab = int(m.group(2))
        pa = int(m.group(3))
        avg = round(hits / ab, 4) if ab > 0 else 0.0
        return {
            "hits": hits,
            "at_bats": ab,
            "plate_appearances": pa,
            "batting_avg": avg,
        }

    return None


def _enrich_from_totals_row(html: str, row: dict) -> dict:
    """Try to extract additional stats from the stat-table totals row."""
    m = _TABLE_TOTALS_RE.search(html)
    if not m:
        return row
    row.setdefault("at_bats", int(m.group(1)))
    row.setdefault("runs", int(m.group(2)))
    row.setdefault("hits", int(m.group(3)))
    row.setdefault("doubles", int(m.group(4)))
    row.setdefault("triples", int(m.group(5)))
    row.setdefault("home_runs", int(m.group(6)))
    row.setdefault("rbi", int(m.group(7)))
    row.setdefault("walks", int(m.group(8)))
    row.setdefault("strikeouts", int(m.group(10)))
    row.setdefault("plate_appearances", int(m.group(11)))
    return row


def _parse_pitcher_vs_team(html: str) -> dict | None:
    """Extract pitcher-vs-team career stats from the StatMuse response."""
    m = _PITCHER_VS_TEAM_RE.search(html)
    if m:
        return {
            "games": int(m.group(5)),
            "era": float(m.group(3)),
            "strikeouts": int(m.group(4)),
        }
    m = _PITCHER_VS_TEAM_ALT_RE.search(html)
    if m:
        return {
            "era": float(m.group(1)),
            "strikeouts": int(m.group(2)),
            "games": int(m.group(3)),
        }
    return None


def _enrich_pvt_from_totals_row(html: str, row: dict) -> dict:
    """Try to extract richer pitcher-vs-team stats from the stat-table Total row.

    Computes ``innings_pitched``, ``earned_runs``, ``whip``, and ``k_per_9``
    from the raw counting stats in the totals row.
    """
    m = _PVT_TOTALS_RE.search(html)
    if not m:
        return row

    games = int(m.group(1))
    k = int(m.group(3))
    ip = float(m.group(7))
    h = int(m.group(8))
    er = int(m.group(10))
    bb = int(m.group(12))

    row.setdefault("games", games)
    row.setdefault("strikeouts", k)
    row["innings_pitched"] = ip
    row["earned_runs"] = er
    row["walks"] = bb
    row["hits"] = h

    if ip > 0:
        row["whip"] = round((h + bb) / ip, 3)
        row["k_per_9"] = round(k * 9 / ip, 2)

    return row


def _parse_platoon(html: str) -> dict | None:
    """Extract batter platoon split stats from the StatMuse response.

    Tries the stat table first (richer data), then falls back to the summary
    sentence pattern.
    """
    # Try the stat-table row first — has all the counting stats + rates.
    m = _PLATOON_TABLE_RE.search(html)
    if m:
        return {
            "games": int(m.group(1)),
            "at_bats": int(m.group(2)),
            "hits": int(m.group(3)),
            "doubles": int(m.group(4)),
            "triples": int(m.group(5)),
            "home_runs": int(m.group(6)),
            "rbi": int(m.group(7)),
            "walks": int(m.group(8)),
            "strikeouts": int(m.group(10)),
            "plate_appearances": int(m.group(11)),
            "batting_avg": float(m.group(12)),
            "obp": float(m.group(13)),
            "slg": float(m.group(14)),
            "ops": float(m.group(15)),
        }

    # Fallback: summary sentence — "batting average of .XXX with HR HR and RBI RBIs …"
    m = _PLATOON_SENTENCE_RE.search(html)
    if m:
        avg = float(m.group(1))
        hr = int(m.group(2))
        rbi = int(m.group(3))
        pa = int(m.group(4))
        return {
            "batting_avg": round(avg, 4),
            "home_runs": hr,
            "rbi": rbi,
            "plate_appearances": pa,
        }

    return None


# ── slate loading ────────────────────────────────────────────────────────────

def _load_slate_matchups(engine, target_date):
    """Return batter/pitcher pairs, pitcher/team pairs, and platoon pairs.

    Returns:
        bvp_pairs: list of (batter_id, batter_name, pitcher_id, pitcher_name)
        pvt_pairs: list of (pitcher_id, pitcher_name, opponent_team)
        platoon_pairs: list of (player_id, player_name, pitcher_hand)
            where pitcher_hand is 'L' or 'R' from the opposing starter.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Get starters for each game on target_date (include pitcher throws)
        starters = conn.execute(text("""
            SELECT ps.game_id, ps.pitcher_id, ps.team, ps.side,
                   COALESCE(dp.full_name, 'Unknown') AS pitcher_name,
                   dp.throws AS pitcher_throws
            FROM pitcher_starts ps
            LEFT JOIN dim_players dp ON dp.player_id = ps.pitcher_id
            WHERE ps.game_date = :td
              AND COALESCE(ps.is_probable, TRUE) = TRUE
        """), {"td": str(target_date)}).fetchall()

        # Get games to map team sides
        games = conn.execute(text("""
            SELECT game_id, home_team, away_team
            FROM games
            WHERE game_date = :td
        """), {"td": str(target_date)}).fetchall()

        game_teams = {g[0]: {"home": g[1], "away": g[2]} for g in games}

        # Build starter map: game_id → {home_pitcher_id, away_pitcher_id, ...}
        starter_map: dict[int, dict] = {}
        for row in starters:
            game_id, pitcher_id, team, side, pitcher_name, pitcher_throws = row
            starter_map.setdefault(game_id, {})[side] = {
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name,
                "team": team,
                "throws": str(pitcher_throws or "").strip().upper()[:1],
            }

        # Get latest lineup snapshot per player per game
        lineup_rows = conn.execute(text("""
            SELECT l.game_id, l.player_id, l.team,
                   COALESCE(dp.full_name, l.player_name) AS player_name
            FROM lineups l
            LEFT JOIN dim_players dp ON dp.player_id = l.player_id
            WHERE l.game_date = :td
              AND l.lineup_slot IS NOT NULL
              AND l.lineup_slot BETWEEN 1 AND 9
        """), {"td": str(target_date)}).fetchall()

        # Deduplicate: keep one row per (game_id, player_id)
        seen_lineup = set()
        unique_lineups = []
        for row in lineup_rows:
            key = (row[0], row[1])
            if key not in seen_lineup:
                seen_lineup.add(key)
                unique_lineups.append(row)

    bvp_pairs = []
    pvt_pairs = []
    pvt_seen = set()
    platoon_pairs = []
    platoon_seen = set()

    for game_id, batter_id, batter_team, batter_name in unique_lineups:
        teams = game_teams.get(game_id)
        if not teams:
            continue
        # Determine opposing side
        if batter_team == teams["home"]:
            opp_side = "away"
        elif batter_team == teams["away"]:
            opp_side = "home"
        else:
            continue

        opp_starter = starter_map.get(game_id, {}).get(opp_side)
        if not opp_starter:
            continue

        bvp_pairs.append((
            batter_id,
            batter_name,
            opp_starter["pitcher_id"],
            opp_starter["pitcher_name"],
        ))

        # Platoon: batter vs. the opposing starter's throwing hand
        pitcher_hand = opp_starter.get("throws")
        if pitcher_hand in {"L", "R"}:
            key = (batter_id, pitcher_hand)
            if key not in platoon_seen:
                platoon_seen.add(key)
                platoon_pairs.append((batter_id, batter_name, pitcher_hand))

    # Pitcher-vs-team pairs: each starter vs. the opposing team
    for game_id, info in starter_map.items():
        teams = game_teams.get(game_id)
        if not teams:
            continue
        for side, starter in info.items():
            opp_team = teams["away"] if side == "home" else teams["home"]
            key = (starter["pitcher_id"], opp_team)
            if key not in pvt_seen:
                pvt_seen.add(key)
                pvt_pairs.append((
                    starter["pitcher_id"],
                    starter["pitcher_name"],
                    opp_team,
                ))

    return bvp_pairs, pvt_pairs, platoon_pairs


def _load_cached_keys(engine, split_type: str, season: int = CACHE_SEASON) -> set[tuple]:
    """Return the set of already-cached (player_id, opponent_id) keys."""
    from sqlalchemy import text

    cutoff = datetime.utcnow().timestamp() - (CACHE_TTL_DAYS * 86400)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT player_id, opponent_id
            FROM matchup_splits
            WHERE split_type = :st AND season = :s
        """), {"st": split_type, "s": season}).fetchall()
    return {(r[0], r[1]) for r in rows}


# ── main pipeline ────────────────────────────────────────────────────────────

def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ingest_date(engine, target_date) -> dict:
    """Ingest matchup splits for a single date. Returns counts dict."""
    bvp_pairs, pvt_pairs, platoon_pairs = _load_slate_matchups(engine, target_date)
    cached_bvp = _load_cached_keys(engine, "bvp")
    cached_pvt = _load_cached_keys(engine, "pitcher_vs_team")

    current_year = target_date.year if hasattr(target_date, "year") else int(str(target_date)[:4])
    cached_lhp = _load_cached_keys(engine, "platoon_lhp", season=current_year)
    cached_rhp = _load_cached_keys(engine, "platoon_rhp", season=current_year)

    bvp_needed = [(b_id, b_name, p_id, p_name)
                   for b_id, b_name, p_id, p_name in bvp_pairs
                   if (b_id, p_id) not in cached_bvp]
    pvt_needed = [(p_id, p_name, opp)
                   for p_id, p_name, opp in pvt_pairs
                   if (p_id, _team_abbr_to_opponent_id(opp)) not in cached_pvt]

    # For platoon, opponent_id = 0.  Only fetch the hand the batter faces today.
    platoon_needed = []
    for player_id, player_name, pitcher_hand in platoon_pairs:
        split_type = "platoon_lhp" if pitcher_hand == "L" else "platoon_rhp"
        cache = cached_lhp if pitcher_hand == "L" else cached_rhp
        if (player_id, 0) not in cache:
            platoon_needed.append((player_id, player_name, pitcher_hand, split_type))

    log.info(
        "Date %s: %d BvP pairs (%d cached, %d to fetch), "
        "%d pitcher-vs-team (%d cached, %d to fetch), "
        "%d platoon (%d cached, %d to fetch)",
        target_date,
        len(bvp_pairs), len(bvp_pairs) - len(bvp_needed), len(bvp_needed),
        len(pvt_pairs), len(pvt_pairs) - len(pvt_needed), len(pvt_needed),
        len(platoon_pairs), len(platoon_pairs) - len(platoon_needed), len(platoon_needed),
    )

    rows_to_upsert: list[dict] = []
    consecutive_failures = 0
    statmuse_ok = True

    # ── BvP queries ──────────────────────────────────────────────────────
    for batter_id, batter_name, pitcher_id, pitcher_name in bvp_needed:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.warning("Circuit breaker: %d consecutive failures, stopping BvP queries", consecutive_failures)
            statmuse_ok = False
            break

        url = _bvp_url(batter_name, pitcher_name)
        time.sleep(REQUEST_DELAY)
        html = _fetch_page(url)

        if html is None:
            consecutive_failures += 1
            continue

        parsed = _parse_bvp(html)
        if parsed is None:
            log.debug("No BvP data for %s vs %s", batter_name, pitcher_name)
            # Store a zero-PA sentinel so we don't re-query
            rows_to_upsert.append({
                "player_id": batter_id,
                "opponent_id": pitcher_id,
                "split_type": "bvp",
                "season": CACHE_SEASON,
                "plate_appearances": 0,
                "at_bats": 0,
                "hits": 0,
                "batting_avg": None,
                "fetched_at": datetime.utcnow(),
                "source_url": url,
            })
            consecutive_failures = 0
            continue

        # Enrich with totals-row data if available
        parsed = _enrich_from_totals_row(html, parsed)

        row = {
            "player_id": batter_id,
            "opponent_id": pitcher_id,
            "split_type": "bvp",
            "season": CACHE_SEASON,
            "games": _safe_int(parsed.get("games")),
            "plate_appearances": _safe_int(parsed.get("plate_appearances")),
            "at_bats": _safe_int(parsed.get("at_bats")),
            "hits": _safe_int(parsed.get("hits")),
            "home_runs": _safe_int(parsed.get("home_runs")),
            "walks": _safe_int(parsed.get("walks")),
            "strikeouts": _safe_int(parsed.get("strikeouts")),
            "rbi": _safe_int(parsed.get("rbi")),
            "runs": _safe_int(parsed.get("runs")),
            "doubles": _safe_int(parsed.get("doubles")),
            "triples": _safe_int(parsed.get("triples")),
            "batting_avg": _safe_float(parsed.get("batting_avg")),
            "obp": _safe_float(parsed.get("obp")),
            "slg": _safe_float(parsed.get("slg")),
            "ops": _safe_float(parsed.get("ops")),
            "fetched_at": datetime.utcnow(),
            "source_url": url,
        }
        rows_to_upsert.append(row)
        consecutive_failures = 0
        log.debug(
            "BvP: %s vs %s → AVG=%s PA=%s",
            batter_name, pitcher_name,
            parsed.get("batting_avg"), parsed.get("plate_appearances"),
        )

    # ── Pitcher-vs-team queries ──────────────────────────────────────────
    consecutive_failures = 0
    for pitcher_id, pitcher_name, opp_team in pvt_needed:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.warning("Circuit breaker: %d consecutive failures, stopping pitcher-vs-team queries", consecutive_failures)
            statmuse_ok = False
            break

        url = _pitcher_vs_team_url(pitcher_name, opp_team)
        time.sleep(REQUEST_DELAY)
        html = _fetch_page(url)

        if html is None:
            consecutive_failures += 1
            continue

        parsed = _parse_pitcher_vs_team(html)
        opponent_id = _team_abbr_to_opponent_id(opp_team)

        if parsed is None:
            log.debug("No pitcher-vs-team data for %s vs %s", pitcher_name, opp_team)
            rows_to_upsert.append({
                "player_id": pitcher_id,
                "opponent_id": opponent_id,
                "split_type": "pitcher_vs_team",
                "season": CACHE_SEASON,
                "games": 0,
                "fetched_at": datetime.utcnow(),
                "source_url": url,
            })
            consecutive_failures = 0
            continue

        # Enrich with IP, WHIP, K/9 from the stat-table Total row
        parsed = _enrich_pvt_from_totals_row(html, parsed)

        row = {
            "player_id": pitcher_id,
            "opponent_id": opponent_id,
            "split_type": "pitcher_vs_team",
            "season": CACHE_SEASON,
            "games": _safe_int(parsed.get("games")),
            "era": _safe_float(parsed.get("era")),
            "strikeouts": _safe_int(parsed.get("strikeouts")),
            "innings_pitched": _safe_float(parsed.get("innings_pitched")),
            "earned_runs": _safe_int(parsed.get("earned_runs")),
            "walks": _safe_int(parsed.get("walks")),
            "hits": _safe_int(parsed.get("hits")),
            "whip": _safe_float(parsed.get("whip")),
            "k_per_9": _safe_float(parsed.get("k_per_9")),
            "fetched_at": datetime.utcnow(),
            "source_url": url,
        }
        rows_to_upsert.append(row)
        consecutive_failures = 0
        log.debug(
            "PvT: %s vs %s → ERA=%s K=%s G=%s IP=%s WHIP=%s K/9=%s",
            pitcher_name, opp_team,
            parsed.get("era"), parsed.get("strikeouts"), parsed.get("games"),
            parsed.get("innings_pitched"), parsed.get("whip"), parsed.get("k_per_9"),
        )

    # ── Platoon queries ──────────────────────────────────────────────────
    consecutive_failures = 0
    for player_id, player_name, pitcher_hand, split_type in platoon_needed:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.warning(
                "Circuit breaker: %d consecutive failures, stopping platoon queries",
                consecutive_failures,
            )
            statmuse_ok = False
            break

        hand_word = "left" if pitcher_hand == "L" else "right"
        url = _platoon_url(player_name, hand_word, current_year)
        time.sleep(REQUEST_DELAY)
        html = _fetch_page(url)

        if html is None:
            consecutive_failures += 1
            continue

        parsed = _parse_platoon(html)
        if parsed is None:
            log.debug("No platoon data for %s vs %sHP %d", player_name, pitcher_hand, current_year)
            rows_to_upsert.append({
                "player_id": player_id,
                "opponent_id": 0,
                "split_type": split_type,
                "season": current_year,
                "plate_appearances": 0,
                "at_bats": 0,
                "hits": 0,
                "batting_avg": None,
                "fetched_at": datetime.utcnow(),
                "source_url": url,
            })
            consecutive_failures = 0
            continue

        row = {
            "player_id": player_id,
            "opponent_id": 0,
            "split_type": split_type,
            "season": current_year,
            "games": _safe_int(parsed.get("games")),
            "plate_appearances": _safe_int(parsed.get("plate_appearances")),
            "at_bats": _safe_int(parsed.get("at_bats")),
            "hits": _safe_int(parsed.get("hits")),
            "home_runs": _safe_int(parsed.get("home_runs")),
            "walks": _safe_int(parsed.get("walks")),
            "strikeouts": _safe_int(parsed.get("strikeouts")),
            "rbi": _safe_int(parsed.get("rbi")),
            "doubles": _safe_int(parsed.get("doubles")),
            "triples": _safe_int(parsed.get("triples")),
            "batting_avg": _safe_float(parsed.get("batting_avg")),
            "obp": _safe_float(parsed.get("obp")),
            "slg": _safe_float(parsed.get("slg")),
            "ops": _safe_float(parsed.get("ops")),
            "fetched_at": datetime.utcnow(),
            "source_url": url,
        }
        rows_to_upsert.append(row)
        consecutive_failures = 0
        log.debug(
            "Platoon: %s vs %sHP → AVG=%s PA=%s OPS=%s",
            player_name, pitcher_hand,
            parsed.get("batting_avg"), parsed.get("plate_appearances"),
            parsed.get("ops"),
        )

    # ── persist ──────────────────────────────────────────────────────────
    if rows_to_upsert:
        upserted = upsert_rows(TABLE_NAME, rows_to_upsert, CONFLICT_COLUMNS)
        log.info("Upserted %d matchup_splits rows for %s", upserted, target_date)
    else:
        log.info("No new matchup_splits rows needed for %s", target_date)

    record_source_health(
        source_name="statmuse",
        is_available=statmuse_ok,
    )

    return {
        "bvp_fetched": len(bvp_needed),
        "pvt_fetched": len(pvt_needed),
        "platoon_fetched": len(platoon_needed),
        "rows_upserted": len(rows_to_upsert),
    }


def _team_abbr_to_opponent_id(team_abbr: str) -> int:
    """Convert team abbreviation to a stable numeric ID for the opponent_id column.

    We use a simple hash to avoid needing a team dimension lookup here.
    The ID only needs to be deterministic for cache deduplication.
    """
    # Use a simple mapping: sum of ASCII values of the 3-char abbreviation.
    # This is intentionally basic — opponent_id for team-level splits is
    # only used as a unique key, never joined elsewhere.
    return sum(ord(c) for c in team_abbr.upper().ljust(3))


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch batter-vs-pitcher and pitcher-vs-team matchup data from StatMuse"
    )
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    engine = get_engine()

    total_bvp = 0
    total_pvt = 0
    total_platoon = 0
    total_upserted = 0

    for current_date in date_range(start_date, end_date):
        counts = _ingest_date(engine, current_date)
        total_bvp += counts["bvp_fetched"]
        total_pvt += counts["pvt_fetched"]
        total_platoon += counts["platoon_fetched"]
        total_upserted += counts["rows_upserted"]

    record_ingest_event(
        source_name="statmuse",
        ingestor_module="src.ingestors.matchup_splits",
        target_date=start_date,
        row_count=total_upserted,
    )

    log.info(
        "Matchup splits complete: %d BvP queries, %d PvT queries, "
        "%d platoon queries, %d rows upserted",
        total_bvp, total_pvt, total_platoon, total_upserted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
