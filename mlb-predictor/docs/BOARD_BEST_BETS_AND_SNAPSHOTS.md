# Best bets, Top EV, snapshots, and the pregame pipeline

This document describes how **team best bets** (green strip) differ from **Top EV** (weighted expected-value headline picks), how **run** vs **lock** snapshots keep picks stable for Daily Results, and how **lineups / markets** ingest interacts with **pregame lock** timing.

For the high-level surface map (green vs watchlist vs slugger vs experimental), see `BOARD_AND_DAILY_RESULTS_PRODUCT.md`.

For **why we track picks**, how **`prediction_outcomes_daily`** feeds **scorecards / calibration / learning**, and what is **automated vs offline retrain**, see **`LEARNING_CHANNEL_AND_EDGE.md`**.

---

## 1. Two different “headline” concepts

| | **Best bets (green strip)** | **Top EV (“EV bet” headline)** |
|---|-----------------------------|--------------------------------|
| **What it is** | Up to one **team/game market card** per game from `BEST_BET_MARKET_KEYS` (moneyline, spreads, game total, first-five markets, etc.). Shown as `best_bets` on `GET /api/games/board` after `flatten_best_bets`. | The single **highest weighted-EV** priced candidate among the same board’s eligible markets for that game (`top_ev_pick` on each game row). |
| **Selection rules** | Must be a **strict positive** (`positive`) *or* pass **soft green** gates (`qualifies_board_green_strip`: weighted EV, probability edge, input trust, per-game certainty, etc.). See `src/utils/best_bets.py`. | Pure **ranking**: `collect_top_ev_candidates` → `select_top_weighted_ev_pick` (`src/utils/top_ev_pick.py`). No separate “soft green”; it is always the EV winner among priced candidates the board considers. |
| **Hitter props** | Excluded from team best-bet surfaces (HR/hits use other panels). | Eligible only when they appear in the Top EV candidate pool (same collector rules). |
| **Typical use** | “What side/market we like most under promotion rules.” | “Which priced line has the best weighted EV right now.” |

They can disagree: the green pick might be a soft-green total while Top EV highlights a first-five side, depending on edges and gates.

---

## 2. Snapshot system (stability for grading and UI)

Snapshots exist so a **later** refresh (after line move, lineup change, or boxscore) does not rewrite what we **record** for Daily Results / analytics.

### 2.1 Run snapshots (earliest stable anchor)

- **Tables:** `board_green_run_snapshots`, `board_top_ev_run_snapshots` (`db/migrations/030_board_run_snapshots.sql`).
- **Top EV run (`board_top_ev_run_snapshots`) — automatic (no runtime toggle):**
  - **Pregame:** The first board load **after** the Top EV lock window opens freezes the headline (same clock as `MLB_PREGAME_INGEST_LOCK_MINUTES` / optional `BOARD_TOP_EV_SNAPSHOT_LOCK_MINUTES`). **Before** that instant, Top EV is **live** and can move as markets/lineups update.
  - **Catch-up:** If you never loaded the board inside that window, the first load **after** first pitch still writes once (`INSERT OR IGNORE`) so Daily Results stops drifting.
  - **API:** Each game on `GET /api/games/board` includes `top_ev_snapshot_info` (`state`, `first_freeze_eligible_after_utc`, `effective_lock_minutes`, `summary`) so you can see *when* freeze becomes eligible without editing `.env` while the server runs.
- **Green run (`board_green_run_snapshots`):** Still first qualifying board build while **before** first pitch (pregame-only), unless changed elsewhere.

### 2.2 Lock snapshots (pregame freeze)

- **Tables:** `board_green_snapshots` (`028`), `board_top_ev_snapshots` (`029`).
- **When written:** First time the game is inside the **Top EV / green lock clock** (see §4) **and** still before first pitch.
- **Meaning:** “What we freeze **near** decision time (configurable minutes before first pitch).”

### 2.3 Resolution order (everywhere we freeze)

For **Top EV** and for **outcomes** builders:

1. **Lock** snapshot row, if present  
2. Else **run** snapshot row, if present  
3. Else **live** recompute from current DB state  

The main **game board** (`_fetch_game_board`) applies the same order to `top_ev_pick` after inserting any missing snapshot rows on that request, so the headline EV pick can stabilize **as soon as** the first run/lock row exists—including on the **same** HTTP response as the first qualifying load.

Green strip merging (`_merge_board_green_snapshots_into_live`) prefers **lock** when the game is in the **ingest** lock window, else **run** while still pregame, else live.

---

## 3. Update lineups, markets, and pregame ingest lock

### 3.1 What “update lineups / markets” does

- **Lineups** (and related prep) refresh who is expected to play; feeds feature and certainty signals on the board.
- **Markets** ingest pulls or refreshes **posted odds/lines** into `game_markets` / related tables so EV and best-bet cards have prices.

Both are **mutating** steps: they change inputs that feed `build_market_cards_for_game`, `flatten_best_bets`, and Top EV.

### 3.2 `MLB_PREGAME_INGEST_LOCK_MINUTES`

- Default in code is **10** (override in `.env`). Implemented in `src/utils/pregame_lock.py` as `is_pregame_ingest_locked`: true from **N minutes before scheduled first pitch onward** (and still true **after** first pitch so late games keep skipping harmful writes).
- **Effect:** Ingest jobs that respect this flag **stop mutating** lineups/markets for that game once the lock window starts, so the DB does not keep shifting under you right at post time. A **narrower** N (e.g. 10 vs 30) means lines and EV can update **later** into the pregame window; a **wider** N freezes earlier.

### 3.3 How this interacts with snapshots

- **Run** snapshots can be taken **hours before** lock—as soon as someone loads the board pregame—so you have a stable “early” anchor even while ingest is still allowed to update.
- **Lock** snapshots are taken when the **separate** Top EV / green lock clock fires (defaults tied to `MLB_PREGAME_INGEST_LOCK_MINUTES` unless you override `BOARD_TOP_EV_SNAPSHOT_LOCK_MINUTES`).

---

## 4. Showing picks earlier (not only right before game time)

**Problem:** If the only frozen state were “minutes before first pitch,” anything that only **displays** at that moment feels too late to bet.

**What we already do:**

1. **Run snapshots** fire on the **first** pregame board computation, not at the last minute.
2. **Green strip** can show a **run-frozen** card pregame even before the ingest-lock window (see merge rules).
3. **Top EV on the board** uses **lock → run → live** so once a snapshot row exists, the API returns that pick without waiting for game time.
4. **Tuning knobs** (see `config/.env.example`):
   - **`BOARD_TOP_EV_SNAPSHOT_LOCK_MINUTES`** (optional): **Larger** value = snapshot taken **earlier** on the clock (more minutes **before** first pitch). **Smaller** = closer to first pitch. Unset = inherit `MLB_PREGAME_INGEST_LOCK_MINUTES`.
   - **`MLB_PREGAME_INGEST_LOCK_MINUTES`**: When **ingest** stops mutating; separate from when we **record** Top EV snapshots unless you inherit.

**Practical schedule (example):**

- **Night before / morning:** Run predictions + lineups + markets as needed; load the board once—**run** snapshots capture early picks.
- **Closer to game:** When the lock window hits, **lock** snapshots record the freeze for Daily Results; ingest mutating stops per lock settings.
- **Re-runs:** `INSERT OR IGNORE` semantics mean **re-running jobs does not overwrite** an existing snapshot for that game/date; you are not forced to wait until post time to “lock in” history.

---

## 5. Daily Results and `prediction_outcomes_daily`

- Daily Results pulls the same board defaults as the UI where possible (`_fetch_daily_results`).
- Top EV rows prefer **lock → run → live** (`_live_top_ev_rows_for_daily_results`, `product_surfaces` Top EV outcome builder).
- Green rows use `_flatten_best_bets`, which triggers snapshot inserts and green merge.

---

## 6. Primary code references

| Topic | Location |
|-------|----------|
| Green / watchlist flatten, thresholds | `src/utils/best_bets.py` |
| Top EV candidate collection + weighted EV | `src/utils/top_ev_pick.py` |
| Board payload, Top EV + snapshot resolution | `src/api/app_logic.py` (`_fetch_game_board`, `_flatten_best_bets`, `_resolve_top_ev_pick_with_snapshots`) |
| Pregame clock helpers | `src/utils/pregame_lock.py` |
| Settings | `src/utils/settings.py`, `config/.env.example` |
| Outcomes / scorecards | `src/transforms/product_surfaces.py` |

---

## 7. Migrations

| Migration | Tables |
|-----------|--------|
| `028_board_green_snapshots.sql` | Lock-time green cards |
| `029_board_top_ev_snapshots.sql` | Lock-time Top EV |
| `030_board_run_snapshots.sql` | First-run green + Top EV |

Apply migrations on Postgres/SQLite seeds before relying on snapshots in production.
