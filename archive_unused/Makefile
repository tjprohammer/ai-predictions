.PHONY: up down psql init games pitchers bullpens offense odds odds_espn odds_fanduel odds_close features train infer strong gameday

# -------- Docker/Postgres ----------
up:
	docker compose up -d

down:
	docker compose down

psql:
	docker compose exec -it db psql -U mlbuser -d mlb

init:
	docker compose exec -T db psql -U mlbuser -d mlb -f /sql/schema.sql

# -------- Ingest ----------
# Usage: make games START=2025-04-01 END=2025-04-07
games:
	python -m ingestors.games --start $(START) --end $(END)

# Usage: make pitchers START=2025-04-01 END=2025-04-07
pitchers:
	python -m ingestors.pitchers_last10 --start $(START) --end $(END)

# Usage: make bullpens START=2025-04-01 END=2025-04-07
bullpens:
	python -m ingestors.bullpens_daily --start $(START) --end $(END)

# Usage: make offense START=2025-04-01 END=2025-04-07
offense:
	python -m ingestors.offense_daily --start $(START) --end $(END)

# ---- Market lines ----
# Aggregator (book filtered by ODDS_BOOK env; e.g., ODDS_BOOK=fanduel)
odds:
	python -m ingestors.odds_totals --snapshot

# ESPN direct (best-effort). Usage: make odds_espn DATE=2025-04-01
odds_espn:
	python -m ingestors.espn_totals --date $(DATE)

# FanDuel via aggregator. Usage: make odds_fanduel
odds_fanduel:
	ODDS_BOOK=fanduel python -m ingestors.odds_totals --snapshot

# Optional “closing line” snapshot near first pitch. Usage: make odds_close DATE=2025-04-01
odds_close:
	python -m ingestors.espn_totals --date $(DATE) --as-close

# -------- Features / Model ----------
features:
	python features/build_features.py --database-url "$$DATABASE_URL" --out features/train.parquet

train:
	python models/train.py --data features/train.parquet --artifacts artifacts

infer:
	python models/infer.py --database-url "$$DATABASE_URL" --artifacts artifacts --out predictions_today.parquet

strong:
	python - <<'PY'
import pandas as pd
from models.filter import strong_plays
pd.set_option('display.max_columns', None)
df = pd.read_parquet('predictions_today.parquet')
sp = strong_plays(df, min_edge=0.6, min_conf=0.80)
sp.to_csv('strong_plays.csv', index=False)
print(sp.head())
PY

# -------- One-shot gameday (today unless overridden) ----------
# Usage: make gameday DATE=2025-04-01  (or set START/END to a range)
gameday:
	$(MAKE) games START=$(DATE) END=$(DATE)
	$(MAKE) pitchers START=$(DATE) END=$(DATE)
	$(MAKE) bullpens START=$(DATE) END=$(DATE)
	$(MAKE) offense START=$(DATE) END=$(DATE)
	$(MAKE) odds_fanduel
	$(MAKE) odds_espn DATE=$(DATE)
	$(MAKE) features
	$(MAKE) infer
	$(MAKE) strong
