# 1) (Re)init clean to be safe
docker compose down -v
docker compose up -d

# 2) Confirm service name and role
docker compose ps
docker compose exec -T db psql -U mlbuser -d mlb -c "\du"   # should list mlbuser

# 3) Apply schema (you already verified /sql/schema.sql exists)
docker compose exec -T db psql -U mlbuser -d mlb -f /sql/schema.sql

# 4) Set DB URL for your shell
$env:DATABASE_URL = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
