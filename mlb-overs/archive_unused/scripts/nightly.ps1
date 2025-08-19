Write-Host "Nightly ingest + train"
$ErrorActionPreference = 'Stop'
if (-not $env:DATABASE_URL) { $env:DATABASE_URL = 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb' }

$y = (Get-Date).AddDays(-1).ToString('yyyy-MM-dd')
python -m ingestors.games --start $y --end $y
python -m ingestors.pitchers_last10 --start $y --end $y
python -m ingestors.bullpens_daily --start $y --end $y
python -m ingestors.offense_daily --start $y --end $y
if ($env:THE_ODDS_API_KEY) { python -m ingestors.odds_totals --snapshot }

python features\build_features.py --database-url "$env:DATABASE_URL" --out features\train.parquet

if ((Get-Date).DayOfWeek -eq 'Sunday') {
  python models\train.py --data features\train.parquet --artifacts artifacts
}