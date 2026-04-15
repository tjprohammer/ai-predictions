-- First-inning team and combined runs from linescore (for NRFI/YRFI grading).
ALTER TABLE games ADD COLUMN IF NOT EXISTS home_runs_inning1 SMALLINT;
ALTER TABLE games ADD COLUMN IF NOT EXISTS away_runs_inning1 SMALLINT;
ALTER TABLE games ADD COLUMN IF NOT EXISTS total_runs_inning1 SMALLINT;
