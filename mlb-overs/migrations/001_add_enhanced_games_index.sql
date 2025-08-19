-- One-time migration: ensure unique constraint on (game_id, date) for enhanced_games upserts
-- Idempotent via conditional check (works even if tooling lacks CREATE INDEX IF NOT EXISTS parsing)
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_class c
		JOIN pg_namespace n ON n.oid = c.relnamespace
		WHERE c.relkind = 'i'
		  AND c.relname = 'uq_enhanced_games_gid_date'
	) THEN
		EXECUTE 'CREATE UNIQUE INDEX uq_enhanced_games_gid_date ON enhanced_games (game_id, "date")';
	END IF;
END$$;
