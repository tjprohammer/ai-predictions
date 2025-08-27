-- Migration script: Add umpire stat columns to enhanced_games table
-- Umpire names and positions
ALTER TABLE enhanced_games ADD home_plate_umpire_name VARCHAR(100);
ALTER TABLE enhanced_games ADD first_base_umpire_name VARCHAR(100);
ALTER TABLE enhanced_games ADD second_base_umpire_name VARCHAR(100);
ALTER TABLE enhanced_games ADD third_base_umpire_name VARCHAR(100);

-- Plate umpire stats (most impactful for totals)
ALTER TABLE enhanced_games ADD plate_umpire_k_pct FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_bb_pct FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_strike_zone_consistency FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_avg_strikes_per_ab FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_rpg FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_ba_against FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_obp_against FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_slg_against FLOAT;
ALTER TABLE enhanced_games ADD plate_umpire_boost_factor FLOAT;

-- Base umpire aggregate stats (less impact but still relevant)
ALTER TABLE enhanced_games ADD base_umpires_experience_avg FLOAT;
ALTER TABLE enhanced_games ADD base_umpires_error_rate FLOAT;
ALTER TABLE enhanced_games ADD base_umpires_close_call_accuracy FLOAT;

-- Overall umpire crew stats
ALTER TABLE enhanced_games ADD umpire_crew_total_experience FLOAT;
ALTER TABLE enhanced_games ADD umpire_crew_consistency_rating FLOAT;
-- Add more columns as needed for additional umpire stats
