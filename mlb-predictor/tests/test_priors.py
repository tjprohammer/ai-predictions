from src.features.priors import blend_with_prior, effective_full_weight, normalize_prior_mode, sample_weight


def test_sample_weight_clamps_to_zero_and_one():
    assert sample_weight(None, 100) == 0.0
    assert sample_weight(0, 100) == 0.0
    assert sample_weight(50, 100) == 0.5
    assert sample_weight(150, 100) == 1.0


def test_blend_with_prior_uses_both_inputs():
    blended = blend_with_prior(0.4, 0.3, 60, 120)
    assert blended == 0.35


def test_blend_with_prior_handles_missing_values():
    assert blend_with_prior(None, 0.2, 30, 100) == 0.2
    assert blend_with_prior(0.4, None, 30, 100) == 0.4


def test_prior_modes_cover_standard_reduced_and_current_only():
    assert normalize_prior_mode("blend") == "standard"
    assert normalize_prior_mode("light") == "reduced"
    assert normalize_prior_mode("off") == "current_only"
    assert effective_full_weight(120, mode="standard", prior_weight_multiplier=1.0) == 120.0
    assert effective_full_weight(120, mode="reduced", prior_weight_multiplier=1.0) == 60.0
    assert effective_full_weight(120, mode="current_only", prior_weight_multiplier=1.0) == 0.0
    assert blend_with_prior(0.4, 0.3, 10, 120, mode="current_only") == 0.4
    assert blend_with_prior(0.4, 0.3, 60, 120, mode="reduced") == 0.4