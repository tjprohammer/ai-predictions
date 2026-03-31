from __future__ import annotations


def normalize_prior_mode(raw: str | None) -> str:
    value = (raw or "standard").strip().lower()
    aliases = {
        "blend": "standard",
        "default": "standard",
        "light": "reduced",
        "lighter": "reduced",
        "reduced": "reduced",
        "current": "current_only",
        "off": "current_only",
        "none": "current_only",
    }
    return aliases.get(value, value if value in {"standard", "reduced", "current_only"} else "standard")


def effective_full_weight(
    full_weight: float | int,
    mode: str = "standard",
    prior_weight_multiplier: float = 1.0,
) -> float:
    resolved_mode = normalize_prior_mode(mode)
    if resolved_mode == "current_only":
        return 0.0
    adjusted = float(full_weight) * max(float(prior_weight_multiplier), 0.0)
    if resolved_mode == "reduced":
        adjusted *= 0.5
    return max(adjusted, 0.0)


def sample_weight(sample_size: float | int | None, full_weight: float | int) -> float:
    if sample_size is None:
        return 0.0
    if full_weight <= 0:
        return 1.0
    return max(0.0, min(float(sample_size) / float(full_weight), 1.0))


def blend_with_prior(
    current_value: float | None,
    prior_value: float | None,
    sample_size: float | int | None,
    full_weight: float | int,
    mode: str = "standard",
    prior_weight_multiplier: float = 1.0,
) -> float | None:
    resolved_mode = normalize_prior_mode(mode)
    if current_value is None and prior_value is None:
        return None
    if resolved_mode == "current_only":
        return current_value
    if current_value is None:
        return prior_value
    if prior_value is None:
        return current_value
    weight = sample_weight(sample_size, effective_full_weight(full_weight, resolved_mode, prior_weight_multiplier))
    return weight * current_value + (1.0 - weight) * prior_value