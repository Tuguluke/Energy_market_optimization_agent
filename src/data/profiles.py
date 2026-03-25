"""
Capacity factor profiles and price scenario generation.

Capacity factor: fraction of max capacity actually available each hour.
  - Solar: follows a bell curve peaking at noon
  - Wind: stochastic, here we use a representative daily shape

Price scenarios: Monte Carlo draws from a distribution around a base forecast.
"""
import numpy as np
from typing import List


def solar_capacity_factors(hours: int = 24) -> List[float]:
    """Typical solar generation profile for a sunny day."""
    factors = []
    for h in range(hours):
        # Bell curve centered at hour 12 (noon), zero at night
        if 6 <= h <= 20:
            cf = np.exp(-0.5 * ((h - 13) / 3.5) ** 2)
        else:
            cf = 0.0
        factors.append(round(cf, 3))
    return factors


def wind_capacity_factors(hours: int = 24, seed: int = 42) -> List[float]:
    """Smoothed wind profile — moderate during day, stronger at night."""
    rng = np.random.default_rng(seed)
    base = [0.5, 0.55, 0.6, 0.65, 0.6, 0.5,   # 0-5
            0.4, 0.35, 0.3, 0.3, 0.35, 0.4,   # 6-11
            0.45, 0.5, 0.5, 0.45, 0.5, 0.6,   # 12-17
            0.65, 0.7, 0.75, 0.7, 0.65, 0.6]  # 18-23
    noise = rng.normal(0, 0.05, hours)
    return [max(0.0, min(1.0, round(base[h % 24] + noise[h], 3))) for h in range(hours)]


def base_price_forecast(hours: int = 24) -> List[float]:
    """
    Typical day-ahead electricity price shape ($/MWh).
    Two peaks: morning ramp (7-9 AM) and evening peak (6-8 PM).
    """
    base = [25, 22, 20, 19, 20, 25,   # midnight to 5 AM (off-peak)
            35, 55, 65, 60, 55, 50,   # 6 AM to 11 AM (morning ramp)
            48, 45, 43, 45, 50, 60,   # noon to 5 PM
            75, 80, 70, 55, 40, 30]   # 6 PM to 11 PM (evening peak + drop)
    return base[:hours]


def generate_price_scenarios(
    hours: int = 24,
    n_scenarios: int = 10,
    volatility: float = 0.20,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate price scenarios via log-normal perturbation of the base forecast.

    Args:
        hours:       number of hours in the planning horizon
        n_scenarios: how many distinct price paths to generate
        volatility:  std dev of log-returns (0.20 = 20% price uncertainty)
        seed:        random seed for reproducibility

    Returns:
        Array of shape (n_scenarios, hours) — each row is one price path ($/MWh)
    """
    rng = np.random.default_rng(seed)
    base = np.array(base_price_forecast(hours), dtype=float)

    # Each scenario is base * exp(normal noise), preserving positivity
    log_shocks = rng.normal(0, volatility, size=(n_scenarios, hours))
    scenarios = base[np.newaxis, :] * np.exp(log_shocks)

    return scenarios.round(2)


def scenario_probabilities(n_scenarios: int) -> List[float]:
    """Equal probability weights across scenarios."""
    return [1.0 / n_scenarios] * n_scenarios
