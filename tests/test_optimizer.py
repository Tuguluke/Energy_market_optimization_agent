"""Basic sanity tests for the optimizer."""
import numpy as np
import pytest

from src.models.assets import default_portfolio
from src.data.profiles import (
    solar_capacity_factors,
    wind_capacity_factors,
    generate_price_scenarios,
    scenario_probabilities,
)
from src.optimizer.stochastic import optimize_dispatch


@pytest.fixture
def portfolio():
    return default_portfolio()


@pytest.fixture
def simple_scenarios():
    return generate_price_scenarios(hours=24, n_scenarios=3, volatility=0.1, seed=0)


def test_solar_profile_zero_at_night():
    cf = solar_capacity_factors(24)
    assert cf[0] == 0.0   # midnight
    assert cf[3] == 0.0   # 3 AM
    assert cf[13] > 0.5   # peak near noon


def test_wind_profile_bounded():
    cf = wind_capacity_factors(24)
    assert all(0.0 <= v <= 1.0 for v in cf)


def test_price_scenarios_shape():
    scenarios = generate_price_scenarios(hours=24, n_scenarios=10)
    assert scenarios.shape == (10, 24)
    assert (scenarios > 0).all()


def test_optimizer_returns_optimal(portfolio, simple_scenarios):
    result = optimize_dispatch(
        gas=portfolio["gas"],
        solar=portfolio["solar"],
        wind=portfolio["wind"],
        battery=portfolio["battery"],
        price_scenarios=simple_scenarios,
        hours=24,
    )
    assert result.status == "Optimal"
    assert result.expected_profit > 0


def test_gas_dispatch_within_capacity(portfolio, simple_scenarios):
    result = optimize_dispatch(
        gas=portfolio["gas"],
        solar=portfolio["solar"],
        wind=portfolio["wind"],
        battery=portfolio["battery"],
        price_scenarios=simple_scenarios,
        hours=24,
    )
    for mw in result.gas_dispatch:
        assert mw <= portfolio["gas"].capacity_mw + 1e-5


def test_battery_soc_within_bounds(portfolio, simple_scenarios):
    result = optimize_dispatch(
        gas=portfolio["gas"],
        solar=portfolio["solar"],
        wind=portfolio["wind"],
        battery=portfolio["battery"],
        price_scenarios=simple_scenarios,
        hours=24,
    )
    bat = portfolio["battery"]
    for soc in result.battery_soc:
        assert bat.min_soc - 1e-5 <= soc <= bat.max_soc + 1e-5
