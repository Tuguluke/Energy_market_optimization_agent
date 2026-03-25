"""
Generation asset definitions.
Each asset has physical and cost parameters that constrain the optimizer.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GasTurbine:
    name: str
    capacity_mw: float          # max output
    min_output_mw: float        # minimum stable generation when online
    fuel_cost_per_mwh: float    # variable fuel cost ($/MWh)
    startup_cost: float         # one-time cost to turn on ($)
    ramp_rate_mw_per_hour: float  # max change in output per hour

    @property
    def asset_type(self) -> str:
        return "gas"


@dataclass
class SolarFarm:
    name: str
    capacity_mw: float
    # Solar output is determined by a capacity factor profile (0.0 to 1.0)
    # e.g., [0, 0, 0, 0.1, 0.4, 0.8, 1.0, ...] over 24 hours

    @property
    def asset_type(self) -> str:
        return "solar"


@dataclass
class WindFarm:
    name: str
    capacity_mw: float
    # Wind output also determined by capacity factor profile

    @property
    def asset_type(self) -> str:
        return "wind"


@dataclass
class Battery:
    name: str
    power_mw: float             # max charge/discharge rate
    energy_mwh: float           # total storage capacity
    efficiency: float           # round-trip efficiency (e.g., 0.90)
    initial_soc: float = 0.5    # starting state of charge (fraction 0-1)
    min_soc: float = 0.1        # minimum allowed state of charge
    max_soc: float = 0.95       # maximum allowed state of charge

    @property
    def asset_type(self) -> str:
        return "battery"


def default_portfolio():
    """A representative small grid portfolio for experimentation."""
    return {
        "gas": GasTurbine(
            name="Gas Peaker",
            capacity_mw=200.0,
            min_output_mw=50.0,
            fuel_cost_per_mwh=40.0,
            startup_cost=500.0,
            ramp_rate_mw_per_hour=80.0,
        ),
        "solar": SolarFarm(
            name="Solar Farm",
            capacity_mw=150.0,
        ),
        "wind": WindFarm(
            name="Wind Farm",
            capacity_mw=100.0,
        ),
        "battery": Battery(
            name="Grid Battery",
            power_mw=50.0,
            energy_mwh=200.0,
            efficiency=0.90,
            initial_soc=0.5,
        ),
    }
