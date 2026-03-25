"""
Stochastic dispatch optimizer.

Problem: choose hourly dispatch for each asset to maximize EXPECTED profit
         across price scenarios, subject to physical constraints.

Formulation (linear program per scenario, averaged):
  maximize  sum_s [ prob_s * sum_t ( price_s_t * total_dispatch_t
                                     - gas_cost_t - startup_cost_t ) ]
  subject to:
    - Gas output within [min, max] when online, 0 when offline
    - Gas ramp rate limits between hours
    - Solar/wind output <= capacity_factor * capacity
    - Battery power within [-power_mw, +power_mw]
    - Battery state of charge within [min_soc, max_soc]
    - Battery energy balance each hour

We solve each scenario independently (wait-and-see relaxation) then
also solve the here-and-now (expected value) problem for the agent.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import pulp
except ImportError:
    raise ImportError("Install PuLP: pip install pulp")

from src.models.assets import GasTurbine, SolarFarm, WindFarm, Battery
from src.data.profiles import (
    solar_capacity_factors,
    wind_capacity_factors,
    scenario_probabilities,
)


@dataclass
class DispatchResult:
    status: str                          # "Optimal" or "Infeasible"
    expected_profit: float               # $ across scenarios
    gas_dispatch: List[float]            # MW per hour
    solar_dispatch: List[float]
    wind_dispatch: List[float]
    battery_dispatch: List[float]        # positive = discharging, negative = charging
    battery_soc: List[float]             # state of charge fraction per hour
    scenario_profits: List[float]        # profit per scenario
    hours: int


def optimize_dispatch(
    gas: GasTurbine,
    solar: SolarFarm,
    wind: WindFarm,
    battery: Battery,
    price_scenarios: np.ndarray,         # shape (n_scenarios, hours)
    probabilities: Optional[List[float]] = None,
    hours: int = 24,
) -> DispatchResult:
    """
    Solve the stochastic dispatch problem.

    Strategy: expected value (EV) formulation — one set of dispatch decisions
    that maximises expected profit across all price scenarios.
    """
    n_scenarios, n_hours = price_scenarios.shape
    assert n_hours == hours, "price_scenarios columns must equal hours"

    if probabilities is None:
        probabilities = scenario_probabilities(n_scenarios)

    solar_cf = solar_capacity_factors(hours)
    wind_cf = wind_capacity_factors(hours)

    prob = pulp.LpProblem("energy_dispatch", pulp.LpMaximize)

    # ── Decision variables ──────────────────────────────────────────────────

    # Gas: continuous output + binary on/off per hour
    gas_mw = [pulp.LpVariable(f"gas_mw_{t}", lowBound=0, upBound=gas.capacity_mw)
              for t in range(hours)]
    gas_on = [pulp.LpVariable(f"gas_on_{t}", cat="Binary")
              for t in range(hours)]
    gas_start = [pulp.LpVariable(f"gas_start_{t}", cat="Binary")
                 for t in range(hours)]

    # Solar and wind: curtailable (can produce less than available)
    solar_mw = [pulp.LpVariable(f"solar_mw_{t}", lowBound=0,
                                upBound=solar.capacity_mw * solar_cf[t])
                for t in range(hours)]
    wind_mw = [pulp.LpVariable(f"wind_mw_{t}", lowBound=0,
                               upBound=wind.capacity_mw * wind_cf[t])
               for t in range(hours)]

    # Battery: positive = discharge (sell), negative = charge (buy)
    bat_mw = [pulp.LpVariable(f"bat_mw_{t}",
                              lowBound=-battery.power_mw,
                              upBound=battery.power_mw)
              for t in range(hours)]
    bat_soc = [pulp.LpVariable(f"bat_soc_{t}",
                               lowBound=battery.min_soc,
                               upBound=battery.max_soc)
               for t in range(hours)]

    # ── Objective: maximise expected revenue minus costs ─────────────────────

    revenue_terms = []
    for s, (scenario, prob_s) in enumerate(zip(price_scenarios, probabilities)):
        for t in range(hours):
            total_mw = gas_mw[t] + solar_mw[t] + wind_mw[t] + bat_mw[t]
            revenue_terms.append(prob_s * scenario[t] * total_mw)

    fuel_costs = [gas.fuel_cost_per_mwh * gas_mw[t] for t in range(hours)]
    startup_costs = [gas.startup_cost * gas_start[t] for t in range(hours)]

    prob += pulp.lpSum(revenue_terms) - pulp.lpSum(fuel_costs) - pulp.lpSum(startup_costs)

    # ── Constraints ──────────────────────────────────────────────────────────

    for t in range(hours):
        # Gas must stay within [min, max] when on, or be zero when off
        prob += gas_mw[t] >= gas.min_output_mw * gas_on[t]
        prob += gas_mw[t] <= gas.capacity_mw * gas_on[t]

        # Startup flag: gas_start[t] = 1 if gas turns on at hour t
        if t == 0:
            prob += gas_start[t] >= gas_on[t]
        else:
            prob += gas_start[t] >= gas_on[t] - gas_on[t - 1]
            # Ramp rate constraint
            prob += gas_mw[t] - gas_mw[t - 1] <= gas.ramp_rate_mw_per_hour
            prob += gas_mw[t - 1] - gas_mw[t] <= gas.ramp_rate_mw_per_hour

        # Battery state of charge balance
        if t == 0:
            # Simplified linear battery model (efficiency applied uniformly)
            prob += bat_soc[t] == battery.initial_soc - bat_mw[t] / battery.energy_mwh
        else:
            prob += bat_soc[t] == bat_soc[t - 1] - bat_mw[t] / battery.energy_mwh

    # ── Solve ────────────────────────────────────────────────────────────────

    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return DispatchResult(
            status=status,
            expected_profit=0,
            gas_dispatch=[0] * hours,
            solar_dispatch=[0] * hours,
            wind_dispatch=[0] * hours,
            battery_dispatch=[0] * hours,
            battery_soc=[battery.initial_soc] * hours,
            scenario_profits=[],
            hours=hours,
        )

    gas_vals = [pulp.value(gas_mw[t]) or 0.0 for t in range(hours)]
    solar_vals = [pulp.value(solar_mw[t]) or 0.0 for t in range(hours)]
    wind_vals = [pulp.value(wind_mw[t]) or 0.0 for t in range(hours)]
    bat_vals = [pulp.value(bat_mw[t]) or 0.0 for t in range(hours)]
    soc_vals = [pulp.value(bat_soc[t]) or 0.0 for t in range(hours)]

    # Calculate per-scenario profits using the solved dispatch
    scenario_profits = []
    for s, (scenario, prob_s) in enumerate(zip(price_scenarios, probabilities)):
        profit = sum(
            scenario[t] * (gas_vals[t] + solar_vals[t] + wind_vals[t] + bat_vals[t])
            - gas.fuel_cost_per_mwh * gas_vals[t]
            for t in range(hours)
        )
        scenario_profits.append(round(profit, 2))

    expected_profit = sum(p * sp for p, sp in zip(probabilities, scenario_profits))

    return DispatchResult(
        status=status,
        expected_profit=round(expected_profit, 2),
        gas_dispatch=[round(v, 2) for v in gas_vals],
        solar_dispatch=[round(v, 2) for v in solar_vals],
        wind_dispatch=[round(v, 2) for v in wind_vals],
        battery_dispatch=[round(v, 2) for v in bat_vals],
        battery_soc=[round(v, 3) for v in soc_vals],
        scenario_profits=scenario_profits,
        hours=hours,
    )
