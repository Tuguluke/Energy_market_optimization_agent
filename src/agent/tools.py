"""
Tool definitions and implementations for the energy agent.

Each tool is:
  1. A JSON schema — tells Claude what parameters it takes
  2. A Python function — executes when Claude calls it
"""
import json
import numpy as np
from typing import Any

from src.models.assets import default_portfolio, GasTurbine
from src.data.profiles import (
    generate_price_scenarios,
    base_price_forecast,
    scenario_probabilities,
)
from src.optimizer.stochastic import optimize_dispatch


# ── Tool schemas (what Claude sees) ─────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "run_optimization",
        "description": (
            "Run the stochastic dispatch optimizer for the energy portfolio. "
            "Returns the optimal hourly generation mix for gas, solar, wind, and battery "
            "that maximises expected profit across price scenarios. "
            "Use this to get a concrete dispatch recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Planning horizon in hours (default 24).",
                    "default": 24,
                },
                "n_scenarios": {
                    "type": "integer",
                    "description": "Number of price scenarios to consider (default 10, max 50).",
                    "default": 10,
                },
                "volatility": {
                    "type": "number",
                    "description": (
                        "Price volatility as a fraction (e.g. 0.20 = 20%). "
                        "Higher values mean more uncertain prices."
                    ),
                    "default": 0.20,
                },
                "gas_fuel_cost": {
                    "type": "number",
                    "description": "Override gas fuel cost in $/MWh (default 40).",
                    "default": 40.0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_price_forecast",
        "description": (
            "Return the base electricity price forecast for a 24-hour horizon, "
            "along with statistics across generated scenarios (min, mean, max per hour). "
            "Useful for understanding the price environment before optimizing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_scenarios": {
                    "type": "integer",
                    "description": "Number of scenarios to sample for statistics.",
                    "default": 20,
                },
                "volatility": {
                    "type": "number",
                    "description": "Price volatility fraction.",
                    "default": 0.20,
                },
            },
            "required": [],
        },
    },
    {
        "name": "compare_scenarios",
        "description": (
            "Run the optimizer under two different conditions and compare the results. "
            "Useful for what-if analysis, e.g. 'what if gas prices double?' or "
            "'what if prices are much more volatile?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "base_volatility": {
                    "type": "number",
                    "description": "Volatility for the base case.",
                    "default": 0.20,
                },
                "alt_volatility": {
                    "type": "number",
                    "description": "Volatility for the alternative case.",
                    "default": 0.40,
                },
                "base_gas_cost": {
                    "type": "number",
                    "description": "Gas fuel cost for base case ($/MWh).",
                    "default": 40.0,
                },
                "alt_gas_cost": {
                    "type": "number",
                    "description": "Gas fuel cost for alternative case ($/MWh).",
                    "default": 40.0,
                },
                "hours": {
                    "type": "integer",
                    "description": "Planning horizon in hours.",
                    "default": 24,
                },
            },
            "required": [],
        },
    },
]


# ── Tool implementations (what actually runs) ────────────────────────────────

def run_optimization(
    hours: int = 24,
    n_scenarios: int = 10,
    volatility: float = 0.20,
    gas_fuel_cost: float = 40.0,
) -> dict[str, Any]:
    portfolio = default_portfolio()

    # Allow overriding gas fuel cost
    portfolio["gas"].fuel_cost_per_mwh = gas_fuel_cost

    scenarios = generate_price_scenarios(
        hours=hours,
        n_scenarios=min(n_scenarios, 50),
        volatility=volatility,
    )
    probs = scenario_probabilities(n_scenarios)

    result = optimize_dispatch(
        gas=portfolio["gas"],
        solar=portfolio["solar"],
        wind=portfolio["wind"],
        battery=portfolio["battery"],
        price_scenarios=scenarios,
        probabilities=probs,
        hours=hours,
    )

    # Summarise dispatch into peak / off-peak blocks
    peak_hours = list(range(7, 10)) + list(range(18, 21))  # morning + evening peak
    offpeak_hours = [h for h in range(hours) if h not in peak_hours]

    def avg(vals, idxs):
        return round(sum(vals[i] for i in idxs if i < len(vals)) / len(idxs), 1)

    return {
        "status": result.status,
        "expected_profit_usd": result.expected_profit,
        "profit_range_usd": {
            "min": round(min(result.scenario_profits), 0),
            "max": round(max(result.scenario_profits), 0),
        },
        "peak_dispatch_mw": {
            "gas": avg(result.gas_dispatch, peak_hours),
            "solar": avg(result.solar_dispatch, peak_hours),
            "wind": avg(result.wind_dispatch, peak_hours),
            "battery": avg(result.battery_dispatch, peak_hours),
        },
        "offpeak_dispatch_mw": {
            "gas": avg(result.gas_dispatch, offpeak_hours),
            "solar": avg(result.solar_dispatch, offpeak_hours),
            "wind": avg(result.wind_dispatch, offpeak_hours),
            "battery": avg(result.battery_dispatch, offpeak_hours),
        },
        "daily_totals_mwh": {
            "gas": round(sum(result.gas_dispatch), 1),
            "solar": round(sum(result.solar_dispatch), 1),
            "wind": round(sum(result.wind_dispatch), 1),
            "battery_net": round(sum(result.battery_dispatch), 1),
        },
        "battery_soc_start": result.battery_soc[0],
        "battery_soc_end": result.battery_soc[-1],
        "hourly_detail": {
            "gas_mw": result.gas_dispatch,
            "solar_mw": result.solar_dispatch,
            "wind_mw": result.wind_dispatch,
            "battery_mw": result.battery_dispatch,
            "battery_soc": result.battery_soc,
        },
        "parameters_used": {
            "hours": hours,
            "n_scenarios": n_scenarios,
            "volatility": volatility,
            "gas_fuel_cost_per_mwh": gas_fuel_cost,
        },
    }


def get_price_forecast(n_scenarios: int = 20, volatility: float = 0.20) -> dict[str, Any]:
    base = base_price_forecast(24)
    scenarios = generate_price_scenarios(
        hours=24,
        n_scenarios=n_scenarios,
        volatility=volatility,
    )

    hourly_stats = []
    for t in range(24):
        prices_at_t = scenarios[:, t]
        hourly_stats.append({
            "hour": t,
            "base_price": base[t],
            "scenario_min": round(float(prices_at_t.min()), 2),
            "scenario_mean": round(float(prices_at_t.mean()), 2),
            "scenario_max": round(float(prices_at_t.max()), 2),
        })

    return {
        "daily_base_avg": round(sum(base) / len(base), 2),
        "daily_peak_avg": round(sum(base[h] for h in range(7, 21)) / 14, 2),
        "price_uncertainty_pct": round(volatility * 100, 1),
        "hourly_stats": hourly_stats,
    }


def compare_scenarios(
    base_volatility: float = 0.20,
    alt_volatility: float = 0.40,
    base_gas_cost: float = 40.0,
    alt_gas_cost: float = 40.0,
    hours: int = 24,
) -> dict[str, Any]:
    base_result = run_optimization(
        hours=hours, n_scenarios=15,
        volatility=base_volatility, gas_fuel_cost=base_gas_cost,
    )
    alt_result = run_optimization(
        hours=hours, n_scenarios=15,
        volatility=alt_volatility, gas_fuel_cost=alt_gas_cost,
    )

    profit_change = alt_result["expected_profit_usd"] - base_result["expected_profit_usd"]
    profit_change_pct = round(profit_change / base_result["expected_profit_usd"] * 100, 1)

    return {
        "base_case": {
            "volatility": base_volatility,
            "gas_fuel_cost": base_gas_cost,
            "expected_profit": base_result["expected_profit_usd"],
            "gas_daily_mwh": base_result["daily_totals_mwh"]["gas"],
        },
        "alternative_case": {
            "volatility": alt_volatility,
            "gas_fuel_cost": alt_gas_cost,
            "expected_profit": alt_result["expected_profit_usd"],
            "gas_daily_mwh": alt_result["daily_totals_mwh"]["gas"],
        },
        "profit_change_usd": round(profit_change, 2),
        "profit_change_pct": profit_change_pct,
        "interpretation": (
            f"Expected profit {'increases' if profit_change > 0 else 'decreases'} by "
            f"${abs(profit_change):,.0f} ({abs(profit_change_pct)}%) "
            f"under the alternative scenario."
        ),
    }


# ── Dispatcher: routes tool calls from Claude ────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool by name and return JSON string result."""
    try:
        if name == "run_optimization":
            result = run_optimization(**inputs)
        elif name == "get_price_forecast":
            result = get_price_forecast(**inputs)
        elif name == "compare_scenarios":
            result = compare_scenarios(**inputs)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, indent=2)
