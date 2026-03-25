"""
Phase 1 entry point — run the stochastic dispatch optimizer locally.

Usage:
    python main.py
    python main.py --scenarios 20 --volatility 0.3 --hours 24
"""
import argparse
import numpy as np

from src.models.assets import default_portfolio
from src.data.profiles import generate_price_scenarios, scenario_probabilities
from src.optimizer.stochastic import optimize_dispatch
from src.optimizer.reporter import print_report


def main():
    parser = argparse.ArgumentParser(description="Energy dispatch optimizer")
    parser.add_argument("--scenarios", type=int, default=10, help="Number of price scenarios")
    parser.add_argument("--volatility", type=float, default=0.20, help="Price volatility (0–1)")
    parser.add_argument("--hours", type=int, default=24, help="Planning horizon in hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"\nRunning optimizer:")
    print(f"  Horizon:    {args.hours} hours")
    print(f"  Scenarios:  {args.scenarios}")
    print(f"  Volatility: {args.volatility:.0%}")

    # Load default asset portfolio
    portfolio = default_portfolio()

    # Generate price scenarios
    price_scenarios = generate_price_scenarios(
        hours=args.hours,
        n_scenarios=args.scenarios,
        volatility=args.volatility,
        seed=args.seed,
    )
    probabilities = scenario_probabilities(args.scenarios)

    print(f"\n  Base price range: ${price_scenarios.mean(axis=0).min():.1f} – "
          f"${price_scenarios.mean(axis=0).max():.1f} /MWh")
    print(f"  Scenario spread:  ±{price_scenarios.std():.1f} $/MWh")

    # Solve
    result = optimize_dispatch(
        gas=portfolio["gas"],
        solar=portfolio["solar"],
        wind=portfolio["wind"],
        battery=portfolio["battery"],
        price_scenarios=price_scenarios,
        probabilities=probabilities,
        hours=args.hours,
    )

    # Print results
    print_report(result, price_scenarios)


if __name__ == "__main__":
    main()
