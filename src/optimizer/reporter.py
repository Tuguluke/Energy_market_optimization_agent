"""
Human-readable report of dispatch results.
"""
from src.optimizer.stochastic import DispatchResult


def print_report(result: DispatchResult, price_scenarios=None):
    print("\n" + "=" * 60)
    print("  OPTIMAL DISPATCH RECOMMENDATION")
    print("=" * 60)
    print(f"  Status:           {result.status}")
    print(f"  Expected Profit:  ${result.expected_profit:,.2f}")

    if result.scenario_profits:
        profits = result.scenario_profits
        print(f"  Profit Range:     ${min(profits):,.0f} – ${max(profits):,.0f}")
        print(f"  Scenarios:        {len(profits)}")

    print("\n  Hourly Dispatch (MW):")
    print(f"  {'Hour':>4}  {'Gas':>8}  {'Solar':>8}  {'Wind':>8}  {'Battery':>9}  {'SOC':>6}")
    print("  " + "-" * 54)

    for t in range(result.hours):
        bat = result.battery_dispatch[t]
        bat_str = f"{bat:+.1f}"  # show sign for charge/discharge
        print(
            f"  {t:>4}  "
            f"{result.gas_dispatch[t]:>8.1f}  "
            f"{result.solar_dispatch[t]:>8.1f}  "
            f"{result.wind_dispatch[t]:>8.1f}  "
            f"{bat_str:>9}  "
            f"{result.battery_soc[t]:>6.1%}"
        )

    print("\n  Daily Totals (MWh):")
    print(f"    Gas:     {sum(result.gas_dispatch):>8.1f} MWh")
    print(f"    Solar:   {sum(result.solar_dispatch):>8.1f} MWh")
    print(f"    Wind:    {sum(result.wind_dispatch):>8.1f} MWh")
    print(f"    Battery: {sum(result.battery_dispatch):>8.1f} MWh (net)")
    print("=" * 60)
