"""
Streamlit app for the Energy Market Optimization Agent.

Run locally:  streamlit run app.py
Deploy:       push to GitHub → connect on share.streamlit.io
"""
import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

from src.data.profiles import generate_price_scenarios, scenario_probabilities
from src.models.assets import default_portfolio
from src.optimizer.stochastic import optimize_dispatch

st.set_page_config(
    page_title="Energy Optimization Agent",
    page_icon="⚡",
    layout="wide",
)

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Parameters")
    st.markdown("---")

    volatility = st.slider(
        "Price Volatility",
        min_value=0.05, max_value=0.60, value=0.20, step=0.05,
        format="%.0f%%",
        help="How uncertain electricity prices are (0% = certain, 60% = very uncertain)"
    )

    gas_cost = st.slider(
        "Gas Fuel Cost ($/MWh)",
        min_value=20, max_value=120, value=40, step=5,
        help="Variable fuel cost for the gas turbine"
    )

    n_scenarios = st.select_slider(
        "Price Scenarios",
        options=[5, 10, 20, 30],
        value=10,
        help="More scenarios = more accurate but slower"
    )

    st.markdown("---")
    run_button = st.button("Run Optimization", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Portfolio: 200MW gas · 150MW solar · 100MW wind · 50MW/200MWh battery")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Energy Market Optimization Agent")
st.caption("Optimal generation dispatch under price uncertainty")

tab1, tab2 = st.tabs(["Optimizer", "Ask the Agent"])

# ── Tab 1: Optimizer ──────────────────────────────────────────────────────────

with tab1:
    if run_button or "result" not in st.session_state:
        with st.spinner("Running stochastic optimization..."):
            portfolio = default_portfolio()
            portfolio["gas"].fuel_cost_per_mwh = gas_cost

            scenarios = generate_price_scenarios(
                hours=24, n_scenarios=n_scenarios, volatility=volatility
            )
            probs = scenario_probabilities(n_scenarios)

            result = optimize_dispatch(
                gas=portfolio["gas"],
                solar=portfolio["solar"],
                wind=portfolio["wind"],
                battery=portfolio["battery"],
                price_scenarios=scenarios,
                probabilities=probs,
                hours=24,
            )
            st.session_state.result = result
            st.session_state.scenarios = scenarios

    result = st.session_state.result
    scenarios = st.session_state.scenarios

    # ── Metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Profit", f"${result.expected_profit:,.0f}")
    col2.metric("Worst Case", f"${min(result.scenario_profits):,.0f}")
    col3.metric("Best Case", f"${max(result.scenario_profits):,.0f}")
    col4.metric(
        "Profit Risk",
        f"${max(result.scenario_profits) - min(result.scenario_profits):,.0f}",
        help="Spread between best and worst scenario"
    )

    st.markdown("---")

    # ── Dispatch chart ──
    st.subheader("Hourly Dispatch Schedule")

    df = pd.DataFrame({
        "Hour": list(range(24)),
        "Gas (MW)": result.gas_dispatch,
        "Solar (MW)": result.solar_dispatch,
        "Wind (MW)": result.wind_dispatch,
        "Battery (MW)": result.battery_dispatch,
    }).set_index("Hour")

    st.bar_chart(df[["Gas (MW)", "Solar (MW)", "Wind (MW)"]])

    # Battery separately (can be negative)
    st.caption("Battery dispatch (positive = discharging, negative = charging)")
    st.line_chart(pd.DataFrame({
        "Hour": list(range(24)),
        "Battery (MW)": result.battery_dispatch,
        "State of Charge (%)": [s * 100 for s in result.battery_soc],
    }).set_index("Hour"))

    st.markdown("---")

    # ── Price scenarios ──
    st.subheader("Price Scenarios")
    price_df = pd.DataFrame(
        scenarios.T,
        index=range(24),
        columns=[f"Scenario {i+1}" for i in range(scenarios.shape[0])]
    )
    st.line_chart(price_df)

    # ── Daily totals ──
    st.markdown("---")
    st.subheader("Daily Totals")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gas", f"{sum(result.gas_dispatch):,.0f} MWh")
    col2.metric("Solar", f"{sum(result.solar_dispatch):,.0f} MWh")
    col3.metric("Wind", f"{sum(result.wind_dispatch):,.0f} MWh")
    col4.metric("Battery (net)", f"{sum(result.battery_dispatch):+,.0f} MWh")

# ── Tab 2: Agent chat ─────────────────────────────────────────────────────────

with tab2:
    st.subheader("Ask the Agent")
    st.caption("Claude will run the optimizer and explain the results in plain language.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        suggestions = [
            "What's the optimal dispatch for today?",
            "What if gas prices spike to $80/MWh?",
            "How does high volatility affect profits?",
            "Compare low vs high price uncertainty",
        ]
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, key=f"sug_{i}"):
                st.session_state.pending_question = suggestion
                st.rerun()

    # Handle suggested question click
    if "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")
    else:
        prompt = st.chat_input("Ask about your energy portfolio...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.agent.agent import EnergyAgent
                    agent = EnergyAgent()
                    # Replay conversation history into agent
                    for msg in st.session_state.messages[:-1]:
                        agent.conversation_history.append(msg)
                    response = agent.chat(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    err = f"Agent error: {e}\n\nMake sure ANTHROPIC_API_KEY is set in your .env or Streamlit secrets."
                    st.error(err)

    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
