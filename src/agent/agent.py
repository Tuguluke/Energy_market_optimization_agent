"""
Energy optimization agent.

Uses Claude (via Anthropic SDK) as the reasoning layer.
Claude decides which tools to call, interprets results, and
explains recommendations in plain language.
"""
import os
from pathlib import Path
import anthropic

def _get_api_key() -> str:
    # 1. environment variable (already set)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # 2. .env file — search from this file's location upward
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except ImportError:
        pass
    # 3. Streamlit secrets (when running on Streamlit Cloud)
    try:
        import streamlit as st
        return st.secrets["ANTHROPIC_API_KEY"]
    except KeyError:
        raise ValueError("ANTHROPIC_API_KEY not found in Streamlit secrets — check the key name is exactly 'ANTHROPIC_API_KEY'")
    except Exception as e:
        raise ValueError(f"ANTHROPIC_API_KEY not found. Streamlit secrets error: {e}")

from src.agent.tools import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """You are an expert energy market optimization agent.

You help energy portfolio managers make dispatch decisions under price uncertainty.
You have access to tools that run stochastic optimization over a portfolio of:
  - Gas turbine (200 MW capacity, $40/MWh fuel cost, can ramp 80 MW/hour)
  - Solar farm (150 MW capacity, follows daylight profile)
  - Wind farm (100 MW capacity, variable output)
  - Grid battery (50 MW / 200 MWh, 90% round-trip efficiency)

When answering questions:
1. Use tools to get concrete numbers — don't guess dispatch values
2. Explain the economic intuition behind recommendations
3. Highlight key risk factors (e.g. high price volatility, gas cost sensitivity)
4. Be concise but precise — this is an operational context

When presenting results, always mention:
- Expected profit and profit range (risk)
- The generation mix rationale (why gas/solar/wind/battery levels make sense)
- Any trade-offs the operator should be aware of
"""

MODEL = "claude-opus-4-6"


class EnergyAgent:
    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or _get_api_key()
        )
        self.conversation_history = []

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, running the tool-use loop."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                thinking={"type": "adaptive"},
                messages=self.conversation_history,
            )

            # Collect all tool calls Claude wants to make
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn" or not tool_calls:
                # Claude is done — extract the text response
                text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })
                return text

            # Claude wants to call tools — execute them all
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            tool_results = []
            for tool_call in tool_calls:
                print(f"  [tool] {tool_call.name}({_fmt_inputs(tool_call.input)})")
                result = execute_tool(tool_call.name, tool_call.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                })

            self.conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []


def _fmt_inputs(inputs: dict) -> str:
    """Format tool inputs for display."""
    parts = [f"{k}={v}" for k, v in inputs.items()]
    return ", ".join(parts) if parts else "defaults"
