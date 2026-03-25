"""
Interactive CLI for the energy optimization agent.

Usage:
    python agent_chat.py

Set your API key in .env:
    ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

# Check for API key before importing agent (gives a clear error message)
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("\nError: ANTHROPIC_API_KEY environment variable is not set.")
    print("Export it before running:")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

from src.agent.agent import EnergyAgent

WELCOME = """
╔══════════════════════════════════════════════════════╗
║       Energy Market Optimization Agent               ║
║                                                      ║
║  Ask me about your generation portfolio:             ║
║  • "What's the optimal dispatch for today?"          ║
║  • "What if gas prices go up to $80/MWh?"            ║
║  • "How does high price volatility affect profits?"  ║
║  • "Compare low vs high volatility scenarios"        ║
║                                                      ║
║  Type 'reset' to start a new conversation            ║
║  Type 'exit' or Ctrl+C to quit                       ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    print(WELCOME)
    agent = EnergyAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation reset.")
            continue

        print("\nAgent: ", end="", flush=True)
        try:
            response = agent.chat(user_input)
            print(response)
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
