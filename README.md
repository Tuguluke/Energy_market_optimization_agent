# Energy Market Optimization Agent

An AI agent that recommends optimal generation mix (gas, solar, wind, battery) under electricity price uncertainty.

## Stack
- **Optimizer** — stochastic linear programming with PuLP (scenario-based dispatch)
- **Agent** — Claude (Anthropic) with tool use for conversational recommendations
- **Cloud** — AWS Lambda + API Gateway + DynamoDB + S3 via SAM

## Usage

### Local
```bash
python main.py                  # run optimizer directly
python agent_chat.py            # chat with the Claude agent
```

### AWS
```bash
make deploy                     # build + deploy to AWS
make invoke-optimize            # call /optimize endpoint
make invoke-recommend           # call /recommend endpoint (uses Claude)
make logs                       # tail CloudWatch logs
```

## API Endpoints
| Method | Path | Description |
|---|---|---|
| POST | `/optimize` | Run optimizer, returns dispatch schedule |
| POST | `/recommend` | Ask the Claude agent a question |
| GET | `/history` | Retrieve past recommendations |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY
```
