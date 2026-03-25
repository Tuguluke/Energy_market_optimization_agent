"""
AWS Lambda entry point.

Routes:
  POST /optimize   — run the optimizer directly (no Claude, free)
  POST /recommend  — run the full Claude agent (costs ~$0.006/call)
  GET  /history    — retrieve past recommendations from DynamoDB
"""
import json
import os
import boto3
from datetime import datetime, timezone

from src.agent.tools import execute_tool
from src.agent.agent import EnergyAgent

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "energy-recommendations")
BUCKET_NAME = os.environ.get("S3_BUCKET", "")


def lambda_handler(event, context):
    path = event.get("rawPath", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")

    try:
        if path == "/optimize" and method == "POST":
            return handle_optimize(event)
        elif path == "/recommend" and method == "POST":
            return handle_recommend(event)
        elif path == "/history" and method == "GET":
            return handle_history(event)
        else:
            return response(404, {"error": f"Unknown route: {method} {path}"})
    except Exception as e:
        return response(500, {"error": str(e)})


# ── /optimize — direct optimizer, no Claude ──────────────────────────────────

def handle_optimize(event):
    body = parse_body(event)

    params = {
        "hours": body.get("hours", 24),
        "n_scenarios": body.get("n_scenarios", 10),
        "volatility": body.get("volatility", 0.20),
        "gas_fuel_cost": body.get("gas_fuel_cost", 40.0),
    }

    result = json.loads(execute_tool("run_optimization", params))
    save_to_dynamodb("optimize", params, result)

    return response(200, result)


# ── /recommend — full Claude agent ───────────────────────────────────────────

def handle_recommend(event):
    body = parse_body(event)
    question = body.get("question", "What is the optimal dispatch for today?")

    agent = EnergyAgent()
    answer = agent.chat(question)

    record = {"question": question, "answer": answer}
    save_to_dynamodb("recommend", {"question": question}, record)

    return response(200, {"answer": answer})


# ── /history — read past recommendations from DynamoDB ───────────────────────

def handle_history(event):
    params = event.get("queryStringParameters") or {}
    limit = int(params.get("limit", 10))

    table = dynamodb.Table(TABLE_NAME)
    result = table.scan(Limit=limit)

    items = sorted(result.get("Items", []), key=lambda x: x["timestamp"], reverse=True)
    return response(200, {"items": items, "count": len(items)})


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_to_dynamodb(route: str, params: dict, result: dict):
    try:
        table = dynamodb.Table(TABLE_NAME)
        table.put_item(Item={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": route,
            "params": params,
            "result_summary": {
                k: result[k]
                for k in ("status", "expected_profit_usd", "profit_range_usd")
                if k in result
            },
        })
    except Exception:
        pass  # don't fail the request if DynamoDB write fails


def parse_body(event) -> dict:
    body = event.get("body", "{}")
    if isinstance(body, str):
        return json.loads(body or "{}")
    return body or {}


def response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }
