# Prepares lambda_pkg/ — the directory SAM packages and deploys.
# SAM needs all source code + requirements.txt in one folder.

.PHONY: build deploy invoke-optimize invoke-recommend logs clean

# Step 1: assemble lambda_pkg/ from project source
build-pkg:
	rm -rf lambda_pkg
	mkdir -p lambda_pkg
	# Copy application source
	cp -r src lambda_pkg/src
	cp infrastructure/lambda/handler.py lambda_pkg/handler.py
	cp infrastructure/lambda/requirements.txt lambda_pkg/requirements.txt

# Step 2: run sam build (compiles deps for Lambda Linux/arm64 inside Docker)
build: build-pkg
	cd infrastructure && sam build

# Step 3: deploy to AWS (reads ANTHROPIC_API_KEY from .env)
deploy: build
	cd infrastructure && sam deploy \
	  --parameter-overrides "AnthropicApiKey=$(shell grep ANTHROPIC_API_KEY .env | cut -d= -f2-)"

# ── Invoke the deployed Lambda directly via CLI ───────────────────────────────

invoke-optimize:
	aws lambda invoke \
	  --function-name energy-optimization-agent \
	  --payload '{"rawPath":"/optimize","requestContext":{"http":{"method":"POST"}},"body":"{\"volatility\":0.25}"}' \
	  --cli-binary-format raw-in-base64-out \
	  /tmp/response.json && cat /tmp/response.json

invoke-recommend:
	aws lambda invoke \
	  --function-name energy-optimization-agent \
	  --payload '{"rawPath":"/recommend","requestContext":{"http":{"method":"POST"}},"body":"{\"question\":\"What is the optimal dispatch for today?\"}"}' \
	  --cli-binary-format raw-in-base64-out \
	  /tmp/response.json && cat /tmp/response.json

# ── Tail CloudWatch logs ──────────────────────────────────────────────────────

logs:
	aws logs tail /aws/lambda/energy-optimization-agent --follow

# ── Teardown ──────────────────────────────────────────────────────────────────

destroy:
	cd infrastructure && sam delete

clean:
	rm -rf lambda_pkg infrastructure/.aws-sam
