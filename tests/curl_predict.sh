#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG (override via env) --------
# Examples:
#   BASE_URL=http://127.0.0.1:8000 ./curl_predict.sh
#   BASE_URL=https://<your-space>.hf.space HF_TOKEN=hf_xxx ENDPOINT=/predict ./curl_predict.sh
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ENDPOINT="${ENDPOINT:-/predict}"        # e.g. /predict | /similarity | /embed | /rerank
HF_TOKEN="${HF_TOKEN:-}"                # only needed if Space is private
JSON_FILE="${1:-}"                      # optional: path to a JSON file to POST

# -------- PAYLOAD (inline fallback) --------
read -r -d '' DEFAULT_PAYLOAD <<'JSON'
{
  "query": "EGFR exon 19 deletion in NSCLC",
  "candidates": [
    "ALK rearrangement NSCLC treatment",
    "EGFR L858R mutation significance",
    "BRCA1 frameshift variant"
  ],
  "top_k": 2
}
JSON

if [[ -n "$JSON_FILE" ]]; then
  if [[ ! -f "$JSON_FILE" ]]; then
    echo "ERROR: JSON file not found: $JSON_FILE" >&2
    exit 1
  fi
  PAYLOAD="$(cat "$JSON_FILE")"
else
  PAYLOAD="$DEFAULT_PAYLOAD"
fi

# -------- Headers --------
AUTH_HEADER=()
if [[ -n "$HF_TOKEN" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
fi

# -------- Request --------
echo "POST ${BASE_URL}${ENDPOINT}"
curl -sS -X POST "${BASE_URL}${ENDPOINT}" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  --data "$PAYLOAD" | jq .

##How to use 
bash
chmod +x scripts/curl_predict.sh

# Local FastAPI/Uvicorn (default /predict)
./scripts/curl_predict.sh

# Point to ngrok tunnel
BASE_URL="https://<your-ngrok-id>.ngrok.io" ./scripts/curl_predict.sh

# Hugging Face Space (public)
BASE_URL="https://<your-space>.hf.space" ENDPOINT="/predict" ./scripts/curl_predict.sh

# Hugging Face Space (private)
BASE_URL="https://<your-space>.hf.space" ENDPOINT="/predict" HF_TOKEN="hf_xxx" ./scripts/curl_predict.sh

# Use a custom JSON body from a file
./scripts/curl_predict.sh payloads/example_request.json
