#!/usr/bin/env bash
set -euo pipefail

# -------- Defaults (can be overridden via env) --------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
APP_MODULE="${APP_MODULE:-app_run:app}"          # e.g., app_run:app  or  app.app:app
MODEL_DIR="${MODEL_DIR:-/models/scimilarity_model_v1_1}"
HEAD_PATH="${HEAD_PATH:-/models/lora/linear_head.pt}"
CLASSES_PATH="${CLASSES_PATH:-/models/lora/label_classes.txt}"

# Honor proxies (HF/Cloud) and forwarded IPs
UVICORN_FLAGS="${UVICORN_FLAGS:---proxy-headers --forwarded-allow-ips='*'}"

# Auto workers (change to 1 if you rely on CUDA and single process)
WORKERS="${WORKERS:-1}"

echo "[start] ================== SCimilarity API =================="
echo "[start] APP_MODULE   = ${APP_MODULE}"
echo "[start] HOST:PORT    = ${HOST}:${PORT}"
echo "[start] MODEL_DIR    = ${MODEL_DIR}"
echo "[start] HEAD_PATH    = ${HEAD_PATH}"
echo "[start] CLASSES_PATH = ${CLASSES_PATH}"
echo "[start] LOG_LEVEL    = ${LOG_LEVEL}"
echo "[start] WORKERS      = ${WORKERS}"
echo "[start] ====================================================="

# -------- Sanity checks --------
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[start][FATAL] MODEL_DIR not found: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/gene_order.tsv" ]]; then
  echo "[start][FATAL] gene_order.tsv missing in MODEL_DIR: ${MODEL_DIR}/gene_order.tsv" >&2
  exit 1
fi

if [[ ! -f "${HEAD_PATH}" ]]; then
  echo "[start][FATAL] Head weights missing: ${HEAD_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CLASSES_PATH}" ]]; then
  echo "[start][FATAL] label_classes.txt missing: ${CLASSES_PATH}" >&2
  exit 1
fi

# -------- Helpful runtime info --------
if python - <<'PY' >/dev/null 2>&1; then
import torch, os
print(f"[start][python] torch {torch.__version__}")
print(f"[start][python] cuda available: {torch.cuda.is_available()}")
print(f"[start][python] device count: {torch.cuda.device_count()}")
print(f"[start][python] visible devices: {os.getenv('CUDA_VISIBLE_DEVICES','<unset>')}")
PY
then
  true
else
  echo "[start][warn] Python/Torch probe failed (continuing)"
fi

# -------- Kill stale listeners on this port (optional) --------
if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP -sTCP:LISTEN -P | grep -q ":${PORT} "; then
    echo "[start] Port ${PORT} already in use; attempting to free it…"
    lsof -ti tcp:${PORT} | xargs -r kill -9 || true
    sleep 1
  fi
fi

# -------- Launch API --------
echo "[start] Launching: uvicorn ${APP_MODULE} --host ${HOST} --port ${PORT} --log-level ${LOG_LEVEL}"
exec uvicorn "${APP_MODULE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level "${LOG_LEVEL}" \
  ${UVICORN_FLAGS} \
  --workers "${WORKERS}"
