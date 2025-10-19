#!/usr/bin/env bash
set -e
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
echo "[start] MODEL_DIR=$MODEL_DIR"
echo "[start] HEAD_PATH=$HEAD_PATH"
echo "[start] CLASSES_PATH=$CLASSES_PATH"

# Optional: download SCimilarity if MODEL_TGZ_URL provided and gene_order missing
if [ ! -f "$MODEL_DIR/gene_order.tsv" ] && [ -n "$MODEL_TGZ_URL" ]; then
  echo "[start] Downloading SCimilarity from $MODEL_TGZ_URL ..."
  mkdir -p "$MODEL_DIR"
  TMP_TGZ="/tmp/model_v1.1.tar.gz"
  curl -L "$MODEL_TGZ_URL" -o "$TMP_TGZ"
  echo "[start] Extracting to $MODEL_DIR"
  tar -xzf "$TMP_TGZ" -C "$MODEL_DIR"
  rm -f "$TMP_TGZ"
fi

exec uvicorn app.app:app --host "$HOST" --port "$PORT"
