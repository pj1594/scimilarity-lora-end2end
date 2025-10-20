#!/usr/bin/env bash
set -e
pip install -q -r requirements_api.txt pyngrok
# Start FastAPI in background
nohup uvicorn app_run:app --host 127.0.0.1 --port 8000 > fastapi.log 2>&1 &
echo "[ok] FastAPI started. Tail logs with: tail -n 50 fastapi.log"
python - <<'PY'
from pyngrok import ngrok
url = ngrok.connect("http://127.0.0.1:8000", bind_tls=True).public_url
print("[ngrok]", url)
PY
