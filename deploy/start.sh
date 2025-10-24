#!/usr/bin/env bash
set -e
exec uvicorn app.app_run:app --host 0.0.0.0 --port ${PORT:-8000}
