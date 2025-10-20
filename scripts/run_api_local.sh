#!/usr/bin/env bash
set -e
pip install -q -r requirements_api.txt
uvicorn app_run:app --host 0.0.0.0 --port 8000
