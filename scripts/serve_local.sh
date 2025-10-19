#!/usr/bin/env bash
set -e
export MODEL_DIR="${MODEL_DIR:-$PWD/scimilarity_model_v1_1}"
export HEAD_PATH="${HEAD_PATH:-$PWD/models/lora/mlp_head_best.pt}"
export CLASSES_PATH="${CLASSES_PATH:-$PWD/models/lora/label_classes.txt}"
export MODEL_TGZ_URL="${MODEL_TGZ_URL:-}"  # optional
python -m uvicorn app.app:app --host 0.0.0.0 --port 8000
