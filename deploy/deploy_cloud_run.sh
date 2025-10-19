#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="cellsimilaritydeployment"
REGION="us-central1"
SERVICE_NAME="scimilarity-api"
AR_REPO="scimilarity-repo"
APP_IMAGE="us-central1-docker.pkg.dev/cellsimilaritydeployment/scimilarity-repo/scimilarity-lora:latest"

MODEL_DIR="/models/scimilarity_model_v1_1"
HEAD_PATH="/models/lora/mlp_head_best.pt"
CLASSES_PATH="/models/lora/label_classes.txt"

MEMORY="8Gi"
CPU="4"
TIMEOUT="900"
PORT="8000"
MAX_INSTANCES="1"
MIN_INSTANCES="0"

echo "==> Config"
gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud config set run/region "${REGION}" >/dev/null

echo "==> Enable services"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

echo "==> Create Artifact Registry (if missing)"
if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" --repository-format=docker --location="${REGION}" --description="SCimilarity images"
fi

echo "==> Build & Push image"
gcloud builds submit --tag "${APP_IMAGE}"

echo "==> Deploy to Cloud Run"
ENV_FLAGS=( --set-env-vars "MODEL_DIR=${MODEL_DIR}" --set-env-vars "HEAD_PATH=${HEAD_PATH}" --set-env-vars "CLASSES_PATH=${CLASSES_PATH}" )
if [ -n "https://zenodo.org/records/10685499/files/model_v1.1.tar.gz" ]; then ENV_FLAGS+=( --set-env-vars "MODEL_TGZ_URL=https://zenodo.org/records/10685499/files/model_v1.1.tar.gz" ); fi

gcloud run deploy "${SERVICE_NAME}"   --image "${APP_IMAGE}"   --platform managed   --region "${REGION}"   --allow-unauthenticated   --memory "${MEMORY}"   --cpu "${CPU}"   --timeout "${TIMEOUT}"   --port "${PORT}"   --max-instances "${MAX_INSTANCES}"   --min-instances "${MIN_INSTANCES}"   "${ENV_FLAGS[@]}"

URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format 'value(status.url)')"
echo "URL = $URL"

echo "==> Health check"
curl -fsS "$URL/healthz" && echo " (health OK)"
echo "==> Sample predict"
curl -fsS -X POST "$URL/predict" -H "Content-Type: application/json" -d '{"expression":{"CD19":8.5,"MS4A1":9.2,"CD79A":7.8}}' | sed 's/.*/Predict → &/'
