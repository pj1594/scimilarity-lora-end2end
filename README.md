# SCimilarity + LoRA-lite (End to End)

This repo includes:
- Data prep and mapping to SCimilarity gene order
- Lightweight (LoRA-lite) MLP head fine-tuning over SCimilarity embeddings
- Baseline vs fine-tuned comparison
- FastAPI inference service
- Docker + Cloud Run deploy

## Quickstart
See `deploy/deploy_cloud_run.sh` for one-command GCP deployment once your artifacts are in place.
