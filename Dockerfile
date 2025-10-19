FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000 \
    MODEL_DIR=/models/scimilarity_model_v1_1 \
    HEAD_PATH=/models/lora/mlp_head_best.pt \
    CLASSES_PATH=/models/lora/label_classes.txt \
    TEMP=0.7 THRESH=0.4

WORKDIR /app
COPY app/requirements.txt /app/app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential tini curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /app/app/requirements.txt

COPY app /app/app
RUN chmod +x /app/app/start.sh && mkdir -p /models/lora /models/scimilarity_model_v1_1
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/app/app/start.sh"]
