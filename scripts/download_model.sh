#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-./scimilarity_model_v1_1}"
URL="${2:-https://zenodo.org/records/10685499/files/model_v1.1.tar.gz}"
TGZ="$(pwd)/model_v1.1.tar.gz"

mkdir -p "$OUT_DIR"
if [ -f "$OUT_DIR/gene_order.tsv" ]; then
  echo "Model present at $OUT_DIR"; exit 0
fi

echo "Downloading SCimilarity from $URL ..."
curl -L "$URL" -o "$TGZ"
echo "Extracting ..."
tar -xzf "$TGZ" -C "$OUT_DIR"
rm -f "$TGZ"
echo "Done → $OUT_DIR"
