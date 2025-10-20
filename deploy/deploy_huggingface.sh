#!/usr/bin/env bash
set -e

echo "Starting deployment of SCimilarity + LoRA on Hugging Face Spaces..."

# 1️⃣ Build context
SPACE_REPO="praj-1594/scimilarity-lora-ui"
LOCAL_DIR="$(pwd)"

echo "[1/4] Verifying required files..."
REQUIRED_FILES=("src/streamlit_app.py" "requirements.txt" "Dockerfile" ".streamlit/secrets.toml")
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" && ! -d "$f" ]]; then
    echo "❌ Missing: $f"
  else
    echo "✅ Found: $f"
  fi
done

# 2️⃣ Git push workflow
echo "[2/4] Committing latest code..."
git add .
git commit -m "Deploy update to Hugging Face Space ${SPACE_REPO}" || echo "No new changes to commit"
git push origin main

# 3️⃣ Instructions for HF deployment
echo "[3/4] Updating Hugging Face Space..."
echo "   - Open https://huggingface.co/spaces/${SPACE_REPO}"
echo "   - Ensure Space type: 'Streamlit'"
echo "   - Ensure 'requirements.txt' lists streamlit, requests, pandas, numpy"
echo "   - Ensure Dockerfile (optional) points to start Streamlit from src/streamlit_app.py"
echo "   - Add secret API_URL in Settings → Variables & Secrets"

# 4️⃣ Verify secrets.toml existence
echo "[4/4] Checking secrets file..."
if [[ -f ".streamlit/secrets.toml" ]]; then
  echo "✅ .streamlit/secrets.toml present. (Do NOT push real URLs in Git.)"
else
  echo "⚠️ Warning: .streamlit/secrets.toml missing locally. Create one if deploying manually:"
  echo "    [general]"
  echo "    API_URL = \"https://<your-ngrok-url>/predict\""
fi

echo "✅ Deployment script complete."
echo "➡️ Now go to your Hugging Face Space UI and click 'Restart Space' to rebuild."
