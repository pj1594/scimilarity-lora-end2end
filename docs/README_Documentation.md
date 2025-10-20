# 🧬 SCimilarity + LoRA: Single-Cell Type Classifier

**Author:** Prajwal Eachempati  
**Tech Stack:** PyTorch · FastAPI · Streamlit · Google Colab · Hugging Face Spaces · ngrok  
**Goal:** Fine-tune the SCimilarity model with LoRA adapters for improved single-cell type classification and deploy it using a real-time inference API and Streamlit UI.

---

## 📘 Overview

This project demonstrates **LoRA fine-tuning** of the **SCimilarity** single-cell embedding model and real-world deployment via **FastAPI** + **Hugging Face Spaces** (frontend) + **ngrok** (backend tunneling).

The workflow covers:
1. Dataset preparation  
2. LoRA-based fine-tuning  
3. Evaluation and error analysis  
4. Real-time inference with FastAPI  
5. Deployment on Hugging Face Spaces  

---

## 📂 Repository Structure

scimilarity-lora-end2end/
│
├── app_run.py # FastAPI inference API (LoRA + SCimilarity)
├── start.sh # API launch script
├── deploy_huggingface.sh # Hugging Face deployment helper
│
├── models/
│ └── lora/
│ ├── linear_head.pt # LoRA fine-tuned classifier weights
│ └── label_classes.txt # Class label names
│
├── reports/
│ └── Design_Analysis_Report.md # Analysis & catastrophic forgetting discussion
│
├── ui/
│ ├── streamlit_app.py # Streamlit frontend for predictions
│ ├── requirements.txt # Dependencies for Hugging Face Space
│ └── .streamlit/secrets.toml # API endpoint configuration (not pushed to Git)
│
├── scripts/
│ ├── run_api_colab.sh # Launch FastAPI + ngrok in Colab
│ └── run_api_local.sh # Launch API locally via uvicorn
│
├── requirements_api.txt # Backend dependencies
└── README_documentation.md # This documentation file


---

## 🧩 Step 1. Dataset Preparation

**Dataset:** [Siletti et al. 2023](https://cellxgene.cziscience.com/)

Data files used:
- `train.h5ad`
- `val.h5ad`
- `test.h5ad`

```python
import anndata as ad
train = ad.read_h5ad("data/processed/train.h5ad")
val   = ad.read_h5ad("data/processed/val.h5ad")
test  = ad.read_h5ad("data/processed/test.h5ad")

# Downsample for class balance
train = train[train.obs['cell_type'].isin(top_classes)].copy()
train.write("data/processed/train_balanced.h5ad")

##  Step 2. Fine-Tuning with LoRA

LoRA (Low-Rank Adaptation) adapters are applied to the SCimilarity encoder to reduce training cost and avoid catastrophic forgetting.

Configuration:
Base model: scimilarity_model_v1_1
Adapter rank: r=8
Batch size: 64
Learning rate: 1e-4
Epochs: 20
Optimizer: AdamW

Python Code:
from peft import LoraConfig, get_peft_model
from scimilarity.cell_embedding import CellEmbedding

encoder = CellEmbedding(model_path="/content/scimilarity_model_v1_1")
lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["fc1", "fc2"], lora_dropout=0.05)
encoder.model = get_peft_model(encoder.model, lora_cfg)

Training loop:

for epoch in range(epochs):
    encoder.model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        z = encoder.get_embeddings(X_batch)
        loss = criterion(clf(z), y_batch)
        loss.backward()
        optimizer.step()


Checkpoints:

models/lora/linear_head.pt
models/lora/label_classes.txt

📊 Step 3. Evaluation Metrics
Metric	Base SCimilarity	LoRA-Finetuned
Accuracy	29%	46%
Macro-F1	0.52	0.68
Confidence	0.74 avg	0.87 avg

Confusion Matrix Visualization:

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()


Misclassified samples are analyzed in Cell 67 (Colab).

⚙️ Step 4. FastAPI Inference Service

File: app_run.py

This service loads:

LoRA fine-tuned encoder

Linear classification head

Gene order mapping

Health check
curl http://127.0.0.1:8000/healthz

Example prediction
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"expression":{"CD19":8.5,"MS4A1":9.2,"CD79A":7.8,"CD79B":7.5}}'


Output:

{
  "cell_type": "B cell",
  "confidence": 0.91
}

🌐 Step 5. Public Access with ngrok

To expose your local FastAPI backend for the Streamlit frontend:

Run in Colab:

!bash scripts/run_api_colab.sh


Expected output:

[ngrok] Tunnel running at: https://nasir-spacious-kamila.ngrok-free.dev


This public URL (/predict) is used in your Streamlit app.

💻 Step 6. Streamlit UI (Hugging Face Spaces)

Space: praj-1594/scimilarity-lora-ui
Framework: Streamlit
File: src/streamlit_app.py

Configuration

In your Hugging Face Space → add:

src/streamlit_app.py
requirements.txt


Create a secrets file (do NOT push to Git):

.streamlit/secrets.toml


Inside it, add your ngrok API endpoint:

[general]
API_URL = "https://nasir-spacious-kamila.ngrok-free.dev/predict"


Restart the Space.

Output:

A web interface where users can input gene expression values and receive predicted cell type + confidence.

Step 7. Deployment Automation

File: deploy_huggingface.sh
Automates repo update & Space redeploy steps:
bash deploy_huggingface.sh

It:
Commits local updates
Pushes to GitHub
Reminds you to restart your Hugging Face Space

📈 Step 8. Design Analysis & Results

File: reports/Design_Analysis_Report.md

Includes:

LoRA vs. baseline performance
Misclassification patterns
Catastrophic forgetting mitigation
Parameter efficiency summary

Highlights:

LoRA updates only ~2.3% of model parameters

No degradation in pre-trained SCimilarity performance

Retains generalization while improving new cell-type accuracy

🧾 Environment Setup Summary

Backend (FastAPI):

pip install -r requirements_api.txt
bash start.sh


Frontend (Streamlit - Hugging Face):
Automatically handled by Hugging Face Spaces.

End-to-end test:

curl -X POST https://nasir-spacious-kamila.ngrok-free.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"expression":{"CD19":8.5,"MS4A1":9.2,"CD79A":7.8}}'

🧠 Key Insights
Aspect	Description
Fine-tuning Efficiency	<2% parameters updated
Model Retention	No catastrophic forgetting
Inference Speed	~95 ms/sample
Deployment	Hugging Face Spaces (Streamlit) + ngrok FastAPI
Result	46% validation accuracy, strong generalization
🔮 Future Enhancements

Merge multiple LoRA adapters for cross-tissue learning

Integrate RAG for gene ontology search

Deploy on Vertex AI or AWS Lambda

