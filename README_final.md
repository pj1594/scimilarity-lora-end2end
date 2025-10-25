# 🧬 SCimilarity + LoRA Fine-Tuning for Single-Cell Classification

**Author:** Prajwal Eachempati  
**Organization:** Navitas Consulting (AI & Process Automation Practice)  
**Objective:** Implement LoRA fine-tuning with Triplet + Reconstruction losses to improve single-cell classification accuracy while mitigating catastrophic forgetting.  
**Repo:** [https://github.com/pj1594/scimilarity-lora-end2end](https://github.com/pj1594/scimilarity-lora-end2end)

---

## 🚀 Overview

This project enhances the **SCimilarity** single-cell embedding model using **LoRA (Low-Rank Adaptation)** with **Triplet** and **Reconstruction** losses to improve biological class separability and mitigate *catastrophic forgetting*.  
The pipeline includes:
- Comparative analysis between **baseline encoder** and **LoRA-adapted encoder**
- Automated metric computation (accuracy, F1, precision, recall, balanced accuracy)
- Triplet loss and reconstruction loss evaluations
- Error analysis (confusion matrices + top misclassified cell populations)
- **FastAPI** service for real-time cell-type inference

---

## 🧭 Methodology

### 1. Dataset & Model Setup
- Input: single-cell RNA expression vectors (28,231 genes)
- Base Model: `CellEmbedding` from SCimilarity v1.1
- LoRA Fine-Tuning:
  - Rank `r=8`, α=16
  - Learning rate `1e-4`
  - 5 epochs (Adam optimizer)
  - Combined loss: `CrossEntropy + TripletLoss + λ * ReconLoss`

### 2. Encoder Sanity Check
To ensure a **fair baseline vs LoRA** comparison:
- `encoder_base` is cloned from LoRA encoder with adapters disabled  
- Both encoders verified for identical embedding dimensions  
- Cosine similarity checks confirm distinct representation subspaces

**Sanity Check Outputs**
```
LoRA embedding shape:     (N, 128)
Baseline embedding shape: (N, 128)
warnings:
 - "Embeddings checked to be different before applying for training."
 - "Ensure head_base retrained on baseline embeddings."
```

---

## 📊 Evaluation Results

| Model | Accuracy | F1 (macro) | Triplet Loss ↓ | Reconstruction MSE ↓ | Balanced Accuracy |
|--------|-----------|------------|----------------|----------------------|-------------------|
| **Baseline** | 34.0% | 0.31 | 0.284 | 0.037 | 0.33 |
| **LoRA (Triplet + Recon)** | 36.0% | 0.35 | 0.221 | 0.031 | 0.36 |

### ✅ Interpretation
- **+2% accuracy improvement** → stronger separability via Triplet loss  
- **Lower reconstruction loss** → reduced signal distortion  
- **Lower triplet loss** → tighter intra-class clusters  
- **Consistent balanced accuracy** → uniform performance across classes  

> These results are well within expected improvement ranges (1–3%) seen in **domain-specific LoRA fine-tuning**, especially for biological embeddings where overfitting is tightly constrained.

---

## 🔍 Error Analysis

### Confusion Matrices
- `artifacts/cm_lora.png`  
- `artifacts/cm_baseline.png`

### Top 5 Misclassified Cell Populations
| True Cell Type | Predicted | Frequency | Biological Plausibility |
|----------------|------------|------------|---------------------------|
| B-cell | Plasma-cell | 0.22 | Similar lineage |
| NK-cell | T-cell | 0.17 | Shared immune marker profile |
| Dendritic | Macrophage | 0.13 | Antigen-presenting overlap |
| Endothelial | Fibroblast | 0.09 | Spatial co-expression |
| Monocyte | Granulocyte | 0.07 | Hematopoietic proximity |

---

## 📈 Metric Interpretation

| Metric | Meaning | Why It Matters |
|---------|----------|----------------|
| **Accuracy** | Overall correctness | Measures prediction strength |
| **F1 (macro)** | Class-averaged harmonic mean | Robust to class imbalance |
| **Triplet Loss** | Enforces intra-class cohesion | Lower = tighter clusters |
| **Reconstruction MSE** | Measures feature retention | Lower = less forgetting |
| **Balanced Accuracy** | Mean of per-class recalls | Stability across rare types |

---

## 🧩 Repository Structure

| Path | Description |
|------|--------------|
| **app/** | Core logic for metrics, embeddings, and evaluation |
| ├── `metrics.py` | Accuracy, F1, Triplet, Recon loss, Confusion matrix plotting |
| ├── `model_io.py` | Batch embedding + normalization utilities |
| ├── `eval_runner.py` | Comparative evaluator for baseline vs LoRA |
| ├── `main.py` | Entry for full evaluation pipeline |
| ├── `app.py` | FastAPI inference service for `/predict` |
| ├── `run_api.py` | API launcher for deployment (Uvicorn-ready) |
| **deploy/start.sh** | Startup script with environment validation |
| **reports/Design_Analysis_Report.md** | In-depth architecture, metrics discussion, and loss analysis |
| **reports/** | Stores confusion matrices, top-5 misclassified CSVs, and summary |
| **requirements.txt** | All dependencies |
| **.gitignore** | Excludes large models & data |

---

## 🧠 Model Artifacts

For reproducibility, trained model weights and sample datasets are hosted externally:  
📦 [Hugging Face Model Repository](https://huggingface.co/praj-1594/scimilarity-lora-end2end/tree/main/models)

Ensure you have:
- `mlp_head_best.pt`
- `label_classes.txt`
- `gene_order.tsv`

> Large `.pt` files are **not** tracked in GitHub to comply with repository limits.

---

## 🧰 Quickstart

### 🔹 Setup
```bash
git clone https://github.com/pj1594/scimilarity-lora-end2end.git
cd scimilarity-lora-end2end
pip install -r requirements.txt
```

### 🔹 Evaluation
```bash
python -m app.main
# Generates summary.csv, confusion matrices, and misclassification logs
```

### 🔹 Launch API
```bash
uvicorn run_api:app --host 0.0.0.0 --port 8000
```
Send requests to:
```bash
POST /predict
{
  "expression": {"CD19": 3.5, "CD3": 0.1, ...}
}
```

---

## 📜 Design & Analysis Report
See [`reports/Design_Analysis_Report.md`] for:
- Detailed model architecture
- Triplet vs Reconstruction loss ablation
- Discussion on catastrophic forgetting mitigation
- Interpretation of confusion matrices and embedding clusters

---

## 📩 Contact
**Prajwal Eachempati**  
AI & Process Automation Consultant | PhD (MIS)  
📧 dataapraj@gmail.com  
🌐 [LinkedIn](https://linkedin.com/in/prajwaleachempati)  
🧠 Passion: AI-driven biological modeling, automation, and explainable GenAI systems.
