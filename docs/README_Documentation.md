# 📘 SCimilarity + LoRA Documentation

---

## 1️⃣ Dataset Preparation

### 🔹 Source
- Select a **single-cell RNA-seq dataset** (post-2022) with annotated cell types.  
- Recommended source: [CellxGene Portal](https://cellxgene.cziscience.com/datasets).

### 🔹 Example Dataset
For demonstration, we used the **Siletti et al., 2023** human cortical dataset (`siletti_pp.h5ad`), containing ~45 K cells and 28 K genes.

### 🔹 Download Instructions (Colab / Cloud Shell)
```bash
# Create folder for data
mkdir -p data/raw

# Example: download Siletti dataset from CellxGene (replace URL with your selected one)
wget -O data/raw/siletti.h5ad "https://datasets.cellxgene.cziscience.com/XXXXXXXXX.h5ad"

# Optionally downsample to 20 K–50 K cells for quick experiments
python scripts/downsample_adata.py --input data/raw/siletti.h5ad --output data/siletti_pp.h5ad --n_cells 30000
import scanpy as sc
adata = sc.read_h5ad("data/siletti_pp.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
adata.write_h5ad("siletti_pp.h5ad")

##2️⃣ Fine-Tuning and Evaluation 
🔹 Step 1 – Load SCimilarity Model
from scimilarity.cell_embedding import CellEmbedding
encoder = CellEmbedding(model_path="/content/scimilarity_model_v1_1", use_gpu=True)

🔹 Step 2 – Attach LoRA Adapters
from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["linear","proj"])
encoder_lora = get_peft_model(encoder, lora_cfg)

🔹 Step 3 – Train MLP Head
clf = torch.nn.Linear(encoder.output_dim, num_classes)
opt = torch.optim.AdamW([...])

🔹 Step 4 – Evaluate
Metric	Baseline	LoRA-Tuned
Accuracy	31 %	99 %
Macro F1	0.04	0.88

Visualization:

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val, display_labels=label_names)

##3️⃣ Inference Service (FastAPI)
🔹 Run Locally
uvicorn app.app:app --host 0.0.0.0 --port 8000

🔹 Example Request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"expression":{"CD19":8.5,"MS4A1":9.2,"CD79A":7.8,"CD3D":0.0,"CD8A":0.0,"CD4":0.0}}'


Expected Response

{
  "cell_type": "B cell",
  "confidence": 0.87
}

🔹 Docker Deployment
docker build -t scimilarity-lora .
docker run -p 8000:8000 scimilarity-lora

🔹 Cloud Run Deployment
gcloud builds submit --tag gcr.io/$PROJECT_ID/scimilarity-lora
gcloud run deploy scimilarity-lora-api \
  --image gcr.io/$PROJECT_ID/scimilarity-lora \
  --platform managed \
  --allow-unauthenticated

##4️⃣ File Structure
📦 scimilarity-lora-end2end
├── data/               # raw + preprocessed datasets
├── models/lora/        # LoRA adapters + classifier head
├── app/                # FastAPI inference service
├── training/           # fine-tuning & evaluation code
├── deploy/             # Docker + Cloud Run scripts
└── notebooks/          # end-to-end Colab notebook

##5️⃣ Reproducibility
git clone https://github.com/pj1594/scimilarity-lora-end2end.git
pip install -r app/requirements.txt
jupyter nbconvert --to notebook --execute notebooks/SCimilarity_LoRA_Colab.ipynb
