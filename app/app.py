"""
app.py — FastAPI inference for SCimilarity + LoRA head
Author: Prajwal Eachempati
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, List

from scimilarity.cell_embedding import CellEmbedding  # type: ignore

# ---------------- CONFIG ----------------
MODEL_DIR    = os.getenv("MODEL_DIR", "scimilarity_model_v1_1")
HEAD_PATH    = os.getenv("HEAD_PATH", "models/lora/linear_head.pt")
CLASSES_PATH = os.getenv("CLASSES_PATH", "models/lora/label_classes.txt")

TEMP       = float(os.getenv("TEMP", "0.7"))     # temperature
UNKNOWN_TH = float(os.getenv("THRESH", "0.10"))  # threshold for "Unknown"
DTYPE      = torch.float32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- FastAPI ----------------
app = FastAPI(title="SCimilarity + LoRA Inference", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------------- Input Schemas ----------------
class CellInput(BaseModel):
    expression: Dict[str, float]

    @validator("expression")
    def _validate(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Expression must be a dict of gene→value.")
        for k, val in v.items():
            _ = float(val)  # ensure numeric
        return v

class BatchInput(BaseModel):
    expressions: List[Dict[str, float]]
    topk: int = 3

# ---------------- Helpers ----------------
def load_gene_order(dirpath):
    f = os.path.join(dirpath, "gene_order.tsv")
    genes = pd.read_csv(f, sep="\t", header=None)[0].astype(str).str.upper().tolist()
    return genes

def normalize_expr(expr):
    vals = np.array(list(expr.values()), dtype=np.float32)
    if vals.sum() <= 0: return expr
    scale = 1e4 / vals.sum()
    return {k: float(np.log1p(v * scale)) for k, v in expr.items()}

def dict_to_tensor(expr, gene2idx, n):
    v = np.zeros(n, dtype=np.float32)
    for g, val in expr.items():
        idx = gene2idx.get(g.upper())
        if idx is not None:
            v[idx] = float(val)
    return torch.from_numpy(v).unsqueeze(0).to(DEVICE)

# ---------------- Load Base Model ----------------
encoder = CellEmbedding(model_path=MODEL_DIR, use_gpu=(DEVICE.type=="cuda"))

MODEL_GENES = load_gene_order(MODEL_DIR)
GENE2IDX = {g: i for i,g in enumerate(MODEL_GENES)}
N_GENES = len(MODEL_GENES)

# Ensure embedding dim
@torch.no_grad()
def infer_dim():
    probe = torch.zeros(1, N_GENES, dtype=DTYPE).to(DEVICE)
    z = encoder.get_embeddings(probe)
    z = torch.as_tensor(z, dtype=DTYPE, device=DEVICE)
    return z.shape[1]

EMB_DIM = infer_dim()

# Load label classes
with open(CLASSES_PATH) as f:
    CLASSES = [ln.strip() for ln in f if ln.strip()]
N_CLASSES = len(CLASSES)

# Load LoRA classification head
def load_head(path, emb_dim, n_cls):
    st = torch.load(path, map_location="cpu")
    # try MLP first
    try:
        h = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, n_cls)
        )
        h.load_state_dict(st)
    except:
        h = nn.Linear(emb_dim, n_cls)
        h.load_state_dict(st)
    return h.to(DEVICE).eval()

head = load_head(HEAD_PATH, EMB_DIM, N_CLASSES)

# ---------------- Core inference ----------------
@torch.no_grad()
def _predict(expr: Dict[str,float], topk=1):
    expr = normalize_expr(expr)
    x = dict_to_tensor(expr, GENE2IDX, N_GENES)
    z = encoder.get_embeddings(x)
    z = torch.as_tensor(z, dtype=DTYPE, device=DEVICE)

    logits = head(z) / max(TEMP,1e-6)
    prob   = torch.softmax(logits,dim=1)[0].cpu().numpy()

    order = np.argsort(prob)[::-1][:topk]
    out = [{"label": CLASSES[i], "confidence": float(prob[i])} for i in order]

    if UNKNOWN_TH>0 and out[0]["confidence"]<UNKNOWN_TH:
        out[0]["label"]="Unknown/Uncertain"

    return out

# ---------------- Routes ----------------
@app.get("/healthz")
def health():
    return dict(ok=True,
                device=str(DEVICE),
                n_genes=N_GENES,
                emb_dim=EMB_DIM,
                n_classes=N_CLASSES)

@app.get("/classes")
def classes():
    return {"classes": CLASSES}

@app.post("/predict")
def predict(inp: CellInput, topk: int = 1):
    try:
        res = _predict(inp.expression, topk=topk)
        return {"predictions": res}
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

@app.post("/predict_batch")
def predict_batch(inp: BatchInput):
    try:
        results=[]
        for ex in inp.expressions:
            results.append({"predictions": _predict(ex, topk=inp.topk)})
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, f"Batch error: {e}")
