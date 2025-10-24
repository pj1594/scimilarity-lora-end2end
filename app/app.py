import os, traceback
import numpy as np, pandas as pd, torch, torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, Optional

# ----------------- CONFIG -----------------
MODEL_DIR    = os.getenv("MODEL_DIR", "/content/scimilarity_model_v1_1")
HEAD_PATH    = os.getenv("HEAD_PATH", "models/lora/mlp_head_best.pt")
CLASSES_PATH = os.getenv("CLASSES_PATH", "models/lora/label_classes.txt")
LABEL_KEY    = os.getenv("LABEL_KEY", "cell_type")
TEMP         = float(os.getenv("TEMP", "0.7"))
THRESH       = float(os.getenv("THRESH", "0.40"))
DTYPE        = torch.float32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- MODEL -----------------
from scimilarity.cell_embedding import CellEmbedding  # type: ignore

app = FastAPI(title="SCimilarity + LoRA Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def _log(msg): print(f"[api] {msg}", flush=True)

# ----- Input schema -----
class CellInput(BaseModel):
    expression: Dict[str, float]
    @validator("expression")
    def _check_expr(cls, v):
        if not isinstance(v, dict):
            raise ValueError("`expression` must be a dict of {gene_symbol: numeric_value}.")
        for k, val in v.items():
            try: float(val)
            except Exception: raise ValueError(f"value for {k} must be numeric (got {val!r})")
        return v

# ----- Normalization -----
def _normalize_expr_dict(expr: Dict[str, float]) -> Dict[str, float]:
    vals = np.asarray(list(expr.values()), dtype=np.float32)
    if vals.size == 0 or float(vals.sum()) <= 0.0:
        return expr
    scale = 1e4 / float(vals.sum())
    return {k: float(np.log1p(float(v) * scale)) for k, v in expr.items()}

# ----- Load encoder -----
try:
    encoder = CellEmbedding(model_path=MODEL_DIR, use_gpu=(DEVICE.type == "cuda"))
    _log(f"Loaded SCimilarity from {MODEL_DIR} on {DEVICE}")
except Exception as e:
    raise RuntimeError(f"Failed to load SCimilarity model from {MODEL_DIR}: {e}")

# ----- Gene order -----
gorder_tsv = os.path.join(MODEL_DIR, "gene_order.tsv")
if not os.path.exists(gorder_tsv):
    raise FileNotFoundError(f"Missing gene_order.tsv at {gorder_tsv}")

MODEL_GENES = pd.read_csv(gorder_tsv, sep="\t", header=None)[0].astype(str).str.upper().tolist()
N_GENES = len(MODEL_GENES)
GENE2IDX = {g: i for i, g in enumerate(MODEL_GENES)}

# ----- Encode (LoRA-safe) -----
@torch.no_grad()
def lora_encode(self, x: torch.Tensor) -> torch.Tensor:
    # Use underlying .model if present (PEFT/LoRA wrapper), else assume self is a nn.Module-like
    m = getattr(self, "model", None)
    x = x.to(DEVICE, dtype=DTYPE, non_blocking=True)
    if m is not None:
        m.eval()
        out = m(x)
    else:
        # fallback to CellEmbedding.get_embeddings
        out = self.get_embeddings(x)
    return torch.as_tensor(out, dtype=DTYPE, device=DEVICE)

# Bind encode
encoder.encode = lora_encode.__get__(encoder, type(encoder))

# ----- Classes -----
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Missing classes at {CLASSES_PATH}")
CLASSES = [ln.strip() for ln in open(CLASSES_PATH, "r", encoding="utf-8") if ln.strip()]
N_CLASSES = len(CLASSES)

# ----- Probe emb dim -----
with torch.no_grad():
    probe = torch.zeros(1, N_GENES, dtype=DTYPE, device=DEVICE)
    z = encoder.encode(probe)
    if z.ndim == 1: z = z.unsqueeze(0)
    EMB_DIM = int(z.shape[1])
_log(f"Embedding dim = {EMB_DIM}, classes = {N_CLASSES}")

# ----- Head loader (MLP or Linear) -----
class MLP(nn.Module):
    def __init__(self, d: int, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, c),
        )
    def forward(self, x): return self.net(x)

if not os.path.exists(HEAD_PATH):
    # fallback to linear head file if mlp not present
    alt = "models/lora/linear_head.pt"
    if os.path.exists(alt):
        HEAD_PATH = alt
    else:
        raise FileNotFoundError(f"Missing head weights at {HEAD_PATH}")

state = torch.load(HEAD_PATH, map_location="cpu")
_head: Optional[nn.Module] = None
try:
    h = MLP(EMB_DIM, N_CLASSES)
    h.load_state_dict(state, strict=True)
    _head = h
    _log(f"Loaded MLP head from {HEAD_PATH}")
except Exception:
    h = nn.Linear(EMB_DIM, N_CLASSES)
    h.load_state_dict(state, strict=True)
    _head = h
    _log(f"Loaded Linear head from {HEAD_PATH}")

head = _head.to(DEVICE).eval()

# ----- Vectorization -----
def dict_to_tensor(expr: Dict[str, float]) -> torch.Tensor:
    vec = np.zeros(N_GENES, dtype=np.float32)
    for g, v in expr.items():
        i = GENE2IDX.get(str(g).upper())
        if i is not None:
            vec[i] = float(v)
    x = torch.from_numpy(vec).to(device=DEVICE, dtype=DTYPE)
    if x.ndim == 1: x = x.unsqueeze(0)
    return x

# ================= Routes =================
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "device": str(DEVICE),
        "n_genes": N_GENES,
        "emb_dim": EMB_DIM,
        "n_classes": N_CLASSES,
        "model_dir": MODEL_DIR,
        "head_path": HEAD_PATH,
    }

@app.get("/classes")
def classes():
    return {"classes": CLASSES}

@app.post("/predict")
@torch.no_grad()
def predict(inp: CellInput):
    try:
        expr = _normalize_expr_dict(inp.expression)
        x = dict_to_tensor(expr)
        z = encoder.encode(x)
        logits = head(z) / max(TEMP, 1e-6)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        i = int(np.argmax(probs))
        label = CLASSES[i]; conf = float(probs[i])
        if THRESH > 0.0 and conf < THRESH:
            label = "Unknown / Uncertain"
        return {"cell_type": label, "confidence": conf}
    except Exception as e:
        print("[api] ERROR /predict:", e, "\n", traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

# Optional batch endpoint (useful for UI/testing)
class BatchInput(BaseModel):
    expressions: Dict[str, Dict[str, float]]  # id -> {gene:value}

@app.post("/predict_batch")
@torch.no_grad()
def predict_batch(inp: BatchInput):
    try:
        ids, mats = [], []
        for k, expr in inp.expressions.items():
            expr_n = _normalize_expr_dict(expr)
            ids.append(k); mats.append(dict_to_tensor(expr_n))
        X = torch.cat(mats, dim=0)
        z = encoder.encode(X)
        logits = head(z) / max(TEMP, 1e-6)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = []
        for rid, p in zip(ids, probs):
            i = int(np.argmax(p)); conf = float(p[i]); label = CLASSES[i]
            if THRESH > 0.0 and conf < THRESH:
                label = "Unknown / Uncertain"
            preds.append({"id": rid, "cell_type": label, "confidence": conf})
        return {"results": preds}
    except Exception as e:
        print("[api] ERROR /predict_batch:", e, "\n", traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {e}")
