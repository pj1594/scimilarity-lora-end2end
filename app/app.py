import os, numpy as np, pandas as pd, torch, torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from scimilarity.cell_embedding import CellEmbedding

MODEL_DIR    = os.getenv("MODEL_DIR", "/models/scimilarity_model_v1_1")
HEAD_PATH    = os.getenv("HEAD_PATH", "/models/lora/mlp_head_best.pt")
CLASSES_PATH = os.getenv("CLASSES_PATH", "/models/lora/label_classes.txt")
TEMP   = float(os.getenv("TEMP", "0.7"))
THRESH = float(os.getenv("THRESH", "0.4"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

app = FastAPI(title="SCimilarity+LoRA-lite API")

class CellInput(BaseModel):
    expression: dict
    @validator("expression")
    def validate_expr(cls, v):
        if not isinstance(v, dict): raise ValueError("expression must be a dict")
        for k,val in v.items(): float(val)
        return v

# Load encoder
encoder = CellEmbedding(model_path=MODEL_DIR, use_gpu=(DEVICE.type == "cuda"))

# Gene order
gorder_tsv = os.path.join(MODEL_DIR, "gene_order.tsv")
if not os.path.exists(gorder_tsv):
    raise FileNotFoundError(f"Missing gene_order.tsv at {gorder_tsv}")
MODEL_GENES = pd.Index(pd.read_csv(gorder_tsv, sep="\t", header=None)[0].astype(str).str.upper())
GENE2IDX = {g:i for i,g in enumerate(MODEL_GENES)}

# Classes
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Missing classes file at {CLASSES_PATH}")
classes = [ln.strip() for ln in open(CLASSES_PATH) if ln.strip()]
num_classes = len(classes)

# Probe emb dim
with torch.no_grad():
    probe = torch.zeros(1, len(MODEL_GENES), dtype=DTYPE, device=DEVICE)
    z = encoder.get_embeddings(probe)
    if isinstance(z, np.ndarray): z = torch.from_numpy(z)
    else: z = torch.as_tensor(z)
    if z.ndim == 1: z = z.unsqueeze(0)
    emb_dim = int(z.shape[1])

# Head (MLP; fallback to Linear)
class MLP(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, c)
        )
    def forward(self, x): return self.net(x)

state = torch.load(HEAD_PATH, map_location="cpu")
head = MLP(emb_dim, num_classes).to(DEVICE).eval()
try:
    head.load_state_dict(state, strict=True)
except Exception:
    head = nn.Linear(emb_dim, num_classes).to(DEVICE).eval()
    head.load_state_dict(state, strict=True)

def normalize_expression_dict(expr: dict) -> dict:
    vals = np.array(list(expr.values()), dtype=np.float32)
    if vals.size == 0 or vals.sum() <= 0: return expr
    scale = 1e4 / float(vals.sum())
    return {k: float(np.log1p(float(v)*scale)) for k, v in expr.items()}

def dict_to_vector(expr: dict) -> torch.Tensor:
    v = np.zeros(len(MODEL_GENES), dtype=np.float32)
    for g, val in expr.items():
        i = GENE2IDX.get(str(g).upper())
        if i is not None: v[i] = float(val)
    x = torch.from_numpy(v).to(device=DEVICE, dtype=DTYPE)
    if x.ndim == 1: x = x.unsqueeze(0)
    return x

def get_embeddings_tensor(x: torch.Tensor) -> torch.Tensor:
    z = encoder.get_embeddings(x)
    if isinstance(z, np.ndarray): z = torch.from_numpy(z)
    else: z = torch.as_tensor(z)
    if z.ndim == 1: z = z.unsqueeze(0)
    return z.to(device=DEVICE, dtype=DTYPE)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
@torch.no_grad()
def predict(inp: CellInput):
    try:
        expr = normalize_expression_dict(inp.expression)
        x = dict_to_vector(expr)
        z = get_embeddings_tensor(x)
        logits = head(z) / TEMP
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        i = int(np.argmax(probs))
        label = classes[i]; conf = float(probs[i])
        if conf < THRESH: label = "Unknown / Uncertain"
        return {"cell_type": label, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
