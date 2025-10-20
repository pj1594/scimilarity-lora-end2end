import os, json, traceback
import numpy as np, pandas as pd, torch, torch.nn as nn
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, validator
from scimilarity.cell_embedding import CellEmbedding
import anndata as ad # Import anndata for fallback class loading

# ---- config / paths ----
MODEL_DIR    = os.getenv("MODEL_DIR", "/content/scimilarity_model_v1_1")
HEAD_PATH    = os.getenv("HEAD_PATH", "/content/models/lora/mlp_head_best.pt") # Use mlp_head_best.pt if it exists, otherwise linear_head.pt
CLASSES_PATH = os.getenv("CLASSES_PATH", "/content/models/lora/label_classes.txt")
TEST_PATH    = "data/processed/test.h5ad" # Path to test data for fallback
LABEL_KEY    = "cell_type" # Default label key for fallback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

app = FastAPI(title="SCimilarity+LoRA Cell Classifier")

# ---------- helpers ----------
def log(msg): print(f"[api] {msg}", flush=True)

class CellInput(BaseModel):
    expression: dict
    @validator("expression")
    def validate_expr(cls, v):
        if not isinstance(v, dict):
            raise ValueError("expression must be a dict")
        for k, val in v.items():
            try:
                float(val)
            except Exception:
                raise ValueError(f"value for gene {k} must be numeric")
        return v

def normalize_expression_dict(expr: dict) -> dict:
    vals = np.array(list(expr.values()), dtype=np.float32)
    if vals.size == 0 or vals.sum() <= 0:
        return expr
    scale = 1e4 / float(vals.sum())
    return {k: float(np.log1p(float(v) * scale)) for k, v in expr.items()}

# ---------- load encoder ----------
# Add encoder initialization here to make this cell self-contained
try:
    encoder = CellEmbedding(model_path=MODEL_DIR, use_gpu=(DEVICE.type == "cuda"))
    log(f"Loaded SCimilarity from {MODEL_DIR} on {DEVICE}")
except Exception as e:
    raise RuntimeError(f"Failed to load SCimilarity model from {MODEL_DIR}: {e}")


# ---------- gene order ----------
gorder_tsv = os.path.join(MODEL_DIR, "gene_order.tsv")
if not os.path.exists(gorder_tsv):
    raise FileNotFoundError(f"Missing gene_order.tsv at {gorder_tsv}")

MODEL_GENES = pd.Index(pd.read_csv(gorder_tsv, sep="\t", header=None)[0].astype(str).str.upper())
if len(MODEL_GENES) != 28231:
    log(f"WARNING: gene_order length is {len(MODEL_GENES)} (expected 28231)")

GENE2IDX = {g: i for i, g in enumerate(MODEL_GENES)}

def dict_to_vector(expr: dict) -> torch.Tensor:
    """Map expression dict → model-ordered 1xN tensor on DEVICE/DTYPE."""
    v = np.zeros(len(MODEL_GENES), dtype=np.float32)
    for g, val in expr.items():
        i = GENE2IDX.get(str(g).upper())
        if i is not None:
            v[i] = float(val)
    x = torch.from_numpy(v).to(device=DEVICE, dtype=DTYPE)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, 28231]
    return x

# ---------- classes ----------
classes = None
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH) as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    log(f"Loaded {len(classes)} classes from {CLASSES_PATH}")
else:
    # Fallback: Load classes from test AnnData if classes file is missing
    log(f"Warning: Missing classes file at {CLASSES_PATH}. Attempting to load classes from {TEST_PATH}.")
    if os.path.exists(TEST_PATH):
        try:
            test_adata = ad.read_h5ad(TEST_PATH)
            if LABEL_KEY in test_adata.obs:
                classes = test_adata.obs[LABEL_KEY].astype("category").cat.categories.tolist()
                log(f"Successfully loaded {len(classes)} classes from {TEST_PATH}")
            else:
                raise ValueError(f"Label key '{LABEL_KEY}' not found in {TEST_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to load classes from {TEST_PATH}: {e}")
    else:
        raise FileNotFoundError(f"Missing classes file at {CLASSES_PATH} and test file at {TEST_PATH}")

if classes is None or len(classes) == 0:
    raise RuntimeError("Failed to load any classes.")

num_classes = len(classes)


# ---------- probe emb dim ----------
# Use the monkey-patched encode method to probe the embedding dimension
@torch.no_grad()
def probe_embedding_dim(encoder_obj, sample_size):
    """Probe the embedding dimension using the encoder_obj's encode method."""
    # Create a dummy input tensor on the correct device
    dummy_input = torch.zeros(1, sample_size, dtype=DTYPE, device=DEVICE)
    # Call the encoder_obj's encode method (monkey-patched)
    z = encoder_obj.encode(dummy_input)
    if not isinstance(z, torch.Tensor):
         z = torch.as_tensor(z, dtype=DTYPE, device=DEVICE)
    return int(z.shape[1])

try:
    # Need to apply the monkey patch for encode method to the newly initialized encoder
    # Define the corrected monkey-patched encode function within this cell
    def lora_encode_peft(self, x):
        # Ensure the model is in evaluation mode for inference
        self.model.eval() # Set the LoRA-adapted model to evaluation mode

        # Ensure input is on the correct device
        # Get device from a parameter of the LoRA-adapted model
        # Use encoder_lora.model to get the device, assuming it's accessible
        # If encoder_lora is not directly accessible here, use self.model
        device = next(self.model.parameters()).device
        x = x.to(device)

        # Apply the LoRA-adapted model
        with torch.no_grad(): # Encoding itself is typically done without gradients
            # Pass through the LoRA-adapted model (which is self.model here)
            z = self.model(x)

        # Optionally, set the model back to train mode if this function might be called during training
        # self.model.train()

        return z

    # Monkey patch the encode method on the original encoder object
    # This will ensure the /predict endpoint uses this corrected function
    # Ensure 'encoder' object is accessible here, which it should be from previous cells.
    # Now encoder is initialized within this cell, so it's accessible.
    encoder.encode = lora_encode_peft.__get__(encoder, type(encoder))


    emb_dim = probe_embedding_dim(encoder, len(MODEL_GENES))
    log(f"Probed Embedding dimension = {emb_dim}")
except Exception as e:
     # Fallback to hardcoded dimension if probing fails
     emb_dim = 128
     log(f"Warning: Failed to probe embedding dimension ({e}). Assuming emb_dim = {emb_dim}")


# ---------- build head to match saved weights ----------
# Prioritize mlp_head_best.pt, fallback to linear_head.pt
actual_head_path = "models/lora/mlp_head_best.pt" if os.path.exists("models/lora/mlp_head_best.pt") else "models/lora/linear_head.pt"
if not os.path.exists(actual_head_path):
     raise FileNotFoundError(f"Missing head weights at both models/lora/mlp_head_best.pt and models/lora/linear_head.pt")

state = torch.load(actual_head_path, map_location="cpu")

def try_load_head(module: nn.Module, state_dict):
    module.load_state_dict(state_dict, strict=True)
    return module.to(DEVICE).eval()

head = None
load_error = None

class MLP(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, c)
        )
    def forward(self, x): return self.net(x)

try:
    # Check if the state dict keys match the MLP head before trying to load as MLP
    # This is a heuristic to guess if the saved state is from an MLP
    mlp_keys = ['net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias']
    is_mlp_state = all(k in state for k in mlp_keys)

    if is_mlp_state:
        head = try_load_head(MLP(emb_dim, num_classes), state)
        log(f"Loaded MLP head weights from {actual_head_path}")
    else:
        log(f"State dict from {actual_head_path} does not match expected MLP keys. Trying Linear…")
        head = try_load_head(nn.Linear(emb_dim, num_classes), state)
        log(f"Loaded Linear head weights from {actual_head_path}")

except Exception as e:
    load_error = e
    log(f"Head load failed from {actual_head_path}. Last error: {e}")
    # Re-raise if head is still None
    if head is None:
         raise RuntimeError(f"Could not load head from {actual_head_path}. Last error: {load_error}")


# ---------- inference settings ----------
TEMP   = float(os.getenv("TEMP", "0.7"))   # temperature scaling
THRESH = float(os.getenv("THRESH", "0.4")) # low-confidence threshold


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
@torch.no_grad()
def predict(inp: CellInput):
    try:
        # 1) normalize + map to model order
        expr = normalize_expression_dict(inp.expression)
        x = dict_to_vector(expr)                 # torch tensor [1, 28231] on DEVICE/DTYPE
        # 2) embed using the monkey-patched encode method
        z = encoder.encode(x)             # torch tensor [1, emb_dim] on DEVICE/DTYPE
        # 3) classify
        logits = head(z) / TEMP                  # stays tensor
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        i      = int(np.argmax(probs))
        label  = classes[i]
        conf   = float(probs[i])
        if conf < THRESH:
            label = "Unknown / Uncertain"
        return {"cell_type": label, "confidence": conf}
    except Exception as e:
        print("[api] ERROR /predict:", e, "\n", traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

