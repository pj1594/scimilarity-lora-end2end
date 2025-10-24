import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Robust converters -----------------
def _as_tensor(x, device=DEVICE, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(x, dtype=dtype, device=device)

def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)

# ----------------- Core indices -----------------
def compute_indices(y_true, y_pred):
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

# ----------------- Triplet loss -----------------
@torch.no_grad()
def batch_hard_triplet_loss(Z, y, margin=0.2, p=2):
    Zt = _as_tensor(Z, device=DEVICE, dtype=torch.float32)
    Zt = torch.nn.functional.normalize(Zt, dim=1)
    yt = _as_tensor(y, device=DEVICE, dtype=torch.long)
    diffs = Zt[:, None, :] - Zt[None, :, :]
    if p == 2:
        D = torch.sqrt((diffs ** 2).sum(-1) + 1e-12)
    else:
        D = torch.norm(diffs, p=p, dim=-1)
    same = (yt[:, None] == yt[None, :]); diff = ~same
    pos = D.clone(); pos[~same] = -1e9
    eye_mask = torch.eye(pos.size(0), dtype=torch.bool, device=pos.device)
    pos[eye_mask] = -1e9
    hardest_pos, _ = pos.max(dim=1)
    neg = D.clone(); neg[~diff] = 1e9
    hardest_neg, _ = neg.min(dim=1)
    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    return float(loss.mean().item())

# ----------------- Reconstruction loss -----------------
def _get_decoder_call(encoder_obj):
    for name in ("decode", "forward_decoder", "decode_from_latent"):
        fn = getattr(encoder_obj, name, None)
        if callable(fn): return fn
    return None

@torch.no_grad()
def reconstruction_loss_mse(encoder_obj, X, bs=512):
    dec = _get_decoder_call(encoder_obj)
    if dec is None: return None
    n = X.shape[0]
    se_sum, n_tot = 0.0, 0
    for i in range(0, n, bs):
        xb = _as_tensor(X[i:i+bs], device=DEVICE, dtype=torch.float32)
        zb = encoder_obj.encode(xb)
        zb = _as_tensor(zb, device=DEVICE, dtype=torch.float32)
        xh = dec(zb)
        xh = _as_tensor(xh, device=DEVICE, dtype=torch.float32)
        se_sum += torch.sum((xb - xh) ** 2).item()
        n_tot += xb.numel()
    return se_sum / max(n_tot, 1)

# ----------------- Confusion + Top-5 misclassified -----------------
def get_topk_misclassified(cm: np.ndarray, classes, k: int = 5) -> pd.DataFrame:
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat_idx = np.argsort(cm_off, axis=None)[::-1]
    r, c = np.unravel_index(flat_idx, cm_off.shape)
    out = []
    for i in range(len(r)):
        cnt = int(cm_off[r[i], c[i]])
        if cnt <= 0: continue
        out.append((classes[r[i]], classes[c[i]], cnt))
        if len(out) >= k: break
    return pd.DataFrame(out, columns=["True_Label", "Predicted_Label", "Count"])

def plot_confusion_matrix(y_true, y_pred, classes, title, out_path=None, topk_path=None):
    y_true = _as_numpy(y_true); y_pred = _as_numpy(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax, xticks_rotation=90, values_format='d', colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    plt.close(fig)

    if topk_path:
        top5_df = get_topk_misclassified(cm, np.array(classes), k=5)
        top5_df.to_csv(topk_path, index=False)
    return cm
