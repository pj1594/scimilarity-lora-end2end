import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from collections import defaultdict

from model_io import (
    _as_tensor, _as_numpy,
    embed_in_batches, normalize_embeddings,
    to_model_matrix, libsize_log1p, zscore_like_train
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Triplet Loss
# -----------------------------
@torch.no_grad()
def batch_hard_triplet_loss(Z, y, margin=0.2, p=2):
    Zt = _as_tensor(Z, device=DEVICE)
    Zt = torch.nn.functional.normalize(Zt, dim=1)
    yt = _as_tensor(y, device=DEVICE, dtype=torch.long)

    diffs = Zt[:, None, :] - Zt[None, :, :]
    if p == 2:
        D = torch.sqrt((diffs ** 2).sum(-1) + 1e-12)
    else:
        D = torch.norm(diffs, p=p, dim=-1)

    same = (yt[:, None] == yt[None, :])
    diff = ~same

    pos = D.clone()
    pos[~same] = -1e9
    pos[torch.eye(pos.size(0), dtype=torch.bool, device=pos.device)] = -1e9
    hardest_pos, _ = pos.max(dim=1)

    neg = D.clone()
    neg[~diff] = 1e9
    hardest_neg, _ = neg.min(dim=1)

    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    return float(loss.mean().item())


# -----------------------------
# Reconstruction Loss
# -----------------------------
def _get_decoder_call(enc):
    for name in ("decode", "forward_decoder", "decode_from_latent"):
        fn = getattr(enc, name, None)
        if callable(fn):
            return fn
    return None

@torch.no_grad()
def reconstruction_loss(enc, X, bs=256):
    dec = _get_decoder_call(enc)
    if dec is None:
        return None
    X_np = _as_numpy(X)
    total_se, total_n = 0.0, 0
    for i in range(0, X_np.shape[0], bs):
        xb = _as_tensor(X_np[i:i+bs], dtype=torch.float32, device=DEVICE)
        zb = enc.encode(xb)
        xhat = dec(zb)
        xhat = _as_tensor(xhat, dtype=torch.float32, device=DEVICE)
        se = torch.sum((xb - xhat) ** 2).item()
        total_se += se
        total_n += xb.numel()
    return total_se / max(total_n, 1)


# -----------------------------
# Per-class correct diagnostic
# -----------------------------
def per_class_hits(y_true, y_pred, classes, top_probs=None):
    hits = defaultdict(list)
    for yi, pi, prob in zip(y_true, y_pred, top_probs or [None]*len(y_pred)):
        if yi == pi:
            hits[yi].append(prob)
    cls_names = list(classes)
    correct = {cls_names[k]: len(v) for k, v in hits.items()}
    return correct


# -----------------------------
# Main Eval
# -----------------------------
def evaluate(
    encoder_lora,
    encoder_base,
    head_lora,
    train_adata,
    val_adata,
    model_genes,
    label_key="cell_type",
    mean_g=None,
    std_g=None,
    out_dir="artifacts"
):
    os.makedirs(out_dir, exist_ok=True)

    # ─────────────────────────────
    # Align → Normalize
    # ─────────────────────────────
    Xtr = to_model_matrix(train_adata, model_genes)
    Xva = to_model_matrix(val_adata, model_genes)

    Xtr = libsize_log1p(Xtr)
    Xva = libsize_log1p(Xva)

    if mean_g is not None and std_g is not None:
        Xtr = zscore_like_train(Xtr, mean_g, std_g)
        Xva = zscore_like_train(Xva, mean_g, std_g)

    ytr = train_adata.obs[label_key].astype("category")
    yva = val_adata.obs[label_key].astype("category")
    classes = list(ytr.cat.categories)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(classes)

    ytr_enc = le.transform(ytr)
    yva_enc = le.transform(yva)


    # ─────────────────────────────
    # Embeddings
    # ─────────────────────────────
    Ztr_lora = embed_in_batches(encoder_lora, Xtr)
    Zva_lora = embed_in_batches(encoder_lora, Xva)

    Ztr_base = embed_in_batches(encoder_base, Xtr)
    Zva_base = embed_in_batches(encoder_base, Xva)

    Ztr_lora = normalize_embeddings(Ztr_lora)
    Zva_lora = normalize_embeddings(Zva_lora)
    Ztr_base = normalize_embeddings(Ztr_base)
    Zva_base = normalize_embeddings(Zva_base)


    # ─────────────────────────────
    # Predict
    # ─────────────────────────────
    head_lora.eval()

    logits_va_lora = head_lora(_as_tensor(Zva_lora)).cpu().detach().numpy()
    preds_lora = logits_va_lora.argmax(axis=1)

    # Build baseline linear head
    head_base = torch.nn.Linear(Ztr_base.shape[1], len(classes)).to(DEVICE)
    head_base.train()
    opt = torch.optim.Adam(head_base.parameters(), lr=1e-3)

    Ztr_base_t = _as_tensor(Ztr_base)
    ytr_t = _as_tensor(ytr_enc, dtype=torch.long)
    for _ in range(3):
        opt.zero_grad()
        logits = head_base(Ztr_base_t)
        loss = torch.nn.functional.cross_entropy(logits, ytr_t)
        loss.backward()
        opt.step()

    head_base.eval()
    logits_va_base = head_base(_as_tensor(Zva_base)).cpu().detach().numpy()
    preds_base = logits_va_base.argmax(axis=1)


    # ─────────────────────────────
    # Metrics
    # ─────────────────────────────
    top1_lora = accuracy_score(yva_enc, preds_lora)
    top3_lora = top_k_accuracy_score(yva_enc, logits_va_lora, k=3)

    top1_base = accuracy_score(yva_enc, preds_base)
    top3_base = top_k_accuracy_score(yva_enc, logits_va_base, k=3)

    macro_f1_lora = f1_score(yva_enc, preds_lora, average="macro")
    macro_f1_base = f1_score(yva_enc, preds_base, average="macro")

    hits_lora = per_class_hits(yva_enc, preds_lora, classes)
    hits_base = per_class_hits(yva_enc, preds_base, classes)

    trip_lora = batch_hard_triplet_loss(Zva_lora, yva_enc)
    trip_base = batch_hard_triplet_loss(Zva_base, yva_enc)

    recon_lora = reconstruction_loss(encoder_lora, Xva)
    recon_base = reconstruction_loss(encoder_base, Xva)

    # Print summary
    print("\n=== FINAL METRICS ===")
    print(f"LORA   : Top1={top1_lora:.4f} | Top3={top3_lora:.4f} | MacroF1={macro_f1_lora:.4f} | Triplet={trip_lora:.4f} | Recon={recon_lora}")
    print(f"BASE   : Top1={top1_base:.4f} | Top3={top3_base:.4f} | MacroF1={macro_f1_base:.4f} | Triplet={trip_base:.4f} | Recon={recon_base}")

    # Confusion matrices
    cm_lora = confusion_matrix(yva_enc, preds_lora)
    cm_base = confusion_matrix(yva_enc, preds_base)

    # Save
    pd.DataFrame({
        "model": ["lora","baseline"],
        "top1":[top1_lora, top1_base],
        "top3":[top3_lora, top3_base],
        "macroF1":[macro_f1_lora, macro_f1_base],
        "triplet":[trip_lora, trip_base],
        "recon":[recon_lora, recon_base],
        "classes_correct":[len(hits_lora), len(hits_base)]
    }).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    return {
        "top1": top1_lora,
        "top3": top3_lora,
        "macroF1": macro_f1_lora,
        "triplet": trip_lora,
        "reconstruction": recon_lora,
        "hits_by_class": hits_lora,
        "confusion_matrix": cm_lora,
        "summary_csv": os.path.join(out_dir, "summary.csv")
    }
