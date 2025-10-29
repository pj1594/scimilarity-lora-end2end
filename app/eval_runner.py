"""
eval_runner.py — Full evaluation for SCimilarity LoRA
Includes:
    Baseline vs LoRA
    Triplet Loss
    Recon Loss (if decoder exists)
    Per-class F1
    Evidence: #classes with correct predictions
    Confusion Matrix export
"""

import os, numpy as np, pandas as pd, torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("artifacts", exist_ok=True)

# Required globals: encoder_base, encoder (LoRA), head_base, head, Xte, yte, le
for must in ["encoder_base","encoder","head_base","head","Xte","yte","le"]:
    assert must in globals(), f"{must} missing."

classes = list(le.classes_)
N = len(yte)

# -------------- L2 Normalize ----------------
def l2norm(z):
    z = torch.as_tensor(z, device=DEVICE, dtype=torch.float32)
    return torch.nn.functional.normalize(z, dim=1)

# -------------- Triplet Loss ----------------
def triplet(Z, y, margin=0.2):
    Z = l2norm(Z)
    y = torch.as_tensor(y, device=DEVICE)
    diff = torch.cdist(Z,Z,p=2)
    same = (y[:,None]==y[None,:])
    pos = diff.clone(); pos[~same] = -1e9
    eye=torch.eye(len(Z),device=DEVICE).bool()
    pos[eye] = -1e9
    hardest_pos = pos.max(1).values
    neg = diff.clone(); neg[same] = 1e9
    hardest_neg = neg.min(1).values
    loss = (hardest_pos-hardest_neg+margin).clamp(min=0)
    return float(loss.mean().item())

# -------------- Reconstruction Loss (if available) --------------
def recon(encoder_obj, X):
    dec=None
    for nm in ("decode","forward_decoder","decode_from_latent"):
        if hasattr(encoder_obj,nm):
            dec=getattr(encoder_obj,nm)
            break
    if dec is None:
        return None
    X = torch.as_tensor(X,device=DEVICE,dtype=torch.float32)
    with torch.no_grad():
        z  = encoder_obj.encode(X)
        z  = torch.as_tensor(z,device=DEVICE,dtype=torch.float32)
        Xh = dec(z)
        return float(((X-Xh)**2).mean().item())

# -------------- Embedding ----------------
@torch.no_grad()
def embed(enc,X,bs=1024):
    outs=[]
    X=np.asarray(X)
    for i in range(0,len(X),bs):
        xb = X[i:i+bs]
        zb = enc.get_embeddings(xb)
        outs.append(zb)
    return np.vstack(outs)

print("Embedding: baseline, LoRA ...")
Zb = embed(encoder_base, Xte)
Zl = embed(encoder,      Xte)
Zb = l2norm(Zb); Zl = l2norm(Zl)

# -------------- Predict ----------------
with torch.no_grad():
    pb = head_base(torch.as_tensor(Zb,device=DEVICE)).argmax(1).cpu().numpy()
    pl = head(torch.as_tensor(Zl,device=DEVICE)).argmax(1).cpu().numpy()

# -------------- Metrics ----------------
acc_b = accuracy_score(yte,pb)
acc_l = accuracy_score(yte,pl)
f1_b  = f1_score(yte,pb,average="macro")
f1_l  = f1_score(yte,pl,average="macro")
tri_b = triplet(Zb,yte)
tri_l = triplet(Zl,yte)
rec_b = recon(encoder_base,Xte)
rec_l = recon(encoder,Xte)

print("\n=== SUMMARY ===")
print(f"Baseline  : ACC={acc_b:.4f} | MF1={f1_b:.4f} | triplet={tri_b:.4f} | recon={rec_b}")
print(f"LoRA      : ACC={acc_l:.4f} | MF1={f1_l:.4f} | triplet={tri_l:.4f} | recon={rec_l}")

# -------------- Classification Report --------------
print("\n--- LoRA Report ---")
print(classification_report(yte,pl,target_names=classes))

# -------------- Per-class evidence: #classes hit --------------
hits = defaultdict(int)
for yi,pi in zip(yte,pl):
    if yi==pi: hits[yi]+=1

hit_names=[]
for k,v in hits.items():
    if v>0: hit_names.append(classes[k])

print(f"\n[EVIDENCE] Correct predictions for {len(hit_names)} / {len(classes)} classes:")
print(hit_names)

# -------------- Confusion Matrix --------------
cm = confusion_matrix(yte,pl)
pd.DataFrame(cm, index=classes, columns=classes).to_csv("artifacts/confusion_matrix.csv")
print("Saved: artifacts/confusion_matrix.csv")

summary = pd.DataFrame([
    dict(model="baseline",acc=acc_b,f1=f1_b,triplet=tri_b,recon=rec_b),
    dict(model="lora",    acc=acc_l,f1=f1_l,triplet=tri_l,recon=rec_l),
])
summary.to_csv("artifacts/summary.csv",index=False)
print("Saved: artifacts/summary.csv")
print("\nDone.")
