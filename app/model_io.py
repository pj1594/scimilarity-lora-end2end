import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------
# tensor / numpy helpers
# --------------------
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


# --------------------
# symbol cleaning
# --------------------
def ensure_symbols(adata):
    """
    Standardize adata.var['gene_symbol'] column.
    Drops NA + duplicates (first occurrence kept).
    """
    ad = adata.copy()
    for cand in ["gene_symbol", "feature_name", "gene_name", "symbol", "name", "SYMBOL"]:
        if cand in ad.var.columns:
            ad.var["gene_symbol"] = ad.var[cand].astype(str).str.upper()
            break
    else:
        ad.var["gene_symbol"] = ad.var_names.astype(str).str.upper()

    keep = ad.var["gene_symbol"].notna() & (ad.var["gene_symbol"] != "")
    ad = ad[:, keep].copy()
    dup = ad.var["gene_symbol"].duplicated(keep="first")
    if dup.any():
        ad = ad[:, ~dup].copy()
    ad.var["gene_symbol"] = ad.var["gene_symbol"].astype(str).str.upper()
    return ad


# --------------------
# model-aligned expression → tensor
# --------------------
def to_model_matrix(adata, model_genes):
    """
    Returns (n_cells, n_model_genes) float32 np.ndarray
    Aligned to SCimilarity (MODEL_GENES).
    """
    ad = ensure_symbols(adata)
    syms = pd.Index(ad.var["gene_symbol"])
    common = syms.intersection(model_genes)

    if len(common) == 0:
        raise ValueError("No overlapping genes with model gene list")

    Xc = ad[:, syms.get_indexer(common)].X
    Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc, dtype=np.float32)

    out = np.zeros((ad.n_obs, len(model_genes)), dtype=np.float32)
    out[:, model_genes.get_indexer(common)] = Xc
    return out


# --------------------
# normalization
# --------------------
def libsize_log1p(X):
    """
    Library-size normalize → log1p
    X: numpy or tensor (cells × genes)
    """
    Xt = _as_tensor(X, device=DEVICE, dtype=torch.float32)
    lib = Xt.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.log1p(Xt * (1e4 / lib))


def zscore_like_train(X, mean_g=None, std_g=None):
    """
    Z-score using train statistics if provided.
    """
    if mean_g is None or std_g is None:
        return X
    Xt = _as_tensor(X, device=DEVICE, dtype=torch.float32)
    return (Xt - mean_g.to(DEVICE)) / std_g.to(DEVICE).clamp_min(1e-6)


# --------------------
# embedding
# --------------------
@torch.no_grad()
def embed_in_batches(encoder_obj, X, bs=1024):
    """
    encoder_obj must expose .encode(x) or .get_embeddings(x)
    X: numpy or tensor
    returns numpy embeddings
    """
    outs = []
    X_np = _as_numpy(X)
    n = X_np.shape[0]

    for i in range(0, n, bs):
        batch = X_np[i:i+bs]

        # If encoder has get_embeddings(), use that
        if hasattr(encoder_obj, "get_embeddings"):
            zb = encoder_obj.get_embeddings(batch)
            zb = _as_tensor(zb, device=DEVICE)
        else:
            xb = _as_tensor(batch, device=DEVICE)
            zb = encoder_obj.encode(xb)

        outs.append(zb.detach().cpu().numpy())

    return np.vstack(outs)


# --------------------
# embedding normalization
# --------------------
def normalize_embeddings(Z):
    Zt = _as_tensor(Z, device=DEVICE, dtype=torch.float32)
    Zt = torch.nn.functional.normalize(Zt, dim=1)
    return Zt.detach().cpu().numpy()
