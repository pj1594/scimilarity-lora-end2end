import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

@torch.no_grad()
def embed_in_batches(encoder_obj, X, bs=1024):
    outs = []
    n = X.shape[0]
    for i in range(0, n, bs):
        xb = _as_tensor(X[i:i+bs], device=DEVICE, dtype=torch.float32)
        zb = encoder_obj.encode(xb)
        outs.append(_as_tensor(zb, device=DEVICE, dtype=torch.float32).detach().cpu().numpy())
    return np.vstack(outs)

def normalize_embeddings(Z):
    Zt = _as_tensor(Z, device=DEVICE, dtype=torch.float32)
    Zt = torch.nn.functional.normalize(Zt, dim=1)
    return Zt.detach().cpu().numpy()
