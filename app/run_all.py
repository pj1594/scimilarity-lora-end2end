"""
run_all.py — Overall pipeline
  1) load encoders + LoRA head
  2) load AnnData
  3) run eval
  4) artifacts → ./artifacts/
"""

import os
import argparse
import torch
import anndata as ad
import pandas as pd

from scimilarity.cell_embedding import CellEmbedding
from main import main


def load_lora_head(path, emb_dim, n_classes):
    import torch.nn as nn
    state = torch.load(path, map_location="cpu")
    try:
        class MLP(nn.Module):
            def __init__(self, d, c, p=0.25):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p),
                    nn.Linear(1024, c)
                )
            def forward(self, x):
                return self.net(x)

        h = MLP(emb_dim, n_classes)
        h.load_state_dict(state)
        return h
    except:
        h = nn.Linear(emb_dim, n_classes)
        h.load_state_dict(state)
        return h


def infer_emb_dim(encoder, n_genes):
    import torch
    with torch.no_grad():
        z = encoder.get_embeddings(torch.zeros(1, n_genes).float())
        if isinstance(z, torch.Tensor):
            return z.shape[1]
        return z.shape[-1]


def get_model_genes(model_dir):
    p = os.path.join(model_dir, "gene_order.tsv")
    g = pd.read_csv(p, sep="\t", header=None)[0].astype(str).str.upper().tolist()
    return g


def get_classes(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5ad", required=True)
    ap.add_argument("--val_h5ad",   required=True)
    ap.add_argument("--model_dir",  default="scimilarity_model_v1_1")
    ap.add_argument("--head_path",  default="models/lora/linear_head.pt")
    ap.add_argument("--classes",    default="models/lora/label_classes.txt")
    ap.add_argument("--stats",      default=None)
    ap.add_argument("--out",        default="artifacts")
    ap.add_argument("--label_key",  default="cell_type")
    args = ap.parse_args()

    train_file = args.train_h5ad
    val_file   = args.val_h5ad

    encoder_lora = CellEmbedding(model_path=args.model_dir)
    encoder_base = CellEmbedding(model_path=args.model_dir)   # no LoRA patches applied

    model_genes  = get_model_genes(args.model_dir)
    classes      = get_classes(args.classes)
    n_classes    = len(classes)

    emb_dim = infer_emb_dim(encoder_lora, len(model_genes))
    head_lora = load_lora_head(args.head_path, emb_dim, n_classes)

    print(">> Running evaluation")
    main(
        train_h5ad=train_file,
        val_h5ad=val_file,
        encoder_lora=encoder_lora,
        encoder_base=encoder_base,
        head_lora=head_lora,
        model_genes=model_genes,
        label_key=args.label_key,
        stats_path=args.stats,
        out_dir=args.out
    )


if __name__ == "__main__":
    run()
