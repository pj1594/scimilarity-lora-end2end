"""
main.py — Entrypoint: run LoRA vs Baseline evaluation
Relies on:
  • eval_runner.py
  • model_io.py
"""

import os
import anndata as ad
import torch

from eval_runner import evaluate
from model_io import load_mean_std   # optional helper if you saved stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    train_h5ad,
    val_h5ad,
    encoder_lora,
    encoder_base,
    head_lora,
    model_genes,
    label_key="cell_type",
    stats_path=None,
    out_dir="artifacts"
):
    train = ad.read_h5ad(train_h5ad)
    val   = ad.read_h5ad(val_h5ad)

    mean_g, std_g = None, None
    if stats_path is not None and os.path.exists(stats_path):
        mean_g, std_g = load_mean_std(stats_path)

    res = evaluate(
        encoder_lora=encoder_lora,
        encoder_base=encoder_base,
        head_lora=head_lora,
        train_adata=train,
        val_adata=val,
        model_genes=model_genes,
        label_key=label_key,
        mean_g=mean_g,
        std_g=std_g,
        out_dir=out_dir
    )

    print("\nDone. Summary CSV →", res["summary_csv"])
    return res


if __name__ == "__main__":
    raise SystemExit(
        "main.py is designed to be imported & driven by run_all.py. "
        "See run_all.py for argument wiring."
    )
