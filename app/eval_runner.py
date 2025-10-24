import os
import numpy as np
import pandas as pd
import torch

from app.metrics import (
    compute_indices,
    batch_hard_triplet_loss,
    reconstruction_loss_mse,
    plot_confusion_matrix,  # writes CM; returns cm
    get_topk_misclassified, # for programmatic use if needed
)
from app.model_io import embed_in_batches, normalize_embeddings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def _predict_from_head(head, Z_np):
    logits = head(torch.from_numpy(Z_np).to(DEVICE))
    return logits.argmax(1).cpu().numpy()

def _evaluate_single_model(
    tag: str,
    encoder_obj,
    head,
    Xte,
    yte,
    classes,
    artifacts_dir: str,
):

    Z = normalize_embeddings(embed_in_batches(encoder_obj, Xte))
    y_pred = _predict_from_head(head, Z)

    # Metrics
    metrics = {
        "model": tag,
        **compute_indices(yte, y_pred),
        "triplet_loss": batch_hard_triplet_loss(Z, yte),
        "reconstruction_mse": reconstruction_loss_mse(encoder_obj, Xte),
    }

    cm_path = os.path.join(artifacts_dir, f"cm_{tag}.png")
    top5_path = os.path.join(artifacts_dir, f"top5_misclassified_{tag}.csv")
    plot_confusion_matrix(
        y_true=yte,
        y_pred=y_pred,
        classes=classes,
        title=f"{tag.upper()} — Confusion Matrix",
        out_path=cm_path,
        topk_path=top5_path,
    )
    return metrics

def evaluate_models(
    encoder_lora,
    encoder_base,
    head,
    head_base,
    Xte,
    yte,
    classes,
    artifacts_dir: str = "artifacts",
) -> pd.DataFrame:

    os.makedirs(artifacts_dir, exist_ok=True)
    head = head.to(DEVICE).eval()
    head_base = head_base.to(DEVICE).eval()

    lora_metrics = _evaluate_single_model(
        tag="lora",
        encoder_obj=encoder_lora,
        head=head,
        Xte=Xte,
        yte=yte,
        classes=classes,
        artifacts_dir=artifacts_dir,
    )
    base_metrics = _evaluate_single_model(
        tag="baseline",
        encoder_obj=encoder_base,
        head=head_base,
        Xte=Xte,
        yte=yte,
        classes=classes,
        artifacts_dir=artifacts_dir,
    )

    df = pd.DataFrame([lora_metrics, base_metrics])
    df_path = os.path.join(artifacts_dir, "summary.csv")
    df.to_csv(df_path, index=False)

    print("\\n✅ Evaluation complete.")
    print(df.to_string(index=False))
    print(f"Artifacts: {os.path.abspath(artifacts_dir)}")
    return df

if __name__ == "__main__":
    # Expect: encoder (LoRA), encoder_base, head, head_base, Xte, yte, le
    missing = [n for n in ["encoder", "encoder_base", "head", "head_base", "Xte", "yte", "le"] if n not in globals()]
    if missing:
        raise RuntimeError(f"Missing in globals: {missing}")
    classes = list(le.classes_)
    evaluate_models(
        encoder_lora=encoder,
        encoder_base=encoder_base,
        head=head,
        head_base=head_base,
        Xte=Xte,
        yte=yte,
        classes=classes,
        artifacts_dir="artifacts",
    )
