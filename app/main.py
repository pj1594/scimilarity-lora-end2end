import torch
from app.eval_runner import evaluate_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assumes: encoder (LoRA), encoder_base, head, head_base, Xte, yte, le are in memory (e.g., Colab)
if __name__ == "__main__":
    classes = list(le.classes_)
    print("Running comparative evaluation ...")
    df_summary = evaluate_models(
        encoder_lora=encoder,
        encoder_base=encoder_base,
        head=head,
        head_base=head_base,
        Xte=Xte,
        yte=yte,
        classes=classes,
        artifacts_dir="artifacts"
    )
    print("\\n✅ Evaluation complete. Summary:")
    print(df_summary)
