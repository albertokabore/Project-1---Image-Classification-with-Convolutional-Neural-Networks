"""Hyperparameter tuning experiment for the regularized CNN. Run: python tune.py."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from train import CLASSES, CNN, DEFAULT_DATA, DEFAULT_DROPOUT, Images, collect, evaluate, train_variant

LEARNING_RATES = [0.0003, 0.001, 0.003]
DROPOUT_RATES = [0.1, DEFAULT_DROPOUT, 0.5]
EPOCHS = 3
BATCH_SIZE = 64
IMAGE_SIZE = 48


def main():
    output = Path("results")
    baseline_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    prior_val_accuracy = baseline_metrics["validation_accuracy"][baseline_metrics["selected_model"]]

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = collect(DEFAULT_DATA)
    labels = [label for _, label in samples]
    train_val, test = train_test_split(samples, test_size=0.15, stratify=labels, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, stratify=[label for _, label in train_val], random_state=42)
    train_loader = DataLoader(Images(train, IMAGE_SIZE, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(Images(val, IMAGE_SIZE), batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(Images(test, IMAGE_SIZE), batch_size=BATCH_SIZE, num_workers=0)
    print(f"Device={device}; train={len(train)}, val={len(val)}, test={len(test)}", flush=True)

    start = time.time()

    lr_results = {}
    for lr in LEARNING_RATES:
        _, history, best_val = train_variant(f"lr={lr}", True, (train_loader, val_loader), EPOCHS, lr, device)
        lr_results[lr] = {"best_val_accuracy": best_val, "history": history}
        print(f"lr={lr}: best val accuracy={best_val:.4f}", flush=True)
    best_lr = max(lr_results, key=lambda k: lr_results[k]["best_val_accuracy"])

    dropout_results = {}
    best_model, best_history, best_dropout_val = None, None, -1
    for dropout in DROPOUT_RATES:
        model, history, best_val = train_variant(f"dropout={dropout}", True, (train_loader, val_loader), EPOCHS, best_lr, device, dropout=dropout)
        dropout_results[dropout] = {"best_val_accuracy": best_val, "history": history}
        print(f"lr={best_lr}, dropout={dropout}: best val accuracy={best_val:.4f}", flush=True)
        if best_val > best_dropout_val:
            best_dropout_val, best_model, best_history = best_val, model, history

    best_dropout = max(dropout_results, key=lambda k: dropout_results[k]["best_val_accuracy"])
    tuned_val_accuracy = dropout_results[best_dropout]["best_val_accuracy"]

    result = {
        "learning_rates_tried": LEARNING_RATES,
        "dropout_rates_tried": DROPOUT_RATES,
        "epochs_per_run": EPOCHS,
        "learning_rate_sweep": {str(k): v["best_val_accuracy"] for k, v in lr_results.items()},
        "dropout_sweep_at_best_lr": {str(k): v["best_val_accuracy"] for k, v in dropout_results.items()},
        "best_learning_rate": best_lr,
        "best_dropout": best_dropout,
        "tuned_validation_accuracy": tuned_val_accuracy,
        "prior_validation_accuracy": prior_val_accuracy,
        "improved_over_prior": tuned_val_accuracy > prior_val_accuracy,
    }

    if tuned_val_accuracy > prior_val_accuracy:
        test_loss, test_accuracy, actual, predicted = evaluate(best_model, test_loader, nn.CrossEntropyLoss(), device)
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="macro", zero_division=0)
        matrix = confusion_matrix(actual, predicted, labels=list(range(len(CLASSES))))
        report = classification_report(actual, predicted, target_names=CLASSES, output_dict=True, zero_division=0)
        result["tuned_test"] = {
            "loss": test_loss, "accuracy": test_accuracy, "macro_precision": precision,
            "macro_recall": recall, "macro_f1": f1, "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }
        torch.save(best_model.state_dict(), output / "tuned.pt")
        print(f"Tuned model improves on prior best ({tuned_val_accuracy:.4f} > {prior_val_accuracy:.4f}); test accuracy={test_accuracy:.4f}", flush=True)
    else:
        print(f"Tuned model does not exceed prior best val accuracy ({tuned_val_accuracy:.4f} <= {prior_val_accuracy:.4f}); keeping original selected model.", flush=True)

    result["elapsed_seconds"] = time.time() - start
    (output / "tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([str(lr) for lr in LEARNING_RATES], [lr_results[lr]["best_val_accuracy"] for lr in LEARNING_RATES])
    axes[0].set(xlabel="Learning rate", ylabel="Best validation accuracy", title=f"Learning rate sweep ({EPOCHS} epochs)")
    axes[1].bar([str(d) for d in DROPOUT_RATES], [dropout_results[d]["best_val_accuracy"] for d in DROPOUT_RATES])
    axes[1].set(xlabel="Dropout rate", ylabel="Best validation accuracy", title=f"Dropout sweep at lr={best_lr}")
    fig.tight_layout()
    fig.savefig(output / "tuning_curves.png", dpi=160)
    plt.close(fig)
    print(f"Saved results/tuning.json and results/tuning_curves.png ({result['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
