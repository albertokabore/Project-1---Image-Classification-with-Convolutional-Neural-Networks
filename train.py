"""Reproducible Natural Images CNN comparison. Run: python train.py --epochs 4."""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASSES = ["airplane", "car", "cat", "dog", "flower", "fruit", "motorbike", "person"]
DEFAULT_DATA = Path(r"C:\Users\alber\OneDrive\Documents\archive\natural_images")
DEFAULT_DROPOUT = 0.3


class Images(Dataset):
    def __init__(self, samples, size, augment=False):
        self.samples, self.size, self.augment = samples, size, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), label


class CNN(nn.Module):
    def __init__(self, regularized=False, dropout=DEFAULT_DROPOUT):
        super().__init__()
        layers = []
        channels = 3
        for width in (16, 32, 64):
            layers += [nn.Conv2d(channels, width, 3, padding=1)]
            if regularized:
                layers += [nn.BatchNorm2d(width)]
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(dropout) if regularized else nn.Identity(), nn.Linear(64, len(CLASSES)))

    def forward(self, x):
        return self.classifier(self.features(x))


def collect(root):
    samples = []
    for label, name in enumerate(CLASSES):
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {directory}")
        samples.extend((p, label) for p in sorted(directory.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not samples:
        raise ValueError(f"No images found in {root}")
    return samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, targets, predictions = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(targets, predictions), targets, predictions


def train_variant(name, regularized, loaders, epochs, lr, device, dropout=DEFAULT_DROPOUT):
    torch.manual_seed(42)
    model = CNN(regularized, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}
    best_accuracy, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        loss_sum, correct = 0.0, 0
        for images, labels in loaders[0]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        val_loss, val_accuracy, _, _ = evaluate(model, loaders[1], criterion, device)
        for key, value in zip(history, (loss_sum / len(loaders[0].dataset), correct / len(loaders[0].dataset), val_loss, val_accuracy)):
            history[key].append(float(value))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{name} epoch {epoch + 1}/{epochs}: train acc={history['train_accuracy'][-1]:.3f}, val acc={val_accuracy:.3f}", flush=True)
    model.load_state_dict(best_state)
    return model, history, best_accuracy


def plot_results(history, matrix, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, values in history.items():
        axes[0].plot(range(1, len(values["train_loss"]) + 1), values["train_loss"], label=f"{name} train")
        axes[0].plot(range(1, len(values["val_loss"]) + 1), values["val_loss"], linestyle="--", label=f"{name} val")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy loss", title="Loss")
    axes[0].legend(fontsize=8)
    for name, values in history.items():
        axes[1].plot(range(1, len(values["train_accuracy"]) + 1), values["train_accuracy"], label=f"{name} train")
        axes[1].plot(range(1, len(values["val_accuracy"]) + 1), values["val_accuracy"], linestyle="--", label=f"{name} val")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1), title="Accuracy")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(8), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(8), CLASSES)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Test confusion matrix")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="white" if matrix[i][j] > matrix.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.image_size < 8:
        parser.error("epochs must be positive and image-size at least 8")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = collect(args.data)
    labels = [label for _, label in samples]
    train_val, test = train_test_split(samples, test_size=0.15, stratify=labels, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, stratify=[label for _, label in train_val], random_state=42)
    args.output.mkdir(parents=True, exist_ok=True)
    loaders = {}
    for name, subset in (("train", train), ("val", val), ("test", test)):
        loaders[name] = DataLoader(Images(subset, args.image_size, augment=name == "train"), batch_size=args.batch_size, shuffle=name == "train", num_workers=0)
    print(f"Device={device}; split: train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    histories, scores = {}, {}
    start = time.time()
    for name, regularized in (("baseline", False), ("regularized", True)):
        model, history, val_accuracy = train_variant(name, regularized, (loaders["train"], loaders["val"]), args.epochs, args.learning_rate, device)
        histories[name] = history
        scores[name] = val_accuracy
        torch.save(model.state_dict(), args.output / f"{name}.pt")
    winner = max(scores, key=scores.get)
    model = CNN(winner == "regularized").to(device)
    model.load_state_dict(torch.load(args.output / f"{winner}.pt", map_location=device, weights_only=True))
    test_loss, test_accuracy, actual, predicted = evaluate(model, loaders["test"], nn.CrossEntropyLoss(), device)
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="macro", zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=list(range(8)))
    report = classification_report(actual, predicted, target_names=CLASSES, output_dict=True, zero_division=0)
    result = {"dataset": str(args.data), "class_counts": {name: labels.count(i) for i, name in enumerate(CLASSES)}, "split": {"train": len(train), "validation": len(val), "test": len(test)}, "seed": 42, "device": str(device), "epochs_per_variant": args.epochs, "image_size": args.image_size, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "dropout": DEFAULT_DROPOUT, "validation_accuracy": scores, "selected_model": winner, "test": {"loss": test_loss, "accuracy": test_accuracy, "macro_precision": precision, "macro_recall": recall, "macro_f1": f1, "classification_report": report, "confusion_matrix": matrix.tolist()}, "history": histories, "elapsed_seconds": time.time() - start}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_results(histories, matrix, args.output)
    print(f"Selected {winner}; test accuracy={test_accuracy:.4f}; macro precision={precision:.4f}; macro recall={recall:.4f}", flush=True)


if __name__ == "__main__":
    main()
