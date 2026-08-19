"""Train a small CNN on a fixed subset of real concrete crack photographs."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import MLP, SmallCNN


LABELS = ["no_crack", "crack"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_class_root(root: Path) -> Path:
    """Find the real archive folder containing Positive and Negative."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Real crack image folder not found at {root}. "
            "Follow the starter README and extract the Kaggle archive."
        )
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for candidate in candidates:
        if (candidate / "Positive").is_dir() and (candidate / "Negative").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find sibling Positive and Negative folders below {root}."
    )


def count_images(folder: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in folder.rglob("*")
    )


def verify_real_data(root: Path) -> dict[str, object]:
    """Verify the full named Kaggle archive before choosing a real subset."""
    class_root = find_class_root(root)
    counts = {
        "Negative": count_images(class_root / "Negative"),
        "Positive": count_images(class_root / "Positive"),
    }
    if counts != {"Negative": 20000, "Positive": 20000}:
        raise ValueError(
            f"Expected 20,000 images per class but found {counts}. "
            "Check that the complete named archive was extracted."
        )
    return {"class_root": str(class_root), "counts": counts}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_real_dataset(root: Path) -> datasets.ImageFolder:
    class_root = find_class_root(root)
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(class_root, transform=transform)
    expected = {"Negative": 0, "Positive": 1}
    if dataset.class_to_idx != expected:
        raise ValueError(
            f"Expected class mapping {expected}, found {dataset.class_to_idx}"
        )
    return dataset


def balanced_split_indices(
    targets: list[int], max_per_class: int, seed: int
) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    generator = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for class_index in (0, 1):
        indices = by_class[class_index][:]
        generator.shuffle(indices)
        selected = indices[: min(max_per_class, len(indices))]
        split_at = int(len(selected) * 0.75)
        train_indices.extend(selected[:split_at])
        test_indices.extend(selected[split_at:])
    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return train_indices, test_indices


def confusion_counts(truth: list[int], predicted: list[int]) -> dict[str, object]:
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, predicted))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
    total = len(truth)
    return {
        "accuracy": (tn + tp) / total if total else None,
        "crack_precision": tp / (tp + fp) if tp + fp else 0.0,
        "crack_recall": tp / (tp + fn) if tp + fn else 0.0,
        "confusion_matrix_labels_no_crack_crack": [[tn, fp], [fn, tp]],
        "false_negative_cracks": fn,
        "total": total,
    }


def majority_baseline(
    dataset: datasets.ImageFolder,
    train_indices: list[int],
    test_indices: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    majority = Counter(dataset.targets[index] for index in train_indices).most_common(1)[0][0]
    truth = [int(dataset.targets[index]) for index in test_indices]
    predicted = [majority] * len(test_indices)
    errors = [
        {
            "path": dataset.samples[index][0],
            "true": LABELS[truth_value],
            "predicted": LABELS[majority],
        }
        for index, truth_value in zip(test_indices, truth)
        if truth_value != majority
    ][:12]
    return confusion_counts(truth, predicted), errors


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(images)
    return total_loss / len(loader.dataset)


def evaluate_cnn(
    model: nn.Module,
    loader: DataLoader,
    dataset: datasets.ImageFolder,
    test_indices: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            output = model(images).argmax(dim=1)
            truth.extend(int(value) for value in labels.tolist())
            predicted.extend(int(value) for value in output.tolist())
    errors = [
        {
            "path": dataset.samples[index][0],
            "true": LABELS[truth_value],
            "predicted": LABELS[predicted_value],
        }
        for index, truth_value, predicted_value in zip(
            test_indices, truth, predicted
        )
        if truth_value != predicted_value
    ][:12]
    return confusion_counts(truth, predicted), errors


def save_error_grid(errors: list[dict[str, object]], output: Path) -> None:
    shown = errors[:6]
    if not shown:
        return
    figure, axes = plt.subplots(2, 3, figsize=(9, 6))
    for axis in axes.flat:
        axis.axis("off")
    for axis, error in zip(axes.flat, shown):
        axis.imshow(plt.imread(str(error["path"])))
        axis.set_title(f"true={error['true']}\npred={error['predicted']}")
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", choices=("baseline", "cnn", "mlp"), default="baseline")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()

    if args.check_data:
        result = verify_real_data(args.data)
        print("REAL DATA CHECK PASSED")
        print(f"class_root: {result['class_root']}")
        print(f"counts: {result['counts']}")
        return 0

    set_seed(args.seed)
    dataset = load_real_dataset(args.data)
    train_indices, test_indices = balanced_split_indices(
        dataset.targets, args.max_per_class, args.seed
    )
    started = time.perf_counter()

    if args.model == "baseline":
        metrics, errors = majority_baseline(dataset, train_indices, test_indices)
        losses: list[float] = []
    else:
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            Subset(dataset, test_indices),
            batch_size=args.batch_size,
            shuffle=False,
        )
        model: nn.Module = SmallCNN() if args.model == "cnn" else MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        losses = [
            train_epoch(model, train_loader, optimizer, loss_fn)
            for _ in range(args.epochs)
        ]
        metrics, errors = evaluate_cnn(model, test_loader, dataset, test_indices)

    result = {
        "dataset": "Surface Crack Detection",
        "source": "Kaggle arunrk7/surface-crack-detection",
        "model": args.model,
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "train_images": len(train_indices),
        "test_images": len(test_indices),
        "epochs": args.epochs if args.model != "baseline" else 0,
        "train_loss": losses,
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation": metrics,
        "first_errors": errors,
    }
    output = Path("runs") / f"{args.model}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_error_grid(errors, Path("runs") / f"{args.model}-errors.png")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
