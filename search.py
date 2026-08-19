"""D2 超参数搜索与架构对比。

在固定真实数据、固定划分（seed）上，比较：
1. 多数类基线（不读图像）
2. 全连接神经网络 MLP（打平像素，忽略空间结构）
3. 起点 SmallCNN（2 层卷积 8/16）
4. 一组不同层数 / 卷积核数量 / 卷积核大小的 CNN 配置

输出 `runs/search-results.json` 与 `runs/search-accuracy.png`，
并在终端打印对比表，用于展示“超参数搜索带来的性能提升过程”。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from models import CNN, MLP, SmallCNN
from train import (
    balanced_split_indices,
    confusion_counts,
    evaluate_cnn,
    load_real_dataset,
    majority_baseline,
    set_seed,
    train_epoch,
)


def train_and_evaluate(
    model: nn.Module,
    dataset,
    train_indices: list[int],
    test_indices: list[int],
    batch_size: int,
    epochs: int,
    lr: float = 0.001,
) -> tuple[dict[str, object], list[float]]:
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = [
        train_epoch(model, train_loader, optimizer, loss_fn) for _ in range(epochs)
    ]
    metrics, _ = evaluate_cnn(model, test_loader, dataset, test_indices)
    return metrics, losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = load_real_dataset(args.data)
    train_indices, test_indices = balanced_split_indices(
        dataset.targets, args.max_per_class, args.seed
    )

    # 需要搜索 / 对比的配置：(名称, 模型工厂, 训练轮数, 说明)
    # 使用工厂函数，在每次训练前用同一个 seed 重新构建模型并重置随机状态，
    # 保证每个配置：相同的权重初始化、相同的训练数据 shuffle，比较公平且可复现。
    configs: list[tuple[str, object, int, str]] = [
        (
            "MLP-256/128",
            lambda: MLP(hidden_units=(256, 128)),
            8,
            "全连接网络：打平 3*64*64 像素，忽略 2D 空间结构",
        ),
        (
            "CNN-2conv-8/16 (SmallCNN base)",
            lambda: SmallCNN(),
            2,
            "课程给定 SmallCNN，2 层卷积、每层 8/16 个核、核 3x3",
        ),
        (
            "CNN-3conv-16/32/64 k3",
            lambda: CNN(filters=(16, 32, 64), kernel_size=3),
            4,
            "3 层卷积，核数量 16/32/64，核 3x3",
        ),
        (
            "CNN-3conv-32/64/128 k3",
            lambda: CNN(filters=(32, 64, 128), kernel_size=3),
            4,
            "3 层卷积，核数量 32/64/128，核 3x3",
        ),
        (
            "CNN-4conv-32/64/128/256 k3",
            lambda: CNN(filters=(32, 64, 128, 256), kernel_size=3),
            4,
            "4 层卷积，核数量 32/64/128/256，核 3x3",
        ),
        (
            "CNN-3conv-32/64/128 k5",
            lambda: CNN(filters=(32, 64, 128), kernel_size=5),
            4,
            "3 层卷积，核数量 32/64/128，核 5x5",
        ),
    ]

    results: list[dict[str, object]] = []

    # 1) 多数类基线
    started = time.perf_counter()
    base_metrics, _ = majority_baseline(dataset, train_indices, test_indices)
    results.append(
        {
            "name": "baseline-majority",
            "description": "永远猜训练集多数类，不读取图像内容",
            "epochs": 0,
            "accuracy": base_metrics["accuracy"],
            "crack_recall": base_metrics["crack_recall"],
            "crack_precision": base_metrics["crack_precision"],
            "false_negative_cracks": base_metrics["false_negative_cracks"],
            "confusion_matrix": base_metrics["confusion_matrix_labels_no_crack_crack"],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )

    # 2) 各候选配置
    for name, make_model, epochs, description in configs:
        set_seed(args.seed)  # 每个配置用同一 seed 重新初始化权重与 shuffle
        model = make_model()
        started = time.perf_counter()
        metrics, losses = train_and_evaluate(
            model, dataset, train_indices, test_indices, args.batch_size, epochs
        )
        results.append(
            {
                "name": name,
                "description": description,
                "epochs": epochs,
                "accuracy": metrics["accuracy"],
                "crack_recall": metrics["crack_recall"],
                "crack_precision": metrics["crack_precision"],
                "false_negative_cracks": metrics["false_negative_cracks"],
                "confusion_matrix": metrics["confusion_matrix_labels_no_crack_crack"],
                "train_loss": losses,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    # 保存 JSON
    output = Path("runs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "search-results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # 终端对比表
    print("\n=== D2 架构对比与超参数搜索结果（同数据同划分）===")
    print(
        f"{'配置':32} {'acc':>7} {'recall':>7} {'prec':>7} {'漏检':>5} {'轮数':>4}"
    )
    for row in results:
        print(
            f"{row['name']:32} "
            f"{row['accuracy']:7.4f} "
            f"{row['crack_recall']:7.4f} "
            f"{row['crack_precision']:7.4f} "
            f"{row['false_negative_cracks']:5d} "
            f"{row['epochs']:4d}"
        )

    # 保存准确率条形图
    names = [row["name"] for row in results]
    accs = [row["accuracy"] for row in results]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, accs, color="tab:blue")
    axis.set_ylabel("accuracy")
    axis.set_title("D2 model comparison: majority baseline / MLP / SmallCNN / searched CNN")
    axis.set_ylim(0.0, 1.0)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "search-accuracy.png", dpi=120)
    plt.close(figure)

    best = max((r for r in results if r["epochs"] > 0), key=lambda r: r["accuracy"])
    print(f"\n最佳候选：{best['name']}（accuracy={best['accuracy']:.4f}）")
    print("结果已写入 runs/search-results.json 与 runs/search-accuracy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
