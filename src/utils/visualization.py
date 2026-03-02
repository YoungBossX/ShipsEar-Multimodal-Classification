# src/utils/visualization.py

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

# 主色板（适合折线图、散点图）
ACADEMIC_COLORS = [
    "#2E86AB",   # 钢蓝
    "#E84855",   # 砖红
    "#3BB273",   # 森绿
    "#F18F01",   # 琥珀
    "#7B2D8B",   # 深紫
    "#1B4332",   # 墨绿
    "#C9184A",   # 玫红
    "#0077B6",   # 深蓝
]

# 训练/验证专用颜色
COLOR_TRAIN = "#2E86AB"    # 钢蓝
COLOR_VAL   = "#E84855"    # 砖红
COLOR_BEST  = "#3BB273"    # 森绿

# 柱状图专用（Precision / Recall / F1）
COLOR_PRECISION = "#2E86AB"
COLOR_RECALL    = "#E84855"
COLOR_F1        = "#3BB273"

# 全局字体与样式设置
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def plot_loss_accuracy_curves(
    train_losses, val_losses, train_accs, val_accs, save_dir, logger=None
):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.subplot(2, 1, 2)
    plt.plot(np.array(train_accs) / 100, label="Train Accuracy", color="blue")
    plt.plot(np.array(val_accs) / 100, label="Validation Accuracy", color="red")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "loss_accuracy_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.info("Loss curves saved to %s", save_path)

def plot_confusion_matrix(cm, class_names, title, save_dir, logger=None):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.info("Confusion matrix saved to: %s", save_path)

def plot_roc_curves(labels, probabilities, class_names, save_dir, logger=None):
    """每个类别的 ROC 曲线 + AUC 值（One-vs-Rest）。"""
    num_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ACADEMIC_COLORS[:num_classes]

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"Class {name} (AUC = {roc_auc:.3f})")

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([
        roc_curve(labels_bin[:, i], probabilities[:, i])[0]
        for i in range(num_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="#333333", lw=2.5, linestyle="--",
            label=f"Macro-average (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info("ROC curves saved to %s", save_path)


def plot_per_class_metrics(
    precision, recall, f1, class_names, save_dir, logger=None
):
    """每个类别的 Precision / Recall / F1 分组柱状图。"""
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision * 100, width, label="Precision",
                   color=COLOR_PRECISION, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x,         recall * 100,    width, label="Recall",
                   color=COLOR_RECALL,    edgecolor="white", linewidth=0.8)
    bars3 = ax.bar(x + width, f1 * 100,        width, label="F1-Score",
                   color=COLOR_F1,        edgecolor="white", linewidth=0.8)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xlabel("Class")
    ax.set_ylabel("Score (%)")
    ax.set_title("Per-Class Precision / Recall / F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "per_class_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info("Per-class metrics saved to %s", save_path)


def plot_tsne(features, labels, class_names, save_dir, perplexity=30, logger=None):
    """融合特征的 t-SNE 二维投影。"""
    if logger:
        logger.info("Running t-SNE (this may take a while)...")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ACADEMIC_COLORS[:len(class_names)]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=color,
            label=name,
            alpha=0.75,
            s=35,
            marker=markers[i % len(markers)],
            edgecolors="white",
            linewidths=0.4,
            n_jobs=8,
        )

    ax.set_title("t-SNE Visualization of Fusion Features")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", markerscale=1.5, frameon=True, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tsne_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info("t-SNE plot saved to %s", save_path)


def plot_modality_contribution(results: dict, save_dir, logger=None):
    """各模态消融实验对比柱状图。"""
    names = list(results.keys())
    accs  = list(results.values())

    # 最高值用深色突出，其余用浅色
    max_idx = accs.index(max(accs))
    colors = ["#AECDE0" if i != max_idx else "#2E86AB" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, accs, color=colors, edgecolor="#555555", linewidth=0.8, width=0.55)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Modality Contribution (Ablation Study)")
    ax.set_ylim(0, max(accs) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "modality_contribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info("Modality contribution plot saved to %s", save_path)


def plot_moe_expert_activation(routing_weights, class_names, save_dir, logger=None):
    """MoE 专家激活权重热力图（各类别 × 各专家）。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        routing_weights,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"Expert {i+1}" for i in range(routing_weights.shape[1])],
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Activation Weight"},
        ax=ax,
    )
    ax.set_title("MoE Expert Activation Weights per Class")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "moe_expert_activation.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info("MoE expert activation saved to %s", save_path)


def plot_lr_curve(lr_history, save_dir, logger=None):
    """训练过程中学习率的变化曲线。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr_history, color="#7B2D8B", lw=2)
    ax.fill_between(range(len(lr_history)), lr_history, alpha=0.1, color="#7B2D8B")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lr_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info("LR curve saved to %s", save_path)