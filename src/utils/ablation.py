# -*-coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ======================================================================
# 全局学术配色
# ======================================================================
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

SAVE_DIR = "saved/plots"

# ======================================================================
# 消融实验数据
# ======================================================================

# 实验组标签
GROUPS = [
    "Time",           # A1: 时序分支
    "Spectrogram",    # A2: 倒谱分支
    "Time+Spec",      # B1: 时序+倒谱
    "Time+Sem",       # B2: 时序+语义
    "Spec+Sem",       # B3: 倒谱+语义
    "All (Fusion)",   # C:  全部
]

# 表5：ShipsEar 数据集
SHIPEAR_DATA = {
    "Accuracy":  [91.64, 93.90, 97.21, 98.08, 97.56, 98.61],
    "Precision": [92.08, 93.79, 97.53, 97.45, 97.22, 98.46],
    "Recall":    [91.69, 94.28, 96.92, 98.10, 97.80, 98.55],
    "F1":        [91.88, 93.96, 97.21, 97.76, 97.46, 98.50],
}

# 表6：DeepShip 数据集
DEEPSHIP_DATA = {
    "Accuracy":  [95.79, 96.13, 98.87, 98.93, 98.82, 99.36],
    "Precision": [95.80, 96.10, 98.86, 98.94, 98.79, 99.33],
    "Recall":    [95.78, 96.18, 98.88, 98.92, 98.85, 99.37],
    "F1":        [95.79, 96.13, 98.87, 98.93, 98.82, 99.35],
}

# ======================================================================
# 图1：双数据集准确率对比折线图
# ======================================================================
def plot_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(GROUPS))
    ax.plot(x, SHIPEAR_DATA["Accuracy"],  marker="o", lw=2, color="#2E86AB",
            label="ShipsEar", markersize=7, linestyle="--")
    ax.plot(x, DEEPSHIP_DATA["Accuracy"], marker="s", lw=2, color="#E84855",
            label="DeepShip", markersize=7, linestyle="--")

    # 标注数值
    for i, (s, d) in enumerate(zip(SHIPEAR_DATA["Accuracy"], DEEPSHIP_DATA["Accuracy"])):
        ax.annotate(f"{s:.2f}", (i, s), textcoords="offset points",
            xytext=(0, -20), ha="center", fontsize=9, color="#2E86AB")
        ax.annotate(f"{d:.2f}", (i, d), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=9, color="#E84855")

    # 高亮最优组
    best_idx = len(GROUPS) - 1
    ax.axvspan(best_idx - 0.4, best_idx + 0.4, alpha=0.08, color="#3BB273")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Accuracy Comparison on Two Datasets")
    ax.set_ylim(89, 101)
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "ablation_accuracy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# 图2：ShipsEar 四指标分组柱状图
# ======================================================================
def plot_metrics_bar(data: dict, dataset_name: str, filename: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#2E86AB", "#E84855", "#3BB273", "#F18F01"]
    x = np.arange(len(GROUPS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[metric], width,
                      label=metric, color=color,
                      edgecolor="white", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    # 高亮最优组
    best_idx = len(GROUPS) - 1
    ax.axvspan(best_idx - 0.45, best_idx + 0.45, alpha=0.08, color="#3BB273")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Ablation Study: Per-Metric Comparison on {dataset_name}")
    ax.set_ylim(88, 102)
    ax.legend(frameon=True, framealpha=0.9, ncol=4)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# 图3：各指标随分支增加的增益热力图
# ======================================================================
def plot_gain_heatmap():
    import seaborn as sns

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    shipear_matrix  = np.array([[SHIPEAR_DATA[m][i]  for m in metrics] for i in range(len(GROUPS))])
    deepship_matrix = np.array([[DEEPSHIP_DATA[m][i] for m in metrics] for i in range(len(GROUPS))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, matrix, title in zip(
        axes,
        [shipear_matrix, deepship_matrix],
        ["ShipsEar", "DeepShip"]
    ):
        sns.heatmap(
            matrix,
            annot=True, fmt=".2f",
            cmap="Blues",
            xticklabels=metrics,
            yticklabels=GROUPS,
            linewidths=0.5, linecolor="white",
            vmin=88, vmax=100,
            cbar_kws={"label": "Score (%)"},
            ax=ax,
        )
        ax.set_title(f"{title} Dataset")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Branch Combination")

    plt.suptitle("Ablation Study Heatmap", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "ablation_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# 主程序
# ======================================================================
if __name__ == "__main__":
    print("Generating ablation study plots...")
    plot_accuracy_comparison()
    plot_metrics_bar(SHIPEAR_DATA,  "ShipsEar", "ablation_shipear_metrics.png")
    plot_metrics_bar(DEEPSHIP_DATA, "DeepShip", "ablation_deepship_metrics.png")
    plot_gain_heatmap()
    print(f"\nAll plots saved to: {SAVE_DIR}/")