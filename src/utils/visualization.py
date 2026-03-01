# src/utils/visualization.py

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    plt.show()
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
    plt.show()
    if logger:
        logger.info("Confusion matrix saved to: %s", save_path)