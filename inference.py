# -*-coding: utf-8 -*-

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm


class Inference:
    def __init__(
        self,
        config,
        test_dataloader,
        model,
        device: str,
        logger: logging.Logger,
    ):
        self.config = config
        self.device = device
        self.test_dataloader = test_dataloader
        self.model = model.to(device)
        self.logger = logger

        # 加载 best model checkpoint
        ckpt_path = os.path.join(
            config.trainer.save_dir,
            config.trainer.ckpt_name,
        )
        if os.path.exists(ckpt_path):
            self.logger.info("Loading checkpoint: %s", ckpt_path)
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state)
            self.logger.info("Checkpoint loaded successfully.")
        else:
            self.logger.warning("Checkpoint not found: %s", ckpt_path)
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ------------------------------------------------------------------
    # 绘制混淆矩阵
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, cm, class_names, title, save_path=None) -> None:
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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info("Confusion matrix saved to: %s", save_path)
        plt.show()

    # ------------------------------------------------------------------
    # 推理主流程
    # ------------------------------------------------------------------
    def predict(self) -> None:
        self.logger.info("******  Running infer  ******")

        all_predictions, all_probabilities, all_labels = [], [], []

        self.model.eval()
        test_bar = tqdm(self.test_dataloader, desc="Testing")

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_bar):
                labels = batch["label"].to(self.device)
                kwargs = self._batch_to_kwargs(batch)

                all_labels.extend(labels.cpu().numpy())
                outputs = self.model(**kwargs)
                logits = outputs["logits"]

                probs = F.softmax(logits, dim=1)
                all_probabilities.extend(probs.cpu().numpy())
                predicted = torch.argmax(logits, dim=1)
                all_predictions.extend(predicted.cpu().numpy())

                test_bar.set_postfix(
                    batch=f"{batch_idx + 1}/{len(self.test_dataloader)}"
                )

        predictions = np.array(all_predictions)
        labels      = np.array(all_labels)

        self.logger.info("Inference summary:")
        self.logger.info("Total samples: %d", len(labels))
        self.logger.info("Predicted labels: %d", len(predictions))
        self.logger.info(
            "Probability matrix shape: %s", str(np.array(all_probabilities).shape)
        )
        self.logger.info("******  Finishing infer  ******")

        # ---- 评估 ----
        self.logger.info("******  Doing evaluate  ******")

        accuracy = (predictions == labels).mean() * 100
        num_classes = len(np.unique(labels))
        class_correct = np.zeros(num_classes)
        class_total   = np.zeros(num_classes)
        for i in range(len(labels)):
            class_total[labels[i]] += 1
            if predictions[i] == labels[i]:
                class_correct[labels[i]] += 1
        class_accuracy = (class_correct / class_total) * 100

        self.logger.info("Total samples: %d", len(labels))
        self.logger.info("Correct predictions: %d", (predictions == labels).sum())
        self.logger.info("Overall accuracy: %.2f%%", accuracy)
        for i in range(num_classes):
            if class_total[i] > 0:
                self.logger.info(
                    "Class %d: %.0f/%.0f (%.2f%%)",
                    i, class_correct[i], class_total[i], class_accuracy[i],
                )

        overall_acc = accuracy_score(labels, predictions) * 100
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average="micro", zero_division=0
        )
        wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )

        self.logger.info("******  Detailed Evaluation Metrics  ******")
        self.logger.info("Overall accuracy: %.2f%%", overall_acc)
        self.logger.info("Macro  Precision=%.2f%% Recall=%.2f%% F1=%.2f%%",
                         macro_p*100, macro_r*100, macro_f1*100)
        self.logger.info("Micro  Precision=%.2f%% Recall=%.2f%% F1=%.2f%%",
                         micro_p*100, micro_r*100, micro_f1*100)
        self.logger.info("Weighted Precision=%.2f%% Recall=%.2f%% F1=%.2f%%",
                         wt_p*100, wt_r*100, wt_f1*100)

        self.logger.info("Per-class metrics:")
        for i in range(num_classes):
            if class_total[i] > 0:
                self.logger.info(
                    "Class %d: Acc=%.2f%% P=%.4f R=%.4f F1=%.4f Support=%d",
                    i, class_accuracy[i], precision[i], recall[i], f1[i], int(support[i]),
                )

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        plot_dir = getattr(self.config.trainer, "plot_dir", self.config.trainer.save_dir)
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(
            cm,
            class_names=["A", "B", "C", "D", "E"],
            title="Confusion Matrix for this Model",
            save_path=save_path,
        )
        self.logger.info("Confusion Matrix:\n%s", str(cm))
        self.logger.info(
            "Classification Report:\n%s",
            classification_report(labels, predictions, digits=2),
        )
        self.logger.info("******  Finishing evaluate  ******")

    def _batch_to_kwargs(self, batch: dict) -> dict:
        kwargs = {}
        if (t := batch.get("time")) is not None:
            kwargs["time"] = t.to(self.device)
        if (s := batch.get("spectrogram")) is not None:
            kwargs["spectrogram"] = s.to(self.device)
        if (txt := batch.get("text")) is not None:
            kwargs["texts"] = txt
        return kwargs