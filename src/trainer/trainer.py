# -*-coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from src.utils.visualization import plot_loss_accuracy_curves
from .base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def train(self) -> None:
        """
        覆写 BaseTrainer.train()，在标准循环基础上额外收集曲线数据，
        训练结束后绘制 loss / accuracy 曲线。
        """
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        self.logger.info("******  Running training  ******")
        self.logger.info("Num Epochs    = %d", self.config.epochs)
        self.logger.info("Batch size    = %d", self.config.batch_size)
        self.logger.info("Learning rate = %s", self.config.optimizer.lr)
        self.logger.info("Early stop    = %s",self._early_stop_patience if self._early_stop_patience > 0 else "disabled")

        for epoch in range(1, self.config.epochs + 1):
            self.logger.info("******************** Epoch: %d/%d ********************", epoch, self.config.epochs)

            train_log = self._train_epoch(epoch)
            val_log = self._valid_epoch(epoch)
            log = {**train_log, **val_log}

            train_losses.append(log["train_loss"])
            val_losses.append(log["val_loss"])
            train_accs.append(log["train_accuracy"])
            val_accs.append(log["val_accuracy"])

            # Writer (WandB)
            if self.writer is not None:
                self.writer.set_step(epoch)
                for k, v in log.items():
                    self.writer.add_scalar(k, v, epoch)

            # LR scheduler
            self._step_scheduler(log)

            # Epoch summary
            self.logger.info(
                "Epoch %d Summary — train_loss: %.6f  val_loss: %.6f  "
                "val_acc: %.2f%%  best_%s: %.4f",
                epoch,
                log["train_loss"], log["val_loss"],
                log["val_accuracy"],
                self._monitor_metric or "-",
                self._best_value if self._best_value is not None else 0,
            )

            # Best model & periodic checkpoint
            is_best = self._is_best(log.get(self._monitor_metric))
            if is_best:
                self._save_checkpoint(epoch, save_best=True)
                self._early_stop_counter = 0
            else:
                self._early_stop_counter += 1

            if epoch % self.config.save_period == 0 or epoch == self.config.epochs:
                self._save_checkpoint(epoch, save_best=False)

            # Early stopping
            if self._early_stop_patience > 0:
                if self._early_stop_counter >= self._early_stop_patience:
                    self.logger.info(
                        "Early stopping triggered at epoch %d. "
                        "Best %s: %.4f",
                        epoch, self._monitor_metric, self._best_value,
                    )
                    break

        # 绘制曲线
        plot_loss_accuracy_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_dir=self.config.plot_dir,
            logger=self.logger,
        )

        if self.writer is not None:
            self.writer.finish()

        self.logger.info("******  Training completed  ******")

    # ------------------------------------------------------------------
    # 训练 epoch（对齐原 _train_epoch，仅返回值改为 dict）
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        train_bar = tqdm(
            enumerate(self.dataloaders["train"]),
            total=len(self.dataloaders["train"]),
            desc=f"Training Epoch {epoch}/{self.config.epochs}",
            unit="batch",
        )

        for batch_idx, batch in train_bar:
            labels = batch["label"].to(self.device)
            kwargs = self._batch_to_kwargs(batch)

            outputs = self.model(**kwargs)
            logits = outputs["logits"]
            balance_loss = outputs.get("balance_loss", 0.0)
            loss = self.loss_fn(logits, labels) + balance_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % 10 == 9:
                self.logger.info(
                    "epoch: %d | batch: %d | train loss: %.4f",
                    epoch, batch_idx + 1, loss.item(),
                )

        avg_loss = total_loss / len(self.dataloaders["train"])
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        self.logger.info(
            "Train Epoch %d:  Loss=%.6f  Accuracy=%.2f%%", epoch, avg_loss, accuracy
        )
        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    # ------------------------------------------------------------------
    # 验证 epoch（对齐原 _validate_epoch，仅返回值改为 dict）
    # ------------------------------------------------------------------

    def _valid_epoch(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        batch_idx = 0

        val_bar = tqdm(
            enumerate(self.dataloaders["val"]),
            total=len(self.dataloaders["val"]),
            desc=f"Validation Epoch {epoch}",
            unit="batch",
        )

        with torch.no_grad():
            for batch_idx, batch in val_bar:
                labels = batch["label"].to(self.device)
                kwargs = self._batch_to_kwargs(batch)

                outputs = self.model(**kwargs)
                logits = outputs["logits"]
                balance_loss = outputs.get("balance_loss", 0.0)
                loss = self.loss_fn(logits, labels) + balance_loss

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                batch_acc = (predicted == labels).float().mean().item() * 100
                val_bar.set_postfix(
                    val_loss=f"{loss.item():.4f}",
                    batch_acc=f"{batch_acc:.2f}%",
                )

        avg_loss = total_loss / len(self.dataloaders["val"])
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        self.logger.info(
            "Val Epoch %d, Batch %d:  Loss=%.6f  Accuracy=%.2f%%",
            epoch, batch_idx + 1, avg_loss, accuracy,
        )
        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _batch_to_kwargs(self, batch: dict) -> dict:
        """将 batch dict 转为模型 forward() 所需 kwargs，Tensor 移到 device。"""
        kwargs = {}
        if (t := batch.get("time")) is not None:
            kwargs["time"] = t.to(self.device)
        if (s := batch.get("spectrogram")) is not None:
            kwargs["spectrogram"] = s.to(self.device)
        if (txt := batch.get("text")) is not None:
            kwargs["texts"] = txt
        return kwargs