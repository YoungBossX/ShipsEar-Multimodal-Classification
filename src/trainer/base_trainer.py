# -*-coding: utf-8 -*-

import logging
import os
from abc import abstractmethod
from pathlib import Path
import torch

class BaseTrainer:
    """
    Abstract base trainer.

    Constructor args mirror what train.py passes via Hydra config:

        trainer = Trainer(
            model, optimizer, lr_scheduler,
            loss_fn, metrics, writer, logger,
            device, dataloaders, config=config.trainer,
        )
    """

    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        loss_fn,
        metrics: list,
        writer,
        logger: logging.Logger,
        device: str,
        dataloaders: dict,
        config,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.writer = writer
        self.logger = logger
        self.device = device
        self.dataloaders = dataloaders
        self.config = config

        # --- checkpoint 路径 ---
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- 监控指标解析："max val_accuracy" 或 "min val_loss" ---
        self._monitor_mode = None
        self._monitor_metric = None
        self._best_value = None

        if hasattr(config, "monitor") and config.monitor:
            parts = str(config.monitor).split()
            assert len(parts) == 2 and parts[0] in ("min", "max"), (
                "config.trainer.monitor 格式应为 'min <metric>' 或 'max <metric>'"
            )
            self._monitor_mode = parts[0]    # "max" or "min"
            self._monitor_metric = parts[1]  # e.g. "val_accuracy"
            self._best_value = -float("inf") if self._monitor_mode == "max" else float("inf")

        # --- early stopping ---
        self._early_stop_patience = int(getattr(config, "early_stopping", 0))
        self._early_stop_counter = 0

        # --- resume ---
        resume_path = getattr(config, "resume", None)
        if resume_path:
            self._resume_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """Return dict with at least {'train_loss': float, 'train_accuracy': float}."""
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch: int) -> dict:
        """Return dict with at least {'val_loss': float, 'val_accuracy': float}."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Full training loop with checkpoint, best-model tracking, early stop."""
        self.logger.info("******  Running training  ******")
        self.logger.info("Num Epochs    = %d", self.config.epochs)
        self.logger.info("Batch size    = %d", self.config.batch_size)
        self.logger.info("Learning rate = %s", self.config.optimizer.lr)
        early_stop_info = (
            self._early_stop_patience if self._early_stop_patience > 0 else "disabled"
        )
        self.logger.info("Early stop    = %s", early_stop_info)

        for epoch in range(1, self.config.epochs + 1):
            self.logger.info(
                "-------------------- Epoch: %d/%d --------------------",
                epoch, self.config.epochs,
            )

            train_log = self._train_epoch(epoch)
            val_log = self._valid_epoch(epoch)
            log = {**train_log, **val_log}

            # Writer (WandB)
            if self.writer is not None:
                self.writer.set_step(epoch)
                for k, v in log.items():
                    self.writer.add_scalar(k, v, epoch)

            # LR scheduler
            self._step_scheduler(log)

            # Epoch summary
            self.logger.info(
                "Epoch %d | train_loss=%.6f  val_loss=%.6f  "
                "train_acc=%.2f%%  val_acc=%.2f%%  best=%s=%.4f",
                epoch,
                log.get("train_loss", 0),
                log.get("val_loss", 0),
                log.get("train_accuracy", 0),
                log.get("val_accuracy", 0),
                self._monitor_metric or "-",
                self._best_value if self._best_value is not None else 0,
            )

            # Best model & checkpoint
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

        if self.writer is not None:
            self.writer.finish()

        self.logger.info("******  Training completed  ******")

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _is_best(self, current_value) -> bool:
        """Return True if current_value beats the stored best."""
        if self._monitor_mode is None or current_value is None:
            return False
        if self._monitor_mode == "max" and current_value > self._best_value:
            self._best_value = current_value
            return True
        if self._monitor_mode == "min" and current_value < self._best_value:
            self._best_value = current_value
            return True
        return False

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """Save model state dict + training state."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_value": self._best_value,
            "config": dict(self.config),
        }
        if save_best:
            path = self.save_dir / "model_best.pt"
            torch.save(state["model_state_dict"], path)
            self.logger.info(
                "New best model saved → %s  (%s=%.4f)",
                path, self._monitor_metric, self._best_value,
            )
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(state, path)
            self.logger.info("Checkpoint saved → %s", path)

    def _resume_checkpoint(self, resume_path: str) -> None:
        """Load checkpoint and restore training state."""
        path = Path(resume_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        self.logger.info("Resuming from checkpoint: %s", path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "best_value" in checkpoint:
                self._best_value = checkpoint["best_value"]
            self.logger.info(
                "Resumed from epoch %d, best_value=%.4f",
                checkpoint.get("epoch", -1), self._best_value or 0,
            )
        else:
            # bare state dict (e.g. best_model.pt)
            self.model.load_state_dict(checkpoint)
            self.logger.info("Loaded bare state dict from %s", path)

    def _step_scheduler(self, log: dict) -> None:
        """Call lr_scheduler.step() with appropriate arguments."""
        if self.lr_scheduler is None:
            return
        name = type(self.lr_scheduler).__name__
        if "ReduceLROnPlateau" in name:
            monitor_val = log.get(self._monitor_metric or "val_loss")
            if monitor_val is not None:
                self.lr_scheduler.step(monitor_val)
        else:
            self.lr_scheduler.step()