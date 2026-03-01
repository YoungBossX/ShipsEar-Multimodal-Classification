# -*-coding: utf-8 -*-

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Fix seed for reproducibility (identical to template's set_random_seed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_saving_and_logging(config) -> logging.Logger:
    """
    Create save/log directories and configure root logger.

    Called exactly like the template:
        logger = setup_saving_and_logging(config)

    Reads:
        config.trainer.save_dir   → checkpoint & plot root
        config.trainer.log_dir    → log file root
    """
    save_dir = Path(config.trainer.save_dir)
    log_dir = Path(config.trainer.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = log_dir / f"shipear_{timestamp}.log"

    # ---- formatter ----
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    # ---- root logger ----
    logger = logging.getLogger("shipear")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    logger.info("Checkpoints will be saved to: %s", save_dir)
    return logger