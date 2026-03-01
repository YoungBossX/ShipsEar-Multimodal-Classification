# -*-coding: utf-8 -*-

import logging
from typing import Optional
import wandb

class WandBWriter:
    """
    Experiment tracking writer using Weights & Biases.

    Instantiated by Hydra:
        writer:
          _target_: src.logger.WandBWriter
          project: ShipEar-Multimodal-Classification
          entity: null
          enabled: false   # set true to activate W&B logging

    The template passes (logger, project_config) as positional args.
    """

    def __init__(
        self,
        logger: logging.Logger,
        config: dict,
        project: str = "ShipEar-Multimodal-Classification",
        entity: Optional[str] = None,
        enabled: bool = False,
    ):
        self.logger = logger
        self.enabled = enabled
        self._step = 0

        if self.enabled:
            try:
                self._wandb = wandb
                wandb.init(project=project, entity=entity, config=config)
                logger.info("WandB initialized: project=%s", project)
            except ImportError:
                logger.warning(
                    "wandb not installed. Falling back to no-op writer."
                )
                self.enabled = False
        else:
            logger.info("WandBWriter disabled (writer.enabled=false).")

    def set_step(self, step: int) -> None:
        self._step = step

    def add_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        s = step if step is not None else self._step
        self._wandb.log({tag: value}, step=s)

    def add_scalars(self, tag: str, values: dict, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        s = step if step is not None else self._step
        self._wandb.log({f"{tag}/{k}": v for k, v in values.items()}, step=s)

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()