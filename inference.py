# -*-coding: utf-8 -*-

import warnings
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.datasets.data_utils import get_dataloaders
from src.trainer.inferencer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config: DictConfig) -> None:
    # ------------------------------------------------------------------
    # 1. 基础初始化
    # ------------------------------------------------------------------
    set_random_seed(config.trainer.seed)
    logger = setup_saving_and_logging(config)

    # 设备：优先读 inference.device，回退到 trainer.device
    infer_cfg = getattr(config, "inference", None)
    raw_device = (
        getattr(infer_cfg, "device", None) or config.trainer.device
    )
    device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ) if raw_device == "auto" else raw_device

    logger.info("Inference device: %s", device)

    # ------------------------------------------------------------------
    # 2. 数据集加载
    # ------------------------------------------------------------------
    logger.info("******  Loading dataset for inference...  ******")
    dataloaders = get_dataloaders(config, device)
    logger.info("Test batches: %d", len(dataloaders["test"]))

    # ------------------------------------------------------------------
    # 3. 模型构建
    # ------------------------------------------------------------------
    logger.info("******  Building model...  ******")
    model = instantiate(config.model).to(device)
    logger.info("******  Model built!  ******")

    # ------------------------------------------------------------------
    # 4. 加载 checkpoint
    # ------------------------------------------------------------------
    import os
    save_dir = getattr(infer_cfg, "save_dir", None) or config.trainer.save_dir
    ckpt_name = getattr(infer_cfg, "ckpt_name", None) or config.trainer.ckpt_name
    ckpt_path = os.path.join(save_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"请先运行 python train.py 训练模型，或指定正确路径：\n"
            f"  python infer.py inference.save_dir=<path> inference.ckpt_name=<file>"
        )

    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    # 兼容两种保存格式：bare state_dict 或完整 checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    logger.info("Checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # 5. 推理 & 评估
    # ------------------------------------------------------------------
    inferencer = Inferencer(config, dataloaders["test"], model, device, logger)
    inferencer.predict()


if __name__ == "__main__":
    main()