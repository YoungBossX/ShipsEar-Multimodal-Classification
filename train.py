# -*-coding: utf-8 -*-
# train.py

import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.datasets.data_utils import get_dataloaders, get_class_weights
from src.trainer.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config: DictConfig) -> None:
    """
    Main training script.

    Usage:
        python train.py                                          # 默认配置
        python train.py trainer.optimizer.lr=...                 # 覆盖学习率
        python train.py datasets.batch_size=...                  # 覆盖 batch size
        python train.py model=multimodal_moe trainer.epochs=200  # 覆盖模型和训练轮数
    """
    # ------------------------------------------------------------------
    # 1. 基础初始化
    # ------------------------------------------------------------------
    set_random_seed(config.trainer.seed)
    project_config = OmegaConf.to_container(config, resolve=True)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    logger.info("Full config:\n%s", OmegaConf.to_yaml(config))

    device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ) if config.trainer.device == "auto" else config.trainer.device

    # ------------------------------------------------------------------
    # 2. 设备信息打印
    # ------------------------------------------------------------------
    print("Training device information:")
    if torch.cuda.is_available():
        device_table = PrettyTable(
            ["number of gpu", "applied gpu index", "applied gpu name"],
            min_table_width=80,
        )
        device_table.add_row(
            [torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name()]
        )
        print(f"{device_table}\n")
    else:
        print("Using CPU\n")

    # ------------------------------------------------------------------
    # 3. 数据集加载
    # ------------------------------------------------------------------
    logger.info("******  Loading dataset...  ******")
    dataloaders = get_dataloaders(config, device)

    train_num = len(dataloaders["train"])
    val_num   = len(dataloaders["val"])
    test_num  = len(dataloaders["test"])
    total_num = train_num + val_num + test_num

    print(f"Use annotations file: {config.datasets.annotations_file}")
    dataset_table = PrettyTable(
        ["total batches", "train", "val", "test", "train%", "val%", "test%"],
        min_table_width=80,
    )
    dataset_table.add_row([
        total_num, train_num, val_num, test_num,
        f"{train_num/total_num:.2f}",
        f"{val_num/total_num:.2f}",
        f"{test_num/total_num:.2f}",
    ])
    print(f"{dataset_table}\n")
    logger.info("******  The dataset is loaded!  ******\n")

    # ------------------------------------------------------------------
    # 4. 多模态分支加载状态检测
    # ------------------------------------------------------------------
    _print_branch_status(config, dataloaders["train"])

    # ------------------------------------------------------------------
    # 5. 模型
    # ------------------------------------------------------------------
    logger.info("******  Start building the model...  ******")
    model = instantiate(config.model).to(device)
    logger.info("Model structure:\n%s", model)
    logger.info("******  The model is built!  ******\n")

    # ------------------------------------------------------------------
    # 6. 优化器
    # ------------------------------------------------------------------
    optimizer = instantiate(config.trainer.optimizer, params=model.parameters())

    # ------------------------------------------------------------------
    # 7. 损失函数 + 类别权重（对应原 compute_class_weights + CrossEntropyLoss）
    # ------------------------------------------------------------------
    loss_fn = instantiate(config.trainer.loss)
    class_weights = get_class_weights(config).to(device)
    loss_fn.weight = class_weights

    # ------------------------------------------------------------------
    # 8. 学习率调度器
    # ------------------------------------------------------------------
    lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)

    # ------------------------------------------------------------------
    # 9. 评估指标列表
    # ------------------------------------------------------------------
    metrics = [instantiate(m) for m in config.trainer.metrics]

    # ------------------------------------------------------------------
    # 10. 训练配置打印
    # ------------------------------------------------------------------
    print("Classes information:")
    class_names = ["A", "B", "C", "D", "E"]
    classes_table = PrettyTable(class_names, min_table_width=80)
    classes_table.add_row(range(len(class_names)))
    print(f"{classes_table}\n")

    print("Train information:")
    train_table = PrettyTable(
        ["model", "batch size", "epochs", "learning rate", "device", "save dir"],
        min_table_width=80,
    )
    train_table.add_row([
        config.model._target_.split(".")[-1],
        config.datasets.batch_size,
        config.trainer.epochs,
        config.trainer.optimizer.lr,
        device,
        config.trainer.save_dir,
    ])
    print(f"{train_table}\n")

    # ------------------------------------------------------------------
    # 11. 训练
    # ------------------------------------------------------------------
    if config.trainer.do_train:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            writer=writer,
            logger=logger,
            device=device,
            dataloaders=dataloaders,
            config=config.trainer,
        )
        trainer.train()

    # ------------------------------------------------------------------
    # 12. 推理
    # ------------------------------------------------------------------
    if config.trainer.do_infer:
        from inference import Inference
        inference = Inference(config, dataloaders["test"], model, device, logger)
        inference.predict()


# ------------------------------------------------------------------
# 辅助：多模态分支状态检测
# ------------------------------------------------------------------

def _print_branch_status(config, train_loader) -> None:
    """打印各模态分支的加载状态表。"""
    ds_cfg = config.datasets
    subset = train_loader.dataset          # torch.utils.data.Subset
    sample = subset.dataset[subset.indices[0]]

    print("Multimodal branches loading status:")
    table = PrettyTable(
        ["Branch", "Enabled", "Data Type", "Status"], min_table_width=80
    )

    # 时域分支
    if ds_cfg.use_time_branch:
        data = sample.get("time")
        status = "✓ Loaded" if data is not None else "✗ Failed"
        shape  = str(data.shape) if data is not None else "N/A"
        table.add_row(["Time Branch", "✓", f"Mel {shape}", status])
    else:
        table.add_row(["Time Branch", "✗", "N/A", "Disabled"])

    # 频谱分支
    if ds_cfg.use_spectrogram_branch:
        data = sample.get("spectrogram")
        status = "✓ Loaded" if data is not None else "✗ Failed"
        shape  = str(data.shape) if data is not None else "N/A"
        table.add_row(["Spectrogram Branch", "✓", f"MFCC {shape}", status])
    else:
        table.add_row(["Spectrogram Branch", "✗", "N/A", "Disabled"])

    # 文本分支
    if ds_cfg.use_text_branch:
        data = sample.get("text")
        if data is None:
            status, display = "✗ Failed", "N/A"
        elif isinstance(data, str):
            status  = "✓ Loaded (Raw text)"
            display = data[:50] + "..." if len(data) > 50 else data
        else:
            status, display = f"⚠ Unknown type: {type(data).__name__}", "N/A"
        table.add_row(["Text Branch", "✓", display, status])
    else:
        table.add_row(["Text Branch", "✗", "N/A", "Disabled"])

    print(f"{table}\n")


if __name__ == "__main__":
    main()