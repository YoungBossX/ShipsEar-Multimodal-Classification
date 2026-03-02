# -*-coding: utf-8 -*-

from typing import Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from .shipear_dataset import ShipEarMultiDataset

def get_dataloaders(config, device: Optional[str] = None) -> dict:
    """
    Build train / val / test DataLoaders from config.

    与模板的调用方式完全一致：
        dataloaders = get_dataloaders(config, device)

    读取 config.datasets（对应 src/configs/datasets/shipear.yaml）。
    """
    ds_cfg = config.datasets
    # 公共 Dataset 构造参数
    dataset_kwargs = dict(
        annotations_file=ds_cfg.annotations_file,
        mel_dir=ds_cfg.mel_dir,
        mfcc_dir=ds_cfg.mfcc_dir,
        mel_col=ds_cfg.mel_col,
        mfcc_col=ds_cfg.mfcc_col,
        text_col=ds_cfg.text_col,
        label_col=ds_cfg.label_col,
        use_time_branch=ds_cfg.use_time_branch,
        use_spectrogram_branch=ds_cfg.use_spectrogram_branch,
        use_text_branch=ds_cfg.use_text_branch,
        noise_factor=ds_cfg.noise_factor,
        time_shift_max=ds_cfg.time_shift_max,
        random_seed=ds_cfg.random_seed,
    )

    # 两个基础数据集（train 开增强，eval 不增强）
    train_base = ShipEarMultiDataset(**dataset_kwargs, is_training=ds_cfg.is_training)
    eval_base  = ShipEarMultiDataset(**dataset_kwargs, is_training=False)

    # 分层划分索引
    train_idx, val_idx, test_idx = _split_indices(
        eval_base,
        ds_cfg.train_ratio,
        ds_cfg.val_ratio,
        ds_cfg.test_ratio,
        ds_cfg.random_seed,
    )

    # DataLoader 构造
    def _make_loader(base_ds, indices, shuffle):
        subset = Subset(base_ds, indices)
        return DataLoader(
            subset,
            batch_size=ds_cfg.batch_size,
            shuffle=shuffle,
            num_workers=getattr(ds_cfg, "num_workers", 0),
            collate_fn=_multimodal_collate_fn,
            pin_memory=(device is not None and "cuda" in str(device)),
        )

    return {
        "train": _make_loader(train_base, train_idx, shuffle=ds_cfg.train_shuffle),
        "val":   _make_loader(eval_base,  val_idx,   shuffle=ds_cfg.val_shuffle),
        "test":  _make_loader(eval_base,  test_idx,  shuffle=ds_cfg.test_shuffle),
    }

def get_class_weights(config, train_dataloader) -> torch.Tensor:
    dataset = train_dataloader.dataset
    labels = dataset.dataset.annotations.iloc[
        dataset.indices, dataset.dataset.label_col
    ].values
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return torch.FloatTensor(weights)

# ------------------------------------------------------------------
# 私有辅助
# ------------------------------------------------------------------
def _split_indices(dataset, train_ratio, val_ratio, test_ratio, random_seed):
    """
    分层划分，与原 split_dataset() 逻辑完全一致。
    """
    labels = dataset.annotations.iloc[:, dataset.label_col].values
    indices = np.arange(len(dataset))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed,
    )
    val_size = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=labels[train_val_idx],
        random_state=random_seed,
    )
    return train_idx, val_idx, test_idx


def _multimodal_collate_fn(batch: list) -> dict:
    """
    自定义 collate，处理 text 字段是 str（不能 stack）的情况。
    其余 Tensor 字段正常 stack。
    """
    collated = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            # text: list[str]，保持原样
            collated[key] = values
    return collated