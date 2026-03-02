# -*-coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ShipEarMultiDataset(Dataset):
    """
    多模态 ShipEar 数据集。

    YAML 中通过 instantiate 创建（见 src/configs/datasets/shipear.yaml）。
    split_dataset / create_data_loader 功能迁移至 data_utils.py。
    """
    def __init__(
        self,
        # --- 数据路径 ---
        annotations_file: str,
        mel_dir: str,
        mfcc_dir: str,
        # --- CSV ---
        mel_col: int = 7,
        mfcc_col: int = 7,
        text_col: int = 12,
        label_col: int = 5,
        # --- 分支开关 ---
        use_time_branch: bool = True,
        use_spectrogram_branch: bool = True,
        use_text_branch: bool = True,
        # --- 训练模式：控制是否应用增强 ---
        is_training: bool = False,
        # --- 增强参数 ---
        noise_factor: float = 0.01,
        time_shift_max: float = 0.1,
        # --- 随机种子 ---
        random_seed: int = 42,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.mel_dir = mel_dir
        self.mfcc_dir = mfcc_dir
        self.mel_col = mel_col
        self.mfcc_col = mfcc_col
        self.text_col = text_col
        self.label_col = label_col

        self.use_time = use_time_branch
        self.use_spectrogram = use_spectrogram_branch
        self.use_text_branch = use_text_branch

        self.is_training = is_training
        self.noise_factor = noise_factor
        self.time_shift_max = time_shift_max
        self.rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict:
        # --- 时域特征 ---
        time_domain = None
        if self.use_time and self.mel_col is not None:
            mel_name = self._get_cell(index, self.mel_col)
            time_domain = self._load_feature(self.mel_dir, mel_name)

        # --- 频谱特征 ---
        spectrogram_domain = None
        if self.use_spectrogram and self.mfcc_col is not None:
            mfcc_name = self._get_cell(index, self.mfcc_col)
            spectrogram_domain = self._load_feature(self.mfcc_dir, mfcc_name)

        # --- 文本特征 ---
        text_domain = None
        if self.use_text_branch and self.text_col is not None:
            raw_text = self._get_cell(index, self.text_col)
            if pd.notna(raw_text):
                text_domain = str(raw_text)

        # --- 数据增强（仅训练集）---
        if time_domain is not None and self.is_training:
            time_domain = self._apply_augmentation(time_domain)
        if spectrogram_domain is not None and self.is_training:
            spectrogram_domain = self._apply_augmentation(spectrogram_domain)

        # --- Channel dim ---
        if time_domain is not None:
            time_domain = self._ensure_channel_dim(time_domain)
        if spectrogram_domain is not None:
            spectrogram_domain = self._ensure_channel_dim(spectrogram_domain)

        label = torch.tensor(self._get_cell(index, self.label_col), dtype=torch.long)

        sample = {"label": label}
        if time_domain is not None:
            sample["time"] = time_domain
        if spectrogram_domain is not None:
            sample["spectrogram"] = spectrogram_domain
        if text_domain is not None:
            sample["text"] = text_domain
        return sample

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _get_cell(self, index: int, column):
        if isinstance(column, int):
            return self.annotations.iloc[index, column]
        return self.annotations.loc[index, column]

    def _load_feature(self, root: str, filename: str) -> torch.Tensor:
        fname = (
            f"{filename}.npy" if not str(filename).endswith(".npy") else str(filename)
        )
        feature = np.load(os.path.join(root, fname))
        return torch.from_numpy(feature).float()

    def _apply_augmentation(self, feature: torch.Tensor) -> torch.Tensor:
        # 高斯噪声
        if self.rng.random() > 0.5:
            feature = feature + torch.randn_like(feature) * self.noise_factor
        # 时间平移
        if self.rng.random() > 0.5:
            shift = int(
                feature.shape[-1] * self.time_shift_max * (self.rng.random() - 0.5)
            )
            feature = torch.roll(feature, shifts=shift, dims=-1)
        # 频率遮蔽
        if feature.ndim >= 2 and feature.shape[-2] > 1 and self.rng.random() > 0.5:
            mask = max(1, int(feature.shape[-2] * 0.1))
            start = self.rng.integers(0, max(1, feature.shape[-2] - mask + 1))
            feature[..., start : start + mask, :] *= 0.1
        return feature

    def _ensure_channel_dim(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        return feature