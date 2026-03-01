# -*-coding: utf-8 -*-

from typing import Iterable, Optional, Sequence, Union
import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor
from .Conformer import AttentivePool, ConformerBlock, SinusoidalPositionalEncoding
from .CNN import CNN

Tensor = torch.Tensor

# ======================================================================
# 各模态分支
# ======================================================================

class TimeConformerBranch(nn.Module):
    """Conformer encoder for 1D audio (Mel) features."""

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 128,
        n_blocks: int = 4,
        ff_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        heads: int = 8,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        max_len: int = 10000,
        target_dim: int = 512,
    ):
        super().__init__()
        self.in_proj = (
            nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim=d_model,
                    ff_expansion_factor=ff_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    heads=heads,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    max_len=max_len,
                )
                for _ in range(n_blocks)
            ]
        )
        self.pool = AttentivePool(d_model)
        self.out_proj = (
            nn.Linear(d_model, target_dim) if d_model != target_dim else nn.Identity()
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

    def forward(self, time, lengths=None):
        time = time.squeeze(1).transpose(1, 2)  # (B, T, F)
        x = self.in_proj(time)
        sequence = x.transpose(0, 1)           # (T, B, D)
        sequence = self.pos_enc(sequence)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = sequence.transpose(0, 1)     # (B, T, D)
        if lengths is None:
            lengths = torch.full(
                (sequence.size(0),), sequence.size(1),
                dtype=torch.long, device=sequence.device,
            )
        pooled = self.pool(sequence, lengths)
        return self.out_proj(pooled)


class SpectrogramCNNBranch(nn.Module):
    """CNN branch for 2D spectrograms (MFCC)."""

    def __init__(self, num_classes: int = 5, target_dim: int = 512):
        super().__init__()
        self.cnn = CNN(num_classes=num_classes)
        with torch.no_grad():
            dummy = torch.randn(1, 1, 40, 94)
            out = self.cnn(dummy)
            feature_dim = out.view(1, -1).size(1)
        self.proj = (
            nn.Linear(feature_dim, target_dim)
            if target_dim != feature_dim
            else nn.Identity()
        )

    def forward(self, spectrogram):
        features = self.cnn(spectrogram)
        return self.proj(features.view(features.size(0), -1))


class ClapTextBranch(nn.Module):
    """Frozen CLAP text encoder branch."""

    def __init__(
        self,
        # 改动：路径改为参数（在 YAML 中配置），不再 hardcode
        pretrained_model_name: str = "laion/clap-htsat-unfused",
        target_dim: int = 512,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(
            pretrained_model_name, local_files_only=local_files_only
        )
        self.text_model = ClapModel.from_pretrained(
            pretrained_model_name, local_files_only=local_files_only
        )
        self.text_model.eval()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.text_model.to(self._device)

        text_dim = self._infer_text_dim()
        self.proj = (
            nn.Linear(text_dim, target_dim) if target_dim != text_dim else nn.Identity()
        )
        for param in self.text_model.parameters():
            param.requires_grad_(False)

    def _infer_text_dim(self) -> int:
        projection = getattr(self.text_model, "text_projection", None)
        if projection is not None:
            if hasattr(projection, "out_features"):
                return projection.out_features
            if hasattr(projection, "weight"):
                return projection.weight.size(0)
        cfg = getattr(self.text_model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "projection_dim"):
                return cfg.projection_dim
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg and hasattr(text_cfg, "hidden_size"):
                return text_cfg.hidden_size
        raise AttributeError("Unable to infer CLAP text embedding dimension.")

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.text_model.get_text_features(**inputs)
        return self.proj(embeddings)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_model.eval()
        return self


# ======================================================================
# 融合模块
# ======================================================================

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, ff_multiplier=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_multiplier * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        attn_out, _ = self.attn(query, context, context, need_weights=False)
        query = self.ln1(query + self.dropout(attn_out))
        return self.ln2(query + self.ffn(query))


class CrossAttentionFusionHead(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        ff_multiplier=4,
        output_dim=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(embed_dim, num_heads, dropout, ff_multiplier)
                for _ in range(num_layers)
            ]
        )
        self.fusion_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        valid = [f for f in features if f is not None]
        if not valid:
            raise ValueError("At least one modality feature required.")
        batch_size = valid[0].size(0)
        device = self.fusion_token.device
        processed = [f.to(device) for f in valid]
        context = torch.stack(processed, dim=1)
        fusion = self.fusion_token.expand(batch_size, -1, -1)
        for layer in self.layers:
            fusion = layer(fusion, context)
        return self.norm(fusion).squeeze(1)


class TopKMoEClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_num: int = 8,
        output_dim: int = 5,
        top_k: int = 3,
        balance_wt: float = 0.01,
    ):
        super().__init__()
        if expert_num < 1:
            raise ValueError("expert_num must be >= 1")
        self.top_k = max(1, min(top_k, expert_num))
        self.expert_num = expert_num
        self.output_dim = output_dim
        self.balance_wt = balance_wt
        self.routing = nn.Linear(input_dim, expert_num)
        self.experts = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(expert_num)]
        )

    def forward(self, features):
        return self.forward_from_features(features)

    def forward_from_features(self, features):
        routing_prob = torch.softmax(self.routing(features), dim=-1)
        top_values, top_indices = routing_prob.topk(self.top_k, dim=1)
        weights = top_values / top_values.sum(dim=1, keepdim=True).clamp_min(1e-9)

        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )
        gathered = expert_outputs.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, self.output_dim)
        )
        logits = (weights.unsqueeze(-1) * gathered).sum(dim=1)

        pi = routing_prob.mean(dim=0)
        balance_loss = self.balance_wt * (
            (pi * pi.clamp_min(1e-9).log()).sum()
            + torch.log(torch.tensor(self.expert_num, dtype=pi.dtype, device=pi.device))
        )
        return logits, balance_loss


# ======================================================================
# 顶层多模态模型（与原代码接口完全一致）
# ======================================================================

class MultimodalModel(nn.Module):
    """
    Complete multimodal encoder with cross-attention fusion + MoE classifier.

    YAML 中通过 Hydra instantiate 创建（见 src/configs/model/multimodal_moe.yaml）。
    forward() 接口与原代码完全一致，返回 dict 包含 logits / balance_loss 等。
    """

    def __init__(
        self,
        time_branch: Optional[nn.Module] = None,
        spectrogram_branch: Optional[nn.Module] = None,
        text_branch: Optional[nn.Module] = None,
        fusion_embed_dim: int = 512,
        fusion_heads: int = 8,
        fusion_layers: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        fusion_output_dim: Optional[int] = None,
        num_classes: int = 5,
        classifier: Optional[nn.Module] = None,
        classifier_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        fusion_output_dim = fusion_embed_dim if fusion_output_dim is None else fusion_output_dim

        self.time_branch = time_branch or TimeConformerBranch(target_dim=fusion_embed_dim)
        self.spectrogram_branch = spectrogram_branch or SpectrogramCNNBranch(
            target_dim=fusion_embed_dim
        )
        self.text_branch = text_branch or ClapTextBranch(target_dim=fusion_embed_dim)

        self.modality_norms = nn.ModuleDict(
            {
                "time": nn.LayerNorm(fusion_embed_dim),
                "spectrogram": nn.LayerNorm(fusion_embed_dim),
                "text": nn.LayerNorm(fusion_embed_dim),
            }
        )
        self.fusion_head = CrossAttentionFusionHead(
            embed_dim=fusion_embed_dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            dropout=dropout,
            ff_multiplier=ff_multiplier,
            output_dim=fusion_output_dim,
        )
        kw = dict(classifier_kwargs or {})
        kw.setdefault("input_dim", fusion_output_dim)
        kw.setdefault("output_dim", num_classes)
        self.classifier = classifier or TopKMoEClassifier(**kw)

    def forward(
        self,
        time: Optional[Tensor] = None,
        spectrogram: Optional[Tensor] = None,
        texts: Optional[Union[str, Iterable[str]]] = None,
        audio_lengths: Optional[Tensor] = None,
    ) -> dict:
        outputs = {}
        features = []

        if time is not None:
            f = self.modality_norms["time"](self.time_branch(time, lengths=audio_lengths))
            outputs["time"] = f
            features.append(f)
        if spectrogram is not None:
            f = self.modality_norms["spectrogram"](self.spectrogram_branch(spectrogram))
            outputs["spectrogram"] = f
            features.append(f)
        if texts is not None:
            f = self.modality_norms["text"](self.text_branch(texts))
            outputs["text"] = f
            features.append(f)

        fusion_feat = self.fusion_head(features)
        outputs["fusion"] = fusion_feat

        logits, balance_loss = self._apply_classifier(fusion_feat)
        outputs["logits"] = logits
        if balance_loss is not None:
            outputs["balance_loss"] = balance_loss
        return outputs

    def _apply_classifier(self, features):
        if hasattr(self.classifier, "forward_from_features"):
            return self.classifier.forward_from_features(features)
        return self.classifier(features), None