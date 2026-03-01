# -*-coding: utf-8 -*-

import torch
from torch import nn
from torchinfo import summary

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * torch.sigmoid(gate)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor, dropout):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, dim, heads, dropout):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pe = self._build_pe(dim, max_len)  # [max_len, 1, dim]
        self.register_buffer("pe", pe, persistent=False)

    def _build_pe(self, dim, max_len):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # [max_len, 1, dim]

    def _maybe_extend(self, need_len, device):
        if need_len <= self.pe.size(0):
            return
        new_pe = self._build_pe(dim=self.pe.size(-1), max_len=need_len)
        self.register_buffer("pe", new_pe.to(device), persistent=False)

    def forward(self, x):
        T = x.size(0)
        self._maybe_extend(T, x.device)
        return x + self.pe[:T].to(dtype=x.dtype)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size, dropout):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * expansion_factor * 2, kernel_size=1)
        self.glu = GLU()
        self.depthwise_conv = nn.Conv1d(dim * expansion_factor, dim * expansion_factor, kernel_size=kernel_size,
                                        groups=dim * expansion_factor, padding=(kernel_size - 1) // 2)
        self.batch_norm = nn.BatchNorm1d(dim * expansion_factor)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim * expansion_factor, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).transpose(0, 1)  # (seq_len, batch_size, dim)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, dim, ff_expansion_factor, conv_expansion_factor, heads, conv_kernel_size, dropout, max_len):
        super(ConformerBlock, self).__init__()
        self.ff_module1 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.mhsa_module = MultiHeadSelfAttentionModule(dim, heads, dropout)
        self.conv_module = ConvolutionModule(dim, conv_expansion_factor, conv_kernel_size, dropout)
        self.ff_module2 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff_module1(x)
        x = x + self.mhsa_module(x)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff_module2(x)
        x = self.layer_norm(x)
        return x

class AttentivePool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )
    def forward(self, x, lengths=None):
        B, T, D = x.shape
        scores = self.score(x).squeeze(-1)
        if lengths is not None:
            device = x.device
            mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
        w = torch.softmax(scores, dim=1)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)
        return pooled

class ConformerClassifier(nn.Module):
    def __init__(self, n_classes=5, f_in=128, d_model=128, heads=8,
                 n_blocks=4, ff_exp=4, conv_exp=2, ksize=31, dropout=0.1, max_len=10000):
        super().__init__()
        self.proj = nn.Linear(f_in, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, ff_exp, conv_exp, heads, ksize, dropout, max_len)
            for _ in range(n_blocks)
        ])
        self.pool = AttentivePool(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

    def forward(self, mel, lengths=None):
        # mel shape: (batch, 1, n_mels, frames) -> (batch, n_mels, frames)
        mel = mel.squeeze(1).transpose(1, 2)
        x = self.proj(mel)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(1, 0, 2)
        if lengths is None:
            lengths = x.new_full((x.size(0),), x.size(1), dtype=torch.long)

        feature = self.pool(x, lengths)
        logits = self.head(feature)
        return logits