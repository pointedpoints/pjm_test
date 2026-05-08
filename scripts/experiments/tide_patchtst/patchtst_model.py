"""
PatchTST (Patch Time Series Transformer) — 2023

Paper: https://arxiv.org/abs/2211.14710

Architecture:
  - Split univariate time series into patches (subseries segments)
  - Each patch is linearly projected to a token
  - Token + positional encoding passed through Transformer Encoder
  - Decoder head maps encoder output to forecast horizon
  - Channel-independent: each variable processed separately

This implementation:
  - Pure PyTorch, no neuralforecast dependency
  - Compatible with the PJM backtest pipeline
  - Supports quantile regression for probabilistic forecasting
  - Channel-independent variant
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ─── Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ─── PatchTST Model ─────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """Channel-independent PatchTST.

    Each channel (variable) is processed independently through the same
    patching + transformer pipeline, with shared weights.

    Args:
        input_size: Length of input time series (history steps).
        h: Forecast horizon.
        patch_len: Length of each patch.
        stride: Stride between patches (patch_len for non-overlapping).
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
        quantiles: List of quantile levels for probabilistic output.
    """

    def __init__(
        self,
        input_size: int = 168,
        h: int = 24,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.quantiles = sorted(quantiles or [0.5])
        self.n_quantiles = len(self.quantiles)

        # Number of patches
        self.num_patches = (input_size - patch_len) // stride + 1
        # If input_size is not evenly divisible, pad
        if (input_size - patch_len) % stride != 0:
            self.num_patches += 1
        self.padded_len = (self.num_patches - 1) * stride + patch_len
        self.pad_amount = self.padded_len - input_size if self.padded_len > input_size else 0

        # Input projection: patch -> token
        self.input_proj = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Decoder head: from [CLS]-style pooled representation to forecast
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, h * self.n_quantiles),
        )

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _patching(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size) — univariate time series

        Returns:
            (batch, num_patches, patch_len)
        """
        # Pad if needed
        if self.pad_amount > 0:
            x = torch.nn.functional.pad(x, (0, self.pad_amount))  # pad right

        # Unfold into patches
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (batch, num_patches, patch_len)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size) — historical target values (univariate)

        Returns:
            (batch, h, n_quantiles) — forecast at each quantile
        """
        batch_size = x.shape[0]

        # Patching
        patches = self._patching(x)  # (batch, num_patches, patch_len)

        # Project to tokens
        tokens = self.input_proj(patches)  # (batch, num_patches, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, 1+num_patches, d_model)

        # Add positional encoding
        tokens = self.pos_encoding(tokens)

        # Transformer encoder
        encoded = self.transformer(tokens)  # (batch, 1+num_patches, d_model)
        encoded = self.norm(encoded)

        # Use CLS token for prediction
        cls_output = encoded[:, 0, :]  # (batch, d_model)

        # Decode to forecast
        out = self.head(cls_output).reshape(batch_size, self.h, self.n_quantiles)
        return out


# ─── Multi-variable PatchTST ────────────────────────────────────────────────
# For handling known future covariates, we concatenate covariate info

class MultiVarPatchTST(nn.Module):
    """PatchTST with future covariate conditioning.

    The univariate target is patched and encoded as in PatchTST.
    Future covariates are separately embedded and concatenated
    before the decoder head.
    """

    def __init__(
        self,
        input_size: int = 168,
        h: int = 24,
        num_future_covariates: int = 0,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.h = h
        self.num_future_covariates = num_future_covariates
        self.quantiles = sorted(quantiles or [0.5])
        self.n_quantiles = len(self.quantiles)

        # Base PatchTST (univariate)
        self.patchtst = PatchTST(
            input_size=input_size,
            h=h,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            quantiles=quantiles,
        )

        # Future covariate embedding (per step, then flatten)
        if num_future_covariates > 0:
            self.cov_embed = nn.Sequential(
                nn.Linear(num_future_covariates, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )
            # Override head to include cov info
            self.patchtst.head = nn.Sequential(
                nn.Linear(d_model * 2, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, h * self.n_quantiles),
            )

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_hist: (batch, input_size) — historical target
            x_future: (batch, h, C) — future covariates, or None

        Returns:
            (batch, h, n_quantiles)
        """
        batch_size = x_hist.shape[0]

        # Get base PatchTST representation (before head)
        patches = self.patchtst._patching(x_hist)
        tokens = self.patchtst.input_proj(patches)
        cls_tokens = self.patchtst.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.patchtst.pos_encoding(tokens)
        encoded = self.patchtst.transformer(tokens)
        encoded = self.patchtst.norm(encoded)
        cls_output = encoded[:, 0, :]  # (batch, d_model)

        if x_future is not None and self.num_future_covariates > 0:
            # Embed covariates: average over time, then combine
            cov_encoded = self.cov_embed(x_future)  # (batch, h, d_model)
            cov_pooled = cov_encoded.mean(dim=1)  # (batch, d_model)
            combined = torch.cat([cls_output, cov_pooled], dim=1)  # (batch, 2*d_model)
        else:
            combined = cls_output

        out = self.patchtst.head(combined).reshape(batch_size, self.h, self.n_quantiles)
        return out


# ─── Quantile Loss ──────────────────────────────────────────────────────────

def quantile_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(-1)
    quantiles_t = torch.tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype).view(1, 1, -1)
    error = y_true - y_pred
    loss = torch.max(quantiles_t * error, (quantiles_t - 1) * error)
    return loss.mean()


# ─── Training ───────────────────────────────────────────────────────────────

class PatchTSTTrainer:
    """Manual training loop for PatchTST."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        quantiles: list[float] | None = None,
    ):
        self.model = model
        self.quantiles = quantiles or [0.5]
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in dataloader:
            self.optimizer.zero_grad()
            # x: (batch, input_size)
            y_pred = self.model(x)  # (batch, h, n_quantiles)
            loss = quantile_loss(y_pred, y, self.quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = self.model(x)
                loss = quantile_loss(y_pred, y, self.quantiles)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)


# ─── Time Series Dataset ────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Sliding window for univariate PatchTST."""

    def __init__(
        self,
        target: np.ndarray,
        input_size: int = 168,
        h: int = 24,
        stride: int = 1,
    ):
        self.input_size = input_size
        self.h = h
        self.stride = stride
        n = len(target)
        self.indices = list(range(0, n - input_size - h + 1, stride))
        self.target = torch.from_numpy(target.astype(np.float32))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        x = self.target[i: i + self.input_size]       # (input_size,)
        y = self.target[i + self.input_size: i + self.input_size + self.h]  # (h,)
        return x, y


# ─── Pipeline Adapter ───────────────────────────────────────────────────────

@dataclass
class PatchTSTPipelineModel:
    """PatchTST model adapter for the PJM backtest pipeline."""

    h: int = 24
    input_size: int = 168
    patch_len: int = 12
    stride: int = 12
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    batch_size: int = 64
    early_stop_patience: int = 10
    quantiles: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995])
    name: str = "patchtst"

    def __post_init__(self) -> None:
        self._model: nn.Module | None = None
        self._trainer: PatchTSTTrainer | None = None
        self._is_fitted: bool = False

    def _build_model(self) -> PatchTST:
        return PatchTST(
            input_size=self.input_size,
            h=self.h,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            quantiles=self.quantiles,
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit PatchTST on training data (univariate — only uses y)."""
        target = train_df["y"].values.astype(np.float64)

        dataset = TimeSeriesDataset(
            target=target,
            input_size=self.input_size,
            h=self.h,
            stride=1,
        )

        if len(dataset) == 0:
            raise ValueError(
                f"Training data too short: need at least {self.input_size + self.h} rows, "
                f"got {len(train_df)}"
            )

        val_size = min(self.h * 14, len(dataset) // 5)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [len(dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
        )

        model = self._build_model()
        self._trainer = PatchTSTTrainer(
            model=model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            quantiles=self.quantiles,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            train_loss = self._trainer.train_epoch(train_loader)
            val_loss = self._trainer.validate(val_loader)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                break

        self._model = model
        self._is_fitted = True

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fit before predict.")

        # Extract last input_size steps
        hist_len = min(len(history_df), self.input_size)
        history_tail = history_df.iloc[-hist_len:]

        if hist_len < self.input_size:
            pad_len = self.input_size - hist_len
            target = np.concatenate([
                np.zeros(pad_len),
                history_tail["y"].values.astype(np.float64),
            ])
        else:
            target = history_tail["y"].values.astype(np.float64)

        x_hist = torch.from_numpy(target[-self.input_size:].astype(np.float32)).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            y_pred = self._model(x_hist)  # (1, h, n_quantiles)

        pred_array = y_pred.squeeze(0).numpy()
        future_ds = future_df["ds"].values[:self.h]
        n_quantiles = len(self.quantiles)

        rows = []
        for t in range(min(self.h, len(future_ds))):
            for q_idx in range(n_quantiles):
                rows.append({
                    "ds": future_ds[t],
                    "y_pred": float(pred_array[t, q_idx]),
                    "quantile": self.quantiles[q_idx],
                })

        result = pd.DataFrame(rows)
        if n_quantiles == 1 and self.quantiles[0] == 0.5:
            result = result.drop(columns=["quantile"])
        return result

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("No fitted model to save.")

        config = {
            "h": self.h,
            "input_size": self.input_size,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "early_stop_patience": self.early_stop_patience,
            "quantiles": self.quantiles,
        }

        torch.save(self._model.state_dict(), path / "model.pt")
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PatchTSTPipelineModel":
        config_path = path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        model = cls(
            h=config["h"],
            input_size=config["input_size"],
            patch_len=config["patch_len"],
            stride=config["stride"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            max_epochs=config["max_epochs"],
            batch_size=config["batch_size"],
            early_stop_patience=config["early_stop_patience"],
            quantiles=config["quantiles"],
        )

        pytorch_model = model._build_model()
        state_dict = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        pytorch_model.load_state_dict(state_dict)
        model._model = pytorch_model
        model._is_fitted = True

        return model

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self._model is not None:
            state["_model_weights"] = self._model.state_dict()
            state["_model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_model_weights" in state:
            model = self._build_model()
            model.load_state_dict(state["_model_weights"])
            self._model = model
            self._is_fitted = True
