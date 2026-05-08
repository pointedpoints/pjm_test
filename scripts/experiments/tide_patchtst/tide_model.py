"""
TiDE (Time-series Dense Encoder) — Google 2023

Paper: https://arxiv.org/abs/2304.08424

Architecture:
  - Encoder: MLP that compresses historical window + known future covariates
    into a low-dimensional latent representation
  - Decoder: MLP that projects the latent back to the forecast horizon
  - Residual connection: skip connection from input to output

This implementation:
  - Pure PyTorch, no neuralforecast dependency
  - Compatible with the PJM backtest pipeline via ForecastModel-like interface
  - Supports quantile regression for probabilistic forecasting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ─── PyTorch Dataset ────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Sliding window dataset for time series forecasting.

    For each window:
      x = [history_target (input_size steps) + future_known_covariates (h steps)]
      y = future_target (h steps)
    """

    def __init__(
        self,
        target: np.ndarray,
        future_covariates: np.ndarray | None = None,
        input_size: int = 168,
        h: int = 24,
        stride: int = 1,
    ):
        self.input_size = input_size
        self.h = h
        self.stride = stride

        n = len(target)
        windows = list(range(0, n - input_size - h + 1, stride))
        self.indices = windows
        self.target = torch.from_numpy(target.astype(np.float32)).unsqueeze(-1)  # (T, 1)
        if future_covariates is not None:
            self.future_cov = torch.from_numpy(future_covariates.astype(np.float32))  # (T, C)
        else:
            self.future_cov = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        # History: target values in [i, i+input_size)
        x_hist = self.target[i: i + self.input_size].squeeze(-1)  # (input_size,)

        # Future: covariates in [i+input_size, i+input_size+h)
        if self.future_cov is not None:
            x_future = self.future_cov[i + self.input_size: i + self.input_size + self.h]  # (h, C)
            x_future = x_future.flatten()  # (h*C,)
            x = torch.cat([x_hist, x_future])  # (input_size + h*C,)
        else:
            x = x_hist

        y = self.target[i + self.input_size: i + self.input_size + self.h].squeeze(-1)  # (h,)
        return x, y


# ─── TiDE Model (Pure PyTorch) ─────────────────────────────────────────────

class TiDE(nn.Module):
    """Time-series Dense Encoder from Google 2023.

    Args:
        input_size: Number of historical time steps.
        h: Forecast horizon.
        num_future_covariates: Number of known future covariates (per step).
        hidden_size: Hidden dimension of encoder/decoder MLPs.
        latent_size: Bottleneck dimension.
        num_encoder_layers: Number of encoder MLP layers.
        num_decoder_layers: Number of decoder MLP layers.
    """

    def __init__(
        self,
        input_size: int = 168,
        h: int = 24,
        num_future_covariates: int = 0,
        hidden_size: int = 256,
        latent_size: int = 64,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.num_future_covariates = num_future_covariates

        # Flattened input dimension
        input_dim = input_size + (h * num_future_covariates)

        # Encoder: MLP compressing to latent
        encoder_layers = []
        dim = input_dim
        for i in range(num_encoder_layers):
            next_dim = latent_size if i == num_encoder_layers - 1 else hidden_size
            encoder_layers.extend([
                nn.Linear(dim, next_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            dim = next_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: MLP from latent to forecast
        decoder_layers = []
        dim = latent_size
        for i in range(num_decoder_layers):
            next_dim = hidden_size if i < num_decoder_layers - 1 else h
            decoder_layers.extend([
                nn.Linear(dim, next_dim),
                nn.ReLU() if i < num_decoder_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if (dropout > 0 and i < num_decoder_layers - 1) else nn.Identity(),
            ])
            dim = next_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Temporal decoder (dense) — projects per-timestep for residual
        self.temporal_decoder = nn.Linear(latent_size, input_size)

        # Residual weight (learned scalar or fixed, per the paper)
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_hist: (batch, input_size) — history target values
            x_future: (batch, h, num_cov) — future covariates, or None

        Returns:
            (batch, h) — forecast
        """
        batch_size = x_hist.shape[0]

        if x_future is not None:
            x_future_flat = x_future.reshape(batch_size, -1)  # (batch, h*C)
            x = torch.cat([x_hist, x_future_flat], dim=1)  # (batch, input_size + h*C)
        else:
            x = x_hist

        # Encode
        latent = self.encoder(x)  # (batch, latent_size)

        # Decode: main forecast
        forecast = self.decoder(latent)  # (batch, h)

        # Temporal decode for residual connection
        temporal_proj = self.temporal_decoder(latent)  # (batch, input_size)

        # Residual: add weighted projection of last input_size//h steps averaged
        # Simplified residual: use the last h values of temporal_proj
        if self.input_size >= self.h:
            residual = temporal_proj[:, -self.h:]  # (batch, h)
        else:
            residual = temporal_proj  # shouldn't happen for our use case
        forecast = forecast + self.residual_weight * residual

        return forecast


class TideQuantileModel(nn.Module):
    """TiDE with multiple quantile heads for probabilistic forecasting."""

    def __init__(
        self,
        input_size: int = 168,
        h: int = 24,
        num_future_covariates: int = 0,
        hidden_size: int = 256,
        latent_size: int = 64,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.quantiles = sorted(quantiles or [0.5])
        self.base_model = TiDE(
            input_size=input_size,
            h=h,
            num_future_covariates=num_future_covariates,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        # Replace decoder's last layer to output h * n_quantiles
        # Instead, simpler: keep base and add a projection head
        in_features = latent_size
        self.quantile_head = nn.Linear(in_features, h * len(self.quantiles))

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            (batch, h, n_quantiles)
        """
        batch_size = x_hist.shape[0]

        if x_future is not None:
            x_future_flat = x_future.reshape(batch_size, -1)
            x = torch.cat([x_hist, x_future_flat], dim=1)
        else:
            x = x_hist

        latent = self.base_model.encoder(x)  # (batch, latent_size)
        forecast = self.base_model.decoder(latent)  # (batch, h)

        # Get temporal residual
        temporal_proj = self.base_model.temporal_decoder(latent)
        if self.base_model.input_size >= self.base_model.h:
            residual = temporal_proj[:, -self.base_model.h:]
        else:
            residual = temporal_proj
        point_forecast = forecast + self.base_model.residual_weight * residual

        # Quantile offsets
        quantile_out = self.quantile_head(latent).reshape(batch_size, self.base_model.h, -1)  # (batch, h, n_quantiles)

        # Combine: point forecast as base + quantile-specific adjustments
        # Center the quantile predictions around the point forecast
        result = point_forecast.unsqueeze(-1) + quantile_out  # (batch, h, n_quantiles)
        return result


# ─── Quantile Loss ──────────────────────────────────────────────────────────

def quantile_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    """Pinball loss for quantile regression.

    Args:
        y_pred: (batch, h, n_quantiles)
        y_true: (batch, h) or (batch, h, 1)
        quantiles: list of quantile levels

    Returns:
        Scalar loss
    """
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(-1)  # (batch, h, 1)
    quantiles_t = torch.tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype).view(1, 1, -1)
    error = y_true - y_pred
    loss = torch.max(quantiles_t * error, (quantiles_t - 1) * error)
    return loss.mean()


# ─── PyTorch Lightning Wrapper ──────────────────────────────────────────────
# Note: if pytorch_lightning is not available, we fall back to manual training

class TideTrainer:
    """Manual training loop for TiDE (avoids pytorch-lightning dependency)."""

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
            # Parse x into hist and future components
            # Our dataset returns (input_size + h*C,) flattened
            # We need to separate
            # For simplicity, assume no future covariates for now
            x_hist = x[:, :self.model.base_model.input_size]  # (batch, input_size)
            if self.model.base_model.num_future_covariates > 0:
                rest = x[:, self.model.base_model.input_size:]
                x_future = rest.reshape(
                    -1, self.model.base_model.h, self.model.base_model.num_future_covariates
                )
            else:
                x_future = None

            y_pred = self.model(x_hist, x_future)  # (batch, h, n_quantiles)
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
                x_hist = x[:, :self.model.base_model.input_size]
                if self.model.base_model.num_future_covariates > 0:
                    rest = x[:, self.model.base_model.input_size:]
                    x_future = rest.reshape(
                        -1, self.model.base_model.h, self.model.base_model.num_future_covariates
                    )
                else:
                    x_future = None
                y_pred = self.model(x_hist, x_future)
                loss = quantile_loss(y_pred, y, self.quantiles)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)


# ─── Sklearn-style Adapter for Backtest Pipeline ────────────────────────────

@dataclass
class TidePipelineModel:
    """
    TiDE model adapter for the PJM backtest pipeline.

    Implements the same interface as NHITSModel (fit/predict/save/load)
    but using pure PyTorch instead of neuralforecast.
    """
    h: int = 24
    input_size: int = 168
    hidden_size: int = 256
    latent_size: int = 64
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    batch_size: int = 64
    early_stop_patience: int = 10
    quantiles: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995])
    name: str = "tide"
    num_future_covariates: int = 0

    def __post_init__(self) -> None:
        self._model: nn.Module | None = None
        self._trainer: TideTrainer | None = None
        self._feature_columns: list[str] | None = None
        self._is_fitted: bool = False

    def _prepare_data(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract target and future covariates from panel dataframe."""
        # Target column
        target = train_df["y"].values.astype(np.float64)

        # Future covariates (we'll use future_exog + calendar features)
        # Determine which columns are "future-known" — those available for both
        # history and future periods
        future_cov_columns = [
            col for col in train_df.columns
            if col not in {"unique_id", "ds", "y", "date", "hour"}
            and not col.startswith("price_lag_")
            and not col.endswith("_lag_24")
        ]

        # Also exclude derived features that depend on y
        exclude_derived = {"prior_day_price_max_ramp", "spike_score"}
        future_cov_columns = [c for c in future_cov_columns if c not in exclude_derived]

        self._feature_columns = future_cov_columns

        if future_cov_columns:
            future_cov = train_df[future_cov_columns].values.astype(np.float64)
            self.num_future_covariates = len(future_cov_columns)
        else:
            future_cov = None
            self.num_future_covariates = 0

        return target, future_cov

    def _build_model(self) -> TideQuantileModel:
        return TideQuantileModel(
            input_size=self.input_size,
            h=self.h,
            num_future_covariates=self.num_future_covariates,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout,
            quantiles=self.quantiles,
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit TiDE on training data."""
        target, future_cov = self._prepare_data(train_df)

        # Build dataset
        dataset = TimeSeriesDataset(
            target=target,
            future_covariates=future_cov,
            input_size=self.input_size,
            h=self.h,
            stride=1,
        )

        if len(dataset) == 0:
            raise ValueError(
                f"Training data too short: need at least {self.input_size + self.h} rows, "
                f"got {len(train_df)}"
            )

        # Split into train/val
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

        # Build model
        model = self._build_model()
        self._trainer = TideTrainer(
            model=model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            quantiles=self.quantiles,
        )

        # Training loop
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
        """Generate predictions for the forecast horizon.

        Args:
            history_df: DataFrame with historical data (input_size steps).
            future_df: DataFrame with future covariates (h steps).

        Returns:
            DataFrame with columns [ds, y_pred, quantile] or [ds, y_pred].
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fit before predict.")

        # Extract last input_size steps of history
        hist_len = min(len(history_df), self.input_size)
        history_tail = history_df.iloc[-hist_len:]

        # Pad with zeros if shorter than input_size
        if hist_len < self.input_size:
            pad_len = self.input_size - hist_len
            target = np.concatenate([
                np.zeros(pad_len),
                history_tail["y"].values.astype(np.float64),
            ])
        else:
            target = history_tail["y"].values.astype(np.float64)

        # Future covariates
        if self._feature_columns and len(self._feature_columns) > 0:
            future_cov = future_df[self._feature_columns].values.astype(np.float64)
            if len(future_cov) < self.h:
                pad = np.zeros((self.h - len(future_cov), future_cov.shape[1]))
                future_cov = np.vstack([future_cov, pad])
        else:
            future_cov = None

        # Prepare input tensors
        x_hist = torch.from_numpy(target[-self.input_size:].astype(np.float32)).unsqueeze(0)  # (1, input_size)

        if future_cov is not None:
            x_future = torch.from_numpy(future_cov[:self.h].astype(np.float32)).unsqueeze(0)  # (1, h, C)
        else:
            x_future = None

        # Predict
        self._model.eval()
        with torch.no_grad():
            y_pred = self._model(x_hist, x_future)  # (1, h, n_quantiles)

        # Build output DataFrame
        pred_array = y_pred.squeeze(0).numpy()  # (h, n_quantiles)
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
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "early_stop_patience": self.early_stop_patience,
            "quantiles": self.quantiles,
            "num_future_covariates": self.num_future_covariates,
            "feature_columns": self._feature_columns,
        }

        # Save model weights
        torch.save(self._model.state_dict(), path / "model.pt")

        # Save config
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TidePipelineModel":
        config_path = path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        model = cls(
            h=config["h"],
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            latent_size=config["latent_size"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            max_epochs=config["max_epochs"],
            batch_size=config["batch_size"],
            early_stop_patience=config["early_stop_patience"],
            quantiles=config["quantiles"],
            num_future_covariates=config.get("num_future_covariates", 0),
        )
        model._feature_columns = config.get("feature_columns")

        # Rebuild and load weights
        pytorch_model = model._build_model()
        state_dict = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        pytorch_model.load_state_dict(state_dict)
        model._model = pytorch_model
        model._is_fitted = True

        return model

    # ─── ─── ───

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
