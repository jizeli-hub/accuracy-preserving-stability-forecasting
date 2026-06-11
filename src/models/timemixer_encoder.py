"""Lightweight multiscale MLP temporal encoder for demand histories.

The encoder learns from historical demand windows, mixes multiple temporal
resolutions using vectorized pooling, and exposes hidden temporal embeddings for
downstream tree models. This module intentionally avoids heavyweight deep
learning dependencies to ensure scalability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.utils.config import RANDOM_SEED


LOGGER = logging.getLogger(__name__)


@dataclass
class MultiscaleEncoderConfig:
    """Configuration for the lightweight temporal encoder."""

    history_length: int = 28
    embedding_dim: int = 16
    max_iter: int = 80
    max_train_rows: int = 50_000
    random_seed: int = RANDOM_SEED


class MultiscaleTemporalEncoder:
    """Multi-scale temporal encoder trained on historical demand sequences."""

    def __init__(self, config: MultiscaleEncoderConfig | None = None) -> None:
        self.config = config or MultiscaleEncoderConfig()
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, self.config.embedding_dim),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=self.config.max_iter,
            random_state=self.config.random_seed,
            early_stopping=True,
            n_iter_no_change=8,
        )
        self.is_fit = False

    def fit(self, sequences: np.ndarray, targets: np.ndarray) -> "MultiscaleTemporalEncoder":
        """Train the encoder to forecast demand from historical sequences."""
        mixed = self._mix_temporal_scales(sequences)
        targets = np.asarray(targets, dtype=float)

        if len(mixed) > self.config.max_train_rows:
            rng = np.random.default_rng(self.config.random_seed)
            indices = rng.choice(len(mixed), size=self.config.max_train_rows, replace=False)
            mixed = mixed[indices]
            targets = targets[indices]

        LOGGER.info("Training multiscale encoder on %s windows", len(mixed))
        scaled = self.scaler.fit_transform(mixed)
        self.model.fit(scaled, targets)
        self.is_fit = True
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Forecast demand directly from temporal sequences."""
        self._check_fit()
        mixed = self._mix_temporal_scales(sequences)
        return self.model.predict(self.scaler.transform(mixed)).clip(min=0)

    def transform(self, sequences: np.ndarray) -> pd.DataFrame:
        """Return learned temporal embeddings for each sequence."""
        self._check_fit()
        mixed = self._mix_temporal_scales(sequences)
        activations = self.scaler.transform(mixed)

        for weights, bias in zip(self.model.coefs_[:-1], self.model.intercepts_[:-1]):
            activations = np.maximum(0, activations @ weights + bias)

        columns = [f"multiscale_emb_{index:02d}" for index in range(activations.shape[1])]
        return pd.DataFrame(activations, columns=columns)

    def _check_fit(self) -> None:
        if not self.is_fit:
            raise ValueError("MultiscaleTemporalEncoder must be fit before use.")

    def _mix_temporal_scales(self, sequences: np.ndarray) -> np.ndarray:
        sequences = np.asarray(sequences, dtype=float)
        if sequences.ndim != 2:
            raise ValueError("Expected a 2D array of demand history sequences.")

        scale_features = [sequences]
        for window in (2, 4, 7, 14):
            scale_features.append(_vectorized_rolling_pool(sequences, window=window, reducer="mean"))
            scale_features.append(_vectorized_rolling_pool(sequences, window=window, reducer="std"))

        scale_features.append(np.diff(sequences, axis=1, prepend=sequences[:, :1]))
        return np.concatenate(scale_features, axis=1)


def _vectorized_rolling_pool(sequences: np.ndarray, window: int, reducer: str) -> np.ndarray:
    """Pool each sequence over non-overlapping windows using vectorization."""
    n_samples, length = sequences.shape
    # Trim to multiple of window
    trimmed_len = (length // window) * window
    reshaped = sequences[:, :trimmed_len].reshape(n_samples, -1, window)
    
    if reducer == "mean":
        return reshaped.mean(axis=2)
    elif reducer == "std":
        return reshaped.std(axis=2)
    else:
        raise ValueError(f"Unknown reducer: {reducer}")


def add_demand_history_sequences(
    frame: pd.DataFrame,
    history_length: int = 28,
    group_col: str = "id",
    target_col: str = "demand",
) -> pd.DataFrame:
    """Add dense historical demand lag columns used by the encoder."""
    output = frame.copy()
    grouped = output.groupby(group_col, sort=False)[target_col]
    for lag in range(1, history_length + 1):
        output[f"{target_col}_history_lag_{lag}"] = grouped.shift(lag)
    return output


def history_feature_columns(history_length: int = 28, target_col: str = "demand") -> list[str]:
    """Return ordered history lag column names."""
    return [f"{target_col}_history_lag_{lag}" for lag in range(1, history_length + 1)]
