"""Classical baseline models for M5 demand forecasting."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    """Minimal interface shared by baseline forecasters."""

    def fit(self, y: pd.Series) -> "Forecaster":
        """Fit the forecaster on a univariate history."""

    def predict(self, horizon: int) -> np.ndarray:
        """Predict the next horizon values."""


@dataclass
class NaiveLastValueForecaster:
    """Forecast every future step as the most recent observed value."""

    last_value_: float | None = None

    def fit(self, y: pd.Series) -> "NaiveLastValueForecaster":
        clean_y = y.dropna()
        if clean_y.empty:
            raise ValueError("Cannot fit NaiveLastValueForecaster on an empty series.")
        self.last_value_ = float(clean_y.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_value_ is None:
            raise ValueError("Forecaster must be fit before calling predict.")
        return np.repeat(self.last_value_, horizon)


@dataclass
class MovingAverageForecaster:
    """Forecast every future step as the trailing average demand."""

    window: int = 28
    mean_value_: float | None = None

    def fit(self, y: pd.Series) -> "MovingAverageForecaster":
        clean_y = y.dropna()
        if clean_y.empty:
            raise ValueError("Cannot fit MovingAverageForecaster on an empty series.")
        self.mean_value_ = float(clean_y.tail(self.window).mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.mean_value_ is None:
            raise ValueError("Forecaster must be fit before calling predict.")
        return np.repeat(self.mean_value_, horizon)


def seasonal_naive_forecast(y: pd.Series, horizon: int, season_length: int = 7) -> np.ndarray:
    """Repeat the last observed seasonal cycle into the forecast horizon."""
    clean_y = y.dropna()
    if len(clean_y) < season_length:
        raise ValueError("Series length must be at least the requested season length.")

    seasonal_values = clean_y.tail(season_length).to_numpy(dtype=float)
    repeats = int(np.ceil(horizon / season_length))
    return np.tile(seasonal_values, repeats)[:horizon]

