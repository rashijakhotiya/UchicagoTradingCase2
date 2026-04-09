from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


N_ASSETS = 25
TICKS_PER_DAY = 30
ASSET_COLUMNS = tuple(f"A{i:02d}" for i in range(N_ASSETS))


@dataclass(frozen=True)
class PublicMeta:
    """Per-asset metadata visible to participants."""
    sector_id: np.ndarray
    spread_bps: np.ndarray
    borrow_bps_annual: np.ndarray


def load_prices(path: str = "prices.csv") -> np.ndarray:
    """Load the price matrix from CSV. Returns shape (n_ticks, 25)."""
    df = pd.read_csv(path, index_col="tick")
    return df[list(ASSET_COLUMNS)].to_numpy(dtype=float)


def load_meta(path: str = "meta.csv") -> PublicMeta:
    """Load asset metadata from CSV."""
    df = pd.read_csv(path)
    return PublicMeta(
        sector_id=df["sector_id"].to_numpy(dtype=int),
        spread_bps=df["spread_bps"].to_numpy(dtype=float),
        borrow_bps_annual=df["borrow_bps_annual"].to_numpy(dtype=float),
    )


class StrategyBase:
    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        pass

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        raise NotImplementedError


class MyStrategy(StrategyBase):
    """
    Signal-based portfolio strategy for UTC 2026 Case 2.

    Signals:
    - short-term momentum
    - short-term mean reversion
    - sector-relative momentum

    Portfolio construction:
    - volatility scaling
    - borrow-cost penalty on short ideas
    - turnover control to reduce transaction costs
    """

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        self.meta = meta
        self.prev_weights = np.zeros(N_ASSETS, dtype=float)

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        prices = np.asarray(price_history, dtype=float)

        # Safety: not enough history yet
        if prices.shape[0] < 2 * TICKS_PER_DAY:
            return np.ones(N_ASSETS, dtype=float) / N_ASSETS

        # Tick log returns
        log_returns = np.zeros_like(prices)
        log_returns[1:] = np.log(prices[1:] / prices[:-1])

        # Only use full days
        n_days = prices.shape[0] // TICKS_PER_DAY
        usable_ticks = n_days * TICKS_PER_DAY
        if n_days < 10:
            return np.ones(N_ASSETS, dtype=float) / N_ASSETS

        daily_log_returns = log_returns[:usable_ticks].reshape(n_days, TICKS_PER_DAY, N_ASSETS).sum(axis=1)

        # Convert daily log returns to simple returns for slightly more intuitive sizing
        daily_returns = np.exp(daily_log_returns) - 1.0

        # Signal 1: 5-day momentum
        momentum = np.mean(daily_returns[-5:], axis=0)

        # Signal 2: 10-day mean reversion via z-score of latest daily return
        reversion_window = daily_returns[-10:]
        mean_10 = np.mean(reversion_window, axis=0)
        std_10 = np.std(reversion_window, axis=0) + 1e-8
        latest_ret = daily_returns[-1]
        zscore = (latest_ret - mean_10) / std_10
        mean_reversion = -zscore

        # Signal 3: sector-relative momentum
        sector_signal = np.zeros(N_ASSETS, dtype=float)
        for sector in np.unique(meta.sector_id):
            idx = np.where(meta.sector_id == sector)[0]
            sector_avg = np.mean(momentum[idx])
            sector_signal[idx] = momentum[idx] - sector_avg

        # Combine signals
        alpha = (
            0.50 * momentum +
            0.30 * mean_reversion +
            0.20 * sector_signal
        )

        # Volatility scaling using recent 10-day vol
        vol = np.std(daily_returns[-10:], axis=0) + 1e-6
        alpha = alpha / vol

        # Penalize assets with high borrow cost, especially to reduce bad shorting
        borrow_cost = meta.borrow_bps_annual / 1e4
        alpha = alpha - 0.10 * borrow_cost

        # Normalize to gross exposure <= 1
        gross_alpha = np.sum(np.abs(alpha))
        if gross_alpha > 0:
            target_weights = alpha / gross_alpha
        else:
            target_weights = np.zeros(N_ASSETS, dtype=float)

        # Turnover control:
        # 1. ignore tiny changes
        # 2. smooth changes toward target
        delta = target_weights - self.prev_weights
        small_change_mask = np.abs(delta) < 0.02
        target_weights[small_change_mask] = self.prev_weights[small_change_mask]

        weights = 0.70 * self.prev_weights + 0.30 * target_weights

        # Final gross-exposure enforcement
        gross = np.sum(np.abs(weights))
        if gross > 1.0:
            weights = weights / gross

        self.prev_weights = weights.copy()
        return weights


def create_strategy() -> StrategyBase:
    """Entry point called by validate.py."""
    return MyStrategy()
