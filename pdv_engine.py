"""
pdv_engine.py — Path-Dependent Volatility Engine
==================================================

Theory
------
Guyon & Lekeufack (2023) demonstrated empirically that realised volatility
depends on the recent *path* of returns, not merely their level. Specifically:

    σ²(t) ≈ a₀ + a₁ · K₁(r, t) + a₂ · K₂(r², t)

Where K₁ and K₂ are exponentially-weighted kernels over the return path:

    K₁(r, t) = Σⱼ w₁ⱼ · r(t−j)        (signed returns — leverage/momentum)
    K₂(r², t) = Σⱼ w₂ⱼ · r²(t−j)       (squared returns — vol-of-vol clustering)

With exponential weights:
    w₁ⱼ = (1−α₁) · α₁ʲ    (fast decay, k1 lags)
    w₂ⱼ = (1−α₂) · α₂ʲ    (slow decay, k2 lags)

Key empirical findings (Guyon & Lekeufack 2023):
1. K₁ is *negative*: recent negative returns → higher future vol (leverage effect)
2. K₂ is *positive*: recent high squared returns → higher future vol (clustering)
3. PDV outperforms GARCH and rough vol at short horizons (1–5 days)

Score Construction
------------------
For ETF ranking, we invert the vol forecast into a signal:

1. pdv_ratio    = PDV-forecast vol / realised vol (EWMA)
                  < 1.0 → PDV expects cooling (positive signal)
                  > 1.0 → PDV expects vol expansion (negative signal)

2. path_sign    = sign(K₁) negated
                  Recent negative returns (negative K₁) → positive score
                  (contrarian: vol spike often precedes reversal)

3. vol_surprise = (EWMA_vol − PDV_vol) / EWMA_vol
                  PDV underpredicts → regime shift, avoid
                  PDV overpredicts → potential mean reversion, favour

Composite score = weighted blend, cross-sectionally z-scored per universe/window.

References
----------
- Guyon, J. & Lekeufack, J. (2023). Volatility is (mostly) path-dependent.
  Quantitative Finance, 23(9), 1221–1258.
- Gatheral, J. et al. (2018). Volatility is rough. Quantitative Finance.
- Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance.
"""

import numpy as np
import pandas as pd
from typing import Tuple

import config


# ── Exponential kernel weights ────────────────────────────────────────────────

def _exp_weights(alpha: float, n_lags: int) -> np.ndarray:
    """
    Normalised exponential weights: w_j = (1−α) · αʲ  for j=0..n_lags-1.
    Returns shape (n_lags,), sums to ~1.
    """
    j = np.arange(n_lags)
    w = (1 - alpha) * (alpha ** j)
    return w / w.sum()


# ── Path kernels ──────────────────────────────────────────────────────────────

def _compute_k1(returns: np.ndarray, alpha: float, k: int) -> float:
    """
    K₁(r, t) = Σⱼ w₁ⱼ · r(t−j)   (signed returns over k lags)
    Negative K₁ → recent drawdown → leverage effect → higher vol.
    """
    if len(returns) < k:
        return np.nan
    w = _exp_weights(alpha, k)
    return float(np.dot(w, returns[-k:][::-1]))


def _compute_k2(returns: np.ndarray, alpha: float, k: int) -> float:
    """
    K₂(r², t) = Σⱼ w₂ⱼ · r²(t−j)  (squared returns over k lags)
    Positive K₂ → recent high variance → clustering → higher vol.
    """
    if len(returns) < k:
        return np.nan
    w = _exp_weights(alpha, k)
    return float(np.dot(w, returns[-k:][::-1] ** 2))


# ── PDV forecast ──────────────────────────────────────────────────────────────

def _pdv_vol_forecast(
    returns: np.ndarray,
    alpha1: float,
    alpha2: float,
    k1: int,
    k2: int,
) -> Tuple[float, float, float]:
    """
    Fit and forecast the PDV model over a return array.

    Returns
    -------
    pdv_vol   : annualised PDV-forecast volatility
    k1_val    : K₁ value (signed path momentum)
    k2_val    : K₂ value (squared path variance)
    """
    k1_val = _compute_k1(returns, alpha1, k1)
    k2_val = _compute_k2(returns, alpha2, k2)

    if np.isnan(k1_val) or np.isnan(k2_val):
        return np.nan, np.nan, np.nan

    # PDV variance = |a₁·K₁| + a₂·K₂  (sign of K₁ enters via leverage effect)
    # We use the absolute values for vol forecast (K₁ contributes via |leverage|)
    # and keep K₁ sign for the path_sign component separately
    pdv_var = abs(k1_val) + k2_val
    pdv_vol = float(np.sqrt(max(pdv_var, 1e-10)) * np.sqrt(252))

    return pdv_vol, k1_val, k2_val


def _ewma_vol(returns: np.ndarray, span: int) -> float:
    """Annualised EWMA volatility."""
    if len(returns) < 5:
        return np.nan
    alpha_ewma = 2.0 / (span + 1)
    var = float(pd.Series(returns).ewm(alpha=alpha_ewma).var().iloc[-1])
    return float(np.sqrt(max(var, 1e-10)) * np.sqrt(252))


# ── Score components ──────────────────────────────────────────────────────────

def _pdv_ratio_score(pdv_vol: float, ewma_vol: float) -> float:
    """
    pdv_ratio = PDV_vol / EWMA_vol
    < 1 → PDV predicts calming    → positive (buy signal)
    > 1 → PDV predicts expansion  → negative (avoid)
    Score = -(pdv_ratio − 1)  so calming → positive
    """
    if np.isnan(pdv_vol) or np.isnan(ewma_vol) or ewma_vol < 1e-10:
        return 0.0
    ratio = pdv_vol / ewma_vol
    return float(-(ratio - 1.0))


def _path_sign_score(k1_val: float) -> float:
    """
    path_sign = -sign(K₁)
    Negative K₁ (recent drawdown) → leverage effect → vol spike → contrarian buy
    Positive K₁ (recent run-up)   → complacency     → vol drop  → momentum avoid
    """
    if np.isnan(k1_val):
        return 0.0
    return float(-np.sign(k1_val))


def _vol_surprise_score(pdv_vol: float, ewma_vol: float) -> float:
    """
    vol_surprise = (EWMA − PDV) / EWMA
    Positive → EWMA overstates vol vs PDV → potential reversion → positive signal
    Negative → PDV overstates vs EWMA   → regime shift coming → negative signal
    """
    if np.isnan(pdv_vol) or np.isnan(ewma_vol) or ewma_vol < 1e-10:
        return 0.0
    return float((ewma_vol - pdv_vol) / ewma_vol)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_pdv_scores(
    prices:  pd.DataFrame,
    tickers: list,
    window:  int,
) -> pd.Series:
    """
    Compute PDV composite scores for all ETFs in the universe.

    Parameters
    ----------
    prices  : DataFrame of closing prices, DatetimeIndex
    tickers : list of ETF tickers in this universe
    window  : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite PDV z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + max(config.PDV_K1, config.PDV_K2) + 10
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    raw_scores = {}

    for ticker in avail:
        price_series = prices[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        # Log returns over the window
        log_ret = np.log(price_series / price_series.shift(1)).dropna()
        ret_window = log_ret.iloc[-window:].values

        if len(ret_window) < config.MIN_OBS:
            continue

        # PDV forecast
        pdv_vol, k1_val, k2_val = _pdv_vol_forecast(
            returns = ret_window,
            alpha1  = config.PDV_ALPHA1,
            alpha2  = config.PDV_ALPHA2,
            k1      = config.PDV_K1,
            k2      = config.PDV_K2,
        )

        if np.isnan(pdv_vol):
            continue

        # EWMA benchmark vol
        ewma_vol = _ewma_vol(ret_window, span=min(21, len(ret_window) // 2))

        # Three score components
        s_ratio    = _pdv_ratio_score(pdv_vol, ewma_vol)
        s_path     = _path_sign_score(k1_val)
        s_surprise = _vol_surprise_score(pdv_vol, ewma_vol)

        composite = (
            config.WEIGHT_PDV_RATIO    * s_ratio
            + config.WEIGHT_PATH_SIGN    * s_path
            + config.WEIGHT_VOL_SURPRISE * s_surprise
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)

    # Cross-sectional z-score
    mu  = scores.mean()
    std = scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)

    return (scores - mu) / std
