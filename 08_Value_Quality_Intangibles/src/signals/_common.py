"""
Helpers shared by several signals (the leading underscore keeps this file out of
the list of signals in src/signals/__init__.py).

- to_matrix / from_matrix: move a column of the monthly panel to a
  (months x stocks) matrix and back. Missing stock-months stay NaN, so "k months
  ago" in the matrix is always exactly k calendar months ago.
- factor_residuals: every stock's return minus what a factor model explains,
  month by month, with betas estimated on the previous WINDOW months only.
  Computed once and cached in data/interim/ (several signals use it).
- cross_rank: rank a signal within each month, from -0.5 to +0.5.
- seasonal_spread: same-calendar-month minus other-months average return
  (Heston and Sadka 2008; Keloharju, Linnainmaa and Nyberg 2016), used by
  seasonality_idio.py.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.panel import CRSP_FILE, INTERIM_DIR
from src.evaluation.alpha import MODELS, load_factors

BASE_DIR = next(
    (p for p in Path(__file__).resolve().parents if (p / "data" / "raw").is_dir()),
    Path(__file__).resolve().parents[2],
)
FACTORS_FILE = BASE_DIR / "data" / "external" / "factors_monthly.csv"


def month_id(dates: pd.Series) -> np.ndarray:
    """Jan 1963 -> 23556, Feb 1963 -> 23557, ... (one number per calendar month)."""
    return (dates.dt.year * 12 + dates.dt.month - 1).to_numpy()


def to_matrix(panel: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(matrix, month ids, permnos): matrix[i, j] = value of `col` for stock j in month i."""
    mid = month_id(panel["MthCalDt"])
    months = np.arange(mid.min(), mid.max() + 1)
    permnos, col_idx = np.unique(panel["PERMNO"].to_numpy(), return_inverse=True)
    M = np.full((len(months), len(permnos)), np.nan)
    M[mid - months[0], col_idx] = panel[col].to_numpy(dtype=float)
    return M, months, permnos


def from_matrix(panel: pd.DataFrame, M: np.ndarray, months: np.ndarray, permnos: np.ndarray) -> pd.Series:
    """Read a (months x stocks) matrix back into one value per panel row."""
    mid = month_id(panel["MthCalDt"])
    rows = mid - months[0]
    cols = np.searchsorted(permnos, panel["PERMNO"].to_numpy())
    ok = (rows >= 0) & (rows < len(months)) & (cols < len(permnos))
    ok &= permnos[np.minimum(cols, len(permnos) - 1)] == panel["PERMNO"].to_numpy()
    out = np.full(len(panel), np.nan)
    out[ok] = M[rows[ok], cols[ok]]
    return pd.Series(out, index=panel.index)


def factor_residuals(panel: pd.DataFrame, model: str = "FF5", window: int = 36):
    """Residual return of every stock-month: excess return minus alpha + betas x factors,
    with alpha and betas estimated by OLS on the `window` months ending in that month
    (only data known at the end of the month). A stock needs `window` consecutive
    months of returns, otherwise its residual is NaN.
    Returns (matrix, month ids, permnos), like to_matrix."""
    cache = INTERIM_DIR / f"residuals_{model}_{window}.npz"
    sources = [CRSP_FILE, FACTORS_FILE]
    if cache.exists() and all(cache.stat().st_mtime > s.stat().st_mtime for s in sources if s.exists()):
        z = np.load(cache)
        return z["R"], z["months"], z["permnos"]

    spec = MODELS[model]
    factors = load_factors(FACTORS_FILE)
    R_raw, months, permnos = to_matrix(panel, "MthRet")
    idx = pd.PeriodIndex([pd.Period(year=m // 12, month=m % 12 + 1, freq="M") for m in months])
    f = factors.reindex(idx)
    F = f[spec["factors"]].to_numpy()
    E = R_raw - f[[spec["rf"]]].to_numpy()  # excess returns

    R = np.full_like(E, np.nan)
    for t in range(window - 1, len(months)):
        Ft = F[t - window + 1: t + 1]
        if np.isnan(Ft).any():
            continue
        Y = E[t - window + 1: t + 1]
        ok = ~np.isnan(Y).any(axis=0)
        if not ok.any():
            continue
        X = np.column_stack([np.ones(window), Ft])
        beta = np.linalg.lstsq(X, Y[:, ok], rcond=None)[0]
        R[t, ok] = Y[-1, ok] - X[-1] @ beta  # residual of month t (alpha removed too)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, R=R.astype(np.float32), months=months, permnos=permnos)
    return R, months, permnos


def cross_rank(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Percentile rank within each month, centred: from -0.5 (lowest) to +0.5 (highest)."""
    return values.groupby(panel["MthCalDt"]).rank(pct=True) - 0.5


YEARS, MIN_YEARS = 20, 5  # seasonal_spread: look back 20 years, need at least 5


def seasonal_spread(M: np.ndarray, years: int = YEARS, min_years: int = MIN_YEARS) -> np.ndarray:
    """Same-calendar-month average minus other-months average over the last `years`
    years, for every row t of a (months x stocks) matrix. Uses rows t-12*years+1..t only."""
    T = M.shape[0]
    filled = np.nan_to_num(M)
    has = (~np.isnan(M)).astype(float)
    # running sums over the last 12*years months (rows t-12*years+1 .. t)
    cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(filled, axis=0)])
    cn = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(has, axis=0)])
    L = 12 * years
    out = np.full(M.shape, np.nan)
    for t in range(11, T):
        lo = max(0, t - L + 1)
        tot_s, tot_n = cs[t + 1] - cs[lo], cn[t + 1] - cn[lo]
        lags = np.arange(t - 11, lo - 1, -12)          # t-11, t-23, ... (same month as t+1)
        same_s, same_n = filled[lags].sum(axis=0), has[lags].sum(axis=0)
        other_n = tot_n - same_n
        ok = (same_n >= min_years) & (other_n >= 11 * min_years)
        out[t, ok] = same_s[ok] / same_n[ok] - (tot_s[ok] - same_s[ok]) / other_n[ok]
    return out
