"""
Benchmark alphas: time-series regressions of portfolio returns on factor models.

For each model we run, month by month over the sample:
    R_p - RF = alpha + beta_1 * F_1 + ... + beta_K * F_K + error
and report alpha with a Newey-West t-statistic (robust to autocorrelation and
heteroskedasticity). For a long-short portfolio (e.g. decile 1 minus decile 10)
we do NOT subtract the risk-free rate: it is already a zero-cost return.

We also run the GRS test (slide 7): are the alphas of ALL test portfolios
jointly zero?

Each model uses the risk-free rate of its own library (French RF for the
Fama-French models, Q_RF for the q-factor models), because each market factor
is defined in excess of its own risk-free rate.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

MODELS: Dict[str, Dict] = {
    "CAPM": {"factors": ["MKT_RF"], "rf": "RF"},
    "FF3": {"factors": ["MKT_RF", "SMB", "HML"], "rf": "RF"},
    "FF4": {"factors": ["MKT_RF", "SMB", "HML", "MOM"], "rf": "RF"},  # Carhart
    "FF5": {"factors": ["MKT_RF", "SMB", "HML", "RMW", "CMA"], "rf": "RF"},
    "FF6": {"factors": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"], "rf": "RF"},
    "q4": {"factors": ["Q_MKT", "Q_ME", "Q_IA", "Q_ROE"], "rf": "Q_RF"},
    "q5": {"factors": ["Q_MKT", "Q_ME", "Q_IA", "Q_ROE", "Q_EG"], "rf": "Q_RF"},
    # the two hardest models plus Ken French's short-term reversal factor (STREV),
    # for signals that use last month's return (e.g. laggard)
    "FF6+STR": {"factors": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "STREV"], "rf": "RF"},
    "q5+STR": {"factors": ["Q_MKT", "Q_ME", "Q_IA", "Q_ROE", "Q_EG", "STREV"], "rf": "Q_RF"},
}


def load_factors(path) -> pd.DataFrame:
    """Read data/external/factors_monthly.csv with a monthly PeriodIndex."""
    f = pd.read_csv(path, index_col=0)
    f.index = pd.PeriodIndex(f.index, freq="M")
    return f


def load_portfolio_returns(path) -> pd.DataFrame:
    """Read a returns file saved by our scripts (months as rows, portfolios as columns)."""
    r = pd.read_csv(path, index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M")
    r.columns = [int(c) if str(c).isdigit() else c for c in r.columns]
    return r


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int):
    """OLS coefficients with Newey-West (Bartlett kernel) standard errors.
    X must already contain the column of ones for the intercept."""
    T = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    scores = X * resid[:, None]                 # one row per month
    S = scores.T @ scores                       # lag 0
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)           # Bartlett weight
        gamma = scores[lag:].T @ scores[:-lag]
        S += weight * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - resid.var() / y.var()
    return beta, se, resid, r2


def regress(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str,
    long_short: bool = False,
    nw_lags: int = 6,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict:
    """Regress one portfolio on one factor model. Alpha is returned in % per month."""
    spec = MODELS[model]
    cols = spec["factors"] + [spec["rf"]]
    data = pd.concat([returns.rename("ret"), factors[cols]], axis=1).sort_index().loc[start:end].dropna()

    y = data["ret"].to_numpy()
    if not long_short:
        y = y - data[spec["rf"]].to_numpy()     # excess return
    X = np.column_stack([np.ones(len(data)), data[spec["factors"]].to_numpy()])

    beta, se, _, r2 = newey_west_ols(y, X, nw_lags)
    out = {
        "alpha %": beta[0] * 100,
        "t(alpha)": beta[0] / se[0],
        "R2": r2,
        "months": len(data),
        "start": str(data.index.min()),
        "end": str(data.index.max()),
    }
    for name, b, s in zip(spec["factors"], beta[1:], se[1:]):
        out[f"b_{name}"] = b
        out[f"t_{name}"] = b / s
    return out


def grs_test(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    model: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict:
    """Gibbons, Ross and Shanken (1989) test that all alphas are jointly zero.
    GRS = (T-N-K)/N * alpha' Sigma^-1 alpha / (1 + mu' Omega^-1 mu)  ~  F(N, T-N-K)
    Sigma: residual covariance (divided by T-K-1), Omega: factor covariance
    (divided by T, as in the original paper), mu: factor means."""
    spec = MODELS[model]
    cols = spec["factors"] + [spec["rf"]]
    data = pd.concat([returns, factors[cols]], axis=1).sort_index().loc[start:end].dropna()

    Y = data[returns.columns].to_numpy() - data[[spec["rf"]]].to_numpy()  # excess returns
    F = data[spec["factors"]].to_numpy()
    T, N, K = Y.shape[0], Y.shape[1], F.shape[1]

    X = np.column_stack([np.ones(T), F])
    coefs = np.linalg.lstsq(X, Y, rcond=None)[0]
    alphas = coefs[0]
    resid = Y - X @ coefs
    sigma = resid.T @ resid / (T - K - 1)
    mu = F.mean(axis=0)
    omega = np.cov(F, rowvar=False, ddof=0).reshape(K, K)

    grs = (T - N - K) / N * (alphas @ np.linalg.solve(sigma, alphas)) / (
        1 + mu @ np.linalg.solve(omega, mu)
    )
    p_value = stats.f.sf(grs, N, T - N - K)
    return {"GRS": grs, "p-value": p_value, "avg |alpha| %": np.abs(alphas).mean() * 100, "months": T}


def alpha_table(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    models: List[str],
    long_short_cols: List = (),
    nw_lags: int = 6,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Alpha (% per month) and its t-stat for every portfolio (columns of `returns`)
    under every model. Rows: model; columns: (portfolio, 'alpha %' / 't')."""
    pieces = {}
    for model in models:
        row = {}
        for col in returns.columns:
            res = regress(returns[col], factors, model, col in long_short_cols, nw_lags, start, end)
            row[(col, "alpha %")] = res["alpha %"]
            row[(col, "t")] = res["t(alpha)"]
        pieces[model] = row
    table = pd.DataFrame(pieces).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table