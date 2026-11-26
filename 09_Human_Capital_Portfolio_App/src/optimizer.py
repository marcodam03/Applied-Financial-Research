"""Optimization engine.

The optimizer chooses only the financial portfolio F, but judges it on total wealth
W = H + P + F: human capital H (the career), pension savings P and the portfolio F.
Every block reuses a course lab, with the lab's notation:

    covariance        Ledoit-Wolf shrinkage to constant correlation          LN3
    expected returns  Black-Litterman: market prior + optional client views  LN4
    portfolio         QP(x) = 0.5 x'Sigma x - gamma x'mu, on total wealth,
                      scipy minimize with Bounds and LinearConstraint        LN1
    gamma             fsolve on a target volatility (a 60/40 portfolio)       LN1
    shadow prices     KKT multipliers of the binding constraints             LN1
    risk              Euler risk contributions, on total wealth              LN5

Total-wealth problem. omega = (H + P) / W is the share of wealth
that cannot be traded, x the weights of the risky assets in the portfolio (the rest,
1 - 1'x, is cash at the CHF risk-free rate), R the monthly excess returns, R_NT the
return of the non-tradable wealth and sigma_H = Cov(R, R_NT):

    R_W = (1 - omega) x'R + omega R_NT
    minimise 0.5 Var(R_W) - gamma E(R_W)
           = 0.5 (1-omega)^2 x'Sigma x + (1-omega) omega x'sigma_H - gamma (1-omega) x'mu + constant

    no constraints:  x* = gamma/(1-omega) inv(Sigma) mu  -  omega/(1-omega) inv(Sigma) sigma_H
                          speculative demand                 hedging demand

With omega = 0 this is exactly the QP of the labs. The pension is credited at a
guaranteed rate, so it has no market risk: it enlarges omega but adds no hedging demand,
and omega x sigma_H = omega_H x Cov(R, R_career).

Units: monthly excess returns inside; annual numbers only in the
tables for the app. gamma and phi are the same with monthly or annual data.

Main functions:
    bl_expected_returns(cov, w_mkt, phi, P, Q, omega_views, tau)  -> ndarray
    optimal_portfolio(mu, cov, sigma_H, omega, gamma, constraints) -> ndarray
    risk_contributions(x, cov)                                      -> ndarray
For the app and the backtest:
    analyze_client(params)                   everything about one client (app pages 3, 4, 5)
    strategy_portfolios(...)                 weights of the strategies (A), (D) and (C)
    build_constraints, calibrate_gamma, shadow_prices,
    total_wealth_risk_contributions, certainty_equivalent, balance_sheet

Run from the project folder for a report on the default client and on every persona:
    python -m src.optimizer
"""
import copy

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, brentq, fsolve, minimize

from src.career import career_betas, career_covariance, human_capital, params_for_persona
from src.config import load_params

MONTHS = 12               # monthly data: mean x 12, variance x 12, volatility x sqrt(12)
BP = 10_000               # basis points in 1
PERCENT_POINT = 0.01      # marginal shadow prices are quoted for 1 percentage point of the portfolio
EMPLOYER = "EMPLOYER"     # column of the employer stock
CASH, CAREER, PENSION = "CASH", "CAREER", "PENSION"
STRATEGIES = {"A": "(A) standard: career ignored",
              "D": "(D) total wealth, no hedge",
              "C": "(C) total wealth"}

# Numerical tolerances (solver settings, not model assumptions)
ACTIVE_TOL = 1e-6         # after scipy, a constraint closer than this is a candidate binding constraint
FEASIBILITY_TOL = 1e-10   # allowed violation of a constraint
DUAL_TOL = 1e-10          # allowed negative multiplier (scaled units)
MAX_ACTIVE_SET_STEPS = 200
SCIPY_OPTIONS = {"ftol": 1e-15, "maxiter": 1000}
CALIBRATION_TOL = 1e-10   # relative error allowed on the target volatility
MAX_DOUBLINGS = 40        # search for an upper bound of gamma in the calibration
CORNER_TOL = 1e-9         # the portfolio no longer moves when gamma doubles: a corner


# =============================================================================
# 1. Covariance: Ledoit-Wolf shrinkage (LN3)
# =============================================================================

def ledoit_wolf(returns):
    """Ledoit and Wolf (2004) shrinkage to the constant-correlation model (LN3, slides 9-10):

        Sigma_hat = delta x F + (1 - delta) x S

    S is the sample covariance, F has the same variances and every correlation equal to
    the average one, delta is the optimal shrinkage constant of "Honey, I Shrunk the
    Sample Covariance Matrix" (Journal of Portfolio Management, 2004), between 0 and 1.

    returns: DataFrame (T months x N assets), no missing values.
    Returns (covariance DataFrame, delta)."""
    data = returns.to_numpy(dtype=float)
    t, n = data.shape
    x = data - data.mean(axis=0)
    sample = x.T @ x / t                                      # the paper divides by T
    if n < 2:
        return pd.DataFrame(sample, index=returns.columns, columns=returns.columns), 0.0
    var = np.diag(sample)
    sd = np.sqrt(var)
    r_bar = ((sample / np.outer(sd, sd)).sum() - n) / (n * (n - 1))     # average correlation
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # pi: sum of the asymptotic variances of the entries of S
    pi_matrix = (x ** 2).T @ (x ** 2) / t - sample ** 2
    pi_hat = pi_matrix.sum()
    # rho: sum of the asymptotic covariances between the entries of F and of S
    theta = (x ** 3).T @ x / t - var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    rho_hat = np.trace(pi_matrix) + r_bar * ((sd[None, :] / sd[:, None]) * theta).sum()
    # gamma of the paper: how far the target is from the sample
    misspecification = ((target - sample) ** 2).sum()

    delta = 0.0 if misspecification == 0 else max(0.0, min(1.0, (pi_hat - rho_hat) / misspecification / t))
    shrunk = delta * target + (1 - delta) * sample
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), float(delta)


def estimate_covariance(returns, params=None):
    """Monthly covariance matrix for the optimizer, with the method in params:
    "ledoit_wolf" (default) or "sample". Returns (DataFrame, shrinkage constant)."""
    if params is None:
        params = load_params()
    method = params["optimization"]["covariance"]
    if method == "ledoit_wolf":
        return ledoit_wolf(returns)
    if method == "sample":
        return returns.cov(), 0.0
    raise ValueError(f"unknown covariance method '{method}': use ledoit_wolf or sample")


# =============================================================================
# 2. Expected returns: Black-Litterman (LN4)
# =============================================================================

def implied_risk_aversion(cov, w_mkt, sharpe_ratio):
    """phi = SR(w_mkt) / sigma(w_mkt): the risk aversion that makes w_mkt optimal (LN4, slide 6).
    It is the MARKET's risk aversion, used only to build the prior."""
    return sharpe_ratio / np.sqrt(w_mkt @ cov @ w_mkt)


def implied_returns(cov, w_mkt, phi):
    """Implied excess returns pi = phi x Sigma x w_mkt (LN4, slide 7): the returns for which
    the market portfolio is the optimal portfolio."""
    return phi * cov @ w_mkt


def gamma_matrix(cov, tau):
    """Uncertainty of the prior, Gamma = tau x Sigma (lab LN4). Not the client's gamma."""
    return tau * cov


def bl_expected_returns(cov, w_mkt, phi, P, Q, omega_views, tau):
    """Black-Litterman expected excess returns (lab LN4):

        mu_bar = pi + Gamma P' (P Gamma P' + Omega)^-1 (Q - P pi)

    pi = phi Sigma w_mkt are the implied returns, Q - P pi the disequilibrium between the
    views and the market. With no views (P is None or empty) mu_bar = pi.
    Same units as cov and Q (the app uses annual numbers, as the lab)."""
    cov, w_mkt = np.asarray(cov, dtype=float), np.asarray(w_mkt, dtype=float)
    implied_mu = implied_returns(cov, w_mkt, phi)
    if P is None or len(P) == 0:
        return implied_mu
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    omega_views = np.atleast_2d(np.asarray(omega_views, dtype=float))
    G = gamma_matrix(cov, tau)
    disequilibrium = Q - P @ implied_mu
    return implied_mu + G @ P.T @ np.linalg.solve(P @ G @ P.T + omega_views, disequilibrium)


def market_weights(assets, params):
    """Weights of the global market portfolio on these assets (params black_litterman),
    0 for the assets that are not in it (their implied return then comes from their
    covariance with the market, as in the CAPM)."""
    weights = params["black_litterman"]["market_weights"]
    missing = [name for name in weights if name not in assets]
    if missing:
        raise ValueError(f"market weights on assets that are not in the universe: {missing}")
    w = np.array([float(weights.get(name, 0.0)) for name in assets])
    if abs(w.sum() - 1.0) > 1e-9:
        raise ValueError(f"market weights must sum to 1, they sum to {w.sum():.6f}")
    return w


def views_matrices(assets, params):
    """P, Q, Omega of the client's views (params black_litterman views; annual numbers).
    A view is {assets: {name: weight}, q: expected excess return, sd: its uncertainty}.
    Views on assets outside this client's universe are skipped.
    Returns (P, Q, omega_views, list of the views used); P is None if there is no view."""
    rows, q, sd, used = [], [], [], []
    for view in params["black_litterman"].get("views") or []:
        if not set(view["assets"]) <= set(assets):
            continue
        row = np.zeros(len(assets))
        for name, weight in view["assets"].items():
            row[assets.index(name)] = weight
        rows.append(row)
        q.append(view["q"])
        sd.append(view["sd"])
        used.append(view)
    if not rows:
        return None, None, None, []
    return np.array(rows), np.array(q, dtype=float), np.diag(np.square(sd)), used


def expected_returns(cov, params=None):
    """Monthly expected excess returns from Black-Litterman, for the assets of cov.

    The model is run in annual units, as in the lab, so tau and the view uncertainties keep
    their usual meaning; the result is divided by 12. phi is the same in both units.
    Returns (Series of monthly mu, dict with phi, implied returns, market weights, views)."""
    if params is None:
        params = load_params()
    bl = params["black_litterman"]
    assets = list(cov.columns)
    cov_annual = cov.to_numpy() * MONTHS
    w_mkt = market_weights(assets, params)
    phi = implied_risk_aversion(cov_annual, w_mkt, bl["sharpe_ratio"])
    P, Q, omega_views, used = views_matrices(assets, params)
    mu_annual = bl_expected_returns(cov_annual, w_mkt, phi, P, Q, omega_views, bl["tau"])
    info = {"phi": float(phi),
            "implied": pd.Series(implied_returns(cov_annual, w_mkt, phi) / MONTHS, index=assets),
            "market_weights": pd.Series(w_mkt, index=assets),
            "views": used}
    return pd.Series(mu_annual / MONTHS, index=assets), info


# =============================================================================
# 3. The total-wealth problem and the constraints (LN1)
# =============================================================================

def total_wealth_QP(x, cov, mu, sigma_H, omega, gamma):
    """Objective to minimise: 0.5 Var(R_W) - gamma E(R_W), without the terms that do not
    depend on x. With omega = 0 it is the QP of the labs: 0.5 x'Sigma x - gamma x'mu."""
    return (0.5 * (1 - omega) ** 2 * x @ cov @ x + (1 - omega) * omega * x @ sigma_H
            - gamma * (1 - omega) * x @ mu)


def unconstrained_portfolio(mu, cov, sigma_H, omega, gamma):
    """Solution without any constraint, split in its two demands.
    Returns (total, speculative, hedging), weights in the portfolio (1 = 100% of F)."""
    mu, cov, sigma_H = (np.asarray(a, dtype=float) for a in (mu, cov, sigma_H))
    speculative = gamma / (1 - omega) * np.linalg.solve(cov, mu)
    hedging = -omega / (1 - omega) * np.linalg.solve(cov, sigma_H)
    return speculative + hedging, speculative, hedging


def no_constraints(n):
    """A constraint set with no constraint at all (for tests and for the theory)."""
    return {"assets": None, "lower": np.full(n, -np.inf), "upper": np.full(n, np.inf),
            "caps": np.full(n, np.inf), "budget": None, "fully_invested": False,
            "long_only": False, "lockup": 0.0}


def build_constraints(assets, params=None, lockup=0.0):
    """The client's constraints, from params['optimization']:

        long_only      x >= 0
        max_leverage   1'x <= max_leverage (1 = no borrowing; the rest is cash)
        position_caps  x_i <= cap_i (one sector or one stock)
        lock-up        x_EMPLOYER >= lockup (deferred shares / portfolio). If the lock-up is
                       above the cap, the cap of the employer stock becomes the lock-up: the
                       deferred shares are held, but no share is bought.

    Returns a dict: assets, lower and upper bounds (arrays), caps, budget, flags."""
    if params is None:
        params = load_params()
    opt = params["optimization"]
    assets = list(assets)
    n = len(assets)
    lower = np.zeros(n) if opt["long_only"] else np.full(n, -np.inf)
    caps = np.full(n, np.inf)
    for name, cap in (opt.get("position_caps") or {}).items():
        if name in assets:
            caps[assets.index(name)] = float(cap)
    upper = caps.copy()
    if lockup > 0:
        if EMPLOYER not in assets:
            raise ValueError("a lock-up needs the EMPLOYER column in the universe")
        i = assets.index(EMPLOYER)
        lower[i] = max(lower[i], float(lockup))
        upper[i] = max(upper[i], float(lockup))
    budget = opt.get("max_leverage")
    return {"assets": assets, "lower": lower, "upper": upper, "caps": caps,
            "budget": None if budget is None else float(budget), "fully_invested": False,
            "long_only": bool(opt["long_only"]), "lockup": float(lockup)}


def _check_constraints(constraints, n):
    """Stop with a clear message if the constraints cannot all hold."""
    lower, upper, budget = constraints["lower"], constraints["upper"], constraints.get("budget")
    if len(lower) != n or len(upper) != n:
        raise ValueError("the constraints do not have one bound per asset")
    if np.any(lower > upper):
        raise ValueError("a lower bound is above an upper bound (for example a lock-up above a cap)")
    if budget is not None and np.all(np.isfinite(lower)) and lower.sum() > budget + FEASIBILITY_TOL:
        raise ValueError("the minimum holdings (lock-up) are above the investment limit")
    if constraints.get("fully_invested") and np.all(np.isfinite(upper)) and upper.sum() < budget:
        raise ValueError("the caps do not allow the portfolio to be fully invested")


def _constraint_rows(constraints, n):
    """Every constraint written as a'x <= b (the fully invested one as a'x = b).
    Returns A (m x n), b (m), equality flags (m) and a label (kind, asset index) per row."""
    rows, rhs, labels = [], [], []
    budget = constraints.get("budget")
    if budget is not None:
        rows.append(np.ones(n))
        rhs.append(budget)
        labels.append(("budget", None))
    for i in range(n):
        if np.isfinite(constraints["upper"][i]):
            rows.append(np.eye(n)[i])
            rhs.append(constraints["upper"][i])
            labels.append(("upper", i))
        if np.isfinite(constraints["lower"][i]):
            rows.append(-np.eye(n)[i])
            rhs.append(-constraints["lower"][i])
            labels.append(("lower", i))
    A = np.array(rows, dtype=float).reshape(len(rows), n)
    equality = np.array([kind == "budget" and constraints.get("fully_invested", False)
                         for kind, _ in labels], dtype=bool)
    return A, np.array(rhs, dtype=float), equality, labels


def _solve_equality_qp(H, c, A, b):
    """min 0.5 x'Hx + c'x subject to A x = b: one linear system (stationarity + constraints).
    Returns x and the multipliers of the rows of A."""
    n, m = len(c), len(b)
    if m == 0:
        return np.linalg.solve(H, -c), np.zeros(0)
    kkt = np.block([[H, A.T], [A, np.zeros((m, m))]])
    solution = np.linalg.lstsq(kkt, np.concatenate([-c, b]), rcond=None)[0]
    return solution[:n], solution[n:]


def _kkt_point(x_start, H, c, A, b, equality):
    """The point that satisfies every KKT condition (LN1), starting from scipy's solution.

    The constraints binding at x_start are held with equality and the system
    "stationarity + binding constraints" is solved exactly. If a constraint is then
    violated it is added; if a multiplier is negative (the constraint is not really
    binding) it is removed. At the end: feasible, multipliers >= 0, complementary slackness.
    Returns x, the multipliers (one per row) and the binding rows."""
    m = len(b)
    active = {j for j in range(m) if equality[j] or b[j] - A[j] @ x_start <= ACTIVE_TOL}
    for _ in range(MAX_ACTIVE_SET_STEPS):
        idx = sorted(active)
        x, lam_active = _solve_equality_qp(H, c, A[idx], b[idx])
        slack = b - A @ x
        violated = [j for j in range(m) if j not in active and slack[j] < -FEASIBILITY_TOL]
        if violated:
            active.add(min(violated, key=lambda j: slack[j]))
            continue
        lam = np.zeros(m)
        lam[idx] = lam_active
        negative = [j for j in idx if not equality[j] and lam[j] < -DUAL_TOL]
        if negative:
            active.remove(min(negative, key=lambda j: lam[j]))
            continue
        lam = np.where(equality, lam, np.where(lam > DUAL_TOL, lam, 0.0))
        return x, lam, idx
    raise RuntimeError("the optimizer did not find a point that satisfies the KKT conditions")


def solve_portfolio(mu, cov, sigma_H, omega, gamma, constraints=None):
    """Optimal portfolio and the Lagrange multipliers of its constraints.

    Step 1, as in the labs: scipy.optimize.minimize (SLSQP) with Bounds and LinearConstraint.
    Step 2, the KKT conditions (LN1): with the binding constraints found by scipy, the
    stationarity condition and the binding constraints are one linear system; its solution
    is the exact optimum and its multipliers are the shadow prices. The objective is divided
    by (1 - omega)^2 x average variance only so that the solver works with numbers of order 1.

    Returns a dict: x, the multipliers of the budget, lower and upper bounds (in units of
    total_wealth_QP, i.e. gamma x certainty equivalent), and scipy's own result."""
    mu, cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
    n = len(mu)
    sigma_H = np.zeros(n) if sigma_H is None else np.asarray(sigma_H, dtype=float)
    if not 0 <= omega < 1:
        raise ValueError(f"omega = {omega}: the portfolio must be a positive part of total wealth (0 <= omega < 1)")
    if constraints is None:
        constraints = no_constraints(n)
    _check_constraints(constraints, n)

    scale = (1 - omega) ** 2 * np.trace(cov) / n
    H = (1 - omega) ** 2 * cov / scale                                   # total_wealth_QP = 0.5 x'Hx + c'x
    c = ((1 - omega) * omega * sigma_H - gamma * (1 - omega) * mu) / scale

    # Step 1: scipy, as in the labs
    lower, upper = constraints["lower"], constraints["upper"]
    x0 = np.clip(np.zeros(n), lower, upper)
    bounds = Bounds(lower, upper) if np.isfinite(np.concatenate([lower, upper])).any() else None
    linear = []
    if constraints.get("budget") is not None:
        budget = constraints["budget"]
        linear.append(LinearConstraint(np.ones(n), lb=budget if constraints.get("fully_invested") else -np.inf,
                                       ub=budget))
    res = minimize(lambda x: total_wealth_QP(x, cov, mu, sigma_H, omega, gamma) / scale, x0,
                   jac=lambda x: H @ x + c, method="SLSQP", bounds=bounds, constraints=linear,
                   options=SCIPY_OPTIONS)

    # Step 2: KKT conditions
    A, b, equality, labels = _constraint_rows(constraints, n)
    x, lam, binding = _kkt_point(res.x, H, c, A, b, equality)
    for j in binding:                                                    # a binding bound holds exactly
        kind, i = labels[j]
        if kind in ("lower", "upper"):
            x[i] = constraints[kind][i]
    lam = lam * scale                                                    # back to total_wealth_QP units
    result = {"x": x, "budget": 0.0, "lower": np.zeros(n), "upper": np.zeros(n),
              "scipy_success": bool(res.success), "scipy_gap": float(np.max(np.abs(res.x - x)))}
    for (kind, i), value in zip(labels, lam):
        if kind == "budget":
            result["budget"] = float(value)
        else:
            result[kind][i] = value
    return result


def optimal_portfolio(mu, cov, sigma_H, omega, gamma, constraints=None):
    """Weights of the risky assets in the portfolio (ndarray; the rest,
    1 - sum, is cash). mu, cov: monthly excess returns; sigma_H = Cov(R, R_NT); omega = share
    of wealth that cannot be traded; gamma = risk tolerance; constraints from
    build_constraints (None = no constraint)."""
    return solve_portfolio(mu, cov, sigma_H, omega, gamma, constraints)["x"]


def strategy_portfolios(mu, cov, sigma_H, omega, gamma, constraints):
    """The strategies of the backtest, same gamma, returns and constraints:
        (A) standard MVO, the career is ignored     omega = 0
        (D) total wealth without the hedging term   sigma_H = 0
        (C) total wealth                            the full problem
    Returns {"A": result, "D": result, "C": result} (results of solve_portfolio)."""
    zero = np.zeros(len(mu))
    return {"A": solve_portfolio(mu, cov, zero, 0.0, gamma, constraints),
            "D": solve_portfolio(mu, cov, zero, omega, gamma, constraints),
            "C": solve_portfolio(mu, cov, sigma_H, omega, gamma, constraints)}


def certainty_equivalent(x, mu, cov, sigma_H, omega, gamma, background_mean=0.0, background_var=0.0):
    """Certainty-equivalent excess return of total wealth, E(R_W) - Var(R_W) / (2 gamma), monthly,
    as a fraction of W (x 12 = annual, x W = CHF). total_wealth_QP = -gamma x this + constant.
    background_mean, background_var: omega x E(R_NT) and omega^2 x Var(R_NT); they do not
    depend on x, so they matter only for the level, never for a difference between portfolios."""
    x = np.asarray(x, dtype=float)
    mean = (1 - omega) * x @ mu + background_mean
    var = (1 - omega) ** 2 * x @ cov @ x + 2 * (1 - omega) * omega * x @ sigma_H + background_var
    return float(mean - var / (2 * gamma))


# =============================================================================
# 4. gamma: one risk tolerance for everybody (LN1, target volatility with fsolve)
# =============================================================================

def calibrate_gamma(mu, cov, benchmark_weights, constraints):
    """gamma such that a client with no career and no pension (omega = 0) holds an optimal
    portfolio as volatile as the benchmark (a 60/40 portfolio), as the lab's target_vol.

    Without constraints vol = gamma x max Sharpe ratio, which gives the starting point.
    The volatility rises with gamma, so if fsolve does not converge a bisection does."""
    mu, cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
    benchmark_weights = np.asarray(benchmark_weights, dtype=float)
    target = float(np.sqrt(benchmark_weights @ cov @ benchmark_weights))
    zero = np.zeros(len(mu))

    def target_vol(gamma, target):
        x = optimal_portfolio(mu, cov, zero, 0.0, float(np.atleast_1d(gamma)[0]), constraints)
        return np.sqrt(x @ cov @ x) - target

    start = target / np.sqrt(mu @ np.linalg.solve(cov, mu))
    gamma, _, ier, _ = fsolve(target_vol, start, args=(target,), full_output=True)
    gamma = float(gamma[0])
    if ier == 1 and gamma > 0 and abs(target_vol(gamma, target)) <= CALIBRATION_TOL * target:
        return gamma

    # Bisection: find a gamma that is too risky, then solve between 0 and it
    if target_vol(0.0, target) > 0:
        raise ValueError("even with gamma = 0 the constraints force a portfolio riskier than the benchmark")
    high, previous = start, None
    for _ in range(MAX_DOUBLINGS):
        x = optimal_portfolio(mu, cov, zero, 0.0, high, constraints)
        if np.sqrt(x @ cov @ x) >= target:
            return float(brentq(target_vol, 0.0, high, args=(target,), xtol=1e-14))
        if previous is not None and np.allclose(x, previous, rtol=0, atol=CORNER_TOL):
            break                                       # a corner: more gamma does not change the portfolio
        previous, high = x, 2 * high
    raise ValueError("no portfolio allowed by the constraints is as volatile as the benchmark")


def risk_tolerance(params, excess_returns):
    """gamma from params: a number, or "auto" = calibrated on the standard universe (the core
    assets without the employer stock) with the standard constraints (no lock-up). So every
    client and every strategy shares the same gamma.
    Returns (gamma, dict with the target volatility and the calibrated portfolio)."""
    opt = params["optimization"]
    core = [name for name in params["data"]["assets"] if name != EMPLOYER]
    cov, _ = estimate_covariance(excess_returns[core], params)
    mu, _ = expected_returns(cov, params)
    benchmark = opt["calibration_portfolio"]
    unknown = [name for name in benchmark if name not in core]
    if unknown:
        raise ValueError(f"calibration_portfolio uses assets outside the core universe: {unknown}")
    w = np.array([float(benchmark.get(name, 0.0)) for name in core])
    constraints = build_constraints(core, params)
    if opt["gamma"] == "auto":
        gamma = calibrate_gamma(mu.to_numpy(), cov.to_numpy(), w, constraints)
    else:
        gamma = float(opt["gamma"])
    x = optimal_portfolio(mu.to_numpy(), cov.to_numpy(), np.zeros(len(core)), 0.0, gamma, constraints)
    info = {"target_vol": float(np.sqrt(w @ cov.to_numpy() @ w * MONTHS)),
            "achieved_vol": float(np.sqrt(x @ cov.to_numpy() @ x * MONTHS)),
            "portfolio": pd.Series(x, index=core)}
    return gamma, info


# =============================================================================
# 5. Shadow prices: what each constraint costs (LN1, KKT)
# =============================================================================

def _without(constraints, group):
    """Copy of the constraints without one group: budget, long_only, lockup or caps."""
    c = copy.deepcopy(constraints)
    n = len(c["lower"])
    employer = c["assets"].index(EMPLOYER) if c["lockup"] > 0 else None
    if group == "budget":
        c["budget"] = None
    elif group == "long_only":
        c["lower"] = np.full(n, -np.inf)
        if employer is not None:
            c["lower"][employer] = c["lockup"]
        c["long_only"] = False
    elif group == "lockup":
        # The client may now sell the deferred shares. The upper bound stays as it is: he can
        # keep them even above the cap (the cap limits buying), so this is a true relaxation.
        c["lower"][employer] = 0.0 if c["long_only"] else -np.inf
        c["lockup"] = 0.0
    elif group == "caps":
        c["upper"] = np.full(n, np.inf)
        c["caps"] = np.full(n, np.inf)
    return c


def shadow_prices(mu, cov, sigma_H, omega, gamma, constraints, wealth, portfolio):
    """What the constraints cost the client, two ways:

    marginal: the KKT multiplier lambda of each binding constraint. total_wealth_QP is
        -gamma x CE, so 1 more unit of room raises the certainty equivalent by lambda / gamma
        (envelope theorem). Quoted for 1 percentage point of the portfolio, per year.
    total: the problem is solved again without the whole group of constraints; the cost is
        the gain in certainty equivalent per year.

    Both in basis points of total wealth W, in CHF per year and in basis points of the
    portfolio F (the CHF amount divided by F). Returns (marginal, total, weights_without):
    two DataFrames and the optimal weights without each group."""
    names = constraints["assets"]
    base = solve_portfolio(mu, cov, sigma_H, omega, gamma, constraints)
    ce_base = certainty_equivalent(base["x"], mu, cov, sigma_H, omega, gamma)
    employer = names.index(EMPLOYER) if constraints["lockup"] > 0 else None

    def amounts(ce_gain_monthly_fraction):
        annual = ce_gain_monthly_fraction * MONTHS
        return {"bp of W per year": annual * BP, "CHF per year": annual * wealth,
                "bp of F per year": annual * wealth / portfolio * BP}

    rows = []
    if base["budget"] > 0:
        rows.append({"group": "No leverage", "constraint": f"invest at most {constraints['budget']:.0%}",
                     "multiplier": base["budget"]})
    for i, name in enumerate(names):
        if base["lower"][i] > 0:
            if i == employer:
                rows.append({"group": "Lock-up", "constraint": f"{name} at least {constraints['lockup']:.1%}",
                             "multiplier": base["lower"][i]})
            else:
                rows.append({"group": "Long-only", "constraint": f"{name} at least 0%",
                             "multiplier": base["lower"][i]})
        if base["upper"][i] > 0:
            rows.append({"group": "Position caps", "constraint": f"{name} at most {constraints['upper'][i]:.0%}",
                         "multiplier": base["upper"][i]})
    for row in rows:
        row.update(amounts(row["multiplier"] / gamma * PERCENT_POINT))
    marginal = pd.DataFrame(rows, columns=["group", "constraint", "multiplier", "bp of W per year",
                                           "CHF per year", "bp of F per year"])

    groups = []
    if constraints.get("budget") is not None:
        groups.append(("budget", "No leverage", base["budget"] > 0))
    if constraints.get("long_only"):
        long_only_binding = any(base["lower"][i] > 0 for i in range(len(names)) if i != employer)
        groups.append(("long_only", "Long-only", long_only_binding))
    if constraints["lockup"] > 0:
        groups.append(("lockup", "Lock-up", base["lower"][employer] > 0))
    if np.isfinite(constraints["upper"]).any():
        groups.append(("caps", "Position caps", bool((base["upper"] > 0).any())))
    rows, weights_without = [], {}
    for key, label, binding in groups:
        relaxed = solve_portfolio(mu, cov, sigma_H, omega, gamma, _without(constraints, key))
        gain = certainty_equivalent(relaxed["x"], mu, cov, sigma_H, omega, gamma) - ce_base
        weights_without[label] = pd.Series(relaxed["x"], index=names)
        rows.append({"group": label, "binding": binding, **amounts(gain),
                     "invested without it": float(relaxed["x"].sum())})
    total = pd.DataFrame(rows, columns=["group", "binding", "bp of W per year", "CHF per year",
                                        "bp of F per year", "invested without it"])
    return marginal, total, weights_without


# =============================================================================
# 6. Risk contributions (LN5), on the portfolio and on total wealth
# =============================================================================

def risk_contributions(x, cov):
    """Absolute risk contributions RC_i = x_i (Sigma x)_i / sqrt(x'Sigma x)
    (lab LN5). They add up to the volatility; divided by it they are the relative ones."""
    x, cov = np.asarray(x, dtype=float), np.asarray(cov, dtype=float)
    vol = np.sqrt(x @ cov @ x)
    if vol == 0:
        return np.zeros_like(x)
    return x * (cov @ x) / vol


def total_wealth_risk_contributions(x, cov, sigma_career, career_var, omega_H, omega_P, assets):
    """ "How much of your risk is your job?".

    Total wealth is a portfolio of the assets (weights (1 - omega) x), the career (weight
    omega_H) and the pension (weight omega_P, no market risk). The LN5 formula is applied to
    this whole balance sheet, with the covariance matrix extended by the career:
    Cov(R, R_career) = sigma_career and Var(R_career) = career_var (monthly).
    Returns a DataFrame: weight in W, risk contribution (annual volatility units), share of risk."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    omega = omega_H + omega_P
    y = np.concatenate([(1 - omega) * x, [omega_H, omega_P]])
    extended = np.zeros((n + 2, n + 2))
    extended[:n, :n] = cov
    extended[:n, n] = extended[n, :n] = sigma_career
    extended[n, n] = career_var
    rc = risk_contributions(y, extended) * np.sqrt(MONTHS)
    total = rc.sum()
    return pd.DataFrame({"weight in W": y, "risk contribution": rc,
                         "share of risk": rc / total if total > 0 else np.zeros_like(rc)},
                        index=list(assets) + [CAREER, PENSION])


# =============================================================================
# 7. The client: balance sheet, universe, data
# =============================================================================

def pension_savings(age, salary, params):
    """2nd-pillar savings P (CHF) with the legal BVG minimum (params['pension']).

    Each year from start_age until today (at most until retirement) the account receives an
    age credit = credit rate of the age band x coordinated salary, and earns the minimum
    interest rate. Coordinated salary = min(salary, max_insured_salary) - coordination_deduction
    (at least 0). Past salaries are today's salary scaled back by salary_growth."""
    pension = params["pension"]
    last = min(age, params["client"]["retirement_age"])
    ages = np.arange(pension["start_age"], last)
    if len(ages) == 0:
        return 0.0
    salaries = salary * (1 + params["career"]["salary_growth"]) ** (ages - age)
    coordinated = np.maximum(np.minimum(salaries, pension["max_insured_salary"])
                             - pension["coordination_deduction"], 0.0)
    bands = sorted(pension["age_credits"].items())
    starts = np.array([start for start, _ in bands])
    rates = np.array([rate for _, rate in bands])
    band = np.searchsorted(starts, ages, side="right") - 1
    credits = np.where(band >= 0, rates[np.maximum(band, 0)], 0.0) * coordinated
    return float((credits * (1 + pension["interest_rate"]) ** (last - 1 - ages)).sum())


def balance_sheet(params=None):
    """The client's balance sheet today: H (career), P (pension), F (portfolio), W, the
    non-tradable shares omega_H, omega_P, omega, the lock-up share and the career betas."""
    if params is None:
        params = load_params()
    client = params["client"]
    age, salary, F = client["age"], client["salary_chf"], float(client["portfolio_chf"])
    if F <= 0:
        raise ValueError("the portfolio must be positive: it is the only part the optimizer chooses")
    H = human_capital(age, salary, params)
    if not client.get("pillar2", True):
        P = 0.0
    elif client.get("pension_chf") in (None, "auto"):
        P = pension_savings(age, salary, params)
    else:
        P = float(client["pension_chf"])
    deferred = float(client.get("deferred_shares_chf") or 0.0)
    if deferred > F:
        raise ValueError("the deferred shares are part of the portfolio: they cannot be larger than it")
    W = H + P + F
    beta_bond, beta_sector, rate = career_betas(age, params)
    return {"H": H, "P": P, "F": F, "W": W, "omega_H": H / W, "omega_P": P / W, "omega": (H + P) / W,
            "deferred_shares": deferred, "lockup": deferred / F,
            "beta_bond": beta_bond, "beta_sector": beta_sector, "discount_rate": rate}


def client_assets(params=None):
    """The client's universe: the core assets, without the employer
    stock if the client holds no deferred shares, plus the client's sector if it comes from
    the sector menu."""
    if params is None:
        params = load_params()
    has_shares = (params["client"].get("deferred_shares_chf") or 0) > 0
    assets = [name for name in params["data"]["assets"] if name != EMPLOYER or has_shares]
    sector = params["data"]["career_sector_asset"]
    if sector is not None and sector not in assets:
        assets.append(sector)
    return assets


def load_excess_returns(start=None, end=None):
    """Monthly excess returns over RF_CHF of every saved series: core assets and sector menu."""
    from src.data import load_returns, load_rf, load_sector_returns
    returns = load_returns(start, end).join(load_sector_returns(start, end))
    return returns.sub(load_rf(start, end), axis=0)


# =============================================================================
# 8. Everything for one client (app pages 3, 4 and 5)
# =============================================================================

def _clean(table):
    """Rounding noise (|value| < 1e-12, for example -1e-17) shown as an exact 0 in the tables."""
    return table.mask(table.abs() < 1e-12, 0.0)


def analyze_client(params=None, excess_returns=None, start=None, end=None):
    """Optimal portfolios, demands, constraint costs and risk sources for the client in params.

    excess_returns: DataFrame with the core assets and (if needed) the client's sector; if None
    the saved data are used (load_excess_returns). Returns a dict of tables (see print_report)."""
    if params is None:
        params = load_params()
    if excess_returns is None:
        excess_returns = load_excess_returns(start, end)
    assets = client_assets(params)
    excess = excess_returns[assets]
    if excess.isna().any().any():
        raise ValueError("the excess returns have missing values")

    sheet = balance_sheet(params)
    cov_df, shrinkage = estimate_covariance(excess, params)
    cov = cov_df.to_numpy()
    mu_series, bl_info = expected_returns(cov_df, params)
    mu = mu_series.to_numpy()
    gamma, gamma_info = risk_tolerance(params, excess_returns)

    # The career: Cov(R, R_career), Var(R_career), and the non-tradable inputs of the problem
    beta_bond, beta_sector = sheet["beta_bond"], sheet["beta_sector"]
    sigma_career = career_covariance(cov_df, beta_bond, beta_sector, params)
    exposure = np.zeros(len(assets))
    exposure[assets.index(params["data"]["career_bond_asset"])] = beta_bond
    sector = params["data"]["career_sector_asset"]
    if sector is not None:
        exposure[assets.index(sector)] += beta_sector
    career_var = exposure @ cov @ exposure + params["career"]["idiosyncratic_vol"] ** 2 / MONTHS
    omega, omega_H = sheet["omega"], sheet["omega_H"]
    sigma_H = omega_H / omega * sigma_career if omega > 0 else np.zeros(len(assets))
    background_mean = omega_H * exposure @ mu
    background_var = omega_H ** 2 * career_var

    constraints = build_constraints(assets, params, sheet["lockup"])
    solutions = strategy_portfolios(mu, cov, sigma_H, omega, gamma, constraints)

    weights = pd.DataFrame({key: result["x"] for key, result in solutions.items()}, index=assets)
    weights.loc[CASH] = 1 - weights.sum()
    weights = _clean(weights)

    total_demand, speculative, hedging = unconstrained_portfolio(mu, cov, sigma_H, omega, gamma)
    demand = _clean(pd.DataFrame({"speculative": speculative, "hedging": hedging, "total": total_demand,
                                  "C (constrained)": solutions["C"]["x"]}, index=assets))

    career_holdings = pd.Series(beta_bond * sheet["H"], index=[params["data"]["career_bond_asset"]])
    if sector is not None and beta_sector != 0:
        career_holdings[sector] = beta_sector * sheet["H"]

    marginal, total, weights_without = shadow_prices(mu, cov, sigma_H, omega, gamma, constraints,
                                                     sheet["W"], sheet["F"])

    rows = {}
    for key, result in solutions.items():
        x = result["x"]
        mean = (1 - omega) * x @ mu + background_mean
        var = (1 - omega) ** 2 * x @ cov @ x + 2 * (1 - omega) * omega * x @ sigma_H + background_var
        ce = certainty_equivalent(x, mu, cov, sigma_H, omega, gamma, background_mean, background_var)
        rows[key] = {"portfolio expected excess return": x @ mu * MONTHS,
                     "portfolio volatility": np.sqrt(x @ cov @ x * MONTHS),
                     "total wealth expected excess return": mean * MONTHS,
                     "total wealth volatility": np.sqrt(var * MONTHS),
                     "certainty equivalent (annual)": ce * MONTHS,
                     "certainty equivalent, CHF per year": ce * MONTHS * sheet["W"]}
    performance = pd.DataFrame(rows)
    performance.loc["gain over (A), CHF per year"] = (performance.loc["certainty equivalent, CHF per year"]
                                                      - performance.loc["certainty equivalent, CHF per year", "A"])

    risk = {key: _clean(total_wealth_risk_contributions(solutions[key]["x"], cov, sigma_career, career_var,
                                                        omega_H, sheet["omega_P"], assets))
            for key in solutions}

    returns_table = pd.DataFrame({"market weight": bl_info["market_weights"],
                                  "implied (prior)": bl_info["implied"] * MONTHS,
                                  "Black-Litterman": mu_series * MONTHS,
                                  "historical mean": excess.mean() * MONTHS,
                                  "volatility": np.sqrt(np.diag(cov) * MONTHS)})

    return {"params": params, "assets": assets, "months": len(excess),
            "sample": (excess.index[0], excess.index[-1]),
            "balance_sheet": sheet, "gamma": gamma, "gamma_info": gamma_info,
            "shrinkage": shrinkage, "phi": bl_info["phi"], "views": bl_info["views"],
            "covariance": cov_df, "expected_returns": returns_table,
            "omega": omega, "sigma_H": sigma_H, "constraints": constraints,
            "solutions": solutions, "weights": weights, "demand": demand,
            "career_holdings": career_holdings, "career_var": career_var,
            "shadow_marginal": marginal, "shadow_total": total, "weights_without": weights_without,
            "performance": performance, "risk": risk}


def personas_summary(params=None, excess_returns=None):
    """One line per persona: balance sheet and the two effects of version 1:
    (A) -> (D) "your job is bond-like, so you can take more risk" (fewer bonds),
    (D) -> (C) "avoid your own sector" (the hedge), with the gains in CHF per year."""
    if params is None:
        params = load_params()
    if excess_returns is None:
        excess_returns = load_excess_returns()
    rows = []
    for name, persona in params["personas"].items():
        result = analyze_client(params_for_persona(params, name), excess_returns)
        sheet, w = result["balance_sheet"], result["weights"]
        ce = result["performance"].loc["certainty equivalent, CHF per year"]
        bonds = [asset for asset in result["assets"] if asset.startswith("BOND")]
        sector = persona["sector"]
        rows.append({"persona": persona["label"], "H": sheet["H"], "P": sheet["P"], "omega": sheet["omega"],
                     "bonds (A)": w.loc[bonds, "A"].sum(), "bonds (D)": w.loc[bonds, "D"].sum(),
                     "own sector (D)": w.loc[sector, "D"] if sector else np.nan,
                     "own sector (C)": w.loc[sector, "C"] if sector else np.nan,
                     "(D) vs (A), CHF/yr": ce["D"] - ce["A"], "(C) vs (D), CHF/yr": ce["C"] - ce["D"]})
    table = _clean(pd.DataFrame(rows).set_index("persona"))
    gains = ["(D) vs (A), CHF/yr", "(C) vs (D), CHF/yr"]
    table[gains] = table[gains].round(2) + 0.0                       # to the cent, no "-0"
    return table


def print_report(result):
    """Readable summary of analyze_client, for the terminal."""
    p, sheet = result["params"], result["balance_sheet"]
    client = p["client"]
    pct = "{:.1%}".format
    chf = "{:,.0f}".format
    print(f"Client: {client.get('persona', 'custom')}, age {client['age']}, salary CHF {chf(client['salary_chf'])}")
    print(f"Data: {result['months']} months, {result['sample'][0]:%Y-%m} to {result['sample'][1]:%Y-%m}")
    print(f"\nBalance sheet: H = CHF {chf(sheet['H'])}, P = CHF {chf(sheet['P'])}, F = CHF {chf(sheet['F'])}, "
          f"W = CHF {chf(sheet['W'])}")
    print(f"  non-tradable share omega = {pct(sheet['omega'])} (career {pct(sheet['omega_H'])}, "
          f"pension {pct(sheet['omega_P'])}); lock-up = {pct(sheet['lockup'])} of the portfolio")
    print(f"  career betas: bond {sheet['beta_bond']:.2f}, sector {sheet['beta_sector']:.2f}. "
          f"The career already holds, in CHF:")
    print("   " + ", ".join(f"{name} {chf(value)}" for name, value in result["career_holdings"].items()))

    info = result["gamma_info"]
    print(f"\ngamma = {result['gamma']:.4f} (risk tolerance): a client with no career holds a portfolio "
          f"with the volatility of 60/40 ({pct(info['target_vol'])})")
    print(f"Covariance: {p['optimization']['covariance']}, shrinkage {result['shrinkage']:.2f}. "
          f"Black-Litterman: phi = {result['phi']:.2f}, {len(result['views'])} view(s)")
    print("\nExpected excess returns (annual):")
    print(result["expected_returns"].to_string(float_format=pct))

    print("\nPortfolios (share of the portfolio F):")
    print(result["weights"].rename(columns=STRATEGIES).to_string(float_format=pct))
    print("\nUnconstrained demands (share of F; 1 = 100%):")
    print(result["demand"].to_string(float_format="{:.2f}".format))

    print("\nWhat the constraints cost, marginal (1 percentage point more room):")
    if result["shadow_marginal"].empty:
        print("  no binding constraint")
    else:
        print(result["shadow_marginal"].drop(columns="multiplier").to_string(
            index=False, float_format="{:,.1f}".format))
    print("\nWhat the constraints cost, total (solved again without the group):")
    print(result["shadow_total"].to_string(index=False, float_format="{:,.1f}".format))

    print("\nWhere the risk of total wealth comes from, (C):")
    print(result["risk"]["C"].to_string(float_format=pct))
    print("\nPerformance (annual, estimated):")
    print(result["performance"].rename(columns=STRATEGIES).to_string(float_format="{:,.4f}".format))


if __name__ == "__main__":
    params = load_params()
    excess_returns = load_excess_returns()
    print_report(analyze_client(params, excess_returns))
    print("\nAll personas: (A) -> (D) take more risk, (D) -> (C) avoid your own sector:")
    print(personas_summary(params, excess_returns).to_string(
        formatters={"H": "{:,.0f}".format, "P": "{:,.0f}".format, "omega": "{:.1%}".format,
                    "bonds (A)": "{:.1%}".format, "bonds (D)": "{:.1%}".format,
                    "own sector (D)": "{:.1%}".format, "own sector (C)": "{:.1%}".format,
                    "(D) vs (A), CHF/yr": "{:,.0f}".format, "(C) vs (D), CHF/yr": "{:,.0f}".format}))
