"""Tests of src/optimizer.py with known answers.

Reference values come from the labs (LN4, LN5), from closed-form solutions and from
independent checks (an exact solution by enumeration, finite differences, the formulas
of Ledoit and Wolf written with loops).

Run from the project folder:
    python tests/test_optimizer.py
"""
import copy
import itertools
import os
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, fsolve, minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.optimizer as O
from src.career import params_for_persona
from src.config import load_params


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def lab_example_1():
    """Example 1 of LN1 (used again in Example 4 of LN4): covariance matrix."""
    vol = np.array([0.15, 0.20, 0.25, 0.30])
    correl = np.array([[1.0, 0.1, 0.4, 0.5], [0.1, 1.0, 0.7, 0.4], [0.4, 0.7, 1.0, 0.8], [0.5, 0.4, 0.8, 1.0]])
    return np.outer(vol, vol) * correl


def random_problem(n=5, seed=0):
    """A random positive definite covariance (monthly), positive expected returns and a
    career with beta_bond on asset 1 and beta_sector on asset 2."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(240, n)) * rng.uniform(0.01, 0.07, size=n)
    cov = np.cov(a.T)
    mu = rng.uniform(0.0005, 0.006, size=n)
    exposure = np.zeros(n)
    exposure[1], exposure[2] = 1.4, 0.3
    return mu, cov, cov @ exposure, exposure


def constraints(n, lower=0.0, upper=np.inf, budget=1.0, fully_invested=False, lockup=0.0, employer=None):
    """A constraint dict written by hand (independent of build_constraints)."""
    names = [f"A{i}" for i in range(n)]
    low = np.full(n, float(lower))
    up = np.full(n, float(upper))
    if lockup > 0:
        names[employer] = O.EMPLOYER
        low[employer] = lockup
        up[employer] = max(up[employer], lockup)
    return {"assets": names, "lower": low, "upper": up, "caps": np.full(n, float(upper)), "budget": budget,
            "fully_invested": fully_invested, "long_only": lower == 0.0, "lockup": lockup}


def exact_solution(mu, cov, sigma_H, omega, gamma, cons):
    """Exact optimum by enumeration: for every set of constraints held with equality, solve
    the linear system; the best feasible point is the optimum (strictly convex problem)."""
    n = len(mu)
    H = (1 - omega) ** 2 * cov
    c = (1 - omega) * omega * sigma_H - gamma * (1 - omega) * mu
    rows, rhs = [np.ones(n)], [cons["budget"]]
    for i in range(n):
        for sign, bound in ((1.0, cons["upper"][i]), (-1.0, cons["lower"][i])):
            if np.isfinite(bound):
                rows.append(sign * np.eye(n)[i])
                rhs.append(sign * bound)
    A, b = np.array(rows), np.array(rhs)
    best, best_value = None, np.inf
    for k in range(n + 1):
        for subset in itertools.combinations(range(len(b)), k):
            idx = list(subset)
            kkt = np.block([[H, A[idx].T], [A[idx], np.zeros((k, k))]])
            try:
                x = np.linalg.solve(kkt, np.concatenate([-c, b[idx]]))[:n]
            except np.linalg.LinAlgError:
                continue
            if np.all(A @ x <= b + 1e-12):
                value = 0.5 * x @ H @ x + c @ x
                if value < best_value:
                    best, best_value = x, value
    return best


# -----------------------------------------------------------------------------
# 1. Lab reference values
# -----------------------------------------------------------------------------

def test_black_litterman_lab_LN4():
    cov = lab_example_1()
    x0, sr, rf = np.array([0.4, 0.3, 0.2, 0.1]), 0.25, 0.03
    vol_x0 = np.sqrt(x0 @ cov @ x0)
    assert abs(vol_x0 - 0.15349267083479914) < 1e-15
    phi = O.implied_risk_aversion(cov, x0, sr)
    assert abs(phi - 1.6287422626782593) < 1e-12
    implied_mu = rf + O.implied_returns(cov, x0, phi)
    assert np.allclose(np.round(implied_mu, 4), [0.0547, 0.0668, 0.0870, 0.0906])     # LN4, slide 8

    # Views of slide 19: q1 = 4%, q2-3 = -1%, uncertainties 10% and 5%. The lab works with total
    # returns, our function with excess returns: the absolute view becomes q1 - rf.
    P = np.array([[1.0, 0, 0, 0], [0, 1.0, -1.0, 0]])
    Q = np.array([0.04, -0.01])
    omega_views = np.diag([0.10 ** 2, 0.05 ** 2])
    q_excess = Q - rf * P.sum(axis=1)

    def mu_bar(tau):
        return rf + O.bl_expected_returns(cov, x0, phi, P, q_excess, omega_views, tau)

    def QP(x, sigma, mu, gamma):
        return 0.5 * x.T @ sigma @ x - gamma * x.T @ mu

    def lab_weights(tau):                       # the lab's call: scipy defaults, starting from x0
        lab = [LinearConstraint(np.ones(4), lb=1, ub=1)]
        return minimize(QP, x0, args=(cov, mu_bar(tau), 1 / phi), constraints=lab, bounds=Bounds(0, 1)).x

    def te(weights):
        return np.sqrt((weights - x0) @ cov @ (weights - x0))

    # 1. Our Black-Litterman returns + the lab's own call: close to the lab's expected output.
    #    Not exact: with scipy's default tolerance the result depends on the scipy version.
    te_lab = te(lab_weights(0.05))
    assert abs(te_lab - 0.013212510637878089) < 3e-3, te_lab

    def target_TE(tau, target, weights_function):
        return te(weights_function(float(np.atleast_1d(tau)[0]))) - target

    # 2. But scipy's default tolerance (ftol = 1e-6, while the objective is about 1e-2) stops after
    #    4 iterations: the lab's weights are not the exact optimum. Our solver checks the KKT
    #    conditions: only the budget binds, so Sigma x - gamma mu must be equal for every asset.
    fully_invested = constraints(4, lower=0.0, budget=1.0, fully_invested=True)

    def our_weights(tau):
        return O.optimal_portfolio(mu_bar(tau), cov, np.zeros(4), 0.0, 1 / phi, fully_invested)

    w = our_weights(0.05)
    stationarity = cov @ w - mu_bar(0.05) / phi
    assert np.allclose(stationarity, stationarity[0], atol=1e-15) and abs(w.sum() - 1) < 1e-15
    lab = lab_weights(0.05)
    assert QP(w, cov, mu_bar(0.05), 1 / phi) < QP(lab, cov, mu_bar(0.05), 1 / phi)  # ours is better
    tight = minimize(QP, x0, args=(cov, mu_bar(0.05), 1 / phi), bounds=Bounds(0, 1), options={"ftol": 1e-15},
                     constraints=[LinearConstraint(np.ones(4), lb=1, ub=1)]).x
    assert np.allclose(w, tight, atol=1e-6)                                         # scipy agrees once precise
    tau = fsolve(target_TE, 0.05, args=(0.02, our_weights))[0]
    assert abs(target_TE(tau, 0.02, our_weights)) < 1e-12
    print(f"Black-Litterman = lab LN4 OK (lab call: TE {te_lab:.6f}; exact optimum: TE {te(w):.6f} at tau 0.05, "
          f"tau {tau:.6f} for TE 2%)")


def test_risk_contributions_lab_LN5():
    vol = np.array([0.3, 0.25, 0.2, 0.15])
    correl = np.full((4, 4), 0.7) + 0.3 * np.eye(4)
    cov = np.outer(vol, vol) * correl
    rc = O.risk_contributions(np.full(4, 0.25), cov)
    assert abs(rc.sum() - 0.19866743568083822) < 1e-15                  # vol_ew of the lab
    # With a constant correlation the ERC portfolio is proportional to 1 / volatility
    x = (1 / vol) / (1 / vol).sum()
    assert np.allclose(O.risk_contributions(x, cov), O.risk_contributions(x, cov)[0])
    print("risk contributions = lab LN5 OK")


def test_lab_QP_when_omega_is_zero():
    # With omega = 0 the total-wealth QP is the lab QP (Example 1, fully invested, short selling allowed)
    cov, mu, gam = lab_example_1(), np.array([0.05, 0.06, 0.08, 0.06]), 0.3

    def QP(x, sigma, mu, gamma):
        return 0.5 * x.T @ sigma @ x - gamma * x.T @ mu

    ours = O.optimal_portfolio(mu, cov, np.zeros(4), 0.0, gam,
                               constraints(4, lower=-np.inf, budget=1.0, fully_invested=True))
    assert abs(O.total_wealth_QP(ours, cov, mu, np.zeros(4), 0.0, gam) - QP(ours, cov, mu, gam)) < 1e-15
    # Closed form of LN3, slide 15
    inv, one = np.linalg.inv(cov), np.ones(4)
    formula = (inv @ one / (one @ inv @ one)
               + gam * ((one @ inv @ one) * inv @ mu - (one @ inv @ mu) * inv @ one) / (one @ inv @ one))
    assert np.allclose(ours, formula, atol=1e-12)
    # The lab's scipy call (LN1 constraints) agrees once its tolerance is tight enough
    lab = [LinearConstraint(np.ones(4), ub=1), LinearConstraint(-np.ones(4), ub=-1)]
    res = minimize(QP, np.full(4, 0.25), args=(cov, mu, gam), constraints=lab, options={"ftol": 1e-15})
    assert np.allclose(ours, res.x, atol=1e-6)
    print("omega = 0 gives the lab QP OK")


# -----------------------------------------------------------------------------
# 2. The total-wealth problem
# -----------------------------------------------------------------------------

def test_unconstrained_equals_formula():
    # Known answer: without constraints optimal_portfolio = the closed-form formula
    mu, cov, sigma_H, exposure = random_problem()
    gamma = 0.3
    for omega in (0.0, 0.3, 0.9, 0.97):
        x = O.optimal_portfolio(mu, cov, sigma_H, omega, gamma, None)
        formula = (gamma / (1 - omega) * np.linalg.solve(cov, mu)
                   - omega / (1 - omega) * np.linalg.solve(cov, sigma_H))
        assert np.allclose(x, formula, rtol=1e-10, atol=1e-10), omega
        # Known answer: the hedging demand is -omega/(1-omega) x (beta_bond e_B + beta_sector e_S)
        _, _, hedging = O.unconstrained_portfolio(mu, cov, sigma_H, omega, gamma)
        assert np.allclose(hedging, -omega / (1 - omega) * exposure, atol=1e-10)
    print("unconstrained = formula OK")


def test_constrained_against_exact_solution():
    for seed in range(6):
        mu, cov, sigma_H, _ = random_problem(n=4, seed=seed)
        for omega, gamma in ((0.0, 0.2), (0.5, 0.3), (0.95, 0.27)):
            cases = [constraints(4, lower=0.0, budget=1.0),
                     constraints(4, lower=0.0, upper=0.4, budget=1.0),
                     constraints(4, lower=0.0, upper=0.3, budget=1.0, lockup=0.15, employer=3),
                     constraints(4, lower=-np.inf, budget=1.0)]
            for cons in cases:
                result = O.solve_portfolio(mu, cov, sigma_H, omega, gamma, cons)
                x = result["x"]
                assert np.allclose(x, exact_solution(mu, cov, sigma_H, omega, gamma, cons), atol=1e-10)
                # KKT: stationarity with the multipliers, all of them >= 0, feasibility
                grad = (1 - omega) ** 2 * cov @ x + (1 - omega) * omega * sigma_H - gamma * (1 - omega) * mu
                residual = grad + result["budget"] + result["upper"] - result["lower"]
                assert np.allclose(residual, 0, atol=1e-12)
                assert result["budget"] >= 0 and np.all(result["lower"] >= 0) and np.all(result["upper"] >= 0)
                assert x.sum() <= 1 + 1e-12
                assert np.all(x >= cons["lower"] - 1e-12) and np.all(x <= cons["upper"] + 1e-12)
    print("constrained solution = exact solution, KKT conditions OK")


def test_shadow_prices():
    mu, cov, sigma_H, _ = random_problem(n=4, seed=3)
    omega, gamma, W, F = 0.9, 0.27, 1_000_000.0, 100_000.0
    cons = constraints(4, lower=0.0, budget=1.0, lockup=0.2, employer=3)
    base = O.solve_portfolio(mu, cov, sigma_H, omega, gamma, cons)
    ce0 = O.certainty_equivalent(base["x"], mu, cov, sigma_H, omega, gamma)
    assert base["budget"] > 0 and base["lower"][3] > 0, "the test needs a binding budget and lock-up"

    # Marginal value = multiplier / gamma (envelope theorem), checked with a small relaxation
    eps = 1e-6
    for kind, change in (("budget", {"budget": 1.0 + eps}), ("lockup", None)):
        relaxed = copy.deepcopy(cons)
        if kind == "budget":
            relaxed.update(change)
            multiplier = base["budget"]
        else:
            relaxed["lower"][3] -= eps
            multiplier = base["lower"][3]
        x = O.optimal_portfolio(mu, cov, sigma_H, omega, gamma, relaxed)
        gain = (O.certainty_equivalent(x, mu, cov, sigma_H, omega, gamma) - ce0) / eps
        assert abs(gain - multiplier / gamma) < 1e-4 * multiplier / gamma, (kind, gain, multiplier / gamma)

    marginal, total, _ = O.shadow_prices(mu, cov, sigma_H, omega, gamma, cons, W, F)
    row = marginal[marginal["group"] == "No leverage"].iloc[0]
    assert abs(row["CHF per year"] - base["budget"] / gamma * 0.01 * 12 * W) < 1e-9
    assert abs(row["bp of F per year"] - row["CHF per year"] / F * 1e4) < 1e-9
    # Total costs are >= 0, and (concave value function) at most marginal value x size of the relaxation
    assert (total["CHF per year"] >= -1e-9).all()
    lockup_total = total.loc[total["group"] == "Lock-up", "CHF per year"].iloc[0]
    assert lockup_total <= base["lower"][3] / gamma * 0.2 * 12 * W + 1e-9
    print("shadow prices (finite differences, units, bounds) OK")


def test_gamma_calibration():
    mu, cov, _, _ = random_problem(n=5, seed=4)
    benchmark = np.array([0.6, 0.4, 0, 0, 0])
    target = np.sqrt(benchmark @ cov @ benchmark)
    # Without constraints: vol = gamma x max Sharpe ratio, so gamma = target / sqrt(mu' inv(Sigma) mu)
    gamma = O.calibrate_gamma(mu, cov, benchmark, None)
    assert abs(gamma - target / np.sqrt(mu @ np.linalg.solve(cov, mu))) < 1e-12
    # With constraints: if the benchmark is the optimal portfolio of some gamma, that gamma comes back
    cons = constraints(5, lower=0.0, budget=1.0)
    for gamma_true in (0.02, 0.05, 0.1):
        benchmark = O.optimal_portfolio(mu, cov, np.zeros(5), 0.0, gamma_true, cons)
        gamma = O.calibrate_gamma(mu, cov, benchmark, cons)
        assert abs(gamma - gamma_true) < 1e-9 * gamma_true, (gamma, gamma_true)
    # A benchmark riskier than every allowed portfolio: a clear error, not a wrong number
    try:
        O.calibrate_gamma(mu, cov, np.array([0.6, 0.4, 0, 0, 0]), cons)
        raise AssertionError("an impossible target was accepted")
    except ValueError:
        pass
    print("gamma calibration OK")


# -----------------------------------------------------------------------------
# 3. Inputs: covariance, Black-Litterman, pension
# -----------------------------------------------------------------------------

def ledoit_wolf_loops(data):
    """Ledoit and Wolf (2004), constant-correlation target, written term by term from the paper."""
    t, n = data.shape
    m = data.mean(axis=0)
    s = np.array([[np.mean((data[:, i] - m[i]) * (data[:, j] - m[j])) for j in range(n)] for i in range(n)])
    r_bar = sum(s[i, j] / np.sqrt(s[i, i] * s[j, j]) for i in range(n) for j in range(i + 1, n)) * 2 / (n * (n - 1))
    f = np.array([[s[i, i] if i == j else r_bar * np.sqrt(s[i, i] * s[j, j]) for j in range(n)] for i in range(n)])
    dev = data - m
    pi = sum(np.mean((dev[:, i] * dev[:, j] - s[i, j]) ** 2) for i in range(n) for j in range(n))

    def theta(i, j):
        return np.mean((dev[:, i] ** 2 - s[i, i]) * (dev[:, i] * dev[:, j] - s[i, j]))

    rho = sum(np.mean((dev[:, i] ** 2 - s[i, i]) ** 2) for i in range(n))
    rho += sum(r_bar / 2 * (np.sqrt(s[j, j] / s[i, i]) * theta(i, j) + np.sqrt(s[i, i] / s[j, j]) * theta(j, i))
               for i in range(n) for j in range(n) if i != j)
    gamma = ((f - s) ** 2).sum()
    delta = max(0.0, min(1.0, (pi - rho) / gamma / t))
    return delta * f + (1 - delta) * s, delta


def test_ledoit_wolf():
    import pandas as pd
    rng = np.random.default_rng(7)
    for t in (24, 60, 300):
        data = rng.standard_t(5, size=(t, 6)) @ rng.normal(size=(6, 6)) * 0.02
        cov, delta = O.ledoit_wolf(pd.DataFrame(data))
        reference, delta_ref = ledoit_wolf_loops(data)
        assert abs(delta - delta_ref) < 1e-12 and np.allclose(cov.to_numpy(), reference, atol=1e-15)
        assert 0 <= delta <= 1 and np.all(np.linalg.eigvalsh(cov.to_numpy()) > 0)
        assert np.allclose(np.diag(cov), np.diag(np.cov(data.T, bias=True)))       # variances are kept
    print("Ledoit-Wolf = formulas of the paper OK")


def test_black_litterman_properties():
    _, cov, _, _ = random_problem(n=4, seed=5)
    w = np.array([0.5, 0.3, 0.2, 0.0])
    phi = O.implied_risk_aversion(cov, w, 0.4)
    pi = O.bl_expected_returns(cov, w, phi, None, None, None, 0.05)
    assert np.allclose(pi, phi * cov @ w)
    # No views, no constraints, omega = 0: the optimum is the market portfolio scaled by gamma x phi
    assert np.allclose(O.optimal_portfolio(pi, cov, np.zeros(4), 0.0, 0.5, None), 0.5 * phi * w)
    # A view held with certainty is met exactly; a view with no confidence changes nothing
    P, Q = np.array([[1.0, -1.0, 0, 0]]), np.array([0.01])
    assert abs(P @ O.bl_expected_returns(cov, w, phi, P, Q, np.array([[1e-14]]), 0.05) - Q) < 1e-9
    assert np.allclose(O.bl_expected_returns(cov, w, phi, P, Q, np.array([[1e6]]), 0.05), pi, atol=1e-9)
    print("Black-Litterman properties OK")


def test_pension_known_answers():
    p = load_params()
    p["career"]["salary_growth"] = 0.0
    p["pension"]["interest_rate"] = 0.0
    coordinated = p["pension"]["max_insured_salary"] - p["pension"]["coordination_deduction"]
    # 10 years at 7% of the coordinated salary (salary above the upper limit)
    assert abs(O.pension_savings(35, 150_000, p) - 10 * 0.07 * coordinated) < 1e-6
    # 10 years at 7% + 10 years at 10%
    assert abs(O.pension_savings(45, 150_000, p) - 10 * (0.07 + 0.10) * coordinated) < 1e-6
    # with interest: each credit grows until today
    p["pension"]["interest_rate"] = 0.0125
    expected = sum(0.07 * coordinated * 1.0125 ** k for k in range(10))
    assert abs(O.pension_savings(35, 150_000, p) - expected) < 1e-6
    # no credit before 25, nothing below the coordination deduction
    assert O.pension_savings(25, 150_000, p) == 0.0 and O.pension_savings(40, 20_000, p) == 0.0
    # self-employed: no pension fund; a number in params is used as it is
    q = load_params()
    q["client"]["pillar2"] = False
    assert O.balance_sheet(q)["P"] == 0.0
    q["client"].update({"pillar2": True, "pension_chf": 123_456})
    assert O.balance_sheet(q)["P"] == 123_456
    print("pension savings OK")


def test_total_wealth_risk():
    mu, cov, sigma_career, exposure = random_problem(n=4, seed=6)
    x = np.array([0.4, 0.2, 0.1, 0.3])
    career_var = exposure @ cov @ exposure
    omega_H, omega_P = 0.6, 0.2
    table = O.total_wealth_risk_contributions(x, cov, sigma_career, career_var, omega_H, omega_P, ["a", "b", "c", "d"])
    omega = omega_H + omega_P
    var = (1 - omega) ** 2 * x @ cov @ x + 2 * (1 - omega) * omega_H * x @ sigma_career + omega_H ** 2 * career_var
    assert abs(table["risk contribution"].sum() - np.sqrt(var * 12)) < 1e-14
    assert table.loc["PENSION", "risk contribution"] == 0 and abs(table["share of risk"].sum() - 1) < 1e-12
    # Without career and pension it is the LN5 formula on the portfolio
    table = O.total_wealth_risk_contributions(x, cov, sigma_career, career_var, 0.0, 0.0, ["a", "b", "c", "d"])
    assert np.allclose(table["risk contribution"].iloc[:4], O.risk_contributions(x, cov) * np.sqrt(12))
    print("risk contributions on total wealth OK")


# -----------------------------------------------------------------------------
# 4. On the saved data: every persona, the signs the theory predicts
# -----------------------------------------------------------------------------

def test_personas_on_saved_data():
    try:
        excess = O.load_excess_returns()
    except Exception:
        print("saved data not found, test skipped")
        return
    params = load_params()
    gammas = []
    for name, persona in params["personas"].items():
        r = O.analyze_client(params_for_persona(params, name), excess)
        cons, w, ce = r["constraints"], r["weights"], r["performance"].loc["certainty equivalent, CHF per year"]
        gammas.append(r["gamma"])
        for key in ("A", "D", "C"):
            x = r["solutions"][key]["x"]
            assert np.all(x >= cons["lower"] - 1e-12) and np.all(x <= cons["upper"] + 1e-12) and x.sum() <= 1 + 1e-12
        # (C) is the best portfolio for total wealth under the same constraints
        assert ce["C"] >= ce["A"] - 1e-6 and ce["C"] >= ce["D"] - 1e-6
        # The hedge has the sign the theory predicts: (C) holds no more of the career's assets than (D)
        for asset in r["career_holdings"].index:
            assert w.loc[asset, "C"] <= w.loc[asset, "D"] + 1e-9, (name, asset)
        assert (r["shadow_total"]["CHF per year"] >= -1e-6).all()
        assert abs(r["risk"]["C"]["share of risk"].sum() - 1) < 1e-12
    assert max(gammas) - min(gammas) < 1e-12                       # one gamma for every client
    # The (1 - omega) scaling: the calibrated no-career client is as volatile as 60/40
    _, info = O.risk_tolerance(params, excess)
    assert abs(info["achieved_vol"] - info["target_vol"]) < 1e-9

    # Retired, no pension fund, no deferred shares: omega = 0 and the three strategies coincide
    p = copy.deepcopy(params)
    p["client"].update({"age": 65, "pillar2": False, "deferred_shares_chf": 0})
    r = O.analyze_client(p, excess)
    assert r["omega"] == 0
    assert np.allclose(r["weights"]["A"], r["weights"]["C"]) and np.allclose(r["weights"]["D"], r["weights"]["C"])

    # Removing a constraint can only help, also when the employer stock is attractive and the
    # lock-up is above the cap (a view makes UBS the best asset)
    p = copy.deepcopy(params)
    p["black_litterman"]["views"] = [{"assets": {"EMPLOYER": 1.0}, "q": 0.10, "sd": 0.02}]
    r = O.analyze_client(p, excess)
    assert (r["shadow_total"]["CHF per year"] >= -1e-6).all(), r["shadow_total"]

    # Negative CHF rates (2015-2021): the optimizer works on that period too
    r = O.analyze_client(params, excess.loc["2015-01-31":"2021-12-31"])
    assert np.isfinite(r["weights"].to_numpy()).all() and r["months"] == 84
    print("all personas on saved data OK")


if __name__ == "__main__":
    test_black_litterman_lab_LN4()
    test_risk_contributions_lab_LN5()
    test_lab_QP_when_omega_is_zero()
    test_unconstrained_equals_formula()
    test_constrained_against_exact_solution()
    test_shadow_prices()
    test_gamma_calibration()
    test_ledoit_wolf()
    test_black_litterman_properties()
    test_pension_known_answers()
    test_total_wealth_risk()
    test_personas_on_saved_data()
    print("ALL TESTS PASSED")
