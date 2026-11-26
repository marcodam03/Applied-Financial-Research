"""Tests of src/backtest.py with known answers.

Run from the project folder (about one minute):
    python tests/test_backtest.py
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.backtest as B
import src.optimizer as O
from src.career import human_capital, params_for_persona
from src.config import load_params


def fake_data(months=30, seed=0, columns=("X", "Y")):
    """Made-up monthly returns and a CHF rate that is sometimes negative."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    returns = pd.DataFrame(rng.normal(0.005, 0.04, size=(months, len(columns))), index=dates, columns=list(columns))
    rf = pd.Series(rng.normal(0.0002, 0.001, size=months), index=dates)
    return returns, rf


def small_params(window=6, cost_bp=0.0):
    p = copy.deepcopy(load_params())
    p["backtest"]["window_months"], p["backtest"]["transaction_cost_bp"] = window, cost_bp
    return p


# -----------------------------------------------------------------------------
# 1. The engine
# -----------------------------------------------------------------------------

def test_engine_known_answers():
    returns, rf = fake_data()
    later = returns.index[6:]
    # Fixed weights, no costs: value = 100 x prod(1 + rf + w'(r - rf)), from the first month after the window
    w = np.array([0.6, 0.3])
    r = B.run_backtest(returns, rf, lambda window, date, state: w, small_params(), initial_value=100.0)
    expected = 100.0 * np.prod(1 + rf[later] + (returns.loc[later] - rf[later].to_numpy()[:, None]) @ w)
    assert r.index.equals(later) and abs(r["value"].iloc[-1] - expected) < 1e-9
    assert np.allclose(r["cash"], 0.1)
    # All cash: the value grows at the CHF rate, negative months included
    r = B.run_backtest(returns, rf, lambda window, date, state: np.zeros(2), small_params(), initial_value=100.0)
    assert abs(r["value"].iloc[-1] - 100.0 * np.prod(1 + rf[later])) < 1e-9

    # Costs: zero returns, 100% X then 100% Y every other month. Turnover 1 the first month (from
    # cash), then 2; each month the value falls by 10 bp x turnover.
    dates = pd.date_range("2000-01-31", periods=10, freq="ME")
    zero = pd.DataFrame(0.0, index=dates, columns=["X", "Y"])
    flip = lambda window, date, state: np.array([1.0, 0.0]) if date.month % 2 else np.array([0.0, 1.0])
    r = B.run_backtest(zero, pd.Series(0.0, index=dates), flip, small_params(window=2, cost_bp=10), initial_value=100.0)
    assert np.allclose(r["turnover"], [1] + [2] * 7)
    assert abs(r["value"].iloc[-1] - 100.0 * np.prod(1 - 0.001 * r["turnover"])) < 1e-12
    assert abs(r["cost"].sum() - (100.0 - r["value"].iloc[-1])) < 1e-12

    # Savings: zero returns and no costs, all cash: the value is the start plus every saving
    path = pd.DataFrame({"savings": 5.0, "H_start": 0.0, "P_start": 0.0}, index=dates)
    r = B.run_backtest(zero, pd.Series(0.0, index=dates), lambda window, date, state: np.zeros(2),
                       small_params(window=2), path=path, initial_value=100.0)
    assert abs(r["value"].iloc[-1] - (100.0 + 5.0 * 8)) < 1e-12 and np.allclose(r["W"], r["value_start"])
    print("engine (fixed weights, cash, costs, savings) OK")


def test_no_look_ahead_and_errors():
    returns, rf = fake_data(months=40)
    seen = {}

    def spy(window, date, state):
        seen[date] = (window.index[0], window.index[-1], len(window))
        return np.array([0.5, 0.5])

    r = B.run_backtest(returns, rf, spy, small_params(window=12))
    for date, (first, last, n) in seen.items():
        assert last < date and n == 12 and last == returns.index[returns.index.get_loc(date) - 1]
    # Changing the future does not change the past: weights depending on the window only
    strategy = lambda window, date, state: np.clip(window.mean().to_numpy() * 50, 0, 0.5)
    base = B.run_backtest(returns, rf, strategy, small_params(window=12))
    future = returns.copy()
    future.iloc[30:] = -0.5
    changed = B.run_backtest(future, rf, strategy, small_params(window=12))
    weights = [c for c in base.columns if c.startswith("w_")]
    assert np.allclose(base[weights].iloc[:18], changed[weights].iloc[:18])          # decisions before month 31
    assert not np.allclose(base["value"].iloc[-1], changed["value"].iloc[-1])

    for bad in (lambda: B.run_backtest(returns, rf.iloc[1:], spy, small_params()),
                lambda: B.run_backtest(returns.iloc[:5], rf.iloc[:5], spy, small_params()),
                lambda: B.run_backtest(returns, rf, lambda w, d, s: np.array([np.nan, 0.0]), small_params()),
                lambda: B.run_backtest(returns, rf, lambda w, d, s: np.zeros(3), small_params())):
        try:
            bad()
            raise AssertionError("bad input accepted")
        except ValueError:
            pass
    print("no look-ahead, errors OK")


# -----------------------------------------------------------------------------
# 2. The client over time
# -----------------------------------------------------------------------------

def client_data(months=60, sector_path=None, params=None):
    """Excess returns where bonds and the sector earn exactly the career model's premia (no shock),
    unless a sector path is given."""
    p = load_params() if params is None else params
    dates = pd.date_range("2005-01-31", periods=months, freq="ME")
    excess = pd.DataFrame({p["data"]["career_bond_asset"]: p["career"]["bond_premium"] / 12,
                           p["data"]["career_sector_asset"]: p["career"]["sector_premium"] / 12}, index=dates)
    if sector_path is not None:
        excess[p["data"]["career_sector_asset"]] = sector_path
    return excess, pd.Series(0.0, index=dates)


def test_client_path():
    p = load_params()
    excess, rf = client_data()
    path = B.client_path(p, excess, rf, excess.index[0])
    c = p["client"]
    # No shock: at every birthday H is exactly the step 2 model value, and in between it is smooth
    for k in (0, 12, 24, 48):
        salary = c["salary_chf"] * (1 + p["career"]["salary_growth"]) ** (k / 12)
        assert abs(path["H_start"].iloc[k] - human_capital(c["age"] + k // 12, salary, p)) < 1e-6
    assert path["H_start"].pct_change().abs().max() < 0.02
    # Savings = savings_rate x salary / 12; age +1/12 per month
    assert np.allclose(path["savings"], p["backtest"]["savings_rate"] * path["salary"] / 12)
    assert abs(path["age"].iloc[12] - (c["age"] + 1)) < 1e-12
    # Pension: P0 from the balance sheet, then interest + age credits (salary growth 0 for a closed form)
    q = copy.deepcopy(p)
    q["career"]["salary_growth"] = 0.0
    path = B.client_path(q, excess, rf, excess.index[0])
    i = q["pension"]["interest_rate"] / 12
    credit = 0.10 * (q["pension"]["max_insured_salary"] - q["pension"]["coordination_deduction"]) / 12   # age 35-44
    expected = O.balance_sheet(q)["P"] * (1 + i) ** 12 + credit * sum((1 + i) ** j for j in range(12))
    assert abs(path["P_start"].iloc[12] - expected) < 1e-6
    # Retired: no career, no savings
    r = copy.deepcopy(p)
    r["client"]["age"] = 66
    path = B.client_path(r, excess, rf, excess.index[0])
    assert (path["H_start"] == 0).all() and (path["savings"] == 0).all()
    print("client path (H = model without shocks, savings, pension) OK")


def test_job_loss():
    p = load_params()
    months = 60
    sector = np.zeros(months)
    sector[[10, 11, 12]] = -0.15        # the sector falls about 39% in three months
    sector[40:43] = -0.15                # again, after the new job
    excess, rf = client_data(months, sector)
    path = B.client_path(p, excess, rf, excess.index[0], job_loss_threshold=0.30)
    lost = list(np.flatnonzero(path["job_lost"]))
    # 0.85^2 = -27.8%: not yet; 0.85^3 = -38.6%: the job is lost at the end of month 12
    assert lost[0] == 12, lost
    k = 12
    assert path["employed"].iloc[k] and not path["employed"].iloc[k + 1:k + 13].any() and path["employed"].iloc[k + 13]
    assert (path["savings"].iloc[k + 1:k + 13] == 0).all()
    # H falls by the year of salary that will not be paid, then is back to the model after the year
    no_loss = B.client_path(p, excess, rf, excess.index[0])
    gap = path["salary"].iloc[k] * 12 / 12
    assert abs(no_loss["H_start"].iloc[k + 1] - path["H_start"].iloc[k + 1] - gap) < 1e-6
    assert abs(path["job_loss_return"].iloc[k] + gap / path["H_start"].iloc[k]) < 1e-15
    assert abs(path["H_start"].iloc[k + 13] - no_loss["H_start"].iloc[k + 13]) < 1e-6
    # No pension credits while unemployed
    assert path["P_start"].iloc[k + 13] < no_loss["P_start"].iloc[k + 13]
    # The peak restarts with the new job: the second fall costs the job again
    assert len(lost) == 2 and lost[1] == 42, lost
    # A civil servant has no sector: never loses his job
    civil = params_for_persona(p, "civil_servant")
    excess_c = excess.drop(columns=p["data"]["career_sector_asset"])
    assert not B.client_path(civil, excess_c, rf, excess.index[0], job_loss_threshold=0.30)["job_lost"].any()
    print("job loss (trigger, unemployment, H, pension, new peak) OK")


# -----------------------------------------------------------------------------
# 3. Metrics and statistics
# -----------------------------------------------------------------------------

def test_metric_formulas():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    assert abs(B.certainty_equivalent_return(r, 0.5) - (12 * r.mean() - 12 * r.var() / 1.0)) < 1e-15
    assert abs(B.max_drawdown([0.10, -0.50, 0.20]) - (-0.5)) < 1e-15
    ret, risk = B.split_gain(r, r * 2, 0.5)
    assert abs(ret + risk - (B.certainty_equivalent_return(r * 2, 0.5) - B.certainty_equivalent_return(r, 0.5))) < 1e-15
    # Newey-West: with 0 lags it is the plain t-stat (variance / T); with 1 lag the Bartlett formula
    x = np.random.default_rng(1).normal(0.001, 0.02, 300)
    mean, se, t = B.newey_west_tstat(x, lags=0)
    assert abs(se - x.std() / np.sqrt(300)) < 1e-15
    d = x - x.mean()
    lr = d @ d / 300 + 2 * 0.5 * (d[1:] @ d[:-1]) / 300
    assert abs(B.newey_west_tstat(x, lags=1)[1] - np.sqrt(lr / 300)) < 1e-15
    # Bootstrap: same series, zero gain; fixed seed, same draws
    g = B.block_bootstrap_gain(x, x, 0.3, 12, 200, 0)
    assert np.allclose(g, 0) and np.allclose(B.block_bootstrap_gain(x, x * 1.1, 0.3, 12, 200, 0),
                                             B.block_bootstrap_gain(x, x * 1.1, 0.3, 12, 200, 0))
    print("metric formulas OK")


def test_wrong_beta_theory():
    # Theory: a hedge h against a true exposure beta saves variance
    # beta^2 - (beta - h)^2 = h (2 beta - h). Total wealth = beta x s + other risk (unrelated to s);
    # the hedged one = (beta - h) x s + the same. The risk part of the gain must be h(2 beta - h) var(s) x 12/(2 gamma).
    rng = np.random.default_rng(3)
    s = rng.normal(0, 0.05, 240)
    other = rng.normal(0, 0.03, 240)
    other = other - other.mean() - s * ((other - other.mean()) @ (s - s.mean())) / ((s - s.mean()) @ (s - s.mean()))
    gamma, h = 0.3, 0.4
    for beta in (0.0, 0.1, 0.2, 0.4, 0.8):
        _, risk = B.split_gain(beta * s + other, (beta - h) * s + other, gamma)
        assert abs(risk - h * (2 * beta - h) * s.var(ddof=1) * 12 / (2 * gamma)) < 1e-15
    print("wrong-beta theory h(2 beta - h) OK")


# -----------------------------------------------------------------------------
# 4. On the saved data
# -----------------------------------------------------------------------------

def test_on_saved_data():
    try:
        data = B.load_backtest_data()
    except Exception:
        print("saved data not found, tests on real data skipped")
        return
    p = load_params()
    run = B.run_strategies(p, data)
    res, path, gamma = run["results"], run["path"], run["gamma"]
    window = p["backtest"]["window_months"]
    returns, rf = data
    assets = O.client_assets(p)

    # gamma from the first window only; the same for the rerun (same numbers on every run)
    assert abs(gamma - O.risk_tolerance(p, returns.sub(rf, axis=0).iloc[:window])[0]) < 1e-15
    again = B.run_strategies(p, data, strategies=("C",))
    assert np.allclose(again["results"]["C"]["value"], res["C"]["value"], rtol=0, atol=1e-9)
    assert res["A"].index[0] == returns.index[window]

    # Month 1 of (C) = the optimizer by hand, with the first window and the client of that month
    first = res["C"].index[0]
    cov, _ = O.estimate_covariance(returns[assets].sub(rf, axis=0).iloc[:window], p)
    mu, _ = O.expected_returns(cov, p)
    F, H, P = res["C"]["value_start"].iloc[0], path["H_start"].iloc[0], path["P_start"].iloc[0]
    W = F + H + P
    from src.career import career_covariance
    sigma_H = H / (H + P) * career_covariance(cov, path["beta_bond"].iloc[0], path["beta_sector"].iloc[0], p)
    deferred = p["client"]["deferred_shares_chf"]
    x = O.optimal_portfolio(mu.to_numpy(), cov.to_numpy(), sigma_H, (H + P) / W, gamma,
                            O.build_constraints(assets, p, deferred / F))
    assert np.allclose(res["C"][[f"w_{a}" for a in assets]].iloc[0], x, atol=1e-12) and first == path.index[0]

    # Every month, every strategy respects the constraints (lock-up, caps, long-only, no leverage)
    for name, r in res.items():
        w = r[[f"w_{a}" for a in assets]].to_numpy()
        lock = np.minimum(deferred / r["value_start"].to_numpy(), 1.0)
        assert (w >= -1e-12).all() and (w.sum(axis=1) <= 1 + 1e-12).all()
        assert (w[:, assets.index(O.EMPLOYER)] >= lock - 1e-12).all()
        assert (w[:, assets.index("BANKS_EU")] <= p["optimization"]["position_caps"]["BANKS_EU"] + 1e-12).all()
    # (B): (100 - age)% in world equities, the rest in CHF bonds, around the deferred shares
    b, age = res["B"], path["age"]
    lock = np.minimum(deferred / b["value_start"], 1.0)
    assert np.allclose(b["w_EQ_WORLD"], (100 - age) / 100 * (1 - lock)) and np.allclose(b["w_EMPLOYER"], lock)
    assert np.allclose(b["w_BOND_CHF"], (age / 100) * (1 - lock))
    # (C) never holds more of the banker's sector than (D) on average (the hedge has the right sign)
    assert res["C"]["w_BANKS_EU"].mean() <= res["D"]["w_BANKS_EU"].mean()

    # Fair comparison: with the shares of (A), gains add up, and parts add up to the gain
    gains = B.gain_table(run)
    assert abs(gains.loc["(C) vs (A)", "CE gain (CHF/yr)"]
               - gains.loc["(C) vs (D)", "CE gain (CHF/yr)"] - gains.loc["(D) vs (A)", "CE gain (CHF/yr)"]) < 1e-6
    assert np.allclose(gains["return part (CHF/yr)"] + gains["risk part (CHF/yr)"], gains["CE gain (CHF/yr)"])
    e_a, e_c = B.total_wealth_excess(res["A"], path, res["A"]), B.total_wealth_excess(res["C"], path, res["A"])
    F, W = res["A"]["value_start"], res["A"]["W"]
    assert np.allclose(e_c - e_a, F / W * (res["C"]["return"] - res["A"]["return"]), atol=1e-15)
    metrics = B.metrics_table(run)
    assert metrics.shape[1] == 4 and np.isfinite(metrics.loc["CE total wealth (CHF/yr)"]).all()
    crises = B.crisis_table(run)
    assert ("2008 financial crisis", "C") in crises.index
    print(f"saved data OK ({len(path)} months, gamma {gamma:.3f}; (C) vs (A) "
          f"CHF {gains.loc['(C) vs (A)', 'CE gain (CHF/yr)']:,.0f}/yr, p = {gains.loc['(C) vs (A)', 'p-value']:.3f})")


def test_robustness_on_saved_data():
    try:
        data = B.load_backtest_data()
    except Exception:
        print("saved data not found, robustness tests skipped")
        return
    p = load_params()
    curve = B.wrong_beta_curve(p, data)
    # Decisions fixed: the risk part of the hedge is a straight line in the true beta, rising
    risk = curve["risk part (CHF/yr)"].to_numpy()
    assert curve.attrs["risk part: R2 of the straight line"] > 0.999 and (np.diff(risk) > 0).all()
    jobs = B.job_loss_table(p, data)
    counts = [len(s.split(", ")) if s != "-" else 0 for s in jobs["jobs lost"]]
    assert counts[0] == 0 and counts[1] >= counts[2] >= counts[3] >= 1, counts      # none, 20%, 30%, 40%
    prior = B.prior_sensitivity(p, data)
    assert len(prior) == 2 and np.isfinite(prior.to_numpy(dtype=float)).all()
    q = copy.deepcopy(p)
    q["backtest"]["hedge_map_portfolios"], q["backtest"]["hedge_map_betas"] = [100_000, 3_000_000], [0.0, 0.60]
    hm = B.hedge_map(q, data)
    assert np.allclose(hm["return part (CHF/yr)"] + hm["risk part (CHF/yr)"], hm["hedge gain (CHF/yr)"])
    # The hedge reduces risk more with a bigger portfolio and a stronger sector link
    assert hm.loc[(3_000_000, 0.60), "risk part (CHF/yr)"] > hm.loc[(100_000, 0.60), "risk part (CHF/yr)"] > 0
    print(f"robustness OK (wrong beta R2 {curve.attrs['risk part: R2 of the straight line']:.4f}; "
          f"job losses at 20/30/40%: {counts[1:]})")


if __name__ == "__main__":
    test_engine_known_answers()
    test_no_look_ahead_and_errors()
    test_client_path()
    test_job_loss()
    test_metric_formulas()
    test_wrong_beta_theory()
    test_on_saved_data()
    test_robustness_on_saved_data()
    print("ALL TESTS PASSED")
