"""Tests of src/career.py with known answers.

Run from the project folder:
    python tests/test_career.py
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.career as C
from src.config import load_params


def base_params(**career):
    """Default params with some career values changed for a test."""
    params = copy.deepcopy(load_params())
    params["career"].update(career)
    return params


def test_career_return():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    bond, sector = pd.Series([0.01, -0.02], index=idx), pd.Series([0.10, 0.05], index=idx)
    r = C.career_return(bond, sector, beta_bond=1.5, beta_sector=0.2)
    assert np.allclose(r.values, [1.5 * 0.01 + 0.2 * 0.10, 1.5 * -0.02 + 0.2 * 0.05]) and r.name == "CAREER"
    print("career_return OK")


def test_human_capital():
    # 1. No salary growth, fixed betas: H = salary x annuity factor (1 - (1+r)^-N) / r
    p = base_params(beta_bond=1.0, beta_sector=0.0, salary_growth=0.0, rf_long_run=0.03, bond_premium=0.01)
    r, n, salary = 0.04, 30, 100_000                           # 3% + 1 x 1% = 4%, age 35 -> 30 years
    expected = salary * (1 - (1 + r) ** -n) / r
    assert abs(C.human_capital(35, salary, p) - expected) < 1e-6, (C.human_capital(35, salary, p), expected)

    # 2. Salary growth equal to the discount rate: every year is worth salary / (1 + r) today
    p = base_params(beta_bond=1.0, beta_sector=0.0, salary_growth=0.04, rf_long_run=0.03, bond_premium=0.01)
    assert abs(C.human_capital(35, salary, p) - n * salary / 1.04) < 1e-6

    # 3. Retired: no human capital left
    assert C.human_capital(65, salary, load_params()) == 0.0 and C.human_capital(70, salary, load_params()) == 0.0

    # 4. H is proportional to the salary
    p = load_params()
    assert abs(C.human_capital(40, 200_000, p) - 2 * C.human_capital(40, 100_000, p)) < 1e-6
    print("human_capital OK")


def test_duration_and_betas():
    # One payment in 1 year: Macaulay duration 1, modified duration 1 / (1 + r)
    pv, dur = C.present_value_and_duration(np.array([1]), np.array([100.0]), 0.05)
    assert abs(pv - 100 / 1.05) < 1e-12 and abs(dur - 1 / 1.05) < 1e-12

    # "auto": the solution satisfies beta_bond = duration(rate(beta_bond)) / bond_duration
    p = load_params()
    beta_bond, beta_sector, rate = C.career_betas(35, p)
    times, flows = C.salary_cash_flows(35, 1.0, p)
    _, duration = C.present_value_and_duration(times, flows, rate)
    link = p["career"]["wage_rate_link"]
    assert abs(beta_bond - (1 - link) * duration / p["career"]["bond_duration"]) < 1e-9
    assert abs(rate - C.discount_rate(beta_bond, beta_sector, p)) < 1e-12

    # beta_bond falls with age (shorter salary stream) and is 0 at retirement
    betas = [C.career_betas(age, p)[0] for age in (35, 45, 55, 64, 65)]
    assert all(a > b for a, b in zip(betas, betas[1:])) and betas[-1] == 0.0

    # a number in params overrides the rule, but never after retirement
    assert C.career_betas(35, base_params(beta_bond=2.5))[0] == 2.5
    assert C.career_betas(65, base_params(beta_bond=2.5))[:2] == (0.0, 0.0)
    print("duration and betas OK")


def test_wage_rate_link():
    # link = 0: pure duration rule (salary = fixed nominal bond)
    p0 = base_params(wage_rate_link=0.0)
    b0, bs, rate0 = C.career_betas(35, p0)
    times, flows = C.salary_cash_flows(35, 1.0, p0)
    assert abs(b0 - C.present_value_and_duration(times, flows, rate0)[1] / p0["career"]["bond_duration"]) < 1e-9
    # link = 1: salaries follow rates completely, no interest-rate risk
    assert C.career_betas(35, base_params(wage_rate_link=1.0))[0] == 0.0
    # beta_bond falls when the link rises
    betas = [C.career_betas(35, base_params(wage_rate_link=k))[0] for k in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a > b for a, b in zip(betas, betas[1:]))
    # values outside [0, 1] are refused
    try:
        C.career_betas(35, base_params(wage_rate_link=1.5))
        raise AssertionError("wage_rate_link = 1.5 should be refused")
    except ValueError:
        pass
    print("wage_rate_link OK")


def test_career_covariance():
    # Known answer: inv(Sigma) sigma_H = beta_bond x e_B + beta_sector x e_S
    p = load_params()
    names = ["EQ_WORLD", "BANKS_EU", "BOND_CHF", "GOLD"]
    rng = np.random.default_rng(1)
    a = rng.normal(size=(200, 4)) * [0.04, 0.07, 0.015, 0.04]
    cov = pd.DataFrame(np.cov(a.T), index=names, columns=names)
    sigma_h = C.career_covariance(cov, beta_bond=1.3, beta_sector=0.2, params=p)
    hedge = np.linalg.solve(cov.to_numpy(), sigma_h)
    assert np.allclose(hedge, [0, 0.2, 1.3, 0])
    print("career_covariance OK")


def test_personas_and_no_sector():
    p = load_params()
    for name, persona in p["personas"].items():
        q = C.params_for_persona(p, name)
        assert q["client"]["age"] == persona["age"] and q["data"]["career_sector_asset"] == persona["sector"]
        assert C.human_capital(persona["age"], persona["salary_chf"], q) > 0
    assert p["client"]["age"] == 35                        # the original params are not changed
    # a career with no sector exposure only hedges the bond
    q = C.params_for_persona(p, "civil_servant")
    names = ["EQ_WORLD", "BANKS_EU", "BOND_CHF"]
    cov = pd.DataFrame(np.diag([0.002, 0.005, 0.0003]), index=names, columns=names)
    sigma_h = C.career_covariance(cov, beta_bond=1.2, beta_sector=0.0, params=q)
    assert np.allclose(np.linalg.solve(cov.to_numpy(), sigma_h), [0, 0, 1.2])
    print("personas OK")


def test_on_saved_data():
    # If the data has been downloaded: the career series has the same dates and no holes
    from src.data import load_returns
    try:
        returns = load_returns()
    except Exception:
        print("saved data not found, test skipped")
        return
    career = C.career_excess_returns()
    assert career.index.equals(returns.index) and career.notna().all()

    # Sanity check: a civil servant (no sector risk) must have LESS hedgeable career risk
    # than the banker. With wage_rate_link = 0 this fails, which is why the link exists.
    p = load_params()
    vol = {name: C.career_excess_returns(p["personas"][name]["age"], C.params_for_persona(p, name)).std()
           for name in ("banker", "civil_servant")}
    assert vol["civil_servant"] < vol["banker"], vol
    print("career on saved data OK (civil servant less risky than banker)")


if __name__ == "__main__":
    test_career_return()
    test_human_capital()
    test_duration_and_betas()
    test_wage_rate_link()
    test_career_covariance()
    test_personas_and_no_sector()
    test_on_saved_data()
    print("ALL TESTS PASSED")
