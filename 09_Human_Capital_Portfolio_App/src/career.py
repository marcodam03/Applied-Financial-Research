"""Career model.

The client's career is treated as a non-tradable asset with two parts:

    career excess return = beta_bond   x (R_B - rf)     bond-like part   (BOND_CHF)
                         + beta_sector x (R_S - rf)     sector part      (BANKS_EU)
                         + noise                        personal, cannot be hedged

H, the value of human capital, is the present value of the remaining salary
until retirement. The same betas give the discount rate of H, so one model gives
both the VALUE and the RISK of the career.

beta_bond follows from duration (params: beta_bond = "auto"): a salary stream is a
series of future cash flows, so its value moves with interest rates by its duration.
But a salary is not a fixed nominal bond: when rates rise with inflation, wages tend to
rise too. wage_rate_link (0 to 1) is the share of rate moves that salaries follow, so
    beta_bond = (1 - wage_rate_link) x modified duration of the salary / duration of BOND_CHF.
With wage_rate_link = 0 every career looks like a long nominal bond, and the safest careers
(civil servant) look riskier than the banker. It falls with age and is 0 at retirement.

Main functions:
    career_return(bond_ret, sector_ret, beta_bond, beta_sector)  -> Series
    human_capital(age, salary, params)                           -> float (CHF)
Helpers used by the optimizer, the backtest and the app:
    params_for_persona(params, name)             -> params set up for one persona (banker, dentist, ...)
    career_betas(age, params)                    -> beta_bond, beta_sector, discount rate
    career_covariance(cov, beta_bond, beta_sector, params) -> sigma_H = Cov(R, R_H)
    career_excess_returns(age, params, start, end)          -> career excess returns on the data

Run from the project folder for a summary of the default client:
    python -m src.career
"""
import copy

import numpy as np
import pandas as pd

from src.config import load_params

MAX_ITERATIONS = 200
TOLERANCE = 1e-12


# =============================================================================
# 0. Personas: the same model for any client
# =============================================================================

def params_for_persona(params, name):
    """Copy of params set up for one persona of params['personas'] (banker, dentist, ...):
    age, salary, sector asset, beta_sector, pension fund and deferred employer shares.
    Every function below (and the optimizer) then works unchanged."""
    persona = params["personas"][name]
    p = copy.deepcopy(params)
    p["client"].update({"persona": name, "age": persona["age"], "salary_chf": persona["salary_chf"],
                        "pillar2": persona.get("pillar2", True),
                        "pension_chf": persona.get("pension_chf", "auto"),
                        "deferred_shares_chf": persona.get("deferred_shares_chf", 0)})
    p["career"]["beta_sector"] = persona["beta_sector"] if persona["sector"] else 0.0
    p["data"]["career_sector_asset"] = persona["sector"]
    return p


# =============================================================================
# 1. Career return
# =============================================================================

def career_return(bond_ret, sector_ret, beta_bond, beta_sector):
    """Monthly excess return of the career (without the personal noise, which
    cannot be hedged and is uncorrelated with the assets).

    bond_ret, sector_ret: monthly EXCESS returns (Series) of BOND_CHF and BANKS_EU."""
    career = beta_bond * bond_ret + beta_sector * sector_ret
    career.name = "CAREER"
    return career


# =============================================================================
# 2. Salary stream, discount rate, duration
# =============================================================================

def salary_cash_flows(age, salary, params):
    """Salaries still to be earned, one per year, paid at the end of each year:
    salary x (1 + g)^(k-1) for k = 1, ..., retirement_age - age.
    Returns (times in years, cash flows in CHF). Empty if the client is retired."""
    years = int(params["client"]["retirement_age"] - age)
    if years <= 0:
        return np.array([]), np.array([])
    g = params["career"]["salary_growth"]
    times = np.arange(1, years + 1)
    flows = salary * (1 + g) ** (times - 1)
    return times, flows


def discount_rate(beta_bond, beta_sector, params):
    """Discount rate of H = long-run risk-free rate + the risk premia of the same betas."""
    c = params["career"]
    return c["rf_long_run"] + beta_bond * c["bond_premium"] + beta_sector * c["sector_premium"]


def present_value_and_duration(times, flows, rate):
    """Present value and MODIFIED duration of a stream of annual cash flows.
    Modified duration = Macaulay duration / (1 + rate): the % change in value
    for a 1 percentage point change in rates, the same unit as a bond index duration."""
    if len(flows) == 0:
        return 0.0, 0.0
    discounted = flows / (1 + rate) ** times
    pv = discounted.sum()
    if pv == 0:                                          # no salary (0 CHF): no value and no duration, not 0/0
        return 0.0, 0.0
    macaulay = (times * discounted).sum() / pv
    return pv, macaulay / (1 + rate)


def career_betas(age, params):
    """beta_bond, beta_sector and the discount rate of H for a client of this age.

    If params['career']['beta_bond'] is a number, it is used as it is (0 once retired).
    If it is "auto", beta_bond = (1 - wage_rate_link) x modified duration of the salary
    stream / bond_duration.
    The discount rate depends on beta_bond and the duration depends on the discount
    rate, so the two are solved together (a few iterations, it converges quickly).
    The salary level does not matter: duration does not change if all cash flows double."""
    c = params["career"]
    beta_sector = c["beta_sector"]
    times, flows = salary_cash_flows(age, 1.0, params)

    if len(flows) == 0:                                  # retired: no career left, no career risk
        return 0.0, 0.0, discount_rate(0.0, 0.0, params)

    if c["beta_bond"] != "auto":
        beta_bond = float(c["beta_bond"])
        return beta_bond, beta_sector, discount_rate(beta_bond, beta_sector, params)

    link = c.get("wage_rate_link", 0.0)
    if not 0.0 <= link <= 1.0:
        raise ValueError(f"wage_rate_link must be between 0 and 1, got {link}")
    beta_bond = 1.0
    for _ in range(MAX_ITERATIONS):
        rate = discount_rate(beta_bond, beta_sector, params)
        _, duration = present_value_and_duration(times, flows, rate)
        new_beta = (1.0 - link) * duration / c["bond_duration"]
        if abs(new_beta - beta_bond) < TOLERANCE:
            break
        beta_bond = new_beta
    return new_beta, beta_sector, discount_rate(new_beta, beta_sector, params)


# =============================================================================
# 3. Value of human capital
# =============================================================================

def human_capital(age, salary, params=None):
    """Value of human capital H in CHF: present value of the remaining salary
    until retirement, discounted at the rate given by the career betas.
    Returns 0 at or after the retirement age."""
    if params is None:
        params = load_params()
    times, flows = salary_cash_flows(age, salary, params)
    _, _, rate = career_betas(age, params)
    pv, _ = present_value_and_duration(times, flows, rate)
    return float(pv)


# =============================================================================
# 4. Inputs for the optimizer and the backtest
# =============================================================================

def career_covariance(cov, beta_bond, beta_sector, params=None):
    """sigma_H = Cov(R, R_H): covariance of each asset with the career, aligned
    with the columns of cov (a DataFrame with the project's asset names).

    Because the career is beta_bond x R_B + beta_sector x R_S + noise, and the noise is
    uncorrelated with the assets: sigma_H = beta_bond x Sigma[:, B] + beta_sector x Sigma[:, S].
    Then inv(Sigma) sigma_H = beta_bond x e_B + beta_sector x e_S: the hedge is simply
    "hold beta_bond less of the bond and beta_sector less of the sector"."""
    if params is None:
        params = load_params()
    bond, sector = params["data"]["career_bond_asset"], params["data"]["career_sector_asset"]
    sigma_h = beta_bond * cov[bond]
    if sector is not None and beta_sector != 0:
        sigma_h = sigma_h + beta_sector * cov[sector]       # the sector must be a column of cov
    return sigma_h.to_numpy()


def career_excess_returns(age=None, params=None, start=None, end=None):
    """Monthly excess returns of the career on the saved data (for the backtest
    and for checks), with the betas of a client of this age. The sector can be a
    core asset (BANKS_EU) or a column of the sector menu (SEC_TECH, ...)."""
    from src.data import load_returns, load_rf, load_sector_returns
    if params is None:
        params = load_params()
    if age is None:
        age = params["client"]["age"]
    rf = load_rf(start, end)
    returns = load_returns(start, end)
    sector = params["data"]["career_sector_asset"]
    if sector is not None and sector not in returns.columns:
        returns = returns.join(load_sector_returns(start, end)[[sector]])
    excess = returns.sub(rf, axis=0)
    beta_bond, beta_sector, _ = career_betas(age, params)
    sector_ret = excess[sector] if sector is not None else 0.0 * excess[params["data"]["career_bond_asset"]]
    return career_return(excess[params["data"]["career_bond_asset"]], sector_ret, beta_bond, beta_sector)


# =============================================================================
# 5. Summary of the default client
# =============================================================================

def summary(params=None):
    """Table by age: betas, discount rate, duration and H for the default salary."""
    if params is None:
        params = load_params()
    salary = params["client"]["salary_chf"]
    rows = []
    for age in range(params["client"]["age"], params["client"]["retirement_age"] + 1, 5):
        beta_bond, beta_sector, rate = career_betas(age, params)
        times, flows = salary_cash_flows(age, salary, params)
        pv, duration = present_value_and_duration(times, flows, rate)
        rows.append({"age": age, "beta_bond": beta_bond, "beta_sector": beta_sector,
                     "discount rate": rate, "duration (mod.)": duration, "H (CHF)": pv})
    return pd.DataFrame(rows).set_index("age")


def personas_table(params=None):
    """H, betas and discount rate today for every persona."""
    if params is None:
        params = load_params()
    rows = []
    for name, persona in params["personas"].items():
        p = params_for_persona(params, name)
        beta_bond, beta_sector, rate = career_betas(persona["age"], p)
        rows.append({"persona": persona["label"], "age": persona["age"], "salary": persona["salary_chf"],
                     "sector": persona["sector"] or "-", "beta_bond": beta_bond, "beta_sector": beta_sector,
                     "discount rate": rate, "H (CHF)": human_capital(persona["age"], persona["salary_chf"], p)})
    return pd.DataFrame(rows).set_index("persona")


if __name__ == "__main__":
    params = load_params()
    c = params["client"]
    print(f"Default client: age {c['age']}, salary CHF {c['salary_chf']:,.0f}, retirement at {c['retirement_age']}")
    print(f"H today = CHF {human_capital(c['age'], c['salary_chf'], params):,.0f}\n")
    print(summary(params).to_string(formatters={"beta_bond": "{:.2f}".format, "beta_sector": "{:.2f}".format,
                                                "discount rate": "{:.2%}".format, "duration (mod.)": "{:.1f}".format,
                                                "H (CHF)": "{:,.0f}".format}))

    print("\nAll personas (today):")
    print(personas_table(params).to_string(formatters={"salary": "{:,.0f}".format, "beta_bond": "{:.2f}".format,
                                                       "beta_sector": "{:.2f}".format, "discount rate": "{:.2%}".format,
                                                       "H (CHF)": "{:,.0f}".format}))

    print("\nHedgeable career risk on the data, 2000 to today (betas of each persona;"
          " the personal part, promotion or job change, is not included):")
    for name, persona in params["personas"].items():
        try:
            career = career_excess_returns(persona["age"], params_for_persona(params, name))
            print(f"  {persona['label']:26s} volatility {career.std() * np.sqrt(12):6.2%} per year")
        except Exception as error:
            print(f"  {persona['label']:26s} not available ({error})")
