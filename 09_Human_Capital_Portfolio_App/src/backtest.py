"""Backtest engine.

The question: would a client who followed the total-wealth portfolio have been better off
than one who followed a standard portfolio, and for which reason?

    client       the persona in params (the banker: 35 years old in Sep 2005), followed month
                 by month: age, salary, career value H, pension P, savings into the portfolio
    strategies   (A) standard MVO, career ignored     (B) 100 - age rule
                 (D) total wealth without the hedge   (C) total wealth
                 (A), (D), (C): same gamma, Ledoit-Wolf covariance, Black-Litterman returns and
                 constraints (lock-up included); only the client's balance sheet differs
    walk-forward the weights of month t are decided with the window_months months BEFORE t only,
                 then held during month t; costs = 10 bp x turnover; cash earns the CHF rate of
                 the month (negative from 2015 to 2022)
    headline     certainty equivalent of total wealth, in CHF per year
    readable     correlation with the sector part of the career, CHF result in the sector's worst
                 months, CHF result in the crises
    robustness   significance (Newey-West, block bootstrap), wrong-beta curve with the theory
                 h(2 beta - h), job-loss thresholds 20-40%, prior (global vs Swiss),
                 hedge map (portfolio size x beta_sector)

Fair comparison. With its own wealth path, a strategy that became richer has a smaller
career share later, a result of past luck rather than of the decision. So strategies are
compared with the SAME wealth shares, those of a reference strategy: the excess return of total wealth of strategy x is
    R_W(x) = (F/W) (r_x - rf) + (H/W) career return + (P/W) pension excess return
with F, H, P, W of the reference. Gains over (A) then add up: (C)-(A) = ((C)-(D)) + ((D)-(A)).

Run from the project folder (about a minute):
    python -m src.backtest
"""
import copy

import numpy as np
import pandas as pd

from src.career import career_betas, career_covariance, human_capital, params_for_persona
from src.config import load_params
from src.optimizer import (EMPLOYER, MONTHS, balance_sheet, build_constraints, client_assets,
                           estimate_covariance, expected_returns, risk_tolerance, solve_portfolio)

BP = 10_000
STRATEGIES = {"A": "(A) standard: career ignored", "B": "(B) 100 - age",
              "D": "(D) total wealth, no hedge", "C": "(C) total wealth", "M": "(M) fewer bonds only"}
EQUITY_ASSET = "EQ_WORLD"        # (B) 100 - age: equities in world equities, the rest in the career's bond


# =============================================================================
# 1. Data
# =============================================================================

def load_backtest_data(start=None, end=None):
    """Monthly CHF returns of every saved series (core assets and sector menu) and RF_CHF."""
    from src.data import load_returns, load_rf, load_sector_returns
    returns = load_returns(start, end).join(load_sector_returns(start, end))
    return returns, load_rf(start, end)


# =============================================================================
# 2. The walk-forward engine
# =============================================================================

def run_backtest(returns, rf, strategy, params, path=None, initial_value=None):
    """Walk-forward backtest of one strategy.

    returns:  DataFrame of monthly CHF returns (total, not excess) of the assets, month-end dates
    rf:       Series of the monthly CHF risk-free rate, same dates
    strategy: function(window_excess, date, state) -> weights of the risky assets (columns of
              returns); the rest is cash. window_excess holds ONLY the window_months months before
              date; state = {"value": portfolio CHF before trading (savings included), "client":
              the row of path for this month or None}.
    path:     client_path (by month) or None. If given, its savings are added at the start of each
              month and the result gets the columns H, P and W (total wealth, start of month).
    initial_value: portfolio CHF at the start (default: params client portfolio_chf).

    Each month t, from the first month with a full window:
      1. the month's savings arrive as cash; 2. the strategy sets target weights from the past;
      3. trading from the drifted holdings costs transaction_cost_bp x turnover x value;
      4. during month t the assets earn their return and the cash earns rf.
    Returns a DataFrame by month: value_start (after savings, before trading), value (end of
    month), return (of the month, net of costs), turnover, cost, savings, cash and w_<asset>."""
    if not returns.index.equals(rf.index):
        raise ValueError("returns and rf must have the same dates")
    if returns.isna().any().any() or rf.isna().any():
        raise ValueError("the backtest data must have no missing values")
    window = int(params["backtest"]["window_months"])
    if len(returns) <= window:
        raise ValueError(f"need more than {window} months of data, got {len(returns)}")
    cost_rate = params["backtest"]["transaction_cost_bp"] / BP
    assets = list(returns.columns)
    excess = returns.sub(rf, axis=0)
    asset_returns, rates = returns.to_numpy(), rf.to_numpy()

    value = float(params["client"]["portfolio_chf"] if initial_value is None else initial_value)
    holdings, cash = np.zeros(len(assets)), value
    rows = []
    for t in range(window, len(returns)):
        date = returns.index[t]
        client = None if path is None else path.loc[date]
        savings = 0.0 if client is None else float(client["savings"])
        cash += savings
        start_value = holdings.sum() + cash
        if start_value <= 0:
            raise ValueError(f"the portfolio is worth {start_value:,.0f} CHF on {date:%Y-%m}")

        weights = np.asarray(strategy(excess.iloc[t - window:t], date,
                                      {"value": start_value, "client": client}), dtype=float)
        if weights.shape != (len(assets),) or not np.isfinite(weights).all():
            raise ValueError(f"the strategy gave invalid weights on {date:%Y-%m}: {weights}")

        turnover = float(np.abs(weights - holdings / start_value).sum())
        cost = cost_rate * turnover * start_value
        invested = start_value - cost
        holdings, cash = weights * invested, invested - weights.sum() * invested
        holdings = holdings * (1 + asset_returns[t])
        cash = cash * (1 + rates[t])
        end_value = holdings.sum() + cash
        row = {"date": date, "value_start": start_value, "value": end_value,
               "return": end_value / start_value - 1, "turnover": turnover, "cost": cost,
               "savings": savings, "cash": 1 - weights.sum()}
        row.update({f"w_{name}": w for name, w in zip(assets, weights)})
        rows.append(row)
    result = pd.DataFrame(rows).set_index("date")
    if path is not None:
        result["H"] = path.loc[result.index, "H_start"]
        result["P"] = path.loc[result.index, "P_start"]
        result["W"] = result["value_start"] + result["H"] + result["P"]
    return result


# =============================================================================
# 3. The client over time
# =============================================================================

def career_state(age, salary, params):
    """H, beta_bond, beta_sector at a fractional age: the step 2 model at the two birthdays
    around it, interpolated linearly. The model pays the salary once a year, so without the
    interpolation H and beta_bond would jump once a year inside the backtest."""
    if age >= params["client"]["retirement_age"]:
        return 0.0, 0.0, 0.0
    low = np.floor(age)
    share = age - low
    values = []
    for a in (low, low + 1):
        beta_bond, beta_sector, _ = career_betas(a, params)
        values.append((human_capital(a, salary, params), beta_bond, beta_sector))
    return tuple((1 - share) * v0 + share * v1 for v0, v1 in zip(*values))


def client_path(params, excess_returns, rf, start, beta_sector=None, job_loss_threshold=None):
    """The client month by month from `start`.

    excess_returns: monthly excess returns with the career's bond and sector assets (all the
                    data: the sector's peak for the job-loss rule can be older than start)
    beta_sector:    the TRUE beta_sector of the career (default: params). The decisions always use
                    the assumed one; the wrong-beta test replays the career with another one.
    job_loss_threshold: None = no job loss (main backtest), or e.g. 0.30 (job-loss variant)

    Month t (k months after start; everything at the start of the month is known to a decision):
      age = age_0 + k/12, salary = salary_0 (1 + salary_growth)^(k/12)
      H_start = model value (step 2) x level - unpaid salary while unemployed
      level: the model value revalued with the realised career returns,
             level x (1 + career excess - expected career excess) each month
      P: starts at the balance-sheet value; each month earns the BVG interest rate / 12 and
         receives the age credit / 12 while employed (no credits while unemployed)
      savings = savings_rate x salary / 12 while employed
    Career excess return of month t = beta_bond R_B + beta_sector R_S (excess returns, as in the
    optimizer); the pension's excess return = BVG rate / 12 - rf (no market risk).

    Job-loss variant: when the sector has fallen more than the threshold below its peak at the
    end of a month, the salary stops for job_loss_months months. H falls at once by the unpaid
    salary (a one-off negative career return), then recovers as those months pass. After the new
    job, the peak restarts from that month, so a later crisis can cost the job again."""
    p = copy.deepcopy(params)
    if beta_sector is not None:
        p["career"]["beta_sector"] = float(beta_sector)
    client, career, pension = p["client"], p["career"], p["pension"]
    bond, sector = p["data"]["career_bond_asset"], p["data"]["career_sector_asset"]
    dates = excess_returns.loc[start:].index
    savings_rate = p["backtest"]["savings_rate"]
    unemployed_months = int(p["backtest"]["job_loss_months"])
    has_pension = client.get("pillar2", True)
    rate = pension["interest_rate"]
    bands = sorted(pension["age_credits"].items())

    sector_index = (1 + excess_returns[sector]).cumprod() if sector is not None else None
    peak = float(sector_index.loc[:start].iloc[:-1].max()) if sector_index is not None and len(
        sector_index.loc[:start]) > 1 else 1.0

    level, gap, months_left = 1.0, 0.0, 0
    P = balance_sheet(p)["P"]
    rows = []
    for k, date in enumerate(dates):
        age = client["age"] + k / MONTHS
        salary = client["salary_chf"] * (1 + career["salary_growth"]) ** (k / MONTHS)
        working = age < client["retirement_age"]
        employed = working and months_left == 0
        H_model, beta_bond, beta_sector_ = career_state(age, salary, p)
        H_start = max(H_model * level - gap, 0.0)

        r_b = excess_returns.loc[date, bond]
        r_s = excess_returns.loc[date, sector] if sector is not None else 0.0
        sector_part = beta_sector_ * r_s
        career_excess = beta_bond * r_b + sector_part
        expected = (beta_bond * career["bond_premium"] + beta_sector_ * career["sector_premium"]) / MONTHS

        started = [r for a, r in bands if a <= age]                # the band of the current age
        credit_rate = started[-1] if started else 0.0
        coordinated = max(min(salary, pension["max_insured_salary"]) - pension["coordination_deduction"], 0.0)
        credit = credit_rate * coordinated / MONTHS if (has_pension and employed) else 0.0

        row = {"date": date, "age": age, "salary": salary, "employed": employed,
               "H_start": H_start, "P_start": P, "W_nontradable": H_start + P,
               "beta_bond": beta_bond, "beta_sector": beta_sector_,
               "savings": savings_rate * salary / MONTHS if employed else 0.0,
               "career_excess": career_excess, "career_sector_part": sector_part,
               "job_lost": False, "job_loss_return": 0.0,
               "rf": rf.loc[date], "pension_excess": rate / MONTHS - rf.loc[date]}

        # end of the month: the career is revalued, the pension grows
        if working:
            level *= 1 + career_excess - expected
        P = P * (1 + rate / MONTHS) + credit
        if months_left > 0:
            gap = max(gap - salary / MONTHS, 0.0)
            months_left -= 1
            if months_left == 0:
                gap = 0.0
                peak = float(sector_index.loc[date]) if sector_index is not None else peak
        elif working and job_loss_threshold is not None and sector_index is not None:
            peak = max(peak, float(sector_index.loc[date]))
            if sector_index.loc[date] / peak - 1 <= -job_loss_threshold:
                gap = salary / MONTHS * unemployed_months
                months_left = unemployed_months
                row["job_lost"] = True
                row["job_loss_return"] = -gap / H_start if H_start > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# 4. The strategies, month by month
# =============================================================================

def make_strategies(params, gamma, assets):
    """The four strategies as functions for run_backtest.

    (A), (D), (C) solve the total-wealth problem of the optimizer each month with the inputs of the
    past window (computed once per month and shared) and the client's balance sheet at the start
    of the month (H and P from the path, F = portfolio value):
        (A) omega = 0 (career ignored)   (D) sigma_H = 0 (no hedge)   (C) the full problem
    (B) puts (100 - age)% in world equities and the rest in CHF bonds.
    Every strategy holds the deferred shares (constant CHF amount: new grants replace the shares
    that vest, an assumption) and the same constraints."""
    deferred = float(params["client"].get("deferred_shares_chf") or 0.0)
    has_employer = EMPLOYER in assets
    inputs = {}

    def market(window_excess, date):
        if date not in inputs:
            cov, _ = estimate_covariance(window_excess[assets], params)
            mu, _ = expected_returns(cov, params)
            inputs[date] = (cov, mu.to_numpy())
        return inputs[date]

    def lockup(value):
        return min(deferred / value, 1.0) if has_employer and deferred > 0 else 0.0

    def optimizer(kind):
        def strategy(window_excess, date, state):
            cov, mu = market(window_excess, date)
            client, value = state["client"], state["value"]
            H, P = client["H_start"], client["P_start"]
            W = H + P + value
            omega, omega_H = (H + P) / W, H / W
            sigma_H = np.zeros(len(assets))
            if kind == "C" and omega > 0:
                sigma_H = omega_H / omega * career_covariance(cov, client["beta_bond"], client["beta_sector"],
                                                              params)
            constraints = build_constraints(assets, params, lockup(value))
            return solve_portfolio(mu, cov.to_numpy(), sigma_H, 0.0 if kind == "A" else omega, gamma,
                                   constraints)["x"]
        return strategy

    def hundred_minus_age(window_excess, date, state):
        x = np.zeros(len(assets))
        held = lockup(state["value"])
        equity = min(max((100 - state["client"]["age"]) / 100, 0.0), 1.0)
        x[assets.index(EQUITY_ASSET)] = equity * (1 - held)
        x[assets.index(params["data"]["career_bond_asset"])] = (1 - equity) * (1 - held)
        if held > 0:
            x[assets.index(EMPLOYER)] = held
        return x

    return {"A": optimizer("A"), "B": hundred_minus_age, "D": optimizer("D"), "C": optimizer("C")}


def run_strategies(params=None, data=None, strategies=("A", "B", "D", "C"), job_loss_threshold=None):
    """Backtest of the strategies for the client in params.

    data: (returns, rf) from load_backtest_data (default: the saved data).
    gamma is calibrated ONCE, on the first window only (no look-ahead), and shared by all.
    Returns a dict: results {name: run_backtest output}, path, gamma, excess (the client's assets)."""
    if params is None:
        params = load_params()
    returns, rf = load_backtest_data() if data is None else data
    assets = client_assets(params)
    excess_all = returns.sub(rf, axis=0)
    window = int(params["backtest"]["window_months"])
    start = returns.index[window]
    gamma, _ = risk_tolerance(params, excess_all.iloc[:window])
    path = client_path(params, excess_all, rf, start, job_loss_threshold=job_loss_threshold)
    functions = make_strategies(params, gamma, assets)
    results = {name: run_backtest(returns[assets], rf, functions[name], params, path) for name in strategies}
    return {"results": results, "path": path, "gamma": gamma, "excess": excess_all, "rf": rf, "params": params}


# =============================================================================
# 5. Metrics
# =============================================================================

def total_wealth_excess(result, path, reference=None):
    """Monthly excess return of total wealth: (F/W)(r - rf) + (H/W) career + (P/W) pension, with the
    shares of `reference` (default: the strategy itself). The career return is its market part
    plus, in the job-loss variant, the one-off fall of H when the job is lost."""
    ref = result if reference is None else reference
    dates = result.index
    F, H, P = ref["value_start"], path.loc[dates, "H_start"], path.loc[dates, "P_start"]
    W = F + H + P
    rf = path.loc[dates, "rf"]
    career = path.loc[dates, "career_excess"] + path.loc[dates, "job_loss_return"]
    return (F * (result["return"] - rf) + H * career + P * path.loc[dates, "pension_excess"]) / W


def certainty_equivalent_return(monthly_excess, gamma):
    """Annual certainty equivalent (lab convention, gamma = risk tolerance):
    CE = 12 x mean - 12 x variance / (2 gamma)."""
    r = pd.Series(monthly_excess).dropna()
    return MONTHS * r.mean() - MONTHS * r.var() / (2 * gamma)


def split_gain(excess_ref, excess_x, gamma):
    """CE gain of x over the reference, annual, split into its two parts:
    return part = 12 x (mean x - mean ref); risk part = -12 x (var x - var ref) / (2 gamma)."""
    a, b = pd.Series(excess_ref), pd.Series(excess_x)
    return MONTHS * (b.mean() - a.mean()), -MONTHS * (b.var() - a.var()) / (2 * gamma)


def max_drawdown(monthly_returns):
    """Largest fall from a peak of the cumulated returns (negative, 0 if none)."""
    index = (1 + pd.Series(monthly_returns)).cumprod()
    return float((index / index.cummax() - 1).min())


def newey_west_tstat(series, lags=None):
    """Mean, Newey-West standard error and t-stat of a monthly series (Bartlett weights;
    default lags = floor(4 (T/100)^(2/9))). Months are not independent: crises last several months."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = x - x.mean()
    long_run = d @ d / n
    for lag in range(1, lags + 1):
        long_run += 2 * (1 - lag / (lags + 1)) * (d[lag:] @ d[:-lag]) / n
    se = np.sqrt(long_run / n)
    return float(x.mean()), float(se), float(x.mean() / se) if se > 0 else np.nan


def block_bootstrap_gain(excess_ref, excess_x, gamma, block, draws, seed):
    """Circular block bootstrap of the annual CE gain of x over the reference: new 21-year
    histories are drawn from blocks of `block` consecutive months, the SAME months for both.
    Returns the gains of the bootstrapped histories (array)."""
    a, b = np.asarray(excess_ref, dtype=float), np.asarray(excess_x, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]

    def ce(sample):
        return MONTHS * sample.mean(axis=1) - MONTHS * sample.var(axis=1, ddof=1) / (2 * gamma)

    return ce(b[idx]) - ce(a[idx])


def gain_table(run, pairs=(("B", "A"), ("D", "A"), ("C", "A"), ("C", "D")), reference="A"):
    """Fair CE gains (same wealth shares, those of `reference`) in CHF per year on the reference's
    average total wealth, split into return and risk parts, with Newey-West t-stats and a block-
    bootstrap 95% interval and p-value (share of histories where the gain is <= 0)."""
    results, path, gamma, bt = run["results"], run["path"], run["gamma"], run["params"]["backtest"]
    ref = results[reference]
    W = ref["W"].mean()
    excess = {name: total_wealth_excess(r, path, ref) for name, r in results.items()}
    rows = []
    for x, y in pairs:
        if x not in results or y not in results:
            continue
        ret, risk = split_gain(excess[y], excess[x], gamma)
        boot = block_bootstrap_gain(excess[y], excess[x], gamma, bt["bootstrap_block"], bt["bootstrap_draws"],
                                    bt["bootstrap_seed"])
        low, high = np.percentile(boot, [2.5, 97.5])
        rows.append({"comparison": f"({x}) vs ({y})", "CE gain (CHF/yr)": (ret + risk) * W,
                     "return part (CHF/yr)": ret * W, "risk part (CHF/yr)": risk * W,
                     "95% low (CHF/yr)": low * W, "95% high (CHF/yr)": high * W,
                     "p-value": float((boot <= 0).mean()),
                     "t-stat total wealth": newey_west_tstat(excess[x] - excess[y])[2],
                     "t-stat portfolio": newey_west_tstat(results[x]["return"] - results[y]["return"])[2]})
    return pd.DataFrame(rows).set_index("comparison")


def metrics_table(run, reference="A"):
    """One row per strategy. Total wealth and the CHF metrics use the
    reference's wealth (fair comparison); the own-path columns describe what the strategy became."""
    results, path, gamma, bt = run["results"], run["path"], run["gamma"], run["params"]["backtest"]
    ref = results[reference]
    dates = ref.index
    rf = path.loc[dates, "rf"]
    sector = path.loc[dates, "career_sector_part"]
    has_sector = sector.abs().sum() > 0
    bad = sector.nsmallest(max(1, int(round(bt["worst_months_share"] * len(sector))))).index
    W = ref["W"].mean()
    rows = {}
    for name, r in results.items():
        excess_W = total_wealth_excess(r, path, ref)
        excess_F = r["return"] - rf
        rows[STRATEGIES[name]] = {
            "CE total wealth (annual)": certainty_equivalent_return(excess_W, gamma),
            "CE total wealth (CHF/yr)": certainty_equivalent_return(excess_W, gamma) * W,
            "total wealth volatility": excess_W.std() * np.sqrt(MONTHS),
            "portfolio excess return (annual)": excess_F.mean() * MONTHS,
            "portfolio volatility": excess_F.std() * np.sqrt(MONTHS),
            "portfolio max drawdown": max_drawdown(r["return"]),
            "correlation with the sector part": excess_F.corr(sector) if has_sector else np.nan,
            "CHF in the sector's worst 10% months": float((ref["value_start"] * r["return"]).loc[bad].mean())
            if has_sector else np.nan,
            "average share outside bonds and cash": float(r[[c for c in r.columns if c.startswith("w_")
                                                             and not c.startswith("w_BOND")]].sum(axis=1).mean()),
            "turnover per year": r["turnover"].mean() * MONTHS,
            "costs (CHF, total)": r["cost"].sum(),
            "tracking error vs (A)": (r["return"] - results["A"]["return"]).std() * np.sqrt(MONTHS)
            if "A" in results else np.nan,
            "final portfolio (own path, CHF)": r["value"].iloc[-1]}
    return pd.DataFrame(rows)


def crisis_table(run, reference="A"):
    """CHF result of each strategy in each crisis (computed, never assumed):
    the portfolio's return over the crisis, its CHF result on the reference's portfolio and the
    CHF change of total wealth (reference's shares, market part only)."""
    results, path, bt = run["results"], run["path"], run["params"]["backtest"]
    ref = results[reference]
    rows = []
    for crisis, (first, last) in bt["crises"].items():
        months = ref.loc[first:last].index
        if len(months) == 0 or pd.Timestamp(last) > ref.index[-1] + pd.offsets.MonthEnd(0):
            continue
        for name, r in results.items():
            excess_W = total_wealth_excess(r, path, ref).loc[months]
            rows.append({"crisis": crisis, "strategy": name,
                         "portfolio return": float((1 + r.loc[months, "return"]).prod() - 1),
                         "portfolio CHF": float((ref.loc[months, "value_start"] * r.loc[months, "return"]).sum()),
                         "total wealth CHF": float((ref.loc[months, "W"] * excess_W).sum())})
    return pd.DataFrame(rows).set_index(["crisis", "strategy"])


# =============================================================================
# 6. Robustness
# =============================================================================

def wrong_beta_curve(params=None, data=None, run=None):
    """Is the hedge right, or only consistent with itself?

    The decisions are made with the ASSUMED beta_sector, as in reality. The career is then replayed
    with a TRUE beta_sector = multiplier x assumed, and the fair gain of (C) over (D) (same shares,
    those of (D)) is measured, with its return and risk parts and a bootstrap p-value.

    Theory: a hedge h against a true exposure beta saves variance
    beta^2 - (beta - h)^2 = h (2 beta - h): a straight line in beta, zero at beta = h / 2. With the
    decisions fixed, the risk part is linear in the true beta here too (the career's sector part
    enters total wealth linearly); the slope and the break-even are reported."""
    if params is None:
        params = load_params()
    if run is None:
        run = run_strategies(params, data, strategies=("D", "C"))
    results, gamma, bt = run["results"], run["gamma"], params["backtest"]
    c, d = results["C"], results["D"]
    assumed = params["career"]["beta_sector"]
    rows = []
    for m in bt["wrong_beta_multipliers"]:
        path = client_path(params, run["excess"], run["rf"], d.index[0], beta_sector=m * assumed)
        ref = d.copy()
        ref["W"] = ref["value_start"] + path.loc[ref.index, "H_start"] + path.loc[ref.index, "P_start"]
        e_d, e_c = total_wealth_excess(d, path, ref), total_wealth_excess(c, path, ref)
        ret, risk = split_gain(e_d, e_c, gamma)
        boot = block_bootstrap_gain(e_d, e_c, gamma, bt["bootstrap_block"], bt["bootstrap_draws"], bt["bootstrap_seed"])
        W = ref["W"].mean()
        rows.append({"multiplier": m, "true beta_sector": m * assumed, "CE gain (C) vs (D) (CHF/yr)": (ret + risk) * W,
                     "return part (CHF/yr)": ret * W, "risk part (CHF/yr)": risk * W,
                     "p-value": float((boot <= 0).mean())})
    table = pd.DataFrame(rows).set_index("multiplier")
    slope, intercept = np.polyfit(table["true beta_sector"], table["risk part (CHF/yr)"], 1)
    table.attrs["risk part: break-even true beta"] = -intercept / slope if slope != 0 else np.nan
    table.attrs["risk part: R2 of the straight line"] = float(np.corrcoef(
        table["true beta_sector"], table["risk part (CHF/yr)"])[0, 1] ** 2) if len(table) > 2 else np.nan
    return table


def job_loss_table(params=None, data=None):
    """The backtest with the job-loss variant: thresholds from params
    (20%, 30%, 40%) and none. For each: when the job was lost, the fair gains, and the CHF result
    of each portfolio from each job loss to the end of the unemployment."""
    if params is None:
        params = load_params()
    rows = []
    for threshold in [None] + list(params["backtest"]["job_loss_thresholds"]):
        run = run_strategies(params, data, strategies=("A", "D", "C"), job_loss_threshold=threshold)
        gains = gain_table(run, pairs=(("D", "A"), ("C", "A"), ("C", "D")))
        path = run["path"]
        row = {"threshold": "no job loss" if threshold is None else f"{threshold:.0%}",
               "jobs lost": ", ".join(f"{d:%Y-%m}" for d in path.index[path["job_lost"]]) or "-"}
        row.update(_gain_columns(gains))
        # Did the portfolio help when the job was lost? Its CHF result (on the wealth of (A)) from the
        # month of each job loss to the end of the unemployment, all losses together.
        months = sorted({d for start in path.index[path["job_lost"]]
                         for d in path.loc[start:].index[:int(run["params"]["backtest"]["job_loss_months"]) + 1]})
        ref = run["results"]["A"]
        for name in ("A", "D", "C"):
            r = run["results"][name]
            row[f"portfolio CHF around the job losses ({name})"] = float(
                (ref.loc[months, "value_start"] * r.loc[months, "return"]).sum()) if months else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def prior_sensitivity(params=None, data=None):
    """Does the result depend on the Black-Litterman prior? The global market portfolio (default)
    against a Swiss pension-fund portfolio (params backtest swiss_prior, rescaled to 100%)."""
    if params is None:
        params = load_params()
    swiss = params["backtest"]["swiss_prior"]
    total = sum(swiss.values())
    priors = {"global market portfolio": params["black_litterman"]["market_weights"],
              "Swiss pension fund (LPP-40)": {k: v / total for k, v in swiss.items()}}
    sector = params["data"]["career_sector_asset"]
    rows = []
    for label, weights in priors.items():
        p = copy.deepcopy(params)
        p["black_litterman"]["market_weights"] = weights
        run = run_strategies(p, data, strategies=("A", "D", "C"))
        gains = gain_table(run, pairs=(("D", "A"), ("C", "A"), ("C", "D")))
        res = run["results"]
        row = {"prior": label, "gamma": run["gamma"]}
        if sector is not None:
            for name in ("D", "C"):
                row[f"own sector in ({name}), average"] = res[name][f"w_{sector}"].mean()
        row.update(_gain_columns(gains))
        rows.append(row)
    return pd.DataFrame(rows).set_index("prior")


def hedge_map(params=None, data=None):
    """When does the hedge matter? The hedge (C) vs (D) for
    other clients: starting portfolio x beta_sector (params backtest hedge_map_*); here the adviser
    knows the true beta. For each: the fair gain (shares of (D)) split into return and risk parts,
    the p-value, total-wealth volatility of (D) and (C), and how much less of the sector (C) holds."""
    if params is None:
        params = load_params()
    bt = params["backtest"]
    sector = params["data"]["career_sector_asset"]
    rows = []
    for portfolio in bt["hedge_map_portfolios"]:
        for beta in bt["hedge_map_betas"]:
            p = copy.deepcopy(params)
            p["client"]["portfolio_chf"] = portfolio
            p["career"]["beta_sector"] = beta
            run = run_strategies(p, data, strategies=("D", "C"))
            gains = gain_table(run, pairs=(("C", "D"),), reference="D")
            res, path = run["results"], run["path"]
            e_d = total_wealth_excess(res["D"], path, res["D"])
            e_c = total_wealth_excess(res["C"], path, res["D"])
            rows.append({"start portfolio (CHF)": portfolio, "beta_sector": beta,
                         "portfolio / total wealth (average)": float((res["D"]["value_start"] / res["D"]["W"]).mean()),
                         "hedge gain (CHF/yr)": gains.loc["(C) vs (D)", "CE gain (CHF/yr)"],
                         "return part (CHF/yr)": gains.loc["(C) vs (D)", "return part (CHF/yr)"],
                         "risk part (CHF/yr)": gains.loc["(C) vs (D)", "risk part (CHF/yr)"],
                         "p-value": gains.loc["(C) vs (D)", "p-value"],
                         "total wealth vol (D)": e_d.std() * np.sqrt(MONTHS),
                         "total wealth vol (C)": e_c.std() * np.sqrt(MONTHS),
                         "sector: (C) - (D), average": float((res["C"][f"w_{sector}"] - res["D"][f"w_{sector}"]).mean())
                         if sector else 0.0})
    return pd.DataFrame(rows).set_index(["start portfolio (CHF)", "beta_sector"])


def personas_backtest(params=None, data=None):
    """The main backtest for every persona: fair gains (D) vs (A), (C) vs (A), (C) vs (D)."""
    if params is None:
        params = load_params()
    rows = []
    for name, persona in params["personas"].items():
        run = run_strategies(params_for_persona(params, name), data, strategies=("A", "D", "C"))
        gains = gain_table(run, pairs=(("D", "A"), ("C", "A"), ("C", "D")))
        row = {"persona": persona["label"]}
        row.update(_gain_columns(gains))
        rows.append(row)
    return pd.DataFrame(rows).set_index("persona")


def _gain_columns(gains):
    """The columns shown by the robustness tables: each gain with its p-value, and the hedge
    ((C) vs (D)) split into its return and risk parts."""
    row = {}
    for comparison in gains.index:
        row[f"{comparison} CHF/yr"] = gains.loc[comparison, "CE gain (CHF/yr)"]
        row[f"{comparison} p-value"] = gains.loc[comparison, "p-value"]
    if "(C) vs (D)" in gains.index:
        row["hedge: return part"] = gains.loc["(C) vs (D)", "return part (CHF/yr)"]
        row["hedge: risk part"] = gains.loc["(C) vs (D)", "risk part (CHF/yr)"]
    return row


# =============================================================================
# 7. Report
# =============================================================================

def print_report(params=None):
    """Main backtest of the client in params, then the robustness tests."""
    if params is None:
        params = load_params()
    data = load_backtest_data()
    run = run_strategies(params, data)
    path, results = run["path"], run["results"]
    first, last = path.index[0], path.index[-1]
    client = params["client"]
    chf = "{:,.0f}".format
    print(f"Client: {client.get('persona', 'custom')}, {client['age']} years old in {first:%b %Y}, salary CHF "
          f"{chf(client['salary_chf'])}, portfolio CHF {chf(client['portfolio_chf'])}")
    print(f"Backtest: {len(path)} months, {first:%Y-%m} to {last:%Y-%m}; gamma = {run['gamma']:.4f} "
          f"(first window only)")
    print(f"Client at the end: age {path['age'].iloc[-1]:.1f}, H = CHF {chf(path['H_start'].iloc[-1])}, "
          f"P = CHF {chf(path['P_start'].iloc[-1])}, savings CHF {chf(path['savings'].sum())} in total")
    print("\nMetrics (total wealth and CHF columns with the wealth of (A): a fair comparison):")
    print(metrics_table(run).to_string(float_format="{:,.4f}".format))
    print("\nGains in certainty equivalent of total wealth (CHF per year):")
    print(gain_table(run).to_string(float_format="{:,.2f}".format))
    crises = crisis_table(run)
    order = [c for c in params["backtest"]["crises"] if c in crises.index.get_level_values(0)]
    print("\nCrises: portfolio return")
    print(crises["portfolio return"].unstack("strategy").loc[order].to_string(float_format="{:.1%}".format))
    print("Crises: CHF result of the portfolio and of total wealth (wealth of (A))")
    print(crises[["portfolio CHF", "total wealth CHF"]].unstack("strategy").loc[order].to_string(
        float_format="{:,.0f}".format))

    print("\nWrong beta: decisions with the assumed beta_sector, career replayed with the true one:")
    curve = wrong_beta_curve(params, run=run_strategies(params, data, strategies=("D", "C")))
    print(curve.to_string(float_format="{:,.2f}".format))
    print(f"  risk part: straight line R2 = {curve.attrs['risk part: R2 of the straight line']:.4f}, "
          f"break-even true beta = {curve.attrs['risk part: break-even true beta']:.3f}")
    print("\nJob loss:")
    print(job_loss_table(params, data).T.to_string(float_format="{:,.2f}".format))
    print("\nPrior:")
    print(prior_sensitivity(params, data).T.to_string(float_format="{:,.3f}".format))
    print("\nHedge map (starting portfolio x beta_sector):")
    print(hedge_map(params, data).to_string(float_format="{:,.3f}".format))
    print("\nAll personas:")
    print(personas_backtest(params, data).to_string(float_format="{:,.2f}".format))


if __name__ == "__main__":
    print_report()
