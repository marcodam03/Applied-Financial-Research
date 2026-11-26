"""Numbers for the app (step 5), computed with the modules of steps 1 to 4.

No Streamlit here, so everything can be tested (tests/test_app.py). app.py only lays out
what these functions return.
"""
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest as B
from src import career as C
from src import optimizer as O
from src.career import params_for_persona
from src.config import load_params

MONTHS = 12
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
GROUPS = ("Shares", "Bonds", "Real estate and gold", "Cash")


# =============================================================================
# 1. The client chosen in the app
# =============================================================================

def client_params(persona, changes=None, base=None):
    """params for a persona, with the changes made in the app's side panel.

    changes (all optional): age, salary_chf, portfolio_chf, deferred_shares_chf,
      pension_chf (None or "auto" = legal estimate), beta_sector, wage_rate_link,
      long_only (bool), no_leverage (bool), sector_cap (share), view (dict asset, q, sd; annual)."""
    params = params_for_persona(copy.deepcopy(base or load_params()), persona)
    changes = changes or {}
    client, career, opt = params["client"], params["career"], params["optimization"]
    for key in ("age", "salary_chf", "portfolio_chf", "deferred_shares_chf"):
        if changes.get(key) is not None:
            client[key] = changes[key]
    if "pension_chf" in changes:
        client["pension_chf"] = "auto" if changes["pension_chf"] in (None, "auto") else float(changes["pension_chf"])
    if changes.get("beta_sector") is not None and params["data"]["career_sector_asset"] is not None:
        career["beta_sector"] = float(changes["beta_sector"])
    if changes.get("wage_rate_link") is not None:
        career["wage_rate_link"] = float(changes["wage_rate_link"])
    if changes.get("long_only") is not None:
        opt["long_only"] = bool(changes["long_only"])
    if changes.get("no_leverage") is not None:
        opt["max_leverage"] = 1.0 if changes["no_leverage"] else None
    if changes.get("sector_cap") is not None:
        opt["position_caps"] = {k: float(changes["sector_cap"]) for k in opt["position_caps"]}
    view = changes.get("view")
    if view:
        params["black_litterman"]["views"] = [{"assets": {view["asset"]: 1.0}, "q": float(view["q"]),
                                               "sd": float(view["sd"])}]
    if client["deferred_shares_chf"] > client["portfolio_chf"]:
        client["deferred_shares_chf"] = client["portfolio_chf"]
    return params


# =============================================================================
# 2. Today: the optimizer
# =============================================================================

def analysis(params, excess_returns=None):
    """analyze_client (step 3) plus the numbers the pages show."""
    result = O.analyze_client(params, excess_returns)
    weights = result["weights"]
    perf = result["performance"]
    W = result["balance_sheet"]["W"]
    sector = params["data"]["career_sector_asset"]
    bonds = [a for a in result["assets"] if a.startswith("BOND")]
    result["story"] = {
        "risky_A": 1 - weights.loc[bonds + [O.CASH], "A"].sum(),
        "risky_D": 1 - weights.loc[bonds + [O.CASH], "D"].sum(),
        "bonds_A": weights.loc[bonds, "A"].sum(), "bonds_D": weights.loc[bonds, "D"].sum(),
        "sector_D": weights.loc[sector, "D"] if sector else 0.0,
        "sector_C": weights.loc[sector, "C"] if sector else 0.0,
        "gain_D_A": perf.loc["certainty equivalent, CHF per year", "D"] - perf.loc["certainty equivalent, CHF per year", "A"],
        "gain_C_D": perf.loc["certainty equivalent, CHF per year", "C"] - perf.loc["certainty equivalent, CHF per year", "D"],
        "career_risk_share": result["risk"]["C"].loc[O.CAREER, "share of risk"],
        "portfolio_share": result["balance_sheet"]["F"] / W,
    }
    return result


def demand_chf(result):
    """Unconstrained demands in CHF (share of F x F): speculative, hedging, total."""
    F = result["balance_sheet"]["F"]
    return result["demand"][["speculative", "hedging", "total"]] * F


def lifecycle(params, result):
    """Expected balance sheet from today to retirement, year by year: the career value H, the
    pension P and the portfolio F, with no market surprise (every asset earns its expected return).
    H and P follow the backtest's client path; F grows at the CHF long-run rate plus the expected
    excess return of portfolio (C) today, and receives the savings."""
    client, career = params["client"], params["career"]
    years = int(np.ceil(client["retirement_age"] - client["age"]))
    if years <= 0:
        sheet = result["balance_sheet"]
        return pd.DataFrame({"H": [0.0], "P": [sheet["P"]], "F": [sheet["F"]]}, index=[client["age"]])
    dates = pd.date_range("2000-01-31", periods=years * MONTHS + 1, freq="ME")
    bond, sector = params["data"]["career_bond_asset"], params["data"]["career_sector_asset"]
    excess = pd.DataFrame({bond: career["bond_premium"] / MONTHS}, index=dates)
    if sector is not None:
        excess[sector] = career["sector_premium"] / MONTHS
    rf = pd.Series(career["rf_long_run"] / MONTHS, index=dates)
    path = B.client_path(params, excess, rf, dates[0])
    x = result["solutions"]["C"]["x"]
    mu = result["expected_returns"]["Black-Litterman"].to_numpy() / MONTHS
    growth = career["rf_long_run"] / MONTHS + x @ mu
    F, values = float(client["portfolio_chf"]), []
    for k in range(len(dates)):
        values.append(F)
        F = F * (1 + growth) + path["savings"].iloc[k]
    path["F"] = values
    yearly = path.iloc[::MONTHS]
    return pd.DataFrame({"H": yearly["H_start"].to_numpy(), "P": yearly["P_start"].to_numpy(),
                         "F": yearly["F"].to_numpy(), "salary": yearly["salary"].to_numpy()},
                        index=np.round(yearly["age"].to_numpy(), 2))


def group_of(asset):
    """The four families the client page talks about."""
    if asset.startswith("BOND"):
        return "Bonds"
    if asset in ("REIT_CH", "GOLD"):
        return "Real estate and gold"
    if asset == O.CASH:
        return "Cash"
    return "Shares"


def groups(weights):
    """Weights (a Series by asset) summed by family, in the order of GROUPS."""
    total = weights.groupby([group_of(a) for a in weights.index]).sum()
    return total.reindex(GROUPS, fill_value=0.0)


def career_years(params):
    """The career year by year: the salary of each remaining year and what it is worth today.
    The values today add up to H (step 2): salary / (1 + discount rate)^years."""
    client = params["client"]
    times, flows = C.salary_cash_flows(client["age"], client["salary_chf"], params)
    rate = C.career_betas(client["age"], params)[2]
    ages = client["age"] + np.asarray(times) - 1
    return pd.DataFrame({"salary": flows, "value today": flows / (1 + rate) ** np.asarray(times)},
                        index=ages), rate


def glide_path(params, result, excess_returns=None):
    """The plan by age: the optimizer solved again every few years until retirement, with the
    balance sheet expected at that age (lifecycle: salary grown, pension credited, portfolio
    grown with the expected return and the savings). Locked shares are assumed sold after today.
    Returns the share of each family for our advice (C) and for standard advice (A)."""
    client = params["client"]
    step = params["app"]["glide_step_years"]
    lc = lifecycle(params, result)
    ages = [a for a in lc.index if a < client["retirement_age"]][::step]
    rows = {}
    for age in ages:
        if age == lc.index[0]:
            res = result
        else:
            p = copy.deepcopy(params)
            p["client"].update(age=float(age), salary_chf=float(lc.loc[age, "salary"]),
                               portfolio_chf=float(lc.loc[age, "F"]), deferred_shares_chf=0.0)
            if client.get("pillar2", True):
                p["client"]["pension_chf"] = float(lc.loc[age, "P"])
            res = O.analyze_client(p, excess_returns)
        sheet = res["balance_sheet"]
        row = {("C", g): v for g, v in groups(res["weights"]["C"]).items()}
        row.update({("A", g): v for g, v in groups(res["weights"]["A"]).items()})
        row[("", "portfolio share of wealth")] = sheet["F"] / sheet["W"]
        row[("", "portfolio")] = sheet["F"]
        rows[age] = row
    table = pd.DataFrame(rows).T
    table.index.name = "age"
    return table


# =============================================================================
# 3. The past: the backtest
# =============================================================================

def backtest(params, data=None):
    """Main backtest (step 4) of the client: values, gains, metrics, crises, client path."""
    run = B.run_strategies(params, data)
    return {"values": pd.DataFrame({k: r["value"] for k, r in run["results"].items()}),
            "weights": {k: r[[c for c in r.columns if c.startswith("w_")]].rename(columns=lambda c: c[2:])
                        for k, r in run["results"].items()},
            "path": run["path"][["age", "H_start", "P_start", "employed"]],
            "gains": B.gain_table(run), "metrics": B.metrics_table(run), "crises": B.crisis_table(run),
            "gamma": run["gamma"], "savings": float(run["path"]["savings"].sum())}


def prior_check(params, data=None):
    """The backtest with the global prior and with the Swiss pension-fund prior (step 4)."""
    return B.prior_sensitivity(params, data)


def robustness(params, data=None):
    """All the robustness tests of step 4 for this client (about half a minute). The tests of the hedge
    (wrong beta, job loss when the sector falls, hedge map) need a sector the salary follows: without
    one (civil servant, or a sector beta of 0) they are None."""
    tests = {"prior": B.prior_sensitivity(params, data), "wrong_beta": None, "job_loss": None, "hedge_map": None}
    if params["data"]["career_sector_asset"] is not None and params["career"]["beta_sector"] > 0:
        tests.update(wrong_beta=B.wrong_beta_curve(params, data), job_loss=B.job_loss_table(params, data),
                     hedge_map=B.hedge_map(params, data))
    return tests


# =============================================================================
# 4. Results kept on disk (the backtest takes seconds: computed once per client)
# =============================================================================

def _plain(value):
    """params with every number as a float, so 130000 and 130000.0 give the same key."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 10)
    return str(value)


def _signature():
    """Changes when the code of the model or the saved data change (then results are recomputed)."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").glob("*.py")):
        if f.name not in ("theme.py", "jobs.py", "precompute.py"):
            h.update(f.name.encode() + f.read_bytes())
    for f in sorted((ROOT / "data").glob("*.csv")):
        h.update(f"{f.name}{f.stat().st_size}{f.stat().st_mtime_ns}".encode())
    return h.hexdigest()


def cache_key(name, params):
    """File name of a saved result: what was computed, for which client (the app section of params
    only changes the pages, so it is left out), with which code and data."""
    plain = _plain({k: v for k, v in params.items() if k != "app"})
    text = json.dumps([name, plain], sort_keys=True) + _signature()
    return f"{name}_{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def saved(name, params):
    """The saved result, or None if it was never computed."""
    path = CACHE_DIR / (cache_key(name, params) + ".pkl")
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:                       # a file cut by an interrupted run: compute it again
        return None


def _backtest2(params, data=None):
    from src import allocation as A
    return A.backtest_summary(params, data)


def _story(params, data=None):
    from src import allocation as A
    return A.backtest_story(params, data)


def _load2():
    from src import allocation as A
    return A.load_data()


COMPUTE = {"backtest": backtest, "prior": prior_check, "robustness": robustness, "backtest2": _backtest2, "story": _story}
LOADERS = {"backtest2": _load2, "story": _load2}


def remember(name, params, data=None):
    """Saved result of COMPUTE[name] for this client; computed and saved the first time."""
    result = saved(name, params)
    if result is None:
        if data is None:
            data = LOADERS.get(name, B.load_backtest_data)()
        result = COMPUTE[name](params, data)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / (cache_key(name, params) + ".pkl")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(result, f)
        os.replace(tmp, path)               # never a half-written file, even with two runs at once
    return result
