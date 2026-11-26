"""
Backtest one signal from src/signals/: portfolio sorts, the long-short spread
before and after trading costs, factor-model alphas, the GRS test, and the same
spread in other universes (robustness). Every setting comes from the config.

Run from the project folder:
    python scripts/run_signal.py --signal momentum
    python scripts/run_signal.py --signal momentum --config configs/my_test.yaml
    python scripts/run_signal.py --signal momentum --holdout

Holdout discipline:
  - The team's holdout starts in HOLDOUT_START (below), whatever the config says.
    Without --holdout, every period is cut at the month before it, so the holdout
    years never appear and nobody can tune on them by accident, even with a config
    whose periods run later.
  - --holdout adds the test period of the config. Run it ONCE per signal, after the
    signal and the config are frozen. Every holdout run is recorded in
    holdout_log.txt in the project folder (in git, so the whole team sees every use)
    and the script warns you if the holdout of that signal was already used.

Outputs in results/signals/<signal>/ (with --holdout: results/signals/<signal>/holdout/),
<w> = ew or vw, <sample> = insample (train + validation) or test:
    returns_<w>.csv           monthly portfolio returns, gross spread and net spread
                              (net of half the bid-ask spread and the price impact)
    turnover_<w>.csv          monthly one-way turnover of each portfolio
    costs_<w>.csv             monthly trading cost of each portfolio, as a return
                              (costs_spread_<w>.csv, costs_impact_<w>.csv: the two parts)
    portfolios_<w>.csv        average return, std and t-stat of each portfolio (in sample)
    spread.csv                spread by period: average, t, Sharpe, before and after costs
    alphas_<sample>_<w>.csv   alpha (% per month) and Newey-West t of every portfolio
    grs_<sample>_<w>.csv      GRS test: are all portfolio alphas jointly zero?
    robustness.csv            the spread and its alphas in the robustness universes
    n_stocks.csv, breakpoints.csv, run_info.json
"""
import argparse
import getpass
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # lets Python find the "src" folder

from src.data.panel import load_spreads  # noqa: E402
from src.evaluation.alpha import MODELS, alpha_table, grs_test, load_factors, regress  # noqa: E402
from src.evaluation.performance import spread_stats  # noqa: E402
from src.portfolio.costs import aum_path, trading_cost, turnover, weight_changes  # noqa: E402
from src.portfolio.sorts import FINANCIALS, UTILITIES, sort_portfolios, summarize  # noqa: E402
from src.signals import build_panel, load_signal  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


PORTFOLIO_KEYS = {"buffer", "nyse_breakpoints", "hold_months"}  # robustness keys of `portfolio`
# robustness keys that belong to `costs` (e.g. a variant with spread costs only)
COST_KEYS = {"price_impact", "aum_musd", "aum_ref_month", "aum_scale", "impact_shape",
             "execution_days"}
COST_DEFAULTS = {"price_impact": False, "aum_musd": 100, "aum_ref_month": "2025-12",
                 "aum_scale": "volume", "impact_shape": "linear", "execution_days": 1}


def cost_settings(panel: pd.DataFrame, spreads: pd.DataFrame, costs_cfg: dict) -> tuple:
    """The `impact` argument of trading_cost (None = spread only) and a label."""
    c = {**COST_DEFAULTS, **(costs_cfg or {})}
    if not c["price_impact"]:
        return None, "half bid-ask spread"
    aum = aum_path(spreads, panel, c["aum_musd"] * 1e6, c["aum_ref_month"], c["aum_scale"])
    impact = {"aum": aum, "shape": c["impact_shape"], "execution_days": c["execution_days"]}
    return impact, (f"half bid-ask spread + {c['impact_shape']} price impact, fund of "
                    f"${c['aum_musd']}m per leg in {c['aum_ref_month']} (scaled with "
                    f"{c['aum_scale'].replace('_', ' ')}), {c['execution_days']} day(s) to trade")

# First month of the team's holdout, fixed here and not in the configs, so that no
# config can show it by accident. Only --holdout runs see it, and they are logged.
HOLDOUT_START = "2016-01"
LAST_OPEN_MONTH = str(pd.Period(HOLDOUT_START, freq="M") - 1)  # "2015-12"
HOLDOUT_LOG = BASE_DIR / "holdout_log.txt"


def sort_options(universe: dict, portfolio: dict) -> dict:
    """Translate the config into the arguments of sort_portfolios."""
    exclude = (FINANCIALS if universe.get("exclude_financials") else []) + (
        UTILITIES if universe.get("exclude_utilities") else []
    )
    # turnover buffer (sorts.py); not possible with fewer than 2*buffer+1 portfolios
    buffer = portfolio.get("buffer") or 0
    if buffer >= portfolio["n_portfolios"] / 2:
        buffer = 0
    return {
        "n_portfolios": portfolio["n_portfolios"],
        "breakpoint_exchanges": ("N",) if portfolio["nyse_breakpoints"] else ("N", "A", "Q"),
        "min_price": universe.get("min_price"),
        "min_nyse_size_pct": universe.get("min_nyse_size_pct"),
        "exclude_sic": exclude or None,
        "buffer": buffer,
        "hold_months": portfolio.get("hold_months") or 1,
    }


def by_formation_month(breakpoints: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    """Index the breakpoints by formation month (June of year t for annual sorts)."""
    if rebalance == "annual":
        idx = pd.PeriodIndex([f"{y}-06" for y in breakpoints.index], freq="M")
    else:  # form_key = year * 12 + month - 1
        idx = pd.PeriodIndex([f"{k // 12}-{k % 12 + 1:02d}" for k in breakpoints.index], freq="M")
    return breakpoints.set_axis(idx).rename_axis("formation")


def backtest(panel, signal, spec, opts, weightings, spreads, start, end, impact=None) -> dict:
    """Sort the stocks, then for each weighting: portfolio returns, turnover, trading
    costs, and the gross and net long-short spread. Everything is cut to the months
    start..end."""
    res = sort_portfolios(panel, signal, rebalance=spec.REBALANCE, return_holdings=True, **opts)
    spread = f"{spec.LONG}-{spec.SHORT}"
    out = {
        "n_stocks": res["n_stocks"].loc[start:end],
        "breakpoints": by_formation_month(res["breakpoints"], spec.REBALANCE).loc[:end],
    }
    for w in weightings:
        rets = res[w].loc[start:end].copy()
        # trades need the month before `start`, so they are computed on all months first
        changes = weight_changes(res["holdings"], w)
        to = turnover(changes).reindex(index=rets.index, columns=rets.columns)
        parts = {k: v.reindex(index=rets.index, columns=rets.columns)
                 for k, v in trading_cost(changes, spreads, impact).items()}
        cost = parts["total"]
        rets[spread] = rets[spec.LONG] - rets[spec.SHORT]
        rets[f"{spread} net"] = rets[spread] - cost[spec.LONG] - cost[spec.SHORT]
        out[w] = rets
        out[f"turnover_{w}"] = to
        out[f"costs_{w}"] = cost
        out[f"costs_spread_{w}"] = parts["spread"]
        out[f"costs_impact_{w}"] = parts["impact"]
    return out


def cut_at_holdout(valid: tuple) -> tuple:
    """Without --holdout: end the validation period (and so every period of the
    run) at LAST_OPEN_MONTH, whatever the config says."""
    if valid[1] <= LAST_OPEN_MONTH:
        return valid
    if valid[0] > LAST_OPEN_MONTH:
        sys.exit(f"The validation period starts after {LAST_OPEN_MONTH}, but the months from "
                 f"{HOLDOUT_START} on are the team's holdout: only --holdout shows them.")
    print("!" * 78)
    print(f"!! The config's periods run to {valid[1]}, but the months from {HOLDOUT_START} on are")
    print(f"!! the team's holdout. Every period is cut at {LAST_OPEN_MONTH}. To see the")
    print("!! holdout (once, when everything is frozen), rerun with --holdout.")
    print("!" * 78)
    return valid[0], LAST_OPEN_MONTH


def log_holdout(signal: str, config_path: Path) -> None:
    """Record every use of the holdout in HOLDOUT_LOG, and warn if the holdout of
    this signal was used before."""
    lines = HOLDOUT_LOG.read_text().splitlines() if HOLDOUT_LOG.exists() else []
    previous = [line for line in lines if f"signal={signal}" in line.split()]
    if previous:
        print("!" * 78)
        print(f"!! The holdout of this signal was already used {len(previous)} time(s):")
        for line in previous:
            print(f"!!   {line}")
        print("!! Anything changed after looking at it is no longer out of sample.")
        print("!" * 78)
    with HOLDOUT_LOG.open("a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M}  signal={signal}  "
                f"config={config_path.name}  user={getpass.getuser()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest one signal from src/signals/.")
    parser.add_argument("--signal", required=True, help="file name in src/signals/, e.g. momentum")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--holdout", action="store_true", help="also evaluate the test period (once!)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    cfg = yaml.safe_load(config_path.read_text())
    name = args.signal
    spec = load_signal(name)

    # Samples
    train, valid, test = (tuple(cfg["periods"][k]) for k in ("train", "validation", "test"))
    if not args.holdout:
        valid = cut_at_holdout(valid)
    in_sample = (train[0], valid[1])
    periods = {"train": train, "validation": valid, "train+validation": in_sample}
    samples = {"insample": in_sample}  # used for alphas, GRS and robustness
    if args.holdout:
        periods.update({"test": test, "full": (train[0], test[1])})
        samples["test"] = test
    start, end = train[0], (test[1] if args.holdout else valid[1])
    # Optional: the periods for alphas and robustness listed in the config
    # (evaluation.alpha_periods, e.g. full sample, last 30 years, last 20 years).
    # They replace "insample"; a period is cut at `end` (the test years only with --holdout).
    alpha_periods = cfg["evaluation"].get("alpha_periods") or {}
    if alpha_periods:
        samples = {k: (v[0], min(v[1], end)) for k, v in alpha_periods.items() if v[0] <= end}
        periods.update(samples)

    out_dir = BASE_DIR / "results" / "signals" / name
    if args.holdout:
        log_holdout(name, config_path)
        out_dir = out_dir / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    weightings = cfg["portfolio"]["weighting"]
    # a signal module can set N_PORTFOLIOS (e.g. 2 for a yes/no signal)
    portfolio_cfg = {**cfg["portfolio"],
                     "n_portfolios": getattr(spec, "N_PORTFOLIOS", cfg["portfolio"]["n_portfolios"])}
    ports = list(range(1, portfolio_cfg["n_portfolios"] + 1))
    spreads = load_spreads()
    ev = cfg["evaluation"]
    factors = load_factors(BASE_DIR / cfg["data"]["factors"])
    # skip models whose factors are not in the file (e.g. STREV before
    # `python src/data/factors.py --add-reversal`)
    usable = [m for m in ev["models"]
              if set(MODELS[m]["factors"] + [MODELS[m]["rf"]]) <= set(factors.columns)]
    if len(usable) < len(ev["models"]):
        print(f"!! Skipping models with factors missing from the factor file: "
              f"{sorted(set(ev['models']) - set(usable))}")
        ev = {**ev, "models": usable,
              "headline_models": [m for m in ev["headline_models"] if m in usable]}

    # 1. Signal and portfolios
    print(f"Loading the panel and computing '{name}' ({spec.REBALANCE} rebalancing)...")
    panel = build_panel(spec)
    impact, cost_label = cost_settings(panel, spreads, cfg.get("costs"))
    print(f"Trading costs: {cost_label}")
    print(f"Sorting into {len(ports)} portfolios, months {start} to {end}...")
    base = backtest(panel, name, spec, sort_options(cfg["universe"], portfolio_cfg),
                    weightings, spreads, start, end, impact)
    spread = f"{spec.LONG}-{spec.SHORT}"
    net = f"{spread} net"
    base["n_stocks"].to_csv(out_dir / "n_stocks.csv")
    base["breakpoints"].to_csv(out_dir / "breakpoints.csv")

    # 2. Portfolio returns and the spread, period by period
    spread_rows = []
    for w in weightings:
        rets, to = base[w], base[f"turnover_{w}"]
        rets.to_csv(out_dir / f"returns_{w}.csv")
        to.to_csv(out_dir / f"turnover_{w}.csv")
        base[f"costs_{w}"].to_csv(out_dir / f"costs_{w}.csv")
        base[f"costs_spread_{w}"].to_csv(out_dir / f"costs_spread_{w}.csv")
        base[f"costs_impact_{w}"].to_csv(out_dir / f"costs_impact_{w}.csv")

        table = summarize(rets.loc[in_sample[0]:in_sample[1], ports], spec.LONG, spec.SHORT)
        table.to_csv(out_dir / f"portfolios_{w}.csv")
        print(f"\n===== {w.upper()}: portfolio returns, {in_sample[0]} to {in_sample[1]} =====")
        print(table.to_string())

        for period, (s, e) in periods.items():
            r = rets.loc[s:e]
            stats = spread_stats(r[spread], r[net], to.loc[s:e, spec.LONG], to.loc[s:e, spec.SHORT])
            spread_rows.append({"weighting": w, "period": period, **stats})

    spread_table = pd.DataFrame(spread_rows).set_index(["weighting", "period"])
    spread_table.to_csv(out_dir / "spread.csv")
    print(f"\n===== Spread {spread} (long {spec.LONG}, short {spec.SHORT}), "
          f"net of {cost_label} =====")
    print(spread_table.to_string())

    # What the long-short spread pays, and why
    cost_rows = []
    for w in weightings:
        legs = [spec.LONG, spec.SHORT]
        for period, (s, e) in periods.items():
            sp = base[f"costs_spread_{w}"].loc[s:e, legs].sum(axis=1)
            im = base[f"costs_impact_{w}"].loc[s:e, legs].sum(axis=1)
            traded = 2 * base[f"turnover_{w}"].loc[s:e, legs].sum(axis=1)
            cost_rows.append({
                "weighting": w, "period": period,
                "Cost % per month": (sp + im).mean() * 100,
                "Spread part %": sp.mean() * 100,
                "Impact part %": im.mean() * 100,
                "bps per $ traded": (sp + im).sum() / traded.sum() * 1e4,
            })
    cost_table = pd.DataFrame(cost_rows).set_index(["weighting", "period"])
    cost_table.to_csv(out_dir / "costs_summary.csv")
    print(f"\n===== Trading costs of the spread {spread} (both legs) =====")
    print(cost_table.to_string())

    # 3. Alphas and GRS
    for sample, (s, e) in samples.items():
        for w in weightings:
            rets = base[w]
            table = alpha_table(rets[ports + [spread, net]], factors, ev["models"],
                                long_short_cols=[spread, net], nw_lags=ev["newey_west_lags"],
                                start=s, end=e)
            table.to_csv(out_dir / f"alphas_{sample}_{w}.csv")
            print(f"\n----- {w.upper()}, {sample} ({s} to {e}): alpha % per month -----")
            print(table.xs("alpha %", axis=1, level=1).to_string())
            print("t-stats (Newey-West):")
            print(table.xs("t", axis=1, level=1).to_string())

            grs = pd.DataFrame({m: grs_test(rets[ports], factors, m, s, e) for m in ev["models"]}).T
            grs.to_csv(out_dir / f"grs_{sample}_{w}.csv")
            print("GRS test (all portfolio alphas = 0):")
            print(grs.to_string(float_format=lambda x: f"{x:.3f}"))

    # 4. Robustness: the same spread in other universes
    variants = {"main": None, **(cfg.get("robustness") or {})}
    rows = []
    for variant, overrides in variants.items():
        if overrides is None:
            res = base
        else:
            print(f"\nRobustness variant '{variant}': {overrides}")
            # a variant can change universe keys and portfolio keys (e.g. buffer)
            port_over = {k: v for k, v in overrides.items() if k in PORTFOLIO_KEYS}
            cost_over = {k: v for k, v in overrides.items() if k in COST_KEYS}
            uni_over = {k: v for k, v in overrides.items()
                        if k not in PORTFOLIO_KEYS | COST_KEYS}
            var_impact = (cost_settings(panel, spreads, {**(cfg.get("costs") or {}), **cost_over})[0]
                          if cost_over else impact)
            res = backtest(panel, name, spec, sort_options({**cfg["universe"], **uni_over},
                           {**portfolio_cfg, **port_over}), weightings, spreads, start, end,
                           var_impact)
        for sample, (s, e) in samples.items():
            n = res["n_stocks"].loc[s:e]
            for w in weightings:
                r, to = res[w].loc[s:e], res[f"turnover_{w}"].loc[s:e]
                row = {"variant": variant, "sample": sample, "weighting": w,
                       **spread_stats(r[spread], r[net], to[spec.LONG], to[spec.SHORT]),
                       "Stocks long": n[spec.LONG].mean(), "Stocks short": n[spec.SHORT].mean()}
                for m in ev["models"]:
                    a = regress(r[spread], factors, m, long_short=True,
                                nw_lags=ev["newey_west_lags"], start=s, end=e)
                    row[f"{m} alpha %"], row[f"{m} t"] = a["alpha %"], a["t(alpha)"]
                    # the same alpha after trading costs (net spread)
                    a = regress(r[net], factors, m, long_short=True,
                                nw_lags=ev["newey_west_lags"], start=s, end=e)
                    row[f"{m} net alpha %"], row[f"{m} net t"] = a["alpha %"], a["t(alpha)"]
                rows.append(row)
    robust = pd.DataFrame(rows).set_index(["variant", "sample", "weighting"])
    robust.to_csv(out_dir / "robustness.csv")
    shown = ["Avg %", "t", "Break-even bps", "Stocks long", "Stocks short"]
    shown += [c for m in ev["headline_models"] for c in (f"{m} alpha %", f"{m} t")]
    print(f"\n===== BEFORE costs: spread {spread} by universe and period "
          f"(all models in robustness.csv) =====")
    print(robust[shown].to_string())
    shown = ["Net avg %", "Net t", "Cost %", "Net months", "Turnover % long", "Turnover % short"]
    shown += [c for m in ev["headline_models"] for c in (f"{m} net alpha %", f"{m} net t")]
    print(f"\n===== AFTER costs ({cost_label}): net spread and net alphas =====")
    print(robust[shown].to_string())

    info = {
        "signal": name, "rebalance": spec.REBALANCE, "long": spec.LONG, "short": spec.SHORT,
        "months": [start, end], "holdout": args.holdout, "config_file": str(config_path),
        "config": cfg, "run_at": f"{datetime.now():%Y-%m-%d %H:%M}",
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nResults saved in: {out_dir}")


if __name__ == "__main__":
    main()
