"""
What the final strategy holds: characteristics and industry weights of the long leg
(decile 10), the short leg (decile 1) and all sorted stocks, at every formation date
1963-07 to 2015-12 (the holdout is not used). Read by scripts/paper_outputs.py
(Table 2 and Figure 1 of the paper).

Run from the project folder (needs data/interim/, about 5 minutes):
    python scripts/describe_legs.py

Same signal, universe and portfolios as the backtest (configs/final.yaml): the
portfolios formed every 3 months, NYSE decile breakpoints, no microcaps.
For each formation date and group, the MEDIAN stock of each characteristic is taken
(the share of firms reporting R&D is a mean), then averaged over the formation dates:
    market cap ($bn)                       firm market cap at formation
    book-to-market                         be / market cap
    book-to-market with intangible capital (be - goodwill + K + O) / market cap
    return on assets (%)                   ib / at
    intangible capital / total capital (%) (K + O) / (at + K + O)
    firms reporting R&D (%)                xrd > 0
    past return, months -12 to -2 (%)      momentum signal at formation
Industry weights: value weight (market cap at formation) of each FF12 industry in
each group, averaged over the formation dates.
Output: results/paper/legs_characteristics.csv and results/paper/industry_weights.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import run_signal as rs  # noqa: E402
from src.portfolio.sorts import sort_portfolios  # noqa: E402
from src.signals import build_panel, load_signal  # noqa: E402
from src.signals._industry import NAMES, ff12  # noqa: E402
from src.signals._qvi_ext import intangible_capital  # noqa: E402
from src.signals.momentum import momentum  # noqa: E402

SIGNAL, CONFIG = "qvi_timely", "configs/final.yaml"
FIRST_FORMATION = "1963-06"  # first portfolios held in July 1963, as in the backtest
LAST_FORMATION = "2015-11"   # held in 2015-12 at the latest: no holdout month
OUT = BASE_DIR / "results" / "paper"
INDUSTRY_NAMES = {"NoDur": "Consumer non-durables", "Durbl": "Consumer durables", "Manuf": "Manufacturing",
                  "Enrgy": "Energy", "Chems": "Chemicals", "BusEq": "Business equipment (tech)",
                  "Telcm": "Telecom", "Utils": "Utilities", "Shops": "Retail and wholesale",
                  "Hlth": "Healthcare", "Money": "Finance", "Other": "Other"}


def main() -> None:
    cfg = yaml.safe_load((BASE_DIR / CONFIG).read_text())
    spec = load_signal(SIGNAL)
    portfolio_cfg = {**cfg["portfolio"], "n_portfolios": getattr(spec, "N_PORTFOLIOS", cfg["portfolio"]["n_portfolios"])}
    opts = rs.sort_options(cfg["universe"], portfolio_cfg)

    print(f"Loading the panel and computing '{SIGNAL}'...", flush=True)
    panel = build_panel(spec)
    print("Sorting...", flush=True)
    res = sort_portfolios(panel, SIGNAL, rebalance=spec.REBALANCE, **opts)
    a = res["assignments"]
    first = pd.Period(FIRST_FORMATION, freq="M")
    last = pd.Period(LAST_FORMATION, freq="M")
    a = a[(a["form_key"] >= first.year * 12 + first.month - 1) & (a["form_key"] <= last.year * 12 + last.month - 1)]

    # characteristics of every stock-month, then the formation rows only
    me = (panel["me_firm"] / 1000).where(panel["me_firm"] > 0)          # $ millions
    at = panel["at"].where(panel["at"] > 0)
    cap = intangible_capital(panel)
    book_int = panel["be"] - panel["gdwl"].fillna(0).clip(lower=0) + cap.fillna(0)
    ch = pd.DataFrame({
        "PERMNO": panel["PERMNO"],
        "form_key": panel["MthCalDt"].dt.year * 12 + panel["MthCalDt"].dt.month - 1,
        "Market capitalisation ($bn)": me / 1000,
        "Book-to-market": (panel["be"] / me).where(panel["be"] > 0),
        "Book-to-market with intangible capital": (book_int / me).where(book_int > 0),
        "Return on assets (%)": panel["ib"] / at * 100,
        "Intangible capital / total capital (%)": cap / (at + cap.fillna(0)) * 100,
        "Firms reporting R&D (%, mean)": (panel["xrd"].fillna(0) > 0).astype(float) * 100,
        "Past return, months -12 to -2 (%)": momentum(panel) * 100,
        "industry": ff12(panel["SICCD"]),
    }).replace([np.inf, -np.inf], np.nan)
    d = a.merge(ch, on=["PERMNO", "form_key"], how="left")
    n_long = int(d["port"].max())
    groups = {"Short (decile 1)": d["port"] == 1, f"Long (decile {n_long})": d["port"] == n_long,
              "All sorted stocks": pd.Series(True, index=d.index)}
    chars = [c for c in ch.columns if c not in ("PERMNO", "form_key", "industry")]

    rows, weights = {}, {}
    for name, mask in groups.items():
        g = d[mask]
        med = g.groupby("form_key")[[c for c in chars if "mean" not in c]].median()
        mean = g.groupby("form_key")[[c for c in chars if "mean" in c]].mean()
        rows[name] = pd.concat([med, mean], axis=1).mean()[chars]
        w = g.groupby(["form_key", "industry"])["mkt_cap"].sum()
        w = (w / w.groupby(level=0).transform("sum")).unstack(fill_value=0)
        weights[name] = w.mean() * 100
    table = pd.DataFrame(rows)
    ind = pd.DataFrame(weights).fillna(0)
    ind.index = [INDUSTRY_NAMES.get(NAMES.get(int(i), "Other"), str(i)) for i in ind.index]
    ind = ind.groupby(level=0).sum().sort_values("All sorted stocks", ascending=False)

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "legs_characteristics.csv")
    ind.to_csv(OUT / "industry_weights.csv")
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())
    print()
    print(ind.round(1).to_string())
    print(f"\nSaved in {OUT}: legs_characteristics.csv, industry_weights.csv "
          f"({a['form_key'].nunique()} formation dates)")


if __name__ == "__main__":
    main()
