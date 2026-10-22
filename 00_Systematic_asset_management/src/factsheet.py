"""Two-page PDF factsheet built from the daily NAV of the paper-trading account.

NAV history source, in order: IBKR Flex Web Service -> last saved Flex XML -> local nav_log.csv.
Returns are time-weighted (deposits and withdrawals removed).
"""
import datetime as dt
import os
import textwrap
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

FLEX_BASE = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"


# ---------------------------------------------------------------------
# 1. NAV history
# ---------------------------------------------------------------------
def _flex_date(s):
    s = str(s).split(";")[0].split(" ")[0].replace("-", "")[:8]
    return pd.to_datetime(s, format="%Y%m%d")


def flex_download(token, query_id):
    h = {"User-Agent": "Python/3"}
    r = requests.get(f"{FLEX_BASE}/SendRequest", params={"t": token, "q": query_id, "v": 3}, headers=h, timeout=30)
    root = ET.fromstring(r.text)
    if root.findtext("Status") != "Success":
        raise RuntimeError(f"Flex: {root.findtext('ErrorMessage')}")
    ref, url = root.findtext("ReferenceCode"), root.findtext("Url") or f"{FLEX_BASE}/GetStatement"
    for _ in range(12):
        time.sleep(5)
        s = requests.get(url, params={"t": token, "q": ref, "v": 3}, headers=h, timeout=60)
        if "<FlexQueryResponse" in s.text:
            return s.text
    raise RuntimeError("Flex: statement not ready yet, retry in a minute.")


def _account_rows(root, tag, account):
    """Elements of one account. The account is taken from the row or, if the Account ID field
    is not in the query, from the enclosing FlexStatement, so multi-account queries are safe."""
    statements = list(root.iter("FlexStatement")) or [root]
    for st in statements:
        st_acc = st.get("accountId")
        for e in st.iter(tag):
            acc = e.get("accountId") or st_acc
            if account and acc not in (None, "", account):
                continue
            yield e


def flex_parse(xml_text, account):
    """Daily NAV and external cash flows (deposits/withdrawals) for one account."""
    root = ET.fromstring(xml_text)
    found = {st.get("accountId") for st in root.iter("FlexStatement")} - {None, ""}
    if account and found and account not in found:
        raise ValueError(f"The Flex statement contains account(s) {sorted(found)}, not {account}: "
                         f"check the Flex token / query id used for this strategy.")
    nav, flows = {}, {}
    for e in _account_rows(root, "EquitySummaryByReportDateInBase", account):
        nav[_flex_date(e.get("reportDate"))] = float(e.get("total"))
    for e in _account_rows(root, "ChangeInNAV", account):
        if not e.get("endingValue"):
            continue
        d = _flex_date(e.get("toDate"))
        nav[d] = float(e.get("endingValue"))
        dep = float(e.get("depositsWithdrawals") or 0)
        if dep:
            flows[d] = flows.get(d, 0.0) + dep
    use_cash_tx = not flows                       # avoid counting deposits twice
    for e in _account_rows(root, "CashTransaction", account):
        if not use_cash_tx:
            continue
        if "deposit" not in (e.get("type") or "").lower():
            continue
        if (e.get("levelOfDetail") or "DETAIL").upper() != "DETAIL":
            continue
        d = _flex_date(e.get("reportDate") or e.get("dateTime"))
        flows[d] = flows.get(d, 0.0) + float(e.get("amount")) * float(e.get("fxRateToBase") or 1)
    return pd.Series(nav).sort_index(), pd.Series(flows, dtype=float)


def _read_nav_log(path):
    log = pd.read_csv(path, dtype={"data": str, "date": str})
    return log.rename(columns={"data": "date", "conto": "account", "strategia": "strategy"})


def update_nav_log(path, account, strategy, nav_today):
    """Append today's NAV to a local CSV (last-resort history if Flex is unavailable)."""
    row = pd.DataFrame([{"date": dt.date.today().isoformat(), "account": account,
                         "strategy": strategy, "nav": round(nav_today, 2)}])
    if os.path.exists(path):
        log = _read_nav_log(path)
        log = log[~((log["date"] == row.at[0, "date"]) & (log["account"] == account))]
        row = pd.concat([log, row], ignore_index=True).sort_values(["account", "date"])
    row.to_csv(path, index=False)


def load_nav_history(account, start_date, flex_token, flex_query_id, flex_file, nav_log, nav_today=None):
    source = None
    if flex_token and flex_query_id:
        try:
            xml_text = flex_download(flex_token, flex_query_id)
            with open(flex_file, "w", encoding="utf-8") as f:
                f.write(xml_text)
            source = "IBKR Flex Web Service"
        except Exception as err:
            print(f"Flex download failed ({err}); using the saved file if available.")
    if os.path.exists(flex_file):
        with open(flex_file, encoding="utf-8") as f:
            nav, flows = flex_parse(f.read(), account)
        source = source or f"file {flex_file}"
    else:
        log = _read_nav_log(nav_log)
        log = log[log["account"] == account]
        nav = pd.Series(log["nav"].values, index=pd.to_datetime(log["date"])).sort_index()
        flows = pd.Series(dtype=float)
        source = nav_log

    # Cross-check: NAVs recorded live on earlier runs must match the Flex history of the same days
    if source != nav_log and os.path.exists(nav_log) and not nav.empty:
        log = _read_nav_log(nav_log)
        log = log[log["account"] == account]
        live = pd.Series(log["nav"].values, index=pd.to_datetime(log["date"]))
        common = live.index.intersection(nav.index)
        if len(common):
            diff = (live[common] / nav[common] - 1).abs().median()
            if diff > 0.05:
                raise ValueError(f"Flex NAV differs by {diff:.1%} (median) from the NAV recorded live for "
                                 f"{account} on {len(common)} day(s): the Flex history belongs to another "
                                 f"account. Check the Flex token / query id and delete {flex_file}.")

    today = pd.Timestamp(dt.date.today())
    if nav_today is not None and not nav.empty:
        gap = nav_today / nav.iloc[-1] - 1
        if abs(gap) > 0.08 and (today - nav.index[-1]).days <= 7:
            raise ValueError(f"Live NAV {nav_today:,.0f} differs by {gap:+.1%} from the last Flex NAV "
                             f"{nav.iloc[-1]:,.0f} ({nav.index[-1].date()}): the history probably belongs "
                             f"to another account. Check the Flex token / query id and {flex_file}.")
    if nav_today is not None and (nav.empty or nav.index[-1] < today):
        nav.loc[today] = nav_today
        nav.index = pd.to_datetime(nav.index)
    flows.index = pd.to_datetime(flows.index)
    nav = nav.sort_index()
    nav = nav[nav.index >= pd.Timestamp(start_date)]
    if len(nav) < 2:
        raise ValueError("Not enough NAV history after the start date. The Flex Query must include "
                         "'Net Asset Value (NAV) in Base' over 'Last 365 Calendar Days'.")
    flows = flows[flows.index > nav.index[0]]
    print(f"NAV history: {len(nav)} days ({nav.index[0].date()} -> {nav.index[-1].date()}), source: {source}")
    return nav, flows


# ---------------------------------------------------------------------
# 2. Statistics
# ---------------------------------------------------------------------
def _perf_since(index_series, since):
    base = index_series[index_series.index <= since]
    base = base.iloc[-1] if not base.empty else index_series.iloc[0]
    return index_series.iloc[-1] / base - 1


def _period_returns(s, last):
    periods = [("MTD", last - pd.offsets.MonthEnd(1)), ("Last month", None),
               ("3M", last - pd.DateOffset(months=3)), ("YTD", pd.Timestamp(last.year - 1, 12, 31)),
               ("Since inception", s.index[0])]
    out = []
    for name, since in periods:
        if name == "Last month":
            m = s.resample("ME").last()
            v = m.iloc[-2] / m.iloc[-3] - 1 if len(m) >= 3 else (m.iloc[-2] / s.iloc[0] - 1 if len(m) == 2 else np.nan)
        else:
            v = _perf_since(s, since)
        out.append("–" if pd.isna(v) else f"{v*100:.2f}")
    return out


# ---------------------------------------------------------------------
# 3. PDF
# ---------------------------------------------------------------------
BLUE, GREY, LIGHT, RED = "#1F3A5F", "#6B7280", "#E5E7EB", "#C62828"


def _title(fig, x, y, text, w):
    fig.add_artist(Rectangle((x, y - 0.004), w, 0.022, color=BLUE, transform=fig.transFigure))
    fig.text(x + 0.008, y + 0.007, text, fontsize=9.5, color="white", va="center", weight="bold")


def _table(fig, x, y, rows, widths, h=0.0175, header=None, align=None, fs=7.5):
    align = align or ["left"] + ["right"] * (len(widths) - 1)
    if header:
        rows = [header] + list(rows)
    for i, r in enumerate(rows):
        yy = y - i * h
        cx = x
        for j, (val, w) in enumerate(zip(r, widths)):
            xx = cx + (w - 0.005 if align[j] == "right" else 0.004)
            fig.text(xx, yy, str(val), fontsize=fs, ha=align[j], va="center",
                     color=GREY if (header and i == 0) else "black",
                     weight="bold" if (header and i == 0) else "normal")
            cx += w
        fig.add_artist(Rectangle((x, yy - h / 2), sum(widths), 0.0006, color=LIGHT, transform=fig.transFigure))
    return y - len(rows) * h


def build_factsheet(*, strategy, account, account_value, holdings, targets, model_weights, ucits_map,
                    asset_class, fund_name, universe, strategy_text, start_date,
                    flex_token="", flex_query_id="", use_ibkr=True,
                    benchmark=("SXR8.DE", "S&P 500 (EUR)"), risk_free=0.02, changes=(),
                    data_dir=os.path.join("..", "data"), out_dir=os.path.join("..", "factsheets")):
    """changes: iterable of (date, label) strategy changes, drawn as vertical lines on the charts."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    nav_log = os.path.join(data_dir, "nav_log.csv")
    flex_file = os.path.join(data_dir, f"flex_{strategy}.xml")

    update_nav_log(nav_log, account, strategy, account_value)
    nav, flows = load_nav_history(account, start_date, flex_token, flex_query_id, flex_file, nav_log,
                                  nav_today=account_value if use_ibkr else None)

    # Time-weighted returns
    flows_d = flows.reindex(nav.index, fill_value=0.0)
    ret = ((nav - flows_d) / nav.shift(1) - 1).dropna()
    idx = pd.concat([pd.Series([100.0], index=[nav.index[0]]), 100 * (1 + ret).cumprod()])
    last = idx.index[-1]
    monthly = idx.resample("ME").last()
    monthly_ret = monthly.pct_change()
    monthly_ret.iloc[0] = monthly.iloc[0] / 100 - 1          # first (partial) month from inception

    bench_idx = None
    try:
        b = yf.download(benchmark[0], start=nav.index[0] - pd.Timedelta(days=7),
                        end=last + pd.Timedelta(days=1), auto_adjust=True, progress=False)["Close"]
        b = b.squeeze().reindex(idx.index, method="ffill").dropna()
        bench_idx = 100 * b / b.iloc[0]
    except Exception as err:
        print(f"Benchmark not downloaded ({err}).")

    days = (last - idx.index[0]).days
    vol = ret.std() * np.sqrt(252)
    ann_mean = ret.mean() * 252
    downside = ret[ret < 0].std() * np.sqrt(252)
    dd = idx / idx.cummax() - 1
    cum = idx.iloc[-1] / 100 - 1

    stats = {
        "Return since inception (%)": f"{cum*100:.2f}",
        "Return annualised (%)": f"{((1+cum)**(365/days)-1)*100:.2f}" if days >= 365 else "n.m. (< 1Y)",
        "Volatility annualised (%)": f"{vol*100:.2f}",
        "Sharpe ratio": f"{(ann_mean - risk_free)/vol:.2f}" if vol > 0 else "n.a.",
        "Sortino ratio": f"{(ann_mean - risk_free)/downside:.2f}" if downside > 0 else "n.a.",
        "Max drawdown (%)": f"{dd.min()*100:.2f}",
        "Current drawdown (%)": f"{dd.iloc[-1]*100:.2f}",
        "Best month (%)": f"{monthly_ret.max()*100:.2f}",
        "Worst month (%)": f"{monthly_ret.min()*100:.2f}",
        "Positive months (%)": f"{(monthly_ret > 0).mean()*100:.0f}",
    }
    if bench_idx is not None and len(bench_idx) > 20:
        br = bench_idx.pct_change().dropna()
        al = pd.concat([ret, br], axis=1, join="inner").dropna()
        al.columns = ["p", "b"]
        stats[f"Beta vs {benchmark[1]}"] = f"{al['p'].cov(al['b'])/al['b'].var():.2f}"
        stats[f"Correlation vs {benchmark[1]}"] = f"{al['p'].corr(al['b']):.2f}"
    stats["Risk-free rate (%)"] = f"{risk_free*100:.2f}"

    # Holdings and allocation history
    positions = sorted([(k, v) for k, v in holdings.items() if abs(v) > 1], key=lambda x: -x[1])
    cash = account_value - sum(v for _, v in positions)
    classes = {}
    for k, v in positions:
        c = asset_class.get(k, "Other")
        classes[c] = classes.get(c, 0) + v
    if abs(cash) > 1:
        classes["Cash"] = classes.get("Cash", 0) + cash

    history = model_weights.copy()
    history.index = history.index + pd.offsets.MonthEnd(1)       # month-end signal -> holding month
    history = history[history.index >= pd.Timestamp(start_date) + pd.offsets.MonthEnd(0)]
    history = history.loc[:, (history > 0.0001).any()]

    palette = plt.get_cmap("tab20").colors
    etf_color = {k: palette[i % 20] for i, k in enumerate(ucits_map)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    def page(fig, n):
        fig.text(0.06, 0.955, fund_name, fontsize=22, color=BLUE, weight="light", va="bottom")
        fig.text(0.06, 0.935, f"Systematic momentum strategy  ·  Factsheet {last:%d %B %Y}", fontsize=9.5, color=GREY)
        fig.add_artist(Rectangle((0.06, 0.925), 0.88, 0.0015, color=BLUE, transform=fig.transFigure))
        fig.text(0.06, 0.03, "Paper trading account (simulated capital). For information only, not investment "
                 "advice. Past performance is not a reliable indicator of future results.", fontsize=6.5, color=GREY)
        fig.text(0.94, 0.03, f"{n}/2", fontsize=7, color=GREY, ha="right")

    pdf_path = os.path.join(out_dir, f"Factsheet_{strategy}_{last:%Y-%m}.pdf")
    with PdfPages(pdf_path) as pdf:
        # ---------------- Page 1 ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        page(fig, 1)
        _title(fig, 0.06, 0.89, "Strategy", 0.42)
        fig.text(0.065, 0.872, textwrap.fill(" ".join(strategy_text.split()), 64), fontsize=7.6, va="top", linespacing=1.45)

        _title(fig, 0.06, 0.635, "Performance (rebased to 100)", 0.42)
        ax = fig.add_axes([0.085, 0.43, 0.39, 0.19])
        ax.plot(idx.index, idx.values, color=BLUE, lw=1.6, label=fund_name)
        if bench_idx is not None:
            ax.plot(bench_idx.index, bench_idx.values, color=GREY, lw=1, ls="--", label=benchmark[1])
        ax.axhline(100, color=LIGHT, lw=0.8)
        for d, lab in changes:
            if idx.index[0] <= pd.Timestamp(d) <= last:
                ax.axvline(pd.Timestamp(d), color=RED, lw=0.8, ls=":")
                ax.text(pd.Timestamp(d), ax.get_ylim()[1], f" {lab}", fontsize=5.5, color=RED, va="top")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.tick_params(labelsize=6.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %y"))

        ax2 = fig.add_axes([0.085, 0.33, 0.39, 0.075])
        ax2.fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.35, lw=0)
        ax2.set_ylabel("Drawdown %", fontsize=6.5)
        ax2.tick_params(labelsize=6.5)
        for s in ["top", "right"]:
            ax2.spines[s].set_visible(False)
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %y"))

        _title(fig, 0.06, 0.29, "Returns (%)", 0.42)
        perf_rows = [["Portfolio"] + _period_returns(idx, last)]
        if bench_idx is not None:
            perf_rows.append([benchmark[1]] + _period_returns(bench_idx, last))
        _table(fig, 0.06, 0.262, perf_rows, [0.11, 0.062, 0.062, 0.062, 0.062, 0.062],
               header=["", "MTD", "Last m.", "3M", "YTD", "Incept."], fs=7)

        _title(fig, 0.52, 0.89, "Portfolio information", 0.42)
        info = [("Date", f"{last:%d %B %Y}"), ("Net asset value", f"EUR {account_value:,.0f}"),
                ("Inception", f"{pd.Timestamp(start_date):%d %B %Y}"), ("Base currency", "EUR"),
                ("Instruments", "UCITS ETFs"), ("Universe", universe), ("Rebalancing", "Monthly"),
                ("Broker", "Interactive Brokers"), ("Status", "Paper trading (forward test)"),
                ("Benchmark (reference)", benchmark[1])]
        _table(fig, 0.52, 0.862, info, [0.20, 0.22], align=["left", "left"])

        _title(fig, 0.52, 0.66, "Portfolio statistics (since inception)", 0.42)
        y_end = _table(fig, 0.52, 0.632, list(stats.items()), [0.32, 0.10])
        fig.text(0.52, y_end - 0.01, textwrap.fill(
            "Returns are time-weighted, in EUR, computed on the daily net asset value of the account "
            "(deposits and withdrawals excluded). Volatility, Sharpe and Sortino are annualised from daily "
            "returns. Annualised return is shown only after one full year.", 62),
            fontsize=6.3, color=GREY, va="top", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 2 ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        page(fig, 2)
        _title(fig, 0.06, 0.89, f"Current holdings ({last:%d %b %Y})", 0.88)
        pos_rows = []
        for k, v in positions:
            d = ucits_map.get(k, {"desc": k, "isin": "–", "ticker": k})
            pos_rows.append([d["desc"], d["isin"], d["ticker"], asset_class.get(k, "Other"),
                             f"{v:,.0f}", f"{100*v/account_value:.1f}"])
        pos_rows.append(["Cash", "–", "–", "Cash", f"{cash:,.0f}", f"{100*cash/account_value:.1f}"])
        pos_rows.append(["Total", "", "", "", f"{account_value:,.0f}", "100.0"])
        y = _table(fig, 0.06, 0.862, pos_rows, [0.28, 0.15, 0.09, 0.14, 0.12, 0.10],
                   header=["Instrument", "ISIN", "Ticker", "Asset class", "Value (EUR)", "Weight (%)"],
                   align=["left", "left", "left", "left", "right", "right"])

        y0 = y - 0.03
        _title(fig, 0.06, y0, "Asset class allocation (%)", 0.42)
        axc = fig.add_axes([0.17, y0 - 0.15, 0.27, 0.13])
        cl = pd.Series(classes).sort_values() / account_value * 100
        axc.barh(cl.index, cl.values, color=BLUE, height=0.55)
        for i, v in enumerate(cl.values):
            axc.text(v + 1, i, f"{v:.1f}", va="center", fontsize=6.5)
        axc.set_xlim(0, max(100, cl.max() + 10))
        axc.tick_params(labelsize=6.5)
        for s in ["top", "right", "bottom"]:
            axc.spines[s].set_visible(False)
        axc.set_xticks([])

        _title(fig, 0.52, y0, "Model target, next rebalancing (%)", 0.42)
        tg = pd.Series(targets).sort_values() * 100
        axt = fig.add_axes([0.66, y0 - 0.15, 0.24, 0.13])
        axt.barh([ucits_map.get(k, {"ticker": k})["ticker"] + f" ({k})" for k in tg.index], tg.values,
                 color=[etf_color.get(k, GREY) for k in tg.index], height=0.55)
        for i, v in enumerate(tg.values):
            axt.text(v + 1, i, f"{v:.1f}", va="center", fontsize=6.5)
        axt.set_xlim(0, 110)
        axt.tick_params(labelsize=6.5)
        axt.set_xticks([])
        for s in ["top", "right", "bottom"]:
            axt.spines[s].set_visible(False)

        y1 = y0 - 0.19
        _title(fig, 0.06, y1, "Allocation history (model weights by holding month, %)", 0.88)
        axh = fig.add_axes([0.09, y1 - 0.2, 0.62, 0.18])
        bottom = np.zeros(len(history))
        labels = [f"{d:%b %y}" for d in history.index]
        for k in history.columns:
            val = history[k].values * 100
            axh.bar(labels, val, bottom=bottom, color=etf_color.get(k, GREY), width=0.65,
                    label=f"{ucits_map.get(k, {'ticker': k})['ticker']} ({k})")
            bottom += val
        axh.set_ylim(0, 100)
        axh.tick_params(labelsize=6.5)
        for s in ["top", "right"]:
            axh.spines[s].set_visible(False)
        axh.legend(frameon=False, fontsize=6.5, bbox_to_anchor=(1.02, 1), loc="upper left")

        y2 = y1 - 0.25
        _title(fig, 0.06, y2, "Monthly returns (%)", 0.88)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        m_rows = []
        for year in sorted(set(monthly_ret.index.year)):
            r = monthly_ret[monthly_ret.index.year == year]
            row = [str(year)] + [""] * 12
            for d, v in r.items():
                row[d.month] = f"{v*100:.2f}"
            prev = monthly[monthly.index.year < year]
            ytd = monthly[monthly.index.year == year].iloc[-1] / (prev.iloc[-1] if len(prev) else 100) - 1
            m_rows.append(row + [f"{ytd*100:.2f}"])
        _table(fig, 0.06, y2 - 0.028, m_rows, [0.06] + [0.06] * 12 + [0.07], header=[""] + months + ["YTD"], fs=6.8)
        fig.text(0.06, y2 - 0.028 - (len(m_rows) + 1) * 0.0175 - 0.005,
                 f"First month measured from inception ({pd.Timestamp(start_date):%d %b %Y}); "
                 "current month is month-to-date.", fontsize=6.3, color=GREY, va="top")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Factsheet saved: {pdf_path}")
    return pdf_path, stats
