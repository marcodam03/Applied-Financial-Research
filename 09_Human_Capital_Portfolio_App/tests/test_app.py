"""Tests of the app (step 5): the numbers in src/dashboard.py, the results kept on disk, the
background jobs, then the app itself, run headless with Streamlit's own test tool (every
persona, other settings).

Run from the project folder (about two minutes the first time, then the backtests are saved):
    python tests/test_app.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.allocation as AL
import src.backtest as B
import src.dashboard as D
import src.optimizer as O
import src.theme as T
from src.jobs import Jobs
from src.career import params_for_persona
from src.config import load_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_client_params():
    base = load_params()
    # No change: exactly the persona of step 2
    assert D.client_params("dentist") == params_for_persona(base, "dentist")
    p = D.client_params("banker", {"age": 50, "salary_chf": 200_000, "portfolio_chf": 500_000,
                                   "deferred_shares_chf": 20_000, "pension_chf": 300_000, "beta_sector": 0.4,
                                   "wage_rate_link": 0.2, "long_only": False, "no_leverage": False,
                                   "sector_cap": 0.25, "view": {"asset": "EQ_CH", "q": 0.06, "sd": 0.05}})
    c = p["client"]
    assert (c["age"], c["salary_chf"], c["portfolio_chf"], c["deferred_shares_chf"], c["pension_chf"]) == \
        (50, 200_000, 500_000, 20_000, 300_000)
    assert p["career"]["beta_sector"] == 0.4 and p["career"]["wage_rate_link"] == 0.2
    assert p["optimization"]["long_only"] is False and p["optimization"]["max_leverage"] is None
    assert set(p["optimization"]["position_caps"].values()) == {0.25}
    assert p["black_litterman"]["views"] == [{"assets": {"EQ_CH": 1.0}, "q": 0.06, "sd": 0.05}]
    assert base["client"]["age"] == 35                       # the shared params are not changed
    # A civil servant has no sector: the sector beta stays 0; locked shares never exceed the portfolio
    assert D.client_params("civil_servant", {"beta_sector": 0.5})["career"]["beta_sector"] == 0.0
    assert D.client_params("banker", {"portfolio_chf": 10_000})["client"]["deferred_shares_chf"] == 10_000
    print("client settings OK")


def test_analysis_and_lifecycle():
    excess = O.load_excess_returns()
    for name in load_params()["personas"]:
        p = D.client_params(name)
        r = D.analysis(p, excess)
        sheet, story = r["balance_sheet"], r["story"]
        assert abs(story["portfolio_share"] - sheet["F"] / sheet["W"]) < 1e-15
        assert 0 <= story["career_risk_share"] <= 1 and story["bonds_A"] >= story["bonds_D"] - 1e-12
        assert story["sector_C"] <= story["sector_D"] + 1e-12
        # unconstrained demands in CHF: the hedging part is minus the career's hidden holdings
        dem = D.demand_chf(r)
        for asset, value in r["career_holdings"].items():
            assert abs(dem.loc[asset, "hedging"] + value) < 1e-6 * max(1.0, value)
        # life cycle: starts at today's balance sheet, the career is worth 0 at retirement
        lc = D.lifecycle(p, r)
        assert abs(lc["H"].iloc[0] - sheet["H"]) < 1e-6 and abs(lc["P"].iloc[0] - sheet["P"]) < 1e-6
        assert lc["F"].iloc[0] == sheet["F"] and lc["H"].iloc[-1] == 0.0
        assert lc.index[-1] == p["client"]["retirement_age"] and (np.diff(lc["F"]) > 0).all()
    print("analysis and life cycle OK")


def test_plan_numbers():
    excess = O.load_excess_returns()
    p = D.client_params("banker")
    r = D.analysis(p, excess)
    # the bars of "How we value your career" add up to H
    years, rate = D.career_years(p)
    assert abs(years["value today"].sum() - r["balance_sheet"]["H"]) < 1e-6
    assert years.index[0] == p["client"]["age"] and years.index[-1] == p["client"]["retirement_age"] - 1
    # families: each portfolio sums to 1; the plan by age starts with today's advice
    for s in ("A", "C"):
        assert abs(D.groups(r["weights"][s]).sum() - 1) < 1e-9
    glide = D.glide_path(p, r, excess)
    assert glide.index[0] == p["client"]["age"] and glide.index[-1] < p["client"]["retirement_age"]
    assert abs(glide.loc[glide.index[0], ("C", "Bonds")] - D.groups(r["weights"]["C"])["Bonds"]) < 1e-12
    shares = [c for c in glide.columns if c[0] == "C"]
    assert np.allclose(glide[shares].sum(axis=1), 1)
    # the banker holds bonds again near retirement, never more than standard advice
    assert glide[("C", "Bonds")].iloc[-1] > glide[("C", "Bonds")].iloc[0]
    assert (glide[("C", "Bonds")] <= glide[("A", "Bonds")] + 1e-9).all()
    print("plan numbers OK")


def test_saved_results():
    # same client written as 130000 or 130000.0: the same saved file; another client: another file
    a = D.client_params("banker")
    b = D.client_params("banker", {"salary_chf": 130_000.0, "age": 35.0})
    assert D.cache_key("backtest", a) == D.cache_key("backtest", b)
    # the app's default inputs give the same client as the persona (so saved results are found)
    p = load_params()["personas"]["banker"]
    page = D.client_params("banker", {"age": p["age"], "salary_chf": p["salary_chf"], "portfolio_chf": 100_000,
                                      "deferred_shares_chf": p["deferred_shares_chf"], "pension_chf": None,
                                      "beta_sector": p["beta_sector"] * 1.0, "wage_rate_link": 0.5, "long_only": True,
                                      "no_leverage": True, "sector_cap": 0.1, "view": None})
    assert D.cache_key("backtest", a) == D.cache_key("backtest", page)
    assert D.cache_key("backtest", a) != D.cache_key("backtest", D.client_params("banker", {"age": 36}))
    first = D.remember("backtest", a)                          # computed (or read) once ...
    again = D.remember("backtest", b)                          # ... then read from disk
    assert first["gains"].equals(again["gains"])
    fresh = D.backtest(a, B.load_backtest_data())
    assert np.allclose(fresh["values"].to_numpy(), first["values"].to_numpy())
    # the background queue: a job returns its result; an older job that has not started is dropped
    jobs = Jobs()
    slow = jobs.get("slow", lambda: __import__("time").sleep(0.5) or 1, group="g")
    old = jobs.get("old", lambda: 2, group="g")
    new = jobs.get("new", lambda: 3, group="g")
    assert old.cancelled() and new.result() == 3 and slow.result() == 1 and not jobs.pending()
    print("saved results and background jobs OK")


def test_allocation():
    base = load_params()
    data = AL.load_data(base)
    excess = data[0].sub(data[1], axis=0)
    p = AL.client_params("banker", None, base)
    mu, cov, _ = AL.estimates(excess, p)
    # risk budgeting with a fixed holding: amounts add up, and y_i (Sigma e)_i is proportional to the budgets
    names = list(cov.columns)
    b = AL.budgets(p, mu, cov, ["equities", "real"], exclude=["IND_MONEY"])
    fixed = pd.Series(0.0, index=names)
    fixed["IND_MONEY"] = 15_000.0
    y = AL.risk_budget_amounts(cov, b, fixed, 85_000.0)
    assert abs(y.sum() - 85_000) < 1e-6 and y["IND_MONEY"] == 0 and (y[b > 0] > 0).all()
    rc = y * (cov.to_numpy() @ (y + fixed).to_numpy())
    ratio = (rc[b > 0] / b[b > 0])
    assert ratio.max() / ratio.min() - 1 < 1e-4
    # plans: weights add up to 1, no short; our plan leaves out the own industry except the stake
    r = AL.analyze(p, data)
    for kind, plan in r["plans"].items():
        w = plan["weights"]
        assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all(), kind
        assert abs(plan["risk_parts"].sum() - 1) < 1e-9, kind
    wC = r["plans"]["C"]["weights"]
    assert abs(wC["IND_MONEY"] - 15_000 / 100_000) < 1e-9
    # a young client with a career holds fewer bonds than typical advice, never 100% in one asset
    assert r["plans"]["C"]["groups"]["bonds"] < r["plans"]["A"]["groups"]["bonds"]
    assert wC.drop(["IND_MONEY", "CASH"]).max() < 0.2
    # the plan by age: more bonds near retirement
    g = AL.glide_path(p, mu, cov)
    assert g[("C", "bonds")].iloc[-1] > g[("C", "bonds")].iloc[0]
    # the range of the savings: today is known, the outcomes spread out until retirement, same seed same page
    fan, pension = AL.savings_range(p, g)
    c = p["client"]
    assert fan.index[0] == c["age"] and fan.index[-1] == c["retirement_age"]
    assert np.allclose(fan.iloc[0], c["portfolio_chf"]) and (fan.iloc[1:].diff(axis=1).iloc[:, 1:] > 0).all().all()
    assert fan.iloc[-1][0.95] - fan.iloc[-1][0.05] > fan.iloc[1][0.95] - fan.iloc[1][0.05]
    assert fan.equals(AL.savings_range(p, g)[0]) and pension > 0
    # the pension later on: today's amount earns the BVG rate and the legal credits are added (as in the
    # backtest), also at 25 when the legal estimate is still 0; extra money does not raise the future credits
    young0 = AL.client_params("civil_servant", {"age": 25, "pension_chf": 0.0}, base)
    young2 = AL.client_params("civil_servant", {"age": 25, "pension_chf": 200_000.0}, base)
    s65 = young0["client"]["salary_chf"] * (1 + base["career"]["salary_growth"]) ** 40
    extra = AL.pension_at(young2, 65, s65) - AL.pension_at(young0, 65, s65)
    assert abs(extra - 200_000 * (1 + base["pension"]["interest_rate"]) ** 40) < 1e-6
    assert AL.pension_at(young0, 65, s65) == O.pension_savings(65, s65, young0)       # nothing yet: the legal path
    assert AL.pension_at(young2, 25, young2["client"]["salary_chf"]) == 200_000          # today: the amount typed
    # the plan by age sees it too (from the next step on: at 25 the bond-like share is above 1, beta_bond > 1,
    # and adding pension moves it towards 1)
    g0, g2 = AL.glide_path(young0, mu, cov), AL.glide_path(young2, mu, cov)
    assert (g2[("", "bond-like share")].iloc[1:] > g0[("", "bond-like share")].iloc[1:]).all()
    # near retirement with large savings the loss limit moves part to cash
    old = AL.client_params("banker", {"age": 64, "portfolio_chf": 2_100_000, "stake_chf": 0, "profile": "prudent"}, base)
    r = AL.analyze(old, data)
    assert r["plans"]["C"]["detail"]["cash_from_loss_limit"] > 0
    # civil servant: no industry left out, no stake
    r = AL.analyze(AL.client_params("civil_servant", None, base), data)
    assert r["plans"]["C"]["detail"]["excluded"] == [] and not r["plans"]["C"]["detail"]["locked"]
    # a market view counts today, fades out in the plan by age, and the backtest never sees it
    view = {"asset": "IND_BUSEQ", "q": 0.15, "sd": 0.02}
    pv = AL.client_params("civil_servant", {"view": view}, base)
    rv = AL.analyze(pv, data)
    assert rv["mu"]["IND_BUSEQ"] > rv["mu_prior"]["IND_BUSEQ"] + 0.02
    plain = AL.analyze(AL.client_params("civil_servant", None, base), data)
    assert np.allclose(rv["mu_prior"], plain["mu"])
    assert rv["plans"]["C"]["weights"]["IND_BUSEQ"] > plain["plans"]["C"]["weights"]["IND_BUSEQ"]
    gv = AL.glide_path(pv, rv["mu"], rv["cov"], mu_prior=rv["mu_prior"])
    gp = AL.glide_path(AL.client_params("civil_servant", None, base), plain["mu"], plain["cov"])
    late = [a for a in gv.index if a - gv.index[0] >= base["allocation"]["view_years"]]
    groups_c = [c for c in gv.columns if c[0] == "C"]                 # after the fade only the savings path differs
    assert np.abs(gv.loc[late, groups_c].to_numpy(float) - gp.loc[late, groups_c].to_numpy(float)).max() < 0.01
    assert D.cache_key("backtest2", AL.without_client_view(pv)) == \
        D.cache_key("backtest2", AL.client_params("civil_servant", None, base))
    print("allocation (version 2) OK")


def test_theme():
    assert T.chf(3_289_732) == "CHF 3.29 m" and T.chf(493_460) == "CHF 493k" and T.chf(1250) == "CHF 1'250"
    assert T.chf(-2500, short=False) == "−CHF 2'500" and T.pct(0.0291, 1) == "2.9%"
    assert "\n\n" not in T.css()                       # a blank line would end the HTML block in Markdown
    sheet = {"H": 3e6, "P": 5e4, "F": 1e5, "W": 3.15e6}
    bar = T.wealth(sheet)
    assert T.asset_color("STAKE") == T.SECTOR and T.asset_name("STAKE") == "Your employer shares"
    assert bar.count("flex:") == 3 and "CHF 3.00 m" in bar and "CHF 3.15 m" in bar
    assert "Typical" in T.bar_row("Typical", {"EQ_WORLD": 0.6, "BOND_CHF": 0.4}, labels=False)
    bar = T.group_bar("Our plan", {"equities": 0.66, "real": 0.24, "bonds": 0.10, "cash": 0.0})
    assert bar.count("flex:") == 3 and "66%" in bar and "10%" in bar          # three groups, no cash
    assert set(T.PALETTES["dark"]) == set(T.PALETTES["light"])       # every colour exists in both modes
    for mode in ("light", "dark"):
        T.use(mode)
        assert T.streamlit_theme()["base"] == mode and T.TEXT == T.PALETTES[mode]["TEXT"]
        assert "\n\n" not in T.css() and T.asset_color("IND_MONEY") in T.PALETTES[mode]["INDUSTRY_COLORS"]
    print("theme OK")


def test_typical_and_fewer_bonds():
    """Typical advice is the profile's model portfolio (balanced 60/40); our plan starts from the same
    bonds; (M) has the bonds of our plan but keeps the industry."""
    data = AL.load_data()
    for persona in ("banker", "civil_servant"):
        for profile in ("prudent", "balanced", "dynamic"):
            p = AL.client_params(persona, {"profile": profile})
            r = AL.analyze(p, data)
            gA, gM, gC = (r["plans"][k]["groups"] for k in "AMC")
            safe = p["allocation"]["profiles"][profile]["safe_share"]
            assert abs(gA["bonds"] - safe) < 1e-9, (persona, profile, gA["bonds"])
            assert abs(r["plans"]["C"]["detail"]["safe_share"] - safe * (1 - r["sheet"]["bond_like"])) < 1e-12
            assert abs(gM["bonds"] - gC["bonds"]) < 1e-9
            assert gC["bonds"] < gA["bonds"]
            for k in "AMC":
                w = r["plans"][k]["weights"]
                assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all(), k
    r = AL.analyze(AL.client_params("banker"), data)
    wM, wC = r["plans"]["M"]["weights"], r["plans"]["C"]["weights"]
    assert r["plans"]["C"]["detail"]["excluded"] == ["IND_MONEY"] and r["plans"]["M"]["detail"]["excluded"] == []
    assert abs(wC["IND_MONEY"] - 0.15) < 1e-9 and wM["IND_MONEY"] > 0.15     # (M) buys finance on top of the stake
    sheet = r["sheet"]
    assert abs(r["plans"]["C"]["detail"]["safe_share"] - 0.40 * (1 - sheet["bond_like"])) < 1e-12
    assert load_params()["allocation"]["profiles"]["balanced"]["safe_share"] == 0.40     # balanced = 60/40
    print("typical advice = model portfolio of the profile, fewer bonds only OK")


def test_story_backtest():
    """The two steps add up to the total gain (all measured with the wealth of typical advice)."""
    s = AL.backtest_story(AL.client_params("banker"))
    g = s["gains"]["CE gain (CHF/yr)"]
    assert abs(g["(M) vs (A)"] + g["(C) vs (M)"] - g["(C) vs (A)"]) < 1e-6 * abs(g["(C) vs (A)"])
    assert set(s["metrics"].columns) == {"A", "B", "M", "C"} and set(s["values"].columns) == {"A", "B", "M", "C"}
    assert g["(M) vs (A)"] > 0
    print(f"story backtest OK: fewer bonds {g['(M) vs (A)']:+,.0f}, no own industry {g['(C) vs (M)']:+,.0f}, "
          f"total {g['(C) vs (A)']:+,.0f} CHF a year")


def test_app_headless():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("streamlit not installed, app test skipped")
        return
    base = load_params()
    for name in base["personas"]:                             # saved first: the page then never waits
        D.remember("story", AL.client_params(name, None, base))
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240).run()
    assert not at.exception, at.exception
    for name in base["personas"]:                             # every example client
        at.selectbox(key="job").set_value(name).run()
        assert not at.exception, (name, at.exception)
    at.selectbox(key="job").set_value("banker").run()
    field = [n for n in at.number_input if n.key and n.key.startswith("pension_banker_")]
    p0 = AL.client_params("banker", None, base)
    estimate = round(O.pension_savings(p0["client"]["age"], p0["client"]["salary_chf"], p0), -3)
    assert len(field) == 1 and field[0].value == estimate        # the estimate is the starting value of the field
    field[0].set_value(estimate + 400_000).run()                 # the client corrects it from the certificate
    assert not at.exception, at.exception
    assert [n.value for n in at.number_input if n.key == field[0].key] == [estimate + 400_000]
    at.number_input(key="age_banker").set_value(62).run()
    at.number_input(key="savings_banker").set_value(2_000_000).run()
    assert not at.exception, at.exception
    at.toggle(key="view_on_banker").set_value(True).run()     # a market view: used today, not in the backtest
    assert not at.exception, at.exception
    at.selectbox(key="industry_banker").set_value(None).run()
    at.toggle(key="self_banker").set_value(True).run()
    assert not at.exception, at.exception
    at.toggle(key="light").set_value(True).run()              # light page, then back to dark
    assert not at.exception, at.exception
    assert T.MODE == "light" and at.session_state["light_mode"]
    at.toggle(key="light").set_value(False).run()
    assert not at.exception and T.MODE == "dark", at.exception
    print("the app runs for every persona and other settings, with a view, in light and dark, OK")


def _page_text(at, wait_seconds=90):
    """All the text of the page as one string; reruns while the backtest is still being replayed."""
    import html as H
    import re
    import time
    for _ in range(int(wait_seconds / 0.5)):
        text = re.sub(r"\s+", " ", H.unescape(" ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown)))
        if "Replaying the past" not in text and "Running the backtest" not in text:
            return text
        time.sleep(0.5)
        at.run()
    raise AssertionError("the backtest of the page did not finish")


def test_page_texts():
    """The sentences of the page agree with the numbers, also in the edge cases found by a stress test:
    a bond-like share above 100%, cash from the loss limit for a young client, a replay that loses, and
    employer shares without a bonus."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("streamlit not installed, page text test skipped")
        return
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240).run()
    assert not at.exception, at.exception
    text = _page_text(at)
    assert "74% of what you own already behaves like a bond" in text and "To cap a bad year" not in text
    assert "better once risk is counted" in text and "already hits your bonus and your employer shares" in text
    at.slider(key="variable_banker").set_value(0.0).run()            # employer shares, no bonus
    text = _page_text(at)
    assert "already hits your employer shares." in text and "your bonus" not in text
    at.number_input(key="age_banker").set_value(25).run()            # a long career: beta_bond above 1
    text = _page_text(at)
    assert "more bonds than everything you own" in text and "of what you own already behaves" not in text
    at.slider(key="variable_banker").set_value(0.3).run()
    at.number_input(key="age_banker").set_value(62).run()
    at.number_input(key="savings_banker").set_value(2_000_000).run()  # the loss limit binds; the replay loses
    text = _page_text(at)
    assert not at.exception, at.exception
    assert "To cap a bad year, 3% goes to cash and the rest shrinks in proportion." in text
    assert "A cap on a bad year." in text and "Close to retirement" not in text
    assert "worse once risk is counted; it helped in" in text and "better once risk is counted" not in text
    print("page texts OK")


if __name__ == "__main__":
    test_client_params()
    test_analysis_and_lifecycle()
    test_plan_numbers()
    test_saved_results()
    test_allocation()
    test_theme()
    test_typical_and_fewer_bonds()
    test_story_backtest()
    test_app_headless()
    test_page_texts()
    print("ALL TESTS PASSED")
