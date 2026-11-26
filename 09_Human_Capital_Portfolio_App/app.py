"""Beyond the Portfolio: the Streamlit app (step 5, version 2 of the advice).

Run from the project folder:
    streamlit run app.py

One page. The client is in the side panel: industry, age, pay, savings, employer shares and when
they can be sold, risk profile. The answer fits on one screen: the plan in CHF and why, the
backtest since 2005, the two changes against typical advice and what each was worth, and the plan by
age with the range of what the savings might be worth at retirement. The tabs below hold the details: holdings, risk and budgets, what you give up, assumptions. Everything updates by itself; the backtest runs in the background.
The numbers come from src/allocation.py (risk budgeting, LN5-LN6) and steps 1 to 4; the look is in
src/theme.py.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit import config

from src import allocation as A
from src import dashboard as D
from src import optimizer as O
from src import theme as T
from src.config import load_params
from src.jobs import Jobs

T.use("light" if st.session_state.get("light_mode", False) else "dark")   # dark unless the switch top right is on


def keep_light():
    """The switch is kept in its own key: on older Streamlit the rerun below would lose the widget's value."""
    st.session_state["light_mode"] = st.session_state["light"]


st.set_page_config(page_title="Beyond the Portfolio", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")
THEME = T.streamlit_theme()
if any(config.get_option(f"theme.{name}") != value for name, value in THEME.items()):
    for name, value in THEME.items():                       # Streamlit's own widgets follow the same palette
        config.set_option(f"theme.{name}", value)
    st.rerun()
st.markdown(T.css(), unsafe_allow_html=True)
CHART_CONFIG = {"displayModeBar": False, "responsive": True}
NAMES = A.STRATEGIES
COLORS = {"A": T.TYPICAL, "C": T.SAVINGS}


# =============================================================================
# Data and results (kept in memory; the backtests also on disk, data/cache/)
# =============================================================================

@st.cache_data(show_spinner=False)
def base_params():
    return load_params()


@st.cache_data(show_spinner=False)
def market_data():
    return A.load_data(base_params())


@st.cache_resource(show_spinner=False)
def jobs():
    """One queue of background jobs for the whole app."""
    return Jobs()


@st.cache_data(show_spinner=False, max_entries=64)
def analysis(key):
    persona, changes = json.loads(key)
    return A.analyze(A.client_params(persona, changes, base_params()), market_data())


@st.cache_data(show_spinner=False, max_entries=64)
def glide(key):
    """The plan by age, and the range of what the savings might be worth until retirement."""
    persona, changes = json.loads(key)
    params = A.client_params(persona, changes, base_params())
    res = analysis(key)
    table = A.glide_path(params, res["mu"], res["cov"], mu_prior=res["mu_prior"])
    return table, A.savings_range(params, table)


def background(name, params):
    """Result of D.remember(name, params): at once if it is saved, otherwise started in the background and
    None for now. The page then checks every second and reruns when it is ready (poll at the end)."""
    result = D.saved(name, params)
    if result is not None:
        return result
    future = jobs().get(D.cache_key(name, params), lambda: D.remember(name, params, market_data()), group=name)
    return future.result() if future.done() else None       # an error in the computation is raised here


def html(text):
    st.markdown(text, unsafe_allow_html=True)


def show(fig):
    """Plotly chart at full width, with the new and the old Streamlit API."""
    try:
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def choice(label, options, default, key, format_func=str):
    """One choice among a few, as a segmented switch (a radio on an old Streamlit)."""
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(label, options, default=default, key=key, format_func=format_func)
        return value if value is not None else default        # a click on the chosen one unselects it
    return st.radio(label, options, index=list(options).index(default), horizontal=True, key=key,
                    format_func=format_func)


def crisis_bands(fig, crises):
    """Shaded crises, with their names on two rows so that close crises do not overlap."""
    for i, (crisis, (first, last)) in enumerate(crises.items()):
        x0 = pd.Timestamp(first) - pd.offsets.MonthEnd(1)
        fig.add_vrect(x0=x0, x1=pd.Timestamp(last) + pd.offsets.MonthEnd(0), fillcolor=T.SECTOR, opacity=0.10,
                      line_width=0)
        fig.add_annotation(x=x0, y=1.0 if i % 2 == 0 else 0.9, yref="paper", text=crisis, showarrow=False,
                           xanchor="left", font=dict(size=9, color=T.FAINT))


# =============================================================================
# Side panel: the client
# =============================================================================

P0 = base_params()
APP, PERSONAS, ALLOC = P0["app"], P0["personas"], P0["allocation"]
INDUSTRIES = APP["industry_names"]
examples = list(PERSONAS)
with st.sidebar:
    html(T.brand())
    persona = st.selectbox("Start from an example", examples, format_func=lambda k: PERSONAS[k]["label"], key="job")
    p = PERSONAS[persona]
    k = f"_{persona}"                                       # each example starts from its own values
    options = list(INDUSTRIES) + [None]
    industry = st.selectbox("I work in", options, index=options.index(p.get("industry")),
                            format_func=lambda i: INDUSTRIES.get(i, APP["no_industry"]), key="industry" + k)
    c1, c2 = st.columns([0.8, 1.2])
    age = c1.number_input("Age", *APP["age_range"], int(p["age"]), 1, key="age" + k)
    # Amounts have no upper limit, except the employer shares: they are part of the savings, so at most the
    # savings. The floors stay: no negative amounts, and savings above 0 (the plan divides by them).
    salary = c2.number_input("Salary, CHF a year", min_value=0, value=int(p["salary_chf"]), step=APP["salary_step"],
                             key="salary" + k)
    savings = st.number_input("Savings to invest, CHF (employer shares included)", min_value=APP["savings_min"],
                              value=int(P0["client"]["portfolio_chf"]), step=APP["savings_step"], key="savings" + k)
    stake, unlock, variable = 0.0, 0.0, 0.0
    if industry:
        stake = st.number_input("Of which shares of my employer, CHF", min_value=0, max_value=int(savings),
                                value=min(int(p.get("stake_chf", 0)), int(savings)), step=APP["shares_step"],
                                key="stake" + k)
        if stake > 0:
            unlock = st.slider("They can be sold in (years)", 0, APP["unlock_max_years"], int(p.get("unlock_years", 0)),
                               key="unlock" + k)
        variable = st.slider("Part of my pay that depends on my industry (bonus)", 0.0, 1.0,
                             float(p.get("variable_pay", 0.0)), APP["variable_pay_step"], key="variable" + k,
                             format="%.2f")
    self_employed = st.toggle("I am self-employed (no pension fund)", value=not p.get("pillar2", True),
                              key="self" + k)
    pension = None
    if not self_employed:                                  # estimated from age and salary; the client can correct it
        estimate = O.pension_savings(age, salary, A.client_params(persona, None, P0))
        pension = st.number_input("Pension fund savings, CHF", min_value=0, value=int(round(estimate, -3)),
                                  step=APP["pension_step"], key=f"pension{k}_{age}_{salary}",
                                  help="Our estimate from your age and salary; put the amount on your certificate.")
    profiles = list(ALLOC["profiles"])
    profile = choice("Risk profile", profiles, ALLOC["default_profile"], "profile" + k, format_func=str.capitalize)
    with st.expander("Advanced"):
        view = None
        if st.toggle("Add my own view on one market", key="view_on" + k):
            asset = st.selectbox("Market", A.assets(P0), format_func=T.asset_name, key="vasset" + k)
            low, high, step = APP["view_return_range"]
            q = st.slider("Expected return above cash, a year", low, high, APP["view_default"], step,
                          key="vq" + k, format="%.3f")
            levels = list(APP["view_confidence"])
            confidence = choice("Confidence", levels, levels[len(levels) // 2], "vc" + k)
            view = {"asset": asset, "q": q, "sd": APP["view_confidence"][confidence]}
            st.caption(f"A view is about the next years: it fades out over {ALLOC['view_years']} years in the "
                       "plan by age, and the backtest does not use it.")
    html('<p class="btp-side-note">Everything updates as you type. The backtest takes a few seconds the first '
         "time for a new client.</p>")

changes = {"industry": industry, "age": age, "salary_chf": salary, "portfolio_chf": savings,
           "stake_chf": float(stake), "unlock_years": float(unlock), "variable_pay": float(variable),
           "pillar2": not self_employed, "pension_chf": pension, "profile": profile, "view": view}
key = json.dumps([persona, changes], sort_keys=True)
try:
    res = analysis(key)
except Exception as error:
    st.error(f"No plan for these settings ({error}). Change the savings or the employer shares.")
    st.stop()

params, sheet = res["params"], res["sheet"]
client = params["client"]
F, W = sheet["F"], sheet["W"]
plans = res["plans"]
C_, A_ = plans["C"], plans["A"]
wC, wA = C_["weights"], A_["weights"]
gC, gA = C_["groups"], A_["groups"]
retire = client["retirement_age"]
ind_name = INDUSTRIES.get(industry, "") if industry else None
bt = background("story", A.without_client_view(params))      # the backtest never uses the client's view
excluded = bool(C_["detail"]["excluded"])
ind_short = ind_name.split(" (")[0].lower() if ind_name else ""


def p_text(p):
    return "p < 0.01" if p < 0.01 else f"p = {p:.2f}"
first_month = market_data()[0].index[int(params["backtest"]["window_months"])]

# =============================================================================
# Header
# =============================================================================
who, switch = st.columns([6, 1], vertical_alignment="center")
with who:
    html(f'<div class="btp-who">{T.esc(ind_name or APP["no_industry"])}, {age} years old, '
         f'{T.chf(salary, short=False)} a year {T.status(True, "Updating the backtest") if bt is None else ""}</div>')
with switch.container(key="theme_switch"):
    st.toggle("Light", value=T.MODE == "light", key="light", on_change=keep_light, help="Light or dark page")
parts = [f"{T.pct(gC['equities'])} in shares", f"{T.pct(gC['real'])} in real assets",
         f"{T.pct(gC['bonds'])} in bonds"]
if gC["cash"] >= APP["min_weight"] * 10:
    parts.append(f"{T.pct(gC['cash'])} in cash")
mix = ", ".join(parts[:-1]) + " and " + parts[-1]
html(f'<div class="btp-headline">Invest your {T.chf(F, short=False)}: <em>{mix}</em>.</div>')
dC, dA = C_["detail"]["safe_share"], A_["detail"]["safe_share"]
bond_like = sheet["bond_like"]                              # above 1 for a long career ahead (beta_bond > 1)
moved = C_["detail"]["cash_from_loss_limit"]                # share of the savings the loss limit moved to cash
if bond_like < 1:
    lede = (f"{T.pct(bond_like)} of what you own already behaves like a bond (your future salary and pension), "
            f"so your savings need {T.pct(dC)} in bonds instead of the usual {T.pct(dA)}")
else:
    lede = ("Your future salary and pension already behave like more bonds than everything you own, so your "
            f"savings need no bonds instead of the usual {T.pct(dA)}")
lede += f", and we leave out {ind_name.lower()}." if C_["detail"]["excluded"] else "."
if moved >= APP["min_weight"] * 10:                         # the cash the headline shows: say where it comes from
    lede += f" To cap a bad year, {T.pct(moved)} goes to cash and the rest shrinks in proportion."
html(f'<p class="btp-lede">{lede}</p>')
html(T.wealth({"H": sheet["H"], "P": sheet["P"], "F": F, "W": W}))

# =============================================================================
# The four panels
# =============================================================================
names = A.assets(params)
order = ["STAKE"] + [a for a in names if max(wC.get(a, 0), wA.get(a, 0)) >= APP["min_weight"]] + ["CASH"]


def with_stake(weights, detail):
    """The weights with the locked employer shares shown apart (in rose), not inside their industry."""
    w = weights.copy()
    w["STAKE"] = 0.0
    if detail["locked"] and industry:
        share = detail["stake"] / F
        w[industry] -= share
        w["STAKE"] = share
    return w.reindex(order).fillna(0).clip(lower=0)



left, right = st.columns([1, 1.25], gap="small")
with left:
    with st.container(key="panel_plan"):
        html(T.panel_title("Your portfolio", T.chf(F, short=False)))
        sC, sA = with_stake(wC, C_["detail"]), with_stake(wA, A_["detail"])
        html('<div class="btp-bars">' + T.group_bar("Our plan", gC, "ours", APP["min_weight"])
             + T.group_bar("Typical", gA, "typical", APP["min_weight"]) + "</div>")
        rows = []
        if sC["STAKE"] > 0:
            rows.append((T.SECTOR, "Your employer shares", T.chf(sC["STAKE"] * F, short=False), T.pct(sC["STAKE"])))
        industries = [a for a in ALLOC["groups"]["equities"] if a.startswith("IND_") and sC.get(a, 0) >= APP["min_weight"]]
        shares = float(sC.reindex(ALLOC["groups"]["equities"]).fillna(0).sum())
        rows.append((T.SAVINGS, f"Shares in {len(industries)} industries and Switzerland", T.chf(shares * F, short=False),
                     T.pct(shares)))
        for g, label in (("real", "Real estate, gold, commodities"), ("bonds", "Bonds: Swiss, global, corporate")):
            if gC[g] >= APP["min_weight"]:
                rows.append((T.GROUP_COLORS[g], label, T.chf(gC[g] * F, short=False), T.pct(gC[g])))
        if gC["cash"] >= APP["min_weight"]:
            rows.append((T.FAINT, "Cash", T.chf(gC["cash"] * F, short=False), T.pct(gC["cash"])))
        html(T.holdings(rows))
        why = []
        if C_["detail"]["excluded"]:
            reason = []
            if client["variable_pay"] > 0:
                reason.append(f"{T.pct(client['variable_pay'])} of your pay")
            if client["stake_chf"] > 0:
                reason.append("your employer shares")
            why.append((T.SECTOR, f"<b>No {ind_name.lower()} in the rest.</b> {' and '.join(reason).capitalize()} "
                                  "already depend on it: a bad year there would hit you twice."))
        if C_["detail"]["locked"]:
            why.append((T.SECTOR, f"<b>Your employer shares stay {unlock:.0f} more year{'s' if unlock != 1 else ''}.</b> "
                                  "The rest is built around them; then they join the plan."))
        if bond_like < 1:
            why.append((T.BONDS, f"<b>Your job works like a bond.</b> {T.pct(bond_like)} of what you own does, so "
                                 f"{T.pct(dA)} × (1 − {T.pct(bond_like)}) = {T.pct(dC)} in bonds, not {T.pct(dA)}."))
        else:
            why.append((T.BONDS, "<b>Your job works like a bond.</b> Your future salary and pension count as more "
                                 f"bonds than everything you own, so no bonds in the savings, not {T.pct(dA)}."))
        if moved > 0:                                       # true at any age, not only close to retirement
            why.append((T.SAVINGS, f"<b>A cap on a bad year.</b> In a bad year (1 in 20) this mix could lose more "
                                   f"than your {profile} profile accepts plus the "
                                   f"{T.chf(C_['detail']['savings_left'])} you can still save before retiring, so "
                                   f"{T.pct(moved, 0 if moved >= 0.01 else 1)} goes to cash."))
        else:
            why.append((T.SAVINGS, f"<b>A bad year (1 in 20) costs about {T.chf(C_['detail']['loss'])}.</b> "
                                   f"Within your {profile} profile, with {T.chf(C_['detail']['savings_left'])} still "
                                   "to save before retiring."))
        html(T.why(why))

with right:
    with st.container(key="panel_backtest"):
        html(T.panel_title(f"Since {first_month:%B %Y}", "you then, our plan against typical advice"))
        if bt is None:
            html(T.waiting("Replaying the past for this client..."))
        else:
            values = bt["values"]
            fig = go.Figure()
            crisis_bands(fig, params["backtest"]["crises"])
            for s, width in (("A", 1.8), ("C", 2.6)):
                fig.add_trace(go.Scatter(x=values.index, y=values[s], name=NAMES[s], mode="lines",
                                         line=dict(color=COLORS[s], width=width),
                                         hovertemplate=NAMES[s] + ": CHF %{y:,.0f}<extra></extra>"))
            fig.update_layout(**T.plot_layout(214, hovermode="x unified", showlegend=False))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
            show(fig)
            m = bt["metrics"]
            g = bt["gains"].loc["(C) vs (A)"]
            gain, p_gain = float(g["CE gain (CHF/yr)"]), float(g["p-value"])
            verdict = (f"better once risk is counted, {p_text(p_gain)}" if gain >= 0 else
                       f"worse once risk is counted; it helped in {T.helped(p_gain)}")
            html(T.stats([(T.chf(values["C"].iloc[-1]), f"today with our plan, against {T.chf(values['A'].iloc[-1])} "
                           "with typical advice", T.SAVINGS),
                          (f"{T.chf(abs(gain))} a year", verdict, None),
                          (T.pct(m.loc["portfolio max drawdown", "C"]), "worst fall, against "
                           f"{T.pct(m.loc['portfolio max drawdown', 'A'])} with typical advice", None)]))
            html(f'<p class="btp-panel-note">Someone like you, {bt["start_age"]:.0f} in {first_month:%B %Y}, saving '
                 f"{T.pct(params['backtest']['savings_rate'])} of the salary; each decision uses only the past "
                 f"{params['backtest']['window_months']} months, with trading costs.</p>")

with st.container(key="panel_changes"):
    html(T.panel_title("The two changes, and what each was worth",
                       f"since {first_month:%B %Y}, CHF a year once risk is counted"))
    if bt is None:
        html(T.waiting("Replaying the past for this client..."))
    else:
        gains = bt["gains"]
        g1, g2, gt = (gains.loc[i] for i in ("(M) vs (A)", "(C) vs (M)", "(C) vs (A)"))
        v1, v2, vt = (float(x["CE gain (CHF/yr)"]) for x in (g1, g2, gt))
        chart_col, steps_col = st.columns([1, 1.2], gap="large")
        with chart_col:
            mix = f"{1 - dA:.0%}/{dA:.0%}".replace("%", "")
            labels = [f"Typical {mix}", "Fewer bonds"] + ([f"No extra {ind_short}"] if excluded else []) + ["Our plan"]
            fig = go.Figure(go.Waterfall(
                x=labels, y=[0, v1] + ([v2] if excluded else []) + [0],
                measure=["absolute", "relative"] + (["relative"] if excluded else []) + ["total"],
                text=["0", T.signed_chf(v1)] + ([T.signed_chf(v2)] if excluded else []) + [T.signed_chf(vt)],
                textposition="outside", textfont=dict(color=T.TEXT, size=12), cliponaxis=False,
                connector=dict(line=dict(color=T.LINE, width=1)),
                increasing=dict(marker=dict(color=T.GOOD)), decreasing=dict(marker=dict(color=T.SECTOR)),
                totals=dict(marker=dict(color=T.SAVINGS)), hovertemplate="%{x}: %{text} a year<extra></extra>"))
            levels = [0.0, v1] + ([v1 + v2] if excluded else []) + [vt]   # where each bar starts or ends
            low, high = min(levels), max(levels)
            pad = 0.3 * ((high - low) or 1.0)
            fig.update_layout(**T.plot_layout(230, showlegend=False))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s", range=[low - pad, high + pad])
            show(fig)
        with steps_col:
            html(T.change(1, "Fewer bonds: your job is a bond",
                          f"Bonds go from {T.pct(dA)} to {T.pct(dC)}; the money goes to shares and real assets, "
                          "which earn more over time.", T.signed_chf(v1), T.GOOD if v1 >= 0 else T.SECTOR,
                          g1["p-value"] <= APP["significance"], T.helped(g1["p-value"])))
            if excluded:                                    # left out because of a bonus, employer shares, or both
                hits = [what for what, on in (("your bonus", client["variable_pay"] > 0),
                                              ("your employer shares", client["stake_chf"] > 0)) if on]
                html(T.change(2, f"No extra {ind_short}: no second bet on your industry",
                              f"A bad year for {ind_short} already hits {' and '.join(hits)}. Leaving it out lowers the risk "
                              "in your industry's worst months: it is insurance, not a source of profit"
                              + (f", and here its premium is {T.chf(-v2, short=False)} a year." if v2 < 0 else
                                 ". Here it also happened to earn a little."),
                              T.signed_chf(v2), T.GOOD if v2 >= 0 else T.SECTOR,
                              g2["p-value"] <= APP["significance"], T.helped(g2["p-value"])))
            html(f'<p class="btp-panel-note">Certainty equivalent on total wealth (gain in return minus the cost of '
                 f"the extra risk), the same wealth for every strategy, "
                 f"{params['backtest']['transaction_cost_bp']} bp trading costs, block bootstrap. "
                 f"Total {T.signed_chf(vt)} a year, {p_text(gt['p-value'])}.</p>")

with st.container(key="panel_glide"):
    gl, (fan, pension_end) = glide(key)
    gl = gl.copy()
    html(T.panel_title("As you get older", f"our plan until you retire at {retire:.0f}"))
    mix_tab, range_tab = st.tabs(["Your mix", "What you might have"])
    with mix_tab:
        if len(gl) > 1:
            gl.loc[retire] = gl.iloc[-1]
            fig = go.Figure()
            for grp in ("equities", "real", "bonds", "cash"):
                if gl[("C", grp)].max() > APP["min_weight"]:
                    fig.add_trace(go.Scatter(x=gl.index, y=gl[("C", grp)], name=T.GROUP_NAMES[grp],
                                             stackgroup="mix", mode="lines",
                                             line=dict(width=0, color=T.GROUP_COLORS[grp]),
                                             fillcolor=T.GROUP_COLORS[grp],
                                             hovertemplate=T.GROUP_NAMES[grp] + ": %{y:.0%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=gl.index, y=1 - gl[("A", "bonds")] - gl[("A", "cash")],
                                     name="Typical advice, not bonds", mode="lines",
                                     line=dict(color=T.TEXT, width=1.5, dash="dot"),
                                     hovertemplate="Typical advice: %{y:.0%} outside bonds<extra></extra>"))
            fig.update_layout(**T.plot_layout(190, hovermode="x unified", legend_traceorder="normal"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
            show(fig)
        html(f'<p class="btp-panel-note">Our plan solved again every {APP["glide_step_years"]} years with the '
             f"wealth you are expected to have then. Every year a salary is paid out, so less of your wealth "
             f"behaves like a bond and the savings hold more bonds; near retirement the loss limit adds cash.</p>")
    with range_tab:
        if len(fan) > 1:
            end = fan.iloc[-1]
            html(T.stats([(T.chf(end[0.5]), f"your savings at {retire:.0f}, the middle outcome", T.SAVINGS),
                          (f"{T.chf(end[0.25])} to {T.chf(end[0.75])}", "where half of the outcomes land", None),
                          (T.chf(end[0.05]), "a bad case: 1 in 20 ends lower", None)]))
            fig = go.Figure()
            for low, high, alpha in ((0.05, 0.95, 0.16), (0.25, 0.75, 0.30)):     # light band 9 in 10, dark band half
                fig.add_trace(go.Scatter(x=fan.index, y=fan[low], mode="lines", line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=fan.index, y=fan[high], mode="lines", line=dict(width=0), fill="tonexty",
                                         fillcolor=T.tint(T.SAVINGS, alpha), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=fan.index, y=fan[0.5], mode="lines", line=dict(color=T.SAVINGS, width=2), showlegend=False,
                customdata=fan[[0.05, 0.25, 0.75, 0.95]].to_numpy(),
                hovertemplate="Middle: CHF %{y:,.0f}<br>Half between CHF %{customdata[1]:,.0f} and "
                              "%{customdata[2]:,.0f}<br>9 in 10 between CHF %{customdata[0]:,.0f} and "
                              "%{customdata[3]:,.0f}<extra></extra>"))
            fig.update_layout(**T.plot_layout(210, hovermode="x unified", margin=dict(l=4, r=8, t=12, b=4)))
            fig.update_xaxes(hoverformat=".0f")
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s", rangemode="tozero")
            show(fig)
            plus = (f" On top, your pension fund should hold about {T.chf(pension_end)} at {retire:.0f}."
                    if pension_end > 0 else "")
            markets = f"{APP['range_paths']:,}".replace(",", "'")
            html(f'<p class="btp-panel-note">{markets} possible markets. Each year your savings earn a random return, '
                 f"with the expected return and the swings of our plan at that age, and you add "
                 f"{T.pct(params['backtest']['savings_rate'])} of your salary. The line is the middle outcome, the "
                 f"dark band holds half of the outcomes, the light band 9 in 10. In francs of the day, not adjusted "
                 f"for inflation.{plus}</p>")

# =============================================================================
# Details
# =============================================================================
tabs = st.tabs(["Holdings", "Risk and budgets", "Backtest", "Assumptions"])

with tabs[0]:
    html('<div class="btp-h3">Every holding, in CHF</div><p class="btp-p">Our plan and typical advice (it knows '
         "neither your job nor your shares).</p>")
    table = pd.DataFrame({NAMES[s]: plans[s]["weights"] * F for s in ("C", "A")})
    table = table[(table.abs() > 1).any(axis=1)]
    table.index = [T.asset_name(a) for a in table.index]
    html(T.table(table, {c: (lambda v: T.chf(v, short=False)) for c in table.columns}))

with tabs[1]:
    html('<div class="btp-h3">Where the risk of your savings comes from</div><p class="btp-p">'
         f"{C_['diversification']['effective bets']:.0f} independent bets in our plan against "
         f"{A_['diversification']['effective bets']:.0f} with typical advice.</p>")
    fig = go.Figure()
    for g, colour in (("stake", T.SECTOR), ("equities", T.SAVINGS), ("real", T.PENSION), ("bonds", T.BONDS)):
        values = [float(plans[s]["risk_parts"][g]) for s in ("C", "A")]
        if max(values) < APP["min_weight"]:
            continue
        label = "Employer shares" if g == "stake" else T.GROUP_NAMES[g]
        fig.add_trace(go.Bar(y=[NAMES["C"], NAMES["A"]], x=values, name=label, orientation="h", marker_color=colour,
                             hovertemplate=label + ": %{x:.0%}<extra></extra>"))
    fig.update_layout(**T.plot_layout(170, barmode="stack", legend_traceorder="normal",
                                      legend=dict(orientation="h", yanchor="top", y=-0.3, x=0,
                                                  font=dict(color=T.MUTED, size=11)),
                                      margin=dict(l=4, r=8, t=4, b=40)))
    fig.update_xaxes(tickformat=".0%", range=[0, 1], showgrid=True, gridcolor=T.LINE)
    fig.update_yaxes(autorange="reversed")
    show(fig)
    dv = C_["diversification"]
    html(f'<p class="btp-panel-note">Risk contributions: the share of the ups and downs of your savings '
         f"that comes from each part. Volatility of our plan {T.pct(dv['volatility'], 1)} a year.</p>")
    html('<div class="btp-h3">How the plan is built</div><p class="btp-p">Risk budgeting: every asset we buy '
         "carries a share of the risk equal to its budget. Between groups the budgets come from your risk "
         "profile; inside a group they follow the expected return per unit of risk (Black-Litterman returns, "
         "starting from the market weights), with a floor so no asset is dropped. Covariance with Ledoit-Wolf "
         "shrinkage.</p>")
    info = pd.DataFrame({"Market weight (Black-Litterman prior)": res["estimates"]["market_weights"],
                         "Expected return above cash": res["mu"], "Risk budget, our plan": C_["detail"]["budgets"],
                         "Risk contribution, our plan": C_["risk"]})
    info.index = [T.asset_name(a) for a in info.index]
    html(T.table(info, {c: (lambda v: T.pct(v, 1)) for c in info.columns}))

with tabs[2]:
    if bt is None:
        html(T.waiting("Running the backtest for this client..."))
    else:
        m = bt["metrics"]
        crises = bt["crises"]["portfolio return"].unstack("strategy")
        gfc = "2008 financial crisis"
        price = [(T.pct(m.loc["portfolio volatility", "C"], 1), "your savings swing, a year; typical "
                  f"{T.pct(m.loc['portfolio volatility', 'A'], 1)}", None),
                 (T.pct(m.loc["portfolio max drawdown", "C"]), "worst fall of the savings; typical "
                  f"{T.pct(m.loc['portfolio max drawdown', 'A'])}", None)]
        if gfc in crises.index:
            price.append((T.pct(crises.loc[gfc, "C"]), f"in the 2008 crisis; typical {T.pct(crises.loc[gfc, 'A'])}", None))
        price.append((T.pct(m.loc["total wealth volatility", "C"], 1), "everything you own swings, a year; typical "
                      f"{T.pct(m.loc['total wealth volatility', 'A'], 1)}", T.SAVINGS))
        if m.loc["portfolio volatility", "C"] > m.loc["portfolio volatility", "A"]:
            paid = float(bt["gains"].loc["(C) vs (A)", "CE gain (CHF/yr)"]) >= 0
            give_up = ("<b>Your savings move more.</b> " + ("You are paid for it (the two changes), and your salary"
                       if paid else "Your salary") + ", which keeps coming in a crash, absorbs it: everything you own "
                       "swings almost as little as before.")
        else:                                               # e.g. a lot of cash from the loss limit
            give_up = "<b>Your savings do not move more than with typical advice.</b> The numbers below compare the two."
        html(f'<div class="btp-h3">What you give up</div><p class="btp-p">{give_up}</p>')
        html(T.cards(price))
        order_c = [c for c in params["backtest"]["crises"] if c in crises.index]
        crises = crises.loc[order_c, ["A", "C"]].rename(columns=NAMES)
        html('<div class="btp-h3">In the crises</div><p class="btp-p">Return of the savings in each crisis, '
             "our plan against typical advice.</p>")
        html(T.table(crises, {c: (lambda v: T.pct(v, 1)) for c in crises.columns}, signed=list(crises.columns)))

with tabs[3]:
    prof = ALLOC["profiles"][profile]
    c = params["career"]
    rows = [
        ("Risk profile", f"{profile}: bonds {T.pct(prof['safe_share'])} of the savings for someone with no career; "
                         f"growth risk {T.pct(prof['equities'])} shares, {T.pct(prof['real'])} real assets", "Our choice"),
        ("Bonds in our plan", "profile's bond share x (1 - bond-like share of wealth)", "Our model"),
        ("Bond-like share of wealth", "(bond beta of the career x career value + pension) / total wealth",
         "Our model"),
        ("Own industry", "left out when part of the pay depends on it or employer shares are held", "Our choice"),
        ("Loss limit", f"bad year = 1 in {1 / (1 - ALLOC['loss_confidence']):.0f}; accepted loss "
                       f"{T.pct(prof['loss_tolerance'])} of the savings plus what can still be saved", "Gaussian value at risk"),
        ("Employer shares", f"industry return plus {T.pct(ALLOC['stake_idio_vol'])} company-specific volatility",
         "Assumption"),
        ("Industries", "12 Fama-French industry portfolios, in CHF", "Kenneth French data library"),
        ("Prior of Black-Litterman", "Doeswijk and Swinkels (2024) shares; equities by MSCI World sectors, Sep 2026",
         "Doeswijk and Swinkels (2024); MSCI"),
        ("Your market view", f"counts fully today, fades out over {ALLOC['view_years']} years in the plan by age, "
                             "not used in the backtest", "Our choice"),
        ("Salary growth", T.pct(c["salary_growth"], 1) + " a year", "Assumption"),
        ("Savings", f"{T.pct(params['backtest']['savings_rate'])} of the gross salary", "Assumption"),
        ("Trading costs", f"{params['backtest']['transaction_cost_bp']} bp of turnover", "Our choice"),
        ("Rebalancing (backtest)", f"plan solved every {ALLOC['rebalance_months']} months, "
                                   f"window {params['backtest']['window_months']} months", "Our choice"),
    ]
    html(T.table(pd.DataFrame(rows, columns=["Assumption", "Value", "Source"]).set_index("Assumption")))
    html('<p class="btp-foot">Every value lives in config/params.yaml. Data: Yahoo Finance, Kenneth French, FRED and '
         "the Swiss National Bank, monthly in CHF. Beyond the Portfolio is a student project (HEC Lausanne, "
         "Quantitative Asset and Risk Management II), not financial advice.</p>")


# =============================================================================
# While a backtest runs in the background: check every second, then show it
# =============================================================================
@st.fragment(run_every=APP["poll_seconds"])
def poll():
    if not jobs().pending():
        st.rerun()


if jobs().pending():
    poll()
