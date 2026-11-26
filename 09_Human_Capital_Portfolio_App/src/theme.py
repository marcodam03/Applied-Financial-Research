"""Look of the app (step 5): colours, font, CSS and small HTML pieces.

The idea: a dark, quiet instrument, like the evening view of a private bank. Graphite
panels, one family of text (Manrope, numbers with tabular figures), and muted colours that
keep the same meaning everywhere: indigo for the career, sage for the pension, champagne for
the savings and for our advice, steel blue for bonds, rose for the client's own sector.
Typical advice is always grey. Everything important fits on one screen.
No Streamlit here: the functions return strings and Plotly settings.
"""
import html

import numpy as np

# ---------------------------------------------------------------------------
# Palettes: dark (default) and light, same meaning for every colour. use(mode) switches them.
# ---------------------------------------------------------------------------
PALETTES = {
    "dark": {
        "BG": "#0E1014", "PANEL": "#15181E", "RAISED": "#1C2028", "LINE": "#262B35", "LINE2": "#1D2129",
        "SIDE": "#111318", "TEXT": "#ECEEF2", "MUTED": "#8B92A1", "FAINT": "#5A6170",
        "CAREER": "#8E9BFF", "PENSION": "#5CC2A7", "SAVINGS": "#D9B46C", "TYPICAL": "#6B7385",
        "BONDS": "#6FA8DC", "SECTOR": "#E46F7F", "GOOD": "#4FC08D", "ON_BAR": "#0E1014",
        "EQ_CH": "#E59A5C", "BOND_GLB_H": "#9CC3E8", "GOLD": "#BFA15A", "EMPLOYER": "#A9876A",
        "BOND_CORP": "#4F7FB0", "COMMOD": "#8FBF9F", "STRATEGY_B": "#A3AEC2",
        "INDUSTRY_COLORS": ["#F2D49B", "#E9C27F", "#DDB066", "#D19E52", "#C48D43", "#B67D38",
                            "#EBC98C", "#DFB873", "#D3A65E", "#C6954C", "#B8853E", "#A97634"],
    },
    "light": {
        "BG": "#F5F6F8", "PANEL": "#FFFFFF", "RAISED": "#ECEEF2", "LINE": "#E1E4EA", "LINE2": "#ECEEF2",
        "SIDE": "#FBFBFC", "TEXT": "#15181E", "MUTED": "#5E6575", "FAINT": "#9AA1AE",
        "CAREER": "#5B6CF0", "PENSION": "#2E9C80", "SAVINGS": "#B8862F", "TYPICAL": "#8A91A0",
        "BONDS": "#3D7CC0", "SECTOR": "#D04A5F", "GOOD": "#2E9C6A", "ON_BAR": "#FFFFFF",
        "EQ_CH": "#C9733A", "BOND_GLB_H": "#7EAEDC", "GOLD": "#A88A3E", "EMPLOYER": "#8C6A4E",
        "BOND_CORP": "#2F5F96", "COMMOD": "#6FA884", "STRATEGY_B": "#7D879A",
        "INDUSTRY_COLORS": ["#E3C27E", "#D8AF66", "#CC9C52", "#BF8A42", "#B17A36", "#A36C2D",
                            "#DDB874", "#D1A55E", "#C4934B", "#B7823E", "#A97333", "#9A652A"],
    },
}
INDUSTRIES = ["IND_NODUR", "IND_DURBL", "IND_MANUF", "IND_ENRGY", "IND_CHEMS", "IND_BUSEQ",
              "IND_TELCM", "IND_UTILS", "IND_SHOPS", "IND_HLTH", "IND_MONEY", "IND_OTHER"]
GROUP_NAMES = {"equities": "Shares", "real": "Real assets", "bonds": "Bonds", "cash": "Cash"}
MODE = "dark"


def use(mode="dark"):
    """Switch every colour of the module to the dark or light palette (call it first in each run)."""
    global MODE, ASSET_COLORS, GROUP_COLORS, STRATEGY_COLORS
    MODE = mode
    p = PALETTES[mode]
    globals().update({k: v for k, v in p.items()})
    ASSET_COLORS = {"EQ_WORLD": p["SAVINGS"], "EQ_CH": p["EQ_CH"], "BANKS_EU": p["SECTOR"], "BOND_CHF": p["BONDS"],
                    "BOND_GLB_H": p["BOND_GLB_H"], "REIT_CH": p["PENSION"], "GOLD": p["GOLD"],
                    "EMPLOYER": p["EMPLOYER"], "CASH": p["FAINT"], "BOND_CORP": p["BOND_CORP"], "COMMOD": p["COMMOD"]}
    ASSET_COLORS.update(dict(zip(INDUSTRIES, p["INDUSTRY_COLORS"])))
    GROUP_COLORS = {"Shares": p["SAVINGS"], "Bonds": p["BONDS"], "Real estate and gold": p["PENSION"],
                    "Cash": p["FAINT"], "equities": p["SAVINGS"], "real": p["PENSION"], "bonds": p["BONDS"],
                    "cash": p["FAINT"]}
    STRATEGY_COLORS = {"A": p["TYPICAL"], "B": p["STRATEGY_B"], "D": p["EQ_CH"], "C": p["SAVINGS"]}


def streamlit_theme():
    """Options of Streamlit's own theme (widgets) for the current palette."""
    return {"base": MODE, "primaryColor": SAVINGS, "backgroundColor": BG, "secondaryBackgroundColor": RAISED,
            "textColor": TEXT}


use("dark")

SHORT_NAMES = {
    "EQ_WORLD": "World equities", "EQ_CH": "Swiss equities", "BANKS_EU": "European banks",
    "BOND_CHF": "Swiss gov. bonds", "BOND_GLB_H": "Global bonds, hedged", "REIT_CH": "Swiss real estate",
    "GOLD": "Gold", "EMPLOYER": "Employer shares", "CASH": "Cash", "CAREER": "Your career",
    "PENSION": "Your pension", "BOND_CORP": "Corporate bonds", "COMMOD": "Commodities",
    "IND_NODUR": "Food and household", "IND_DURBL": "Cars and appliances", "IND_MANUF": "Manufacturing",
    "IND_ENRGY": "Energy", "IND_CHEMS": "Chemicals", "IND_BUSEQ": "Technology", "IND_TELCM": "Telecom",
    "IND_UTILS": "Utilities", "IND_SHOPS": "Retail", "IND_HLTH": "Health care", "IND_MONEY": "Finance",
    "IND_OTHER": "Construction and other", "STAKE": "Your employer shares",
}

FONTS = "https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap"
FONT = "'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif"


def asset_name(asset, params=None):
    """Short readable name of a column (sector menu names come from params)."""
    if asset in SHORT_NAMES:
        return SHORT_NAMES[asset]
    if params is not None and asset in params["data"].get("sectors", {}):
        return params["data"]["sectors"][asset]["label"].split(" (")[0]
    return asset


def asset_color(asset):
    if asset == "STAKE":
        return SECTOR
    return ASSET_COLORS.get(asset, SECTOR if str(asset).startswith("SEC_") else TYPICAL)


# ---------------------------------------------------------------------------
# Number formats
# ---------------------------------------------------------------------------

def chf(x, short=True):
    """CHF amount: 3.44 m, 493k, 1'250."""
    x = float(x)
    sign = "−" if x < 0 else ""
    a = abs(x)
    if short and a >= 999_500:
        return f"{sign}CHF {a / 1e6:.2f} m"
    if short and a >= 10_000:
        return f"{sign}CHF {a / 1e3:.0f}k"
    return f"{sign}CHF {a:,.0f}".replace(",", "'")


def pct(x, digits=0):
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-digits - 2):
        x = 0.0
    return f"{x:.{digits}%}".replace("-", "−")


def esc(text):
    return html.escape(str(text))


def tint(color, alpha):
    """A #RRGGBB colour as rgba with this opacity (shaded areas of a chart)."""
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alpha})"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def css():
    text = f"""<style>
@import url('{FONTS}');
:root {{ --bg:{BG}; --panel:{PANEL}; --raised:{RAISED}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
        --faint:{FAINT}; --career:{CAREER}; --pension:{PENSION}; --savings:{SAVINGS}; --typical:{TYPICAL};
        --bonds:{BONDS}; --sector:{SECTOR}; --good:{GOOD}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg); color: var(--text);
        font-family: {FONT}; }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea, .stApp table,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-baseweb="select"] div,
[role="tab"] {{ font-family: {FONT}; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding: 3.8rem 2rem 3rem; max-width: 1500px; }}   /* top: room for the toolbar of Streamlit Cloud */
.stApp .num, .btp-table td {{ font-variant-numeric: tabular-nums; }}
/* side panel: the client */
[data-testid="stSidebar"] {{ background: {SIDE}; border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] {{ padding-top: 1.2rem; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: var(--muted); font-size: .82rem; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 14px;
        background: transparent; }}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{ border: none; }}
.btp-brand {{ font-weight: 800; font-size: 1.12rem; letter-spacing: -0.01em; display: flex; align-items: center;
        gap: .6rem; margin: 0 0 1.3rem; }}
.btp-brand i {{ display: inline-block; width: 22px; height: 22px; border-radius: 50%;
        background: conic-gradient(var(--career) 0 70%, var(--pension) 70% 84%, var(--savings) 84% 100%); }}
.stApp .btp-side-note {{ color: var(--faint); font-size: .78rem; line-height: 1.45; margin: 1rem 0 0; }}
/* header */
.stApp .st-key-theme_switch {{ display: flex; justify-content: flex-end; }}
.stApp .st-key-theme_switch label p {{ color: var(--muted); font-size: .85rem; }}
.btp-who {{ color: var(--muted); font-size: .92rem; margin: 0 0 .35rem; display: flex; gap: 14px;
        align-items: center; flex-wrap: wrap; }}
.btp-status {{ display: inline-flex; align-items: center; gap: 7px; padding: 3px 10px; border-radius: 999px;
        background: var(--raised); font-size: .78rem; color: var(--muted); }}
.btp-status i {{ width: 7px; height: 7px; border-radius: 50%; background: var(--good); display: inline-block; }}
.btp-status.busy i {{ background: var(--savings); }}
.btp-headline {{ font-weight: 700; font-size: clamp(1.5rem, 2.3vw, 2.15rem); line-height: 1.15;
        letter-spacing: -0.025em; margin: 0 0 .35rem; max-width: 46ch; }}
.btp-headline em {{ font-style: normal; color: var(--savings); }}
.stApp .btp-lede {{ color: var(--muted); font-size: 1rem; line-height: 1.5; margin: 0 0 1.1rem; max-width: 92ch; }}
.btp-lede b {{ color: var(--text); font-weight: 600; }}
/* wealth bar */
.btp-wealth {{ display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: center; margin: 0 0 1.1rem; }}
.btp-ribbon {{ display: flex; height: 14px; border-radius: 999px; overflow: hidden; gap: 3px; }}
.btp-ribbon > div {{ min-width: 4px; }}
.btp-legend {{ display: flex; flex-wrap: wrap; gap: 6px 22px; margin: 8px 0 0; font-size: .86rem; color: var(--muted); }}
.btp-legend b {{ color: var(--text); font-weight: 600; }}
.btp-total {{ text-align: right; }}
.btp-total b {{ display: block; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }}
.btp-total span {{ color: var(--muted); font-size: .8rem; }}
.btp-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }}
/* panels: bordered containers with a key starting with "panel" */
[class*="st-key-panel"] {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
        padding: 18px 20px 22px; }}
.btp-panel-title {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin: 0 0 2px; }}
.btp-panel-title b {{ font-weight: 700; font-size: 1.02rem; }}
.btp-panel-title span {{ color: var(--muted); font-size: .82rem; }}
.stApp .btp-panel-note {{ color: var(--muted); font-size: .86rem; line-height: 1.45; margin: 4px 0 0; }}
.btp-panel-note b {{ color: var(--text); font-weight: 600; }}
[data-testid="stPlotlyChart"] {{ background: transparent; }}
/* allocation bars */
.btp-bars {{ margin: 12px 0 4px; }}
.btp-barrow {{ display: grid; grid-template-columns: 92px 1fr; gap: 12px; align-items: center; margin: 0 0 9px; }}
.btp-barrow > span {{ color: var(--muted); font-size: .84rem; }}
.btp-barrow.ours > span {{ color: var(--savings); font-weight: 600; }}
.btp-bar {{ display: flex; height: 30px; border-radius: 9px; overflow: hidden; gap: 2px; }}
.btp-barrow.typical .btp-bar {{ height: 16px; border-radius: 6px; opacity: .85; }}
.btp-barrow.groups .btp-bar {{ height: 30px; border-radius: 9px; opacity: 1; }}
.btp-barrow.groups.typical .btp-bar > div {{ opacity: .7; }}
.btp-bar > div {{ display: flex; align-items: center; padding: 0 10px; font-size: .78rem; font-weight: 600;
        color: {ON_BAR}; white-space: nowrap; overflow: hidden; min-width: 0; }}
.btp-hold {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 0 22px;
        margin: 8px 0 0; }}
.btp-hold > div {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; padding: 7px 0;
        border-bottom: 1px solid var(--line); font-size: .88rem; }}
.btp-hold b {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.btp-hold small {{ color: var(--muted); margin-left: 6px; }}
.btp-why {{ margin: 12px 0 2px; display: grid; gap: 8px; }}
.stApp .btp-why > div {{ display: grid; grid-template-columns: 3px 1fr; gap: 12px; font-size: .88rem; line-height: 1.45;
        color: var(--muted); }}
.btp-why > div > i {{ border-radius: 3px; }}
.btp-why b {{ color: var(--text); font-weight: 600; }}
/* small numbers under a chart */
.btp-change {{ display: grid; grid-template-columns: 26px 1fr auto; gap: 12px; padding: 12px 0 10px;
  border-top: 1px solid var(--line); align-items: start; }}
.btp-change > i {{ font-style: normal; width: 24px; height: 24px; border-radius: 50%; background: var(--raised);
  color: var(--muted); font-size: .78rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
.btp-change b {{ color: var(--text); font-size: .92rem; }}
.btp-change p {{ color: var(--muted); font-size: .84rem; line-height: 1.45; margin: .2rem 0 .45rem; }}
.btp-change > strong {{ font-size: 1.02rem; font-weight: 700; white-space: nowrap; }}
.btp-tag {{ display: inline-block; font-size: .74rem; padding: 2px 9px; border-radius: 999px; background: var(--raised);
  color: var(--muted); }}
.btp-tag.good {{ color: var(--good); }}
.btp-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 6px 0 4px; }}
.btp-stat b {{ display: block; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.015em;
        font-variant-numeric: tabular-nums; white-space: nowrap; }}
.btp-stat span {{ color: var(--muted); font-size: .8rem; line-height: 1.35; display: block; }}
.stApp .btp-wait {{ padding: 46px 6px; color: var(--muted); font-size: .9rem; }}
.btp-wait .track {{ height: 4px; border-radius: 999px; background: var(--line); overflow: hidden; margin: 12px 0 0;
        max-width: 360px; }}
.btp-wait .track i {{ display: block; height: 100%; width: 35%; border-radius: 999px; background: var(--savings); }}
/* details below the panels */
.stTabs {{ margin-top: 1.6rem; }}
.stTabs [role="tablist"] {{ gap: 2px; border-bottom: 1px solid var(--line); }}
.stTabs [role="tablist"]::after {{ background: var(--line); }}
.stTabs [role="tab"] {{ color: var(--muted); padding: 0 12px; }}
.stTabs [role="tab"][data-selected], .stTabs [role="tab"][aria-selected="true"] {{ color: var(--text); }}
.stTabs .react-aria-SelectionIndicator {{ background: var(--savings) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--savings); }}
[class*="st-key-panel"] .stTabs {{ margin-top: .5rem; }}
.btp-h3 {{ font-weight: 700; font-size: 1.05rem; margin: 1.3rem 0 .2rem; }}
.stApp .btp-p {{ color: var(--muted); font-size: .92rem; line-height: 1.55; margin: 0 0 .9rem; max-width: 90ch; }}
.btp-p b {{ color: var(--text); font-weight: 600; }}
.btp-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin: .6rem 0 0; }}
.btp-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; }}
.btp-card b.v {{ display: block; font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em; white-space: nowrap;
        font-variant-numeric: tabular-nums; }}
.btp-card p {{ color: var(--muted); font-size: .86rem; line-height: 1.45; margin: .35rem 0 0; }}
.stApp .btp-table {{ width: 100%; border-collapse: collapse; font-size: .88rem; margin: .4rem 0 0; border: none; }}
.stApp .btp-table th, .stApp .btp-table td {{ border: none; }}
.stApp .btp-table th {{ text-align: right; font-weight: 600; color: var(--muted); padding: 9px 12px;
        border-bottom: 1px solid var(--line); }}
.stApp .btp-table td {{ text-align: right; padding: 8px 12px; border-bottom: 1px solid {LINE2}; }}
.stApp .btp-table th.txt, .stApp .btp-table td.txt {{ text-align: left; }}
.stApp .btp-table td:first-child, .stApp .btp-table td:not(.txt) {{ white-space: nowrap; }}
.btp-pos {{ color: var(--good); font-weight: 600; }}
.btp-neg {{ color: var(--sector); font-weight: 600; }}
.stApp .btp-foot {{ color: var(--faint); font-size: .78rem; margin: 2.4rem 0 0; line-height: 1.5; }}
@media (max-width: 900px) {{
  .block-container {{ padding: 3.8rem 1rem 3rem; }}
  .btp-wealth {{ grid-template-columns: 1fr; }} .btp-total {{ text-align: left; }}
  .btp-hold, .btp-stats {{ grid-template-columns: 1fr; }}
}}
@media (prefers-reduced-motion: no-preference) {{
  .btp-wait .track i {{ animation: slide 1.4s ease-in-out infinite; }}
  @keyframes slide {{ from {{ transform: translateX(-100%); }} to {{ transform: translateX(290%); }} }}
}}
</style>"""
    # no blank line: Markdown would end the HTML block there
    return "\n".join(line for line in text.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------

def brand():
    return '<div class="btp-brand"><i></i>Beyond the Portfolio</div>'


def status(busy, text):
    return f'<span class="btp-status{" busy" if busy else ""}"><i></i>{esc(text)}</span>'


def wealth(sheet):
    """Total wealth as one thin bar: career, pension, savings in proportion, with the total."""
    W = sheet["W"]
    parts = [(CAREER, sheet["H"], "Career"), (PENSION, sheet["P"], "Pension"), (SAVINGS, sheet["F"], "Savings")]
    bar = "".join(f'<div style="flex:{v / W:.5f} 1 0;background:{c}" title="{label}: {chf(v)}"></div>'
                  for c, v, label in parts if v > 0)
    legend = "".join(f'<span><i class="btp-dot" style="background:{c}"></i>{label} <b>{chf(v)}</b> '
                     f'{pct(v / W, 1)}</span>' for c, v, label in parts if v > 0)
    return (f'<div class="btp-wealth"><div><div class="btp-ribbon">{bar}</div><div class="btp-legend">{legend}</div>'
            f'</div><div class="btp-total"><b>{chf(W)}</b><span>everything you own</span></div></div>')


def panel_title(title, note=None):
    return f'<div class="btp-panel-title"><b>{esc(title)}</b>{f"<span>{esc(note)}</span>" if note else ""}</div>'


def bar_row(label, weights, params=None, cls="", min_weight=0.0005, labels=True):
    """One portfolio as a bar, one segment per holding (the label inside when it fits)."""
    cells = []
    for asset, w in weights.items():
        if w < min_weight:
            continue
        text = ""
        if labels:
            text = f"{asset_name(asset, params)} {pct(w)}" if w >= 0.3 else (pct(w) if w >= 0.08 else "")
        cells.append(f'<div style="flex:{w:.5f} 1 0;background:{asset_color(asset)}" '
                     f'title="{esc(asset_name(asset, params))}: {pct(w, 1)}">{esc(text)}</div>')
    return f'<div class="btp-barrow {cls}"><span>{esc(label)}</span><div class="btp-bar">{"".join(cells)}</div></div>'


def group_bar(label, groups, cls="", min_weight=0.0005):
    """One portfolio as a bar of its groups only (shares, real assets, bonds, cash), the share inside."""
    cells = []
    for g in ("equities", "real", "bonds", "cash"):
        w = float(groups.get(g, 0.0))
        if w < min_weight:
            continue
        text = pct(w) if w >= 0.07 else ""
        cells.append(f'<div style="flex:{w:.5f} 1 0;background:{GROUP_COLORS[g]}" '
                     f'title="{esc(GROUP_NAMES[g])}: {pct(w, 1)}">{esc(text)}</div>')
    return (f'<div class="btp-barrow groups {cls}"><span>{esc(label)}</span>'
            f'<div class="btp-bar">{"".join(cells)}</div></div>')


def change(number, title, text, value, colour, reliable, helped):
    """One step of the gain: number, title, explanation, CHF a year, and how often it helped in the bootstrap."""
    tag = (f'<span class="btp-tag good">Reliable: it helped in {esc(helped)}</span>' if reliable
           else f'<span class="btp-tag">Not proven: it helped in {esc(helped)}</span>')
    return (f'<div class="btp-change"><i>{number}</i><div><b>{esc(title)}</b><p>{text}</p>{tag}</div>'
            f'<strong style="color:{colour}">{esc(value)}</strong></div>')


def helped(p):
    """The bootstrap p-value (share of replayed histories where the change did not help) in plain words."""
    return "more than 99% of replayed histories" if 1 - p > 0.99 else f"{1 - p:.0%} of replayed histories"


def signed_chf(x):
    return ("+" if x >= 0 else "") + chf(x, short=False)


def holdings(rows):
    """rows: list of (colour, name, amount text, share text)."""
    return '<div class="btp-hold">' + "".join(
        f'<div><span><i class="btp-dot" style="background:{c}"></i>{esc(name)}</span>'
        f'<b>{amount}<small>{share}</small></b></div>' for c, name, amount, share in rows) + "</div>"


def why(items):
    """items: list of (colour, html text): the reasons of the advice, one line each."""
    return '<div class="btp-why">' + "".join(
        f'<div><i style="background:{c}"></i><span>{text}</span></div>' for c, text in items) + "</div>"


def stats(items):
    """items: list of (value, label, colour or None)."""
    return '<div class="btp-stats">' + "".join(
        f'<div class="btp-stat"><b style="color:{c or TEXT}">{v}</b><span>{label}</span></div>'
        for v, label, c in items) + "</div>"


def cards(items):
    """items: list of (value, text, colour or None)."""
    return '<div class="btp-cards">' + "".join(
        f'<div class="btp-card"><b class="v" style="color:{c or TEXT}">{v}</b><p>{t}</p></div>'
        for v, t, c in items) + "</div>"


def waiting(text):
    return f'<div class="btp-wait">{esc(text)}<div class="track"><i></i></div></div>'


def table(df, formats=None, signed=(), index=True):
    """A DataFrame as an HTML table. formats: {column: function}; signed: columns coloured green when
    positive and red when negative. Text columns are aligned left, numbers right."""
    formats = formats or {}
    cols = list(df.columns)
    numeric = {c: all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                      for v in df[c]) for c in cols}
    head = ('<th class="txt"></th>' if index else "") + "".join(
        f'<th{"" if numeric[c] else " class=txt"}>{esc(c)}</th>' for c in cols)
    rows = []
    for idx, row in df.iterrows():
        cells = [f'<td class="txt">{esc(idx)}</td>'] if index else []
        for c in cols:
            v = row[c]
            if numeric[c] and np.isnan(float(v)):
                text, cls = "", ""
            else:
                text = formats[c](v) if c in formats else esc(v)
                cls = "" if numeric[c] else " class=txt"
                if c in signed and numeric[c]:
                    cls = ' class="btp-pos"' if v > 0 else (' class="btp-neg"' if v < 0 else "")
            cells.append(f"<td{cls}>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="btp-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

def plot_layout(height=260, **extra):
    """Common layout for every chart: transparent on the dark panel, light grid, legend on top."""
    layout = dict(
        height=height, margin=dict(l=4, r=8, t=28, b=4), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED, size=12),
        hoverlabel=dict(bgcolor=RAISED, bordercolor=LINE, font=dict(family=FONT, color=TEXT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=MUTED, size=11)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=LINE, tickcolor=LINE, color=MUTED),
        yaxis=dict(gridcolor=LINE, zeroline=False, linecolor="rgba(0,0,0,0)", color=MUTED),
    )
    layout.update(extra)
    return layout
