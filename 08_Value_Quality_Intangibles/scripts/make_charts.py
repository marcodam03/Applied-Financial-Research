"""
Charts of the strategy against the US market, like the regime-monitor page.

Run from the project folder, after run_signal.py for the signal you want:
    python scripts/make_charts.py                          # the final strategy, qvi_timely
    python scripts/make_charts.py --signal qvi_intan_full

Writes to results/charts/:
    performance.html     interactive page: growth of $100 (log scale) with hover, period
                         buttons and a custom month range, drawdowns, a statistics table that
                         recomputes for the visible window, and a calendar-year table
    growth.png, drawdowns.png, annual_returns.png   static charts for the paper and slides

Series (monthly, 1963-07 to 2015-12; the holdout is never read):
    US market              CRSP value-weighted market (Fama-French MKT_RF + RF). SPY only
                           exists from 1993; this is the same index idea over the whole sample.
    Long-short fund        cash in T-bills + long decile 10 / short decile 1, after costs
    Long only              decile 10 only, after its own trading costs
    Market + strategy      the market plus the long-short positions (overlay, k = 1)
Keep the outputs local: they are built from WRDS data.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.alpha import load_factors  # noqa: E402

END = pd.Period("2015-12", freq="M")
COLORS = {  # validated categorical slots (light, dark); the market is neutral ink
    "US market": ("#52514e", "#c3c2b7"),
    "Long-short fund": ("#2a78d6", "#3987e5"),
    "Long only": ("#eb6834", "#d95926"),
    "Market + strategy": ("#1baf7a", "#199e70"),
}


def read(signal, kind):
    x = pd.read_csv(BASE_DIR / "results" / "signals" / signal / f"{kind}_vw.csv", index_col=0)
    x.index = pd.PeriodIndex(x.index, freq="M")
    return x[x.index <= END]


def build(signal: str) -> pd.DataFrame:
    m = importlib.import_module(f"src.signals.{signal}")
    L, S = str(m.LONG), str(m.SHORT)
    r, c = read(signal, "returns"), read(signal, "costs")
    f = load_factors(BASE_DIR / "data" / "external" / "factors_monthly.csv").reindex(r.index)
    net = r[f"{L}-{S} net"]
    out = pd.DataFrame({
        "US market": f["MKT_RF"] + f["RF"],
        "Long-short fund": f["RF"] + net,
        "Long only": r[L] - c[L],
        "Market + strategy": f["MKT_RF"] + f["RF"] + net,
    })
    out["_rf"] = f["RF"]
    return out.dropna()


def pngs(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [c for c in df.columns if not c.startswith("_")]
    x = df.index.to_timestamp()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})

    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ends = []
    for s in series:
        w = 100 * (1 + df[s]).cumprod()
        ax.plot(x, w, color=COLORS[s][0], lw=2 if s != "US market" else 1.6,
                ls="-" if s != "US market" else (0, (5, 3)), label=s)
        ends.append((np.log10(w.iloc[-1]), s, w.iloc[-1]))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    lo, hi = np.log10(ax.get_ylim())
    gap, last = (hi - lo) * 0.045, -np.inf   # end labels never overlap
    for ly, s, v in sorted(ends):
        ly = max(ly, last + gap)
        last = ly
        ax.annotate(f"{s}  ${v:,.0f}", (x[-1], 10 ** ly), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color="#0b0b0b", annotation_clip=False)
    ax.set_ylabel("Value of $100 invested in July 1963 (log scale)")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Growth of $100, after trading costs, 1963-2015", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "growth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for s in series:
        w = (1 + df[s]).cumprod()
        ax.plot(x, 100 * (w / w.cummax() - 1), color=COLORS[s][0], lw=1.4 if s != "US market" else 1.2,
                ls="-" if s != "US market" else (0, (5, 3)), label=s)
    ax.set_ylabel("Drawdown from previous peak (%)")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.legend(frameon=False, loc="lower left", ncol=3, fontsize=8.5)
    ax.set_title("Drawdowns", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "drawdowns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    yr = (1 + df[["US market", "Long-short fund"]]).groupby(df.index.year).prod() - 1
    fig, ax = plt.subplots(figsize=(11, 3.8))
    xs = np.arange(len(yr))
    ax.bar(xs - 0.2, yr["US market"] * 100, 0.38, color=COLORS["US market"][0], label="US market")
    ax.bar(xs + 0.2, yr["Long-short fund"] * 100, 0.38, color=COLORS["Long-short fund"][0], label="Long-short fund")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(xs[::4], yr.index[::4])
    ax.set_ylabel("Calendar-year return (%)")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Every year: the strategy against the market (1963 = July to December)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "annual_returns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy vs market</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--good:#006300;--ring:rgba(11,11,11,.10);__LIGHT__}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--good:#0ca30c;--ring:rgba(255,255,255,.10);__DARK__}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--good:#0ca30c;--ring:rgba(255,255,255,.10);__DARK__}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font:600 26px/1.2 Georgia,serif;margin:0 0 4px}.sub{color:var(--ink2);margin:0 0 18px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px;margin-bottom:16px}
.ctrl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
button,select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:var(--ink);color:var(--surface)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 4px;color:var(--ink2)}
.legend span{cursor:pointer;display:inline-flex;align-items:center;gap:6px}.legend span.off{opacity:.35}
.sw{width:18px;height:3px;border-radius:2px;display:inline-block}
svg{width:100%;display:block}text{fill:var(--muted);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.12);display:none;z-index:5}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}.scroll{max-height:360px;overflow:auto}
.note{color:var(--muted);font-size:12px;margin-top:8px}
</style></head><body><main>
<h1>Cheap and cash-profitable, against the market</h1>
<p class="sub">__SIGNAL__, value-weighted, after trading costs. Monthly, __START__ to __END__. Holdout 2016-2025 not shown.</p>
<div class="card">
 <div class="ctrl" id="periods"></div>
 <div class="ctrl"><span style="color:var(--ink2)">From</span><select id="from"></select><span style="color:var(--ink2)">to</span><select id="to"></select></div>
 <div class="legend" id="legend"></div>
 <svg id="growth" viewBox="0 0 1000 380" role="img" aria-label="Growth of 100 dollars, log scale"></svg>
 <svg id="dd" viewBox="0 0 1000 170" role="img" aria-label="Drawdowns"></svg>
 <p class="note">Growth of $100 invested at the start of the window, log scale. Drawdown = fall from the previous peak. Click a legend item to hide it.</p>
</div>
<div class="card"><table id="stats"></table>
<p class="note">Excess of the T-bill rate for Sharpe. Beta and correlation are against the US market. Recomputed for the window shown.</p></div>
<div class="card"><div class="scroll"><table id="years"></table></div></div>
<div class="tip" id="tip"></div>
</main><script>
const D=__DATA__;const N=D.months.length;const names=D.names;let hidden=new Set();let a=0,b=N-1;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const col=i=>css('--s'+i);
function cum(s,i0,i1){let w=[1];for(let i=i0+1;i<=i1;i++)w.push(w[w.length-1]*(1+D.series[s][i]));return w}
function fmt(x,d=1){return (x>=0?'':'−')+Math.abs(x).toFixed(d)}
function draw(){const g=document.getElementById('growth'),dd=document.getElementById('dd');const W=1000,H=380,L=56,R=150,T=12,B=26;
 const vis=names.map((n,i)=>i).filter(i=>!hidden.has(i));const curves={};let lo=1,hi=1;
 vis.forEach(i=>{const w=cum(i,a,b);curves[i]=w;w.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v)})});
 const ly=v=>Math.log(v),y0=ly(lo*0.95),y1=ly(hi*1.05);const X=k=>L+(W-L-R)*k/Math.max(1,b-a);const Y=v=>T+(H-T-B)*(1-(ly(v)-y0)/(y1-y0));
 let s='';const ticks=[];for(let e=Math.floor(Math.log10(lo*100));e<=Math.ceil(Math.log10(hi*100));e++)for(const m of [1,2,5]){const v=m*10**e/100;if(v>=lo*0.95&&v<=hi*1.05)ticks.push(v)}
 ticks.forEach(v=>{s+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css('--grid')}"/><text x="${L-6}" y="${Y(v)+4}" text-anchor="end">$${(v*100).toLocaleString()}</text>`});
 const yrs=[];for(let k=a;k<=b;k++)if(D.months[k].endsWith('-01')||k===a)yrs.push(k);const step=Math.max(1,Math.ceil(yrs.length/10));
 yrs.filter((k,j)=>j%step===0).forEach(k=>{s+=`<text x="${X(k-a)}" y="${H-8}" text-anchor="middle">${D.months[k].slice(0,4)}</text>`});
 s+=`<line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="${css('--axis')}"/>`;
 const ends=[];vis.forEach(i=>{const w=curves[i];const p=w.map((v,k)=>`${k?'L':'M'}${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join('');
  s+=`<path d="${p}" fill="none" stroke="${col(i)}" stroke-width="2" ${i===0?'stroke-dasharray="6 4"':''} stroke-linejoin="round"/>`;ends.push([Y(w[w.length-1]),i,w[w.length-1]])});
 ends.sort((p,q)=>p[0]-q[0]);let last=-99;ends.forEach(e=>{let yy=Math.max(e[0],last+14);last=yy;
  s+=`<text x="${W-R+6}" y="${yy+4}" style="fill:${css('--ink')}">${names[e[1]]} $${Math.round(e[2]*100).toLocaleString()}</text>`});
 s+=`<line id="cx" y1="${T}" y2="${H-B}" stroke="${css('--axis')}" visibility="hidden"/><rect id="hit" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/>`;
 g.innerHTML=s;
 const H2=170;let lo2=0;const dds={};vis.forEach(i=>{const w=curves[i];let pk=w[0];dds[i]=w.map(v=>{pk=Math.max(pk,v);return v/pk-1});lo2=Math.min(lo2,...dds[i])});
 const Y2=v=>8+(H2-8-22)*(v/(lo2*1.05||-1));let s2='';[0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7].filter(v=>v>=lo2*1.05).forEach(v=>{s2+=`<line x1="${L}" x2="${W-R}" y1="${Y2(v)}" y2="${Y2(v)}" stroke="${css('--grid')}"/><text x="${L-6}" y="${Y2(v)+4}" text-anchor="end">${Math.round(v*100)}%</text>`});
 vis.forEach(i=>{s2+=`<path d="${dds[i].map((v,k)=>`${k?'L':'M'}${X(k).toFixed(1)},${Y2(v).toFixed(1)}`).join('')}" fill="none" stroke="${col(i)}" stroke-width="1.5" ${i===0?'stroke-dasharray="6 4"':''}/>`});
 dd.innerHTML=s2;
 const tip=document.getElementById('tip'),hit=document.getElementById('hit'),cx=document.getElementById('cx');
 hit.onmousemove=ev=>{const r=g.getBoundingClientRect();const px=(ev.clientX-r.left)*W/r.width;let k=Math.round((px-L)/(W-L-R)*(b-a));k=Math.max(0,Math.min(b-a,k));
  cx.setAttribute('x1',X(k));cx.setAttribute('x2',X(k));cx.setAttribute('visibility','visible');
  tip.innerHTML=`<b>${D.months[a+k]}</b><br>`+vis.map(i=>`<span class="sw" style="background:${col(i)}"></span> ${names[i]}: $${Math.round(curves[i][k]*100).toLocaleString()} (${fmt(dds[i][k]*100)}% from peak)`).join('<br>');
  tip.style.display='block';tip.style.left=Math.min(ev.clientX+14,innerWidth-300)+'px';tip.style.top=(ev.clientY+14)+'px'};
 hit.onmouseleave=()=>{tip.style.display='none';cx.setAttribute('visibility','hidden')};
 stats();}
function stats(){const rf=D.rf.slice(a+1,b+1),m=D.series[0].slice(a+1,b+1),n=m.length;const rows=[];
 const mean=x=>x.reduce((p,q)=>p+q,0)/x.length,sd=x=>{const u=mean(x);return Math.sqrt(x.reduce((p,q)=>p+(q-u)**2,0)/(x.length-1))};
 names.forEach((nm,i)=>{const x=D.series[i].slice(a+1,b+1);if(x.length<12)return;const w=x.reduce((p,q)=>p*(1+q),1);const ex=x.map((v,k)=>v-rf[k]);
  let pk=1,c=1,mdd=0;x.forEach(v=>{c*=1+v;pk=Math.max(pk,c);mdd=Math.min(mdd,c/pk-1)});
  const um=mean(m),ux=mean(x);let cov=0,vm=0,vx=0;x.forEach((v,k)=>{cov+=(v-ux)*(m[k]-um);vm+=(m[k]-um)**2;vx+=(v-ux)**2});
  rows.push([nm,(w**(12/n)-1)*100,sd(x)*Math.sqrt(12)*100,mean(ex)/sd(ex)*Math.sqrt(12),mdd*100,cov/vm,cov/Math.sqrt(vm*vx),w*100])});
 const base=rows[0];let h='<tr><th>Portfolio</th><th>Return per year</th><th>Volatility</th><th>Sharpe</th><th>Max drawdown</th><th>Beta</th><th>Correlation</th><th>$100 became</th></tr>';
 rows.forEach(r=>{h+=`<tr><td>${r[0]}</td><td>${fmt(r[1])}%</td><td>${fmt(r[2])}%</td><td>${fmt(r[3],2)}</td><td>${fmt(r[4])}%</td><td>${fmt(r[5],2)}</td><td>${fmt(r[6],2)}</td><td>$${Math.round(r[7]).toLocaleString()}</td></tr>`});
 document.getElementById('stats').innerHTML=h;}
function years(){let h='<tr><th>Year</th>'+names.map(n=>`<th>${n}</th>`).join('')+'</tr>';Object.keys(D.years).forEach(y=>{h+=`<tr><td>${y}</td>`+D.years[y].map(v=>`<td>${fmt(v*100)}%</td>`).join('')+'</tr>'});
 document.getElementById('years').innerHTML=h}
function setRange(i0,i1){a=i0;b=i1;document.getElementById('from').value=a;document.getElementById('to').value=b;draw()}
const P=document.getElementById('periods');[['5Y',60],['10Y',120],['20Y',240],['30Y',360],['Full',N]].forEach(([l,m])=>{const bt=document.createElement('button');bt.textContent=l;
 bt.onclick=()=>{[...P.children].forEach(c=>c.classList.remove('on'));bt.classList.add('on');setRange(Math.max(0,N-1-m),N-1)};if(l==='Full')bt.classList.add('on');P.appendChild(bt)});
['from','to'].forEach(id=>{const s=document.getElementById(id);D.months.forEach((m,k)=>{if(m.endsWith('-01')||m.endsWith('-07')||k===0||k===N-1){const o=document.createElement('option');o.value=k;o.textContent=m;s.appendChild(o)}});
 s.onchange=()=>{const f=+document.getElementById('from').value,t=+document.getElementById('to').value;if(t>f+12){[...P.children].forEach(c=>c.classList.remove('on'));a=f;b=t;draw()}}});
const Lg=document.getElementById('legend');names.forEach((n,i)=>{const sp=document.createElement('span');sp.innerHTML=`<i class="sw" style="background:${col(i)}"></i>${n}`;
 sp.onclick=()=>{if(hidden.has(i))hidden.delete(i);else if(hidden.size<names.length-1)hidden.add(i);sp.classList.toggle('off',hidden.has(i));draw()};Lg.appendChild(sp)});
years();setRange(0,N-1);matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
</script></body></html>"""


def html(df: pd.DataFrame, out: Path) -> None:
    series = [c for c in df.columns if not c.startswith("_")]
    yr = (1 + df[series]).groupby(df.index.year).prod() - 1
    data = {"months": [str(p) for p in df.index], "names": series,
            "series": [df[s].round(6).tolist() for s in series], "rf": df["_rf"].round(6).tolist(),
            "years": {str(y): yr.loc[y].round(4).tolist() for y in yr.index}}
    light = ";".join(f"--s{i}:{COLORS[s][0]}" for i, s in enumerate(series))
    dark = ";".join(f"--s{i}:{COLORS[s][1]}" for i, s in enumerate(series))
    page = (HTML.replace("__DATA__", json.dumps(data)).replace("__LIGHT__", light).replace("__DARK__", dark)
            .replace("__SIGNAL__", df.attrs.get("signal", ""))
            .replace("__START__", str(df.index[0])).replace("__END__", str(df.index[-1])))
    (out / "performance.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", default="qvi_timely")
    args = ap.parse_args()
    df = build(args.signal)
    df.attrs["signal"] = args.signal
    out = BASE_DIR / "results" / "charts"
    out.mkdir(parents=True, exist_ok=True)
    html(df, out)
    pngs(df, out)
    w = (1 + df[[c for c in df.columns if not c.startswith("_")]]).prod() * 100
    print("$100 in", df.index[0], "became, by", df.index[-1], ":")
    print(w.round(0).to_string())
    print(f"\nSaved in {out}: performance.html (open it in a browser), growth.png, drawdowns.png, annual_returns.png")


if __name__ == "__main__":
    main()
