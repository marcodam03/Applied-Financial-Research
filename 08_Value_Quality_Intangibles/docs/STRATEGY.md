# Cheap and cash-profitable, within each industry, with intangibles counted

Branch `roberto` (Roberto and Marco). Final signal: `src/signals/qvi_intan_full.py`
(built by `src/signals/_qvi_plus.py`). Every number below comes from
`python scripts/run_roberto.py` (section 8).

## 1. The idea in one sentence

Buy the US stocks that are cheap **for their industry** and generate real **cash** from their
operations, once the accounts are corrected for the investment in intangibles (R&D, brands,
organisation) that accounting treats as a cost; short the opposite.

## 2. The rules

Once a year, at the end of June of year t, using the accounts of the fiscal year that ended
in calendar year t-1 (public by then) and the market value of December t-1:

1. **Universe (team standard, `configs/strategies.yaml`):** US common stocks (NYSE, AMEX,
   NASDAQ) above the NYSE 20th size percentile, so no microcaps.
2. **Industry:** each stock gets one of the 12 Fama-French industries from its SIC code.
3. **Intangible capital** (`src/signals/_intangibles.py`, Peters and Taylor 2017), rebuilt
   year by year with the perpetual inventory method:
   - knowledge capital: K(t) = 0.85 K(t-1) + R&D(t)
   - organisation capital: O(t) = 0.80 O(t-1) + 0.30 x (SG&A without R&D)(t)
4. **Cheapness score.** Inside each industry, rank every firm on four ratios and average the
   ranks (at least three needed):
   - book value / market value, with book = book equity - goodwill + K + O
     (Eisfeldt, Kim and Papanikolaou 2022)
   - earnings / market value
   - cash flow (earnings + depreciation) / market value
   - EBITDA / enterprise value

   Banks, insurers and other financials (SIC 6000-6999) use only the first two: for them
   debt is raw material and EBITDA has no meaning.
5. **Quality score.** Inside each industry, rank every firm on cash-based operating
   profitability (Ball, Gerakos, Linnainmaa and Nikolaev 2016), corrected for intangibles:
   (sales - cost of goods - SG&A + R&D + 0.30 x SG&A without R&D - increase in receivables
   - increase in inventory + increase in payables) / (total assets + K + O).
   A missing item counts as no change.
6. **Final score:** the average of the two ranks, ranked again inside the industry.
   Industries need at least 10 stocks that year.
7. **Portfolio:** deciles with NYSE breakpoints. Long decile 10 (cheap and profitable),
   short decile 1 (expensive and unprofitable), each stock weighted by its market value.
8. **Hold for 12 months**, then repeat.

Trading costs follow the team's model (`src/portfolio/costs.py`): half the bid-ask spread
plus price impact, for a fund of $100m per side in 2025 dollars.

## 3. Why it should work

- **Industry-relative cheapness removes a bet that does not pay.** A bank always looks
  cheap next to a software firm, so a plain value portfolio is mostly an industry bet.
  Inside an industry, cheapness is real information (Asness, Porter and Stevens 2000).
- **Cash, not accounting profit.** Profit that never arrives as cash is often an accounting
  choice, and investors overpay for it (Sloan 1996; Ball et al. 2016).
- **Each half protects the other.** Cheap stocks alone are often cheap for a reason (value
  traps); profitable stocks alone are often expensive. Requiring both keeps the cheap firms
  that are healthy (Novy-Marx 2013; Asness, Frazzini and Pedersen 2019).
- **Intangibles fix the accounts where they are most wrong.** R&D and brand building are
  investments, but accounting expenses them. Without the correction, a pharma or software
  firm looks both expensive (book value too low) and unprofitable (profit too low), so the
  strategy would short good firms by mistake. The correction matters most after 1995, when
  intangible-heavy firms became a large part of the market, and that is where the gain is
  (section 4).

## 4. Results, 1963-07 to 2015-12, value-weighted, no microcaps, after trading costs

Net long-short spread (decile 10 minus decile 1), % per month, t-stats in brackets
(alphas: Newey-West, 6 lags).

| | Final: intangibles in value and quality | First version (no intangibles) |
|---|---|---|
| Net return 1963-2015 | **0.46 (3.64)** | 0.38 (2.95) |
| Net return 1963-1994 | 0.50 (3.20) | 0.46 (2.90) |
| Net return 1995-2015 | 0.41 (1.90) | 0.24 (1.16) |
| Net Sharpe (annual) | 0.50 | 0.41 |
| CAPM net alpha | 0.55 (4.17) | 0.46 (3.49) |
| Fama-French 5 net alpha | **0.50 (3.78)** | 0.37 (2.83) |
| Fama-French 6 net alpha | **0.45 (3.13)** | 0.35 (2.31) |
| q5 net alpha | 0.24 (1.60) | 0.15 (0.97) |
| With every stock (microcaps in), net | 0.56 | 0.47 |

Reading: after costs the final strategy has a net return and FF5/FF6 alphas with t above 3,
the Harvey, Liu and Zhu (2016) bar. The improvement over the first version is on every
line and is largest after 1995, as the intangibles story predicts. On its own the
improvement is not yet statistically proven: final minus first version is +0.08% per month
(t 1.27) over 1963-2015 and +0.16% (t 1.17) over 1995-2015.

### Which piece does the work (same table, `results/roberto/summary.txt`)

| Version | Net % (t) | FF5 net alpha % (t) | q5 net alpha % (t) |
|---|---|---|---|
| Final (intangibles in value and quality) | 0.46 (3.64) | 0.50 (3.78) | 0.24 (1.60) |
| Intangibles in book value only | 0.45 (3.57) | 0.42 (3.32) | 0.16 (1.15) |
| First version | 0.38 (2.95) | 0.37 (2.83) | 0.15 (0.97) |
| First version, 49 industries | 0.35 (2.72) | 0.33 (2.46) | 0.06 (0.42) |
| Quality + value without industries | 0.57 (3.66) | 0.32 (2.40) | 0.06 (0.35) |
| Value half alone (within industries) | 0.38 (2.98) | 0.18 (1.70) | 0.12 (0.73) |
| Quality half alone (cash profitability) | 0.26 (1.80) | 0.52 (4.16) | 0.16 (0.99) |

- The version without industries earns the most, but almost all of it is the known value
  and profitability factors: its FF5 alpha is lower and its q5 alpha is close to zero.
  Measuring within industries is what turns a factor bet into alpha.
- The intangible book value raises the return; the intangible profitability raises the
  alphas. 49 industries instead of 12 do not help.

### Luck, years, market, capacity

`run_roberto.py` also runs, for the final strategy: `analyse_strategy.py` (bootstrap,
Bonferroni for the 20 ideas tested, results without the best years, behaviour in crises,
market + strategy), `run_capacity.py` ($100m to $2.5bn per side), `yearly_performance.py`
and `make_charts.py`. Their full output is in `results/roberto/logs/`; the summary of those
numbers will be added here.

## 5. Limits

- The q5 model explains a good part of the alpha after costs (q5 net alpha t 1.60).
- The gain from the intangible correction over the first version is consistent with the
  story but not statistically significant on its own.
- 1995-2015 alone is weaker (net t 1.90).
- It needs short selling (borrow costs and availability are not in the cost model) and
  holds a few hundred stocks per side: a strategy for a fund, not a small account.

## 6. Holdout 2016-2025

To be run once by the team on the final rules
(`python scripts/run_signal.py --signal qvi_intan_full --config configs/strategies.yaml --holdout`).

## 7. Files on this branch

| File | What it does |
|---|---|
| `src/signals/qvi_intan_full.py`, `_qvi_plus.py` | **the final strategy** (Marco) |
| `src/signals/quality_value_industry.py` | the first version |
| `src/signals/value_industry.py`, `cash_op.py` | its two halves (also tested alone) |
| `src/signals/quality_value.py`, `value_composite.py` | the same ideas without industries |
| `src/signals/quality_value_industry_ff49.py`, `quality_value_industry_intan.py`, `qvi_intensity.py` | variants |
| `src/signals/_industry.py`, `_quality.py`, `_intangibles.py` | helpers (industries, ranks, universe, intangible capital) |
| `scripts/run_roberto.py` | runs everything for this document |
| `scripts/run_fast.py`, `configs/fast.yaml` | quick comparison of several signals (Marco) |
| `scripts/analyse_strategy.py`, `run_capacity.py`, `yearly_performance.py`, `make_charts.py` | analysis of the final strategy |

## 8. How to reproduce

```
git pull origin roberto
python src/data/clean.py              # once, needs the WRDS files in data/raw/
python scripts/run_roberto.py         # every number above (2 backtests at a time; --workers 3 is faster)
open results/charts/performance.html  # interactive chart against the market
```
