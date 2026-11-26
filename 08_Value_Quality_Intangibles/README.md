# Industry-relative value and quality with intangible capital

Marco D'Amico with Roberto Berardi, Roatanak Cédric Nim, Viola Brignoli, Ana Lucía Clavijo Martínez and Claudio Marcoli. Applied Investments, Prof. Amit Goyal, HEC Lausanne, November 2026. The full report is in `paper.pdf`.

A long-short US equity strategy built for the Applied Investments course (MSc Finance,
HEC Lausanne, fall 2026): buy the stocks that are cheap for their industry and generate
real cash, once the accounts are corrected for intangible investment, and short the
opposite. Tested on CRSP/Compustat 1963-2025 against the CAPM, Fama-French and q-factor
models, after realistic trading costs, with 2016-2025 kept as an untouched holdout.

**Headline (value-weighted, net of trading costs, % per month, t-stat):**

| | 1963-2015 (research) | 2016-2025 (holdout, run once) |
|---|---|---|
| Net return | 0.33 (2.6) | 0.73 (2.2) |
| Fama-French 6-factor alpha | 0.53 (3.8) | 0.75 (2.7) |
| q5 alpha (Hou, Mo, Xue and Zhang) | 0.41 (2.6) | 0.58 (2.4) |

The rules, the reasoning, every result and the limits are in
[docs/STRATEGY.md](docs/STRATEGY.md).

## Project structure

```
configs/
  final.yaml            settings of the final strategy (sample, universe, costs, robustness)
  annual.yaml           settings of the annual version used for comparison
  default.yaml          settings for the replication of published anomalies
src/
  data/                 cleaning of the WRDS files, trading-cost inputs, factor data
  signals/              one file per signal
    qvi_timely.py       THE FINAL STRATEGY (built in _qvi_ext.py)
    qvi_intan_full.py   its annual version (built in _qvi_plus.py)
    asset_growth.py, book_to_market.py, momentum.py
                        published anomalies, used to check the pipeline
    _*.py               helpers: industries, intangible capital, value and quality scores
  portfolio/            portfolio sorts and the trading-cost model
  evaluation/           factor-model regressions and performance statistics
scripts/
  run_signal.py         backtest one signal (portfolios, costs, alphas, robustness)
  run_final_analysis.py every analysis of the final strategy in one command
  analyse_strategy.py   luck (bootstrap), dependence on a few years, crises, overlay
  run_capacity.py       fund size and cost-model sensitivity
  overfit_tests.py      placebo, random half universes, neighbouring parameters
  yearly_performance.py year-by-year returns as a fund
  make_charts.py        interactive chart (results/charts/performance.html)
  compare_strategies.py final strategy vs its annual version
  run_asset_growth.py   reproduces the professor's asset-growth numbers
  validate_signal.py    compares a signal's portfolios with Ken French's
holdout_log.txt         every use of the 2016-2025 holdout
```

## How to run

Run every command from the project folder. Python 3.10+, then once:
`pip install -r requirements.txt`.

1. **Data (once).** The CRSP and Compustat files come from WRDS (licensed, not in git).
   Put `crsp_monthly.csv`, `ccm_funda.csv` and `crsp_daily.csv` in `data/raw/`, then:
   ```
   python convert_to_parquet.py crsp_daily.csv
   python src/data/clean.py
   ```
   The factor file `data/external/factors_monthly.csv` is in git (public Fama-French and
   q-factor data), so every run uses the same vintage.
2. **Check the pipeline.**
   ```
   python scripts/run_asset_growth.py                     # professor's numbers
   python scripts/validate_signal.py --signal momentum    # vs Ken French's portfolios
   ```
3. **Final strategy, everything.**
   ```
   python scripts/run_final_analysis.py
   python scripts/compare_strategies.py
   ```

**Holdout discipline.** The years from 2016 are fixed as holdout in `scripts/run_signal.py`
(`HOLDOUT_START`): without `--holdout` every period is cut at 2015-12, whatever the config
says. Each `--holdout` run is recorded in `holdout_log.txt`. The holdout of the final
strategy has been run once.

## Team

Marco D'Amico, Roberto Berardi, Roatanak Cédric Nim, Viola Brignoli, Ana Lucía Clavijo Martínez, Claudio Marcoli.
