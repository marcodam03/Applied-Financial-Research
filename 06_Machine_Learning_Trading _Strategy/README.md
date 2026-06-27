# Machine Learning–Based Stock Selection and Trading Strategy
Marco D’Amico — Datascience And Advanced Programming

This project builds a simple Machine Learning pipeline that tries to predict
whether each S&P 500 stock will go up the following month.
Predictions are used to build and backtest simple investment strategies.

The pipeline automatically performs:

1. Download daily S&P 500 price data (Yahoo Finance)
2. Build monthly dataset and features (momentum, volatility, MA50 ratio, RSI)
3. Train several ML models (Logistic Regression, Random Forest, XGBoost, etc.)
4. Select the best model based on out-of-sample AUC
5. Backtest the following strategies:
   - ML Long Top 10%
   - Equal-Weight (benchmark)
   - Momentum 6M Top 10% (benchmark)
   - ML Long–Short Top 10% vs Bottom 10%
6. Generate plots (equity curve, drawdown, calibration)
7. Export all results to the `report/outputs/` folder

# Project structure
```
Marco_D-Amico_Project/
│
├── data/ # ticker list + generated datasets (ignored in git)
├── features/ # feature construction (momentum, vol, MA ratios, RSI)
├── models/ # ML models + walk-forward expanding window
├── metrics/ # evaluation metrics (AUC, Brier, Sharpe, MaxDD, etc.)
├── notebooks/ # exploratory analysis and prototyping
├── portfolio/ # portfolio strategies and sensitivity analysis
├── report/ # plotting utilities + exported figures / tables
│
├── main.py # full pipeline runner
├── settings.py # global configuration variables
├── environment.yml # conda environment for reproducibility
├── requirements.txt
└── README.md
```

# How to run
Clone the repository:

```bash
git clone https://github.com/marcodamico03/Marco_D-Amico_Project
cd Marco_D-Amico_Project

Create and activate the conda environment
conda env create -f environment.yml
conda activate ml-finance
```

# Run the full pipeline
python main.py

This command will coordinate the different python files and run the whole project, generate the results and the given outputs inside `report/outputs/`:

- `performance_summary.csv` — performance of all strategies
- `backtest_full.csv` — equity curves
- `calibration_points.csv` — probability calibration
- `turnover.csv` — monthly turnover of ML top bucket
- `sensitivity.csv` — sensitivity grid
- `model_metrics.csv` — results of all models
- `plots`:
  - equity curve
  - underwater (drawdown)
  - calibration curve
  - feature importance

# Interpretation
- AUC is usually around 0.48–0.50 → ML has very limited predictive power
  (this is normal in financial monthly forecasting).
- ML Long Top 10% behaves similarly to an equal-weight portfolio.
- Momentum 6M often performs best (well-known factor premium).
- ML Long–Short tends to perform poorly → confirms no alpha.

These outcomes are realistic and consistent with financial literature
and the Efficient Market Hypothesis.

# Author
Marco D’Amico
MSc Finance — HEC Lausanne
Datascience And Advanced Programming Project
