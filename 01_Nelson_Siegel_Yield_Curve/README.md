# TERM STRUCTURE ESTIMATION: A PARAMETRIC APPROACH BASED ON NELSON SIEGEL MODELS
**Marco D'Amico — Bachelor Thesis Project**

**Project Porting:** This codebase represents a complete refactoring and translation of the original thesis work, initially developed in **MATLAB**. The transition to **Python** was undertaken to leverage open-source data science libraries (Pandas, SciPy) and to enhance reproducibility.

This project implements a complete computational pipeline to estimate the **Term Structure of Interest Rates** for Italian Government Bonds (BTP). It applies parametric models (**Nelson-Siegel** and **Nelson-Siegel-Svensson**) to calibrate the yield curve against observed market prices.

The pipeline automatically performs:

1.  **Data Ingestion**: Loads raw daily BTP data (clean prices, coupons, maturities) from Excel.
2.  **Financial Engineering**: Calculates accrued interest, dirty prices, and generates cashflow matrices.
3.  **Model Calibration**: Optimizes model parameters ($\beta_0, \beta_1, \beta_2, \tau, \dots$) by minimizing the Sum of Squared Errors (SSE) between model prices and market prices using the **L-BFGS-B** algorithm.
4.  **Cross-Sectional Analysis**: Fits the curve for a specific reference date (May 20, 2025).
5.  **Time-Series Analysis**: Performs a rolling calibration over a 30-day trading window (April–May 2025) to analyze parameter stability.
6.  **Reporting**: Exports all figures and tables required for the thesis to the `output/` folder.

# Project Structure

```text
Tesi_BTP_Final/
│
├── data/
│   └── raw/               # Input dataset (30giorni.xlsx)
│
├── output/                # Generated CSV results and PNG plots
│
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── config.py          # Global configuration (paths, dates)
│   ├── data_loader.py     # Excel ingestion and cleaning
│   ├── finance.py         # Financial math (Accrued, Cashflows, Tau)
│   ├── models.py          # NS/NSS formulas and optimization engine
│   └── plotting.py        # Professional plotting utilities
│
├── main.py                # Full pipeline runner
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

# How to Run
1. Clone the repository:
```bash
git clone https://github.com/marcodamico03/Tesi_BTP_Final.git
cd Tesi_BTP_Final
```

2. Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

3. Run the full pipeline
```bash
python main.py
```

This command will coordinate the modules, run both Single-Day and 30-Day analyses, and generate the following outputs inside the `output/` folder:

- `Tables_7_8_Full_Results.csv` — Full time-series parameters
- `Table5_Summary_NS.csv` — Descriptive statistics for Nelson-Siegel
- `Table6_Summary_NSS.csv` — Descriptive statistics for Nelson-Siegel-Svensson
- `plots`:
  - Fig4_Spot_NS_Single.png — Spot rate curve (Single Day)
  - Fig8_NS_Lambda_Time.png — Evolution of the decay factor $\lambda$
  - Fig11_NS_Misprice_Time.png — Pricing error evolution
  - Fig13_NS_Curves_Comparison.png — Yield curve shifts over distinct dates

# Interpretation
- Model Comparison: The Nelson-Siegel-Svensson (NSS) model consistently achieves a lower SSE compared to the standard Nelson-Siegel (NS) model, thanks to its ability to capture the "hump" and complex shapes of the BTP curve.
- Parameter Stability: The decay parameter $\lambda$ shows stability over the analyzed month, validating the robustness of the optimization procedure.
- Computational Efficiency: The vectorized implementation allows for calibrating 30 days of trading data (involving hundreds of cashflows per bond) in under 60 seconds.

These outcomes are realistic and consistent with financial literature
and the Efficient Market Hypothesis.

## 📄 Full Documentation

For a detailed explanation of the mathematical models (Nelson-Siegel & NSS), the optimization methodology, and the financial interpretation of the results, please refer to the full thesis document:

**[Read the Full Thesis (PDF)](thesis_damico.pdf)**

# Author
Marco D'Amico
University of Rome "Tor Vergata"
Bachelor of Science in Business Administration and Economics
Thesis in Mathematical Finance
