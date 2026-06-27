# Project Name: Sustainable Finance & Portfolio Allocation

## Overview
This project explores the integration of sustainability constraints (e.g., carbon footprint reduction) into an emerging market equity portfolio. The objective is to assess the trade-off between environmental impact and financial performance (Sharpe Ratio, Tracking Error).

## Folder Structure
- `data/`: Contains raw emissions/returns data and cleaned datasets.
- `notebooks/`: Contains the Jupyter notebooks for EDA and initial strategy testing.
- `src/`: Modular Python scripts for data cleaning, signal generation, and optimization.
- `output/`: Generated performance reports, backtest statistics, and visualization plots.

## Key Features
*   **Universe:** Emerging Markets equities (2014-2025).
*   **Methodology:** Rolling 10-year estimation window, tracking error minimization with CO2 emission constraints.
*   **Technologies:** Python (pandas, numpy, scipy, scikit-learn).

## Getting Started
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the pipeline: `FINAL.ipynb`.
