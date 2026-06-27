# Reproducibility and shared configuration
SEED = 42

# fraction of stocks selected in the ML portfolio (10% and 20%)
TOP_FRACS = [0.10, 0.20]

# monthly trading costs in return space (0 / 5 / 10 bps)
COSTS = [0.00, 0.0005, 0.001]

# sample splits for subperiod analysis
SUBPERIODS = [
    ("2019-01-01","2021-12-31"),
    ("2022-01-01","2024-12-31")
]

# baseline feature set for all models
FEATURES_BASE = [
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "vol_3m",
    "ma_ratio_50D",
    "RSI_14D"
]

import numpy as np, random, os

# fix random seeds for reproducibility
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"]=str(SEED)
