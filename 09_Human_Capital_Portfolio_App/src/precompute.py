"""Optional: compute the backtests of the app in advance (step 5).

Not needed to use the app: it computes the backtest of each client by itself, in the background,
the first time the client is shown (a few seconds), and keeps it in data/cache/. This script only
does it in advance for every persona with its default settings.

Run from the project folder:
    python -m src.precompute                 backtest of every example client (about 1 minute)
    python -m src.precompute --robustness    also the version 1 backtests and robustness tests (about 5 minutes)
"""
import sys
import time

from src import backtest as B
from src import dashboard as D
from src.config import load_params


def main(robustness=False):
    base = load_params()
    data = B.load_backtest_data()
    from src import allocation as A
    data2 = A.load_data(base)
    names = ["story"] + (["backtest", "prior", "robustness"] if robustness else [])
    for persona in base["personas"]:
        for name in names:
            params = A.client_params(persona, None, base) if name == "story" else D.client_params(persona, None, base)
            start = time.time()
            was_saved = D.saved(name, params) is not None
            D.remember(name, params, data2 if name == "story" else data)
            print(f"{persona:18s} {name:11s} {'already saved' if was_saved else 'computed'}"
                  f"  {time.time() - start:5.1f} s")
    print(f"Saved in {D.CACHE_DIR}")


if __name__ == "__main__":
    main(robustness="--robustness" in sys.argv)
