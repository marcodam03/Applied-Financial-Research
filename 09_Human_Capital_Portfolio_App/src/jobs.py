"""Slow computations in the background, so the app never freezes (step 5).

The backtest of a client takes a few seconds. The app asks for it, shows the rest of the
page at once, and checks again every second; meanwhile the client can keep changing the
inputs. One queue per kind of job (backtest, prior check, robustness), so a long robustness
test never delays a backtest. In each queue one job runs at a time, and a job that has not
started yet is dropped when a newer one arrives (the client changed the inputs again).
No Streamlit here.
"""
import threading
from concurrent.futures import ThreadPoolExecutor


class Jobs:
    def __init__(self):
        self._pools = {}                   # group -> its own queue
        self._futures = {}                 # key -> (group, future)
        self._lock = threading.Lock()

    def get(self, key, compute, group):
        """The future of the job `key`, started now if it is new. Pending jobs of the same group
        that have not started are cancelled: only the newest request matters."""
        with self._lock:
            if key in self._futures and not self._futures[key][1].cancelled():
                return self._futures[key][1]
            for other, (g, future) in list(self._futures.items()):
                if g == group and future.cancel():
                    del self._futures[other]
                elif future.done():
                    del self._futures[other]
            if group not in self._pools:
                self._pools[group] = ThreadPoolExecutor(max_workers=1)
            future = self._pools[group].submit(compute)
            self._futures[key] = (group, future)
            return future

    def pending(self):
        """True while some job is waiting or running."""
        with self._lock:
            return any(not future.done() for _, future in self._futures.values())
