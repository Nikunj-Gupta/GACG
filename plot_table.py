#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- Configuration ---
ROOT_DIR    = "gather_0,2,3,4,7- all"
METRIC_NAME = "test_battle_won_mean"
SMOOTHING   = 0.75
MAX_STEPS   = 900_000
MILESTONE   = 900_000  # only sample at 1M steps

# --- Helpers ---
def ema_smooth(x: np.ndarray, α: float) -> np.ndarray:
    """Compute exponential moving average of array x with smoothing α."""
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = α * out[i-1] + (1 - α) * x[i]
    return out

def load_single_run(path: str):
    """Load (steps, values) up to MAX_STEPS from one TensorBoard event file."""
    ea = EventAccumulator(path)
    ea.Reload()
    if METRIC_NAME not in ea.Tags().get("scalars", []):
        return None
    evs = ea.Scalars(METRIC_NAME)
    steps = np.array([e.step  for e in evs])
    vals  = np.array([e.value for e in evs])
    mask  = steps <= MAX_STEPS
    return steps[mask], vals[mask]

def collect_all(root_dir: str):
    """
    Walk subfolders named like:
      gather__<algo>__rnn__seed_<n>[_<variant>]
    and collect all runs per variant key.
    """
    data = defaultdict(list)
    pat  = re.compile(r"^gather__(?P<a>[^_]+)__rnn__seed_(?P<s>\d+)(?:_(?P<v>.+))?$")
    for d in sorted(os.listdir(root_dir)):
        m = pat.match(d)
        if not m:
            continue
        key = m.group("a") + (f"_{m.group('v')}" if m.group("v") else "")
        evf = sorted(glob.glob(os.path.join(root_dir, d, "events.out.tfevents*")))
        if not evf:
            continue
        run = load_single_run(evf[-1])
        if run:
            data[key].append(run)
    return data

def aggregate_runs(runs):
    """
    Align runs by shortest length, then compute:
      steps, EMA-smoothed mean, EMA-smoothed std
    """
    # find common length
    L = min(len(vals) for _, vals in runs)
    steps = runs[0][0][:L]
    arr   = np.stack([v[:L] for _, v in runs], axis=0)
    mean  = arr.mean(axis=0)
    std   = arr.std(axis=0)
    return steps, ema_smooth(mean, SMOOTHING), ema_smooth(std, SMOOTHING)

# --- Main ---
if __name__ == "__main__":
    all_data = collect_all(ROOT_DIR)

    # Print header
    print(f"{'Variant':30s}  {'Mean@1M':>8s}  {'Std@1M':>8s}")

    for key, runs in sorted(all_data.items()):
        if not runs:
            continue

        # get steps, mean, std
        if len(runs) > 1:
            steps, mean, std = aggregate_runs(runs)
        else:
            steps, vals = runs[0]
            mean = ema_smooth(vals, SMOOTHING)
            std  = np.zeros_like(mean)

        # locate index at or just before MILESTONE
        idx = np.searchsorted(steps, MILESTONE, side="right") - 1
        idx = max(0, min(idx, len(steps) - 1))

        # output
        print(f"{key:30s}  {mean[idx]:8.3f}  {std[idx]:8.3f}")
