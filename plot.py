import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- Configuration ---
#ROOT_DIR    = "gather_0,2,3,4,7-mixers"
#ROOT_DIR    = "gather_0,2,3,4,7-graphs"
ROOT_DIR    = "gather-ABLATIONS"
METRIC_NAME = "test_battle_won_mean"
SMOOTHING   = 0.75 
# SMOOTHING   = 15
MAX_STEPS   = 1_000_000

# Display names in bold
DISPLAY_NAMES = {
    "qmix-TGAT": "TIGER",
    "dicg-TGAT": "TIGER-DICG",
    "qmix":      "QMIX",
    "dicg":      "DICG",
    "qgnn":      "QGNN",
    "vdn":       "VDN",
    "graphmix":  "GraphMix",
    "casec":     "CASEC",
    "GACG":      "GACG",
    "ltscg":     "LTSCG",
}

# New colour assignments
COLOR_MAP = {
    "qmix-TGAT": "#9467bd",  # purple
    "dicg-TGAT": "#e377c2",  # pink
    "qmix":      "#2ca02c",  # green
    "dicg":      "#DAA520",  # goldenrod
    "qgnn":      "#d62728",  # red
    "vdn":       "#1f77b4",  # blue
    "graphmix":  "#ff7f0e",  # orange
    "casec":     "#17becf",  # cyan
    "GACG":      "#bcbd22",  # olive
    "ltscg":     "#7f7f7f",  # gray
}

params = {
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'legend.fontsize': 14,
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        #'text.usetex': True,
        'figure.figsize': [10, 8]
    } 

from pylab import rcParams

def ema_smooth(data, weight):
    """Apply exponential moving average smoothing to a 1D array."""
    smoothed = []
    last = data[0]
    for point in data:
        last = last * weight + (1 - weight) * point
        smoothed.append(last)
    return np.array(smoothed)

# def ema_smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     # return np.array(pd.Series(y).rolling(box_pts).mean())
#     return y_smooth

def extract_events(event_file):
    """Read a single TensorBoard event file and extract (step, value) arrays for METRIC_NAME."""
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags()
    if 'scalars' not in tags or METRIC_NAME not in tags['scalars']:
        print(f"[warn] Metric {METRIC_NAME} not in {event_file}")
        return None

    events = ea.Scalars(METRIC_NAME)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])

    mask = steps <= MAX_STEPS
    return steps[mask], values[mask]

def gather_experiments(root_dir):
    """Collect (steps, values) tuples grouped by algo and variant."""
    exp_data = defaultdict(list)
    pattern = re.compile(r'^gather__(?P<algo>[^_]+)__rnn__seed_(?P<seed>\d+)(?:_(?P<variant>.+))?$')
    for subdir in sorted(os.listdir(root_dir)):
        if not os.path.isdir(os.path.join(root_dir, subdir)):
            continue
        m = pattern.match(subdir)
        if not m:
            continue
        base_algo = m.group('algo')
        variant = m.group('variant')
        key = f"{base_algo}_{variant}" if variant else base_algo

        event_files = glob.glob(os.path.join(root_dir, subdir, 'events.out.tfevents*'))
        if not event_files:
            print(f"[warn] No event files in {subdir}")
            continue
        data = extract_events(sorted(event_files)[-1])
        if data is None:
            continue
        exp_data[key].append(data)

    return exp_data

def align_and_aggregate(runs):
    """Truncate to shortest run, compute mean+std, and smooth them."""
    min_len = min(len(vals) for _, vals in runs)
    steps = runs[0][0][:min_len]
    arr = np.stack([vals[:min_len] for _, vals in runs], axis=0)
    mean = ema_smooth(arr.mean(axis=0), SMOOTHING)
    std = ema_smooth(arr.std(axis=0), SMOOTHING)
    return steps, mean, std

def plot_all(exp_data):
    print("Summary of runs per variant:")
    for key, runs in sorted(exp_data.items()):
        print(f"  {key}: {len(runs)} run(s)")

    # plt.figure(figsize=(12, 7))
    plt.figure(figsize=(12, 10))

    # 1. TIGER first
    priority    = ["qmix-TGAT", "dicg-TGAT"]
    # 2. Then all others
    ordered_keys = [k for k in priority if k in exp_data] \
                 + [k for k in sorted(exp_data) if k not in priority]

    # 3. Loop over ordered_keys instead of exp_data.items()
    for key in ordered_keys:
        runs = exp_data[key]
        label = DISPLAY_NAMES.get(key, key)
        color = COLOR_MAP.get(key)

        if len(runs) == 1:
            steps, values = runs[0]
            mean = ema_smooth(values, SMOOTHING)
            std  = np.zeros_like(mean)
        else:
            steps, mean, std = align_and_aggregate(runs)

        plt.plot(steps, mean, label=label, color=color, linewidth=2.0)
        plt.fill_between(steps, mean - std, mean + std,
                         color=color, alpha=0.1)

    plt.xlim(0, 1000_000)
    plt.ylim(0, 1.1)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1], ["0.0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xticks([0, 200_000, 400_000, 600_000, 800_000, 1000_000], ["0.0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xlabel("Timesteps (in millions)", fontsize=26, fontweight="bold")
    plt.ylabel("Test win rate", fontsize=26, fontweight="bold")
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.title("Gather", fontsize=36, fontweight="bold", y=1.02)
    plt.legend(loc="upper left", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("rl_results_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    data = gather_experiments(ROOT_DIR)
    plot_all(data)
