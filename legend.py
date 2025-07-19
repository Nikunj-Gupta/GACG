import matplotlib.pyplot as plt

DISPLAY_NAMES = {
    "qmix-TGAT": "TIGER-MIX",
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

# Suppose these are your ordered keys from plot_all:
ordered_keys = [ "dicg-TGAT", "dicg", "casec", "GACG", "ltscg"]
#ordered_keys = ["qmix-TGAT",  "qmix",  "qgnn", "vdn", "graphmix"]

# Create one Line2D proxy per key (same color & linewidth as your real plots)
handles = [
    plt.Line2D([0], [0],
               color=COLOR_MAP[k],
               linewidth=2.0)
    for k in ordered_keys
]
labels = [DISPLAY_NAMES.get(k, k) for k in ordered_keys]

# Make a wide, short figure just for the legend
fig = plt.figure(figsize=(12, 2))
fig.legend(handles, labels,
           ncol=len(labels),     # all entries in one row
           loc='center',         # center of the figure
           frameon=False)        # no box around it

# Hide axes
plt.axis('off')
plt.tight_layout()
# Save it:
fig.savefig("legend_only.png", dpi=300, bbox_inches="tight")

# (Optional) display it
plt.show()
