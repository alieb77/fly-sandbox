"""Schematic of the extracted reward microcircuit with REAL synapse counts."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(os.path.dirname(__file__), "reward_circuit.png")

# synapse-weighted role->role totals (from filter step)
W = {
    ("PAM", "KC"): 42911, ("PAM", "MBON"): 11092, ("PAM", "PAM"): 1401,
    ("PPL1", "KC"): 16366, ("PPL1", "MBON"): 4127,
    ("KC", "MBON"): 256719, ("KC", "PAM"): 85527, ("KC", "PPL1"): 39222,
    ("KC", "KC"): 379338,
    ("MBON", "KC"): 13206, ("MBON", "PAM"): 5480, ("MBON", "PPL1"): 3195,
    ("MBON", "MBON"): 19983,
}
pos = {"KC": (0.5, 0.72), "MBON": (0.5, 0.20),
       "PAM": (0.12, 0.46), "PPL1": (0.88, 0.46)}
color = {"KC": "#4C78A8", "MBON": "#E45756", "PAM": "#59A14F", "PPL1": "#B07AA1"}
label = {"KC": "KC\nKenyon cells\n(odor code, ACh)",
         "MBON": "MBON\noutput neurons\n(behavior)",
         "PAM": "PAM\nDA reward\n(sugar/water)",
         "PPL1": "PPL1\nDA punish"}

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor("white")

wmax = max(W.values())
def lw(w):
    return 0.6 + 5.5 * (np.log1p(w) / np.log1p(wmax))

# draw edges (skip tiny self loops for clarity except KC/MBON self)
for (a, b), w in W.items():
    x0, y0 = pos[a]; x1, y1 = pos[b]
    if a == b:
        continue
    rad = 0.18 if (a, b) in W and (b, a) in W else 0.0
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          connectionstyle=f"arc3,rad={rad}",
                          arrowstyle="-|>", mutation_scale=16,
                          lw=lw(w), color=color[a], alpha=0.75,
                          shrinkA=26, shrinkB=26, zorder=1)
    ax.add_patch(arr)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    off = 0.05 * (1 if rad >= 0 else -1)
    ax.text(mx + off * (y1 - y0) * 2, my - off * (x1 - x0) * 2,
            f"{w:,}", fontsize=8, color=color[a], ha="center",
            va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

# nodes
for n, (x, y) in pos.items():
    ax.add_patch(Circle((x, y), 0.075, fc=color[n], ec="white", lw=2, zorder=4))
    ax.text(x, y, label[n], ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=5)

ax.set_title("Drosophila mushroom-body REWARD circuit — FlyWire connectome (v783)\n"
             "arrow width & label = total synapses between cell classes  |  "
             "5,596 neurons, 573,036 intra-circuit edges",
             fontsize=11, pad=14)
ax.text(0.5, 0.015,
        "Reward learning: PAM dopamine (reward) reshapes KC→MBON synapses so odors "
        "predicting reward drive approach.\nPPL1 does the same for punishment. MBONs "
        "feed back onto the dopaminergic neurons (recurrent loop).",
        ha="center", va="bottom", fontsize=8.5, style="italic", color="#333")
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved ->", OUT)
