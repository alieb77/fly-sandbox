import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(os.path.join(os.path.dirname(__file__), "sim_results.npz"))
rA, rB, w = d["rA_curve"], d["rB_curve"], d["w_curve"]
tr = np.arange(len(rA))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
fig.patch.set_facecolor("white")

# 1. behavioural learning curve
ax[0].plot(tr, rA, "o-", color="#59A14F", lw=2.2, label="odor A (rewarded)")
ax[0].plot(tr, rB, "s--", color="#4C78A8", lw=2, label="odor B (control)")
ax[0].set_xlabel("reward pairing (trial)"); ax[0].set_ylabel("MBON response (Hz)")
ax[0].set_title("Appetitive learning curve\n(MBON output = avoidance drive)")
ax[0].axvspan(0.5, len(rA)-0.5, color="#59A14F", alpha=0.05)
ax[0].legend(frameon=False); ax[0].set_ylim(0, max(rA.max(), rB.max())*1.15)
ax[0].grid(alpha=0.25)

# 2. synaptic mechanism
ax[1].plot(tr, w, "o-", color="#E45756", lw=2.2)
ax[1].set_xlabel("reward pairing (trial)")
ax[1].set_ylabel("mean KC(A)->MBON weight")
ax[1].set_title("Mechanism: dopamine depresses\nthe odor-A synapses")
ax[1].grid(alpha=0.25); ax[1].set_ylim(0, w.max()*1.1)

# 3. before / after bars
labels = ["odor A\n(rewarded)", "odor B\n(control)"]
before = [d["rA0"], d["rB0"]]; after = [d["rA1"], d["rB1"]]
x = np.arange(2); bw = 0.36
ax[2].bar(x-bw/2, before, bw, label="before", color="#BAB0AC")
ax[2].bar(x+bw/2, after, bw, label="after", color=["#59A14F", "#4C78A8"])
ax[2].set_xticks(x); ax[2].set_xticklabels(labels)
ax[2].set_ylabel("MBON response (Hz)")
ax[2].set_title("Before vs after reward learning")
ax[2].legend(frameon=False); ax[2].grid(alpha=0.25, axis="y")
for i,(b,a) in enumerate(zip(before, after)):
    ax[2].text(i+bw/2, a+0.2, f"{100*(a-b)/b:+.0f}%", ha="center",
               fontsize=9, fontweight="bold")

fig.suptitle("LIF simulation of the FlyWire mushroom-body reward circuit  |  "
             "5,596 real neurons, connectome-constrained  |  "
             "reward pairing -> odor-specific plasticity",
             fontsize=11.5, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "sim_results.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved ->", out)
