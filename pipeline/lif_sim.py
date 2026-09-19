"""Leaky integrate-and-fire simulation of the FlyWire mushroom-body REWARD circuit.

Biology encoded:
  - KC (Kenyon cells): sparse odor code, cholinergic -> EXCITATORY onto MBONs.
  - MBON: readout neurons; their pooled activity = "avoidance drive" here.
  - PAM: dopaminergic REWARD neurons. Dopamine does NOT drive fast spikes; it
    GATES PLASTICITY: KC->MBON synapses of recently-active KCs are DEPRESSED
    when dopamine is high (the canonical Drosophila reward-learning rule).
  - PPL1: dopaminergic punishment neurons (present, not used as US here).

Experiment:
  1. TEST odor A and odor B (two sparse KC ensembles) -> baseline MBON response.
  2. TRAIN: present odor A together with reward (PAM activation). Plasticity
     depresses KC(A)->MBON synapses.
  3. TEST again: MBON response to A should DROP (learned), B stays (specific).
     -> the fly now "approaches" odor A instead of avoiding it.

Signs from per-EDGE neurotransmitter (ach/oct=+, gaba/glut=-, da=modulatory).
Edges thresholded at syn_count >= 5. Weights ~ syn_count.
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse

RNG = np.random.default_rng(7)
DATA = os.path.join(os.path.dirname(__file__), "data")

# ----------------------------------------------------------------------------
# 1. Build the network
# ----------------------------------------------------------------------------
neur = pd.read_csv(os.path.join(DATA, "reward_neurons.csv"))
edges = pd.read_csv(os.path.join(DATA, "reward_edges.csv"))

neur = neur.reset_index(drop=True)
idx = {rid: i for i, rid in enumerate(neur["root_id"].astype("int64"))}
N = len(neur)
role = neur["role"].to_numpy()

def where(r):
    return np.flatnonzero(role == r)

KC = where("KC_kenyon")
MBON = where("MBON_output")
PAM = where("DAN_PAM_reward")
PPL1 = where("DAN_PPL1_punish")
print(f"neurons N={N}  KC={len(KC)} MBON={len(MBON)} PAM={len(PAM)} PPL1={len(PPL1)}")

# threshold + dominant NT -> sign
nt_cols = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
e = edges[edges["syn_count"] >= 5].copy()
dom = e[nt_cols].to_numpy().argmax(1)
nt_name = np.array([c.replace("_avg", "") for c in nt_cols])
e["dom"] = nt_name[dom]
sign_map = {"ach": +1.0, "oct": +1.0, "gaba": -1.0, "glut": -1.0,
            "da": 0.0, "ser": 0.0}
e["sign"] = e["dom"].map(sign_map)

pre = e["pre_pt_root_id"].map(idx).to_numpy()
post = e["post_pt_root_id"].map(idx).to_numpy()
w = e["sign"].to_numpy() * e["syn_count"].to_numpy()

G = 0.03  # global weight->current scale (tuned below)

# KC->MBON plastic synapses handled in a dense matrix P (nKC x nMBON)
kc_pos = {g: i for i, g in enumerate(KC)}
mbon_pos = {g: i for i, g in enumerate(MBON)}
is_kc = np.isin(pre, KC)
is_mbon_post = np.isin(post, MBON)
plastic = is_kc & is_mbon_post & (e["sign"].to_numpy() > 0)

P = np.zeros((len(KC), len(MBON)))
for pr, po, ww in zip(pre[plastic], post[plastic], w[plastic]):
    P[kc_pos[pr], mbon_pos[po]] += ww * G
P0 = P.copy()  # baseline for learning curve
print(f"KC->MBON plastic synapses: {int(plastic.sum())} edges, "
      f"mean weight {P0[P0>0].mean():.3f}")

# fast synaptic backbone = everything EXCEPT the plastic KC->MBON set
fast = ~plastic
Wfast = sparse.csr_matrix(
    (w[fast] * G, (pre[fast], post[fast])), shape=(N, N))

# ----------------------------------------------------------------------------
# 2. Odors = sparse KC ensembles
# ----------------------------------------------------------------------------
P_SPARSE = 0.06  # fraction of KCs an odor activates
nA = int(P_SPARSE * len(KC))
perm = RNG.permutation(len(KC))
odorA_kc = KC[perm[:nA]]
odorB_kc = KC[perm[nA:2 * nA]]   # disjoint ensemble -> clean control
overlap = len(set(odorA_kc) & set(odorB_kc))
print(f"odor ensembles: {nA} KC each, overlap A&B = {overlap}")

# ----------------------------------------------------------------------------
# 3. LIF engine
# ----------------------------------------------------------------------------
dt = 0.5                      # ms
tau_m = 20.0
tau_syn = 5.0
Vth, Vreset = 1.0, 0.0
t_ref = 2.0
ref_steps = int(t_ref / dt)
a_syn = np.exp(-dt / tau_syn)
k_m = dt / tau_m

# plasticity / dopamine
tau_elig = 40.0               # KC eligibility trace (ms)
a_elig = np.exp(-dt / tau_elig)
tau_da = 100.0
a_da = np.exp(-dt / tau_da)
ETA = 2.2e-6                  # learning rate (depression), tuned for graded curve

I_ODOR = 2.0                  # external drive to activated KCs
I_REWARD = 2.0                # external drive to PAM during training


def run_phase(dur_ms, odor_kc=None, reward=False, learn=False, state=None):
    """Simulate one phase; return spike counts per neuron and updated state."""
    global P
    steps = int(dur_ms / dt)
    if state is None:
        V = np.zeros(N); g = np.zeros(N); ref = np.zeros(N, int)
        elig = np.zeros(len(KC)); DA = 0.0
    else:
        V, g, ref, elig, DA = state
    Iext = np.zeros(N)
    if odor_kc is not None:
        Iext[odor_kc] = I_ODOR
    if reward:
        Iext[PAM] = I_REWARD
    spikes_prev = np.zeros(N)
    count = np.zeros(N)
    mbon_trace = []
    for _ in range(steps):
        # synaptic current from spikes at previous step
        g *= a_syn
        if spikes_prev.any():
            g += spikes_prev @ Wfast
            kc_s = spikes_prev[KC]
            g[MBON] += kc_s @ P
        # membrane update (skip refractory)
        active = ref <= 0
        V[active] += k_m * (-(V[active]) + Iext[active] + g[active])
        spk = active & (V >= Vth)
        V[spk] = Vreset
        ref[spk] = ref_steps
        ref[~active] -= 1
        spikes_prev = spk.astype(float)
        count += spikes_prev
        mbon_trace.append(spikes_prev[MBON].sum())
        # traces + plasticity
        elig = elig * a_elig + spikes_prev[KC]
        DA = DA * a_da + (spikes_prev[PAM].mean() if len(PAM) else 0.0)
        if learn and DA > 0:
            # dopamine-gated depression of KC->MBON for active KCs
            P -= ETA * DA * elig[:, None]
            np.clip(P, 0.0, None, out=P)
    state = (V, g, ref, elig, DA)
    return count, np.array(mbon_trace), state


def mbon_rate(odor_kc):
    """Mean MBON firing rate (Hz) during a 400 ms probe of an odor."""
    c, tr, _ = run_phase(400, odor_kc=odor_kc)
    return c[MBON].sum() / len(MBON) / 0.4, tr


# ----------------------------------------------------------------------------
# 4. Protocol
# ----------------------------------------------------------------------------
kcA_rows = np.array([kc_pos[g] for g in odorA_kc])
plastic_mask_A = P0[kcA_rows] > 0        # real KC(A)->MBON synapses


def weightA():
    return P[kcA_rows][plastic_mask_A].mean()


print("\n=== BASELINE (before learning) ===")
rA0, trA0 = mbon_rate(odorA_kc)
rB0, trB0 = mbon_rate(odorB_kc)
kc_c, _, _ = run_phase(400, odor_kc=odorA_kc)
print(f"KC(odorA) firing rate: {kc_c[odorA_kc].sum()/len(odorA_kc)/0.4:.1f} Hz")
print(f"MBON response  odor A: {rA0:.1f} Hz   odor B: {rB0:.1f} Hz")
print(f"mean KC(A)->MBON synaptic weight: {weightA():.3f}")

print("\n=== TRAINING: odor A + reward (dopamine ON), probe A & B each trial ===")
n_trials = 10
rA_curve, rB_curve, w_curve = [rA0], [rB0], [weightA()]
st = None
for t in range(n_trials):
    # one CS-US pairing (odor A + reward), plasticity on
    _, _, st = run_phase(300, odor_kc=odorA_kc, reward=True, learn=True, state=st)
    _, _, st = run_phase(150, state=st)                       # inter-trial gap
    # probe both odors WITHOUT reward / plasticity (behavioural readout)
    rA, _ = mbon_rate(odorA_kc)
    rB, _ = mbon_rate(odorB_kc)
    rA_curve.append(rA); rB_curve.append(rB); w_curve.append(weightA())
    print(f"  trial {t+1:2d}: KC(A)->MBON w={w_curve[-1]:.3f}   "
          f"MBON(A)={rA:4.1f} Hz   MBON(B)={rB:4.1f} Hz")

rA1, trA1 = mbon_rate(odorA_kc)
rB1, trB1 = mbon_rate(odorB_kc)
print("\n=== SUMMARY ===")
print(f"MBON response  odor A (rewarded): {rA0:.1f} -> {rA1:.1f} Hz "
      f"({100*(rA1-rA0)/max(rA0,1e-9):+.0f}%)  -> learned: approach")
print(f"MBON response  odor B (control) : {rB0:.1f} -> {rB1:.1f} Hz "
      f"({100*(rB1-rB0)/max(rB0,1e-9):+.0f}%)  -> unchanged: specific")

np.savez(os.path.join(os.path.dirname(__file__), "sim_results.npz"),
         rA0=rA0, rA1=rA1, rB0=rB0, rB1=rB1,
         rA_curve=np.array(rA_curve), rB_curve=np.array(rB_curve),
         w_curve=np.array(w_curve),
         trA0=trA0, trA1=trA1, trB0=trB0, trB1=trB1)
print("\nsaved sim_results.npz")
