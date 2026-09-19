"""Export the reward circuit to a compact JSON the browser sandbox loads.

Output: sandbox/network.json
  N            : neuron count
  roles        : per-neuron role code (0=KC 1=MBON 2=PAM 3=PPL1)
  idxKC/MBON/PAM/PPL1 : global indices per class
  fast         : signed non-plastic synapses {pre,post,w}  (w = sign*syn_count)
  plastic      : KC->MBON synapses {kc,mbon,w}  (LOCAL idx into idxKC/idxMBON)
"""
import os, json
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")
OUTDIR = os.path.join(os.path.dirname(__file__), "sandbox")
os.makedirs(OUTDIR, exist_ok=True)

neur = pd.read_csv(os.path.join(DATA, "reward_neurons.csv")).reset_index(drop=True)
edges = pd.read_csv(os.path.join(DATA, "reward_edges.csv"))

idx = {rid: i for i, rid in enumerate(neur["root_id"].astype("int64"))}
N = len(neur)
role_code = {"KC_kenyon": 0, "MBON_output": 1,
             "DAN_PAM_reward": 2, "DAN_PPL1_punish": 3}
roles = neur["role"].map(role_code).to_numpy()

def gidx(code):
    return np.flatnonzero(roles == code).tolist()
idxKC, idxMBON, idxPAM, idxPPL1 = gidx(0), gidx(1), gidx(2), gidx(3)

nt_cols = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
e = edges[edges["syn_count"] >= 5].copy()
dom = e[nt_cols].to_numpy().argmax(1)
nt = np.array([c.replace("_avg", "") for c in nt_cols])
e["dom"] = nt[dom]
sign = {"ach": 1, "oct": 1, "gaba": -1, "glut": -1, "da": 0, "ser": 0}
e["sign"] = e["dom"].map(sign)

pre = e["pre_pt_root_id"].map(idx).to_numpy()
post = e["post_pt_root_id"].map(idx).to_numpy()
w = (e["sign"].to_numpy() * e["syn_count"].to_numpy()).astype(int)

is_kc = np.isin(pre, idxKC)
is_mbon = np.isin(post, idxMBON)
plastic = is_kc & is_mbon & (e["sign"].to_numpy() > 0)
fast = ~plastic

kc_local = {g: i for i, g in enumerate(idxKC)}
mbon_local = {g: i for i, g in enumerate(idxMBON)}

net = {
    "N": int(N),
    "roles": roles.astype(int).tolist(),
    "idxKC": idxKC, "idxMBON": idxMBON, "idxPAM": idxPAM, "idxPPL1": idxPPL1,
    "fast": {
        "pre": pre[fast].astype(int).tolist(),
        "post": post[fast].astype(int).tolist(),
        "w": w[fast].tolist(),
    },
    "plastic": {
        "kc": [kc_local[g] for g in pre[plastic]],
        "mbon": [mbon_local[g] for g in post[plastic]],
        "w": w[plastic].tolist(),
    },
}
out = os.path.join(OUTDIR, "network.json")
with open(out, "w") as f:
    json.dump(net, f, separators=(",", ":"))
sz = os.path.getsize(out) / 1e6
print(f"N={N}  KC={len(idxKC)} MBON={len(idxMBON)} PAM={len(idxPAM)} PPL1={len(idxPPL1)}")
print(f"fast edges={fast.sum()}  plastic KC->MBON edges={plastic.sum()}")
print(f"saved -> {out}  ({sz:.2f} MB)")
