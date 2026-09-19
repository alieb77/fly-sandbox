"""Build the reward-circuit neuron subset from FlyWire annotations.

Circuit of appetitive/reward learning in the Drosophila mushroom body (MB):
  - PAM  : dopaminergic neurons signalling REWARD (sugar, water) -> MB lobes
  - PPL1 : dopaminergic neurons signalling PUNISHMENT (for contrast)
  - KC   : Kenyon cells, intrinsic MB neurons (sparse odor code)
  - MBON : MB output neurons (drive approach / avoidance)
Output: data/reward_neurons.csv  (one row per neuron, with a 'role' tag)
"""
import os, pandas as pd
DATA = os.path.join(os.path.dirname(__file__), "data")
df = pd.read_csv(os.path.join(DATA, "neuron_annotations.tsv"), sep="\t", low_memory=False)

ct = df["cell_type"].astype(str)
hb = df["hemibrain_type"].astype(str)
cc = df["cell_class"].astype(str)


def tag(row_ct, row_hb, row_cc):
    s = f"{row_ct} {row_hb}"
    if "PAM" in s:
        return "DAN_PAM_reward"
    if "PPL1" in s:
        return "DAN_PPL1_punish"
    if "MBON" in s:
        return "MBON_output"
    if row_cc == "Kenyon_Cell" or "KC" in s:
        return "KC_kenyon"
    return None


df["role"] = [tag(a, b, c) for a, b, c in zip(ct, hb, cc)]
sub = df[df["role"].notna()].copy()

keep = ["root_id", "role", "cell_type", "hemibrain_type", "cell_class",
        "super_class", "top_nt", "top_nt_conf", "side", "soma_x", "soma_y", "soma_z"]
sub = sub[keep]

print("Reward-circuit neurons:", len(sub))
print("\nBy role:")
print(sub["role"].value_counts())
print("\nPredicted neurotransmitter (top_nt) by role:")
print(pd.crosstab(sub["role"], sub["top_nt"]))
print("\nSide distribution:")
print(pd.crosstab(sub["role"], sub["side"]))

out = os.path.join(DATA, "reward_neurons.csv")
sub.to_csv(out, index=False)
print("\nsaved ->", out)
