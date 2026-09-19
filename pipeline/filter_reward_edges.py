"""Filter the full FlyWire edge list down to the reward microcircuit.

Reads proofread_connections_783.feather (852 MB, ~all neuron-neuron-neuropil
edges) memory-mapped, keeps edges where BOTH endpoints are reward-circuit
neurons {PAM, PPL1, KC, MBON}, annotates each edge with pre/post role, and
saves a small CSV.
"""
import os, sys
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.compute as pc

DATA = os.path.join(os.path.dirname(__file__), "data")
CONN = os.path.join(DATA, "proofread_connections_783.feather")
NEUR = os.path.join(DATA, "reward_neurons.csv")

neur = pd.read_csv(NEUR)
role = dict(zip(neur["root_id"].astype("int64"), neur["role"]))
ids = pa.array(list(role.keys()), type=pa.int64())

# 1. inspect schema
t0 = feather.read_table(CONN, memory_map=True)
print("edge table rows:", t0.num_rows)
print("columns:", t0.column_names)

# figure out column names (be tolerant of naming)
cols = t0.column_names
def find(*cands):
    for c in cands:
        if c in cols:
            return c
    return None
pre = find("pre_root_id", "pre_pt_root_id", "pre")
post = find("post_root_id", "post_pt_root_id", "post")
syn = find("syn_count", "n_syn", "count", "weight")
npil = find("neuropil", "neuropil_region", "region")
print(f"using pre={pre} post={post} syn={syn} neuropil={npil}")

# 2. filter: both endpoints in reward set
mask = pc.and_(pc.is_in(t0[pre], value_set=ids),
               pc.is_in(t0[post], value_set=ids))
sub = t0.filter(mask)
print("intra-circuit edges:", sub.num_rows)

df = sub.to_pandas()
df["pre_role"] = df[pre].map(role)
df["post_role"] = df[post].map(role)

# tidy column order
front = [c for c in [pre, post, syn, npil, "pre_role", "post_role"] if c]
nt_cols = [c for c in df.columns if c.endswith("_avg") or c in
           ("gaba", "ach", "glut", "oct", "ser", "da")]
df = df[front + [c for c in df.columns if c not in front]]

out = os.path.join(DATA, "reward_edges.csv")
df.to_csv(out, index=False)
print("saved ->", out, f"({len(df)} edges)")

# 3. role x role connectivity summary (sum of synapses)
print("\nSynapse-weighted role->role connectivity:")
piv = df.pivot_table(index="pre_role", columns="post_role",
                     values=syn, aggfunc="sum", fill_value=0)
print(piv.astype(int))
print("\nEdge-count role->role:")
piv2 = df.pivot_table(index="pre_role", columns="post_role",
                      values=syn, aggfunc="count", fill_value=0)
print(piv2.astype(int))
