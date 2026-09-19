# 🪰 Fly Sandbox — a connectome-driven virtual fruit fly

An interactive 3D sandbox where a stylized *Drosophila* is driven by a **live
simulation of 5,596 real neurons** taken from the
[FlyWire](https://flywire.ai) whole-brain connectome — specifically the
**mushroom-body reward circuit** (appetitive learning).

Drop odor sources in the arena, lead the fly to one, then hold **🍬 Reward**
(dopaminergic PAM neurons) or **⚡ Punish** (PPL1). The KC→MBON synapses
re-weight in real time and the fly *learns* to approach or avoid that odor —
specifically for the trained odor, exactly as a real fly does.

## What is real vs modelled

| Real (FlyWire connectome v783) | Added by the model |
|---|---|
| Who connects to whom (573k synapses) | Leaky integrate-and-fire dynamics |
| Sign of each synapse (per-edge neurotransmitter) | Firing threshold, refractory period |
| Weights ∝ synapse counts | Dopamine-gated KC→MBON plasticity |
| Cell classes PAM / PPL1 / KC / MBON | Odor = sparse KC ensemble |

The wiring is data; only the *dynamics* are modelled. This is a
**connectome-constrained** model.

## Circuit

- **KC** (Kenyon cells, 5177): sparse odor code, cholinergic → excitatory.
- **MBON** (96): output neurons; pooled activity read out as "avoidance drive".
- **PAM** (307): dopaminergic **reward** neurons → depress active KC→MBON synapses.
- **PPL1** (16): dopaminergic **punishment** neurons → potentiate them.

## Run locally

Static site — just serve the folder:

```bash
python -m http.server 4607
# open http://localhost:4607
```

## Pipeline (provenance)

`pipeline/` contains the Python that produced `network.json`:

1. `download.py` — fetch FlyWire annotations (GitHub) + edge list (Zenodo 10676866).
2. `build_reward_set.py` — tag the 5,596 reward-circuit neurons.
3. `filter_reward_edges.py` — extract the 573k intra-circuit synapses.
4. `lif_sim.py` — the reference LIF simulation + learning experiment.
5. `export_network.py` — compact `network.json` for the browser.

`index.html` re-implements the LIF engine in JavaScript (Three.js) to run it live.

## Data & credits

- Connectome: FlyWire (Dorkenwald, Schlegel et al., *Nature* 2024); annotations
  from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations);
  edges from [Zenodo 10676866](https://zenodo.org/records/10676866).
- Original EM reconstruction: FAFB-FFN1 (Google Research / Janelia).

Educational project. Not affiliated with FlyWire or Google.
