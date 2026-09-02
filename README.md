# Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization

Code and experiment artifacts for the paper *Thinking Deeper, Not Longer:
Depth-Recurrent Transformers for Compositional Generalization*
([arXiv:2603.21676](https://arxiv.org/abs/2603.21676)).

A standard Transformer has a fixed computational depth set at architecture time.
This project studies a **depth-recurrent Transformer** that decouples computational
depth from parameter count: a single shared-weight block is applied iteratively for
*T* steps in latent space, so the model can trade recurrence steps for deeper
reasoning at inference time ("thinking deeper"), without emitting any intermediate
tokens ("not longer").

Across three compositional-reasoning tasks we observe a consistent **computational
frontier**: a diagonal boundary in the (task difficulty x thinking steps) accuracy
grid where performance jumps from chance to near-perfect once enough steps are
allocated. The tasks differ in the inductive bias of their input interface, and
this produces qualitatively different out-of-distribution (OOD) behaviour:
precise-but-brittle (graph), approximate-but-robust (logic), and autonomous latent
routing with no structural hints (relational text).

## Architecture

```
input tokens -> embedding -> [ ThinkingBlock  x T steps ] -> [CLS] readout -> prediction
                                    ^ shared weights
```

The recurrent **ThinkingBlock** (one pre-norm Transformer block, applied *T* times):

- **Gated recurrence** `h <- z * h_new + (1 - z) * h_old`, with the update gate
  bias initialised to **-2.0** (`sigma(-2) ~ 0.12`), biasing the block toward
  keeping its previous state. This creates a near-identity path that keeps
  gradients healthy across 20+ unrolled steps.
- **Depth embeddings**: a learnable per-step vector added at each iteration so the
  block knows which step it is on.
- **Silent-thinking objective**: cross-entropy is computed **only at the final
  step**. The model is never supervised on intermediate steps, forcing it to learn
  a genuine multi-step algorithm rather than an early-exit heuristic.
- **Randomised step count**: *T* is sampled from a range each batch, so the model
  is robust across compute budgets and can be unrolled beyond its training range.
- **RoPE + LayerScale** (sequence tasks only): rotary positions for relative-order
  awareness, LayerScale (`gamma` init `1e-4`) to protect fragile symbolic states
  from untrained-block noise early in training. The graph task uses a hard
  adjacency mask instead and needs neither.

## Tasks

All three tasks are **binary classification**; difficulty is a single integer
(hop count / nesting depth / relation-chain length) that the model must match
with enough thinking steps.

| | `graph/` | `nested-expr/` | `family-reason/` |
|---|---|---|---|
| Problem | given a directed graph and two nodes `s`, `r`, decide whether a path `s -> r` exists | given a nested boolean expression such as `!((T&F)\|(!(T\|F)))`, decide whether it evaluates to `True` | given a shuffled bag of kinship facts and a queried relation between two entities, decide whether that relation is correct |
| Difficulty axis | path length (hops) | expression nesting depth | length of the parent/child relation chain |
| Input bias | adjacency-masked attention (1 step = 1 hop) | full attention + RoPE | full attention + RoPE, facts shuffled |
| `d` / heads / `d_ff` | 128 / 4 / 256 | 256 / 8 / 1024 | 256 / 8 / 1024 |
| RoPE / LayerScale | no / no | yes / yes | yes / yes |
| Train step range `T` / `T_max` | [5, 8] / 20 | [4, 16] / 28 | [1, 12] / 20 |
| Train -> eval difficulty | 1-5 -> 12 hops | depth 1-8 -> 14 | depth 2-5 -> 9 |

The `family-reason` task is CLUTRR-inspired but synthetic: the correct relation
is a signed offset over parent (`+1`) / child (`-1`) hops rather than literal
kinship semantics, and hard negatives share the offset parity of the true
answer, so the model must actually track the running sum rather than count
words. All models are under 1M parameters; the scale is deliberate, to isolate
the recurrence mechanism from pretraining confounds.

## Repository layout

```
graph/
  graph_experiment.py        Experiment I: train + full-grid eval + heatmap.
                             Flags: --per-step-loss (alpha=1 objective),
                             --emb-ablation, --seed.
  plot_results.py            rebuild the paper heatmap PDF (no retraining)
  supervision_sweep.py       one (alpha, d, seed) point of the silent-thinking
                             vs intermediate-supervision interpolation sweep
  plot_supervision_sweep.py  aggregate sweep_results/ -> summary table + figures
  sweep_results/             per-(alpha, d, seed) JSON from the sweep
nested-expr/
  logic_experiment.py        Experiment II (flags: --emb-ablation)
  plot_results.py
family-reason/
  family_experiment.py       Experiment III
  embedding_ablation.py      depth-embedding OOD-confound diagnostic (family)
  plot_results.py
emb_ablation_common.py       shared code for the depth-embedding diagnostic
emb_ablation_report.py       aggregate the three embedding_ablation_results.json
run_*.sh                     Slurm submission templates (see "Cluster runs")
```

Each task directory keeps its committed outputs: the result figure
(`<task>_results.pdf`) and the accuracy grid behind it (an array literal at the
top of `plot_results.py`, plus `family_results.npy` for that task), the trained
seed-42 checkpoint (`*_model.pt`), and the depth-embedding diagnostic's outputs
(`embedding_ablation_results.json`, `gate_stats.png`,
`embedding_ablation_summary.png`). The supervision-sweep figures under `graph/`
are regenerated by `plot_supervision_sweep.py` from `graph/sweep_results/`.

## Setup

```bash
pip install -r requirements.txt   # torch>=2.0, numpy, matplotlib, networkx
```

A CUDA GPU is recommended; every experiment also runs on CPU (slower).

## Reproducing the results

Each experiment script generates its own data, trains, evaluates over the full
(difficulty x steps) grid, prints the table, and writes a heatmap. Run each from
its own directory (outputs are written next to the script):

```bash
cd graph         && python graph_experiment.py     # Experiment I  (Fig. 2)
cd nested-expr   && python logic_experiment.py      # Experiment II (Fig. 3)
cd family-reason && python family_experiment.py     # Experiment III (Fig. 4)
```

**Silent thinking vs. intermediate supervision** (paper Sec. 4.4 / Table III).
The blended objective is
`L(alpha) = (1 - alpha) * CE(final step) + (alpha / T) * sum_t CE(step t)`,
so `alpha = 0` is silent thinking and `alpha = 1` is full per-step supervision.
Run the grid of `(alpha, d, seed)` points, then aggregate:

```bash
cd graph
python supervision_sweep.py --alpha 0 --d-model 256 --seed 42   # one point
# ... sweep alpha in {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}, d in {128, 192, 256, 384},
#     seeds 42-46 (see run_supervision_sweep*.sh) ...
python plot_supervision_sweep.py    # prints the summary table, writes the figures
```

**Depth-embedding OOD-confound diagnostic** (paper Sec. 5.2): checks that
beyond-training-range stability is not an artifact of untrained depth-embedding
rows, by re-evaluating each trained checkpoint with the untrained rows replaced
by zeros / clamping / re-initialisation.

```bash
(cd graph        && python graph_experiment.py --emb-ablation)
(cd nested-expr  && python logic_experiment.py --emb-ablation)
(cd family-reason && python embedding_ablation.py)
python emb_ablation_report.py       # summary table across the three tasks
```

## Cluster runs

The `run_*.sh` scripts are **Slurm submission templates**, not turn-key scripts.
Before submitting, edit:

- `#SBATCH -A YOUR_ACCOUNT` -> your allocation,
- `ENV_BIN=/path/to/your/conda/env/bin` -> your Python environment,
- partition / time / GPU lines for your site.

Then `mkdir -p slurm_logs` and `sbatch run_<name>.sh` from the repo root. The
scripts write per-run logs under `slurm_logs/` (git-ignored).

## Citation

```bibtex
@article{chen2026thinking,
  title   = {Thinking Deeper, Not Longer: Depth-Recurrent Transformers for
             Compositional Generalization},
  author  = {Chen, Hung-Hsuan},
  journal = {arXiv preprint arXiv:2603.21676},
  year    = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
