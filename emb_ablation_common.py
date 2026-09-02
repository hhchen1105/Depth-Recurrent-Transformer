"""
Shared logic for the depth-embedding OOD confound diagnostic.

Motivation
----------
The three thinking-transformer models use a learnable lookup table
`self.depth_emb = nn.Embedding(max_steps, d_model)` to tell the recurrent
block which iteration it is on. During training `n_steps` is only sampled
inside the training range, so depth-embedding rows for step indices at or
beyond the largest trained iteration are NEVER touched by a gradient --
they sit at (near) their random N(0, 1) initialisation.

The paper reports that the models still generalise to thinking-step counts
well beyond the training range. This module tests whether that OOD
stability is (a) a real architectural property (the recurrence is
insensitive to those untrained rows) or (b) an artefact of the particular
random draw that happened to be initialised.

Part A -- embedding replacement ablation
    For step indices >= t_max_trained, at inference time only, swap the
    depth embedding for one of:
      baseline : the trained model as-is (untrained rows = random init)
      zero     : e_t = 0            (inject no depth signal)
      clamp    : e_t = e_{t_max_trained - 1}  (reuse last trained vector)
      reinit   : e_t ~ fresh N(0, sigma) with a different seed
                 (sigma = std of the untrained rows; 3 seeds)
    Each variant re-runs the full (difficulty x thinking-steps) eval grid
    with the SAME trained weights; only the lookup is patched.

Part B -- gate behaviour logging
    While running the baseline grid at the largest step budget, record the
    mean and std of the update gate z^(t) at every thinking step t, inside
    and beyond the training range.

Nothing here retrains the model or changes the architecture.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


VARIANTS = ["baseline", "zero", "clamp", "reinit"]
REINIT_SEEDS = [1234, 5678, 9012]


# ----------------------------------------------------------------------
# Part A: depth-embedding patching
# ----------------------------------------------------------------------

def _set_depth_emb_mode(model, mode, t_max_trained, orig_weight, reinit_seed):
    """
    Overwrite model.thinking_block.depth_emb.weight in place for `mode`.
    Rows [0, t_max_trained) (the trained rows) are always kept exactly as
    trained; only rows [t_max_trained, max_steps) are patched.
    `orig_weight` is a CPU copy of the trained embedding table.
    """
    W = model.thinking_block.depth_emb.weight.data
    W.copy_(orig_weight.to(W.device, W.dtype))          # restore trained table
    if mode == "baseline":
        return
    if mode == "zero":
        W[t_max_trained:].zero_()
    elif mode == "clamp":
        W[t_max_trained:] = W[t_max_trained - 1].clone().unsqueeze(0)
    elif mode == "reinit":
        untrained_std = float(orig_weight[t_max_trained:].std())
        g = torch.Generator().manual_seed(reinit_seed)
        noise = torch.randn(
            W.shape[0] - t_max_trained, W.shape[1], generator=g
        )
        W[t_max_trained:] = (noise * untrained_std).to(W.device, W.dtype)
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _eval_grid(model, evaluate_fn, test_loaders, difficulties, steps):
    grid = np.zeros((len(difficulties), len(steps)))
    for i, d in enumerate(difficulties):
        for j, s in enumerate(steps):
            grid[i, j] = evaluate_fn(model, test_loaders[d], s)
    return grid


def _quadrant_means(grid, difficulties, steps, id_difficulties, t_max_trained):
    """
    Split the grid into difficulty {ID, OOD} x thinking-steps
    {within training range, beyond training range} and return the four
    cell means. A step count s is "within range" iff running s iterations
    only ever indexes trained depth-embedding rows, i.e. s <= t_max_trained.
    """
    diff_is_id = np.array([d in id_difficulties for d in difficulties])
    step_in_rng = np.array([s <= t_max_trained for s in steps])
    out = {}
    for dkey, dmask in (("id_diff", diff_is_id), ("ood_diff", ~diff_is_id)):
        for skey, smask in (("steps_in_range", step_in_rng),
                            ("steps_beyond_range", ~step_in_rng)):
            block = grid[np.ix_(dmask, smask)]
            out[f"{dkey}__{skey}"] = (
                float(block.mean()) if block.size else float("nan")
            )
    return out


# ----------------------------------------------------------------------
# Part B: gate recorder
# ----------------------------------------------------------------------

class GateRecorder:
    """
    Forward hook on `gate_proj` that accumulates z = sigmoid(gate_proj(.))
    statistics per thinking-step index. Assumes every forward pass of the
    model runs exactly `n_steps` recurrent iterations, so the step index
    can be tracked with a modulo counter (no per-batch reset needed).
    """

    def __init__(self, gate_proj, n_steps):
        self.n_steps = n_steps
        self.step = 0
        # step index -> [sum z, sum z^2, count]
        self.acc = {t: [0.0, 0.0, 0] for t in range(n_steps)}
        self._handle = gate_proj.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        z = torch.sigmoid(output.detach().float())
        s = self.acc[self.step]
        s[0] += float(z.sum())
        s[1] += float((z * z).sum())
        s[2] += z.numel()
        self.step = (self.step + 1) % self.n_steps

    def close(self):
        self._handle.remove()

    def table(self):
        rows = []
        for t in range(self.n_steps):
            s0, s1, c = self.acc[t]
            if c == 0:
                continue
            mean = s0 / c
            var = max(s1 / c - mean * mean, 0.0)
            rows.append((t, mean, var ** 0.5))
        return rows


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run_embedding_ablation(
    model,
    evaluate_fn,
    test_loaders,
    difficulties,
    steps,
    t_max_trained,
    id_difficulties,
    out_dir,
    task_name,
    seed,
    difficulty_label="difficulty",
    extra_meta=None,
):
    """
    model            : trained thinking-transformer (already on DEVICE, eval())
    evaluate_fn      : fn(model, loader, n_steps) -> accuracy in [0, 1]
    test_loaders     : dict {difficulty_value -> DataLoader}
    difficulties     : ordered list of difficulty values (grid rows)
    steps            : ordered list of thinking-step counts (grid cols)
    t_max_trained    : first depth-embedding row index never trained
                       (= largest #iterations ever run during training)
    id_difficulties  : difficulty values seen during training
    out_dir          : task directory to write outputs into
    task_name        : "graph" | "logic" | "family"
    seed             : training seed (for provenance / filenames)
    """
    model.eval()
    device = next(model.parameters()).device
    orig_weight = model.thinking_block.depth_emb.weight.data.detach().cpu().clone()
    max_row = model.thinking_block.depth_emb.weight.shape[0]

    ood_difficulties = [d for d in difficulties if d not in id_difficulties]
    suffix = "" if seed == 42 else f"_s{seed}"

    print("\n" + "=" * 70)
    print(f"  DEPTH-EMBEDDING OOD CONFOUND DIAGNOSTIC  ({task_name})")
    print("=" * 70)
    print(f"  depth_emb rows            : {max_row}")
    print(f"  trained rows (indices)    : 0..{t_max_trained - 1}")
    print(f"  untrained rows (indices)  : {t_max_trained}..{max_row - 1}")
    print(f"  ID {difficulty_label:<20}: {id_difficulties}")
    print(f"  OOD {difficulty_label:<19}: {ood_difficulties}")
    print(f"  thinking steps            : {steps}")
    steps_in = [s for s in steps if s <= t_max_trained]
    steps_beyond = [s for s in steps if s > t_max_trained]
    print(f"  steps within train range  : {steps_in}")
    print(f"  steps beyond train range  : {steps_beyond}")
    print("=" * 70)

    # -- Part A --------------------------------------------------------
    variant_runs = {}          # label -> {"grid", "quadrants"}
    for mode in VARIANTS:
        seeds = REINIT_SEEDS if mode == "reinit" else [None]
        for rs in seeds:
            label = mode if rs is None else f"reinit_s{rs}"
            _set_depth_emb_mode(model, mode, t_max_trained, orig_weight,
                                rs if rs is not None else 0)
            grid = _eval_grid(model, evaluate_fn, test_loaders,
                              difficulties, steps)
            quad = _quadrant_means(grid, difficulties, steps,
                                   id_difficulties, t_max_trained)
            variant_runs[label] = {"grid": grid.tolist(), "quadrants": quad}
            print(f"\n  [{label}]")
            _print_grid(grid, difficulties, steps, difficulty_label)
            for k, v in quad.items():
                print(f"      {k:<32}: {v:.4f}")

    # restore trained table before Part B
    _set_depth_emb_mode(model, "baseline", t_max_trained, orig_weight, 0)

    # sanity: for step counts within the training range no untrained row is
    # ever indexed, so every variant grid must match baseline on those columns.
    base_grid = np.array(variant_runs["baseline"]["grid"])
    in_cols = [j for j, s in enumerate(steps) if s <= t_max_trained]
    consistency = {}
    for label, run in variant_runs.items():
        if label == "baseline":
            continue
        diff = float(np.abs(np.array(run["grid"])[:, in_cols]
                            - base_grid[:, in_cols]).max())
        consistency[label] = diff
    max_incons = max(consistency.values()) if consistency else 0.0
    print(f"\n  [sanity] max |variant - baseline| on within-range step "
          f"columns: {max_incons:.2e}  (must be ~0)")

    # reinit aggregate (mean/std across the 3 seeds)
    reinit_labels = [f"reinit_s{s}" for s in REINIT_SEEDS]
    reinit_quads = {
        k: [variant_runs[l]["quadrants"][k] for l in reinit_labels]
        for k in variant_runs[reinit_labels[0]]["quadrants"]
    }
    reinit_summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in reinit_quads.items()
    }

    # -- Part B --------------------------------------------------------
    max_steps = max(steps)
    print(f"\n  [Part B] recording gate z^(t) at {max_steps} steps "
          f"over all {difficulty_label} buckets ...")
    recorder = GateRecorder(model.thinking_block.gate_proj, max_steps)
    for d in difficulties:
        evaluate_fn(model, test_loaders[d], max_steps)
    recorder.close()
    gate_rows = recorder.table()
    print("      t   mean_z   std_z   (| = train/beyond boundary)")
    for t, m, s in gate_rows:
        bar = " <" if t == t_max_trained else ""
        print(f"      {t:2d}  {m:.4f}  {s:.4f}{bar}")

    # -- Outputs ------------------------------------------------------
    results = {
        "task": task_name,
        "seed": seed,
        "depth_emb_rows": int(max_row),
        "t_max_trained": int(t_max_trained),
        "difficulty_label": difficulty_label,
        "test_difficulties": list(difficulties),
        "test_steps": list(steps),
        "id_difficulties": list(id_difficulties),
        "ood_difficulties": list(ood_difficulties),
        "steps_within_range": steps_in,
        "steps_beyond_range": steps_beyond,
        "variants": variant_runs,
        "reinit_seed_summary": reinit_summary,
        "within_range_consistency": consistency,
        "gate_stats": [[int(t), float(m), float(s)] for t, m, s in gate_rows],
        "meta": extra_meta or {},
    }
    res_path = os.path.join(out_dir, f"embedding_ablation_results{suffix}.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to {res_path}")

    _plot_gate_stats(gate_rows, t_max_trained, task_name,
                     os.path.join(out_dir, f"gate_stats{suffix}.png"))
    _plot_variant_summary(variant_runs, reinit_summary, task_name,
                          os.path.join(out_dir,
                                       f"embedding_ablation_summary{suffix}.png"))

    _print_headline(variant_runs, reinit_summary, task_name)
    return results


# ----------------------------------------------------------------------
# Printing / plotting helpers
# ----------------------------------------------------------------------

def _print_grid(grid, difficulties, steps, difficulty_label):
    header = f"      {difficulty_label[:8]:>8} \\ steps |" + \
             "".join(f" {s:>4} " for s in steps)
    print(header)
    for i, d in enumerate(difficulties):
        row = f"      {d:>16} |" + "".join(f" {grid[i, j]:.2f} "
                                           for j in range(len(steps)))
        print(row)


def _plot_gate_stats(gate_rows, t_max_trained, task_name, path):
    ts = [r[0] for r in gate_rows]
    means = np.array([r[1] for r in gate_rows])
    stds = np.array([r[2] for r in gate_rows])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, means, "-o", color="#1f77b4", label="mean z^(t)")
    ax.fill_between(ts, means - stds, means + stds, alpha=0.25,
                    color="#1f77b4", label="+/- 1 std")
    ax.axvline(t_max_trained - 0.5, color="crimson", ls="--", lw=1.5,
               label=f"train/beyond boundary (idx {t_max_trained})")
    ax.set_xlabel("thinking step t")
    ax.set_ylabel("update gate z^(t)")
    ax.set_title(f"Gate behaviour vs thinking step  ({task_name})\n"
                 f"indices >= {t_max_trained} use never-trained depth embeddings")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Gate plot written to {path}")


def _plot_variant_summary(variant_runs, reinit_summary, task_name, path):
    keys = ["id_diff__steps_in_range", "id_diff__steps_beyond_range",
            "ood_diff__steps_in_range", "ood_diff__steps_beyond_range"]
    klabels = ["ID diff\nsteps in-range", "ID diff\nsteps beyond",
               "OOD diff\nsteps in-range", "OOD diff\nsteps beyond"]
    bars = ["baseline", "zero", "clamp", "reinit"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    def val_err(bar, k):
        if bar == "reinit":
            return reinit_summary[k]["mean"], reinit_summary[k]["std"]
        return variant_runs[bar]["quadrants"][k], 0.0

    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for bi, bar in enumerate(bars):
        vals = [val_err(bar, k)[0] for k in keys]
        errs = [val_err(bar, k)[1] for k in keys]
        ax.bar(x + (bi - 1.5) * w, vals, w, yerr=errs, capsize=3,
               label=bar, color=colors[bi])
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(klabels)
    ax.set_ylabel("mean accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Depth-embedding replacement ablation  ({task_name})")
    ax.legend(fontsize=9, ncol=5, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Summary plot written to {path}")


def _print_headline(variant_runs, reinit_summary, task_name):
    k = "ood_diff__steps_beyond_range"
    print("\n" + "-" * 70)
    print(f"  HEADLINE ({task_name}): mean accuracy on OOD difficulty x "
          f"beyond-range thinking steps")
    print("-" * 70)
    for bar in ["baseline", "zero", "clamp"]:
        print(f"    {bar:<10}: {variant_runs[bar]['quadrants'][k]:.4f}")
    rs = reinit_summary[k]
    print(f"    {'reinit':<10}: {rs['mean']:.4f} +/- {rs['std']:.4f} "
          f"(over {len(REINIT_SEEDS)} seeds)")
    base = variant_runs["baseline"]["quadrants"][k]
    spread = max(
        abs(variant_runs["zero"]["quadrants"][k] - base),
        abs(variant_runs["clamp"]["quadrants"][k] - base),
        abs(reinit_summary[k]["mean"] - base),
    )
    print(f"    max |variant - baseline| = {spread:.4f}")
    print("    (small spread  -> OOD stability is a real architectural "
          "property)")
    print("    (large spread  -> untrained depth embeddings are a real "
          "confound)")
    print("-" * 70)
