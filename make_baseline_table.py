"""
Aggregate baseline_results.json from all three tasks (+ the full-budget "ours"
results already saved in each experiment dir) into the LaTeX comparison table.

Metric (robust across tasks, no fixed 90% threshold):
  ID Acc  = mean over in-distribution difficulty levels of the best accuracy
            achieved at any sufficient step budget.
  OOD Acc = same, averaged over out-of-distribution difficulty levels.

Prints a summary and the LaTeX tabular rows for Table~\\ref{tab:baselines}.
"""

import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# -- difficulty buckets per task ---------------------------------------
BUCKETS = {
    "graph":  dict(id=[1, 2, 3, 4],  ood=[6, 8, 10, 12]),
    "logic":  dict(id=[2, 4, 6, 8],  ood=[10, 12, 14]),
    "family": dict(id=[2, 3, 4, 5],  ood=[6, 7, 8, 9]),
}

# -- "ours" full-budget grids: one per seed, best over thinking steps per
#    difficulty. Each ours_<task>() returns a LIST of {difficulty: acc} dicts
#    (one per seed) so the table can report a 3-seed mean +/- std, matching the
#    baseline rows. Falls back to a single hard-coded literal (the paper's
#    primary run) when no per-seed result files are present yet.

# Graph: silent-thinking (alpha=0), d=128 -- the main-model config -- from the
# supervision sweep (graph/sweep_results/e1_a0_d128_s*.json).
GRAPH_HOPS_LIT = [1, 2, 3, 4, 6, 8, 10, 12]
GRAPH_FINAL_GRID_LIT = [
    [0.97,0.97,1,1,1,1,1,1],[0.51,0.98,1,1,1,1,1,1],[0.51,0.54,1,1,1,1,1,1],
    [0.52,0.54,0.50,1,1,1,1,1],[0.50,0.52,0.50,0.50,1,1,1,1],
    [0.51,0.54,0.50,0.50,0.50,0.97,1,1],[0.51,0.52,0.50,0.50,0.50,0.50,0.50,0.50],
    [0.51,0.53,0.50,0.50,0.50,0.50,0.50,0.50]]

LOGIC_DEPTHS_LIT = [2, 4, 6, 8, 10, 12, 14]
LOGIC_GRID_LIT = [
    [0.59,0.82,1,1,1,1,1,1,1,1],[0.54,0.74,0.94,0.99,1,1,1,1,1,1],
    [0.56,0.67,0.91,0.97,0.98,0.99,0.99,1,0.99,0.99],
    [0.58,0.73,0.90,0.96,0.98,0.97,0.97,0.98,0.97,0.97],
    [0.58,0.64,0.82,0.88,0.92,0.93,0.94,0.96,0.95,0.94],
    [0.58,0.66,0.85,0.89,0.92,0.93,0.93,0.92,0.93,0.92],
    [0.54,0.67,0.82,0.89,0.91,0.92,0.90,0.91,0.90,0.90]]

FAMILY_DEPTHS_LIT = [2, 3, 4, 5, 6, 7, 8, 9]


def _grid_best(rows, difficulties):
    """rows: 2D array/list (difficulty x steps). -> {difficulty: best acc over steps}."""
    return {d: float(np.max(row)) for d, row in zip(difficulties, rows)}


def ours_graph():
    import glob
    paths = sorted(glob.glob(os.path.join(HERE, "graph", "sweep_results",
                                          "e1_a0_d128_s*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        out.append(_grid_best(d["grid"], d["test_hops"]))
    return out or [_grid_best(GRAPH_FINAL_GRID_LIT, GRAPH_HOPS_LIT)]


def ours_logic():
    import glob
    paths = sorted(glob.glob(os.path.join(HERE, "nested-expr", "logic_grid_s*.npy")))
    if paths:
        return [_grid_best(np.load(p), LOGIC_DEPTHS_LIT) for p in paths]
    return [_grid_best(LOGIC_GRID_LIT, LOGIC_DEPTHS_LIT)]


def ours_family():
    import glob
    d = os.path.join(HERE, "family-reason")
    paths = sorted(set(glob.glob(os.path.join(d, "family_results_s*.npy"))
                       + glob.glob(os.path.join(d, "family_results.npy"))))
    # de-dup: seed 42 is written to both names
    grids, seen = [], set()
    for p in paths:
        arr = np.load(p)
        key = arr.tobytes()
        if key in seen:
            continue
        seen.add(key)
        grids.append(_grid_best(arr, FAMILY_DEPTHS_LIT))
    return grids

OURS = {"graph": ours_graph, "logic": ours_logic, "family": ours_family}


def ours_params(task):
    """Trainable-parameter count of our full model per task -- the count printed
    by instantiating the model class in each experiment script and summing
    parameters() with requires_grad:
        graph : GraphThinkingTransformer(d=128)   -> graph_experiment.py
        logic : LogicThinkingTransformer()  (d=256) -> nested-expr/logic_experiment.py
        family: FamilyThinkingTransformer() (d=256) -> family-reason/family_experiment.py
    All three round to the P column of tab:baselines (0.2 / 1.0 / 1.0 M).
    """
    return {"graph": 205_441, "logic": 998_145, "family": 1_007_617}[task]


def bucket_means(grid, task):
    """grid: {difficulty:int -> best acc}. Returns (id_acc, ood_acc)."""
    g = {int(k): v for k, v in grid.items()}
    b = BUCKETS[task]
    id_acc = np.mean([g[d] for d in b["id"] if d in g])
    ood_acc = np.mean([g[d] for d in b["ood"] if d in g])
    return float(id_acc), float(ood_acc)


def load_baselines(task, dirname):
    """Mean over all baseline_results*.json in the task dir (graph has 3 seeds;
    logic/family a single file). Returns name -> (id_acc, ood_acc, params, ood_std)."""
    import glob
    d = os.path.join(HERE, dirname)
    paths = [os.path.join(d, "baseline_results.json")] + sorted(
        glob.glob(os.path.join(d, "baseline_results_s*.json")))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return {}
    per_name = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for name, v in data["results"].items():
            id_acc, ood_acc = bucket_means(v["grid"], task)
            per_name.setdefault(name, []).append((id_acc, ood_acc, v.get("params")))
    out = {}
    for name, runs in per_name.items():
        ids = np.mean([r[0] for r in runs])
        oods = [r[1] for r in runs]
        out[name] = (ids, float(np.mean(oods)), runs[0][2], float(np.std(oods)))
    return out


def load_act(task, dirname):
    """Genuine Universal Transformer with ACT halting + ponder cost.
    Reads <dirname>/act_results.json ({seed: {params, grid:{difficulty:acc}}})
    written by ut_act.py (graph) / ut_act_seq.py (logic, family).
    Returns {} if absent, else {"Universal Transformer (ACT)": (id,ood,params,ood_std)}."""
    path = os.path.join(HERE, dirname, "act_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    ids, oods, params = [], [], None
    for _seed, v in sorted(data.items()):
        id_acc, ood_acc = bucket_means(v["grid"], task)
        ids.append(id_acc); oods.append(ood_acc)
        params = v.get("params", params)
    return {"Universal Transformer (ACT)":
            (float(np.mean(ids)), float(np.mean(oods)), params, float(np.std(oods)))}


def main():
    tasks = [("graph", "graph"), ("logic", "nested-expr"), ("family", "family-reason")]
    # method label -> {task: (id,ood)}
    table = {}

    def put(label, task, vals):
        table.setdefault(label, {})[task] = vals

    for task, dirname in tasks:
        bl = load_baselines(task, dirname)
        bl.update(load_act(task, dirname))
        for name, vals in bl.items():
            # unify naming across tasks
            label = {"Fixed-Depth Transformer": "Graph Transformer"}.get(name, name)
            put(label, task, vals)
        ours_pairs = [bucket_means(g, task) for g in OURS[task]()]
        ids = [p[0] for p in ours_pairs]
        oods = [p[1] for p in ours_pairs]
        put("Depth-Recurrent Transformer (ours)", task,
            (float(np.mean(ids)), float(np.mean(oods)), ours_params(task),
             float(np.std(oods))))

    # "Weight-Tied Transformer" is our recurrent core with LayerScale and the
    # gate removed -- printed last, as an ablation row under "ours", not as a
    # standalone baseline.
    order = ["Fixed-Depth GNN", "Graph Transformer",
             "Universal Transformer (ACT)",
             "Depth-Recurrent Transformer (ours)", "Weight-Tied Transformer"]

    print("\n=== ID / OOD accuracy by method and task ===")
    hdr = f"{'Method':38s} | " + " | ".join(f"{t:^13}" for t, _ in tasks)
    print(hdr); print("-" * len(hdr))
    for label in order:
        row = f"{label:38s} | "
        cells = []
        for task, _ in tasks:
            v = table.get(label, {}).get(task)
            cells.append(f"{v[0]*100:4.0f}/{v[1]*100:4.0f}" if v else f"{'--':^11}")
        print(row + " | ".join(f"{c:^13}" for c in cells))

    # -- LaTeX: the exact body of tab:baselines, copy-paste ready ----------
    # Formatting mirrors the paper: P to 1 decimal, ID/OOD as integers, and
    # OOD "{\pm}std" (integer percent) only when std rounds to >= 1 -- per the
    # caption, "omitted when it rounds to zero".
    def cell(v):
        if not v:
            return r"--- & --- & ---"
        p = f"${v[2] / 1e6:.1f}$" if len(v) > 2 and v[2] else "---"
        ood = f"{v[1] * 100:.0f}"
        if len(v) > 3 and v[3] is not None and round(v[3] * 100) >= 1:
            ood += f"{{\\pm}}{v[3] * 100:.0f}"
        return f"{p} & ${v[0] * 100:.0f}$ & ${ood}$"

    # (internal name in `table`, displayed model name) in paper row order.
    layout = [
        ("group", r"\multirow{2}{2.1cm}{\emph{Fixed depth}}"),
        ("Fixed-Depth GNN", "Fixed-depth GNN"),
        ("Graph Transformer", "Fixed-depth Transformer"),
        ("midrule", None),
        ("group", r"\multirow{3}{2.1cm}{\emph{Weight-sharing / recurrent}}"),
        ("Universal Transformer (ACT)",
         r"Universal Transformer (ACT)~\cite{dehghani2018universal,graves2016adaptive}"),
        ("Depth-Recurrent Transformer (ours)", "Depth-Recurrent (ours)"),
        ("Weight-Tied Transformer",
         "Depth-Recurrent w/o LayerScale, w/o gate (ablation)"),
    ]
    print("\n=== tab:baselines body (paste into exp.tex) ===")
    for key, disp in layout:
        if key == "group":
            print(f"        {disp}")
        elif key == "midrule":
            print(r"        \midrule")
        else:
            cells = " & ".join(cell(table.get(key, {}).get(task))
                               for task, _ in tasks)
            print(f"        & {disp} & {cells} \\\\")


if __name__ == "__main__":
    main()
