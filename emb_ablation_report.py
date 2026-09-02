"""
Summarise the depth-embedding OOD confound diagnostic across the three
tasks. Reads:
  graph/embedding_ablation_results.json
  nested-expr/embedding_ablation_results.json
  family-reason/embedding_ablation_results.json
and prints a compact comparison table (baseline / zero / clamp / reinit
in the four difficulty x thinking-step-range quadrants) plus the gate
in-range vs beyond-range summary.

Pure file IO -- safe to run on the login node.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = [
    ("graph", "graph/embedding_ablation_results.json"),
    ("logic", "nested-expr/embedding_ablation_results.json"),
    ("family", "family-reason/embedding_ablation_results.json"),
]

QUADS = [
    ("id_diff__steps_in_range", "ID diff / steps in-range"),
    ("id_diff__steps_beyond_range", "ID diff / steps BEYOND"),
    ("ood_diff__steps_in_range", "OOD diff / steps in-range"),
    ("ood_diff__steps_beyond_range", "OOD diff / steps BEYOND"),
]


def variant_val(res, label, qkey):
    if label == "reinit":
        s = res["reinit_seed_summary"][qkey]
        return s["mean"], s["std"]
    return res["variants"][label]["quadrants"][qkey], 0.0


def main():
    for name, rel in TASKS:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            print(f"\n### {name}: {rel} NOT FOUND (job not finished?)")
            continue
        res = json.load(open(path))
        print("\n" + "=" * 72)
        print(f"  {name.upper()}   seed={res['seed']}  "
              f"t_max_trained={res['t_max_trained']}  "
              f"(depth_emb rows {res['depth_emb_rows']})")
        print(f"  ID {res['difficulty_label']}: {res['id_difficulties']}   "
              f"OOD: {res['ood_difficulties']}")
        print(f"  steps in-range: {res['steps_within_range']}   "
              f"beyond: {res['steps_beyond_range']}")
        wc = res.get("within_range_consistency", {})
        if wc:
            print(f"  within-range consistency (max |var-base|): "
                  f"{max(wc.values()):.2e}")
        print("-" * 72)
        print(f"  {'quadrant':<28}" + "".join(f"{v:>15}"
              for v in ["baseline", "zero", "clamp", "reinit_x3"]))
        for qkey, qlabel in QUADS:
            cells = []
            for label in ["baseline", "zero", "clamp", "reinit"]:
                v, e = variant_val(res, label, qkey)
                cells.append(f"{v:.3f}+-{e:.3f}" if e >= 5e-4 else f"{v:.3f}")
            print(f"  {qlabel:<28}" + "".join(f"{c:>15}" for c in cells))

        base = res["variants"]["baseline"]["quadrants"]["ood_diff__steps_beyond_range"]
        z = res["variants"]["zero"]["quadrants"]["ood_diff__steps_beyond_range"]
        c = res["variants"]["clamp"]["quadrants"]["ood_diff__steps_beyond_range"]
        r = res["reinit_seed_summary"]["ood_diff__steps_beyond_range"]["mean"]
        spread = max(abs(z - base), abs(c - base), abs(r - base))
        print("-" * 72)
        print(f"  HEADLINE spread on (OOD diff / steps BEYOND): {spread:.3f}")

        gate = res["gate_stats"]
        tmt = res["t_max_trained"]
        inr = [m for (t, m, s) in gate if t < tmt]
        byd = [m for (t, m, s) in gate if t >= tmt]
        if inr and byd:
            print(f"  gate z^(t) mean: in-range {sum(inr)/len(inr):.3f}  "
                  f"beyond {sum(byd)/len(byd):.3f}  "
                  f"delta {sum(byd)/len(byd) - sum(inr)/len(inr):+.3f}")
    print()


if __name__ == "__main__":
    main()
