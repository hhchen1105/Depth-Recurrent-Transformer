"""
Aggregate sweep_results/*.json (written by supervision_sweep.py) into the
figures for the supervision analysis (each written as both .pdf and .png):

  supervision_sweep_alpha       E1: metrics vs alpha at d_model = 128
  supervision_sweep_d256_alpha  E3: same dose-response at d_model = 256
                                    (the regime where the OOD cost resolves)
  supervision_sweep_capacity    E2: OOD acc / generalisation gap vs d_model
                                    for alpha in {0, 1}

Also prints a plain-text summary table (mean +/- std over seeds).
Pure matplotlib; no GPU required.
"""

import os
import glob
import json
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "sweep_results")

E1_DMODEL   = 128
E2_ALPHAS   = [0.0, 1.0]
ALPHA_LABEL = {0.0: r"$\alpha=0$ (silent thinking)",
               1.0: r"$\alpha=1$ (intermediate supervision)"}


def load_all():
    # (alpha, d_model, seed) -> row. Training is deterministic in these three, so
    # when a point exists under more than one tag (e.g. d=256 alpha in {0,1} was
    # run under both "e2" and the "e3" dose-response) the values are identical;
    # keep the first seen so the aggregates are not double-counted.
    by_point = {}
    for path in sorted(glob.glob(os.path.join(RES, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if d.get("tag") == "smoke":
            continue
        m = d["metrics"]
        key = (float(d["alpha"]), int(d["d_model"]), int(d["seed"]))
        if key in by_point:
            continue
        by_point[key] = dict(
            tag=d.get("tag", "?"),
            alpha=key[0], d_model=key[1], seed=key[2], n_params=d["n_params"],
            ood=m["ood_acc"], id=m["id_acc"],
            ood10=m.get("ood_per_hop", {}).get("10"),
            shortcut=m["step1_acc_12hop"],
            train_final=m["train_acc_final"],
            train_meanstep=m["train_acc_meanstep"],
            gap=m["gen_gap"], conv=m["conv_epoch_95"],
        )
    return list(by_point.values())


def agg(rows, key_fn, val_key):
    """Group rows by key_fn -> (sorted keys, mean array, std array, n array)."""
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r[val_key])
    keys = sorted(buckets)
    mean = np.array([np.mean(buckets[k]) for k in keys])
    std  = np.array([np.std(buckets[k]) for k in keys])
    n    = np.array([len(buckets[k]) for k in keys])
    return keys, mean, std, n


def plot_alpha_doseresponse(rows, d_model, out_name, panel):
    """alpha dose-response at a fixed width. `panel` is the short label (E1/E3)."""
    r = [x for x in rows if x["d_model"] == d_model]
    if not r:
        print("[%s] no d_model=%d rows; skipping" % (panel, d_model))
        return
    n_seeds = max(len(set(x["seed"] for x in r if x["alpha"] == a))
                  for a in set(x["alpha"] for x in r))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    series = [("ood", "OOD acc (6-12 hops)", "tab:red", "o"),
              ("id", "ID acc (1-4 hops)", "tab:green", "s"),
              ("train_final", "train acc (final step)", "tab:blue", "^"),
              ("train_meanstep", "train acc (mean over steps)", "tab:cyan", "v"),
              ("shortcut", "step-1 acc on 12 hops (shortcut)", "tab:orange", "D")]
    for key, lab, c, mk in series:
        a, mean, std, _ = agg(r, lambda x: x["alpha"], key)
        ax.errorbar(a, mean, yerr=std, label=lab, color=c, marker=mk,
                    capsize=3, linewidth=1.8)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(r"supervision weight $\alpha$", fontsize=13)
    ax.set_ylabel("accuracy", fontsize=13)
    ax.set_title(f"{panel}: metrics vs $\\alpha$  "
                 f"(d_model={d_model}, {n_seeds} seeds)", fontsize=13)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    a, gmean, gstd, _ = agg(r, lambda x: x["alpha"], "gap")
    ax2.errorbar(a, gmean, yerr=gstd, color="tab:purple", marker="o",
                 capsize=3, linewidth=1.8, label="train_final $-$ OOD")
    ax2.set_xlabel(r"supervision weight $\alpha$", fontsize=13)
    ax2.set_ylabel("generalisation gap", fontsize=13)
    ax2.set_title(f"{panel}: generalisation gap vs $\\alpha$", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"[{panel}] saved {out} (+ .png)")


def plot_e1(rows):
    plot_alpha_doseresponse(rows, E1_DMODEL, "supervision_sweep_alpha.pdf", "E1")


def plot_e3(rows):
    # d=256 dose-response (route-1 follow-up: the regime where the OOD cost of
    # intermediate supervision is actually resolvable).
    plot_alpha_doseresponse(rows, 256, "supervision_sweep_d256_alpha.pdf", "E3")


def plot_e2(rows):
    r = [x for x in rows if x["alpha"] in E2_ALPHAS]
    if not r:
        print("[E2] no alpha in {0,1} rows; skipping")
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = {0.0: "tab:blue", 1.0: "tab:orange"}

    for alpha in E2_ALPHAS:
        ra = [x for x in r if x["alpha"] == alpha]
        if not ra:
            continue
        d, mean, std, _ = agg(ra, lambda x: x["d_model"], "ood")
        ax.errorbar(d, mean, yerr=std, color=colors[alpha], marker="o",
                    capsize=3, linewidth=1.8, label=ALPHA_LABEL[alpha])
        d, gmean, gstd, _ = agg(ra, lambda x: x["d_model"], "gap")
        ax2.errorbar(d, gmean, yerr=gstd, color=colors[alpha], marker="o",
                     capsize=3, linewidth=1.8, label=ALPHA_LABEL[alpha])

    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xticks(sorted({x["d_model"] for x in r}))
        a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.set_xlabel("model width $d$", fontsize=13)
        a.legend(fontsize=10)
        a.grid(alpha=0.3)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_ylabel("OOD acc (6-12 hops)", fontsize=13)
    ax.set_title("E2: OOD accuracy vs capacity", fontsize=13)
    ax.set_ylim(0.4, 1.02)
    ax2.set_ylabel("generalisation gap (train_final $-$ OOD)", fontsize=13)
    ax2.set_title("E2: generalisation gap vs capacity", fontsize=13)

    fig.tight_layout()
    out = os.path.join(HERE, "supervision_sweep_capacity.pdf")
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"[E2] saved {out} (+ .png)")


def print_table(rows):
    print("\n=== supervision sweep summary (mean +/- std over seeds) ===")
    hdr = ("alpha", "d", "n", "ID", "OOD", "OOD_10hop", "shortcut", "train_fin",
           "train_mean", "gap", "conv95")
    print("  " + " ".join(f"{h:>9}" for h in hdr))
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["alpha"], r["d_model"])].append(r)
    for key in sorted(grp):
        g = grp[key]
        def ms(k):
            v = [x[k] for x in g if x[k] is not None]
            return (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)
        cells = [f"{key[0]:>9g}", f"{key[1]:>9d}", f"{len(g):>9d}"]
        for k in ("id", "ood", "ood10", "shortcut", "train_final", "train_meanstep",
                  "gap", "conv"):
            mu, sd = ms(k)
            cells.append(f"{mu:>6.3f}+/-{sd:>.2f}" if k != "conv"
                         else f"{mu:>6.1f}+/-{sd:>.1f}")
        print("  " + " ".join(cells))


def main():
    rows = load_all()
    if not rows:
        print(f"no result JSONs in {RES}")
        return
    print(f"loaded {len(rows)} runs from {RES}")
    print_table(rows)
    plot_e1(rows)
    plot_e3(rows)
    plot_e2(rows)


if __name__ == "__main__":
    main()
