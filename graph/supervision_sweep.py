"""
E1 (supervision-weighting sweep) + E2 (capacity x supervision) on graph reachability.

One process trains ONE (alpha, d_model, seed) configuration and writes a JSON with
the full training history, the hop x step accuracy grid, and the derived metrics
used by plot_supervision_sweep.py.

Objective (generalises the two extremes used in the paper):

    L(alpha) = (1 - alpha) * CE(y_T)  +  alpha * mean_t CE(y_t)

    alpha = 0  -> silent thinking / final-step-only  (== graph_experiment.py default)
    alpha = 1  -> intermediate supervision           (== graph_experiment.py --per-step-loss)

The full grid (both run scripts) is alpha in {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
crossed with d_model in {32, 64, 128, 192, 256, 384}: the alpha dose-response at
d = 128 and 256, and the alpha in {0, 1} capacity trace over all widths.

This is a GPU training job (one point takes a few minutes on a GPU). Use
run_supervision_sweep.sh / run_supervision_sweep_d256.sh for the full grid.
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# graph_experiment defines the dataset, model and eval loop; reuse them so this
# sweep stays in lockstep with the headline run.
from graph_experiment import (
    GraphReachabilityDataset,
    GraphThinkingTransformer,
    evaluate,
    VOCAB_SIZE,
    DEVICE,
)

# -- Evaluation grid (identical to graph_experiment.py) --
TEST_HOPS  = [1, 2, 3, 4, 6, 8, 10, 12]
TEST_STEPS = [1, 2, 3, 5, 8, 12, 15, 20]
ID_HOPS    = [1, 2, 3, 4]
OOD_HOPS   = [6, 8, 10, 12]

# -- Fixed training config (matches the graph_experiment.py headline run) --
MAX_NODES        = 20
MAX_THINK        = 20
TRAIN_THINK      = 5
BATCH_SIZE       = 64
LR               = 3e-4
WEIGHT_DECAY     = 1e-2
TRAIN_HOP_RANGE  = (1, 5)
TRAIN_NODES      = (8, 16)
TRAIN_STEP_RANGE = (5, 8)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_test_loaders(test_samples_per: int):
    loaders = {}
    for hops in TEST_HOPS:
        n_min = max(hops + 2, 10)
        n_max = min(max(hops + 6, 16), MAX_NODES)
        n_min = min(n_min, n_max)
        ds = GraphReachabilityDataset(
            n_samples=test_samples_per,
            hop_range=(hops, hops),
            num_nodes_range=(n_min, n_max),
            max_nodes=MAX_NODES,
        )
        loaders[hops] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loaders


def blended_loss(logits_all, labels, alpha):
    """logits_all: (T, B) tensor of per-step readout logits."""
    per_step = torch.stack([
        F.binary_cross_entropy_with_logits(lg, labels) for lg in logits_all])
    final = per_step[-1]
    mean = per_step.mean()
    return (1.0 - alpha) * final + alpha * mean


@torch.no_grad()
def eval_grid(model, loaders):
    grid = np.zeros((len(TEST_HOPS), len(TEST_STEPS)))
    for i, h in enumerate(TEST_HOPS):
        for j, s in enumerate(TEST_STEPS):
            grid[i, j] = evaluate(model, loaders[h], n_steps=s)
    return grid


@torch.no_grad()
def mean_step_acc(model, loader, max_t):
    """Mean over t = 1..max_t of the final-readout accuracy at t steps."""
    return float(np.mean([evaluate(model, loader, n_steps=t)
                          for t in range(1, max_t + 1)]))


def best_sufficient(grid, hop):
    i = TEST_HOPS.index(hop)
    js = [j for j, s in enumerate(TEST_STEPS) if s >= hop]
    return float(grid[i, js].max())


def train_one(alpha, d_model, seed, epochs, train_samples, test_samples_per,
              nhead, dim_ff, dropout, save_ckpt, tag):
    set_seed(seed)

    # Data BEFORE model init, so a fixed seed yields identical train/test
    # regardless of d_model (model init draws a d-dependent number of RNG values).
    print(f"[data] generating {train_samples} train + "
          f"{len(TEST_HOPS)}x{test_samples_per} test samples ...", flush=True)
    train_ds = GraphReachabilityDataset(
        n_samples=train_samples, hop_range=TRAIN_HOP_RANGE,
        num_nodes_range=TRAIN_NODES, max_nodes=MAX_NODES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True)
    test_loaders = build_test_loaders(test_samples_per)

    model = GraphThinkingTransformer(
        vocab_size=VOCAB_SIZE, d_model=d_model, nhead=nhead, dim_ff=dim_ff,
        dropout=dropout, max_nodes=MAX_NODES, max_thinking_steps=MAX_THINK,
        n_thinking_steps=TRAIN_THINK).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] d_model={d_model} nhead={nhead} dim_ff={dim_ff} "
          f"params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    history = {"epoch": [], "train_loss": [], "train_acc_running": [],
               "train_acc_final": [], "probe_ood_8h_12s": [],
               "probe_ood_10h_15s": [], "probe_id_4h_5s": []}
    probe_every = max(1, epochs // 8)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for node_ids, adj_mask, src_idx, tgt_idx, labels in train_loader:
            node_ids = node_ids.to(DEVICE); adj_mask = adj_mask.to(DEVICE)
            src_idx = src_idx.to(DEVICE); tgt_idx = tgt_idx.to(DEVICE)
            labels = labels.to(DEVICE)
            n_steps = random.randint(*TRAIN_STEP_RANGE)

            if alpha == 0.0:
                logits = model(node_ids, adj_mask, src_idx, tgt_idx,
                               n_steps=n_steps)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                logits_all = model(node_ids, adj_mask, src_idx, tgt_idx,
                                   n_steps=n_steps, return_all=True)
                loss = blended_loss(logits_all, labels, alpha)
                logits = logits_all[-1]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            tot_loss += loss.item() * node_ids.size(0)
            correct += ((logits > 0).float() == labels).sum().item()
            total += node_ids.size(0)

        ep_loss = tot_loss / total
        ep_acc = correct / total
        history["epoch"].append(epoch)
        history["train_loss"].append(ep_loss)
        history["train_acc_running"].append(ep_acc)

        if epoch % probe_every == 0 or epoch in (1, epochs):
            taf = evaluate(model, train_loader, n_steps=TRAIN_THINK)
            p8  = evaluate(model, test_loaders[8],  n_steps=12)
            p10 = evaluate(model, test_loaders[10], n_steps=15)
            p4  = evaluate(model, test_loaders[4],  n_steps=5)
            history["train_acc_final"].append([epoch, taf])
            history["probe_ood_8h_12s"].append([epoch, p8])
            history["probe_ood_10h_15s"].append([epoch, p10])
            history["probe_id_4h_5s"].append([epoch, p4])
            print(f"  epoch {epoch:3d}/{epochs}  loss={ep_loss:.4f}  "
                  f"train_acc(run)={ep_acc:.3f}  train_acc(final@5)={taf:.3f}  "
                  f"OOD 8h@12s={p8:.3f}  10h@15s={p10:.3f}", flush=True)

    train_secs = time.time() - t0

    # -- Final evaluation --
    grid = eval_grid(model, test_loaders)
    id_acc  = float(np.mean([best_sufficient(grid, h) for h in ID_HOPS]))
    ood_acc = float(np.mean([best_sufficient(grid, h) for h in OOD_HOPS]))
    step1_12hop = float(grid[TEST_HOPS.index(12), TEST_STEPS.index(1)])
    low_js  = [j for j, s in enumerate(TEST_STEPS) if s <= 3]
    deep_is = [i for i, h in enumerate(TEST_HOPS) if h >= 6]
    shortcut = float(np.mean([grid[i, j] for i in deep_is for j in low_js]))
    train_acc_final = evaluate(model, train_loader, n_steps=TRAIN_THINK)
    train_acc_meanstep = mean_step_acc(model, train_loader, TRAIN_STEP_RANGE[1])

    # epoch at which the running train accuracy first reaches 0.95
    conv_epoch = next((e for e, a in zip(history["epoch"],
                                         history["train_acc_running"]) if a >= 0.95),
                      None)

    out = {
        "tag": tag, "alpha": alpha, "d_model": d_model, "seed": seed,
        "nhead": nhead, "dim_ff": dim_ff, "epochs": epochs,
        "train_samples": train_samples, "n_params": n_params,
        "train_secs": round(train_secs, 1),
        "test_hops": TEST_HOPS, "test_steps": TEST_STEPS,
        "grid": grid.tolist(),
        "metrics": {
            "id_acc": id_acc,
            "ood_acc": ood_acc,
            "step1_acc_12hop": step1_12hop,
            "shortcut_lowstep_deep": shortcut,
            "train_acc_final": train_acc_final,
            "train_acc_meanstep": train_acc_meanstep,
            "gen_gap": train_acc_final - ood_acc,
            "conv_epoch_95": conv_epoch,
            "ood_per_hop": {str(h): best_sufficient(grid, h) for h in OOD_HOPS},
        },
        "history": history,
    }

    res_dir = os.path.join(HERE, "sweep_results")
    os.makedirs(res_dir, exist_ok=True)
    fname = f"{tag}_a{alpha:g}_d{d_model}_s{seed}.json"
    with open(os.path.join(res_dir, fname), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {fname}  ID={id_acc:.3f}  OOD={ood_acc:.3f}  "
          f"shortcut(step1-12h)={step1_12hop:.3f}  gap={out['metrics']['gen_gap']:+.3f}",
          flush=True)

    if save_ckpt:
        ckpt = os.path.join(res_dir, f"{tag}_a{alpha:g}_d{d_model}_s{seed}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"[done] checkpoint -> {ckpt}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, required=True,
                    help="0=silent thinking, 1=intermediate supervision")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-samples", type=int, default=12_000)
    ap.add_argument("--test-samples-per", type=int, default=500)
    ap.add_argument("--nhead", type=int, default=0,
                    help="0 -> d_model // 32 (keeps head dim = 32)")
    ap.add_argument("--dim-ff", type=int, default=0,
                    help="0 -> 2 * d_model")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tag", type=str, default="e1",
                    help="output-file prefix, e.g. e1 / e2")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="also save the trained weights (for later probing / E4)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run (512 samples, 2 epochs)")
    args = ap.parse_args()

    if args.quick:
        args.train_samples = 512
        args.epochs = 2
        args.test_samples_per = 100

    nhead = args.nhead or max(1, args.d_model // 32)
    dim_ff = args.dim_ff or (2 * args.d_model)

    print("=" * 70)
    print(f"  supervision_sweep  tag={args.tag}  alpha={args.alpha}  "
          f"d_model={args.d_model}  seed={args.seed}")
    print(f"  device={DEVICE}  epochs={args.epochs}  "
          f"train_samples={args.train_samples}")
    print("=" * 70, flush=True)

    train_one(alpha=args.alpha, d_model=args.d_model, seed=args.seed,
              epochs=args.epochs, train_samples=args.train_samples,
              test_samples_per=args.test_samples_per, nhead=nhead, dim_ff=dim_ff,
              dropout=args.dropout, save_ckpt=args.save_ckpt, tag=args.tag)


if __name__ == "__main__":
    main()
